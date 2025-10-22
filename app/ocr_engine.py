"""Shared OCR engine helpers for DeepSeek-OCR web and CLI tooling."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import fitz  # type: ignore
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama import modeling_llama

# ---------------------------------------------------------------------------
# Compatibility patches for varying transformers installations
# ---------------------------------------------------------------------------

if not hasattr(modeling_llama, "LlamaFlashAttention2"):
    from transformers.models.llama.modeling_llama import LlamaAttention

    class LlamaFlashAttention2(LlamaAttention):  # type: ignore[misc]
        """Fallback shim delegating to the standard LlamaAttention."""

        pass

    modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2


if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())  # type: ignore[attr-defined]


if not hasattr(DynamicCache, "get_max_length"):

    def _get_max_length(self: DynamicCache):  # type: ignore[valid-type]
        try:
            return self.max_cache_len
        except AttributeError:
            return None

    DynamicCache.get_max_length = _get_max_length  # type: ignore[attr-defined]


if not hasattr(DynamicCache, "get_usable_length"):

    def _get_usable_length(self: DynamicCache, seq_length: int):  # type: ignore[valid-type]
        max_len = None
        try:
            max_len = self.get_max_length()
        except AttributeError:
            pass
        if isinstance(max_len, int):
            return min(seq_length, max_len)
        return seq_length

    DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OCRDocumentResult:
    """Container describing a single OCR output artefact."""

    input_path: Path
    image_path: Path
    artifact_dir: Path
    text_path: Path
    text_plain: str
    text_markdown: str


WEB_HISTORY_DIR = Path(os.getenv("OCR_HISTORY_DIR", "/workspace/web_history"))
WEB_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def render_pdf(pdf_path: Path, target_dir: Path) -> List[Path]:
    """Render each page of ``pdf_path`` to PNG images in ``target_dir``."""

    doc = fitz.open(pdf_path)  # noqa: SLF001 - PyMuPDF API
    rendered: List[Path] = []
    zoom_matrix = fitz.Matrix(2, 2)  # render at 144 DPI for clarity

    for page_index, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=zoom_matrix)
        out_path = target_dir / f"{pdf_path.stem}_page{page_index:02d}.png"
        pix.save(out_path)
        rendered.append(out_path)

    if not rendered:
        raise RuntimeError(f"PDF '{pdf_path}' contains no pages")

    return rendered


def collect_image_paths(inputs: Iterable[Path], work_dir: Path) -> List[Path]:
    """Expand the provided paths into concrete image files."""

    image_dir = work_dir / "pdf_pages"
    image_dir.mkdir(parents=True, exist_ok=True)

    collected: List[Path] = []
    for path in inputs:
        if path.suffix.lower() == ".pdf":
            collected.extend(render_pdf(path, image_dir))
        else:
            collected.append(path)

    if not collected:
        raise RuntimeError("No image inputs detected")

    return collected


def as_text(result: object) -> str:
    """Best-effort coercion of ``model.infer`` outputs to string."""

    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "markdown", "output", "outputs"):
            value = result.get(key) if key in result else None  # type: ignore[index]
            if value:
                return str(value)
        return json.dumps(result, ensure_ascii=False, indent=2)
    if isinstance(result, (list, tuple)):
        return "\n".join(str(item) for item in result)
    return str(result)


def encode_file_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


# ---------------------------------------------------------------------------
# Model loading and execution
# ---------------------------------------------------------------------------


_MODEL_CACHE: dict[tuple[str, str, str], tuple[AutoTokenizer, AutoModel]] = {}
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-OCR"


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_name: str, attn_impl: str) -> tuple[AutoTokenizer, AutoModel]:
    device = _get_device()
    key = (model_name, attn_impl, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    load_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation=attn_impl,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=load_dtype,
    )

    if device == "cuda":
        model = model.eval().to(device="cuda", dtype=torch.bfloat16)
    else:
        model = model.eval().to(torch.float32)

    _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def run_ocr_documents(
    input_paths: Sequence[Path],
    output_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    attn_impl: str = "flash_attention_2",
    save_json: bool = False,
) -> List[OCRDocumentResult]:
    """Run OCR on the provided files and persist artefacts under ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    text_dir = output_dir / "texts"
    text_dir.mkdir(parents=True, exist_ok=True)

    resolved_inputs = [Path(p).expanduser().resolve() for p in input_paths]
    for path in resolved_inputs:
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

    tokenizer, model = get_model(model_name, attn_impl)
    image_paths = collect_image_paths(resolved_inputs, output_dir)

    results: List[OCRDocumentResult] = []
    for image_path in image_paths:
        image_output_dir = artifact_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        inference_output = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=str(image_output_dir),
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=True,
            test_compress=True,
        )

        plain_text = as_text(inference_output).strip()
        text_candidates = [plain_text]
        result_mmd = image_output_dir / "result.mmd"
        if result_mmd.exists():
            text_candidates.append(result_mmd.read_text(encoding="utf-8"))
        result_md = image_output_dir / "result.md"
        if result_md.exists():
            text_candidates.append(result_md.read_text(encoding="utf-8"))

        markdown_text = next(
            (c.strip() for c in text_candidates if c and c.strip() and c.strip().lower() != "none"),
            "",
        )
        if not plain_text or plain_text.lower() == "none":
            plain_text = markdown_text

        text_path = text_dir / f"{image_path.stem}.md"
        text_path.write_text(markdown_text, encoding="utf-8")

        if save_json:
            payload = {"raw": inference_output}
            json_path = text_dir / f"{image_path.stem}.json"
            json_path.write_text(json.dumps(payload, default=str, ensure_ascii=False, indent=2), encoding="utf-8")

        results.append(
            OCRDocumentResult(
                input_path=image_path,
                image_path=image_path,
                artifact_dir=image_output_dir,
                text_path=text_path,
                text_plain=plain_text,
                text_markdown=markdown_text,
            )
        )

    return results


def run_ocr_bytes(
    image_bytes: bytes,
    filename: str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    attn_impl: str = "flash_attention_2",
) -> dict[str, object]:
    """Run OCR on an in-memory image and return artefacts for web responses."""

    safe_name = filename or "upload.png"
    if "/" in safe_name:
        safe_name = Path(safe_name).name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / safe_name
        input_path.write_bytes(image_bytes)

        output_dir = tmp_path / "outputs"
        results = run_ocr_documents(
            [input_path],
            output_dir,
            model_name=model_name,
            prompt=prompt,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            attn_impl=attn_impl,
            save_json=False,
        )

        if not results:
            raise RuntimeError("OCR produced no output")

        doc = results[0]
        metadata = _persist_history_entry(doc, safe_name, image_bytes)
        bounding_url, crops, preview_url = _build_image_urls(metadata["id"], metadata)

        return {
            "history_id": metadata["id"],
            "created_at": metadata["created_at"],
            "filename": metadata["filename"],
            "text_plain": doc.text_plain,
            "text_markdown": doc.text_markdown,
            "bounding_image_url": bounding_url,
            "crops": crops,
            "preview_image_url": preview_url,
            "metadata": {
                "input": doc.input_path.name,
            },
        }


def _persist_history_entry(
    doc: OCRDocumentResult,
    original_filename: str,
    original_bytes: bytes,
) -> dict[str, object]:
    entry_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    entry_dir = WEB_HISTORY_DIR / entry_id
    entry_dir.mkdir(parents=True, exist_ok=True)

    artifacts_target = entry_dir / "artifacts"
    shutil.copytree(doc.artifact_dir, artifacts_target)

    texts_target = entry_dir / "texts"
    texts_target.mkdir(parents=True, exist_ok=True)
    shutil.copy(doc.text_path, texts_target / doc.text_path.name)

    input_target_dir = entry_dir / "input"
    input_target_dir.mkdir(parents=True, exist_ok=True)
    if doc.image_path.exists():
        shutil.copy(doc.image_path, input_target_dir / doc.image_path.name)
    else:
        (input_target_dir / original_filename).write_bytes(original_bytes)

    crop_files: List[str] = []
    images_dir = artifacts_target / "images"
    if images_dir.exists():
        crop_files = sorted([p.name for p in images_dir.iterdir() if p.is_file()])

    bounding_name = None
    possible = ["result_with_boxes.jpg", "result_with_boxes.png"]
    for candidate in possible:
        candidate_path = artifacts_target / candidate
        if candidate_path.exists():
            bounding_name = candidate_path.name
            break

    created_at = datetime.utcnow().isoformat() + "Z"
    preview_source = (doc.text_markdown or doc.text_plain or "").replace("\n", " ").strip()
    preview = preview_source[:160]

    metadata = {
        "id": entry_id,
        "filename": original_filename,
        "created_at": created_at,
        "text_plain": doc.text_plain,
        "text_markdown": doc.text_markdown,
        "crops": crop_files,
        "bounding_image": bounding_name,
        "preview": preview,
    }

    (entry_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata


def _build_image_urls(entry_id: str, metadata: dict[str, object]) -> tuple[str | None, List[dict[str, str]], str | None]:
    base_url = f"/api/history/{entry_id}"

    bounding_url = None
    bounding_name = metadata.get("bounding_image")
    if bounding_name:
        bounding_url = f"{base_url}/image/bounding"

    crops: List[dict[str, str]] = []
    for name in metadata.get("crops", []):
        crops.append({"name": name, "url": f"{base_url}/image/crop/{name}"})

    preview_url = bounding_url or (crops[0]["url"] if crops else None)
    return bounding_url, crops, preview_url


def _load_entry_metadata(entry_id: str) -> tuple[dict[str, object], Path]:
    entry_dir = WEB_HISTORY_DIR / entry_id
    if not entry_dir.exists():
        raise FileNotFoundError(entry_id)
    meta_file = entry_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(entry_id)
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    return metadata, entry_dir


def list_history_entries(limit: int | None = None) -> List[dict[str, object]]:
    entries: List[dict[str, object]] = []
    if not WEB_HISTORY_DIR.exists():
        return entries

    for entry_dir in WEB_HISTORY_DIR.iterdir():
        if not entry_dir.is_dir():
            continue
        meta_file = entry_dir / "metadata.json"
        if not meta_file.exists():
            continue
        try:
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        entry_id = metadata.get("id", entry_dir.name)
        bounding_url, crops, preview_url = _build_image_urls(entry_id, metadata)

        entry = {
            "id": entry_id,
            "filename": metadata.get("filename", entry_dir.name),
            "created_at": metadata.get("created_at"),
            "preview": metadata.get("preview", ""),
            "preview_image_url": preview_url,
        }
        entries.append(entry)

    entries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    if limit is not None:
        entries = entries[:limit]
    return entries


def load_history_entry(entry_id: str) -> dict[str, object]:
    metadata, _ = _load_entry_metadata(entry_id)
    bounding_url, crops, preview_url = _build_image_urls(entry_id, metadata)

    return {
        "history_id": metadata.get("id", entry_id),
        "filename": metadata.get("filename", entry_id),
        "created_at": metadata.get("created_at"),
        "text_plain": metadata.get("text_plain", ""),
        "text_markdown": metadata.get("text_markdown", ""),
        "bounding_image_url": bounding_url,
        "crops": crops,
        "preview_image_url": preview_url,
        "metadata": {
            "input": metadata.get("filename", entry_id),
        },
    }


def delete_history_entry(entry_id: str) -> None:
    entry_dir = WEB_HISTORY_DIR / entry_id
    if not entry_dir.exists():
        raise FileNotFoundError(entry_id)
    shutil.rmtree(entry_dir)
