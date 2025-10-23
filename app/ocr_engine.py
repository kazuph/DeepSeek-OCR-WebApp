"""Shared OCR engine helpers for DeepSeek-OCR web and CLI tooling."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import quote

import importlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np

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


@dataclass(frozen=True)
class ModelDescriptor:
    key: str
    label: str
    description: str = ""


@dataclass
class OCRVariantArtifacts:
    key: str
    label: str
    text_plain: str
    text_markdown: str
    artifact_dir: Path
    text_path: Path
    bounding_image: Optional[Path]
    crop_images: List[Path]
    preview_text: str
    preview_image: Optional[Path]
    extras: Dict[str, object] | None = None


WEB_HISTORY_DIR = Path(os.getenv("OCR_HISTORY_DIR", "/workspace/web_history"))
WEB_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


MODEL_DESCRIPTORS: List[ModelDescriptor] = [
    ModelDescriptor(key="deepseek", label="DeepSeek OCR"),
    ModelDescriptor(key="yomitoku", label="YomiToku Document Analyzer"),
]

MODEL_DESCRIPTOR_MAP: Dict[str, ModelDescriptor] = {item.key: item for item in MODEL_DESCRIPTORS}
DEFAULT_MODEL_KEY = MODEL_DESCRIPTORS[0].key

_CV2_MODULE = None
_YOMITOKU_EXECUTOR: Optional[ThreadPoolExecutor] = None
LOGGER = logging.getLogger(__name__)


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
_YOMITOKU_ANALYZERS: Dict[tuple[str, bool], object] = {}
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




def _normalize_model_selection(models: Optional[Sequence[str] | str]) -> List[str]:
    if models is None:
        return [DEFAULT_MODEL_KEY]
    if isinstance(models, str):
        raw_items = re.split(r"[,\s]+", models)
    else:
        raw_items = list(models)
    normalized: List[str] = []
    for item in raw_items:
        key = (item or "").strip().lower()
        if not key:
            continue
        if key not in MODEL_DESCRIPTOR_MAP:
            continue
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        normalized.append(DEFAULT_MODEL_KEY)
    return normalized


def _descriptor_for(key: Optional[str]) -> ModelDescriptor:
    effective_key = (key or "").strip().lower() or DEFAULT_MODEL_KEY
    descriptor = MODEL_DESCRIPTOR_MAP.get(effective_key)
    if descriptor:
        return descriptor
    return ModelDescriptor(key=effective_key, label=effective_key)


def _trim_preview(*candidates: str, limit: int = 160) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        normalized = re.sub(r"\s+", " ", candidate).strip()
        if normalized:
            return normalized[:limit]
    return ""


def _append_model_query(url: str, model: Optional[str]) -> str:
    if not model:
        return url
    separator = '&' if '?' in url else '?'
    return f"{url}{separator}model={model}"


def _require_cv2() -> None:
    _get_cv2()


def _get_cv2():
    global _CV2_MODULE
    if _CV2_MODULE is None:
        try:
            _CV2_MODULE = importlib.import_module("cv2")
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "OpenCV (cv2) is required for YomiToku integration but is not available."
            ) from exc
    return _CV2_MODULE


def _get_yomitoku_executor() -> ThreadPoolExecutor:
    global _YOMITOKU_EXECUTOR
    if _YOMITOKU_EXECUTOR is None:
        _YOMITOKU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yomitoku")
    return _YOMITOKU_EXECUTOR


def _get_yomitoku_analyzer(device: str):
    key = (device, False)
    analyzer = _YOMITOKU_ANALYZERS.get(key)
    if analyzer is None:
        from yomitoku import DocumentAnalyzer  # type: ignore

        analyzer = DocumentAnalyzer(device=device, visualize=True, split_text_across_cells=True)
        _YOMITOKU_ANALYZERS[key] = analyzer
    return analyzer


def _load_yomitoku_pages(input_path: Path) -> List[np.ndarray]:
    from yomitoku.data.functions import load_image, load_pdf  # type: ignore

    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        pages = load_pdf(str(input_path))
    else:
        pages = load_image(str(input_path))
    if not pages:
        raise RuntimeError("YomiToku produced no renderable pages.")
    return pages


def _markdown_to_plain_text(markdown_text: str) -> str:
    if not markdown_text:
        return ""
    text_value = markdown_text.replace("<br>", "\n")
    text_value = re.sub(r"<[^>]+>", "", text_value)
    text_value = text_value.replace("|", " ")
    text_value = re.sub(r"\n{3,}", "\n\n", text_value)
    text_value = re.sub(r"[ \t]+", " ", text_value)
    return text_value.strip()


def _yomitoku_plain_from_results(results) -> str:
    lines: List[str] = []
    for paragraph in getattr(results, "paragraphs", []):
        contents = getattr(paragraph, "contents", None)
        if contents:
            lines.append(contents)
    for table in getattr(results, "tables", []):
        for cell in getattr(table, "cells", []):
            contents = getattr(cell, "contents", None)
            if contents:
                lines.append(contents)
    return "\n".join(lines).strip()


def _run_deepseek_variant(
    input_path: Path,
    work_dir: Path,
    *,
    model_name: str,
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    attn_impl: str,
) -> OCRVariantArtifacts:
    results = run_ocr_documents(
        [input_path],
        work_dir,
        model_name=model_name,
        prompt=prompt,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        attn_impl=attn_impl,
        save_json=False,
    )
    if not results:
        raise RuntimeError("DeepSeek OCR produced no output")

    doc = results[0]
    descriptor = _descriptor_for("deepseek")

    bounding_path: Optional[Path] = None
    for candidate in ("result_with_boxes.jpg", "result_with_boxes.png"):
        candidate_path = doc.artifact_dir / candidate
        if candidate_path.exists():
            bounding_path = candidate_path
            break

    crop_paths: List[Path] = []
    images_dir = doc.artifact_dir / "images"
    if images_dir.exists():
        crop_paths = sorted([p for p in images_dir.iterdir() if p.is_file()])

    preview_image = bounding_path or (crop_paths[0] if crop_paths else None)
    preview_text = _trim_preview(doc.text_markdown, doc.text_plain)

    return OCRVariantArtifacts(
        key=descriptor.key,
        label=descriptor.label,
        text_plain=doc.text_plain,
        text_markdown=doc.text_markdown,
        artifact_dir=doc.artifact_dir,
        text_path=doc.text_path,
        bounding_image=bounding_path,
        crop_images=crop_paths,
        preview_text=preview_text,
        preview_image=preview_image,
        extras={"attn_impl": attn_impl},
    )


def _rewrite_yomitoku_markdown_links(markdown: str, figures_dir: Path) -> str:
    relative_prefix = Path("artifacts") / "figures" / figures_dir.name
    replacement = relative_prefix.as_posix()
    candidates = {str(figures_dir), str(figures_dir.resolve())}
    candidates |= {candidate.replace('\\', '/') for candidate in list(candidates)}

    updated = markdown
    for candidate in list(candidates):
        if candidate.rstrip('/'):
            updated = updated.replace(f"{candidate}/", f"{replacement}/")
        updated = updated.replace(candidate, replacement)

    return updated


def _run_yomitoku_variant(input_path: Path, work_dir: Path) -> OCRVariantArtifacts:
    _require_cv2()
    device = _get_device()
    analyzer = _get_yomitoku_analyzer(device)
    pages = _load_yomitoku_pages(input_path)

    work_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = work_dir / "artifacts"
    text_root = work_dir / "texts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    text_root.mkdir(parents=True, exist_ok=True)

    descriptor = _descriptor_for("yomitoku")
    markdown_segments: List[str] = []
    plain_segments: List[str] = []
    crop_paths: List[Path] = []
    bounding_path: Optional[Path] = None
    preview_image: Optional[Path] = None

    executor = _get_yomitoku_executor()

    for index, image in enumerate(pages, start=1):
        future = executor.submit(analyzer, image)
        results, ocr_vis, _ = future.result()
        page_slug = f"{input_path.stem}_page{index:02d}"
        page_text_path = text_root / f"{page_slug}.md"
        figures_dir = artifact_root / "figures" / page_slug
        figures_dir.mkdir(parents=True, exist_ok=True)
        markdown_text = results.to_markdown(  # type: ignore[attr-defined]
            str(page_text_path),
            img=image,
            figure_dir=str(figures_dir),
        )
        markdown_text = _rewrite_yomitoku_markdown_links(markdown_text, figures_dir)
        markdown_segments.append(markdown_text.strip())

        plain_markdown = _markdown_to_plain_text(markdown_text)
        if plain_markdown:
            plain_segments.append(plain_markdown)
        else:
            extracted_plain = _yomitoku_plain_from_results(results)
            if extracted_plain:
                plain_segments.append(extracted_plain)

        if ocr_vis is not None:
            page_bounding_path = artifact_root / f"{page_slug}_ocr.jpg"
            cv2_module = _get_cv2()
            cv2_module.imwrite(str(page_bounding_path), ocr_vis)
            if bounding_path is None:
                bounding_path = page_bounding_path

        page_figures_dir = artifact_root / "figures" / page_slug
        if page_figures_dir.exists():
            crop_paths.extend(sorted(p for p in page_figures_dir.rglob("*") if p.is_file()))

    combined_markdown = "\n\n".join(segment for segment in markdown_segments if segment).strip()
    combined_plain = "\n\n".join(segment for segment in plain_segments if segment).strip()
    if not combined_plain and combined_markdown:
        combined_plain = _markdown_to_plain_text(combined_markdown)

    combined_text_path = text_root / f"{input_path.stem}.md"
    combined_text_path.write_text(combined_markdown, encoding="utf-8")

    if crop_paths and preview_image is None:
        preview_image = crop_paths[0]
    if preview_image is None:
        preview_image = bounding_path

    preview_text = _trim_preview(combined_markdown, combined_plain)

    return OCRVariantArtifacts(
        key=descriptor.key,
        label=descriptor.label,
        text_plain=combined_plain,
        text_markdown=combined_markdown,
        artifact_dir=artifact_root,
        text_path=combined_text_path,
        bounding_image=bounding_path,
        crop_images=crop_paths,
        preview_text=preview_text,
        preview_image=preview_image,
        extras={"pages": len(pages), "device": device},
    )


def run_ocr_bytes(
    image_bytes: bytes,
    filename: str,
    *,
    models: Optional[Sequence[str] | str] = None,
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

    selected_models = _normalize_model_selection(models)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / safe_name
        input_path.write_bytes(image_bytes)

        variant_results: List[OCRVariantArtifacts] = []
        variant_errors: List[str] = []
        for key in selected_models:
            work_dir = tmp_path / f"{key}_work"
            descriptor = _descriptor_for(key)
            try:
                if key == "deepseek":
                    variant = _run_deepseek_variant(
                        input_path,
                        work_dir,
                        model_name=model_name,
                        prompt=prompt,
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        attn_impl=attn_impl,
                    )
                elif key == "yomitoku":
                    variant = _run_yomitoku_variant(input_path, work_dir)
                else:
                    raise ValueError(f"Unsupported OCR model '{key}'")
                variant_results.append(variant)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("%s failed during OCR", descriptor.label)
                variant_errors.append(f"{descriptor.label}: {exc}")
                continue

        if not variant_results:
            detail = "; ".join(variant_errors) or "All OCR models failed."
            raise RuntimeError(detail)

        metadata = _persist_history_entry(variant_results, safe_name, image_bytes, input_path, warnings=variant_errors)
        entry_id = metadata["id"]

        variant_payloads: List[dict[str, object]] = []
        variants_meta = metadata.get("variants")
        if isinstance(variants_meta, list) and variants_meta:
            for variant_meta in variants_meta:
                variant_payloads.append(_build_variant_response(entry_id, metadata, variant_meta))
        else:
            descriptor = _descriptor_for(metadata.get("primary_model"))
            bounding_url, crops, preview_url = _build_image_urls(entry_id, metadata)
            variant_payloads.append(
                {
                    "model": descriptor.key,
                    "label": descriptor.label,
                    "text_plain": metadata.get("text_plain", ""),
                    "text_markdown": metadata.get("text_markdown", ""),
                    "bounding_image_url": bounding_url,
                    "crops": crops,
                    "preview_image_url": preview_url,
                    "preview": metadata.get("preview", ""),
                    "metadata": {},
                }
            )

        primary_variant = variant_payloads[0]

        return {
            "history_id": entry_id,
            "created_at": metadata.get("created_at"),
            "filename": metadata.get("filename", safe_name),
            "text_plain": primary_variant.get("text_plain", ""),
            "text_markdown": primary_variant.get("text_markdown", ""),
            "bounding_image_url": primary_variant.get("bounding_image_url"),
            "crops": primary_variant.get("crops", []),
            "preview_image_url": primary_variant.get("preview_image_url"),
            "preview": primary_variant.get("preview", metadata.get("preview", "")),
            "variants": variant_payloads,
            "metadata": {
                "input": safe_name,
                "models": [variant.get("model") for variant in variant_payloads],
                "warnings": variant_errors,
            },
        }


def _persist_history_entry(
    variants: Sequence[OCRVariantArtifacts],
    original_filename: str,
    original_bytes: bytes,
    source_path: Path,
    *,
    warnings: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if not variants:
        raise ValueError("No OCR variants to persist")

    entry_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    entry_dir = WEB_HISTORY_DIR / entry_id
    entry_dir.mkdir(parents=True, exist_ok=True)

    variant_entries: List[dict[str, object]] = []
    for variant in variants:
        descriptor = _descriptor_for(variant.key)

        variant_root = entry_dir / descriptor.key
        if variant_root.exists():
            shutil.rmtree(variant_root)
        variant_root.mkdir(parents=True, exist_ok=True)

        artifacts_target = variant_root / "artifacts"
        shutil.copytree(variant.artifact_dir, artifacts_target)

        texts_target = variant_root / "texts"
        shutil.copytree(variant.text_path.parent, texts_target)

        bounding_rel = None
        if variant.bounding_image:
            try:
                rel = Path("artifacts") / variant.bounding_image.relative_to(variant.artifact_dir)
            except ValueError:
                rel = Path("artifacts") / variant.bounding_image.name
            bounding_rel = rel.as_posix()

        crop_rel_paths: List[str] = []
        for crop_path in variant.crop_images:
            try:
                rel = Path("artifacts") / crop_path.relative_to(variant.artifact_dir)
            except ValueError:
                rel = Path("artifacts") / crop_path.name
            crop_rel_paths.append(rel.as_posix())

        preview_rel = None
        if variant.preview_image:
            try:
                rel = Path("artifacts") / variant.preview_image.relative_to(variant.artifact_dir)
            except ValueError:
                rel = Path("artifacts") / variant.preview_image.name
            preview_rel = rel.as_posix()

        text_rel = Path("texts") / variant.text_path.relative_to(variant.text_path.parent)

        variant_entries.append(
            {
                "model": descriptor.key,
                "label": variant.label,
                "text_plain": variant.text_plain,
                "text_markdown": variant.text_markdown,
                "bounding_image": bounding_rel,
                "crops": crop_rel_paths,
                "preview": variant.preview_text,
                "preview_image": preview_rel,
                "text_path": text_rel.as_posix(),
                "extras": variant.extras or {},
            }
        )

    input_target_dir = entry_dir / "input"
    input_target_dir.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        shutil.copy(source_path, input_target_dir / source_path.name)
    else:
        (input_target_dir / original_filename).write_bytes(original_bytes)

    created_at = datetime.utcnow().isoformat() + "Z"
    primary_variant = variant_entries[0]
    preview_text = _trim_preview(primary_variant.get("preview", ""))

    metadata = {
        "id": entry_id,
        "filename": original_filename,
        "created_at": created_at,
        "variants": variant_entries,
        "primary_model": primary_variant.get("model"),
        "preview": preview_text,
        "preview_image": primary_variant.get("preview_image"),
        "models": [entry.get("model") for entry in variant_entries],
        "text_plain": primary_variant.get("text_plain", ""),
        "text_markdown": primary_variant.get("text_markdown", ""),
        "bounding_image": primary_variant.get("bounding_image"),
        "crops": primary_variant.get("crops", []),
    }
    if warnings:
        metadata["warnings"] = list(warnings)

    (entry_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata


def select_variant(metadata: dict[str, object], model: Optional[str]) -> tuple[dict[str, object], Optional[str]]:
    variants = metadata.get("variants")
    if isinstance(variants, list) and variants:
        if model:
            for variant in variants:
                if variant.get("model") == model:
                    return variant, variant.get("model")
        chosen = variants[0]
        return chosen, chosen.get("model")
    fallback_key = model or metadata.get("primary_model")
    if isinstance(fallback_key, str) and fallback_key:
        key = fallback_key
    else:
        key = DEFAULT_MODEL_KEY
    return metadata, key


def _build_image_urls(
    entry_id: str,
    metadata: dict[str, object],
    model: Optional[str] = None,
) -> tuple[Optional[str], List[dict[str, str]], Optional[str]]:
    base_url = f"/api/history/{entry_id}"
    variant_meta, variant_key = select_variant(metadata, model)

    bounding_url = None
    bounding_rel = variant_meta.get("bounding_image")
    if bounding_rel:
        bounding_url = _append_model_query(f"{base_url}/image/bounding", variant_key)

    crops: List[dict[str, str]] = []
    for rel_path in variant_meta.get("crops", []):
        rel_str = str(rel_path)
        crop_url = f"{base_url}/image/crop/{quote(rel_str, safe='/')}"
        crop_url = _append_model_query(crop_url, variant_key)
        crops.append(
            {
                "name": Path(rel_str).name,
                "path": rel_str,
                "url": crop_url,
            }
        )

    preview_rel = variant_meta.get("preview_image")
    preview_url = None
    if preview_rel:
        preview_str = str(preview_rel)
        preview_url = f"{base_url}/image/crop/{quote(preview_str, safe='/')}"
        preview_url = _append_model_query(preview_url, variant_key)

    if preview_url is None:
        preview_url = bounding_url or (crops[0]["url"] if crops else None)

    return bounding_url, crops, preview_url


def _build_variant_response(entry_id: str, metadata: dict[str, object], variant_meta: dict[str, object]) -> dict[str, object]:
    variant_key = variant_meta.get("model")
    descriptor = _descriptor_for(variant_key)
    bounding_url, crops, preview_url = _build_image_urls(entry_id, metadata, variant_key)
    extras = variant_meta.get("extras") or {}
    variant_metadata = {
        "text_path": variant_meta.get("text_path"),
        "bounding_image_path": variant_meta.get("bounding_image"),
        **extras,
    }
    return {
        "model": descriptor.key,
        "label": variant_meta.get("label", descriptor.label),
        "text_plain": variant_meta.get("text_plain", ""),
        "text_markdown": variant_meta.get("text_markdown", ""),
        "bounding_image_url": bounding_url,
        "crops": crops,
        "preview_image_url": preview_url,
        "preview": variant_meta.get("preview", ""),
        "metadata": variant_metadata,
    }


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
        _, _, preview_url = _build_image_urls(entry_id, metadata)

        models = metadata.get("models")
        if not models and isinstance(metadata.get("variants"), list):
            models = [variant.get("model") for variant in metadata["variants"]]
        if isinstance(models, list):
            models = [model for model in models if model]

        entry = {
            "id": entry_id,
            "filename": metadata.get("filename", entry_dir.name),
            "created_at": metadata.get("created_at"),
            "preview": metadata.get("preview", ""),
            "preview_image_url": preview_url,
            "models": models or [],
        }
        entries.append(entry)

    entries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    if limit is not None:
        entries = entries[:limit]
    return entries


def load_history_entry(entry_id: str) -> dict[str, object]:
    metadata, _ = _load_entry_metadata(entry_id)

    variants_payload: List[dict[str, object]] = []
    variants_meta = metadata.get("variants")
    if isinstance(variants_meta, list) and variants_meta:
        for variant_meta in variants_meta:
            variants_payload.append(_build_variant_response(entry_id, metadata, variant_meta))
    else:
        descriptor = _descriptor_for(metadata.get("primary_model"))
        bounding_url, crops, preview_url = _build_image_urls(entry_id, metadata)
        variants_payload.append(
            {
                "model": descriptor.key,
                "label": descriptor.label,
                "text_plain": metadata.get("text_plain", ""),
                "text_markdown": metadata.get("text_markdown", ""),
                "bounding_image_url": bounding_url,
                "crops": crops,
                "preview_image_url": preview_url,
                "preview": metadata.get("preview", ""),
                "metadata": {},
            }
        )

    primary_variant = variants_payload[0]

    return {
        "history_id": metadata.get("id", entry_id),
        "filename": metadata.get("filename", entry_id),
        "created_at": metadata.get("created_at"),
        "text_plain": primary_variant.get("text_plain", ""),
        "text_markdown": primary_variant.get("text_markdown", ""),
        "bounding_image_url": primary_variant.get("bounding_image_url"),
        "crops": primary_variant.get("crops", []),
        "preview_image_url": primary_variant.get("preview_image_url"),
        "preview": primary_variant.get("preview", metadata.get("preview", "")),
        "variants": variants_payload,
        "metadata": {
            "input": metadata.get("filename", entry_id),
            "models": [variant.get("model") for variant in variants_payload],
        },
    }


def delete_history_entry(entry_id: str) -> None:
    entry_dir = WEB_HISTORY_DIR / entry_id
    if not entry_dir.exists():
        raise FileNotFoundError(entry_id)
    shutil.rmtree(entry_dir)
