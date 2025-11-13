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
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import quote

import importlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import fitz  # type: ignore
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
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
class ModelOptionDescriptor:
    key: str
    label: str
    description: str = ""
    default: bool = False


@dataclass(frozen=True)
class ModelDescriptor:
    key: str
    label: str
    description: str = ""
    options: tuple[ModelOptionDescriptor, ...] = tuple()


@dataclass
class OCRVariantArtifacts:
    key: str
    label: str
    text_plain: str
    text_markdown: str
    artifact_dir: Path
    text_path: Path
    bounding_image: Optional[Path] = None
    crop_images: List[Path] = field(default_factory=list)
    bounding_images: List[Path] = field(default_factory=list)
    layout_image: Optional[Path] = None
    layout_images: List[Path] = field(default_factory=list)
    preview_text: str = ""
    preview_image: Optional[Path] = None
    extras: Dict[str, object] | None = None


@dataclass
class InputSnapshot:
    """Represents an input image stored for history playback."""

    name: str
    path: Path


WEB_HISTORY_DIR = Path(os.getenv("OCR_HISTORY_DIR", "/workspace/web_history"))
WEB_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


MODEL_DESCRIPTORS: List[ModelDescriptor] = [
    ModelDescriptor(
        key="yomitoku",
        label="YomiToku Document Analyzer (GPU)",
        options=(
            ModelOptionDescriptor(
                key="figure_letter",
                label="絵や図の中の文字も抽出",
                description="YomiToku の --figure_letter 相当。ページ全体を図として検出してしまう絵本・縦書き原稿向け。",
            ),
            ModelOptionDescriptor(
                key="reading_order",
                label="読み順序を表示",
                description="レイアウト解析画像に読み順序番号を表示します（YomiToku v0.9.5以降）。",
                default=False,
            ),
        ),
    ),
    ModelDescriptor(
        key="yomitoku-cpu",
        label="YomiToku Document Analyzer (CPU)",
        description="CPU推論版。GPU版より処理時間が長くなりますが、GPUなしでも動作します。",
        options=(
            ModelOptionDescriptor(
                key="figure_letter",
                label="絵や図の中の文字も抽出",
                description="YomiToku の --figure_letter 相当。ページ全体を図として検出してしまう絵本・縦書き原稿向け。",
            ),
            ModelOptionDescriptor(
                key="reading_order",
                label="読み順序を表示",
                description="レイアウト解析画像に読み順序番号を表示します（YomiToku v0.9.5以降）。",
                default=False,
            ),
        ),
    ),
    ModelDescriptor(key="deepseek", label="DeepSeek OCR"),
#    ModelDescriptor(
#        key="deepseek-8bit",
#        label="DeepSeek OCR (8-bit Quantized)",
#        description="BitsAndBytes 8-bit quantization for mid-range GPUs.",
#    ),
    ModelDescriptor(
        key="deepseek-4bit",
        label="DeepSeek OCR (4-bit Quantized)",
        description="BitsAndBytes 4-bit build for lower VRAM CUDA GPUs.",
    ),
]

MODEL_DESCRIPTOR_MAP: Dict[str, ModelDescriptor] = {item.key: item for item in MODEL_DESCRIPTORS}
MODEL_ORDER: Dict[str, int] = {descriptor.key: index for index, descriptor in enumerate(MODEL_DESCRIPTORS)}
DEFAULT_MODEL_KEY = "deepseek"
MODEL_EXECUTION_PRIORITY = {
    "yomitoku": 0,
    "yomitoku-cpu": 1,
    "deepseek": 2,
    # "deepseek-8bit": 3,
    "deepseek-4bit": 3,
}

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


def _unique_filename(name: str, used: set[str], fallback: str) -> str:
    """Return a sanitized filename unique within ``used``."""

    candidate = Path(name).name or fallback
    if not candidate:
        candidate = fallback

    # Collapse whitespace and disallow path separators
    candidate = re.sub(r"[\s]+", "_", candidate)
    candidate = candidate.replace("/", "_").replace("\\", "_")

    stem, suffix = os.path.splitext(candidate)
    if not suffix:
        suffix = ""
    normalized_key = candidate.lower()
    counter = 2
    while normalized_key in used:
        candidate = f"{stem}_{counter}{suffix}"
        normalized_key = candidate.lower()
        counter += 1

    used.add(normalized_key)
    return candidate


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
_QUANT_MODEL_CACHE: dict[tuple[str, str, str], tuple[AutoTokenizer, AutoModel]] = {}
# _EIGHTBIT_MODEL_CACHE: dict[tuple[str, str, str], tuple[AutoTokenizer, AutoModel]] = {}
_YOMITOKU_ANALYZERS: Dict[tuple[str, bool], object] = {}
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
QUANTIZED_MODEL_NAME = "Jalea96/DeepSeek-OCR-bnb-4bit-NF4"
# EIGHTBIT_MODEL_ALIAS = "deepseek-ai/DeepSeek-OCR#8bit"


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_name: str, attn_impl: str) -> tuple[AutoTokenizer, AutoModel]:
    if model_name == QUANTIZED_MODEL_NAME:
        return _get_quantized_model(model_name, attn_impl)
    # if model_name == EIGHTBIT_MODEL_ALIAS:
    #     return _get_8bit_model(DEFAULT_MODEL_NAME, attn_impl)
    device = _get_device()
    key = (model_name, attn_impl, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if device == "cuda":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        load_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    else:
        load_dtype = torch.float32
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation=attn_impl,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=load_dtype,
    )

    if device == "cuda":
        model = model.eval().to(device="cuda", dtype=load_dtype)
    else:
        model = model.eval().to(torch.float32)

    _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def _get_quantized_model(model_repo: str, attn_impl: str) -> tuple[AutoTokenizer, AutoModel]:
    device = _get_device()
    cache_key = (model_repo, attn_impl, device)
    cached = _QUANT_MODEL_CACHE.get(cache_key)
    if cached:
        return cached

    if device != "cuda":
        raise RuntimeError(
            "DeepSeek OCR (4-bit Quantized) requires a CUDA-enabled GPU. "
            "Run with the full precision model when GPU is unavailable."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)

        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

        quant_kwargs: dict[str, object] = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": compute_dtype,
        }

        def _build_quant_config(kwargs: dict[str, object]) -> BitsAndBytesConfig:
            try:
                return BitsAndBytesConfig(**kwargs)
            except TypeError:
                fallback = dict(kwargs)
                fallback.pop("bnb_4bit_use_double_quant", None)
                return BitsAndBytesConfig(**fallback)

        device_index = torch.cuda.current_device()

        def _load_model(current_kwargs: dict[str, object]):
            return AutoModel.from_pretrained(
                model_repo,
                _attn_implementation=attn_impl,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=compute_dtype,
                quantization_config=_build_quant_config(current_kwargs),
                device_map={"": device_index},
                low_cpu_mem_usage=True,
            )

        try:
            model = _load_model(quant_kwargs)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                quant_kwargs["bnb_4bit_use_double_quant"] = True
                model = _load_model(quant_kwargs)
            else:
                raise
    except Exception as exc:  # noqa: BLE001 - surface dependency issues
        raise RuntimeError(
            "Failed to load the 4-bit quantized model. Ensure `bitsandbytes`, `accelerate`, "
            "and compatible CUDA drivers are installed."
        ) from exc

    model = model.eval()
    if hasattr(model, "config") and getattr(model.config, "use_cache", True) is False:
        try:
            model.config.use_cache = True  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - best-effort safeguard
            pass
    _QUANT_MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def _release_cached_model(model_name: str, attn_impl: str) -> None:
    device = _get_device()
    cache_key = (model_name, attn_impl, device)
    if model_name == QUANTIZED_MODEL_NAME:
        _QUANT_MODEL_CACHE.pop(cache_key, None)
    if cache_key in _MODEL_CACHE:
        _MODEL_CACHE.pop(cache_key, None)

# def _get_8bit_model(model_repo: str, attn_impl: str) -> tuple[AutoTokenizer, AutoModel]:
#     ...


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
    device = _get_device()
    autocast_dtype = None
    if device == "cuda":
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        autocast_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    image_paths = collect_image_paths(resolved_inputs, output_dir)

    results: List[OCRDocumentResult] = []
    try:
        for image_path in image_paths:
            image_output_dir = artifact_dir / image_path.stem
            image_output_dir.mkdir(parents=True, exist_ok=True)
            autocast_context = (
                torch.cuda.amp.autocast(dtype=autocast_dtype) if autocast_dtype else nullcontext()
            )
            with torch.inference_mode():
                with autocast_context:
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
    finally:
        if device == "cuda" and model_name == QUANTIZED_MODEL_NAME:
            _release_cached_model(model_name, attn_impl)
            torch.cuda.empty_cache()

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
    normalized.sort(key=lambda key: MODEL_EXECUTION_PRIORITY.get(key, len(MODEL_EXECUTION_PRIORITY)))
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


def _get_yomitoku_analyzer(device: str = "cuda"):
    """Get or create a YomiToku DocumentAnalyzer for the specified device.

    Args:
        device: Device to use for inference. Either "cuda" (GPU) or "cpu".

    Returns:
        DocumentAnalyzer instance configured for the specified device.
    """
    key = (device, False)
    analyzer = _YOMITOKU_ANALYZERS.get(key)
    if analyzer is None:
        from yomitoku import DocumentAnalyzer  # type: ignore

        analyzer = DocumentAnalyzer(device=device, visualize=True, split_text_across_cells=True)
        _YOMITOKU_ANALYZERS[key] = analyzer
        LOGGER.info(f"Initialized YomiToku DocumentAnalyzer with device={device}")
    return analyzer


def _load_yomitoku_pages(input_path: Path) -> List[np.ndarray]:
    from yomitoku.data.functions import load_image, load_pdf  # type: ignore

    suffix = input_path.suffix.lower()
    LOGGER.info(f"YomiToku loading file: {input_path} (suffix: {suffix})")
    
    if suffix == ".pdf":
        pages = load_pdf(str(input_path))
    else:
        try:
            pages = load_image(str(input_path))
        except ValueError as exc:
            # Provide more helpful error message for unsupported formats
            supported_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'pdf']
            error_msg = (
                f"YomiToku does not support the image format '{suffix or '(no extension)'}'. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
            LOGGER.error(f"{error_msg}. File: {input_path}")
            raise ValueError(error_msg) from exc
    
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


def _word_centroid(points) -> tuple[float, float]:
    xs: List[float] = []
    ys: List[float] = []
    if isinstance(points, (list, tuple)):
        for entry in points:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            x, y = entry[0], entry[1]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                xs.append(float(x))
                ys.append(float(y))
    if xs and ys:
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return 0.0, 0.0


def _yomitoku_words_fallback(results) -> str:
    words = getattr(results, "words", None)
    if not words:
        return ""

    entries: List[dict[str, object]] = []
    for word in words:
        if isinstance(word, dict):
            content = word.get("content")
            points = word.get("points")
            direction = word.get("direction")
        else:
            content = getattr(word, "content", None)
            points = getattr(word, "points", None)
            direction = getattr(word, "direction", None)
        text = (content or "").strip()
        if not text:
            continue
        cx, cy = _word_centroid(points)
        entries.append(
            {
                "direction": (direction or "horizontal"),
                "x": cx,
                "y": cy,
                "text": text,
            }
        )

    if not entries:
        return ""

    bucket = 40.0
    lines: List[str] = []

    horizontals = [entry for entry in entries if entry["direction"] != "vertical"]
    horizontals.sort(key=lambda item: (round(item["y"] / bucket) if bucket else 0, item["x"]))
    current_bucket = None
    buffer: List[str] = []
    for entry in horizontals:
        bucket_index = round(entry["y"] / bucket) if bucket else 0
        if current_bucket is None:
            current_bucket = bucket_index
        if bucket_index != current_bucket and buffer:
            line = "".join(buffer).strip()
            if line:
                lines.append(line)
            buffer = [entry["text"]]  # type: ignore[index]
            current_bucket = bucket_index
        else:
            buffer.append(entry["text"])  # type: ignore[index]
    if buffer:
        line = "".join(buffer).strip()
        if line:
            lines.append(line)

    verticals = [entry for entry in entries if entry["direction"] == "vertical"]
    verticals.sort(key=lambda item: (-round(item["x"] / bucket) if bucket else 0, item["y"]))
    current_bucket = None
    buffer = []
    for entry in verticals:
        bucket_index = round(entry["x"] / bucket) if bucket else 0
        if current_bucket is None:
            current_bucket = bucket_index
        if bucket_index != current_bucket and buffer:
            line = "\n".join(buffer).strip()
            if line:
                lines.append(line)
            buffer = [entry["text"]]  # type: ignore[index]
            current_bucket = bucket_index
        else:
            buffer.append(entry["text"])  # type: ignore[index]
    if buffer:
        line = "\n".join(buffer).strip()
        if line:
            lines.append(line)

    return "\n\n".join(line for line in lines if line).strip()


def _run_deepseek_variant(
    input_paths: Sequence[Path],
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
        input_paths,
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

    descriptor = _descriptor_for("deepseek")

    artifact_root = work_dir / "artifacts"
    text_root = work_dir / "texts"

    combined_markdown_segments: List[str] = []
    combined_plain_segments: List[str] = []
    crop_paths: List[Path] = []
    bounding_path: Optional[Path] = None
    bounding_paths: List[Path] = []
    preview_image: Optional[Path] = None

    for doc in results:
        if doc.text_markdown:
            combined_markdown_segments.append(doc.text_markdown.strip())
        if doc.text_plain:
            combined_plain_segments.append(doc.text_plain.strip())

        page_bounding: Optional[Path] = None
        for candidate in ("result_with_boxes.jpg", "result_with_boxes.png"):
            candidate_path = doc.artifact_dir / candidate
            if candidate_path.exists():
                page_bounding = candidate_path
                if bounding_path is None:
                    bounding_path = candidate_path
                if preview_image is None:
                    preview_image = candidate_path
                bounding_paths.append(candidate_path)
                break

        images_dir = doc.artifact_dir / "images"
        if images_dir.exists():
            crop_paths.extend(sorted(p for p in images_dir.iterdir() if p.is_file()))

    combined_markdown = "\n\n".join(segment for segment in combined_markdown_segments if segment).strip()
    combined_plain = "\n\n".join(segment for segment in combined_plain_segments if segment).strip()
    if not combined_plain and combined_markdown:
        combined_plain = _markdown_to_plain_text(combined_markdown)

    text_root.mkdir(parents=True, exist_ok=True)
    combined_text_path = text_root / "combined.md"
    combined_text_path.write_text(combined_markdown, encoding="utf-8")

    if preview_image is None and crop_paths:
        preview_image = crop_paths[-1]
    if preview_image is None:
        preview_image = bounding_path

    preview_text = _trim_preview(combined_markdown, combined_plain)

    extras = {"attn_impl": attn_impl, "documents": len(results)}

    return OCRVariantArtifacts(
        key=descriptor.key,
        label=descriptor.label,
        text_plain=combined_plain,
        text_markdown=combined_markdown,
        artifact_dir=artifact_root,
        text_path=combined_text_path,
        bounding_image=bounding_path,
        bounding_images=bounding_paths,
        crop_images=crop_paths,
        preview_text=preview_text,
        preview_image=preview_image,
        extras=extras,
    )

# def _run_8bit_deepseek_variant(...):
#     pass


def _run_quantized_deepseek_variant(
    input_paths: Sequence[Path],
    work_dir: Path,
    *,
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    attn_impl: str,
) -> OCRVariantArtifacts:
    results = run_ocr_documents(
        input_paths,
        work_dir,
        model_name=QUANTIZED_MODEL_NAME,
        prompt=prompt,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        attn_impl=attn_impl,
        save_json=False,
    )
    if not results:
        raise RuntimeError("DeepSeek OCR (4-bit Quantized) produced no output")

    descriptor = _descriptor_for("deepseek-4bit")

    artifact_root = work_dir / "artifacts"
    text_root = work_dir / "texts"

    combined_markdown_segments: List[str] = []
    combined_plain_segments: List[str] = []
    crop_paths: List[Path] = []
    bounding_path: Optional[Path] = None
    bounding_paths: List[Path] = []
    preview_image: Optional[Path] = None

    for doc in results:
        if doc.text_markdown:
            combined_markdown_segments.append(doc.text_markdown.strip())
        if doc.text_plain:
            combined_plain_segments.append(doc.text_plain.strip())

        page_bounding: Optional[Path] = None
        for candidate in ("result_with_boxes.jpg", "result_with_boxes.png"):
            candidate_path = doc.artifact_dir / candidate
            if candidate_path.exists():
                page_bounding = candidate_path
                if bounding_path is None:
                    bounding_path = candidate_path
                if preview_image is None:
                    preview_image = candidate_path
                bounding_paths.append(candidate_path)
                break

        images_dir = doc.artifact_dir / "images"
        if images_dir.exists():
            crop_paths.extend(sorted(p for p in images_dir.iterdir() if p.is_file()))

    combined_markdown = "\n\n".join(segment for segment in combined_markdown_segments if segment).strip()
    combined_plain = "\n\n".join(segment for segment in combined_plain_segments if segment).strip()
    if not combined_plain and combined_markdown:
        combined_plain = _markdown_to_plain_text(combined_markdown)

    text_root.mkdir(parents=True, exist_ok=True)
    combined_text_path = text_root / "combined.md"
    combined_text_path.write_text(combined_markdown, encoding="utf-8")

    if preview_image is None and crop_paths:
        preview_image = crop_paths[-1]
    if preview_image is None:
        preview_image = bounding_path

    preview_text = _trim_preview(combined_markdown, combined_plain)

    extras = {
        "attn_impl": attn_impl,
        "quantized": True,
        "model_repo": QUANTIZED_MODEL_NAME,
        "documents": len(results),
    }

    return OCRVariantArtifacts(
        key=descriptor.key,
        label=descriptor.label,
        text_plain=combined_plain,
        text_markdown=combined_markdown,
        artifact_dir=artifact_root,
        text_path=combined_text_path,
        bounding_image=bounding_path,
        bounding_images=bounding_paths,
        crop_images=crop_paths,
        preview_text=preview_text,
        preview_image=preview_image,
        extras=extras,
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


def _run_yomitoku_variant(
    input_paths: Sequence[Path],
    work_dir: Path,
    *,
    export_figure_letter: bool = False,
    export_reading_order: bool = False,
    device: str = "cuda",
    model_key: str = "yomitoku",
) -> OCRVariantArtifacts:
    """Run YomiToku OCR variant.

    Args:
        input_paths: List of input image/PDF paths to process.
        work_dir: Working directory for output artifacts.
        export_figure_letter: Whether to extract text from figures.
        export_reading_order: Whether to export reading order visualization.
        device: Device for inference. Either "cuda" (GPU) or "cpu".
        model_key: Model key for descriptor lookup ("yomitoku" or "yomitoku-cpu").

    Returns:
        OCRVariantArtifacts containing results and output paths.
    """
    _require_cv2()
    analyzer = _get_yomitoku_analyzer(device)

    work_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = work_dir / "artifacts"
    text_root = work_dir / "texts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    text_root.mkdir(parents=True, exist_ok=True)

    descriptor = _descriptor_for(model_key)
    markdown_segments: List[str] = []
    plain_segments: List[str] = []
    crop_paths: List[Path] = []
    bounding_path: Optional[Path] = None
    bounding_paths: List[Path] = []
    layout_path: Optional[Path] = None
    layout_paths: List[Path] = []
    preview_image: Optional[Path] = None
    used_word_fallback = False

    executor = _get_yomitoku_executor()
    total_pages = 0

    for input_path in input_paths:
        pages = _load_yomitoku_pages(input_path)
        for page_index, image in enumerate(pages, start=1):
            total_pages += 1
            future = executor.submit(analyzer, image)
            results, ocr_vis, layout_vis = future.result()
            page_slug = f"{input_path.stem}_page{page_index:02d}"
            page_text_path = text_root / f"{page_slug}.md"
            figures_dir = artifact_root / "figures" / page_slug
            figures_dir.mkdir(parents=True, exist_ok=True)
            markdown_text = results.to_markdown(  # type: ignore[attr-defined]
                str(page_text_path),
                img=image,
                figure_dir=str(figures_dir),
                export_figure_letter=export_figure_letter,
            )
            markdown_text = _rewrite_yomitoku_markdown_links(markdown_text, figures_dir)
            normalized_markdown = markdown_text.strip()
            fallback_text = ""
            if not normalized_markdown:
                fallback_text = _yomitoku_words_fallback(results)
                if fallback_text:
                    normalized_markdown = fallback_text
                    used_word_fallback = True

            markdown_segments.append(normalized_markdown)

            plain_markdown = _markdown_to_plain_text(normalized_markdown)
            if plain_markdown:
                plain_segments.append(plain_markdown)
            else:
                extracted_plain = _yomitoku_plain_from_results(results)
                if not extracted_plain:
                    extracted_plain = fallback_text or _yomitoku_words_fallback(results)
                    if extracted_plain and not fallback_text:
                        used_word_fallback = True
                        markdown_segments[-1] = f"{normalized_markdown}\n\n{extracted_plain}".strip()
                if extracted_plain:
                    plain_segments.append(extracted_plain)

            if ocr_vis is not None:
                page_bounding_path = artifact_root / f"{page_slug}_ocr.jpg"
                cv2_module = _get_cv2()
                cv2_module.imwrite(str(page_bounding_path), ocr_vis)
                bounding_paths.append(page_bounding_path)
                if bounding_path is None:
                    bounding_path = page_bounding_path
                if preview_image is None:
                    preview_image = page_bounding_path

            if export_reading_order and layout_vis is not None:
                page_layout_path = artifact_root / f"{page_slug}_layout.jpg"
                cv2_module = _get_cv2()
                cv2_module.imwrite(str(page_layout_path), layout_vis)
                layout_paths.append(page_layout_path)
                if layout_path is None:
                    layout_path = page_layout_path

            page_figures_dir = artifact_root / "figures" / page_slug
            if page_figures_dir.exists():
                crop_paths.extend(sorted(p for p in page_figures_dir.rglob("*") if p.is_file()))

    combined_markdown = "\n\n".join(segment for segment in markdown_segments if segment).strip()
    combined_plain = "\n\n".join(segment for segment in plain_segments if segment).strip()
    if not combined_plain and combined_markdown:
        combined_plain = _markdown_to_plain_text(combined_markdown)

    combined_text_path = text_root / "combined.md"
    combined_text_path.write_text(combined_markdown, encoding="utf-8")

    if crop_paths and preview_image is None:
        preview_image = crop_paths[-1]
    if preview_image is None:
        preview_image = bounding_path

    preview_text = _trim_preview(combined_markdown, combined_plain)

    extras = {
        "pages": total_pages,
        "device": device,
        "figure_letter": export_figure_letter,
        "reading_order": export_reading_order,
        "word_fallback_used": used_word_fallback,
    }

    return OCRVariantArtifacts(
        key=descriptor.key,
        label=descriptor.label,
        text_plain=combined_plain,
        text_markdown=combined_markdown,
        artifact_dir=artifact_root,
        text_path=combined_text_path,
        bounding_image=bounding_path,
        bounding_images=bounding_paths,
        layout_image=layout_path,
        layout_images=layout_paths,
        crop_images=crop_paths,
        preview_text=preview_text,
        preview_image=preview_image,
        extras=extras,
    )


def _build_input_images(entry_id: str, entry_dir: Path) -> List[dict[str, str]]:
    images: List[dict[str, str]] = []
    input_dir = entry_dir / "input"
    if not input_dir.exists():
        return images

    manifest_path = input_dir / "input_manifest.json"
    manifest_entries: List[dict[str, str]] = []
    if manifest_path.exists():
        try:
            manifest_entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - tolerate manual edits
            manifest_entries = []

    seen: set[Path] = set()

    def append_image(path: Path, display_name: str) -> None:
        try:
            relative = path.relative_to(input_dir).as_posix()
        except ValueError:
            relative = path.name
        url_path = quote(relative, safe="/")
        images.append(
            {
                "name": display_name,
                "path": relative,
                "url": f"/api/history/{entry_id}/image/input/{url_path}",
            }
        )

    # Preserve manifest order when available
    for entry in manifest_entries:
        file_name = entry.get("file") if isinstance(entry, dict) else None
        if not file_name:
            continue
        candidate = input_dir / file_name
        if not candidate.is_file():
            continue
        seen.add(candidate)
        display_name = entry.get("name") if isinstance(entry, dict) else None
        append_image(candidate, display_name or candidate.name)

    for path in sorted(input_dir.iterdir()):
        if path == manifest_path or not path.is_file() or path in seen:
            continue
        append_image(path, path.name)

    return images


def run_ocr_bytes(
    image_bytes: bytes,
    filename: str,
    *,
    models: Optional[Sequence[str] | str] = None,
    history_id: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    attn_impl: str = "flash_attention_2",
    model_options: Optional[Mapping[str, Any]] = None,
) -> dict[str, object]:
    """Run OCR on an in-memory image and return artefacts for web responses."""

    safe_name = filename or "upload.png"
    if "/" in safe_name:
        safe_name = Path(safe_name).name

    return run_ocr_uploads(
        [(safe_name, image_bytes)],
        models=models,
        history_id=history_id,
        model_name=model_name,
        prompt=prompt,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        attn_impl=attn_impl,
        model_options=model_options,
    )


def run_ocr_uploads(
    uploads: Sequence[tuple[str, bytes]],
    *,
    models: Optional[Sequence[str] | str] = None,
    history_id: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    attn_impl: str = "flash_attention_2",
    model_options: Optional[Mapping[str, Any]] = None,
) -> dict[str, object]:
    """Run OCR on one or more uploaded files and persist results."""

    if not uploads:
        raise ValueError("No uploads provided")

    selected_models = _normalize_model_selection(models)
    options_by_model: Mapping[str, Any]
    if isinstance(model_options, Mapping):
        options_by_model = model_options
    else:
        options_by_model = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[Path] = []
        original_names: List[str] = []
        used_names: set[str] = set()

        for index, (raw_name, content) in enumerate(uploads, start=1):
            suffix = Path(raw_name or "").suffix or ".png"
            fallback_name = f"upload_{index:03d}{suffix}"
            safe_name = _unique_filename(raw_name or fallback_name, used_names, fallback_name)
            target_path = upload_dir / safe_name
            target_path.write_bytes(content)
            saved_paths.append(target_path)
            original_names.append(raw_name or safe_name)

        processed_images = collect_image_paths(saved_paths, tmp_path)
        if not processed_images:
            raise RuntimeError("No image inputs detected")

        snapshots = [InputSnapshot(name=path.name, path=path) for path in processed_images]
        snapshot_names = [snapshot.name for snapshot in snapshots]

        display_filename = original_names[0] if original_names else processed_images[0].name
        if len(original_names) > 1:
            display_filename = f"{display_filename} 他{len(original_names) - 1}件"

        variant_results: List[OCRVariantArtifacts] = []
        variant_errors: List[str] = []

        for key in selected_models:
            work_dir = tmp_path / f"{key}_work"
            descriptor = _descriptor_for(key)
            try:
                start_time = time.perf_counter()
                if key == "deepseek":
                    variant = _run_deepseek_variant(
                        processed_images,
                        work_dir,
                        model_name=model_name,
                        prompt=prompt,
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        attn_impl=attn_impl,
                    )
                elif key == "deepseek-4bit":
                    variant = _run_quantized_deepseek_variant(
                        processed_images,
                        work_dir,
                        prompt=prompt,
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        attn_impl=attn_impl,
                    )
                elif key in ("yomitoku", "yomitoku-cpu"):
                    # Determine device based on model key
                    device = "cpu" if key == "yomitoku-cpu" else "cuda"

                    # Get model options
                    yomitoku_options: Any = None
                    try:
                        yomitoku_options = options_by_model.get(key)
                    except AttributeError:
                        yomitoku_options = None

                    export_figure_letter = False
                    export_reading_order = False
                    if isinstance(yomitoku_options, Mapping):
                        export_figure_letter = bool(yomitoku_options.get("figure_letter"))
                        export_reading_order = bool(yomitoku_options.get("reading_order"))
                    elif isinstance(yomitoku_options, bool):
                        export_figure_letter = yomitoku_options

                    variant = _run_yomitoku_variant(
                        processed_images,
                        work_dir,
                        export_figure_letter=export_figure_letter,
                        export_reading_order=export_reading_order,
                        device=device,
                        model_key=key,
                    )
                else:
                    raise ValueError(f"Unsupported OCR model '{key}'")
                elapsed = time.perf_counter() - start_time
                extras = variant.extras.copy() if variant.extras else {}
                extras["elapsed_seconds"] = round(elapsed, 3)
                extras["inputs"] = len(processed_images)
                variant.extras = extras
                variant_results.append(variant)
            except ValueError as exc:
                # Handle YomiToku image size validation errors
                error_msg = str(exc)
                if "Image size is too small" in error_msg:
                    user_friendly_msg = f"{descriptor.label}: 画像サイズが小さすぎます（最小32ピクセル、推奨720ピクセル以上）"
                    LOGGER.warning("%s: %s", descriptor.label, user_friendly_msg)
                    variant_errors.append(user_friendly_msg)
                else:
                    LOGGER.exception("%s failed during OCR", descriptor.label)
                    variant_errors.append(f"{descriptor.label}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("%s failed during OCR", descriptor.label)
                variant_errors.append(f"{descriptor.label}: {exc}")
                continue

        if not variant_results:
            detail = "; ".join(variant_errors) or "All OCR models failed."
            raise RuntimeError(detail)

        metadata = None
        entry_id: Optional[str] = None

        if history_id:
            try:
                metadata = _append_history_entry(
                    history_id,
                    variant_results,
                    display_filename,
                    snapshots,
                    original_filenames=snapshot_names,
                    warnings=variant_errors,
                )
                entry_id = metadata.get("id", history_id)
            except FileNotFoundError:
                metadata = None
                entry_id = None

        if metadata is None:
            metadata = _persist_history_entry(
                variant_results,
                display_filename,
                snapshots,
                input_filenames=snapshot_names,
                warnings=variant_errors,
            )
            entry_id = metadata["id"]

        entry_dir = WEB_HISTORY_DIR / entry_id
        input_images = _build_input_images(entry_id, entry_dir)

        variant_payloads: List[dict[str, object]] = []
        variants_meta = metadata.get("variants")
        if isinstance(variants_meta, list) and variants_meta:
            for variant_meta in variants_meta:
                variant_payloads.append(_build_variant_response(entry_id, entry_dir, metadata, variant_meta))
        else:
            descriptor = _descriptor_for(metadata.get("primary_model"))
            bounding_url, bounding_images, layout_url, layout_images, crops, preview_url = _build_image_urls(entry_id, entry_dir, metadata)
            variant_payloads.append(
                {
                    "model": descriptor.key,
                    "label": descriptor.label,
                    "text_plain": metadata.get("text_plain", ""),
                    "text_markdown": metadata.get("text_markdown", ""),
                    "bounding_image_url": bounding_url,
                    "bounding_images": bounding_images,
                    "layout_image_url": layout_url,
                    "layout_images": layout_images,
                    "crops": crops,
                    "preview_image_url": preview_url,
                    "preview": metadata.get("preview", ""),
                    "metadata": {},
                }
            )

        primary_variant = variant_payloads[0]

        response_metadata = {
            "input": display_filename,
            "input_files": snapshot_names,
            "models": [variant.get("model") for variant in variant_payloads],
            "warnings": variant_errors,
        }

        return {
            "history_id": entry_id,
            "created_at": metadata.get("created_at"),
            "filename": metadata.get("filename", display_filename),
            "text_plain": primary_variant.get("text_plain", ""),
            "text_markdown": primary_variant.get("text_markdown", ""),
            "bounding_image_url": primary_variant.get("bounding_image_url"),
            "bounding_images": primary_variant.get("bounding_images", []),
            "crops": primary_variant.get("crops", []),
            "preview_image_url": primary_variant.get("preview_image_url"),
            "preview": primary_variant.get("preview", metadata.get("preview", "")),
            "variants": variant_payloads,
            "input_images": input_images,
            "metadata": response_metadata,
        }


def _persist_history_entry(
    variants: Sequence[OCRVariantArtifacts],
    display_filename: str,
    snapshots: Sequence[InputSnapshot],
    *,
    input_filenames: Optional[Sequence[str]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if not variants:
        raise ValueError("No OCR variants to persist")

    entry_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    entry_dir = WEB_HISTORY_DIR / entry_id
    entry_dir.mkdir(parents=True, exist_ok=True)

    variant_entries = [_store_variant_entry(entry_dir, variant) for variant in variants]

    _ensure_input_snapshots(entry_dir, snapshots)

    warnings_list = None
    if warnings:
        warnings_list = [warning for warning in warnings if warning]

    metadata = _compose_history_metadata(
        entry_id,
        display_filename,
        variant_entries,
        input_files=[str(name) for name in input_filenames] if input_filenames else None,
        warnings=warnings_list,
    )

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


def _collect_bounding_rel_paths(
    entry_dir: Path,
    variant_key: Optional[str],
    variant_meta: dict[str, object],
    metadata: dict[str, object],
) -> List[str]:
    existing: List[str] = []
    raw = variant_meta.get("bounding_images")
    if isinstance(raw, list):
        existing.extend(str(item) for item in raw if item)

    single = variant_meta.get("bounding_image")
    if single:
        existing.append(str(single))

    if not existing and variant_meta is not metadata:
        meta_level = metadata.get("bounding_images")
        if isinstance(meta_level, list):
            existing.extend(str(item) for item in meta_level if item)
        single_meta = metadata.get("bounding_image")
        if single_meta:
            existing.append(str(single_meta))

    normalized: List[str] = []
    seen: set[str] = set()
    for rel_str in existing:
        if not rel_str:
            continue
        if rel_str in seen:
            continue
        seen.add(rel_str)
        normalized.append(rel_str)

    if normalized:
        return normalized

    if not variant_key:
        return []

    variant_root = entry_dir / variant_key
    artifacts_root = variant_root / "artifacts"
    if not artifacts_root.exists():
        return []

    candidates: List[str] = []
    for path in sorted(p for p in artifacts_root.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        name = path.name.lower()
        if "ocr" not in name and "box" not in name and "bounding" not in name:
            continue
        rel = Path("artifacts") / path.relative_to(artifacts_root)
        rel_str = rel.as_posix()
        if rel_str in seen:
            continue
        seen.add(rel_str)
        candidates.append(rel_str)

    if not candidates:
        for fallback in ("result_with_boxes.jpg", "result_with_boxes.png"):
            candidate = artifacts_root / fallback
            if candidate.exists():
                rel_str = (Path("artifacts") / fallback).as_posix()
                if rel_str not in seen:
                    candidates.append(rel_str)
                    seen.add(rel_str)

    return candidates


def _build_image_urls(
    entry_id: str,
    entry_dir: Path,
    metadata: dict[str, object],
    model: Optional[str] = None,
) -> tuple[Optional[str], List[dict[str, str]], Optional[str], List[dict[str, str]], List[dict[str, str]], Optional[str]]:
    base_url = f"/api/history/{entry_id}"
    variant_meta, variant_key = select_variant(metadata, model)

    bounding_url = None
    bounding_rel = variant_meta.get("bounding_image")
    bounding_rel_str = str(bounding_rel) if bounding_rel else None
    if bounding_rel_str:
        bounding_url = _append_model_query(f"{base_url}/image/bounding", variant_key)

    raw_bounding_paths = _collect_bounding_rel_paths(entry_dir, variant_key, variant_meta, metadata)
    if not raw_bounding_paths and bounding_rel_str:
        raw_bounding_paths = [bounding_rel_str]

    bounding_path_set = {rel for rel in raw_bounding_paths}
    bounding_name_set = {Path(rel).name for rel in raw_bounding_paths}

    bounding_images: List[dict[str, str]] = []
    for rel_str in raw_bounding_paths:
        url = f"{base_url}/image/bounding/{quote(rel_str, safe='/')}"
        url = _append_model_query(url, variant_key)
        bounding_images.append(
            {
                "name": Path(rel_str).name,
                "path": rel_str,
                "url": url,
            }
        )

    if bounding_url is None and bounding_images:
        bounding_url = bounding_images[0]["url"]

    layout_url = None
    layout_rel = variant_meta.get("layout_image")
    layout_rel_str = str(layout_rel) if layout_rel else None
    if layout_rel_str:
        layout_url = _append_model_query(f"{base_url}/image/layout", variant_key)

    raw_layout_paths = []
    layout_list = variant_meta.get("layout_images")
    if isinstance(layout_list, list):
        raw_layout_paths = [str(rel_path) for rel_path in layout_list]
    if not raw_layout_paths and layout_rel_str:
        raw_layout_paths = [layout_rel_str]

    layout_images: List[dict[str, str]] = []
    for rel_str in raw_layout_paths:
        url = f"{base_url}/image/layout/{quote(rel_str, safe='/')}"
        url = _append_model_query(url, variant_key)
        layout_images.append(
            {
                "name": Path(rel_str).name,
                "path": rel_str,
                "url": url,
            }
        )

    if layout_url is None and layout_images:
        layout_url = layout_images[0]["url"]

    raw_crop_paths = [str(rel_path) for rel_path in variant_meta.get("crops", [])]
    crop_path_set = set(raw_crop_paths)
    crop_name_set = {Path(rel_str).name for rel_str in raw_crop_paths}

    crops: List[dict[str, str]] = []
    for rel_str in raw_crop_paths:
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
        preview_name = Path(preview_str).name
        if bounding_rel_str and Path(preview_str) == Path(bounding_rel_str):
            preview_url = bounding_url
        elif preview_str in bounding_path_set or preview_name in bounding_name_set:
            preview_url = next(
                (item["url"] for item in bounding_images if item["path"] == preview_str or Path(item["path"]).name == preview_name),
                None,
            )
        elif preview_str in crop_path_set or preview_name in crop_name_set:
            preview_url = f"{base_url}/image/crop/{quote(preview_str, safe='/')}"
            preview_url = _append_model_query(preview_url, variant_key)
        else:
            preview_url = bounding_url

    if preview_url is None:
        preview_url = bounding_url or (bounding_images[0]["url"] if bounding_images else None) or (crops[0]["url"] if crops else None)

    return bounding_url, bounding_images, layout_url, layout_images, crops, preview_url


def _build_variant_response(
    entry_id: str,
    entry_dir: Path,
    metadata: dict[str, object],
    variant_meta: dict[str, object],
) -> dict[str, object]:
    variant_key = variant_meta.get("model")
    descriptor = _descriptor_for(variant_key)
    bounding_url, bounding_images, layout_url, layout_images, crops, preview_url = _build_image_urls(entry_id, entry_dir, metadata, variant_key)
    extras = variant_meta.get("extras") or {}
    variant_metadata = {
        "text_path": variant_meta.get("text_path"),
        "bounding_image_path": variant_meta.get("bounding_image"),
        "layout_image_path": variant_meta.get("layout_image"),
        **extras,
    }
    if bounding_images:
        variant_metadata.setdefault("bounding_images", [item["path"] for item in bounding_images])
    if layout_images:
        variant_metadata.setdefault("layout_images", [item["path"] for item in layout_images])
    return {
        "model": descriptor.key,
        "label": variant_meta.get("label", descriptor.label),
        "text_plain": variant_meta.get("text_plain", ""),
        "text_markdown": variant_meta.get("text_markdown", ""),
        "bounding_image_url": bounding_url,
        "bounding_images": bounding_images,
        "layout_image_url": layout_url,
        "layout_images": layout_images,
        "crops": crops,
        "preview_image_url": preview_url,
        "preview": variant_meta.get("preview", ""),
        "metadata": variant_metadata,
        "elapsed_seconds": variant_metadata.get("elapsed_seconds") or extras.get("elapsed_seconds"),
    }


def _sort_variant_entries(entries: List[dict[str, object]]) -> List[dict[str, object]]:
    return sorted(
        entries,
        key=lambda entry: (
            MODEL_ORDER.get(entry.get("model"), 999),
            entry.get("model") or "",
        ),
    )


def _store_variant_entry(entry_dir: Path, variant: OCRVariantArtifacts) -> dict[str, object]:
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
    bounding_rel_paths: List[str] = []
    if variant.bounding_image:
        try:
            rel = Path("artifacts") / variant.bounding_image.relative_to(variant.artifact_dir)
        except ValueError:
            rel = Path("artifacts") / variant.bounding_image.name
        bounding_rel = rel.as_posix()

    for bounding_path in variant.bounding_images:
        try:
            rel = Path("artifacts") / bounding_path.relative_to(variant.artifact_dir)
        except ValueError:
            rel = Path("artifacts") / bounding_path.name
        rel_str = rel.as_posix()
        if rel_str not in bounding_rel_paths:
            bounding_rel_paths.append(rel_str)

    if bounding_rel is None and bounding_rel_paths:
        bounding_rel = bounding_rel_paths[0]

    layout_rel = None
    layout_rel_paths: List[str] = []
    if variant.layout_image:
        try:
            rel = Path("artifacts") / variant.layout_image.relative_to(variant.artifact_dir)
        except ValueError:
            rel = Path("artifacts") / variant.layout_image.name
        layout_rel = rel.as_posix()

    for layout_path in variant.layout_images:
        try:
            rel = Path("artifacts") / layout_path.relative_to(variant.artifact_dir)
        except ValueError:
            rel = Path("artifacts") / layout_path.name
        rel_str = rel.as_posix()
        if rel_str not in layout_rel_paths:
            layout_rel_paths.append(rel_str)

    if layout_rel is None and layout_rel_paths:
        layout_rel = layout_rel_paths[0]

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

    return {
        "model": descriptor.key,
        "label": variant.label,
        "text_plain": variant.text_plain,
        "text_markdown": variant.text_markdown,
        "bounding_image": bounding_rel,
        "bounding_images": bounding_rel_paths,
        "layout_image": layout_rel,
        "layout_images": layout_rel_paths,
        "crops": crop_rel_paths,
        "preview": variant.preview_text,
        "preview_image": preview_rel,
        "text_path": text_rel.as_posix(),
        "extras": variant.extras or {},
    }


def _compose_history_metadata(
    entry_id: str,
    filename: str,
    variant_entries: List[dict[str, object]],
    *,
    created_at: Optional[str] = None,
    input_files: Optional[Iterable[str]] = None,
    warnings: Optional[Iterable[str]] = None,
) -> dict[str, object]:
    if not variant_entries:
        raise ValueError("No variant entries to persist")

    ordered_entries = _sort_variant_entries(variant_entries)
    primary = ordered_entries[0]

    metadata: dict[str, object] = {
        "id": entry_id,
        "filename": filename,
        "created_at": created_at or datetime.utcnow().isoformat() + "Z",
        "variants": ordered_entries,
        "primary_model": primary.get("model"),
        "preview": _trim_preview(primary.get("preview", "")),
        "preview_image": primary.get("preview_image"),
        "models": [entry.get("model") for entry in ordered_entries if entry.get("model")],
        "text_plain": primary.get("text_plain", ""),
        "text_markdown": primary.get("text_markdown", ""),
        "bounding_image": primary.get("bounding_image"),
        "bounding_images": primary.get("bounding_images", []),
        "crops": primary.get("crops", []),
    }

    if input_files:
        metadata["input_files"] = [str(item) for item in input_files if item]

    if warnings:
        warning_set = {warning for warning in warnings if warning}
        if warning_set:
            metadata["warnings"] = list(warning_set)

    return metadata


def _ensure_input_snapshots(
    entry_dir: Path,
    snapshots: Sequence[InputSnapshot],
) -> None:
    if not snapshots:
        return

    input_target_dir = entry_dir / "input"
    existing_files: set[str] = set()
    if input_target_dir.exists():
        for candidate in input_target_dir.iterdir():
            if candidate.is_file() and candidate.name != "input_manifest.json":
                existing_files.add(candidate.name.lower())
    if existing_files:
        # Preserve the first captured inputs to retain original history context.
        return

    input_target_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[dict[str, str]] = []
    used: set[str] = set(existing_files)

    for index, snapshot in enumerate(snapshots, start=1):
        if not snapshot.path.exists():
            continue
        fallback = f"input_{index:03d}{snapshot.path.suffix or '.png'}"
        stored_name = _unique_filename(snapshot.path.name or fallback, used, fallback)
        target_path = input_target_dir / stored_name
        shutil.copy(snapshot.path, target_path)
        manifest.append({
            "file": stored_name,
            "name": snapshot.name or stored_name,
        })

    if manifest:
        manifest_path = input_target_dir / "input_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _append_history_entry(
    entry_id: str,
    variants: Sequence[OCRVariantArtifacts],
    display_filename: str,
    snapshots: Sequence[InputSnapshot],
    *,
    original_filenames: Optional[Sequence[str]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    metadata, entry_dir = _load_entry_metadata(entry_id)

    existing_entries: List[dict[str, object]] = []
    if isinstance(metadata.get("variants"), list):
        existing_entries = [entry for entry in metadata["variants"] if isinstance(entry, dict)]

    replacement_keys = {variant.key for variant in variants}
    filtered_entries = [entry for entry in existing_entries if entry.get("model") not in replacement_keys]
    new_entries = [_store_variant_entry(entry_dir, variant) for variant in variants]
    combined_entries = filtered_entries + new_entries

    combined_warnings: List[str] = []
    if isinstance(metadata.get("warnings"), list):
        combined_warnings.extend(str(item) for item in metadata["warnings"] if item)
    if warnings:
        combined_warnings.extend(str(item) for item in warnings if item)

    _ensure_input_snapshots(entry_dir, snapshots)

    input_files: List[str] = []
    existing_inputs = metadata.get("input_files")
    if isinstance(existing_inputs, list):
        input_files.extend(str(item) for item in existing_inputs if item)
    if not input_files and original_filenames:
        input_files.extend(str(name) for name in original_filenames if name)

    updated_metadata = _compose_history_metadata(
        entry_id,
        metadata.get("filename", display_filename),
        combined_entries,
        created_at=metadata.get("created_at"),
        input_files=input_files or None,
        warnings=combined_warnings,
    )

    (entry_dir / "metadata.json").write_text(
        json.dumps(updated_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return updated_metadata


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
        _, _, _, _, _, preview_url = _build_image_urls(entry_id, entry_dir, metadata)

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
    metadata, entry_dir = _load_entry_metadata(entry_id)
    input_images = _build_input_images(entry_id, entry_dir)

    variants_payload: List[dict[str, object]] = []
    variants_meta = metadata.get("variants")
    if isinstance(variants_meta, list) and variants_meta:
        for variant_meta in variants_meta:
            variants_payload.append(_build_variant_response(entry_id, entry_dir, metadata, variant_meta))
    else:
        descriptor = _descriptor_for(metadata.get("primary_model"))
        bounding_url, bounding_images, layout_url, layout_images, crops, preview_url = _build_image_urls(entry_id, entry_dir, metadata)
        variants_payload.append(
            {
                "model": descriptor.key,
                "label": descriptor.label,
                "text_plain": metadata.get("text_plain", ""),
                "text_markdown": metadata.get("text_markdown", ""),
                "bounding_image_url": bounding_url,
                "bounding_images": bounding_images,
                "layout_image_url": layout_url,
                "layout_images": layout_images,
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
        "bounding_images": primary_variant.get("bounding_images", []),
        "crops": primary_variant.get("crops", []),
        "preview_image_url": primary_variant.get("preview_image_url"),
        "preview": primary_variant.get("preview", metadata.get("preview", "")),
        "variants": variants_payload,
        "input_images": input_images,
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
