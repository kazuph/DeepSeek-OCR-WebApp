#!/usr/bin/env python3
"""Batch OCR utility for DeepSeek-OCR.

This script loads the DeepSeek-OCR Hugging Face model once, then runs
`model.infer` over a list of image or PDF inputs. PDF files are rendered to
PNG images via PyMuPDF before inference. The decoded markdown outputs are
saved alongside optional JSON metadata for quick inspection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import fitz  # type: ignore
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama import modeling_llama


# Provide a graceful fallback if the installed Transformers build lacks the
# optional flash-attention specialization expected by the released checkpoints.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR on sample inputs")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to images or PDFs")
    parser.add_argument("--output-dir", default="/workspace/outputs", help="Directory for OCR outputs")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR", help="Hugging Face model id or path")
    parser.add_argument("--prompt", default="<image>\n<|grounding|>Convert the document to markdown.", help="Prompt passed to infer")
    parser.add_argument("--base-size", type=int, default=1024, help="Vision encoder base size")
    parser.add_argument("--image-size", type=int, default=640, help="Image tile size")
    parser.add_argument("--no-crop", action="store_true", help="Disable cropping mode")
    parser.add_argument("--attn-impl", default="eager", choices=["eager", "flash_attention_2", "sdpa"], help="Attention backend override")
    parser.add_argument("--save-json", action="store_true", help="Persist per-image JSON metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    text_dir = output_dir / "texts"
    text_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(p).expanduser().resolve() for p in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

    print(f"Loading model '{args.model}' on {device}…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    load_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        args.model,
        _attn_implementation=args.attn_impl,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=load_dtype,
    )

    if device == "cuda":
        model = model.eval().to(device="cuda", dtype=torch.bfloat16)
    else:
        model = model.eval().to(torch.float32)

    image_paths = collect_image_paths(input_paths, output_dir)
    print(f"Found {len(image_paths)} page(s) to process", flush=True)

    crop_mode = not args.no_crop

    summary = []
    for image_path in image_paths:
        print(f"\n>>> OCR: {image_path}", flush=True)
        image_output_dir = artifact_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        result = model.infer(
            tokenizer,
            prompt=args.prompt,
            image_file=str(image_path),
            output_path=str(image_output_dir),
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=crop_mode,
            save_results=True,
            test_compress=True,
        )

        text = as_text(result).strip()
        result_candidates = [text]
        result_mmd = image_output_dir / "result.mmd"
        if result_mmd.exists():
            result_candidates.append(result_mmd.read_text(encoding="utf-8"))
        result_md = image_output_dir / "result.md"
        if result_md.exists():
            result_candidates.append(result_md.read_text(encoding="utf-8"))
        resolved_text = next(
            (c.strip() for c in result_candidates if c and c.strip() and c.strip().lower() != "none"),
            "",
        )

        text_file = text_dir / f"{image_path.stem}.md"
        text_file.write_text(resolved_text, encoding="utf-8")
        summary.append({"image": str(image_path), "text_file": str(text_file), "preview": resolved_text[:200]})
        print(resolved_text[:500] + ("…" if len(resolved_text) > 500 else ""), flush=True)

        if args.save_json:
            json_path = text_dir / f"{image_path.stem}.json"
            json_path.write_text(json.dumps({"raw": result}, default=str, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote summary to {report_path}")


if __name__ == "__main__":
    main()
