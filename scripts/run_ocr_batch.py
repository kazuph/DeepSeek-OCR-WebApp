#!/usr/bin/env python3
"""Batch OCR utility for DeepSeek-OCR.

This script loads the DeepSeek-OCR Hugging Face model once, then runs
`model.infer` over a list of image or PDF inputs. PDF files are rendered to
PNG images via PyMuPDF before inference. The decoded markdown outputs are
saved alongside optional JSON metadata for quick inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ocr_engine import DEFAULT_MODEL_NAME, OCRDocumentResult, run_ocr_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR on sample inputs")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to images or PDFs")
    parser.add_argument("--output-dir", default="/workspace/outputs", help="Directory for OCR outputs")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Hugging Face model id or path")
    parser.add_argument("--prompt", default="<image>\n<|grounding|>Convert the document to markdown.", help="Prompt passed to infer")
    parser.add_argument("--base-size", type=int, default=1024, help="Vision encoder base size")
    parser.add_argument("--image-size", type=int, default=640, help="Image tile size")
    parser.add_argument("--no-crop", action="store_true", help="Disable cropping mode")
    parser.add_argument("--attn-impl", default="eager", choices=["eager", "flash_attention_2", "sdpa"], help="Attention backend override")
    parser.add_argument("--save-json", action="store_true", help="Persist per-image JSON metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    crop_mode = not args.no_crop

    print(f"Running DeepSeek-OCR on {len(args.inputs)} input(s)…", flush=True)
    documents: list[OCRDocumentResult] = run_ocr_documents(
        [Path(p) for p in args.inputs],
        output_dir,
        model_name=args.model,
        prompt=args.prompt,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=crop_mode,
        attn_impl=args.attn_impl,
        save_json=args.save_json,
    )

    print(f"Found {len(documents)} processed page(s)", flush=True)

    summary = []
    for doc in documents:
        print(f"\n>>> OCR: {doc.image_path}", flush=True)
        preview = doc.text_markdown[:200]
        summary.append({"image": str(doc.image_path), "text_file": str(doc.text_path), "preview": preview})
        display_text = doc.text_markdown[:500]
        if len(doc.text_markdown) > 500:
            display_text += "…"
        print(display_text, flush=True)

    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote summary to {report_path}")


if __name__ == "__main__":
    main()
