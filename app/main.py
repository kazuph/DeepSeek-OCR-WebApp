from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.ocr_engine import (
    WEB_HISTORY_DIR,
    MODEL_DESCRIPTORS,
    delete_history_entry,
    list_history_entries,
    load_history_entry,
    run_ocr_bytes,
    select_variant,
    _load_entry_metadata,
)

app = FastAPI(title="DeepSeek OCR Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def _resolve_history_file(entry_dir: Path, relative_path: str, variant_key: Optional[str] = None) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=403, detail="Forbidden")

    candidates = []
    if variant_key:
        candidates.append(entry_dir / variant_key / rel)
    candidates.append(entry_dir / rel)
    candidates.append(entry_dir / "artifacts" / rel)
    candidates.append(entry_dir / "artifacts" / "images" / rel)

    base = entry_dir.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.exists() or not resolved.is_file():
            continue
        if resolved == base or base in resolved.parents:
            return resolved

    raise HTTPException(status_code=404, detail="File not found")


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    index_path = static_dir / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(static_dir / "favicon.svg")


@app.get("/api/ping")
async def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
async def models_list() -> List[dict[str, str]]:
    return [
        {
            "key": descriptor.key,
            "label": descriptor.label,
            "description": descriptor.description,
        }
        for descriptor in MODEL_DESCRIPTORS
    ]


@app.post("/api/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    models: str | None = Form(default=None),
) -> dict[str, object]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        prompt_value = (prompt or "").strip() or None
        models_value = (models or "").strip() or None
        kwargs: dict[str, object] = {}
        if prompt_value is not None:
            kwargs["prompt"] = prompt_value
        if models_value:
            kwargs["models"] = models_value
        result = run_ocr_bytes(content, file.filename or "upload.png", **kwargs)
    except Exception as exc:  # noqa: BLE001 - surface to client
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc

    return result


@app.get("/api/history")
async def history_list() -> List[dict[str, object]]:
    return list_history_entries()


@app.get("/api/history/{entry_id}")
async def history_detail(entry_id: str) -> dict[str, object]:
    try:
        return load_history_entry(entry_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="History entry not found") from exc


@app.delete("/api/history/{entry_id}")
async def history_delete(entry_id: str) -> dict[str, str]:
    try:
        delete_history_entry(entry_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="History entry not found") from exc
    return {"status": "deleted"}


@app.get("/api/history/{entry_id}/image/bounding")
async def history_bounding_image(entry_id: str, model: str | None = Query(default=None)) -> FileResponse:
    try:
        metadata, entry_dir = _load_entry_metadata(entry_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="History entry not found") from exc

    variant_meta, variant_key = select_variant(metadata, model)
    bounding_name = variant_meta.get("bounding_image")
    if not bounding_name:
        raise HTTPException(status_code=404, detail="Bounding image not found")

    path = _resolve_history_file(entry_dir, bounding_name, variant_key)
    return FileResponse(path)


@app.get("/api/history/{entry_id}/image/crop/{file_path:path}")
async def history_crop_image(entry_id: str, file_path: str, model: str | None = Query(default=None)) -> FileResponse:
    try:
        metadata, entry_dir = _load_entry_metadata(entry_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="History entry not found") from exc

    variant_meta, variant_key = select_variant(metadata, model)

    allowed_raw = variant_meta.get("crops", [])
    if not allowed_raw and variant_meta is not metadata:
        allowed_raw = metadata.get("crops", [])

    allowed = {str(item) for item in allowed_raw} if isinstance(allowed_raw, list) else set()
    candidate = file_path
    if candidate not in allowed and Path(candidate).name in allowed:
        candidate = Path(candidate).name

    if candidate not in allowed:
        raise HTTPException(status_code=404, detail="Crop not found")

    path = _resolve_history_file(entry_dir, candidate, variant_key)
    return FileResponse(path)


@app.get("/api/history/{entry_id}/image/input/{file_path:path}")
async def history_input_image(entry_id: str, file_path: str) -> FileResponse:
    try:
        _, entry_dir = _load_entry_metadata(entry_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="History entry not found") from exc

    if not (entry_dir / "input").exists():
        raise HTTPException(status_code=404, detail="Input image not found")

    relative = Path("input") / file_path
    path = _resolve_history_file(entry_dir, relative.as_posix())
    return FileResponse(path)
