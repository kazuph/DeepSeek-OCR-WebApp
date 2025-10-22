from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.ocr_engine import (
    WEB_HISTORY_DIR,
    delete_history_entry,
    list_history_entries,
    load_history_entry,
    _load_entry_metadata,
    run_ocr_bytes,
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


@app.post("/api/ocr")
async def ocr_endpoint(file: UploadFile = File(...)) -> dict[str, object]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        result = run_ocr_bytes(content, file.filename or "upload.png")
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
async def history_bounding_image(entry_id: str) -> FileResponse:
    try:
        metadata, entry_dir = _load_entry_metadata(entry_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="History entry not found") from exc

    bounding_name = metadata.get("bounding_image")
    if not bounding_name:
        raise HTTPException(status_code=404, detail="Bounding image not found")

    path = (entry_dir / "artifacts" / bounding_name).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Bounding image not found")

    if WEB_HISTORY_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=403, detail="Forbidden")

    return FileResponse(path)


@app.get("/api/history/{entry_id}/image/crop/{filename}")
async def history_crop_image(entry_id: str, filename: str) -> FileResponse:
    try:
        metadata, entry_dir = _load_entry_metadata(entry_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="History entry not found") from exc

    allowed = set(metadata.get("crops", []))
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Crop not found")

    path = (entry_dir / "artifacts" / "images" / filename).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Crop not found")

    if WEB_HISTORY_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=403, detail="Forbidden")

    return FileResponse(path)
