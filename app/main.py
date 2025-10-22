from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.ocr_engine import run_ocr_bytes

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
