"""Image upload endpoint — per API_CONTRACT.md § Step 1."""

import os
import uuid

import cv2
from fastapi import APIRouter, File, UploadFile, HTTPException
from loguru import logger

from app.models.responses import UploadResponse

router = APIRouter(tags=["upload"])

OUTPUTS_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Accept an image and save it to outputs/, returning image metadata.

    Field name must be 'file' (multipart/form-data).
    """
    # ── Validate extension ───────────────────────────────────────
    ext = os.path.splitext(file.filename or "unknown.jpg")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    if ext == ".jpeg":
        ext = ".jpg"

    # ── Read into memory ─────────────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(contents)} bytes — max {MAX_FILE_SIZE}",
        )

    # ── Reject empty files ────────────────────────────────────────
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file — upload aborted",
        )

    # ── Decode with OpenCV to get dimensions ─────────────────────
    np_arr = cv2.imdecode(
        __import__("numpy").frombuffer(contents, __import__("numpy").uint8),
        cv2.IMREAD_COLOR,
    )
    if np_arr is None:
        raise HTTPException(status_code=400, detail="Failed to decode image — corrupted or unsupported format")

    height, width = np_arr.shape[:2]

    # ── Generate image_id and save ───────────────────────────────
    image_id = uuid.uuid4().hex
    filename = f"{image_id}_original{ext}"
    save_path = os.path.join(OUTPUTS_DIR, filename)

    # WSL-safe atomic write: tmp → rename
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(contents)
    os.replace(tmp_path, save_path)

    logger.info(
        "Image uploaded | image_id={} name={} size={}x{} bytes={}",
        image_id, file.filename, width, height, len(contents),
    )

    return UploadResponse(
        image_id=image_id,
        width=width,
        height=height,
        original_url=f"/outputs/{filename}",
    )
