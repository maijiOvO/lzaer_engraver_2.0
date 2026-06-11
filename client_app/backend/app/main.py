import sys
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# ── Loguru configuration ──────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)
logger.add(
    "/tmp/backend.log",
    rotation="10 MB",
    retention="3 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
)

app = FastAPI(title="Laser Engraver V2", version="2.0")

# ── CORS: allow frontend dev server (Vite on 5173) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files: serve /outputs/ from the outputs directory ──────
import os
OUTPUTS_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


# ── Global Exception Handler (with full traceback via loguru) ──────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": type(exc).__name__,
            "error_msg": tb,
        },
    )


# ── Routers ───────────────────────────────────────────────────────
from app.api.lineart import router as lineart_router
app.include_router(lineart_router, prefix="/api")


# ── Health endpoint (per API_CONTRACT.md § Step 0) ────────────────
@app.get("/api/health")
async def health():
    logger.info("Health check requested")
    return {"status": "ok", "version": "2.0"}


# ── Startup ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("🚀 Laser Engraver V2 backend started")
