#!/usr/bin/env python3
"""Bare-metal Canny edge-detection sandbox test.

Run directly on the host (NO Docker, NO HTTP):
    python3 dev_tools/scripts/test_canny.py

This script dynamically adds client_app/backend to sys.path so it can
import the canny_lineart engine without polluting the Web stack.
"""

import os
import sys
import time
from pathlib import Path

import cv2

# ── Resolve project root and inject backend into sys.path ─────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # lzaer_engraver_2.0/
BACKEND_DIR = PROJECT_ROOT / "client_app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# ── Pick a test image ─────────────────────────────────────────────
TEST_IMGS_DIR = PROJECT_ROOT / "dev_tools" / "test_imgs"
OUTPUT_DIR = PROJECT_ROOT / "dev_tools" / "outputs" / "canny"

os.makedirs(OUTPUT_DIR, exist_ok=True)

test_image = None
for f in sorted(TEST_IMGS_DIR.iterdir()):
    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
        test_image = f
        break

if test_image is None:
    print("\u274c No test images found in dev_tools/test_imgs/")
    sys.exit(1)

print(f"\U0001f4f7 Test image: {test_image.name}")

# ── Import the engine ─────────────────────────────────────────────
try:
    from app.utils.canny_lineart import canny_lineart
except ImportError as e:
    print(f"\u274c Cannot import canny_lineart: {e}")
    print("   Check that app/utils/canny_lineart.py exists.")
    sys.exit(1)

# ── Load image as OpenCV BGR numpy array ──────────────────────────
image_path = str(test_image)
image = cv2.imread(image_path)
if image is None:
    print(f"\u274c cv2.imread() returned None for: {image_path}")
    sys.exit(1)
print(f"   Shape: {image.shape}  dtype: {image.dtype}")

# ── Run extraction ────────────────────────────────────────────────
print("\U0001f3a8 Running canny_lineart() with defaults (low=50, high=150, smooth=0)...")
t0 = time.perf_counter()

try:
    result = canny_lineart(image, low=50, high=150, smooth_level=0)
except Exception as e:
    print(f"\u274c canny_lineart() raised an error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

elapsed_ms = int((time.perf_counter() - t0) * 1000)
fg_pct = (result > 0).mean() * 100

# ── Save result ───────────────────────────────────────────────────
stem = test_image.stem
out_path = OUTPUT_DIR / f"{stem}_canny.png"
cv2.imwrite(str(out_path), result)

print(f"   Done in {elapsed_ms} ms")
print(f"\u2705 Result saved to: {out_path}")
print(f"   Result shape: {result.shape}  dtype: {result.dtype}")
print(f"   Edge density: {fg_pct:.1f}%")
