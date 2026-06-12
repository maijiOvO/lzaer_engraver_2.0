#!/usr/bin/env python3
"""Sandbox test: Canny lineart → Denoise (Step 5).

Run on host (NO Docker, NO HTTP):
    python3 dev_tools/scripts/test_denoise.py

Reads a pre-generated canny image, applies connected-component area
filtering, saves the denoised result to dev_tools/outputs/denoise/.
"""

import os
import sys
import time
from pathlib import Path

import cv2

# ── Project root and path injection ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "client_app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# ── Directories ─────────────────────────────────────────────────
CANNY_DIR = PROJECT_ROOT / "dev_tools" / "outputs" / "canny"
DENOISE_DIR = PROJECT_ROOT / "dev_tools" / "outputs" / "denoise"

os.makedirs(DENOISE_DIR, exist_ok=True)

# ── Pick input image ────────────────────────────────────────────
test_image = None
for f in sorted(CANNY_DIR.glob("*_canny.png")):
    test_image = f
    break

if test_image is None:
    print("❌ No *_canny.png found in dev_tools/outputs/canny/")
    print("   Run test_canny.py first to generate line-art images.")
    sys.exit(1)

stem = test_image.stem.replace("_canny", "")
print(f"📷 Input: {test_image.name}")

# ── Load binary image ───────────────────────────────────────────
img = cv2.imread(str(test_image), cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"❌ Failed to load: {test_image}")
    sys.exit(1)
print(f"   Shape: {img.shape}  dtype: {img.dtype}  "
      f"foreground: {(img > 0).sum():,} px")

# ── Import denoise engine ───────────────────────────────────────
try:
    from app.utils.denoise import denoise_binary
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# ── Test multiple thresholds ─────────────────────────────────────
thresholds = [1, 4, 10, 25, 50]
print(f"\n🧪 Testing min_component_area = {thresholds}")

for min_area in thresholds:
    t0 = time.perf_counter()

    try:
        denoised = denoise_binary(img, min_component_area=min_area)
    except Exception as e:
        print(f"   ❌ min_area={min_area}: {e}")
        continue

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    before_px = int((img > 0).sum())
    after_px = int((denoised > 0).sum())
    removed_px = before_px - after_px

    out_path = DENOISE_DIR / f"{stem}_denoised_a{min_area}.png"
    cv2.imwrite(str(out_path), denoised)

    print(f"   min_area={min_area:>3}: {before_px:,} → {after_px:,} px "
          f"(-{removed_px:,}, {removed_px*100/before_px:.1f}%)  "
          f"{elapsed_ms} ms  → {out_path.name}")

print(f"\n✅ All denoise results saved to {DENOISE_DIR}/")
