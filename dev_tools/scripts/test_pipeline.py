#!/usr/bin/env python3
"""End-to-end sandbox test: Canny → Connectivity → SVG.

Run on host (NO Docker, NO HTTP):
    python3 dev_tools/scripts/test_pipeline.py

Reads a pre-generated canny image, applies connectivity repair,
generates an SVG, and saves all outputs per dev_tools/outputs/README.md.
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
CONNECT_DIR = PROJECT_ROOT / "dev_tools" / "outputs" / "connectivity"
SVG_DIR = PROJECT_ROOT / "dev_tools" / "outputs" / "svg"

os.makedirs(CONNECT_DIR, exist_ok=True)
os.makedirs(SVG_DIR, exist_ok=True)

# ── Pick input image ────────────────────────────────────────────
test_image = None
for f in sorted(CANNY_DIR.glob("*_canny.png")):
    test_image = f
    break

if test_image is None:
    print("\u274c No *_canny.png found in dev_tools/outputs/canny/")
    print("   Run test_canny.py first to generate line-art images.")
    sys.exit(1)

stem = test_image.stem.replace("_canny", "")
print(f"\U0001f4f7 Input: {test_image.name}")

# ── Load binary image ───────────────────────────────────────────
img = cv2.imread(str(test_image), cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"\u274c Failed to load: {test_image}")
    sys.exit(1)
print(f"   Shape: {img.shape}  dtype: {img.dtype}  "
      f"foreground: {(img > 0).sum():,} px")

# ── Import engines ──────────────────────────────────────────────
try:
    from app.utils.connectivity import repair_connectivity
    from app.utils.svg_generator import generate_svg
except ImportError as e:
    print(f"\u274c Import failed: {e}")
    sys.exit(1)

# ── Step 6: Connectivity repair ─────────────────────────────────
print("\n\u267f Step 6: Connectivity repair (gap_tolerance=5)...")
t0 = time.perf_counter()

try:
    repaired = repair_connectivity(img, gap_tolerance=5)
except Exception as e:
    print(f"\u274c repair_connectivity error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

conn_elapsed = int((time.perf_counter() - t0) * 1000)
conn_px = (repaired > 0).sum()
orig_px = (img > 0).sum()
bridges_px = conn_px - orig_px

conn_path = CONNECT_DIR / f"{stem}_connected.png"
cv2.imwrite(str(conn_path), repaired)
print(f"   Done in {conn_elapsed} ms")
print(f"   Pixels before: {orig_px:,}  after: {conn_px:,}  "
      f"bridged: {bridges_px:,}")
print(f"\u2705 Saved: {conn_path}")

# ── Step 7: SVG generation ──────────────────────────────────────
print("\n\u267f Step 7: SVG generation (simplify_tolerance=1.0)...")
t0 = time.perf_counter()

try:
    svg_str = generate_svg(repaired, simplify_tolerance=1.0)
except Exception as e:
    print(f"\u274c generate_svg error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

svg_elapsed = int((time.perf_counter() - t0) * 1000)

# Count paths in SVG
path_count = svg_str.count('<path d=')
# Count 'C ' commands as Bezier segments (rough count)
bezier_count = svg_str.count('C ')

svg_path = SVG_DIR / f"{stem}.svg"
svg_path.write_text(svg_str, encoding="utf-8")
print(f"   Done in {svg_elapsed} ms")
print(f"   SVG size: {len(svg_str):,} bytes  "
      f"paths: {path_count}  bezier segments: {bezier_count}")
print(f"\u2705 Saved: {svg_path}")

# ── Summary ────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Pipeline complete: {stem}")
print(f"  Connectivity: {conn_elapsed} ms, {bridges_px:,} px bridged")
print(f"  SVG:          {svg_elapsed} ms, {path_count} paths")
print(f"  Total:        {conn_elapsed + svg_elapsed} ms")
