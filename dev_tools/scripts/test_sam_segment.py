#!/usr/bin/env python3
"""Sandbox test: SAM segmentation → Layer clustering (Steps 2-3).

Run on host (bare-metal Python with torch/mobile-sam installed):
    python3 dev_tools/scripts/test_sam_segment.py

Takes the first available test image, runs SAM fragment extraction,
clusters into N layers, and saves masks + overlay.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Project root and path injection ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "client_app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Set MODEL_DIR to the project's persistent models directory
os.environ.setdefault(
    "MODEL_DIR",
    str(PROJECT_ROOT / "client_app" / "backend" / "models"),
)

# ── Directories ─────────────────────────────────────────────────
TEST_IMGS_DIR = PROJECT_ROOT / "dev_tools" / "test_imgs"
OUTPUT_DIR = PROJECT_ROOT / "dev_tools" / "outputs" / "sam"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Pick input image ────────────────────────────────────────────
test_image = None
for f in sorted(TEST_IMGS_DIR.glob("*")):
    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
        test_image = f
        break

if test_image is None:
    print("❌ No test image found in dev_tools/test_imgs/")
    sys.exit(1)

stem = test_image.stem
print(f"📷 Input: {test_image.name}")

# ── Load image ──────────────────────────────────────────────────
image = cv2.imread(str(test_image))
if image is None:
    print(f"❌ Failed to load: {test_image}")
    sys.exit(1)
print(f"   Shape: {image.shape}  dtype: {image.dtype}")

# ── Import SAM engine ───────────────────────────────────────────
try:
    from app.utils.sam_engine import run_sam_automatic
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("   Ensure torch and mobile-sam are installed in the host Python.")
    sys.exit(1)

# ── Step 2: SAM fragment extraction ─────────────────────────────
print("\n🔬 Step 2: SAM AutomaticMaskGenerator (paper_sculpture)...")
t0 = time.perf_counter()

try:
    masks, region_map = run_sam_automatic(image)
except Exception as e:
    print(f"❌ SAM error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

sam_elapsed = int((time.perf_counter() - t0) * 1000)
n_fragments = int(region_map.max())
print(f"   Done in {sam_elapsed} ms — {n_fragments} fragments")

# Save region_map visualization
region_viz = (region_map.astype(np.float32) / max(n_fragments, 1) * 255).astype(np.uint8)
region_viz_color = cv2.applyColorMap(region_viz, cv2.COLORMAP_JET)
region_path = OUTPUT_DIR / f"{stem}_regions.png"
cv2.imwrite(str(region_path), region_viz_color)
print(f"✅ Region map saved: {region_path}")

# ── Import clustering helpers ───────────────────────────────────
from app.services.segmentation_service import (
    _extract_fragment_features,
    _cluster_kmeans,
    _sort_layers_front_to_back,
    _merge_small_layers,
    _build_layer_masks,
    _render_overlay,
)

# ── Step 3: Layer clustering ────────────────────────────────────
if n_fragments == 0:
    print("❌ Zero fragments — nothing to cluster")
    sys.exit(1)

print("\n🔬 Step 3: K-means clustering into 3 layers...")
features = _extract_fragment_features(region_map, image)
print(f"   Features shape: {features.shape}")

n_layers = min(3, n_fragments)
labels = _cluster_kmeans(features, n_layers, "balanced")
labels = _merge_small_layers(features, labels, n_layers, 5.0, 1.0)

# Remap labels
unique = sorted(set(labels.tolist()))
label_map = {old: new for new, old in enumerate(unique)}
labels = np.array([label_map[l] for l in labels], dtype=np.int32)
n_actual = len(unique)

layer_order = _sort_layers_front_to_back(features, labels, n_actual)
print(f"   Layers: {n_actual} (front→back order: {layer_order})")

layer_masks = _build_layer_masks(region_map, labels, layer_order)

# Save per-layer masks
for rank, lid in enumerate(layer_order):
    mask_path = OUTPUT_DIR / f"{stem}_mask_{rank}.png"
    cv2.imwrite(str(mask_path), layer_masks[rank])
    area_px = np.count_nonzero(layer_masks[rank])
    area_pct = area_px / (image.shape[0] * image.shape[1]) * 100
    print(f"   Layer {rank} saved: {mask_path.name}  "
          f"area={area_px:,}px ({area_pct:.1f}%)")

# Save overlay
overlay = _render_overlay(image, layer_masks)
overlay_path = OUTPUT_DIR / f"{stem}_segmented.png"
cv2.imwrite(str(overlay_path), overlay)
print(f"✅ Overlay saved: {overlay_path}")

# ── Summary ────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"SAM segmentation complete: {stem}")
print(f"  Fragments: {n_fragments}")
print(f"  Layers:    {n_actual}")
print(f"  SAM time:  {sam_elapsed} ms")
print(f"  Outputs:   {OUTPUT_DIR}/")
