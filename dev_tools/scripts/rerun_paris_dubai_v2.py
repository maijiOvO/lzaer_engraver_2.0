#!/usr/bin/env python3
"""按 outputs/README.md 规范重新分割巴黎 + 迪拜（导向滤波版）。

输出结构：
    dev_tools/outputs/sam/{巴黎,迪拜}/
    ├── depth/
    │   ├── {城市}_depth.npy
    │   └── {城市}_depth.png        # 热力图
    └── n3_f50_i100_gf/             # n3_f50_i100_gf (导向滤波)
        ├── {城市}_n3_f50_i100_gf_frame_0~2.png
        ├── {城市}_n3_f50_i100_gf_mask_0~2.png
        └── {城市}_n3_f50_i100_gf.png   # 叠加图
"""
from __future__ import annotations

import sys, os, time, cv2, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "client_app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MODEL_DIR", str(BACKEND_DIR / "models"))
os.environ.setdefault("OUTPUT_DIR", str(BACKEND_DIR / "outputs"))

from app.utils.depth_engine import estimate_depth
from app.utils.sam_engine import run_sam_automatic, _snap_mask_to_edges
from app.utils.structural_segmentation import (
    build_sam_driven_layers, suggest_n_layers,
)

TEST_IMGS_DIR = PROJECT_ROOT / "dev_tools" / "test_imgs" / "train"
SAM_OUTPUT_DIR = PROJECT_ROOT / "dev_tools" / "outputs" / "sam"

IMAGES = [
    ("巴黎.jpg", "巴黎"),
    ("迪拜.jpg", "迪拜"),
]

PARAMS = {
    "n_layers": 3,
    "frame_width": 50,
    "min_island_area": 100,
    "method": "gf",
}

LAYER_COLORS = [
    (231, 76, 60),    # red
    (46, 204, 113),   # green
    (52, 152, 219),   # blue
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
]


def render_overlay(image, layer_masks, alpha=0.4):
    h, w = image.shape[:2]
    canvas = image.copy().astype(np.float32)
    for i, mask in enumerate(layer_masks):
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        color_arr = np.array(color, dtype=np.float32)
        fg = mask > 0
        canvas[fg] = canvas[fg] * (1 - alpha) + color_arr * alpha
    return np.clip(canvas, 0, 255).astype(np.uint8)


def process_city(img_name: str, city_name: str):
    img_path = TEST_IMGS_DIR / img_name
    if not img_path.exists():
        print(f"\u274c 图片不存在: {img_path}")
        return

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"\u274c 无法解码: {img_path}")
        return

    orig_h, orig_w = image.shape[:2]
    stem = city_name  # keep Chinese city name in filenames per existing convention
    n = PARAMS["n_layers"]
    fw = PARAMS["frame_width"]
    mi = PARAMS["min_island_area"]
    method = PARAMS["method"]
    param_tag = f"n{n}_f{fw}_i{mi}_{method}"
    param_dir = f"n{n}_f{fw}_i{mi}_{method}"

    # ── Output directories (README v2: version-first) ────
    ver_dir = SAM_OUTPUT_DIR / param_dir
    city_dir = ver_dir / city_name
    depth_dir = city_dir / "depth"
    ver_dir.mkdir(parents=True, exist_ok=True)
    city_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"\U0001f4f7 {city_name} ({orig_w}\u00d7{orig_h}) \u2192 sam/{city_name}/{param_dir}/")
    print(f"{'='*60}")

    # ── Step 1: Depth estimation ────────────────────────
    depth_npy = depth_dir / f"{stem}_depth.npy"
    depth_png = depth_dir / f"{stem}_depth.png"

    if depth_npy.exists() and depth_png.exists():
        print("   \U0001f4e6 深度缓存命中")
        depth_map = np.load(str(depth_npy))
    else:
        t0 = time.perf_counter()
        depth_map = estimate_depth(image)
        dt = time.perf_counter() - t0
        print(f"   深度估计: {dt:.1f}s")
        np.save(str(depth_npy), depth_map)
        # Save heatmap
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-8:
            depth_u8 = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_u8 = np.zeros_like(depth_map, dtype=np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
        cv2.imwrite(str(depth_png), depth_color)
        print(f"   \U0001f4be 深度保存: depth.npy + depth.png")

    # ── Step 2: SAM auto-segment (force re-run, no cache) ──
    sam_cache = city_dir / f"{stem}_sam_region.npz"
    print("   \U0001f52c SAM 推理 \u2026")
    t0 = time.perf_counter()
    sam_masks, _rm = run_sam_automatic(image, cache_path=str(sam_cache))
    print(f"   SAM 完成: {len(sam_masks)} masks, {time.perf_counter()-t0:.1f}s")

    # ── Step 3: SAM-driven layering ─────────────────────
    print(f"   \U0001f4d0 分层 (n={n}, frame={fw}px, island>{mi}px\u00b2) \u2026")
    layer_masks, frame_mask, stats = build_sam_driven_layers(
        sam_masks, depth_map,
        n_layers=n,
        image_shape=(orig_h, orig_w),
        frame_width=fw,
        min_island_area=mi,
        skip_connectivity_repair=True,
    )

    # ── Step 4: Guided Filter edge snap ─────────────────
    print("   \U0001f9f2 导向滤波 边缘吸附 \u2026")
    frame_bin = frame_mask > 0
    gf_success = 0
    gf_fallback = 0
    for i, mask in enumerate(layer_masks):
        if not mask.any():
            continue
        content_only = (mask > 0) & ~frame_bin
        if content_only.sum() < 10:
            continue
        t0 = time.perf_counter()
        try:
            snapped = _snap_mask_to_edges(content_only, image, edge_band=5)
            ms = time.perf_counter() - t0
            before = content_only.sum()
            after = snapped.sum()
            snapped_u8 = snapped.astype(np.uint8) * 255
            snapped_u8[frame_bin] = 255
            layer_masks[i] = snapped_u8
            stats[i]["fg_pixels"] = int(np.count_nonzero(snapped_u8))
            stats[i]["fg_pct"] = round(stats[i]["fg_pixels"] / (orig_h * orig_w) * 100, 2)
            if abs(after - before) > before * 0.01:  # >1% change means GF worked
                gf_success += 1
            else:
                gf_fallback += 1
            print(f"      层{i}: {before}\u2192{after} ({after-before:+d}px, {ms*1000:.0f}ms)")
        except Exception as e:
            gf_fallback += 1
            print(f"      层{i}: 失败 fallback ({e})")
    print(f"   导向滤波: {gf_success} 层有效 / {gf_fallback} 层 fallback")

    # ── Step 5: uint8 normalize ─────────────────────────
    for i, mask in enumerate(layer_masks):
        if mask.dtype == bool:
            layer_masks[i] = mask.astype(np.uint8) * 255
        elif mask.dtype != np.uint8:
            layer_masks[i] = mask.astype(np.uint8)

    # ── Step 6: Pad image to match padded layer masks ───
    if frame_mask.shape != image.shape[:2]:
        image_padded = cv2.copyMakeBorder(
            image, fw, fw, fw, fw, cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
    else:
        image_padded = image

    # ── Step 7: Save outputs ────────────────────────────
    file_prefix = f"{stem}_{param_tag}"
    print(f"\n   \U0001f4be 保存为 {file_prefix}_*")

    # Overlay
    overlay = render_overlay(image_padded, layer_masks)
    overlay_path = city_dir / f"{file_prefix}.png"
    cv2.imwrite(str(overlay_path), overlay)

    # Per-layer mask + frame
    for rank, mask in enumerate(layer_masks):
        mask_path = city_dir / f"{file_prefix}_mask_{rank}.png"
        cv2.imwrite(str(mask_path), mask)

        pure = mask.copy()
        pure[frame_mask > 0] = 0
        frame_path = city_dir / f"{file_prefix}_frame_{rank}.png"
        cv2.imwrite(str(frame_path), pure)

        fg_pct = round(np.count_nonzero(mask) / mask.size * 100, 1)
        print(f"      层{rank}: {fg_pct}% fg")

    # Manifest
    suggested_n = int(suggest_n_layers(depth_map))
    manifest = {
        "city": city_name,
        "image": img_name,
        "params": PARAMS,
        "suggested_n_layers": suggested_n,
        "actual_n_layers": n,
        "image_shape": [orig_h, orig_w],
        "sam_masks_count": len(sam_masks),
        "guided_filter_success_layers": gf_success,
        "guided_filter_fallback_layers": gf_fallback,
        "layer_stats": stats,
        "output_dir": str(city_dir.relative_to(PROJECT_ROOT)),
    }
    import json
    manifest_path = city_dir / f"{file_prefix}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"   \u2705 {city_name} 完成 \u2192 {city_dir.relative_to(PROJECT_ROOT)}")
    print(f"   文件数: {1 + len(layer_masks)*2 + 1} (1 overlay + {len(layer_masks)*2} masks + 1 manifest)")


def main():
    for img_name, city_name in IMAGES:
        process_city(img_name, city_name)
    print(f"\n{'='*60}")
    print("\U0001f3c1 全部完成")


if __name__ == "__main__":
    main()