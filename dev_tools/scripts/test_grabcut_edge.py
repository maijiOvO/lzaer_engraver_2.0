#!/usr/bin/env python3
"""GrabCut 边缘吸附独立测试 — 巴黎 + 迪拜。

直接调用 sam_engine 的 _snap_mask_to_edges 验证细边缘保留效果。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "client_app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.utils.sam_engine import (
    run_sam_automatic,
    _snap_mask_to_edges,
    _upscale_mask_smooth,
)

TEST_IMGS = PROJECT_ROOT / "dev_tools" / "test_imgs"
OUTPUT_DIR = PROJECT_ROOT / "dev_tools" / "outputs" / "sam"

IMAGES = [
    ("train/巴黎.jpg", "巴黎"),
    ("train/迪拜.jpg", "迪拜"),
]


def test_grabcut_on_image(img_rel: str, stem: str):
    """完整管线：SAM → 上采样 → GrabCut 边缘精修 → 保存对比图。"""
    img_path = TEST_IMGS / img_rel
    if not img_path.exists():
        print(f"❌ 图片不存在: {img_path}")
        return

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"❌ 无法解码: {img_path}")
        return
    orig_h, orig_w = image.shape[:2]
    print(f"\n{'='*60}")
    print(f"📷 {stem} ({orig_w}×{orig_h})")
    print(f"{'='*60}")

    # ── SAM 自动分割 ────────────────────────────────
    t0 = time.perf_counter()
    sam_cache = OUTPUT_DIR / f"{stem}_sam_region.npz"
    if sam_cache.exists():
        print("   📦 SAM 缓存命中，清除以强制重新推理...")
        sam_cache.unlink()

    masks, _rm = run_sam_automatic(image, cache_path=str(sam_cache))
    print(f"   SAM 推理: {len(masks)} masks, {time.perf_counter()-t0:.1f}s")

    if not masks:
        print("   ⚠ 无 mask，跳过")
        return

    # ── 对所有 mask 逐一测试 GrabCut ────────────────
    gc_total_ms = 0.0
    before_px_total = 0
    after_px_total = 0
    grabcut_failures = 0
    thin_preserved = 0

    for i, mask_data in enumerate(masks[:10]):  # 只测试前10个（性能考虑）
        seg_bool = mask_data["segmentation"]
        h_sam, w_sam = seg_bool.shape[:2]

        # 上采样到原图分辨率
        if (h_sam, w_sam) != (orig_h, orig_w):
            seg_bool = _upscale_mask_smooth(seg_bool, orig_h, orig_w)

        before_px = seg_bool.sum()

        gc_t0 = time.perf_counter()
        try:
            refined = _snap_mask_to_edges(seg_bool, image, edge_band=3)
            gc_ms = (time.perf_counter() - gc_t0) * 1000
            gc_total_ms += gc_ms
            after_px = refined.sum()
            before_px_total += before_px
            after_px_total += after_px

            # 检测是否保留了细结构：如果 GrabCut 后像素比上采样后还多 → 保留了细结构
            if after_px >= before_px * 0.99:
                thin_preserved += 1
        except Exception as e:
            grabcut_failures += 1
            print(f"   ⚠ mask {i} GrabCut 失败: {e}")

    avg_ms = gc_total_ms / min(10, len(masks)) if gc_total_ms > 0 else 0
    print(f"   GrabCut 平均耗时: {avg_ms:.1f}ms/mask")
    print(f"   GrabCut 失败: {grabcut_failures}/10")
    print(f"   像素保有率: {after_px_total}/{before_px_total} = {(after_px_total/before_px_total*100) if before_px_total > 0 else 0:.1f}%")
    print(f"   细结构保留(≥99%像素): {thin_preserved}/10 masks")

    # ── 对第一个 mask 做详细的 A/B 对比 ──────────────
    if len(masks) > 0:
        print(f"\n   ── mask[0] A/B 对比 ──")
        mask_data = masks[0]
        seg_bool = mask_data["segmentation"]
        h_sam, w_sam = seg_bool.shape[:2]
        if (h_sam, w_sam) != (orig_h, orig_w):
            seg_bool = _upscale_mask_smooth(seg_bool, orig_h, orig_w)

        # 上采样版 (无边缘精修)
        seg_u8 = seg_bool.astype(np.uint8) * 255
        cv2.imwrite(str(OUTPUT_DIR / f"{stem}_grabcut_before_mask0.png"), seg_u8)

        # GrabCut 版
        refined = _snap_mask_to_edges(seg_bool, image, edge_band=3)
        refined_u8 = refined.astype(np.uint8) * 255
        cv2.imwrite(str(OUTPUT_DIR / f"{stem}_grabcut_after_mask0.png"), refined_u8)

        # 差异图 (绿色=GrabCut新增, 红色=GrabCut移除)
        diff = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        added = refined & ~seg_bool
        removed = seg_bool & ~refined
        diff[added] = (0, 255, 0)      # 绿色
        diff[removed] = (0, 0, 255)    # 红色
        cv2.imwrite(str(OUTPUT_DIR / f"{stem}_grabcut_diff_mask0.png"), diff)

        before = seg_bool.sum()
        after = refined.sum()
        print(f"   上采样像素数: {before}")
        print(f"   GrabCut后像素数: {after} ({after-before:+d}, {(after-before)/max(1,before)*100:+.2f}%)")
        print(f"   GrabCut新增: {added.sum()} px")
        print(f"   GrabCut移除: {removed.sum()} px")
        print(f"   💾 对比图已保存: {stem}_grabcut_before_mask0.png / _after_mask0.png / _diff_mask0.png")

    print(f"   ✅ {stem} 完成")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for img_rel, stem in IMAGES:
        test_grabcut_on_image(img_rel, stem)
    print(f"\n{'='*60}")
    print("🏁 全部完成")


if __name__ == "__main__":
    main()