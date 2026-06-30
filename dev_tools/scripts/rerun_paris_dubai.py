#!/usr/bin/env python3
"""用最新 GrabCut 管线重新分割巴黎 + 迪拜。

直接调用 labeler_server.run_segmentation，参数：
  - n_layers=3, frame_width=50, min_island_area=100
  - quality=standard → sam_driven + GrabCut → 文件后缀 _gc
"""
from __future__ import annotations

import sys, os, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "client_app" / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "dev_tools" / "scripts"
LABELER_DIR = PROJECT_ROOT / "dev_tools" / "labeler"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(LABELER_DIR))

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MODEL_DIR", str(BACKEND_DIR / "models"))
os.environ.setdefault("OUTPUT_DIR", str(BACKEND_DIR / "outputs"))

from labeler_server import run_segmentation, SegmentRequest, TEST_IMGS_DIR

IMAGES = ["巴黎.jpg", "迪拜.jpg"]

def main():
    for img_name in IMAGES:
        print(f"\n{'='*60}")
        print(f"🔄 处理: {img_name}")
        print(f"{'='*60}")
        t0 = time.perf_counter()

        req = SegmentRequest(
            image_name=img_name,
            n_layers=3,
            frame_width=50,
            min_island_area=100,
            quality="standard",
            refine_mode="sam_driven",
        )
        result = run_segmentation(req)
        elapsed = time.perf_counter() - t0

        print(f"✅ 完成 ({elapsed:.0f}s)")
        print(f"   叠加图: {result['overlay_url']}")
        print(f"   mask_key: {result['mask_key']}")
        print(f"   层数: {len(result['layers'])}")
        for l in result['layers']:
            print(f"     层{l['layer_index']}: {l['fg_pct']}% fg")
        if result.get('scores'):
            s = result['scores']
            print(f"   评分: {s.get('score_label', '?')} ({s.get('combined_score', '?'):.2f})")


if __name__ == "__main__":
    main()