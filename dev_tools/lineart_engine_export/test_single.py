"""
快速验证 — 对单张图片跑 canny_lineart 并输出结果。

用法:
    python test_single.py <图片路径>
    python test_single.py input.jpg --low 30 --high 100
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

from canny_lineart import canny_lineart

if __name__ == "__main__":
    # 参数
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    low = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--low" else 50
    high = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[4] == "--high" else 150

    if img_path is None:
        print("用法: python test_single.py <图片路径> [--low 50] [--high 150]")
        sys.exit(1)

    if not Path(img_path).exists():
        print(f"文件不存在: {img_path}")
        sys.exit(1)

    print(f"图片: {img_path}")
    print(f"参数: low={low} high={high} smooth_level=0")

    t0 = time.perf_counter()
    result = canny_lineart(img_path, low=low, high=high, smooth_level=0)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    fg_pct = (result > 0).mean() * 100
    print(f"耗时: {elapsed_ms}ms")
    print(f"输出: {result.shape}  (线条密度 {fg_pct:.2f}%)")

    # 保存
    stem = Path(img_path).stem
    out_path = f"{stem}_lineart.png"
    cv2.imwrite(out_path, result)
    print(f"保存: {out_path}")
