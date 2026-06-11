"""
批量处理 — 对目录下所有图片批量跑 canny_lineart。

用法:
    python batch_run.py <输入目录> [输出目录]
    python batch_run.py ./test_imgs ./output --low 30 --high 100
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2

from canny_lineart import canny_lineart

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    import argparse

    p = argparse.ArgumentParser(description="Canny LineArt 批量处理")
    p.add_argument("input_dir", help="输入图片目录")
    p.add_argument("output_dir", nargs="?", default=None,
                   help="输出目录 (默认: <input_dir>_lineart)")
    p.add_argument("--low", type=int, default=50, help="Canny 低阈值 (默认 50)")
    p.add_argument("--high", type=int, default=150, help="Canny 高阈值 (默认 150)")
    p.add_argument("--smooth", type=int, default=0, choices=[0, 1, 2],
                   help="平滑强度 (默认 0)")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        print(f"目录不存在: {in_dir}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else Path(f"{in_dir}_lineart")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集图片
    images = []
    for f in sorted(in_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
            images.append(f)

    if not images:
        print(f"目录中没有图片: {in_dir}")
        sys.exit(1)

    print(f"输入: {in_dir} ({len(images)} 张)")
    print(f"输出: {out_dir}")
    print(f"参数: low={args.low} high={args.high} smooth={args.smooth}")
    print()

    total_ms = 0
    ok = 0

    for idx, img_path in enumerate(images, 1):
        stem = img_path.stem
        out_path = out_dir / f"{stem}_lineart.png"
        rel = img_path.relative_to(in_dir)

        print(f"[{idx}/{len(images)}] {rel} ... ", end="", flush=True)

        try:
            t0 = time.perf_counter()
            result = canny_lineart(
                str(img_path),
                low=args.low,
                high=args.high,
                smooth_level=args.smooth,
            )
            elapsed = int((time.perf_counter() - t0) * 1000)
            cv2.imwrite(str(out_path), result)
            fg_pct = (result > 0).mean() * 100
            print(f"OK  {elapsed}ms  fg={fg_pct:.1f}%")
            total_ms += elapsed
            ok += 1
        except Exception as e:
            print(f"FAIL: {e}")

    if ok:
        print(f"\n完成: {ok}/{len(images)} 张, 平均 {total_ms // ok}ms/张")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
