"""
Canny LineArt 线稿引擎 — 零项目依赖，可直接复制到任何项目使用。

这是 laser-engraver-app 项目中经过 117 张图片批量对比验证的最优线稿算法。
通过 CLAHE 局部对比度增强 + Canny 边缘检测，在城市场景、自然风景、建筑
等多种题材上均表现突出。

依赖 (pip install):
    opencv-python numpy

用法:
    from canny_lineart import canny_lineart

    # 从文件路径
    result = canny_lineart("input.jpg")
    cv2.imwrite("output.png", result)

    # 从 numpy 数组 (BGR)
    import cv2
    img = cv2.imread("input.jpg")
    result = canny_lineart(img)

    # 调参
    result = canny_lineart(img, low=30, high=100)  # 更多细节
    result = canny_lineart(img, smooth_level=1)    # 去除微小纹理

核心算法:
    BGR → Gray → CLAHE(2.0, 8×8) → 平滑 → Canny(low, high) → 二值图

参数:
    low=50, high=150        最佳默认值（经过 117 图批量验证）
    smooth_level=0          原始细节 / 1=轻量去噪 / 2=中等平滑
"""

from __future__ import annotations

import cv2
import numpy as np


def canny_lineart(
    image: np.ndarray | str,
    low: int = 50,
    high: int = 150,
    smooth_level: int = 0,
) -> np.ndarray:
    """Canny 线稿提取 — 适用于激光雕刻 / SVG 路径追踪。

    Args:
        image: BGR numpy 数组 (H,W,3) 或图片文件路径
        low: Canny 低阈值 (0-255, 默认 50)
             越小线条越多，越大越稀疏
        high: Canny 高阈值 (0-255, 默认 150)
              推荐 high = low × 3
        smooth_level: 预平滑强度
            0 = 默认 (Gaussian 3×3) — 保留最多细节，推荐
            1 = 轻量 (Bilateral 5×5 + Gaussian 3×3) — 去除微小纹理
            2 = 中等 (Bilateral 7×7 + Gaussian 5×5) — 抹平密集纹理

    Returns:
        uint8 二值图像 (0=背景, 255=线条)

    典型参数组合:
        通用最佳:   low=50,  high=150, smooth=0   ← 经过 117 图验证
        更多细节:   low=30,  high=100, smooth=0
        更少噪点:   low=80,  high=200, smooth=1
        密集纹理图: low=50,  high=150, smooth=2
    """
    # ── 加载图像 ──
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image}")
    else:
        img = image

    if img.ndim < 2 or img.ndim > 3:
        raise ValueError(f"需要 2D 或 3D 数组，收到 ndim={img.ndim}")

    # ── 转灰度 ──
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # ── CLAHE 局部对比度增强 ──
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ── 多级预平滑 ──
    if smooth_level == 0:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    elif smooth_level == 1:
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    elif smooth_level == 2:
        gray = cv2.bilateralFilter(gray, 7, 75, 75)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        raise ValueError(f"smooth_level 必须是 0/1/2，收到 {smooth_level}")

    # ── Canny 边缘检测 ──
    return cv2.Canny(gray, low, high)


# ── 便捷函数 ──

def canny_lineart_to_file(
    image: np.ndarray | str,
    output_path: str,
    **kwargs,
) -> str:
    """提取线稿并保存为 PNG。返回输出路径。"""
    result = canny_lineart(image, **kwargs)
    cv2.imwrite(output_path, result)
    return output_path


def canny_lineart_to_bgra(
    image: np.ndarray | str, **kwargs
) -> np.ndarray:
    """提取线稿返回 BGRA (透明背景+黑线)，适合叠加显示。"""
    binary = canny_lineart(image, **kwargs)
    h, w = binary.shape
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[binary > 0, 3] = 255
    return bgra


# ── CLI ──

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Canny LineArt 线稿提取")
    p.add_argument("input", help="输入图片路径")
    p.add_argument("output", nargs="?", help="输出路径 (默认: <input>_lineart.png)")
    p.add_argument("--low", type=int, default=50, help="Canny 低阈值 (默认 50)")
    p.add_argument("--high", type=int, default=150, help="Canny 高阈值 (默认 150)")
    p.add_argument("--smooth", type=int, default=0, choices=[0, 1, 2],
                   help="平滑强度: 0=默认 1=轻量 2=中等 (默认 0)")
    args = p.parse_args()

    out = args.output or f"{args.input.rsplit('.', 1)[0]}_lineart.png"

    print(f"Canny LineArt: {args.input}")
    print(f"  low={args.low} high={args.high} smooth={args.smooth}")

    result = canny_lineart(args.input, low=args.low, high=args.high,
                           smooth_level=args.smooth)
    cv2.imwrite(out, result)
    fg_pct = (result > 0).mean() * 100
    print(f"完成 → {out}  (线条密度 {fg_pct:.1f}%)")
