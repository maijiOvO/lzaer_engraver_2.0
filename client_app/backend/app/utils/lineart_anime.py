# lineart_anime.py — Core line-art extraction engine
# The user will paste the actual implementation here.
# Expected function signature:
#   def lineart_anime(image: Image.Image, detect_resolution: int = 768,
#                     line_strength: int = 55, thin: bool = True) -> Image.Image:
#       ...

"""
Standalone LineArt Anime — 从图像生成黑白线稿，可独立用于任何项目。

依赖 (pip install):
    opencv-python numpy Pillow controlnet_aux

controlnet_aux 首次运行自动从 HuggingFace 下载 ControlNet LineArt Anime 模型。

用法:
    from lineart_anime_standalone import lineart_anime

    # 从文件
    result = lineart_anime("input.jpg")
    cv2.imwrite("output.png", result)

    # 从 numpy 数组 (BGR)
    img = cv2.imread("input.jpg")
    result = lineart_anime(img)

    # 调整参数
    result = lineart_anime(img, detect_resolution=1024, line_strength=60)

    # CLI
    python lineart_anime_standalone.py input.jpg output.png
    python lineart_anime_standalone.py input.jpg --detect-resolution 1024 --line-strength 60

核心算法:
    1. ControlNet LineArt Anime 模型提取线稿 (白底黑线)
    2. 二值化输出 → 可选细化到 1px
    3. 可选短连通域过滤

模型下载位置: ~/.cache/huggingface/
"""

from __future__ import annotations

import cv2
import numpy as np

# ── 模型缓存 ──

_LINEART_PROCESSOR = None


def _load_processor():
    """延迟加载 ControlNet LineArt Anime 处理器 (全局单例)。"""
    global _LINEART_PROCESSOR
    if _LINEART_PROCESSOR is None:
        import os
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

        from controlnet_aux.processor import Processor
        _LINEART_PROCESSOR = Processor("lineart_anime")
        # 设置默认参数 (API 通过 params dict 传递)
        _LINEART_PROCESSOR.params.setdefault("detect_resolution", 768)
        _LINEART_PROCESSOR.params.setdefault("image_resolution", 768)
    return _LINEART_PROCESSOR


# ── 核心函数 ──

def lineart_anime(
    image: np.ndarray | str,
    detect_resolution: int = 768,
    line_strength: int = 55,
    *,
    thin: bool = True,
    min_component_area: int = 4,
) -> np.ndarray:
    """提取 LineArt Anime 黑白线稿。

    Args:
        image: BGR numpy 数组 (H,W,3) 或图片文件路径
        detect_resolution: 检测分辨率 (默认 768，越大细节越多)
        line_strength: 线条强度 0-255 (默认 55，越大越稀疏)
        thin: 是否细化到 1px (默认 True，适合激光雕刻/SVG 路径)
        min_component_area: 最小连通域面积 (小于此值的碎片剔除，默认 4)

    Returns:
        uint8 二值图像 (0=背景, 255=线条)

    典型参数:
        - 简洁线稿: detect_resolution=512, line_strength=80, thin=True
        - 丰富细节: detect_resolution=1024, line_strength=30, thin=True
        - 厚轮廓:   detect_resolution=768,  line_strength=55, thin=False
    """
    # 加载图像
    if isinstance(image, str):
        img_bgr = cv2.imread(image)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {image}")
    else:
        img_bgr = image

    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"需要 BGR 3 通道图像，收到 shape={img_bgr.shape}")

    # 1. ControlNet LineArt 检测
    from PIL import Image
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    proc = _load_processor()
    # 更新参数 dict (controlnet_aux Processor 通过 params 传递参数)
    proc.params["detect_resolution"] = detect_resolution
    proc.params["image_resolution"] = detect_resolution
    result = proc(pil_img)
    gray = np.array(result.convert("L"))

    # 2. 二值化 (白底黑线 → 黑底白线)
    threshold = max(10, min(255, 255 - line_strength))
    binary = ((gray < threshold).astype(np.uint8)) * 255

    # 3. 确保尺寸匹配原图 (模型可能自动缩放)
    h, w = img_bgr.shape[:2]
    if binary.shape[:2] != (h, w):
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)

    # 4. 后处理: 细化 + 短连通域过滤
    if thin:
        binary = _thin_to_1px(binary)
        # 细化后可能产生碎片，再滤一次
        binary = _filter_short_components(binary, min_area=max(2, min_component_area // 2))
    elif min_component_area > 1:
        binary = _filter_short_components(binary, min_area=min_component_area)

    return binary


# ── 后处理工具 ──

def _thin_to_1px(binary: np.ndarray) -> np.ndarray:
    """细化到 1px 骨架 (Zhang-Suen)。"""
    if binary.size == 0 or binary.max() == 0:
        return binary.copy()
    try:
        return cv2.ximgproc.thinning(
            binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
    except (ImportError, AttributeError):
        # 回退: 自实现 Zhang-Suen
        return _zhang_suen(binary)


def _zhang_suen(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen 细化 (纯 Python 回退)。"""
    skel = (binary > 0).astype(np.uint8)
    h, w = skel.shape
    while True:
        changed = False
        # Sub-iteration 1
        to_remove = np.zeros((h, w), dtype=bool)
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if skel[r, c] == 0:
                    continue
                p2, p3, p4 = skel[r - 1, c], skel[r - 1, c + 1], skel[r, c + 1]
                p5, p6, p7 = skel[r + 1, c + 1], skel[r + 1, c], skel[r + 1, c - 1]
                p8, p9 = skel[r, c - 1], skel[r - 1, c - 1]
                p = [p2, p3, p4, p5, p6, p7, p8, p9]
                transitions = sum(1 for a, b in zip(p, p[1:] + [p[0]]) if a == 0 and b == 1)
                neighbors = sum(p)
                if (2 <= neighbors <= 6 and transitions == 1
                        and p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0):
                    to_remove[r, c] = True
                    changed = True
        skel[to_remove] = 0
        # Sub-iteration 2
        to_remove = np.zeros((h, w), dtype=bool)
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if skel[r, c] == 0:
                    continue
                p2, p3, p4 = skel[r - 1, c], skel[r - 1, c + 1], skel[r, c + 1]
                p5, p6, p7 = skel[r + 1, c + 1], skel[r + 1, c], skel[r + 1, c - 1]
                p8, p9 = skel[r, c - 1], skel[r - 1, c - 1]
                p = [p2, p3, p4, p5, p6, p7, p8, p9]
                transitions = sum(1 for a, b in zip(p, p[1:] + [p[0]]) if a == 0 and b == 1)
                neighbors = sum(p)
                if (2 <= neighbors <= 6 and transitions == 1
                        and p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0):
                    to_remove[r, c] = True
                    changed = True
        skel[to_remove] = 0
        if not changed:
            break
    return (skel * 255).astype(np.uint8)


def _filter_short_components(
    binary: np.ndarray, min_area: int, connectivity: int = 8
) -> np.ndarray:
    """删除像素数 < min_area 的连通域。"""
    if min_area <= 1 or binary.max() == 0:
        return binary.copy()
    fg = (binary > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg, connectivity=connectivity
    )
    if n <= 1:
        return binary.copy()
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[0] = False
    areas = stats[:, cv2.CC_STAT_AREA]
    keep_mask[1:] = areas[1:] >= min_area
    result = np.zeros_like(binary)
    result[keep_mask[labels]] = 255
    return result


# ── 便捷函数 ──

def lineart_anime_to_file(
    image: np.ndarray | str,
    output_path: str,
    **kwargs,
) -> str:
    """提取线稿并保存为 PNG 文件。返回输出路径。"""
    result = lineart_anime(image, **kwargs)
    cv2.imwrite(output_path, result)
    return output_path


def lineart_anime_to_bgra(
    image: np.ndarray | str, **kwargs
) -> np.ndarray:
    """提取线稿并返回 BGRA (透明背景 + 黑线)，适合叠加显示。"""
    binary = lineart_anime(image, **kwargs)
    h, w = binary.shape
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    fg = binary > 0
    bgra[fg, 3] = 255  # 线条不透明
    return bgra


# ── CLI ──

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LineArt Anime 线稿提取")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("output", nargs="?", help="输出路径 (默认: <input>_lineart.png)")
    parser.add_argument("--detect-resolution", type=int, default=768,
                        help="检测分辨率 (默认 768)")
    parser.add_argument("--line-strength", type=int, default=55,
                        help="线条强度 0-255 (默认 55)")
    parser.add_argument("--no-thin", action="store_true",
                        help="不细化到 1px")
    parser.add_argument("--min-area", type=int, default=4,
                        help="最小连通域面积 (默认 4)")

    args = parser.parse_args()

    output = args.output or f"{args.input.rsplit('.', 1)[0]}_lineart.png"

    print(f"提取线稿: {args.input}")
    print(f"  分辨率={args.detect_resolution}, 强度={args.line_strength}")
    print(f"  细化={'否' if args.no_thin else '是'}, 最小面积={args.min_area}")

    result = lineart_anime(
        args.input,
        detect_resolution=args.detect_resolution,
        line_strength=args.line_strength,
        thin=not args.no_thin,
        min_component_area=args.min_area,
    )

    cv2.imwrite(output, result)
    fg_pct = (result > 0).mean() * 100
    print(f"完成 → {output}  ({fg_pct:.1f}% 线条密度)")
