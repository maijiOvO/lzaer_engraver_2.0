"""Layer frame utility — renders outer clamping-frame borders on layer masks
and merges per-layer SVGs into a tiled multi-layer SVG.

Frame width = max(image_width, image_height) / 50, equal on all four sides.
Matches the reference SVGs in dev_tools/references/multiple/ where each
paper layer has a fixed-width outer rectangular clamping border.
"""

from __future__ import annotations

import re
from typing import Literal

import cv2
import numpy as np


def render_layer_frame(
    mask: np.ndarray,
    frame_width: int,
) -> np.ndarray:
    """Draw an outer rectangular clamping frame on a binary layer mask.

    The frame is drawn as a 1px white stroke around the expanded bounding box
    of the mask content. The mask is first padded with frame_width on all sides,
    then a rectangle is drawn at the padded bounding box.

    Args:
        mask: (H, W) uint8 binary mask (255 = in layer, 0 = background).
        frame_width: frame border width in pixels (same on all four sides).

    Returns:
        (H+2*fw, W+2*fw) uint8 binary mask with outer frame border.
    """
    h, w = mask.shape[:2]
    fw = frame_width

    # Find the bounding box of the mask content
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        # Empty mask — return a minimal frame rectangle
        result = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(result, (0, 0), (w - 1, h - 1), 255, 1)
        return result

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    # Expand bbox outward by frame_width, clamped to image bounds
    fy_min = max(0, y_min - fw)
    fy_max = min(h - 1, y_max + fw)
    fx_min = max(0, x_min - fw)
    fx_max = min(w - 1, x_max + fw)

    # Draw frame rectangle on a copy of the mask
    result = mask.copy()
    cv2.rectangle(
        result,
        (int(fx_min), int(fy_min)),
        (int(fx_max), int(fy_max)),
        255,  # white, 1px
        1,
    )

    return result


def _svg_bbox(svg_str: str) -> tuple[int, int, int, int] | None:
    """Extract bounding box from a single-layer SVG string.

    Parses path data to find min/max x and y coordinates.

    Returns:
        (x_min, y_min, width, height) in SVG pixel coords, or None if empty.
    """
    # Find all coordinate pairs in path data
    coords = re.findall(r'([\d.]+)\s+([\d.]+)', svg_str)
    if not coords:
        return None

    xs = [float(x) for x, _ in coords]
    ys = [float(y) for _, y in coords]

    x_min = min(xs)
    y_min = min(ys)
    w = max(xs) - x_min
    h = max(ys) - y_min

    return (int(x_min), int(y_min), int(w), int(h))


def _svg_strip_decls(svg_str: str) -> str:
    """Strip XML declaration and outer <svg> tags, returning inner content.

    Extracts everything between the first <svg ...> and the closing </svg>.
    """
    # Remove XML declaration
    svg_str = re.sub(r'<\?xml[^>]*\?>\s*', '', svg_str)

    # Find inner content (between <svg ...> and </svg>)
    m_open = re.search(r'<svg\b[^>]*>', svg_str)
    if not m_open:
        return svg_str
    inner_start = m_open.end()

    # Find the matching </svg> (last occurrence)
    m_close = list(re.finditer(r'</svg>', svg_str))
    if not m_close:
        return svg_str[inner_start:]

    inner_end = m_close[-1].start()
    return svg_str[inner_start:inner_end].strip()


def merge_layer_svgs(
    layer_svgs: list[str],
    layer_bboxes: list[tuple[int, int, int, int]],
    frame_width: int,
    direction: Literal["horizontal", "vertical"],
    output_width: int | None = None,
    output_height: int | None = None,
) -> str:
    """Merge N single-layer SVGs into one tiled multi-layer SVG.

    Each layer's content is wrapped in a <g> group with a transform
    that positions it according to the stacking direction and its
    bounding box. A frame <rect> is added around each layer.

    Args:
        layer_svgs: list of SVG strings, one per layer.
        layer_bboxes: list of (x, y, w, h) per layer.
        frame_width: frame border width (same as used in render_layer_frame).
        direction: stacking direction — 'horizontal' or 'vertical'.
        output_width: optional override for SVG viewBox width.
        output_height: optional override for SVG viewBox height.

    Returns:
        Complete SVG string with all layers merged.
    """
    if not layer_svgs:
        return ''

    n = len(layer_svgs)
    gap = frame_width * 2  # spacing between tiles

    # Compute total viewBox dimensions
    max_layer_w = max(b[2] for b in layer_bboxes) + frame_width * 2
    max_layer_h = max(b[3] for b in layer_bboxes) + frame_width * 2

    if direction == "horizontal":
        total_w = n * max_layer_w + (n - 1) * gap
        total_h = max_layer_h
    else:
        total_w = max_layer_w
        total_h = n * max_layer_h + (n - 1) * gap

    if output_width is not None:
        total_w = output_width
    if output_height is not None:
        total_h = output_height

    # Build SVG
    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'  viewBox="0 0 {total_w} {total_h}"',
        '  style="shape-rendering:geometricPrecision; fill-rule:evenodd; clip-rule:evenodd">',
        '  <style type="text/css"><![CDATA[',
        '    .frame { fill: none; stroke: red; stroke-width: 1; }',
        '  ]]></style>',
    ]

    for i, (svg_str, (bx, by, bw, bh)) in enumerate(zip(layer_svgs, layer_bboxes)):
        if direction == "horizontal":
            dx = i * (max_layer_w + gap)
            dy = 0
        else:
            dx = 0
            dy = i * (max_layer_h + gap)

        # Frame rect
        frame_x = dx + frame_width
        frame_y = dy + frame_width
        frame_w = bw + frame_width * 2
        frame_h = bh + frame_width * 2

        inner_content = _svg_strip_decls(svg_str)

        lines.append(f'  <g id="layer-{i}" transform="translate({dx}, {dy})">')
        lines.append(
            f'    <rect class="frame" x="{frame_width}" y="{frame_width}" '
            f'width="{frame_w}" height="{frame_h}" />'
        )
        lines.append(f'    <g transform="translate({frame_width}, {frame_width})">')
        lines.append(f'      {inner_content}')
        lines.append('    </g>')
        lines.append('  </g>')

    lines.append('</svg>')

    return '\n'.join(lines)


def compute_frame_width(image_shape: tuple[int, int]) -> int:
    """Compute frame border width from image dimensions.

    Rule: max(width, height) / 50, minimum 1px.

    Args:
        image_shape: (height, width) of the original image.

    Returns:
        Frame width in pixels.
    """
    h, w = image_shape[:2]
    return max(1, int(max(w, h) / 50))
