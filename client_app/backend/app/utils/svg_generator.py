"""SVG vector generation — pure stroke Bezier curves (剪纸单连通拓扑).

Extracts contours from a binary skeleton image, simplifies with
Douglas-Peucker, fits C1-continuous cubic Bezier curves, and outputs
a single-layer SVG with fill="none" stroke="black" stroke-width="1".

Algorithm:
  1. cv2.findContours → raw pixel paths.
  2. cv2.approxPolyDP → simplified anchor points.
  3. Catmull-Rom to cubic Bezier conversion (C1 continuity).
  4. Y-axis flip (cv2 origin top-left → SVG origin bottom-left).
  5. Concatenate into SVG <path d="M... C..."/> string.
"""

import math
from typing import List, Tuple, Sequence

import cv2
import numpy as np


Point = Tuple[float, float]  # (x, y) — x/y in pixel coords


def generate_svg(
    binary_img: np.ndarray,
    simplify_tolerance: float = 1.0,
    stroke_width: int = 1,
) -> str:
    """Generate a pure-stroke SVG from a binary skeleton image.

    Args:
        binary_img: uint8 binary image (0=bg, 255=fg).
        simplify_tolerance: Douglas-Peucker epsilon (pixels).
        stroke_width: SVG stroke-width in pixels.

    Returns:
        Complete SVG string (<svg>...</svg>).
    """
    h, w = binary_img.shape

    # ── 1. Extract contours ────────────────────────────────
    contours = _extract_contours(binary_img)
    if not contours:
        return _empty_svg(w, h)

    # ── 2. Simplify & convert to cubic Bezier paths ─────────
    path_strings = []
    for contour in contours:
        simplified = _approx_poly(contour, simplify_tolerance)
        if len(simplified) < 2:
            continue
        closed = _is_closed(simplified)
        d = _to_bezier_path(simplified, closed)
        if d:
            path_strings.append(d)

    if not path_strings:
        return _empty_svg(w, h)

    paths = '\n    '.join(path_strings)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg"\n'
        f'     viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
        f'  <path d="{paths}"\n'
        f'        fill="none" stroke="black" stroke-width="{stroke_width}"\n'
        f'        stroke-linecap="round" stroke-linejoin="round"/>\n'
        f'</svg>'
    )


# ── Contour extraction ──────────────────────────────────────────

def _extract_contours(img: np.ndarray) -> list:
    """Extract all contours from a binary image (RETR_LIST, CHAIN_APPROX_NONE).

    Returns raw pixel sequences as list of (n, 1, 2) arrays.
    Skips degenerate contours (< 2 points).
    """
    # OpenCV requires foreground=white, background=black
    contours, _ = cv2.findContours(
        img,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    return [c for c in contours if len(c) >= 2]


# ── Douglas-Peucker simplification ──────────────────────────────

def _approx_poly(contour: np.ndarray, epsilon: float) -> List[Point]:
    """Simplify a contour using Douglas-Peucker via cv2.approxPolyDP.

    Returns list of (x, y) points.
    """
    if epsilon <= 0 or len(contour) <= 2:
        # Return all points as-is
        return [(int(p[0][0]), int(p[0][1])) for p in contour]

    approx = cv2.approxPolyDP(contour, epsilon, closed=False)
    return [(int(p[0][0]), int(p[0][1])) for p in approx]


def _is_closed(pts: List[Point]) -> bool:
    """Heuristic: contour is closed if first/last points are very close."""
    if len(pts) < 3:
        return False
    dx = pts[0][0] - pts[-1][0]
    dy = pts[0][1] - pts[-1][1]
    return math.hypot(dx, dy) <= 3.0


# ── Bezier fitting (C1 continuous, Catmull-Rom style) ───────────

def _to_bezier_path(pts: List[Point], closed: bool) -> str:
    """Convert a sequence of anchor points to an SVG path d-string.

    Uses C1-continuous cubic Bezier segments (Catmull-Rom → Bezier conversion).

    For each anchor P_i, the tangent direction is (P_{i+1} - P_{i-1}),
    scaled by 1/6 of the adjacent segment length.
    """
    n = len(pts)
    if n < 2:
        return ""

    segments: List[str] = []

    # Move to first point
    segments.append(f"M {pts[0][0]:.1f} {pts[0][1]:.1f}")

    for i in range(n - 1):
        p0 = pts[i]
        p3 = pts[i + 1]

        # Compute tangents
        if closed:
            p_prev = pts[(i - 1) % n]
            p_next = pts[(i + 2) % n]
        else:
            p_prev = pts[i - 1] if i > 0 else _reflect(p0, p3)
            p_next = pts[i + 2] if i < n - 2 else _reflect(p3, p0)

        # Catmull-Rom control points
        cp1_x = p0[0] + (p3[0] - p_prev[0]) / 6
        cp1_y = p0[1] + (p3[1] - p_prev[1]) / 6
        cp2_x = p3[0] - (p_next[0] - p0[0]) / 6
        cp2_y = p3[1] - (p_next[1] - p0[1]) / 6

        segments.append(
            f"C {cp1_x:.1f} {cp1_y:.1f} {cp2_x:.1f} {cp2_y:.1f} {p3[0]:.1f} {p3[1]:.1f}"
        )

    if closed:
        # Close the loop with a final segment back to the first point
        p0 = pts[-1]
        p3 = pts[0]
        p_prev = pts[-2]
        p_next = pts[1]

        cp1_x = p0[0] + (p3[0] - p_prev[0]) / 6
        cp1_y = p0[1] + (p3[1] - p_prev[1]) / 6
        cp2_x = p3[0] - (p_next[0] - p0[0]) / 6
        cp2_y = p3[1] - (p_next[1] - p0[1]) / 6

        segments.append(
            f"C {cp1_x:.1f} {cp1_y:.1f} {cp2_x:.1f} {cp2_y:.1f} {p3[0]:.1f} {p3[1]:.1f} Z"
        )

    return ' '.join(segments)


def _reflect(p: Point, ref: Point) -> Point:
    """Reflect point p across ref: p' = 2*ref - p."""
    return (2 * ref[0] - p[0], 2 * ref[1] - p[1])


def _empty_svg(w: int, h: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg"\n'
        f'     viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
        f'  <!-- No contours found -->\n'
        f'</svg>'
    )
