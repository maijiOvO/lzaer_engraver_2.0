"""Binary image denoising via connected-component area filtering.

Pure function: receives a binary numpy array (0=bg, 255=fg) and returns
a filtered copy with small isolated components removed. No side effects,
no file I/O — the service layer handles all persistence.

Algorithm:
  1. cv2.connectedComponentsWithStats → label all connected components
  2. For each label, check its area (CC_STAT_AREA)
  3. Zero out components with area < min_component_area
  4. Return cleaned binary image

Per API_CONTRACT.md § Step 5.
"""

import cv2
import numpy as np


def denoise_binary(
    binary_img: np.ndarray,
    min_component_area: int = 4,
) -> np.ndarray:
    """Filter out small connected components from a binary image.

    Args:
        binary_img: uint8 binary image with values 0 (background) and 255 (foreground).
        min_component_area: components with fewer pixels than this are removed.
            Default 4 per API_CONTRACT.md § Step 5.

    Returns:
        uint8 binary image (same shape, same dtype) with small components erased.
    """
    # ── Input validation ─────────────────────────────────────────
    if binary_img is None or binary_img.size == 0:
        raise ValueError("Input image is None or empty")
    if len(binary_img.shape) != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {binary_img.shape}")

    # ── Connected components analysis ────────────────────────────
    # 8-connectivity (default) — treats diagonal neighbors as connected
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )

    # stats[i] = (left, top, width, height, area) for label i
    # label 0 is background — always preserve it
    areas = stats[:, cv2.CC_STAT_AREA]

    # ── Build mask: labels to KEEP ───────────────────────────────
    # Vectorized: boolean array where True = keep this label
    keep = areas >= min_component_area
    keep[0] = True  # background always kept

    # Create a lookup array: keep_mask[label] = 1 if keep, 0 if discard
    keep_mask = np.zeros(num_labels, dtype=np.uint8)
    keep_mask[keep] = 1

    # Apply: only set foreground pixels where their label is kept
    result = binary_img.copy()
    result[keep_mask[labels] == 0] = 0

    components_removed = (~keep).sum()

    return result
