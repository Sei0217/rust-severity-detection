"""
Rust severity analysis module for RustWatch.

Coverage is measured by counting pixels that actually *look like* rust
(red/orange/brown in HSV) within each detected bounding box, rather than
using raw box area. This prevents large boxes around small rust spots from
inflating the severity score.

Each detection dict must have a 'bbox' key: [x1, y1, x2, y2].
image_size is always (height, width) to match numpy array convention.
Pass image_bgr (the captured frame) to analyze_rust for accurate
color-based coverage; omit it to fall back to box-area coverage.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

# HSV ranges that match rust/corrosion color
# Range A: red-orange-brown  (H 0-25)
# Range B: wraparound dark red (H 165-180)
_RUST_HSV_RANGES = [
    (np.array([0,   60,  40]), np.array([25,  255, 255])),
    (np.array([165, 60,  40]), np.array([180, 255, 255])),
]


def _rust_mask_in_boxes(
    image_bgr: np.ndarray,
    detections: list[dict],
) -> np.ndarray:
    """Return a binary mask of rust-colored pixels inside detection boxes."""
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    color_mask = np.zeros((h, w), dtype=np.uint8)
    for lo, hi in _RUST_HSV_RANGES:
        color_mask |= cv2.inRange(hsv, lo, hi)

    box_mask = np.zeros((h, w), dtype=np.uint8)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            box_mask[y1:y2, x1:x2] = 1

    return (color_mask > 0) & (box_mask > 0)


def create_detection_mask(
    detections: list[dict],
    image_size: tuple[int, int],
) -> np.ndarray:
    """Convert bounding boxes to a binary mask (box-area fallback).

    Args:
        detections: List of dicts with a 'bbox' key ([x1, y1, x2, y2]).
        image_size: (height, width) of the source image.

    Returns:
        uint8 numpy array — 0 = no rust, 1 = rust.
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    h, w = image_size
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def count_connected_patches(mask: np.ndarray) -> int:
    """Count distinct rust patches using connected components.

    Args:
        mask: Binary mask from create_detection_mask.

    Returns:
        Number of connected rust regions.
    """
    _labeled, num_components = ndimage.label(mask)
    return int(num_components)


def calculate_metrics(
    detections: list[dict],
    image_size: tuple[int, int],
    image_bgr: np.ndarray | None = None,
) -> dict:
    """Calculate all severity metrics from detections.

    Args:
        detections: List of dicts with 'bbox' key.
        image_size: (height, width) of the source image.
        image_bgr: Optional captured frame for color-based coverage.
                   When provided, coverage is measured by counting
                   rust-colored pixels instead of box area.

    Returns:
        Dict with keys: num_patches, coverage_ratio,
        avg_patch_size, max_patch_size, coverage_method.
    """
    # Patch count always comes from box connectivity
    box_mask = create_detection_mask(detections, image_size)
    labeled, num_patches = ndimage.label(box_mask)

    if num_patches > 0:
        patch_sizes = [int((labeled == i).sum()) for i in range(1, num_patches + 1)]
        avg_patch_size = float(np.mean(patch_sizes))
        max_patch_size = int(max(patch_sizes))
    else:
        avg_patch_size = 0.0
        max_patch_size = 0

    # Coverage: use color-based pixel count when image is available
    if image_bgr is not None:
        rust_pixels = _rust_mask_in_boxes(image_bgr, detections)
        total = image_size[0] * image_size[1]
        coverage_ratio = float(rust_pixels.sum()) / total
        coverage_method = "color"
    else:
        coverage_ratio = float(box_mask.sum()) / box_mask.size
        coverage_method = "box_area"

    return {
        "num_patches": int(num_patches),
        "coverage_ratio": coverage_ratio,
        "avg_patch_size": avg_patch_size,
        "max_patch_size": max_patch_size,
        "coverage_method": coverage_method,
    }


def classify_severity(metrics: dict) -> str:
    """Map metrics to a severity label.

    Thresholds are tuned for color-based coverage (which is smaller and
    more accurate than box-area coverage).

    Args:
        metrics: Dict from calculate_metrics.

    Returns:
        'LOCALIZED', 'DISTRIBUTED', or 'EXTENSIVE'.
    """
    cov = metrics["coverage_ratio"]
    patches = metrics["num_patches"]

    if patches <= 2 and cov < 0.15:
        return "LOCALIZED"
    elif cov < 0.35:
        return "DISTRIBUTED"
    else:
        return "EXTENSIVE"


def analyze_rust(
    detections: list[dict],
    image_size: tuple[int, int],
    image_bgr: np.ndarray | None = None,
) -> dict:
    """Main analysis entry point.

    Args:
        detections: List of dicts with 'bbox' key ([x1, y1, x2, y2]).
        image_size: (height, width) of the source image.
        image_bgr: Optional captured frame for accurate color-based coverage.

    Returns:
        Dict with num_patches, coverage_ratio, avg_patch_size,
        max_patch_size, coverage_method, severity, and suspicious flag.
    """
    if not detections:
        return {
            "num_patches": 0,
            "coverage_ratio": 0.0,
            "avg_patch_size": 0.0,
            "max_patch_size": 0,
            "coverage_method": "color" if image_bgr is not None else "box_area",
            "severity": "NONE",
            "suspicious": False,
        }

    metrics = calculate_metrics(detections, image_size, image_bgr)
    severity = classify_severity(metrics)

    # Flag if any single box covers more than 80% of the frame
    total_pixels = image_size[0] * image_size[1]
    suspicious = any(
        (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) / total_pixels > 0.80
        for d in detections
    )

    return {**metrics, "severity": severity, "suspicious": suspicious}


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate two rust patches on a 480x640 image (no real image available)
    sample_detections = [
        {"bbox": [50, 60, 200, 180]},
        {"bbox": [300, 100, 420, 250]},
    ]
    image_size = (480, 640)

    result = analyze_rust(sample_detections, image_size)

    print("=== Rust Severity Analysis (box-area fallback) ===")
    print(f"  Severity      : {result['severity']}")
    print(f"  Patches       : {result['num_patches']}")
    print(f"  Coverage      : {result['coverage_ratio']*100:.1f}%  [{result['coverage_method']}]")
    print(f"  Avg patch size: {result['avg_patch_size']:.0f} px")
    print(f"  Max patch size: {result['max_patch_size']} px")
