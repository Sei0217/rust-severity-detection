"""
Rust severity aggregation for RustWatch.

The YOLO model classifies each detection as 'low', 'medium', or 'high'. This
module aggregates per-detection labels into a single per-image severity using
max-wins: any HIGH → HIGH, else any MEDIUM → MEDIUM, else any LOW → LOW, else
NONE. Unknown class labels are ignored.

Coverage and patch count are computed as informational metrics for the
dashboard; they do not drive the severity label.

Each detection dict must have:
  - 'bbox':  [x1, y1, x2, y2]
  - 'class': 'low' | 'medium' | 'high' (case-insensitive)

image_size is (height, width). Pass image_bgr for color-based coverage; omit
it to fall back to box-area coverage.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

# HSV ranges that match rust/corrosion color (informational coverage only)
# Range A: red-orange-brown  (H 0-25)
# Range B: wraparound dark red (H 165-180)
_RUST_HSV_RANGES = [
    (np.array([0,   60,  40]), np.array([25,  255, 255])),
    (np.array([165, 60,  40]), np.array([180, 255, 255])),
]

_SEVERITY_RANK = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}


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
    """Convert bounding boxes to a binary mask (box-area fallback)."""
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
    """Count distinct rust patches using connected components."""
    _labeled, num_components = ndimage.label(mask)
    return int(num_components)


def calculate_metrics(
    detections: list[dict],
    image_size: tuple[int, int],
    image_bgr: np.ndarray | None = None,
) -> dict:
    """Calculate informational coverage/patch metrics from detections."""
    box_mask = create_detection_mask(detections, image_size)
    labeled, num_patches = ndimage.label(box_mask)

    if num_patches > 0:
        patch_sizes = [int((labeled == i).sum()) for i in range(1, num_patches + 1)]
        avg_patch_size = float(np.mean(patch_sizes))
        max_patch_size = int(max(patch_sizes))
    else:
        avg_patch_size = 0.0
        max_patch_size = 0

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


def aggregate_severity(detections: list[dict]) -> str:
    """Aggregate per-detection class labels into a single severity (max-wins)."""
    best, best_rank = 'NONE', 0
    for det in detections:
        label = (det.get('class') or '').upper()
        rank = _SEVERITY_RANK.get(label, 0)
        if rank > best_rank:
            best, best_rank = label, rank
    return best


def analyze_rust(
    detections: list[dict],
    image_size: tuple[int, int],
    image_bgr: np.ndarray | None = None,
) -> dict:
    """Main analysis entry point.

    Returns:
        num_patches, coverage_ratio, avg_patch_size, max_patch_size,
        coverage_method, severity ('NONE'/'LOW'/'MEDIUM'/'HIGH'), suspicious.
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
    severity = aggregate_severity(detections)

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
    sample_detections = [
        {"bbox": [50, 60, 200, 180],   "class": "low"},
        {"bbox": [300, 100, 420, 250], "class": "medium"},
    ]
    image_size = (480, 640)

    result = analyze_rust(sample_detections, image_size)

    print("=== Rust Severity Analysis (box-area fallback) ===")
    print(f"  Severity      : {result['severity']}")
    print(f"  Patches       : {result['num_patches']}")
    print(f"  Coverage      : {result['coverage_ratio']*100:.1f}%  [{result['coverage_method']}]")
    print(f"  Avg patch size: {result['avg_patch_size']:.0f} px")
    print(f"  Max patch size: {result['max_patch_size']} px")
