"""
Rust severity analysis module for RustWatch.

Converts YOLO bounding-box detections into coverage metrics and a
three-level severity classification (LOCALIZED / DISTRIBUTED / EXTENSIVE).

Each detection dict must have a 'bbox' key: [x1, y1, x2, y2].
image_size is always (height, width) to match numpy array convention.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def create_detection_mask(
    detections: list[dict],
    image_size: tuple[int, int],
) -> np.ndarray:
    """Convert bounding boxes to a binary mask.

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
) -> dict:
    """Calculate all severity metrics from detections.

    Args:
        detections: List of dicts with 'bbox' key.
        image_size: (height, width) of the source image.

    Returns:
        Dict with keys: num_patches, coverage_ratio,
        avg_patch_size, max_patch_size.
    """
    mask = create_detection_mask(detections, image_size)
    labeled, num_patches = ndimage.label(mask)
    coverage_ratio = float(mask.sum()) / mask.size

    if num_patches > 0:
        patch_sizes = [int((labeled == i).sum()) for i in range(1, num_patches + 1)]
        avg_patch_size = float(np.mean(patch_sizes))
        max_patch_size = int(max(patch_sizes))
    else:
        avg_patch_size = 0.0
        max_patch_size = 0

    return {
        "num_patches": int(num_patches),
        "coverage_ratio": coverage_ratio,
        "avg_patch_size": avg_patch_size,
        "max_patch_size": max_patch_size,
    }


def classify_severity(metrics: dict) -> str:
    """Map metrics to a severity label.

    Args:
        metrics: Dict from calculate_metrics.

    Returns:
        'LOCALIZED', 'DISTRIBUTED', or 'EXTENSIVE'.
    """
    if metrics["num_patches"] <= 2 and metrics["coverage_ratio"] < 0.10:
        return "LOCALIZED"
    elif metrics["num_patches"] >= 3 and metrics["coverage_ratio"] < 0.25:
        return "DISTRIBUTED"
    else:
        return "EXTENSIVE"


def analyze_rust(
    detections: list[dict],
    image_size: tuple[int, int],
) -> dict:
    """Main analysis entry point.

    Args:
        detections: List of dicts with 'bbox' key ([x1, y1, x2, y2]).
        image_size: (height, width) of the source image.

    Returns:
        Dict with num_patches, coverage_ratio, avg_patch_size,
        max_patch_size, and severity string.
        Returns severity='NONE' when there are no detections.
    """
    if not detections:
        return {
            "num_patches": 0,
            "coverage_ratio": 0.0,
            "avg_patch_size": 0.0,
            "max_patch_size": 0,
            "severity": "NONE",
            "suspicious": False,
        }
    metrics = calculate_metrics(detections, image_size)
    severity = classify_severity(metrics)

    # Flag if any single box covers more than 80% of the frame — likely a false positive
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
    # Simulate two overlapping rust patches on a 480×640 image
    sample_detections = [
        {"bbox": [50, 60, 200, 180]},   # patch 1
        {"bbox": [300, 100, 420, 250]}, # patch 2
    ]
    image_size = (480, 640)  # (height, width)

    result = analyze_rust(sample_detections, image_size)

    print("=== Rust Severity Analysis ===")
    print(f"  Severity      : {result['severity']}")
    print(f"  Patches       : {result['num_patches']}")
    print(f"  Coverage      : {result['coverage_ratio']*100:.1f}%")
    print(f"  Avg patch size: {result['avg_patch_size']:.0f} px")
    print(f"  Max patch size: {result['max_patch_size']} px")
