"""
Rust severity aggregation for RustWatch.

The YOLO model classifies each detection as 'low', 'medium', or 'high'. This
module aggregates per-detection labels into a single per-image severity using
max-wins: any HIGH → HIGH, else any MEDIUM → MEDIUM, else any LOW → LOW, else
NONE. Unknown class labels are ignored.

Each detection dict must have:
  - 'bbox':  [x1, y1, x2, y2]
  - 'class': 'low' | 'medium' | 'high' (case-insensitive)

image_size is (height, width). Used only for the 'suspicious' flag that catches
a single box covering more than 80% of the frame (likely false positive).
"""

from __future__ import annotations

_SEVERITY_RANK = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}


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
) -> dict:
    """Main analysis entry point.

    Returns:
        severity:   'NONE' | 'LOW' | 'MEDIUM' | 'HIGH'
        suspicious: True if any single box covers > 80% of the frame
    """
    severity = aggregate_severity(detections)

    if not detections:
        return {"severity": severity, "suspicious": False}

    total_pixels = image_size[0] * image_size[1]
    suspicious = any(
        (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) / total_pixels > 0.80
        for d in detections
    )

    return {"severity": severity, "suspicious": suspicious}


if __name__ == "__main__":
    sample = [
        {"bbox": [50, 60, 200, 180],   "class": "low"},
        {"bbox": [300, 100, 420, 250], "class": "medium"},
    ]
    print(analyze_rust(sample, (480, 640)))
