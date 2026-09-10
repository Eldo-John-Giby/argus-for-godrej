"""Face/ID blurring for stored clips — Responsible AI, implemented (plan §5).

Identity protection by default: demo/stored clips get the face band of every
tracked person pixelated before the clip is written to disk. Un-blurred
footage never leaves the raw video file.

Usage:
    from face_blur import blur_regions, face_boxes_from_persons

    boxes = face_boxes_from_persons(person_detections)
    frame = blur_regions(frame, boxes)          # in-place, returns frame
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def face_boxes_from_persons(
    persons: Sequence,
    band: float = 0.22,
    x_inset: float = 0.12,
) -> list[tuple[float, float, float, float]]:
    """Face-region boxes from person detections (upper `band` of each bbox).

    Works with anything exposing .bbox (Detection dataclass) or a raw
    (x1, y1, x2, y2) tuple — no dependency on the perception types.
    """
    boxes = []
    for p in persons:
        bbox = getattr(p, "bbox", p)
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        fx1 = x1 + x_inset * w
        fx2 = x2 - x_inset * w
        fy1 = y1
        fy2 = y1 + band * h
        boxes.append((fx1, fy1, fx2, fy2))
    return boxes


def blur_regions(
    frame: np.ndarray,
    boxes: Sequence[tuple[float, float, float, float]],
    strength: int = 14,
    method: str = "pixelate",
) -> np.ndarray:
    """Blur/pixelate regions of a frame in place. Frame is returned for chaining.

    method="pixelate" is cheap and un-reversible at strength<=16;
    method="blur" uses a heavy Gaussian instead.
    """
    if not boxes:
        return frame
    fh, fw = frame.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        ix1 = max(0, int(x1))
        iy1 = max(0, int(y1))
        ix2 = min(fw, int(x2))
        iy2 = min(fh, int(y2))
        if ix2 - ix1 < 4 or iy2 - iy1 < 4:
            continue
        roi = frame[iy1:iy2, ix1:ix2]
        if method == "pixelate":
            rh = max(2, int(strength * (iy2 - iy1) / max(1, ix2 - ix1)))
            small = cv2_resize(roi, (strength, rh))
            frame[iy1:iy2, ix1:ix2] = cv2_resize(small, (ix2 - ix1, iy2 - iy1), interp="nearest")
        else:
            import cv2

            frame[iy1:iy2, ix1:ix2] = cv2.GaussianBlur(roi, (31, 31), 0)
    return frame


def cv2_resize(img, dsize, interp: str = "linear"):
    import cv2

    interpolation = cv2.INTER_NEAREST if interp == "nearest" else cv2.INTER_LINEAR
    return cv2.resize(img, dsize, interpolation=interpolation)


def maybe_blur_frame(
    frame: np.ndarray,
    person_boxes: Optional[Sequence[tuple[float, float, float, float]]],
    enabled: bool,
) -> np.ndarray:
    """Convenience wrapper: blur only when enabled and boxes exist."""
    if not enabled or not person_boxes:
        return frame
    return blur_regions(frame, list(person_boxes))
