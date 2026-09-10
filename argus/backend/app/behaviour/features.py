"""Geometry and trajectory feature extraction for behaviour detection.

Computes: velocity, tilt angle, footprint ratio, zone membership, jerk,
object density, overlap ratio, orientation angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GeometryFeatures:
    """Computed geometry features for a tracked object at a point in time."""
    velocity_magnitude: float = 0.0
    velocity_angle: float = 0.0  # degrees from horizontal
    acceleration_magnitude: float = 0.0
    jerk_magnitude: float = 0.0
    tilt_angle: float = 0.0  # degrees from vertical
    footprint_ratio: float = 0.0  # width / height of bbox
    aspect_ratio: float = 0.0
    bottom_y: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    area: float = 0.0
    in_zone: bool = True
    zone_id: Optional[str] = None
    orientation_angle: float = 0.0  # degrees from configured correct orientation


def compute_velocity(
    center_curr: tuple[float, float],
    center_prev: tuple[float, float],
    dt: float,
) -> tuple[float, float]:
    """Compute velocity in pixels per second."""
    if dt <= 0:
        return (0.0, 0.0)
    vx = (center_curr[0] - center_prev[0]) / dt
    vy = (center_curr[1] - center_prev[1]) / dt
    return (vx, vy)


def compute_jerk(
    accel_curr: tuple[float, float],
    accel_prev: tuple[float, float],
    dt: float,
) -> tuple[float, float]:
    """Compute jerk (rate of acceleration change) in px/s³."""
    if dt <= 0:
        return (0.0, 0.0)
    jx = (accel_curr[0] - accel_prev[0]) / dt
    jy = (accel_curr[1] - accel_prev[1]) / dt
    return (jx, jy)


def magnitude(v: tuple[float, float]) -> float:
    """Euclidean magnitude of a 2D vector."""
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


def compute_tilt_angle(bbox_width: float, bbox_height: float) -> float:
    """Estimate tilt angle from bbox aspect ratio skew.

    Returns angle in degrees from vertical (0 = perfectly upright).
    """
    if bbox_height <= 0:
        return 0.0
    ratio = bbox_width / bbox_height
    # A perfect vertical box has ratio ~0.3; horizontal ~3.0
    # Map ratio to angle: ratio 1.0 → 45°, ratio 0.3 → 0°, ratio 3.0 → 90°
    angle = math.degrees(math.atan(ratio))
    return angle


def compute_footprint_ratio(top_bbox: tuple[float, float, float, float],
                            bottom_bbox: tuple[float, float, float, float]) -> float:
    """Compute ratio of top footprint area to bottom footprint area.

    Used for improper stacking detection (behaviour 4).
    Returns top_area / bottom_area. >1.0 means top is larger (improper).
    """
    top_area = (top_bbox[2] - top_bbox[0]) * (top_bbox[3] - top_bbox[1])
    bottom_area = (bottom_bbox[2] - bottom_bbox[0]) * (bottom_bbox[3] - bottom_bbox[1])
    if bottom_area <= 0:
        return 0.0
    return top_area / bottom_area


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def compute_object_density(
    centers: list[tuple[float, float]],
    zone_polygon: list[tuple[float, float]],
    radius: float = 100.0,
) -> float:
    """Compute local object density within a zone (objects per area unit)."""
    # Count objects whose center is within the zone
    in_zone_count = sum(1 for c in centers if point_in_polygon(c, zone_polygon))
    # Approximate zone area from bounding box of polygon
    xs = [p[0] for p in zone_polygon]
    ys = [p[1] for p in zone_polygon]
    zone_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if zone_area <= 0:
        return 0.0
    return in_zone_count / (zone_area / 10000)  # normalize


def compute_overlap_ratio(
    bboxes: list[tuple[float, float, float, float]],
) -> float:
    """Compute average pairwise overlap ratio of bounding boxes in a region."""
    if len(bboxes) < 2:
        return 0.0

    total_overlap = 0.0
    pairs = 0

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            x1 = max(bboxes[i][0], bboxes[j][0])
            y1 = max(bboxes[i][1], bboxes[j][1])
            x2 = min(bboxes[i][2], bboxes[j][2])
            y2 = min(bboxes[i][3], bboxes[j][3])

            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area_i = (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1])
                area_j = (bboxes[j][2] - bboxes[j][0]) * (bboxes[j][3] - bboxes[j][1])
                union = area_i + area_j - intersection
                if union > 0:
                    total_overlap += intersection / union
            pairs += 1

    return total_overlap / pairs if pairs > 0 else 0.0


def extract_features(
    bbox: tuple[float, float, float, float],
    prev_bbox: Optional[tuple[float, float, float, float]] = None,
    prev_velocity: Optional[tuple[float, float]] = None,
    prev_accel: Optional[tuple[float, float]] = None,
    dt: float = 1 / 30,
    correct_orientation_angle: float = 0.0,
) -> GeometryFeatures:
    """Extract all geometry features from current and previous frame data."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    area = width * height

    features = GeometryFeatures(
        center_x=center[0],
        center_y=center[1],
        bottom_y=y2,
        area=area,
        footprint_ratio=width / height if height > 0 else 0,
        aspect_ratio=width / height if height > 0 else 0,
        tilt_angle=compute_tilt_angle(width, height),
    )

    # Velocity
    if prev_bbox:
        prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
        vx, vy = compute_velocity(center, prev_center, dt)
        features.velocity_magnitude = magnitude((vx, vy))
        features.velocity_angle = math.degrees(math.atan2(vy, vx))

        # Acceleration
        if prev_velocity:
            ax = (vx - prev_velocity[0]) / dt
            ay = (vy - prev_velocity[1]) / dt
            features.acceleration_magnitude = magnitude((ax, ay))

            # Jerk
            if prev_accel:
                jx, jy = compute_jerk((ax, ay), prev_accel, dt)
                features.jerk_magnitude = magnitude((jx, jy))

    # Orientation angle
    if width > 0 and height > 0:
        features.orientation_angle = abs(math.degrees(math.atan(height / width)) - correct_orientation_angle)

    return features
