"""Camera calibration — homography setup, reference-object height/velocity calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CalibrationData:
    """Camera calibration parameters."""
    # Homography matrix (4-point warp from image to floor plane)
    homography_matrix: Optional[list[list[float]]] = None
    # Reference object properties for velocity calibration
    reference_object_height_m: float = 0.3  # standard carton height in metres
    reference_object_height_px: float = 100.0  # measured in calibration frame
    # Pixels-per-metre scale
    pixels_per_metre: float = 333.0  # default: 100px / 0.3m
    # Camera position
    camera_id: Optional[str] = None
    bay_id: Optional[str] = None


class CameraCalibrator:
    """Handle camera calibration for velocity and spatial measurements."""

    def __init__(self, calibration_dir: str = "./storage/calibration"):
        self.calibration_dir = Path(calibration_dir)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

    def calibrate_from_reference(
        self,
        reference_height_m: float,
        reference_height_px: float,
    ) -> CalibrationData:
        """Calibrate using a known reference object in frame.

        Args:
            reference_height_m: Real-world height of reference object in metres.
            reference_height_px: Measured height of reference object in pixels.

        Returns:
            CalibrationData with computed scale.
        """
        pixels_per_metre = reference_height_px / reference_height_m
        return CalibrationData(
            reference_object_height_m=reference_height_m,
            reference_object_height_px=reference_height_px,
            pixels_per_metre=pixels_per_metre,
        )

    def calibrate_from_homography(
        self,
        image_points: list[tuple[float, float]],
        world_points: list[tuple[float, float]],
    ) -> CalibrationData:
        """Compute homography from 4+ point correspondences.

        Args:
            image_points: Points in image coordinates (at least 4).
            world_points: Corresponding points in world/floor coordinates.

        Returns:
            CalibrationData with homography matrix.
        """
        import numpy as np

        if len(image_points) < 4 or len(world_points) < 4:
            raise ValueError("Need at least 4 point correspondences")

        src = np.array(image_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)

        # Compute homography using DLT
        A = []
        for i in range(len(src)):
            x, y = src[i]
            u, v = dst[i]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        H = H / H[2, 2]

        return CalibrationData(
            homography_matrix=H.tolist(),
        )

    def save_calibration(self, cal: CalibrationData, camera_id: str):
        """Save calibration data to disk."""
        path = self.calibration_dir / f"{camera_id}.json"
        with open(path, "w") as f:
            json.dump({
                "homography_matrix": cal.homography_matrix,
                "reference_object_height_m": cal.reference_object_height_m,
                "reference_object_height_px": cal.reference_object_height_px,
                "pixels_per_metre": cal.pixels_per_metre,
                "camera_id": cal.camera_id,
                "bay_id": cal.bay_id,
            }, f, indent=2)

    def load_calibration(self, camera_id: str) -> Optional[CalibrationData]:
        """Load calibration data from disk."""
        path = self.calibration_dir / f"{camera_id}.json"
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return CalibrationData(
            homography_matrix=data.get("homography_matrix"),
            reference_object_height_m=data.get("reference_object_height_m", 0.3),
            reference_object_height_px=data.get("reference_object_height_px", 100.0),
            pixels_per_metre=data.get("pixels_per_metre", 333.0),
            camera_id=data.get("camera_id"),
            bay_id=data.get("bay_id"),
        )

    def pixels_to_metres(self, pixels: float, cal: CalibrationData) -> float:
        """Convert pixel displacement to metres using calibration."""
        if cal.pixels_per_metre <= 0:
            return 0.0
        return pixels / cal.pixels_per_metre

    def velocity_in_metres_per_sec(
        self,
        pixel_velocity: float,
        cal: CalibrationData,
    ) -> float:
        """Convert pixel velocity (px/s) to real-world velocity (m/s)."""
        return self.pixels_to_metres(pixel_velocity, cal)
