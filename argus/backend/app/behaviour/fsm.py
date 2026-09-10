"""Per-track finite state machine for warehouse handling behaviour.

States: idle → carry → place/drop/throw
Tracks object state transitions to detect candidate events.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Optional


class TrackState(enum.Enum):
    IDLE = "idle"
    PICKING_UP = "picking_up"
    CARRYING = "carrying"
    PLACING = "placing"
    DROPPING = "dropping"
    THROWING = "throwing"
    DRAGGING = "dragging"
    STEPPING = "stepping"
    STACKING = "stacking"


class EventType(enum.Enum):
    """Candidate event types the FSM can emit."""
    PRODUCT_DROPPED = "1_product_dropped"
    PRODUCT_DRAGGED = "2_product_dragged"
    PRODUCT_THROWN = "3_product_thrown"
    IMPROPER_STACKING = "4_improper_stacking"
    UNSTABLE_STACKING = "5_unstable_stacking"
    STEPPING_ON = "6_stepping_on_packages"
    OUTSIDE_ZONE = "7_outside_designated_zone"
    DRAGGING_NO_TROLLEY = "8_dragging_no_trolley"
    PRODUCT_OVERHANG = "9_product_overhang"
    ROUGH_HANDLING = "10_rough_handling"
    WRONG_ORIENTATION = "11_wrong_orientation"
    CLUTTERED_STAGING = "12_cluttered_staging"


@dataclass
class FrameData:
    """Per-frame data for a single tracked object."""
    timestamp: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    center: tuple[float, float]
    bottom_y: float
    area: float
    keypoints: Optional[dict] = None  # pose keypoints if person
    velocity: Optional[tuple[float, float]] = None
    acceleration: Optional[tuple[float, float]] = None
    zone_id: Optional[str] = None
    # Bottom-y of every person in the same frame (feet positions). Used for
    # perspective-invariant floor estimation: in a tilted camera view the
    # floor is a region, not a horizontal line, so "on the floor" means
    # "near some person's feet at this depth".
    person_bottom_ys: Optional[tuple[float, ...]] = None


@dataclass
class CandidateEvent:
    """An event emitted by the FSM for downstream processing."""
    event_type: EventType
    timestamp: float
    features: dict = field(default_factory=dict)
    confidence: float = 0.0
    track_id: int = 0


@dataclass
class TrackFSM:
    """Finite state machine for a single tracked object."""
    track_id: int
    object_class: str
    state: TrackState = TrackState.IDLE
    frame_history: list[FrameData] = field(default_factory=list)
    state_since: float = 0.0
    max_history: int = 120  # ~4s at 30fps

    # Running statistics
    _prev_center: Optional[tuple[float, float]] = None
    _prev_velocity: Optional[tuple[float, float]] = None
    _prev_timestamp: Optional[float] = None

    def update(self, frame: FrameData) -> Optional[CandidateEvent]:
        """Process a new frame and potentially emit a candidate event."""
        self.frame_history.append(frame)
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)

        dt = frame.timestamp - self._prev_timestamp if self._prev_timestamp is not None else 0.0
        self._prev_timestamp = frame.timestamp

        # Compute instantaneous velocity
        if self._prev_center and dt > 0:
            vx = (frame.center[0] - self._prev_center[0]) / dt
            vy = (frame.center[1] - self._prev_center[1]) / dt
            frame.velocity = (vx, vy)

        # Compute acceleration
        if frame.velocity and self._prev_velocity and dt > 0:
            ax = (frame.velocity[0] - self._prev_velocity[0]) / dt
            ay = (frame.velocity[1] - self._prev_velocity[1]) / dt
            frame.acceleration = (ax, ay)

        # State transition
        event = self._transition(frame)

        if frame.velocity:
            self._prev_velocity = frame.velocity
        self._prev_center = frame.center

        return event

    def _transition(self, frame: FrameData) -> Optional[CandidateEvent]:
        """Core state machine transition logic."""
        event = None

        if self.state in (TrackState.IDLE, TrackState.PICKING_UP):
            # Check for sudden ballistic throw or real drop
            recent = self.frame_history[-10:]
            is_unthrowable = (self.object_class in {"cupboard", "cabinet", "appliance", "furniture", "wardrobe", "pallet", "trolley"} or (frame.area and frame.area >= 50000.0))
            if len(recent) >= 3 and self._check_parabolic_trajectory(recent):
                if frame.velocity and not is_unthrowable:
                    vx, vy = frame.velocity
                    # Real drop: predominantly downward fall with significant vertical displacement
                    dy = recent[-1].center[1] - recent[0].center[1]
                    floor_ys = self._floor_ys()
                    min_floor = min(floor_ys) if floor_ys else 450.0
                    near_floor = any(abs(frame.bottom_y - fy) <= self.FLOOR_TOLERANCE for fy in floor_ys) or frame.bottom_y >= min_floor - 40.0
                    if vy >= 150.0 and vy > 1.3 * abs(vx) and dy >= 35.0 and near_floor:
                        self.state = TrackState.IDLE
                        self.state_since = frame.timestamp
                        return CandidateEvent(
                            event_type=EventType.PRODUCT_DROPPED,
                            timestamp=frame.timestamp,
                            features={"downward_velocity": vy, "horizontal_velocity": vx, "vertical_drop": dy, "frame_bottom_y": frame.bottom_y},
                            confidence=min(max(vy / 300.0, 0.5), 1.0),
                            track_id=self.track_id,
                        )
                import numpy as np
                curr_speed = float(np.hypot(frame.velocity[0], frame.velocity[1])) if frame.velocity else 0.0
                # Real throw: must be airborne with high launch speed and horizontal motion
                floor_ys = self._floor_ys()
                min_floor = min(floor_ys) if floor_ys else 450.0
                is_grounded = self._check_dragging(frame) or (frame.bottom_y >= min_floor - 20.0 and abs(frame.velocity[1] if frame.velocity else 0) < 35.0)
                is_unthrowable = (self.object_class in {"cupboard", "cabinet", "appliance", "furniture", "wardrobe", "pallet", "trolley"} or (frame.area and frame.area >= 50000.0))
                if curr_speed >= 135.0 and not is_grounded and not is_unthrowable and frame.velocity and abs(frame.velocity[0]) >= 55.0:
                    self.state = TrackState.THROWING
                    self.state_since = frame.timestamp
                    return CandidateEvent(
                        event_type=EventType.PRODUCT_THROWN,
                        timestamp=frame.timestamp,
                        features={"throw_velocity": frame.velocity or (0.0, 0.0)},
                        confidence=0.75,
                        track_id=self.track_id,
                    )

        if self.state == TrackState.IDLE:
            # Detect pickup: object moving upward or being carried
            if frame.velocity and frame.velocity[1] < -10:  # moving up
                self.state = TrackState.PICKING_UP
                self.state_since = frame.timestamp

            # Detect drag-from-idle: enter DRAGGING state and require sustained motion
            elif frame.velocity and self._check_dragging(frame):
                self.state = TrackState.DRAGGING
                self.state_since = frame.timestamp
                self._drag_accum = 0.0
                self._drag_emitted = False
                self._drag_start_x = frame.center[0]

        elif self.state == TrackState.PICKING_UP:
            if frame.velocity and abs(frame.velocity[1]) < 5:
                self.state = TrackState.CARRYING
                self.state_since = frame.timestamp

        elif self.state == TrackState.CARRYING:
            event = self._check_carry_events(frame)

        elif self.state == TrackState.DRAGGING:
            event = self._check_drag_events(frame)

        elif self.state == TrackState.THROWING:
            # Already emitted, transition to idle after ballistic phase
            if frame.velocity and frame.velocity[1] > 0 and any(
                abs(frame.bottom_y - fy) <= self.FLOOR_TOLERANCE for fy in self._floor_ys()
            ):
                self.state = TrackState.IDLE
                self.state_since = frame.timestamp

        return event

    def _check_carry_events(self, frame: FrameData) -> Optional[CandidateEvent]:
        """Check for events during carry phase."""
        if not frame.velocity:
            return None

        vx, vy = frame.velocity

        is_unthrowable = (self.object_class in {"cupboard", "cabinet", "appliance", "furniture", "wardrobe", "pallet", "trolley"} or (frame.area and frame.area >= 50000.0))
        # Product dropped: predominantly downward motion reaching floor level
        floor_ys = self._floor_ys()
        min_floor = min(floor_ys) if floor_ys else 450.0
        was_airborne = sum(
            1 for f in self.frame_history[-10:-1]
            if any(f.bottom_y < fy - 30 for fy in floor_ys)
        ) >= 2
        near_floor = any(abs(frame.bottom_y - fy) <= self.FLOOR_TOLERANCE for fy in floor_ys) or frame.bottom_y >= min_floor - 40.0
        recent = self.frame_history[-10:]
        dy = frame.center[1] - recent[0].center[1] if recent else 0.0
        # If moving primarily downward towards the floor with sustained fall
        if not is_unthrowable and vy >= 150.0 and was_airborne and near_floor and vy > 1.3 * abs(vx) and dy >= 35.0:
            self.state = TrackState.IDLE
            self.state_since = frame.timestamp
            return CandidateEvent(
                event_type=EventType.PRODUCT_DROPPED,
                timestamp=frame.timestamp,
                features={"downward_velocity": vy, "horizontal_velocity": vx, "vertical_drop": dy, "frame_bottom_y": frame.bottom_y},
                confidence=min(max(vy / 300.0, 0.5), 1.0),
                track_id=self.track_id,
            )

        # Product thrown: ballistic parabolic fit or sudden airborne release with forward/outward velocity
        recent = self.frame_history[-12:]
        if len(recent) >= 3 and self._check_parabolic_trajectory(recent):
            dy_rec = recent[-1].center[1] - recent[0].center[1]
            if not is_unthrowable and vy >= 150.0 and vy > 1.3 * abs(vx) and dy_rec >= 35.0 and near_floor:
                self.state = TrackState.IDLE
                self.state_since = frame.timestamp
                return CandidateEvent(
                    event_type=EventType.PRODUCT_DROPPED,
                    timestamp=frame.timestamp,
                    features={"downward_velocity": vy, "horizontal_velocity": vx, "vertical_drop": dy_rec, "frame_bottom_y": frame.bottom_y},
                    confidence=0.8,
                    track_id=self.track_id,
                )
            import numpy as np
            curr_speed = float(np.hypot(vx, vy))
            is_grounded = self._check_dragging(frame) or (frame.bottom_y >= min_floor - 20.0 and abs(vy) < 35.0)
            if curr_speed >= 135.0 and not is_grounded and not is_unthrowable and abs(vx) >= 55.0:
                self.state = TrackState.THROWING
                self.state_since = frame.timestamp
                return CandidateEvent(
                    event_type=EventType.PRODUCT_THROWN,
                    timestamp=frame.timestamp,
                    features={"throw_velocity": (vx, vy)},
                    confidence=0.75,
                    track_id=self.track_id,
                )

        # Product dragged: bottom edge near floor while moving horizontally
        if self._check_dragging(frame):
            self.state = TrackState.DRAGGING
            self.state_since = frame.timestamp
            self._drag_accum = 0.0
            self._drag_emitted = False
            self._drag_start_x = frame.center[0]
            return None

        return None

    def _check_drag_events(self, frame: FrameData) -> Optional[CandidateEvent]:
        """Check for events during drag phase."""
        if not frame.velocity:
            return None

        # dt via frame history (update() already advanced _prev_timestamp)
        if len(self.frame_history) >= 2:
            dt = frame.timestamp - self.frame_history[-2].timestamp
        else:
            dt = 0.0
        if self._check_dragging(frame):
            self._drag_misses = 0
            self._drag_accum = getattr(self, "_drag_accum", 0.0) + max(dt, 0.0)
            net_disp = abs(frame.center[0] - getattr(self, "_drag_start_x", frame.center[0]))
            # Emit PRODUCT_DRAGGED after sustained floor motion (>= 0.4s and >= 25px displacement)
            if self._drag_accum >= 0.4 and net_disp >= 25.0 and not getattr(self, "_drag_emitted", False):
                self._drag_emitted = True
                return CandidateEvent(
                    event_type=EventType.PRODUCT_DRAGGED,
                    timestamp=frame.timestamp,
                    features={
                        "horizontal_speed": abs(frame.velocity[0]),
                        "drag_duration": round(self._drag_accum, 2),
                        "net_displacement": round(net_disp, 1),
                        "bottom_y": round(frame.bottom_y, 1),
                        "bbox": frame.bbox,
                    },
                    confidence=0.7,
                    track_id=self.track_id,
                )
        else:
            self._drag_misses = getattr(self, "_drag_misses", 0) + 1
            if self._drag_misses >= self.DRAG_EXIT_FRAMES:
                self.state = TrackState.IDLE
                self.state_since = frame.timestamp

        return None

    def _check_parabolic_trajectory(self, frames: list[FrameData]) -> bool:
        """Check if recent positions fit a ballistic trajectory or rapid throw."""
        if len(frames) < 3:
            return False

        import numpy as np

        cx = np.array([f.center[0] for f in frames])
        cy = np.array([f.center[1] for f in frames])
        t = np.array([(f.timestamp - frames[0].timestamp) for f in frames])

        # Instantaneous speeds
        if len(t) > 1:
            dt = np.diff(t)
            speeds = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2) / np.maximum(dt, 1e-6)
            max_speed = float(speeds.max()) if len(speeds) > 0 else 0.0
        else:
            max_speed = 0.0

        net_disp = float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))

        floor_ys = self._floor_ys()
        min_floor = min(floor_ys) if floor_ys else 450.0
        is_on_floor = (
            any(abs(frames[-1].bottom_y - fy) <= 65 for fy in floor_ys)
            or frames[-1].bottom_y >= min_floor - 20.0
            or self._check_dragging(frames[-1])
        )

        # Normal floor handling (dragging or rolling on floor) is NOT an airborne ballistic trajectory
        if is_on_floor and max_speed < 300.0:
            return False

        # Condition 1: Extremely high speed throw / fling (>= 300 px/s)
        if max_speed >= 300.0 and net_disp >= 45.0:
            return True

        # Condition 2: Airborne rapid movement (above floor level)
        if not is_on_floor and max_speed >= 120.0 and net_disp >= 45.0:
            return True

        # Condition 3: Quadratic fit on vertical trajectory (ballistic arc under gravity)
        if len(frames) >= 5:
            try:
                coeffs = np.polyfit(t, cy, 2)
                y_pred = np.polyval(coeffs, t)
                ss_res = np.sum((cy - y_pred) ** 2)
                ss_tot = np.sum((cy - np.mean(cy)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                if r_squared > 0.70 and max_speed >= 60.0 and coeffs[0] > 0:
                    return True
            except Exception:
                pass

        return False

    FLOOR_TOLERANCE = 140.0  # px below/above a person's feet at same depth
    DRAG_EXIT_FRAMES = 8     # consecutive non-drag frames before leaving DRAGGING

    def _floor_ys(self) -> tuple[float, ...]:
        """Floor reference lines for the current camera view."""
        if self.frame_history and self.frame_history[-1].person_bottom_ys:
            return self.frame_history[-1].person_bottom_ys
        return (450.0, 720.0)

    def _check_dragging(self, frame: FrameData) -> bool:
        """Check if object appears to be dragged along the floor."""
        floor_ys = self._floor_ys()
        near_floor = any(
            abs(frame.bottom_y - fy) <= self.FLOOR_TOLERANCE
            for fy in floor_ys
        ) or frame.bottom_y >= 350.0
        if not near_floor:
            return False

        # Has horizontal velocity consistent with human pulling/dragging (18 to 450 px/s)
        if frame.velocity:
            speed = abs(frame.velocity[0])
            vy = abs(frame.velocity[1])
            if 18.0 <= speed <= 450.0 and vy <= 0.7 * speed:
                return True

        return False

    def reset(self):
        """Reset FSM to idle state."""
        self.state = TrackState.IDLE
        self.state_since = 0.0
        self.frame_history.clear()
        self._prev_center = None
        self._prev_velocity = None
        self._prev_timestamp = None
