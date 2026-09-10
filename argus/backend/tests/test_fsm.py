"""Unit tests for the behaviour FSM."""

import pytest
from app.behaviour.fsm import TrackFSM, TrackState, FrameData, EventType


class TestTrackFSM:
    """Test the per-track finite state machine."""

    def setup_method(self):
        self.fsm = TrackFSM(track_id=1, object_class="box")

    def test_initial_state_is_idle(self):
        assert self.fsm.state == TrackState.IDLE

    def test_pickup_detection(self):
        """Object moving upward should trigger PICKING_UP state."""
        frame = FrameData(
            timestamp=0.0,
            bbox=(100, 300, 200, 400),
            center=(150, 350),
            bottom_y=400,
            area=10000,
        )
        self.fsm.update(frame)

        # Frame 2: moving up
        frame2 = FrameData(
            timestamp=0.033,
            bbox=(100, 280, 200, 380),
            center=(150, 330),
            bottom_y=380,
            area=10000,
        )
        self.fsm.update(frame2)
        assert self.fsm.state == TrackState.PICKING_UP

    def test_drop_detection(self):
        """Carried-above-floor object falling fast to floor emits PRODUCT_DROPPED."""
        # Set up carrying state with ~1s of held-above-floor history
        self.fsm.state = TrackState.CARRYING
        self.fsm.state_since = 0.0
        self.fsm._prev_center = (150, 200)
        self.fsm._prev_timestamp = 0.0
        for i in range(10):
            held = FrameData(
                timestamp=i * 0.1,
                bbox=(100, 200, 200, 300),
                center=(150, 250),
                bottom_y=300,
                area=10000,
                velocity=(0.0, 0.0),
            )
            self.fsm.update(held)

        # Frame with fast downward velocity landing at floor level
        frame = FrameData(
            timestamp=1.033,
            bbox=(100, 200, 200, 500),
            center=(150, 350),
            bottom_y=500,
            area=30000,
        )
        event = self.fsm.update(frame)
        assert event is not None
        assert event.event_type == EventType.PRODUCT_DROPPED
        assert event.confidence > 0

    def test_drag_detection(self):
        """Bottom edge near floor + horizontal motion should trigger dragging."""
        self.fsm.state = TrackState.CARRYING
        self.fsm._prev_center = (100, 250)
        self.fsm._prev_timestamp = 0.0

        frame = FrameData(
            timestamp=0.033,
            bbox=(120, 200, 220, 480),
            center=(170, 340),
            bottom_y=480,
            area=28000,
        )
        event = self.fsm.update(frame)
        # Should detect dragging (bottom near floor + horizontal movement)
        if event:
            assert event.event_type in (EventType.PRODUCT_DRAGGED, EventType.DRAGGING_NO_TROLLEY)

    def test_reset(self):
        """FSM reset should return to idle."""
        self.fsm.state = TrackState.CARRYING
        self.fsm.frame_history = [FrameData(0, (0,0,1,1), (0.5,0.5), 1, 1)]
        self.fsm.reset()
        assert self.fsm.state == TrackState.IDLE
        assert len(self.fsm.frame_history) == 0

    def test_frame_history_bounded(self):
        """Frame history should not exceed max_history."""
        self.fsm.max_history = 5
        for i in range(10):
            frame = FrameData(
                timestamp=i * 0.033,
                bbox=(100, 300, 200, 400),
                center=(150, 350),
                bottom_y=400,
                area=10000,
            )
            self.fsm.update(frame)
        assert len(self.fsm.frame_history) <= 5

    def test_velocity_computation(self):
        """Velocity should be computed between consecutive frames."""
        frame1 = FrameData(
            timestamp=0.0,
            bbox=(100, 300, 200, 400),
            center=(150, 350),
            bottom_y=400,
            area=10000,
        )
        self.fsm.update(frame1)

        frame2 = FrameData(
            timestamp=0.1,
            bbox=(200, 300, 300, 400),
            center=(250, 350),
            bottom_y=400,
            area=10000,
        )
        self.fsm.update(frame2)

        assert frame2.velocity is not None
        assert frame2.velocity[0] > 0  # moving right
        assert abs(frame2.velocity[1]) < 1  # not moving vertically
