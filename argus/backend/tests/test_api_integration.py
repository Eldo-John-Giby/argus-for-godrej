"""Integration tests for all API endpoints.

Tests the full chain: HTTP request → FastAPI route → business logic → DB → response.
Uses in-memory SQLite via the `client` fixture from conftest.py.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import AsyncClient

from tests.conftest import seed_camera, seed_event, seed_feedback, seed_track


# =========================================================================
# Health API
# =========================================================================


class TestHealthAPI:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client: AsyncClient):
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["service"] == "argus"

    @pytest.mark.asyncio
    async def test_root_endpoint(self, client: AsyncClient):
        res = await client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "ARGUS"
        assert "docs" in data


# =========================================================================
# Events API
# =========================================================================


class TestEventsAPI:
    @pytest.mark.asyncio
    async def test_list_events_empty(self, client: AsyncClient):
        res = await client.get("/api/events/")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_list_events_with_data(self, client: AsyncClient, seeded_data):
        res = await client.get("/api/events/")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 5  # 3 bay_A + 2 bay_B

        # Verify field names match what LiveFeed.tsx expects
        event = data[0]
        for field in ["id", "behaviour_type", "tier", "risk_score", "status", "confidence", "timestamp"]:
            assert field in event, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_list_events_filter_by_tier(self, client: AsyncClient, seeded_data):
        res = await client.get("/api/events/?tier=Critical")
        assert res.status_code == 200
        data = res.json()
        assert all(e["tier"] == "Critical" for e in data)
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_list_events_filter_by_behaviour(self, client: AsyncClient, seeded_data):
        res = await client.get("/api/events/?behaviour_type=3_product_thrown")
        assert res.status_code == 200
        data = res.json()
        assert all(e["behaviour_type"] == "3_product_thrown" for e in data)
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_list_events_filter_by_status(self, client: AsyncClient, seeded_data):
        res = await client.get("/api/events/?status=potential")
        assert res.status_code == 200
        data = res.json()
        assert all(e["status"] == "potential" for e in data)

    @pytest.mark.asyncio
    async def test_list_events_filter_by_bay_id(self, client: AsyncClient, seeded_data):
        """bay_id filter should only return events from that bay."""
        res = await client.get("/api/events/?bay_id=bay_A")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 3  # bay_A has 3 events

        res_b = await client.get("/api/events/?bay_id=bay_B")
        assert res_b.status_code == 200
        data_b = res_b.json()
        assert len(data_b) == 2  # bay_B has 2 events

    @pytest.mark.asyncio
    async def test_list_events_nonexistent_bay(self, client: AsyncClient, seeded_data):
        res = await client.get("/api/events/?bay_id=bay_Z")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_list_events_limit(self, client: AsyncClient, seeded_data):
        res = await client.get("/api/events/?limit=2")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_list_events_ordered_by_time_desc(self, client: AsyncClient, seeded_data):
        res = await client.get("/api/events/")
        data = res.json()
        timestamps = [e["timestamp"] for e in data if e["timestamp"]]
        assert timestamps == sorted(timestamps, reverse=True)

    @pytest.mark.asyncio
    async def test_get_event_detail(self, client: AsyncClient, seeded_data):
        event_id = seeded_data["events"][0].id
        res = await client.get(f"/api/events/{event_id}")
        assert res.status_code == 200
        data = res.json()

        # Verify explain_event response fields (needed by IncidentReplay.tsx)
        for field in [
            "event_id", "behaviour_type", "risk_score", "tier",
            "confidence", "status", "features", "vlm_explanation",
            "risk_breakdown", "explanation",
        ]:
            assert field in data, f"Missing field: {field}"

        assert data["event_id"] == event_id
        assert data["tier"] == "Critical"
        assert data["risk_breakdown"]["behaviour_severity"] == 90

    @pytest.mark.asyncio
    async def test_get_event_not_found(self, client: AsyncClient):
        res = await client.get("/api/events/99999")
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_top_behaviours(self, client: AsyncClient, seeded_data):
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        start = (now - dt.timedelta(hours=1)).isoformat()
        end = now.isoformat()
        # Strip tz info for SQLite compatibility: +00:00 -> empty
        start = start.replace("+00:00", "")
        end = end.replace("+00:00", "")

        res = await client.get(f"/api/events/top-behaviours?start_time={start}&end_time={end}&limit=5")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Top behaviour should be product_thrown (2 events)
        assert data[0]["behaviour_type"] == "3_product_thrown"
        assert data[0]["count"] == 2

        # Verify field names match Dashboard.tsx expectations
        for field in ["behaviour_type", "count", "avg_risk"]:
            assert field in data[0], f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_bay_summary(self, client: AsyncClient, seeded_data):
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        start = (now - dt.timedelta(hours=1)).isoformat().replace("+00:00", "")
        end = now.isoformat().replace("+00:00", "")

        res = await client.get(f"/api/events/bay-summary/bay_A?start_time={start}&end_time={end}")
        assert res.status_code == 200
        data = res.json()
        assert data["bay_id"] == "bay_A"
        assert data["total_events"] == 3
        assert "tier_counts" in data

    @pytest.mark.asyncio
    async def test_explain_endpoint(self, client: AsyncClient, seeded_data):
        event_id = seeded_data["events"][0].id
        res = await client.get(f"/api/events/explain/{event_id}")
        assert res.status_code == 200
        data = res.json()
        assert "explanation" in data
        assert data["event_id"] == event_id

    @pytest.mark.asyncio
    async def test_explain_not_found(self, client: AsyncClient):
        res = await client.get("/api/events/explain/99999")
        assert res.status_code == 404


# =========================================================================
# Feedback API
# =========================================================================


class TestFeedbackAPI:
    @pytest.mark.asyncio
    async def test_submit_confirm(self, client: AsyncClient, seeded_data):
        event_id = seeded_data["events"][4].id  # status: "observed"
        res = await client.post("/api/feedback/", json={
            "event_id": event_id,
            "reviewer": "test_supervisor",
            "action": "confirm",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["event_id"] == event_id
        assert data["action"] == "confirm"
        assert data["old_status"] == "observed"
        assert data["new_status"] == "potential"  # observed → potential on confirm
        assert data["reviewer"] == "test_supervisor"

    @pytest.mark.asyncio
    async def test_submit_dismiss(self, client: AsyncClient, seeded_data):
        event_id = seeded_data["events"][4].id
        res = await client.post("/api/feedback/", json={
            "event_id": event_id,
            "reviewer": "test_supervisor",
            "action": "dismiss",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["new_status"] == "dismissed"

    @pytest.mark.asyncio
    async def test_submit_invalid_action(self, client: AsyncClient, seeded_data):
        event_id = seeded_data["events"][0].id
        res = await client.post("/api/feedback/", json={
            "event_id": event_id,
            "reviewer": "test",
            "action": "invalid_action",
        })
        assert res.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_nonexistent_event(self, client: AsyncClient):
        res = await client.post("/api/feedback/", json={
            "event_id": 99999,
            "reviewer": "test",
            "action": "confirm",
        })
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_feedback_stats(self, client: AsyncClient, seeded_data):
        res = await client.get("/api/feedback/stats")
        assert res.status_code == 200
        data = res.json()
        assert "confirms" in data
        assert "dismissals" in data
        assert "total" in data
        assert "false_positive_rate" in data
        # 2 confirms, 2 dismissals from seeded data
        assert data["confirms"] == 2
        assert data["dismissals"] == 2
        assert data["total"] == 4
        assert data["false_positive_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_recalibrate_with_feedback(self, client: AsyncClient, seeded_data):
        res = await client.post("/api/feedback/recalibrate")
        assert res.status_code == 200
        data = res.json()
        assert "false_positive_rate" in data
        assert "adjustment_direction" in data
        assert "per_behaviour" in data
        # 50% FPR should trigger "stable" or "decrease_sensitivity"
        assert data["adjustment_direction"] in ("stable", "decrease_sensitivity")

    @pytest.mark.asyncio
    async def test_recalibrate_no_feedback(self, client: AsyncClient):
        """With no feedback, recalibration should return no_data message."""
        res = await client.post("/api/feedback/recalibrate")
        assert res.status_code == 200
        data = res.json()
        assert data["adjustment_direction"] == "no_data"
        assert data["total_feedback"] == 0
        assert "message" in data

    @pytest.mark.asyncio
    async def test_feedback_updates_event_status(self, client: AsyncClient, seeded_data):
        """After confirming an observed event, it should move to potential."""
        event_id = seeded_data["events"][4].id  # observed
        res = await client.post("/api/feedback/", json={
            "event_id": event_id,
            "reviewer": "test",
            "action": "confirm",
        })
        assert res.status_code == 200

        # Fetch event detail and verify status changed
        res2 = await client.get(f"/api/events/{event_id}")
        assert res2.status_code == 200
        assert res2.json()["status"] == "potential"


# =========================================================================
# Assistant API
# =========================================================================


class TestAssistantAPI:
    @pytest.mark.asyncio
    async def test_chat_returns_grounded_response(self, client: AsyncClient, seeded_data):
        res = await client.post("/api/assistant/chat", json={
            "query": "Show me all high-risk handling events",
        })
        assert res.status_code == 200
        data = res.json()
        assert "response" in data
        assert "tool_calls" in data
        assert "grounded" in data
        assert data["grounded"] is True
        assert isinstance(data["tool_calls"], list)
        assert len(data["tool_calls"]) > 0  # Should have called get_events tool

    @pytest.mark.asyncio
    async def test_chat_mentions_real_data(self, client: AsyncClient, seeded_data):
        """The mock LLM should execute tools against real DB and return actual data."""
        res = await client.post("/api/assistant/chat", json={
            "query": "What were the most common risky behaviours?",
        })
        assert res.status_code == 200
        data = res.json()
        # The response should contain actual data from the DB
        assert data["grounded"] is True
        # Tool call should be get_top_behaviours
        tool_names = [tc["tool"] for tc in data["tool_calls"]]
        assert "get_top_behaviours" in tool_names

    @pytest.mark.asyncio
    async def test_chat_bay_query(self, client: AsyncClient, seeded_data):
        res = await client.post("/api/assistant/chat", json={
            "query": "Which loading bay had the highest number of risky events?",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["grounded"] is True
        tool_names = [tc["tool"] for tc in data["tool_calls"]]
        assert "get_bay_summary" in tool_names

    @pytest.mark.asyncio
    async def test_chat_explain_query(self, client: AsyncClient, seeded_data):
        event_id = seeded_data["events"][0].id
        res = await client.post("/api/assistant/chat", json={
            "query": f"Why was event {event_id} classified as high risk?",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["grounded"] is True
        # Tool calls should include get_events or explain_event
        tool_names = [tc["tool"] for tc in data["tool_calls"]]
        assert len(tool_names) > 0

    @pytest.mark.asyncio
    async def test_list_tools(self, client: AsyncClient):
        res = await client.get("/api/assistant/tools")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        tool_names = [t["name"] for t in data]
        assert "get_events" in tool_names
        assert "get_top_behaviours" in tool_names
        assert "get_bay_summary" in tool_names
        assert "explain_event" in tool_names


# =========================================================================
# Bays API
# =========================================================================


class TestBaysAPI:
    @pytest.mark.asyncio
    async def test_list_bays_empty(self, client: AsyncClient):
        res = await client.get("/api/bays/")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        # No cameras in DB → no bays
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_list_bays_with_events(self, client: AsyncClient, seeded_data):
        res = await client.get("/api/bays/")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 2  # bay_A and bay_B

        # Verify field names match Heatmap.tsx expectations
        bay = data[0]
        for field in ["bay_id", "event_count", "avg_risk", "critical", "high", "medium", "low"]:
            assert field in bay, f"Missing field: {field}"

        # bay_A should have 3 events, bay_B should have 2
        bay_a = next(b for b in data if b["bay_id"] == "bay_A")
        bay_b = next(b for b in data if b["bay_id"] == "bay_B")
        assert bay_a["event_count"] == 3
        assert bay_b["event_count"] == 2

    @pytest.mark.asyncio
    async def test_list_bays_tier_breakdown(self, client: AsyncClient, seeded_data):
        res = await client.get("/api/bays/")
        data = res.json()
        bay_a = next(b for b in data if b["bay_id"] == "bay_A")
        # bay_A: 2 Critical, 1 High, 0 Medium, 0 Low
        assert bay_a["critical"] == 2
        assert bay_a["high"] == 1
        assert bay_a["medium"] == 0
        assert bay_a["low"] == 0

    @pytest.mark.asyncio
    async def test_list_operators_empty(self, client: AsyncClient):
        res = await client.get("/api/bays/operators")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_list_operators_with_data(self, client: AsyncClient, seeded_data):
        res = await client.get("/api/bays/operators")
        assert res.status_code == 200
        data = res.json()
        # Seeded data has person tracks with events
        assert isinstance(data, list)
        if len(data) > 0:
            op = data[0]
            for field in ["operator_id", "name", "events_total", "avg_risk"]:
                assert field in op, f"Missing field: {field}"


# =========================================================================
# Ingestion API
# =========================================================================


class TestIngestionAPI:
    @pytest.mark.asyncio
    async def test_upload_returns_metadata(self, client: AsyncClient):
        """Upload endpoint should return video_id, filename, path, metadata."""
        # Create a minimal fake file (won't have valid video metadata but tests the route)
        files = {"file": ("test_video.mp4", b"fake_video_content", "video/mp4")}
        res = await client.post("/api/ingestion/upload", files=files)
        assert res.status_code == 200
        data = res.json()
        assert "video_id" in data
        assert data["filename"] == "test_video.mp4"
        assert "path" in data
        assert "metadata" in data

    @pytest.mark.asyncio
    async def test_upload_deterministic_id(self, client: AsyncClient):
        """Same file should produce same video_id."""
        files1 = {"file": ("test.mp4", b"content_abc", "video/mp4")}
        files2 = {"file": ("test.mp4", b"content_abc", "video/mp4")}
        res1 = await client.post("/api/ingestion/upload", files=files1)
        res2 = await client.post("/api/ingestion/upload", files=files2)
        assert res1.json()["video_id"] == res2.json()["video_id"]

    @pytest.mark.asyncio
    async def test_upload_different_files_different_ids(self, client: AsyncClient):
        files1 = {"file": ("a.mp4", b"content_a", "video/mp4")}
        files2 = {"file": ("b.mp4", b"content_b", "video/mp4")}
        res1 = await client.post("/api/ingestion/upload", files=files1)
        res2 = await client.post("/api/ingestion/upload", files=files2)
        assert res1.json()["video_id"] != res2.json()["video_id"]

    @pytest.mark.asyncio
    async def test_metadata_endpoint(self, client: AsyncClient):
        # Upload a file first
        files = {"file": ("test.mp4", b"fake", "video/mp4")}
        upload_res = await client.post("/api/ingestion/upload", files=files)
        video_path = upload_res.json()["path"]

        # Get metadata — won't return real video info for fake content, but tests the route
        res = await client.get(f"/api/ingestion/metadata?video_path={video_path}")
        # Could be 404 if OpenCV can't read fake content, or 200 with empty metadata
        assert res.status_code in (200, 404)


# =========================================================================
# Frontend ↔ Backend Contract Tests
# =========================================================================


class TestFrontendContract:
    """Verify that every API response field matches what the frontend expects.

    Frontend expectations extracted from:
    - LiveFeed.tsx: Event interface
    - IncidentReplay.tsx: EventDetail interface
    - Dashboard.tsx: TopBehaviour, FeedbackStats interfaces
    - Heatmap.tsx: BayData interface
    - Assistant.tsx: ChatResponse interface
    """

    @pytest.mark.asyncio
    async def test_livefeed_event_contract(self, client: AsyncClient, seeded_data):
        """LiveFeed.tsx Event interface fields."""
        res = await client.get("/api/events/")
        data = res.json()
        assert len(data) > 0

        event = data[0]
        # Fields LiveFeed.tsx reads:
        assert isinstance(event["id"], int)
        assert isinstance(event["behaviour_type"], str)
        assert isinstance(event["tier"], str)
        assert isinstance(event["risk_score"], (int, float))
        assert isinstance(event["status"], str)
        assert isinstance(event["confidence"], (int, float))
        assert isinstance(event["timestamp"], str)  # ISO format
        # Optional field
        if event.get("vlm_explanation") is not None:
            assert isinstance(event["vlm_explanation"], str)

    @pytest.mark.asyncio
    async def test_incident_replay_event_contract(self, client: AsyncClient, seeded_data):
        """IncidentReplay.tsx EventDetail interface fields."""
        event_id = seeded_data["events"][0].id
        res = await client.get(f"/api/events/{event_id}")
        assert res.status_code == 200
        data = res.json()

        # Fields IncidentReplay.tsx reads:
        assert isinstance(data["event_id"], int)
        assert isinstance(data["behaviour_type"], str)
        assert isinstance(data["risk_score"], (int, float))
        assert isinstance(data["tier"], str)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["status"], str)
        assert isinstance(data["features"], dict)
        assert isinstance(data["vlm_explanation"], (str, type(None)))
        assert isinstance(data["risk_breakdown"], dict)
        assert isinstance(data["explanation"], str)

    @pytest.mark.asyncio
    async def test_dashboard_top_behaviours_contract(self, client: AsyncClient, seeded_data):
        """Dashboard.tsx TopBehaviour interface fields."""
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        start = (now - dt.timedelta(hours=1)).isoformat().replace("+00:00", "")
        end = now.isoformat().replace("+00:00", "")

        res = await client.get(f"/api/events/top-behaviours?start_time={start}&end_time={end}&limit=10")
        assert res.status_code == 200
        data = res.json()
        assert len(data) > 0

        for item in data:
            assert isinstance(item["behaviour_type"], str)
            assert isinstance(item["count"], int)
            assert isinstance(item["avg_risk"], (int, float))

    @pytest.mark.asyncio
    async def test_dashboard_feedback_stats_contract(self, client: AsyncClient, seeded_data):
        """Dashboard.tsx FeedbackStats interface fields."""
        res = await client.get("/api/feedback/stats")
        assert res.status_code == 200
        data = res.json()

        assert isinstance(data["confirms"], int)
        assert isinstance(data["dismissals"], int)
        assert isinstance(data["total"], int)
        assert isinstance(data["false_positive_rate"], (int, float))

    @pytest.mark.asyncio
    async def test_heatmap_bay_data_contract(self, client: AsyncClient, seeded_data):
        """Heatmap.tsx BayData interface fields."""
        res = await client.get("/api/bays/")
        assert res.status_code == 200
        data = res.json()
        assert len(data) > 0

        bay = data[0]
        assert isinstance(bay["bay_id"], str)
        assert isinstance(bay["event_count"], int)
        assert isinstance(bay["avg_risk"], (int, float))
        assert isinstance(bay["critical"], int)
        assert isinstance(bay["high"], int)
        assert isinstance(bay["medium"], int)
        assert isinstance(bay["low"], int)

    @pytest.mark.asyncio
    async def test_assistant_chat_response_contract(self, client: AsyncClient, seeded_data):
        """Assistant.tsx ChatResponse interface fields."""
        res = await client.post("/api/assistant/chat", json={"query": "Show events"})
        assert res.status_code == 200
        data = res.json()

        assert isinstance(data["response"], str)
        assert isinstance(data["tool_calls"], list)
        assert isinstance(data["grounded"], bool)

        if data["tool_calls"]:
            tc = data["tool_calls"][0]
            assert isinstance(tc["tool"], str)
            assert isinstance(tc["arguments"], dict)
            assert isinstance(tc["result"], str)

    @pytest.mark.asyncio
    async def test_feedback_submit_response_contract(self, client: AsyncClient, seeded_data):
        """IncidentReplay.tsx reads feedback response fields."""
        event_id = seeded_data["events"][4].id
        res = await client.post("/api/feedback/", json={
            "event_id": event_id,
            "reviewer": "supervisor",
            "action": "confirm",
        })
        assert res.status_code == 200
        data = res.json()

        assert isinstance(data["event_id"], int)
        assert isinstance(data["action"], str)
        assert isinstance(data["old_status"], str)
        assert isinstance(data["new_status"], str)
        assert isinstance(data["reviewer"], str)

    @pytest.mark.asyncio
    async def test_recalibration_response_contract(self, client: AsyncClient, seeded_data):
        """Dashboard.tsx reads recalibration response fields."""
        res = await client.post("/api/feedback/recalibrate")
        assert res.status_code == 200
        data = res.json()

        assert isinstance(data["false_positive_rate"], (int, float))
        assert isinstance(data["adjustment_direction"], str)
        assert isinstance(data["confidence_threshold_adjustment"], (int, float))
        assert isinstance(data["per_behaviour"], dict)

    @pytest.mark.asyncio
    async def test_no_null_timestamps(self, client: AsyncClient, seeded_data):
        """Events should never have null timestamps (LiveFeed renders them)."""
        res = await client.get("/api/events/")
        for event in res.json():
            assert event["timestamp"] is not None, f"Event {event['id']} has null timestamp"
