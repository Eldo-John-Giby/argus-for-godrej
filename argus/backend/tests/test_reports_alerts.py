"""Integration tests for shift-report and alert endpoints."""

from __future__ import annotations

import datetime as dt

import pytest
from httpx import AsyncClient

from app.models import Event
from tests.conftest import seed_camera, seed_event, seed_feedback, seed_track


@pytest.fixture()
def past_seeded_data(db_session):
    """Seed events in the past 24h (the report window used in these tests)."""
    now = dt.datetime.now(dt.timezone.utc)

    async def _seed():
        cam = await seed_camera(db_session, bay_id="bay_A", name="cam_A1")
        cam2 = await seed_camera(db_session, bay_id="bay_B", name="cam_B1")
        track = await seed_track(db_session, camera_id=cam.id, object_class="person")
        track2 = await seed_track(db_session, camera_id=cam2.id, object_class="person")

        events = []
        for i, (behaviour, score, tier) in enumerate([
            ("3_product_thrown", 87.2, "Critical"),
            ("4_improper_stacking", 55.0, "High"),
            ("2_product_dragged", 35.0, "Medium"),
        ]):
            e = Event(
                track_id=track.id if i < 2 else track2.id,
                behaviour_type=behaviour,
                ts=now - dt.timedelta(hours=1),
                features_json={},
                risk_score=score,
                tier=tier,
                confidence=0.8,
                status="potential",
                vlm_explanation="test explanation",
            )
            db_session.add(e)
            events.append(e)
        await db_session.commit()
        for e in events:
            await db_session.refresh(e)
        await seed_feedback(db_session, event_id=events[0].id, action="confirm")
        await seed_feedback(db_session, event_id=events[1].id, action="dismiss")
        return events

    return _seed


def _window(hours: float = 8.0) -> tuple[str, str]:
    """ISO start/end with timezone stripped for SQLite naive datetimes."""
    now = dt.datetime.now(dt.timezone.utc)
    start = (now - dt.timedelta(hours=hours)).isoformat().replace("+00:00", "")
    end = now.isoformat().replace("+00:00", "")
    return start, end


class TestShiftReportAPI:
    @pytest.mark.asyncio
    async def test_report_json(self, client: AsyncClient, past_seeded_data):
        await past_seeded_data()
        start, end = _window()
        res = await client.get(f"/api/reports/shift?start_time={start}&end_time={end}")
        assert res.status_code == 200
        data = res.json()

        assert data["total_events"] == 3
        assert data["tier_counts"]["Critical"] == 1
        assert data["tier_counts"]["High"] == 1
        assert data["tier_counts"]["Medium"] == 1
        assert data["top_behaviour"] is not None
        assert len(data["behaviours"]) == 3
        assert len(data["bays"]) == 2

        # Feedback in window: 1 confirm + 1 dismiss
        assert data["feedback"]["confirms"] == 1
        assert data["feedback"]["dismissals"] == 1
        assert data["feedback"]["false_positive_rate"] == 0.5

        # Report text is rendered and mentions the numbers
        assert "report_text" in data
        assert "3 events" in data["report_text"]
        assert "Top behaviour" in data["report_text"]
        assert "Recommended action" in data["report_text"]

    @pytest.mark.asyncio
    async def test_report_text_format(self, client: AsyncClient, past_seeded_data):
        await past_seeded_data()
        start, end = _window()
        res = await client.get(f"/api/reports/shift?start_time={start}&end_time={end}&format=text")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/plain")
        assert "Shift report" in res.text

    @pytest.mark.asyncio
    async def test_report_hours_param(self, client: AsyncClient, past_seeded_data):
        await past_seeded_data()
        res = await client.get("/api/reports/shift?hours=24")
        assert res.status_code == 200
        assert res.json()["total_events"] == 3

    @pytest.mark.asyncio
    async def test_report_bay_filter(self, client: AsyncClient, past_seeded_data):
        await past_seeded_data()
        start, end = _window()
        res = await client.get(f"/api/reports/shift?start_time={start}&end_time={end}&bay_id=bay_A")
        assert res.status_code == 200
        data = res.json()
        assert data["total_events"] == 2  # only bay_A events
        assert data["bay_id"] == "bay_A"

    @pytest.mark.asyncio
    async def test_report_empty_window(self, client: AsyncClient):
        start, end = _window(hours=1)
        res = await client.get(f"/api/reports/shift?start_time={start}&end_time={end}")
        assert res.status_code == 200
        data = res.json()
        assert data["total_events"] == 0
        assert data["top_behaviour"] is None
        assert data["feedback"]["false_positive_rate"] is None

    @pytest.mark.asyncio
    async def test_report_no_fabricated_action(self, client: AsyncClient, past_seeded_data):
        """Known behaviours must map to their fixed recommendation text."""
        await past_seeded_data()
        start, end = _window()
        res = await client.get(f"/api/reports/shift?start_time={start}&end_time={end}&bay_id=bay_A")
        text = res.json()["report_text"]
        # product thrown -> fixed recommendation
        assert "corrective briefing" in text.lower()


class TestAlertsAPI:
    @pytest.mark.asyncio
    async def test_languages(self, client: AsyncClient):
        res = await client.get("/api/alerts/languages")
        assert res.status_code == 200
        codes = [item["code"] for item in res.json()]
        assert "en" in codes
        assert "hi" in codes

    @pytest.mark.asyncio
    async def test_alert_text_english(self, client: AsyncClient, past_seeded_data):
        events = await past_seeded_data()
        res = await client.get(f"/api/alerts/text?event_id={events[0].id}&lang=en")
        assert res.status_code == 200
        data = res.json()
        assert data["event_id"] == events[0].id
        assert data["translated"] is False
        assert "87" in data["text"]  # risk score rendered
        assert "thrown" in data["text"]

    @pytest.mark.asyncio
    async def test_alert_text_translation_fallback(self, client: AsyncClient, past_seeded_data):
        """No translation API configured -> English text, translated=False, no error."""
        events = await past_seeded_data()
        res = await client.get(f"/api/alerts/text?event_id={events[0].id}&lang=hi")
        assert res.status_code == 200
        data = res.json()
        assert data["language"] in ("hi", "en")
        assert data["text"]  # always speakable text
        if not data["translated"]:
            assert data["text"] == data["base_text"]

    @pytest.mark.asyncio
    async def test_alert_text_unknown_event(self, client: AsyncClient):
        res = await client.get("/api/alerts/text?event_id=99999")
        assert res.status_code == 200
        assert res.json()["text"] is None

    @pytest.mark.asyncio
    async def test_speak_without_tts_returns_204(self, client: AsyncClient, past_seeded_data):
        """Default TTS_PROVIDER=none -> 204, frontend falls back to browser TTS."""
        events = await past_seeded_data()
        res = await client.get(f"/api/alerts/speak?event_id={events[0].id}&lang=en")
        assert res.status_code == 204

    @pytest.mark.asyncio
    async def test_speak_unknown_event_returns_404(self, client: AsyncClient):
        res = await client.get("/api/alerts/speak?event_id=99999")
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_lang_param_normalised(self, client: AsyncClient, past_seeded_data):
        """'hi-IN' style codes should work the same as 'hi'."""
        events = await past_seeded_data()
        res = await client.get(f"/api/alerts/text?event_id={events[0].id}&lang=hi-IN")
        assert res.status_code == 200
        assert res.json()["text"]
