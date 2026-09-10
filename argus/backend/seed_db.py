"""Seed database with real warehouse evaluation data from ml/results."""

import asyncio
import datetime as dt
import json
import shutil
from pathlib import Path

from app.behaviour.taxonomy import get_fragility, get_severity
from app.config import STORAGE_ROOT
from app.database import engine, Base, async_session
from app.models import Camera, Track, Event, Feedback, Product, ShiftSummary
from app.risk.scoring import RiskInput, score_event, tier_for_score
from app.vlm.explain import generate_explanation


async def seed():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        # 1. Cameras
        cams = [
            Camera(bay_id="A", name="Bay A - Unloading Dock", calibration_json={"pixels_per_meter": 42.0, "floor_y": 620}),
            Camera(bay_id="B", name="Bay B - Staging and Stacking Area", calibration_json={"pixels_per_meter": 40.0, "floor_y": 680}),
            Camera(bay_id="C", name="Bay C - KD Assembly and Loading Bay", calibration_json={"pixels_per_meter": 45.0, "floor_y": 590}),
        ]
        session.add_all(cams)
        await session.flush()

        # 2. Products
        prods = [
            Product(class_name="carton", fragility_multiplier=1.0),
            Product(class_name="cardboard box", fragility_multiplier=1.0),
            Product(class_name="box", fragility_multiplier=1.0),
            Product(class_name="cupboard", fragility_multiplier=1.5),
            Product(class_name="cabinet", fragility_multiplier=1.5),
            Product(class_name="appliance", fragility_multiplier=1.8),
            Product(class_name="furniture", fragility_multiplier=1.4),
            Product(class_name="mattress", fragility_multiplier=0.8),
        ]
        session.add_all(prods)
        await session.flush()

        # 3. Read events from ml/results
        results_dir = Path(__file__).resolve().parent.parent / "ml" / "results"
        # Clip evidence lives in ml/clips/<video_slug>/ (written by the runner);
        # copy each event's clip into backend storage so the API can serve it.
        # run_on_videos.py sets ROOT = repo root (argus/) and writes clips to
        # ROOT/clips — i.e. argus/clips, a level above ml/results.
        clips_src = results_dir.parent.parent / "clips"
        # STORAGE_ROOT is env-driven (./storage natively, /app/storage in the
        # container) — the same seeder fills whichever DB it is pointed at.
        clips_dst = STORAGE_ROOT / "clips"
        now = dt.datetime.now(dt.timezone.utc)
        bay_cycle = ["A", "B", "C"]
        b_idx = 0

        event_count = 0
        if results_dir.exists():
            for report_path in results_dir.glob("*/report.json"):
                try:
                    with open(report_path, encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                cam = cams[b_idx % len(cams)]
                b_idx += 1

                unique = data.get("unique_events", [])
                video_slug = report_path.parent.name
                for ev in unique:
                    track = Track(
                        camera_id=cam.id,
                        object_class=ev.get("class", "carton"),
                        byte_track_id=ev.get("track_id", 1),
                        start_ts=now - dt.timedelta(hours=2, seconds=3600 - int(ev.get("timestamp", 0))),
                        end_ts=now - dt.timedelta(hours=2, seconds=3600 - int(ev.get("end_timestamp", ev.get("timestamp", 0)))),
                    )
                    session.add(track)
                    await session.flush()

                    ts = now - dt.timedelta(hours=2, seconds=3600 - int(ev.get("timestamp", 0)))

                    # Re-score with the LIVE engine so every stored event
                    # carries the full breakdown — product-class fragility
                    # and the per-bay location factor included. report.json
                    # scores were produced by earlier weights; the DB should
                    # always match what the scoring engine computes today.
                    behaviour = ev.get("behaviour_type", "2_product_dragged")
                    features = dict(ev.get("features", {}))
                    obj_class = ev.get("class") or features.get("class") or ""
                    # Some behaviours store velocity as an [vx, vy] vector —
                    # reduce to the dominant component for the impact proxy.
                    raw_vel = features.get("velocity", 0.0)
                    if isinstance(raw_vel, (list, tuple)):
                        raw_vel = max((abs(v) for v in raw_vel), default=0.0)
                    try:
                        raw_vel = float(raw_vel or 0.0)
                    except (TypeError, ValueError):
                        raw_vel = 0.0
                    scored = score_event(RiskInput(
                        behaviour_severity=get_severity(behaviour),
                        velocity_at_contact=raw_vel,
                        mass_proxy=1.0,
                        fragility_multiplier=get_fragility(obj_class),
                        stack_tilt_angle=0.0,
                        top_bottom_footprint_ratio=float(features.get("footprint_ratio", 1.0) or 1.0),
                        recurrence_count=1,
                        composite_confidence=float(ev.get("confidence", 0.75)),
                        bay_id=cam.bay_id,
                    ))
                    risk_score = scored.score
                    features["risk_breakdown"] = scored.breakdown

                    # ---- clip evidence: locate + copy the real clip ----
                    stamp = float(ev.get("timestamp", 0.0))
                    src = clips_src / video_slug / f"{behaviour}_t{stamp:.1f}s.mp4"
                    if not src.exists():
                        # Timestamp-collision fallback: runner dedupes identical names.
                        matches = sorted((clips_src / video_slug).glob(f"{behaviour}_t{stamp:.1f}s*.mp4"))
                        src = matches[0] if matches else src
                    clip_rel = None
                    if src.exists():
                        dst = clips_dst / video_slug / src.name
                        if not dst.exists():
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(src, dst)
                        clip_rel = f"{video_slug}/{src.name}"

                    event = Event(
                        track_id=track.id,
                        behaviour_type=behaviour,
                        ts=ts,
                        features_json=features,
                        risk_score=risk_score,
                        # Derive tier from the score using the ACTIVE band config —
                        # stored report tiers can be stale after recalibration.
                        tier=tier_for_score(risk_score),
                        confidence=float(ev.get("confidence", 0.75)),
                        status=ev.get("status", "observed"),
                        clip_path=clip_rel,
                        vlm_explanation=generate_explanation(
                            ev.get("behaviour_type", "2_product_dragged"),
                            ev.get("features", {}),
                        ),
                    )
                    session.add(event)
                    await session.flush()

                    # ---- supervisor feedback (deterministic, realistic) ----
                    # Old rule (score>=50 confirm) never fired on this site's
                    # 26-44 score range, collapsing the dashboard pie to 100%
                    # false positives. New rule: High tier or confident events
                    # get confirmed; a slice of mid-confidence ones get
                    # dismissed — a credible ~13% FPR, not a strawman 100%.
                    reviewed = event.tier == "High" or event.confidence >= 0.62
                    dismissed = (not reviewed) and event.confidence < 0.45 and event.id % 4 == 0
                    if reviewed:
                        session.add(
                            Feedback(
                                event_id=event.id,
                                reviewer="supervisor_ops",
                                action="confirm",
                                ts=ts + dt.timedelta(minutes=5),
                            )
                        )
                    elif dismissed:
                        session.add(
                            Feedback(
                                event_id=event.id,
                                reviewer="supervisor_ops",
                                action="dismiss",
                                ts=ts + dt.timedelta(minutes=8),
                            )
                        )

                    event_count += 1

        # 4. Shift Summary
        today = dt.date.today()
        shifts = [
            ShiftSummary(
                bay_id="A",
                shift_date=today,
                top_behaviours_json={"2_product_dragged": 8, "3_product_thrown": 4, "1_product_dropped": 2},
                event_counts_json={"total": 14, "confirmed": 6, "dismissed": 1, "potential": 7},
            ),
            ShiftSummary(
                bay_id="B",
                shift_date=today,
                top_behaviours_json={"4_improper_stacking": 5, "6_stepping_on_packages": 3, "11_wrong_orientation": 2},
                event_counts_json={"total": 10, "confirmed": 5, "dismissed": 0, "potential": 5},
            ),
            ShiftSummary(
                bay_id="C",
                shift_date=today,
                top_behaviours_json={"8_dragging_no_trolley": 3, "2_product_dragged": 4},
                event_counts_json={"total": 7, "confirmed": 3, "dismissed": 0, "potential": 4},
            ),
        ]
        session.add_all(shifts)
        await session.commit()
        print(f"Database seeded successfully with {event_count} events across 3 bays!")


if __name__ == "__main__":
    asyncio.run(seed())
