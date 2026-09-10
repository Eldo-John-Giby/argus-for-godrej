"""SQLAlchemy ORM models — all core tables."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bay_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128))
    calibration_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    tracks: Mapped[list["Track"]] = relationship(back_populates="camera")


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), index=True)
    object_class: Mapped[str] = mapped_column(String(64))  # person, forklift, box …
    byte_track_id: Mapped[int] = mapped_column(Integer)
    start_ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    end_ts: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    camera: Mapped["Camera"] = relationship(back_populates="tracks")
    events: Mapped[list["Event"]] = relationship(back_populates="track")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), index=True)
    behaviour_type: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    clip_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    keyframe_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    features_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    vlm_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    tier: Mapped[str] = mapped_column(String(32), default="Low")  # Low/Medium/High/Critical
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(
        String(32), default="observed", index=True
    )  # observed/potential/confirmed/dismissed

    track: Mapped["Track"] = relationship(back_populates="events")
    feedback_entries: Mapped[list["Feedback"]] = relationship(back_populates="event")


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(32))  # confirm / dismiss
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    event: Mapped["Event"] = relationship(back_populates="feedback_entries")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    class_name: Mapped[str] = mapped_column(String(64), unique=True)
    fragility_multiplier: Mapped[float] = mapped_column(Float, default=1.0)


class ShiftSummary(Base):
    __tablename__ = "shift_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bay_id: Mapped[str] = mapped_column(String(64), index=True)
    shift_date: Mapped[dt.date] = mapped_column(Date())
    top_behaviours_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    event_counts_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
