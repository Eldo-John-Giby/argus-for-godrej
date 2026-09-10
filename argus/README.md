# ARGUS — AI Field Intelligence for Warehouse Handling

> *"Argus never blinks — an AI that watches every load so nothing gets dropped."*

**Godrej Enterprises Group × graVITas'2026 AI Hackathon (Track 02)**

---

## Quick Start

### Demo / presenter path — one command, no setup

Prerequisites on the machine: **Python 3.10+** and **Node.js 18+**. Nothing else —
no Postgres, no Docker, no API keys. The SQLite database, the pre-computed events
in `ml/results`, and the replay clips all ship with this folder.

```bash
cd argus
python run_demo.py            # or double-click start_demo.bat on Windows
```

That single command:
1. installs backend + frontend dependencies (first run only)
2. seeds the database from `ml/results` (only if empty)
3. starts the backend (8000) and frontend (5173)
4. runs the sanity checks a judge could ask about
5. prints the 60-second demo path

```bash
python run_demo.py stop       # stop the stack (also frees the ports)
python run_demo.py seed       # wipe + reseed the demo data
python run_demo.py doctor     # pre-flight checks only
python run_demo.py --detach   # keep the stack running after the terminal closes
./start_demo.sh               # bash alias for the same thing (also has a docker mode)
```

First run downloads ~100 MB of packages and takes 2-3 minutes; every run after
that starts in under 15 seconds. If the demo machine has no internet, run the
launcher once beforehand — after that the demo works fully offline (grounded
assistant uses its DB fallback, alerts use browser TTS; only live VLM
verification degrades).

### Docker (alternative)
```bash
cd argus
docker compose up --build
```

- **Backend API**: http://localhost:8000
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

### Manual (development)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python seed_db.py     # SQLite by default; set DATABASE_URL for Postgres
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ARGUS Pipeline                         │
├─────────────┬───────────────┬──────────────┬───────────────┤
│ Perception  │  Behaviour    │  Risk &      │  Product      │
│ (YOLO26 +   │  Engine       │  Verification│  (Dashboard,  │
│  ByteTrack) │  (FSM + Geo)  │  (VLM)       │  Assistant,   │
│             │               │              │  Heatmap)     │
└─────────────┴───────────────┴──────────────┴───────────────┘
```

### Three Core Differentiators

1. **Physics-informed risk score** — computed from drop height, impact velocity proxy, stacking geometry, and fragility. Not a lookup table dressed as AI.

2. **Confidence ladder with feedback loop** — Observed → Potential → Confirmed, with human-in-the-loop confirm/dismiss that visibly recalibrates thresholds. Almost nobody builds this.

3. **Grounded assistant** — answers ONLY from the events database via tool calling. Responds to the brief's own sample questions verbatim.

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Detection/Tracking | Ultralytics YOLO + ByteTrack |
| Behaviour Engine | Python FSM + OpenCV geometry |
| VLM Verification | Qwen3-VL / Gemini 3 Flash |
| Assistant | Claude/GPT via tool-calling |
| Backend | FastAPI (Python) |
| Database | PostgreSQL |
| Frontend | React + Tailwind + Recharts |
| Deployment | Docker Compose |

---

## Behaviour Taxonomy (13 behaviours)

| # | Behaviour | CV Signal |
|---|-----------|-----------|
| 1 | Product Dropped | Downward velocity spike + ground contact |
| 2 | Product Dragged | Bottom-edge near floor + horizontal motion |
| 3 | Product Thrown | Ballistic parabolic trajectory fit |
| 4 | Improper Stacking | Top footprint > bottom footprint |
| 5 | Unstable Stacking | Stack vertical axis tilt |
| 6 | Stepping on Packages | Foot keypoints intersect package bbox |
| 7 | Outside Designated Zone | Homography-mapped zone violation |
| 8 | Dragging No Trolley | Drag signal + no trolley detected |
| 9 | Product Overhang | Product vs pallet footprint ratio |
| 10 | Rough Handling | High jerk at hand-off points |
| 11 | Wrong Orientation | Aspect ratio vs configured orientation |
| 12 | Cluttered Staging | Object density in staging zone |
| 13 | PPE Non-compliance | Second YOLO detector: NO-helmet / NO-vest box on a tracked person |

**PPE module** (`ml/ppe_detector.py`): an optional second detector running on the same frames.
Drop a pretrained helmet/vest weight at `ml/ppe_best.pt` (or `ml/best.pt`, or set `PPE_MODEL_PATH`)
and it activates automatically; without a weight it stays dormant at zero cost.
Validate the weights on your own footage first — they are trained on construction-site imagery:

```bash
python ml/ppe_detector.py path/to/frame.jpg     # sanity-check CLI, writes ppe_sanity_check.jpg
python ml/run_on_videos.py --no-ppe             # run pipeline without PPE
python ml/run_on_videos.py --ppe-model best.pt  # explicit weight path

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/events/` | List events with filters |
| GET | `/api/events/{id}` | Get event details |
| GET | `/api/events/top-behaviours` | Most common behaviours |
| GET | `/api/events/bay-summary/{bay_id}` | Bay event summary |
| GET | `/api/events/explain/{id}` | Why this event was flagged |
| POST | `/api/feedback/` | Submit human feedback |
| GET | `/api/feedback/stats` | Feedback statistics |
| POST | `/api/feedback/recalibrate` | Trigger threshold recalibration |
| POST | `/api/assistant/chat` | Chat with grounded assistant |
| GET | `/api/assistant/tools` | List available tools |
| GET | `/api/reports/shift` | Automatic shift report (JSON or `&format=text`) |
| GET | `/api/alerts/languages` | Voice-alert languages |
| GET | `/api/alerts/text` | Composed + translated alert text for an event |
| GET | `/api/alerts/speak` | Alert as audio (204 → browser TTS fallback) |

---

## Frontend Pages

- **Live Feed** — Multi-bay live event stream with tier badges and status indicators
- **Incident Replay** — Video + canvas overlay with bbox/skeleton, risk breakdown, "why flagged" panel, confirm/dismiss buttons
- **Dashboard** — Shift summary, top behaviours chart, precision/recall, false-positive rate, recalibration button
- **Heatmap** — Bay/location risk heatmap with tier distribution
- **Assistant** — Chat widget with visible tool-call trace (judges can see the grounding)

---

## Responsible AI Features

- **Face/ID blurring** toggle on stored clips (default: on)
- **Retention policy config** (auto-purge non-flagged footage after N days)
- **Role-based access** — Operator (own shift) / Supervisor (bay-wide) / Auditor (full history)
- **No automated punitive action** — reports events and locations, identity behind explicit logged review
- **False-positive rate displayed** on the dashboard — honest error reporting

---

## Project Structure

```
argus/
├── run_demo.py               # ONE-COMMAND launcher: deps + seed + start + sanity checks
├── start_demo.sh             # bash wrapper (delegates to run_demo.py; docker mode)
├── start_demo.bat            # Windows double-click wrapper
├── backend/
│   ├── app/
│   │   ├── api/              # REST routes (events, feedback, assistant)
│   │   ├── perception/       # YOLO26 + ByteTrack wrapper
│   │   ├── behaviour/        # FSM, features, taxonomy
│   │   ├── vlm/              # VLM verification + clip extraction
│   │   ├── risk/             # Risk scoring + confidence gate
│   │   ├── events/           # SQLAlchemy CRUD
│   │   ├── feedback/         # Human feedback + recalibration
│   │   ├── assistant/        # Grounded tool-calling agent
│   │   ├── calibration/      # Camera calibration
│   │   ├── ingestion/        # Video upload + frame extraction
│   │   ├── config.py         # Central configuration
│   │   ├── database.py       # Async DB setup
│   │   └── models.py         # ORM models
│   ├── tests/                # FSM + risk scoring unit tests
│   └── main.py               # FastAPI entry point
├── frontend/
│   ├── src/
│   │   ├── pages/            # LiveFeed, IncidentReplay, Dashboard, Heatmap, Assistant
│   │   └── components/       # OverlayCanvas
│   └── ...config files
├── ml/
│   ├── clips/                # Labeled clips
│   └── eval.py               # Precision/recall/FPR evaluation
├── docker-compose.yml
└── README.md
```

---

## Evaluation

Video-level evaluation against the ground truth encoded in each judge video's name:

```bash
cd ml
python run_on_videos.py        # process all 7 pilot videos -> results/<video>/report.json
python eval.py                 # -> results/eval_metrics.json
```

Latest measured numbers (`ml/results/eval_metrics.json` — regenerate after pipeline changes):

- **Core behaviours (the 12 with ground truth): precision 0.39 · recall 0.42 · F1 0.40**
- PPE module: 18 non-compliance events detected across 7 videos (reported separately — the
  judge video names carry no PPE ground truth, so including it would count true detections
  as false positives)
- Per-video breakdown is in `eval_metrics.json` — present the honest numbers, not rounded-up claims

Risk-tier bands (`RISK_TIER_*` in `backend/app/config.py`) are per-site calibration; this
pilot footage's operational score range is 26–44, so bands are set to 20/35/50. Event scores
are never adjusted — only the display bands move per deployment.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./argus.db` (no setup needed; Postgres optional via `postgresql+asyncpg://argus:argus@localhost:5432/argus`) | Database connection |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model path |
| `VLM_PROVIDER` | `qwen` | VLM provider (qwen/gemini) |
| `QWEN_API_KEY` | - | Qwen API key |
| `GEMINI_API_KEY` | - | Gemini API key |
| `ASSISTANT_API_KEY` | - | Claude/OpenAI API key |
| `ASSISTANT_LLM` | `claude` | Assistant LLM provider |
| `FACE_BLUR` | `true` | Enable face blurring |
| `RETENTION_DAYS` | `30` | Footage retention period |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Gemini model id (REST) |
| `ALERT_TRANSLATION_URL` | - | OpenAI-compatible translation endpoint |
| `ALERT_TRANSLATION_API_KEY` | - | Translation API key |
| `TTS_PROVIDER` | `none` | `gtts` enables server-side speech (else browser TTS) |
| `PPE_MODEL_PATH` | - | Explicit path to PPE weights |
| `W7` | `0.10` | Location factor weight in risk score |
| `BAY_HAZARD_A` / `_B` / `_C` | `1.0` / `1.15` / `1.25` | Per-bay hazard multipliers (location input) |
| `RETENTION_SWEEP_HOURS` | `24` | How often the retention worker purges expired events |

Fragility multipliers per product class live in
`backend/app/behaviour/taxonomy.yaml` (Fragile 1.5 / Standard 1.0 / Rugged
0.7); detector classes map to tiers via `CLASS_TIER` in `taxonomy.py`.

> Copy `backend/.env` (gitignored) for local keys — the config loads it automatically;
> real environment variables always take precedence.

---

## Demo-day runbook

### Minimal path for the presenter (recommended)

Copy the whole `argus/` folder to the demo machine (including `clips/` and
`ml/results/` — they carry the replay evidence and seed data). Then:

```bash
cd argus
python run_demo.py        # Windows: double-click start_demo.bat
```

Stop with `python run_demo.py stop`; reseed anytime with `python run_demo.py seed`.

### Full runbook

```bash
# 1. Fresh data (regenerates reports with the current pipeline, then reseeds)
cd argus/ml && python run_on_videos.py --no-clips && python eval.py
cd ../backend && python seed_db.py

# 2. Start the stack — one command, with built-in sanity checks
python run_demo.py             # self-bootstrapping: deps + seed + start + checks
./start_demo.sh docker         # full container path (needs Docker Desktop)

# Docker + seeded data: seed the compose Postgres from the host
DATABASE_URL=postgresql+asyncpg://argus:argus@localhost:5432/argus python seed_db.py

# 3. 60-second sanity check
open http://localhost:5173                 # Live Feed shows events, tiers, +N badge
# - click an event row  -> replay + risk breakdown + confirm/dismiss
# - Dashboard           -> generate shift report, read aloud, download
# - Assistant           -> ask "Which loading bay had the highest number of risky events?"
# - flip Alert voice to Hindi, then Confirm a High event in another tab -> toast + voice

# 4. Pre-pitch checks
python -m pytest tests/ -q                 # backend: 93 tests
cd ../frontend && npx tsc -b               # frontend: typecheck
```

If the demo machine has no internet: everything critical works offline — the grounded
assistant answers from the DB via its fallback LLM, alerts speak via browser TTS, and
reports/alerts need no external API. Only VLM verification and server-side translation
degrade (loudly and gracefully).

## License

Built for Godrej Enterprises Group × graVITas'2026 AI Hackathon.
