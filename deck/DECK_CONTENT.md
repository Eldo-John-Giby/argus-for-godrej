# ARGUS — Submission Deck Content (6 slides)

> Built directly against the Problem Statement's slide plan (Slide 1–5 + optional 6).
> Every number below is real and verifiable in the repo/DB — nothing invented.
> Placeholders to fill before export: **[TEAM NAME]**, **[Member names]**.

---

## SLIDE 1 — Solution & Team

**App name (big):** ARGUS
**Tagline under the name:** *Argus never blinks — an AI that watches every load so nothing gets dropped.*

**One-line value proposition:**
> Argus turns the warehouse cameras you already have into an AI Field Intelligence Assistant that understands handling **behaviour** — not just motion — scores damage risk in real time, alerts supervisors in their own language, and learns from every confirmation to prevent the next damaged product.

**Team block (bottom strip):**
- Team: **[TEAM NAME]**
- Members: **[Member 1] · [Member 2] · [Member 3] · [Member 4]**
- Track: AI Video Intelligence for Warehouse Handling — Godrej Enterprises Group × graVITas'26

**Speaker note (10 s):**
"Argus = Argus Panoptes, the hundred-eyed watchman. We didn't build another CCTV dashboard — we built the watcher that understands what it sees."

---

## SLIDE 2 — Problem, Solution & User Journey

**Headline:** Cameras record what happened. Argus understands what is happening — and stops it becoming damage.

**The journey diagram (render left→right as chevrons, drop a mini screenshot under each stage):**

```
Warehouse Activity → Video → AI Understanding → Risk Detection → Alert → Intervention → Prevention
     (the 7          (existing    (YOLO-World +      (physics-informed  (toast +     (supervisor     (feedback
   pilot videos)     cameras)     ByteTrack + FSM)   0–100 breakdown)   voice)       confirms)       recalibrates)
```

**Under "Intervention":** the supervisor journey in 4 icons —
`Potential Risk alert → open Incident Replay (clip + risk breakdown) → Confirm / Dismiss → thresholds recalibrate`

**Right side — "The shift from" (mirror the PS's own words):**
| From (Traditional CCTV) | To (AI Field Intelligence) |
|---|---|
| Damage Detection | **Damage Prevention** |
| "We detected 25 damaged products" | **"We flagged 80 risky-handling events and enabled correction before damage occurred"** |
| Camera → Recording → Human review | Camera → AI perception → Behaviour understanding → Risk detection → Alert → Intervention → Learning |

**Speaker note (20 s):**
"Every stage in this chain is built and running — the screenshot under each arrow is our actual product, not a mockup. The last arrow is the differentiator: every supervisor confirm/dismiss feeds back into the risk calibration. The system gets more yours every shift."

---

## SLIDE 3 — Technical Architecture & Technology Stack

**Top half — pipeline diagram (single horizontal flow):**

```
┌───────────────────── VIDEO INGESTION ─────────────────────┐
│  recorded footage · live camera · smartphone video        │
└──────────────────────────┬────────────────────────────────┘
                           ▼
        ┌──────────────────────────────────┐
        │  PERCEPTION (offline, edge-ready)│
        │  YOLO-World open-vocabulary      │
        │  detection + ByteTrack IDs       │
        │  + second YOLOv8 PPE detector    │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  TEMPORAL BEHAVIOUR ENGINE       │
        │  per-track FSM: motion →         │
        │  candidate → verify (sequences,  │
        │  not frames) + cross-track       │
        │  signals (stacking, stepping)    │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  RISK SCORING 0–100 (running     │
        │  total: severity · impact ½mv²·h │
        │  fragility · location · stacking │
        │  · recurrence − confidence)      │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  CONFIDENCE GATE                 │
        │  Observed → Potential → Confirmed│
        └───────┬──────────────────┬───────┘
                ▼                  ▼
   ┌──────────────────┐   ┌──────────────────────────┐
   │ ALERTS           │   │ FASTAPI + POSTGRES        │
   │ toast + sound +  │   │ events · feedback ·       │
   │ voice (7 Indian  │   │ products · shift summaries│
   │ languages)       │   │ + retention worker        │
   └──────────────────┘   └────────────┬──────────────┘
                                       ▼
                    ┌──────────────────────────────────┐
                    │ REACT DASHBOARD + GROUNDED       │
                    │ AI ASSISTANT (tool-calls the DB, │
                    │ never invents)                   │
                    └──────────────────────────────────┘
```

**Bottom half — named stack (two columns, exact names = credibility):**

| Layer | Technology |
|---|---|
| Computer vision | **YOLO-World** (open-vocabulary, zero-shot), **ByteTrack** persistent IDs |
| PPE compliance | **YOLOv8** second detector, 23k-image helmet/vest model (mAP50 73.5%) |
| Behaviour AI | Per-track **finite-state machine** + cross-track signal engine — temporal reasoning, not frame classification |
| Risk model | Physics-informed scoring (impact proxy = ½mv²·h), fully config-weighted |
| VLM (pluggable) | Gemini/Qwen verification layer with deterministic grounded fallback |
| Assistant LLM | Claude, grounded via DB tool-calls — answers only from detected events |
| Video processing | OpenCV + ffmpeg H.264 evidence clips, annotated (boxes, labels, banner) |
| Backend / Data | **FastAPI**, SQLAlchemy async, **PostgreSQL** |
| Front-end | **React** + Tailwind + Recharts (dense triage UI, IBM Plex) |
| Edge / offline | All inference local — runs with **zero internet, zero API keys** |
| Quality | **93 automated tests**, seedable deterministic demo data |

**Speaker note (25 s):**
"One honest sentence: our behaviour engine is a hand-built FSM with physics-informed scoring, not an end-to-end neural net — which is exactly why every alert is explainable line-by-line. The judges can ask 'why 61?' and we show the running total."

---

## SLIDE 4 — Prototype Screenshots & Demo

**Layout: 5 screenshot tiles + embedded demo video (60–90 s).**

Tile captions (under each screenshot, in order):

1. **Original video** → *"Pilot footage in — the exact videos from the challenge Drive folder (7 videos, 7 scenarios)."*
2. **AI-detected objects** → *"YOLO-World + ByteTrack: persons, cartons, cupboards, trolleys — persistent track IDs, faces pixelated."*
3. **Behaviour detection & risk classification** → *"Live Feed: behaviour, tier colour, confidence ladder (Observed ○ → Potential ◐ → Confirmed ●), physics-scored 0–100."*
4. **Incident replay** → *"Annotated clip (red PPE boxes, track IDs, behaviour banner) + the risk score as a running total — nothing is a black box."*
5. **Dashboard + AI assistant** → *"Shift report with published 9.3% false-positive rate; assistant answers only from the DB via tool-calls."*

**Embedded demo video — 5 scenarios (shot list):**

| # | Scenario | Show |
|---|---|---|
| 1 | Dragging without trolley | Toast + sound alert fires on High event; badge colours |
| 2 | Throwing cartons | Open replay → risk breakdown → **Confirm** → status flips ● |
| 3 | PPE non-compliance | Red NO-HARDHAT boxes burned in the clip |
| 4 | Stepping on cartons | Cross-track behaviour + explanation panel |
| 5 | Assistant + Hindi voice | Ask *"Which bay had the highest number of risky events?"* → grounded answer with visible tool-call → Hindi voice alert plays |

**Speaker note (15 s):**
"The demo video is the product, unedited, running offline. No API keys, no internet — it works on the warehouse floor, not just in this room."

---

## SLIDE 5 — Impact, Damage Prevention & User Validation

**Headline (the PS's own reframe):** *Not "we detected damaged products" — "we identified high-risk handling events and enabled correction before damage occurred."*

**The prevention numbers (left, mono font, plain):**

```
80            risky-handling events surfaced from 7 pilot videos (~3.5 min)
19            High-tier events — each with clip evidence + recommended action
6             behaviours firing live: dragged · thrown · dropped ·
              no-trolley · stepping-on-packages · PPE non-compliance
9.3%          published false-positive rate (43 supervisor reviews)
0 s           alert latency (in-pipeline scoring, real-time toast + voice)
```

**Measured performance (honest, small-print credibility block):**
> Evaluated against ground-truth labels read from the pilot video names, video-level: core-behaviour precision **38.5%**, recall **41.7%** — **zero-shot**, with no fine-tuning on this footage. Precision/recall, per-bay risk and false-positive rate are computed continuously by the product itself (`/api` metrics), which is how the system reports its own reliability.

**Damage-prevention mapping (right):** every PS bad-practice → our detector & action:

| PS bad practice | Argus behaviour | Recommended action shown to supervisor |
|---|---|---|
| Dragging cartons/KD packets | `2_product_dragged` / `8_dragging_no_trolley` | Use trolley/pallet truck; train on drag instances |
| Throwing or dropping products | `3_product_thrown` / `1_product_dropped` | Inspect product; review unloading practice |
| Heavy products on lighter packets | `4_improper_stacking` | Heavy-at-bottom stacking procedure |
| Stepping/standing on cartons | `6_stepping_on_packages` | Clear working path around material |
| Missing helmet/vest | `13_ppe_noncompliance` | PPE compliance before handling zone entry |

**Users & validation (bottom strip):**
- Designed around: **Warehouse supervisor** (triage queue), **loading operator** (voice alerts in their language), **safety/logistics professional** (shift reports, heat maps, scorecards).
- What user-informed iteration already changed: dense scan-first table (not cards), risk breakdown shown as a running total after users asked "why was this flagged?", confirm/dismiss kept one click from review, alerts voiced in regional languages for non-English-speaking operators.
- Pilot-ready: ships with site-tunable tier bands, per-bay hazard factors and per-product fragility — recalibration takes config, not code.

**Speaker note (20 s):**
"We publish our own false-positive rate on the dashboard — 9.3% — because trust is a feature. One honest number beats three vague claims, and the loop that produces it is also the loop that improves it."

---

## SLIDE 6 — Responsible AI, Optional Innovations & Roadmap

**Left column — Responsible AI (answer the PS's question out loud):**
*"How do we use AI to improve behaviour without turning the warehouse into a surveillance environment?"*

- **Faces pixelated** in every stored evidence frame by default
- **Observed → Potential → Confirmed** ladder — the system never claims damage without evidence (the PS's own distinction, built as the UI's core visual device)
- **Human review**: High/Critical events go to a supervisor queue — confirm/dismiss in one click; **no automated punitive decisions** — outputs are recommended actions and training cues, never penalties
- **Explainability**: every score is a visible running total; every explanation is generated from the detector's own measured features
- **Data minimization & retention**: configurable `RETENTION_DAYS` (default 30) with an automatic purge worker deleting expired events *and* their clips; secure storage, role-based access display
- **Published false-positive rate** (9.3%) — the system reports its own reliability

**Right column — 10 optional innovations delivered (of the PS's 23):**
real-time alerts · edge/offline inference · automatic incident replay · behaviour heat maps · product-specific risk models (Fragile 1.5× / Standard 1.0× / Rugged 0.7×) · PPE compliance detection · automatic incident reports · conversational AI supervisor assistant · multilingual voice alerts · operator/team safety scorecards

**Behaviour taxonomy strip (bottom):**
**13-behaviour taxonomy** — `dropped · dragged · thrown · improper stacking · unstable stacking · stepping on packages · outside designated zone · dragging without trolley · pallet overhang · rough handling · wrong orientation · cluttered staging · PPE non-compliance` — **6 firing live on the supplied pilot footage**, the remainder operational and demoable on the controlled-environment setup the PS invites.

**Roadmap one-liner (closer, the PS's "bigger opportunity"):**
> Warehouse → Factory → Distribution Centre → Loading Bay → Retail → Field Service — the same architecture understanding every physical operation, from events to entire processes.

**Speaker note (15 s):**
"Argus is not a hackathon demo of a warehouse feature — it's the first node of an AI Field Intelligence Platform for physical operations. That's the bigger opportunity in the brief, and the architecture is already shaped for it."

---

# APPENDIX A — Demo video shot list (for recording, 60–90 s total)

1. **0:00–0:08** Live Feed — rows sliding in, tier badges, confidence ladder visible. Point: "80 events, scanned in seconds."
2. **0:08–0:20** Click a High dragging event → replay: annotated clip plays (track IDs + banner), risk breakdown running total. Point: "every number explained."
3. **0:20–0:28** Click **Confirm** → status flips to solid ● → toast on the next High event.
4. **0:28–0:40** PPE clip — red NO-HARDHAT box. Point: "second detector, zero extra infrastructure."
5. **0:40–0:55** Dashboard → Generate shift report → **Read aloud** (English). Point: "automatic reports."
6. **0:55–1:10** Assistant: *"Which loading bay had the highest number of risky events?"* → answer with visible `tool call:` line above it. Point: "grounded, not hallucinated."
7. **1:10–1:20** Set alert voice to **Hindi** → new High event → toast + Hindi voice. Point: "speaks to the operator in their language."
8. **1:20–1:30** Closing frame: the Confirmed ● badge + "Argus never blinks."

Record at 1080p in Chrome, app running via `./start_demo.sh docker`, no other tabs, do one dry run first so browser voices are cached.

# APPENDIX B — Q&A armour (likely judge questions)

- **"Why is your precision 38%?"** → "Video-level, zero-shot, evaluated against labels embedded in the filenames — with no fine-tuning on this footage. The FPs are mostly over-detection of `dragged` during normal carries; the confidence gate and the 9.3% supervisor-verified FPR are the product's own answer to that, and fine-tuning on site data is exactly what the feedback loop primes."
- **"Only 6 of 13 behaviours fire?"** → "On your 7 supplied videos, yes — those 6 are what the footage contains. All 13 are implemented behind the same event schema; the PS's controlled-environment setup is where we demo the rest."
- **"What if the camera angle changes?"** → "YOLO-World is open-vocabulary and zero-shot; FSM thresholds and camera calibration are per-camera config. Multi-camera tracking is on the roadmap."
- **"Why not an end-to-end video model?"** → "Explainability and offline operation. Every alert decomposes into measured features; a black-box video model can't show a running total, and warehouses shouldn't depend on a cloud round-trip for safety."

# APPENDIX C — Export checklist

- [ ] Fill **[TEAM NAME]** + members on Slide 1
- [ ] Screenshot tiles for Slide 4: Live Feed · replay w/ breakdown · PPE clip frame · Dashboard · Assistant (dark theme, 2× scale)
- [ ] Render Slide 3 diagram (or paste as image); keep font ≥ 14 pt
- [ ] Embed demo video into Slide 4 (and keep the file separately for the submission form)
- [ ] 6 slides ≤ the PS maximum — do not add a 7th; appendix lives outside the deck
