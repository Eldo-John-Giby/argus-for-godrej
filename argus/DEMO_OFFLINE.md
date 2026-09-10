# ARGUS — Offline Demo Runbook (presenter)

Everything the demo needs lives **inside the `argus/` folder**: code, SQLite
seeding data (`ml/results`), replay evidence clips (`clips/`), and the YOLO
weights (`backend/*.pt`, `ml/yolov8s-world.pt`, `ml/best.pt`). No Postgres, no
Docker, no API keys, no internet required on demo day.

---

## 1. What works offline (and what doesn't)

| Feature | Offline behaviour |
|---|---|
| Live Feed, Replay, Dashboard, Heatmap | ✅ fully working (served from local API + SQLite) |
| Incident replay clips | ✅ played from local `clips/` evidence |
| Shift report (JSON/text/download) | ✅ computed locally |
| Risk scoring + confidence ladder + feedback | ✅ computed locally |
| Grounded assistant | ✅ answers from the events DB via its **fallback LLM** — no API key |
| Voice alerts | ✅ **browser TTS** (Chrome/Edge built-in, incl. Hindi) — server TTS not needed |
| Translated alert text | ⚠️ needs the translation endpoint; without it the English text is used |
| VLM verification (live pipeline) | ⚠️ degrades gracefully — flagged events keep FSM explanations |
| Live video inference | ⚠️ only if `requirements-ml.txt` was pre-installed (see step 2B) |

## 2. One-time prep — on a machine WITH internet

Do this at home / in the hotel, never at the venue.

**2A. Get the code** (either way works):

```bash
git clone https://github.com/Eldo-John-Giby/argus-for-godrej.git
cd argus-for-godrej/argus
# — or — copy the whole argus/ folder to a USB stick / the demo laptop
```

**2B. Warm the demo (installs deps + seeds the DB, one command):**

```bash
python run_demo.py
```

First run takes 2–3 minutes (~100 MB of pip/npm packages). Verify the sanity
checks all print `OK`, click through the 60-second demo path, then:

```bash
python run_demo.py stop
```

**Optional — if you plan to run LIVE video inference at the booth**
(otherwise skip; the seeded demo doesn't need it):

```bash
cd backend && pip install -r requirements-ml.txt && cd ..
```

**Optional — true offline installs** (venue Wi-Fi is usually firewalled; if you
want zero network dependence even on a *fresh* machine, pre-download the
packages too):

```bash
cd backend && pip download -r requirements.txt -d wheels/ && cd ..
cd frontend && npm ci --pack-destination ../npm-cache && cd ..
```

Then on the demo machine:
`pip install --no-index --find-links wheels/ -r backend/requirements.txt`
and `npm install --offline --cache ../npm-cache` (from `frontend/`).

## 3. Demo-day steps (no internet needed)

1. Copy the warmed `argus/` folder to the demo laptop (or `git clone` — the
   clone already contains clips + results + weights, so seeding works as-is).
2. Requirements on that machine: **Python 3.10+** and **Node.js 18+**. Nothing else.
3. Pre-flight:

   ```bash
   cd argus
   python run_demo.py doctor
   ```

   All green → go. If it flags anything, fix before the audience arrives.
4. Start:

   ```bash
   python run_demo.py            # Windows: double-click start_demo.bat
   ```

   Sanity checks (events count, shift report, alert languages, grounded
   assistant) run automatically and print `OK`.
5. Frontend: **http://localhost:5173** · API docs: **http://localhost:8000/docs**
6. **Airplane-mode test** (do this once, before doors open): turn off Wi-Fi,
   restart the stack (`run_demo.py stop` → `run_demo.py`), and click through the
   demo path below. If it works now, it works on stage.
7. After the pitch: `python run_demo.py stop`.

### The 60-second demo path

1. **Live Feed** → click a High-tier row → replay + risk breakdown opens
2. **Confirm** it → status flips on the confidence ladder
3. **Dashboard** → Generate shift report → Read aloud (browser TTS)
4. **Assistant** → ask: *"Which loading bay had the highest number of risky events?"*
5. Flip **Alert voice to Hindi** in Live Feed → Confirm a High event in another
   tab → toast + voice fires

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| Ports 8000/5173 busy | `python run_demo.py stop` (kills strays), then start again |
| Database looks wrong / empty | `python run_demo.py seed` (wipes + reseeds from `ml/results`) |
| Replay clips 404 | `clips/` folder missing — recopy it; the app still boots without it |
| Assistant answers look generic | It fell back to the offline DB brain — that's expected and still grounded; show the tool-call trace |
| No alert voice | Use Chrome/Edge (best browser-TTS support); check the voice isn't muted in OS settings |
| `npm not found` / `python not found` | Install Node 18+ / Python 3.10+ — or use the pre-warmed venv copy (ship the whole folder incl. `backend/venv` and `frontend/node_modules` on the USB stick) |

**Presenter's belt-and-braces:** carry the entire `argus/` folder on a USB
stick *with* `backend/venv/` and `frontend/node_modules/` included (skip
`videos/` and `weights/clip/` — they're not used). Then even a machine with no
Python/Node downloads available can run the demo via the existing venv.
