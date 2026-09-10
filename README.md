# argus-for-godrej

**ARGUS — AI Field Intelligence for Warehouse Handling**
Godrej Enterprises Group × graVITas'2026 AI Hackathon (Track 02)

The full product, demo launcher, and offline-presenter runbook live in **[`argus/`](argus/)**:

- [`argus/README.md`](argus/README.md) — architecture, endpoints, evaluation numbers, dev setup
- [`argus/DEMO_OFFLINE.md`](argus/DEMO_OFFLINE.md) — **presenter runbook: offline demo steps**
- `python run_demo.py` — one command: installs deps, seeds SQLite from `ml/results`, starts backend (8000) + frontend (5173), runs sanity checks

Quick start:

```bash
cd argus
python run_demo.py
```
