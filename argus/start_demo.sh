#!/usr/bin/env bash
# ARGUS demo launcher — one command, one terminal.
#
#   ./start_demo.sh          # start backend + frontend (self-bootstrapping)
#   ./start_demo.sh docker   # full container path (requires Docker Desktop running)
#   ./start_demo.sh stop     # stop any running demo processes
#   ./start_demo.sh seed     # reseed the database from ml/results, then exit
#   ./start_demo.sh doctor   # pre-flight checks only
#
# Everything is delegated to run_demo.py, which is self-bootstrapping: it
# creates the backend venv, installs pip/npm dependencies, seeds the database
# when empty, starts the stack and runs the sanity checks. No Postgres and no
# API keys are needed — SQLite is the default database.

set -u
cd "$(dirname "$0")"

MODE="${1:-start}"

if [ "$MODE" = "docker" ]; then
    docker info >/dev/null 2>&1 || { echo "!! Docker engine not running — start Docker Desktop first"; exit 1; }
    echo "==> Starting container stack (db + backend + frontend)"
    docker compose up -d --build
    echo "==> Waiting for backend health"
    for i in $(seq 1 30); do
        curl -sf http://localhost:8000/health >/dev/null 2>&1 && break
        sleep 2
    done
    echo "==> Seeding compose Postgres from host (needs backend deps installed)"
    (cd backend && DATABASE_URL="postgresql+asyncpg://argus:argus@localhost:5432/argus" python seed_db.py) || \
        echo "!! seeding skipped — run: cd backend && pip install -r requirements.txt && python seed_db.py with DATABASE_URL set"
    echo ""
    echo "  ARGUS is ready:  http://localhost:5173"
    echo "  (logs: docker compose logs -f backend)"
    exit 0
fi

# start / stop / seed / doctor — all handled by the canonical launcher
exec python3 run_demo.py "$MODE"
