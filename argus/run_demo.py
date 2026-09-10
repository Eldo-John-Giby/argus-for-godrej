#!/usr/bin/env python3
"""ARGUS demo launcher — the one command that makes everything work.

    python run_demo.py            # start backend + frontend (installs deps, seeds if needed)
    python run_demo.py --detach   # same, but returns immediately (stack stays up)
    python run_demo.py stop       # stop any running demo processes
    python run_demo.py seed       # (re)seed the database from ml/results, then exit
    python run_demo.py doctor     # pre-flight checks only — no servers started
    python run_demo.py --port 8000 --frontend-port 5173   # custom ports

Safe to run repeatedly: detects existing installs, reuses the venv, and skips
seeding when the database already has events. No Postgres, no Docker, no API
keys needed — SQLite is the default database and every external LLM call has a
grounded fallback.

Cross-platform: works on Windows, macOS and Linux.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent          # argus/
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV_DIR = BACKEND / "venv"
BACKEND_PORT = 8000
FRONTEND_PORT = 5173
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"

IS_WINDOWS = os.name == "nt"

# ---------------------------------------------------------------- utilities


def info(msg: str) -> None:
    print(f"==> {msg}")


def ok(msg: str) -> None:
    print(f"    OK {msg}")


def die(msg: str) -> None:
    print(f"!! {msg}")
    sys.exit(1)


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for(url: str, tries: int = 30, delay: float = 1.0) -> bool:
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=3):
                return True
        except urllib.error.HTTPError:
            return True  # reachable, even on 4xx/5xx
        except OSError:
            time.sleep(delay)
    return False


def find_python() -> list[str]:
    """Return an argv prefix for a Python 3.10+ interpreter (system or venv).

    A list (not a string) so paths with spaces survive, e.g.
    'C:\\Program Files\\Python310\\python.exe'.
    """
    candidates: list[list[str]] = [
        [sys.executable],
        ["python"],
        ["python3"],
        ["py", "-3"],
    ]
    for cand in candidates:
        try:
            out = subprocess.run(
                cand + ["--version"], capture_output=True, text=True, timeout=15
            )
            ver = (out.stdout or out.stderr).strip()  # e.g. 'Python 3.10.7'
            parts = ver.split()
            if len(parts) == 2 and parts[1][0].isdigit():
                major_minor = parts[1].split(".")
                if int(major_minor[0]) >= 3:
                    return cand
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            continue
    die("Python 3.10+ not found — install it from https://python.org")
    return [sys.executable]


def venv_python() -> Optional[str]:
    """Path to the venv python if the venv exists."""
    if IS_WINDOWS:
        cand = VENV_DIR / "Scripts" / "python.exe"
    else:
        cand = VENV_DIR / "bin" / "python"
    return str(cand) if cand.exists() else None


def run(cmd: list[str], cwd: Path, title: str, quiet: bool = False) -> None:
    info(f"{title}…")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as e:
        die(f"{title}: could not run {cmd[0]!r} — {e}")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        print("\n".join(tail))
        die(f"{title} failed")
    if not quiet:
        last = (proc.stdout or "").strip().splitlines()
        if last:
            print(f"    {last[-1]}")


# ------------------------------------------------------------- environment


def ensure_python_deps() -> list[str]:
    """Return an argv prefix for a python with all backend deps installed.

    Creates backend/venv on first run (or reuses it), installs
    requirements.txt only when imports are missing.
    """
    # 1) Existing venv?
    vp = venv_python()
    if vp is not None:
        r = subprocess.run(
            [vp, "-c", "import fastapi, uvicorn, aiosqlite, sqlalchemy"],
            capture_output=True,
        )
        if r.returncode == 0:
            ok("backend venv ready")
            return [vp]
        info("backend venv incomplete — installing requirements…")
        run([vp, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            cwd=BACKEND, title="pip install (backend venv)")
        return [vp]

    # 2) System python already has everything?
    base = find_python()
    r = subprocess.run(
        base + ["-c", "import fastapi, uvicorn, aiosqlite, sqlalchemy"],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        ok("system python has all backend deps")
        return base

    # 3) Create a venv and install.
    info("creating backend venv (first run — one-time, ~1 min)…")
    run(base + ["-m", "venv", "venv"], cwd=BACKEND, title="python -m venv")
    vp = venv_python()
    if vp is None:
        die("venv creation failed")
    run([vp, "-m", "pip", "install", "-q", "--upgrade", "pip"],
        cwd=BACKEND, title="pip upgrade", quiet=True)
    run([vp, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=BACKEND, title="pip install (backend deps)")
    return [vp]


def ensure_node() -> str:
    """Resolve npm to a Win32-executable path (npm.cmd) on Windows."""
    if IS_WINDOWS:
        # shutil.which("npm") can resolve to the bash shim (no extension),
        # which Popen cannot execute — require a real .cmd/.exe.
        for cand in ("npm.cmd", "npm"):
            p = shutil.which(cand)
            if p and p.lower().endswith((".cmd", ".exe", ".bat")):
                return p
    else:
        p = shutil.which("npm")
        if p:
            return p
    die("npm not found — install Node.js 18+ from https://nodejs.org")
    return ""


def ensure_frontend_deps(npm: str) -> None:
    if (FRONTEND / "node_modules" / ".package-lock.json").exists():
        ok("frontend node_modules present")
        return
    info("installing frontend dependencies (first run — one-time, ~2 min)…")
    run([npm, "install", "--no-audit", "--no-fund"], cwd=FRONTEND,
        title="npm install", quiet=True)


# ------------------------------------------------------------------ data


_COUNT_EVENTS = """
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def main():
    engine = create_async_engine("sqlite+aiosqlite:///./argus.db")
    try:
        async with engine.connect() as conn:
            n = (await conn.execute(text("select count(*) from events"))).scalar()
            print(n if n is not None else 0)
    except Exception:
        print(0)
    finally:
        await engine.dispose()

asyncio.run(main())
"""


def db_has_events(python: list[str]) -> bool:
    r = subprocess.run(
        python + ["-c", _COUNT_EVENTS],
        cwd=BACKEND, capture_output=True, text=True,
    )
    if r.returncode != 0:
        return False
    try:
        return int(r.stdout.strip().splitlines()[-1]) > 0
    except (ValueError, IndexError):
        return False


def seed(python: list[str]) -> None:
    if db_has_events(python):
        ok("database already seeded")
        return
    info("seeding database from ml/results…")
    run(python + ["seed_db.py"], cwd=BACKEND, title="python seed_db.py")


# ---------------------------------------------------------------- servers


def spawn(cmd: list[str], cwd: Path, log_path: Path, name: str) -> subprocess.Popen:
    log = open(log_path, "w", encoding="utf-8")
    info(f"starting {name} (log: {log_path})")
    try:
        return subprocess.Popen(
            cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0),
        )
    except FileNotFoundError as e:
        die(f"starting {name}: {e}")
        raise


def stop_stale(port: int) -> None:
    """Kill whatever is on our ports so a fresh start always works."""
    if not port_in_use(port):
        return
    info(f"port {port} busy — stopping the previous process")
    if IS_WINDOWS:
        for pid in _pids_on(port):
            subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True)
    else:
        killed = subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
        if killed.returncode != 0:
            r = subprocess.run(["lsof", "-t", f"-i:{port}", "-sTCP:LISTEN"],
                               capture_output=True, text=True)
            for pid in r.stdout.split():
                subprocess.run(["kill", "-9", pid], capture_output=True)
    deadline = time.time() + 10
    while port_in_use(port) and time.time() < deadline:
        time.sleep(0.5)


def _pids_on(port: int) -> list[str]:
    """All PIDs listening on the port (IPv4 + IPv6 rows can differ)."""
    out = subprocess.run(
        ["netstat", "-ano"], capture_output=True, text=True
    ).stdout
    pids: list[str] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and f":{port}" in parts[1] and parts[3] == "LISTENING":
            if parts[4].isdigit() and parts[4] not in pids:
                pids.append(parts[4])
    return pids


def write_pidfile(popen: subprocess.Popen, path: Path) -> None:
    path.write_text(str(popen.pid))


# ------------------------------------------------------------ sanity checks


def sanity_checks() -> None:
    info("sanity checks (what the judges will ask about)…")

    def get(url: str) -> dict:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)

    events = get(f"{BACKEND_URL}/api/events/?limit=500")
    tiers: dict[str, int] = {}
    for e in events:
        tiers[e["tier"]] = tiers.get(e["tier"], 0) + 1
    ppe = sum(1 for e in events if e["behaviour_type"] == "13_ppe_noncompliance")
    print(f"    events: {len(events)} | tiers: {tiers} | PPE: {ppe}")
    assert len(events) > 0, "no events in DB — run: python run_demo.py seed"

    rep = get(f"{BACKEND_URL}/api/reports/shift?hours=8")
    print(f"    shift report: {rep['total_events']} events, top: {rep['top_behaviour']}")

    langs = get(f"{BACKEND_URL}/api/alerts/languages")
    print(f"    alert languages: {len(langs)}")

    req = urllib.request.Request(
        f"{BACKEND_URL}/api/assistant/chat",
        data=json.dumps({"query": "Which loading bay had the highest number of risky events?"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        chat = json.load(r)
    print(f"    assistant grounded: {chat['grounded']} | tools: {[t['tool'] for t in chat['tool_calls']]}")
    print("    OK all sanity checks passed")


def doctor() -> None:
    info("pre-flight checks")
    find_python()
    ok(f"python: {sys.version.split()[0]}")
    ensure_node()
    ok("npm available")
    ensure_python_deps()
    ensure_frontend_deps(ensure_node())
    if (ROOT / "ml" / "results").exists():
        ok("ml/results present (seeding source)")
    else:
        die("ml/results missing — needed to seed the demo data")
    if (ROOT / "clips").exists():
        ok("clips present (replay evidence)")
    else:
        print("    !! clips/ missing — replay evidence will 404 (demo still boots)")
    if db_has_events(ensure_python_deps()):
        ok("database seeded")
    else:
        print("    database empty — will seed on next start")
    print("\nAll checks passed. Start with: python run_demo.py")


# -------------------------------------------------------------------- main


def do_stop() -> None:
    info("stopping demo processes…")
    # 1) Kill anything recorded in pidfiles (covers detached runs)
    for pf in (BACKEND / "demo.pid", FRONTEND / "demo.pid"):
        if pf.exists():
            pid = pf.read_text().strip()
            if pid.isdigit():
                if IS_WINDOWS:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True)
                else:
                    subprocess.run(["kill", "-9", pid], capture_output=True)
            pf.unlink(missing_ok=True)
    # 2) Free the ports regardless (covers attached runs and strays)
    for port, label in ((BACKEND_PORT, "backend"), (FRONTEND_PORT, "frontend")):
        if port_in_use(port):
            if IS_WINDOWS:
                for pid in _pids_on(port):
                    subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True)
            else:
                subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
            ok(f"stopped {label} (port {port})")
        else:
            ok(f"{label} not running")
    print("Stopped.")


def main() -> None:
    global BACKEND_PORT, FRONTEND_PORT, BACKEND_URL
    args = sys.argv[1:]

    if "--port" in args:
        BACKEND_PORT = int(args[args.index("--port") + 1])
        BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
    if "--frontend-port" in args:
        FRONTEND_PORT = int(args[args.index("--frontend-port") + 1])

    if args and args[0] == "stop":
        do_stop()
        return
    if args and args[0] == "seed":
        vp = ensure_python_deps()
        seed(vp)
        print("Seeded. Start with: python run_demo.py")
        return
    if args and args[0] == "doctor":
        doctor()
        return
    if args and args[0] in ("help", "--help", "-h"):
        print(__doc__)
        return

    # ---------- start ----------
    info(f"ARGUS demo setup (root: {ROOT})")
    python = ensure_python_deps()
    npm = ensure_node()
    ensure_frontend_deps(npm)
    seed(python)

    # Fresh start: clear anything already on our ports
    stop_stale(BACKEND_PORT)
    stop_stale(FRONTEND_PORT)

    backend = spawn(
        python + ["-m", "uvicorn", "main:app", "--port", str(BACKEND_PORT)],
        cwd=BACKEND, log_path=BACKEND / "demo_backend.log", name="backend",
    )
    write_pidfile(backend, BACKEND / "demo.pid")

    vite_env = {**os.environ, "VITE_API_URL": f"http://localhost:{BACKEND_PORT}"}
    log = open(FRONTEND / "demo_frontend.log", "w", encoding="utf-8")
    info("starting frontend")
    try:
        frontend = subprocess.Popen(
            [npm, "run", "dev", "--", "--port", str(FRONTEND_PORT)],
            cwd=str(FRONTEND), env=vite_env,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0),
        )
    except FileNotFoundError as e:
        die(f"starting frontend: {e}")

    if not wait_for(f"{BACKEND_URL}/health"):
        print((BACKEND / "demo_backend.log").read_text(encoding="utf-8", errors="replace")[-2000:])
        die(f"backend did not become healthy — see {BACKEND / 'demo_backend.log'}")

    sanity_checks()

    detach = "--detach" in args
    if detach:
        (FRONTEND / "demo.pid").write_text(str(frontend.pid))
        print()
        print("  Stack is running in the background.")
        print("  Stop it with:  python run_demo.py stop")
        return

    print()
    print("=" * 60)
    print(f"  ARGUS is ready:  http://localhost:{FRONTEND_PORT}")
    print(f"  API docs:        {BACKEND_URL}/docs")
    print("=" * 60)
    print()
    print("  60-second demo path:")
    print("   1. Live Feed -> click a High-tier row (replay + breakdown)")
    print("   2. Confirm it -> status flips on the confidence ladder")
    print("   3. Dashboard -> Generate shift report -> Read aloud")
    print("   4. Assistant -> 'What were the three most common risky behaviours during the morning shift?'")
    print("   5. Set Alert voice to Hindi in Live Feed -> new High event fires toast + voice")
    print()
    print(f"  stop everything:  python run_demo.py stop")
    print(f"  backend log:      {BACKEND / 'demo_backend.log'}")
    print(f"  frontend log:     {FRONTEND / 'demo_frontend.log'}")

    try:
        while backend.poll() is None and frontend.poll() is None:
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        info("shutting down…")
        for proc in (frontend, backend):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("Stopped.")


if __name__ == "__main__":
    main()
