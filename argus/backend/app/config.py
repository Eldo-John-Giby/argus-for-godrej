"""Central configuration for the Argus backend."""

import os
from pathlib import Path

# --- .env loading (no python-dotenv dependency; real env vars win) ---

def _load_env_file() -> None:
    """Load KEY=VALUE pairs from backend/.env into os.environ (once).

    Existing environment variables take precedence, so docker-compose / shell
    exports always override the file.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_env_file()

# --- Database ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./argus.db",
)

# --- Storage ---
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "./storage"))
CLIPS_DIR = STORAGE_ROOT / "clips"
KEYFRAMES_DIR = STORAGE_ROOT / "keyframes"


def ensure_storage_dirs():
    """Create storage directories on demand (not at import time)."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)

# --- Perception ---
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")  # Placeholder for YOLO26
DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv("DET_CONF", "0.35"))
TRACKING_IOU_THRESHOLD = float(os.getenv("TRACK_IOU", "0.7"))

# --- VLM ---
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "qwen")  # "qwen" | "gemini"
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Gemini model id for REST calls (Generative Language API v1beta)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
VLM_TIMEOUT_SECONDS = int(os.getenv("VLM_TIMEOUT", "30"))

# --- Assistant ---
ASSISTANT_LLM_PROVIDER = os.getenv("ASSISTANT_LLM", "claude")  # "claude" | "openai"
ASSISTANT_API_KEY = os.getenv("ASSISTANT_API_KEY", "")
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", "claude-sonnet-4-20250514")

# --- Alerts: translation + TTS ---
# Optional: OpenAI-compatible chat endpoint used to translate alert text.
# Falls back to GEMINI_API_KEY, then to the English base text.
ALERT_TRANSLATION_URL = os.getenv("ALERT_TRANSLATION_URL", "")
ALERT_TRANSLATION_API_KEY = os.getenv("ALERT_TRANSLATION_API_KEY", "")
ALERT_TRANSLATION_MODEL = os.getenv("ALERT_TRANSLATION_MODEL", "gpt-4o-mini")
# "gtts" enables server-side speech synthesis; anything else = browser TTS only.
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "none")
TTS_TIMEOUT_SECONDS = int(os.getenv("TTS_TIMEOUT", "15"))
TTS_VOICE_LANG = os.getenv("TTS_VOICE_LANG", "en")

# --- Risk Scoring Weights ---
RISK_WEIGHTS = {
    "w1_behaviour_severity": float(os.getenv("W1", "0.35")),
    "w2_impact_proxy": float(os.getenv("W2", "0.25")),
    "w3_fragility": float(os.getenv("W3", "0.10")),
    "w4_stack_instability": float(os.getenv("W4", "0.10")),
    "w5_recurrence": float(os.getenv("W5", "0.12")),
    "w6_confidence_penalty": float(os.getenv("W6", "0.08")),
    "w7_location": float(os.getenv("W7", "0.10")),
}

# Location factor (PS names location as a risk-score input): per-bay hazard
# multipliers, config-tunable per deployment. Applied to the fragility +
# location component of the score — a fragile product mishandled in a
# high-hazard bay scores higher than the same event in a baseline bay.
BAY_HAZARD_FACTORS = {
    "default": float(os.getenv("BAY_HAZARD_DEFAULT", "1.0")),
    "A": float(os.getenv("BAY_HAZARD_A", "1.0")),   # Unloading dock — baseline
    "B": float(os.getenv("BAY_HAZARD_B", "1.15")),  # Staging/stacking zone
    "C": float(os.getenv("BAY_HAZARD_C", "1.25")),  # KD assembly/loading — mixed traffic
}


def bay_hazard_factor(bay_id: str | None) -> float:
    """Hazard multiplier for a bay; unknown/absent bays use the default."""
    if not bay_id:
        return BAY_HAZARD_FACTORS["default"]
    return BAY_HAZARD_FACTORS.get(str(bay_id).strip(), BAY_HAZARD_FACTORS["default"])

# --- Confidence Gate Thresholds ---
CONFIDENCE_LOW = 0.5
CONFIDENCE_MEDIUM = 0.8

# --- Risk Tier Thresholds ---
# Per-site calibration (the plan calls for site-tunable bands): this pilot
# footage's operational score range is ~26-44, so bands are set relative to
# it. Scores themselves are never adjusted — only the display bands move.
RISK_TIER_LOW = 20
RISK_TIER_MEDIUM = 35
RISK_TIER_HIGH = 50

# --- Responsible AI ---
FACE_BLUR_ENABLED = os.getenv("FACE_BLUR", "true").lower() == "true"
# Retention policy: events (and their stored clips) older than this are
# purged by the background retention worker. 0 disables purging.
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "30"))
# How often the retention worker runs (hours).
RETENTION_SWEEP_HOURS = float(os.getenv("RETENTION_SWEEP_HOURS", "24"))
