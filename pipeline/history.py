"""Session history persistence."""

import json
from datetime import datetime
from pathlib import Path

import config

HISTORY_FILE = config.DATA_DIR / "history.json"
MAX_ENTRIES = 50


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history: list[dict]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def add_entry(
    audio_filename: str,
    duration_sec: float,
    transcript: str,
    summary: str,
    emotion: dict,
    segments: list[dict],
) -> dict:
    entry = {
        "id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "audio_file": audio_filename,
        "duration_sec": duration_sec,
        "transcript": transcript,
        "summary": summary,
        "emotion_label": emotion.get("label", ""),
        "emotion_confidence": emotion.get("confidence", 0.0),
        "segments": segments,
    }
    history = load_history()
    history.insert(0, entry)
    save_history(history[:MAX_ENTRIES])
    return entry


def get_entry(entry_id: str) -> dict | None:
    for entry in load_history():
        if entry["id"] == entry_id:
            return entry
    return None


def clear_history() -> None:
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
