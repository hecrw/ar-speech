"""Audio loading, resampling, chunking, and validation utilities."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from config import SAMPLE_RATE, MAX_AUDIO_LENGTH_SEC, CHUNK_DURATION_SEC, CHUNK_OVERLAP_SEC

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".mp4"}


def load_audio(file_path: str | Path, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to target sample rate.

    Returns (waveform, sample_rate) where waveform is a 1-D float32 array.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    waveform, sr = librosa.load(str(file_path), sr=target_sr, mono=True)

    if len(waveform) == 0:
        raise ValueError(f"Audio file is empty: {file_path}")

    return waveform.astype(np.float32), sr


def chunk_audio(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    chunk_duration_sec: float = CHUNK_DURATION_SEC,
    overlap_sec: float = CHUNK_OVERLAP_SEC,
) -> list[dict]:
    """Split audio into overlapping chunks for processing.

    Each chunk dict contains: waveform, start_sec, end_sec.
    """
    total_samples = len(waveform)
    chunk_samples = int(chunk_duration_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    step_samples = chunk_samples - overlap_samples

    if total_samples <= chunk_samples:
        return [{"waveform": waveform, "start_sec": 0.0, "end_sec": total_samples / sr}]

    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[start:end]
        chunks.append({
            "waveform": chunk,
            "start_sec": start / sr,
            "end_sec": end / sr,
        })
        if end >= total_samples:
            break
        start += step_samples

    return chunks


def get_audio_duration(file_path: str | Path) -> float:
    """Return audio duration in seconds."""
    return librosa.get_duration(path=str(file_path))


def validate_audio(file_path: str | Path) -> tuple[bool, str]:
    """Validate that an audio file is suitable for processing.

    Returns (is_valid, message).
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False, f"File not found: {file_path}"

    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {file_path.suffix}. Supported: {SUPPORTED_FORMATS}"

    try:
        duration = get_audio_duration(file_path)
    except Exception as e:
        return False, f"Cannot read audio file: {e}"

    if duration <= 0:
        return False, "Audio file is empty (0 seconds)"

    if duration > MAX_AUDIO_LENGTH_SEC:
        return False, f"Audio too long: {duration:.1f}s (max: {MAX_AUDIO_LENGTH_SEC}s)"

    return True, f"Valid audio: {duration:.1f}s"
