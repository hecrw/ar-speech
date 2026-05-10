"""Central configuration for the Arabic Audio Understanding system."""

import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = PROJECT_ROOT / "models"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()


def _local_or_hf(local_name: str, hf_id: str) -> str:
    local = MODELS_DIR / local_name
    if local.exists() and (local / "config.json").exists():
        return str(local)
    return hf_id


WHISPER_MODEL_ID = _local_or_hf(
    "whisper-turbo-arabic",
    "openai/whisper-large-v3-turbo",
)
WAV2VEC2_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

OLLAMA_URL = "http://localhost:11434"
OLLAMA_SUMMARIZER_MODEL = "aya-expanse:8b"
SUMMARIZER_MODEL_ID = OLLAMA_SUMMARIZER_MODEL

SUMMARIZER_FALLBACK_ID = _local_or_hf(
    "mt5-arabic-summarization",
    "csebuetnlp/mT5_multilingual_XLSum",
)

EMOTION_MODEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"

EMBEDDING_MODEL_ID = _local_or_hf(
    "nomic-embed-arabic-search",
    "nomic-ai/nomic-embed-text-v2-moe",
)

SAMPLE_RATE = 16_000
MAX_AUDIO_LENGTH_SEC = 600

CHUNK_DURATION_SEC = 30.0
CHUNK_OVERLAP_SEC = 2.0

MAX_INPUT_LENGTH = 512
MAX_SUMMARY_LENGTH = 128
MIN_SUMMARY_LENGTH = 30

EMBEDDING_DIM = 768
TOP_K_DEFAULT = 5
