"""Central configuration for the Arabic Audio Understanding system."""

import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Device ─────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()

# ── Local-or-HF resolution ─────────────────────────────────────────────
# If a fine-tuned model exists under MODELS_DIR/<name>/, use it.
# Otherwise fall back to the HuggingFace Hub ID so the app still runs
# before the user has downloaded their fine-tuned weights.

def _local_or_hf(local_name: str, hf_id: str) -> str:
    local = MODELS_DIR / local_name
    if local.exists() and (local / "config.json").exists():
        return str(local)
    return hf_id

# ── ASR Models ─────────────────────────────────────────────────────────
WHISPER_MODEL_ID = _local_or_hf(
    "whisper-turbo-arabic",
    "openai/whisper-large-v3-turbo",
)
WAV2VEC2_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

# ── Summarization (Ollama LLM) ─────────────────────────────────────────
# Ollama-served LLM for high-quality Arabic summarization. Run locally:
#   ollama pull aya-expanse:8b   (or qwen2.5:7b, llama3.1:8b)
#   ollama serve
OLLAMA_URL = "http://localhost:11434"
OLLAMA_SUMMARIZER_MODEL = "aya-expanse:8b"
SUMMARIZER_MODEL_ID = OLLAMA_SUMMARIZER_MODEL  # for display in System Info

# Local seq2seq fallback if Ollama is unreachable.
SUMMARIZER_FALLBACK_ID = _local_or_hf(
    "mt5-arabic-summarization",
    "csebuetnlp/mT5_multilingual_XLSum",
)

# ── Emotion (Arabic sentiment over the transcript) ─────────────────────
EMOTION_MODEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"

# ── Embedding Models ───────────────────────────────────────────────────
EMBEDDING_MODEL_ID = _local_or_hf(
    "nomic-embed-arabic-search",
    "nomic-ai/nomic-embed-text-v2-moe",
)

# ── Audio Settings ─────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
MAX_AUDIO_LENGTH_SEC = 600  # 10 minutes

# ── Chunking ───────────────────────────────────────────────────────────
CHUNK_DURATION_SEC = 30.0
CHUNK_OVERLAP_SEC = 2.0

# ── Summarization ──────────────────────────────────────────────────────
MAX_INPUT_LENGTH = 512        # mT5 input cap
MAX_SUMMARY_LENGTH = 128      # mT5 output cap
MIN_SUMMARY_LENGTH = 30

# ── Search ─────────────────────────────────────────────────────────────
EMBEDDING_DIM = 768  # nomic-embed-text-v2-moe outputs 768-dim
TOP_K_DEFAULT = 5
