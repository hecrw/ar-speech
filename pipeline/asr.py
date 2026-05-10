"""Arabic Automatic Speech Recognition (ASR) pipeline.

Supports two backends:
  - OpenAI Whisper (large-v3): primary, best Arabic WER
  - Wav2Vec2 (XLSR-53 Arabic): secondary, for comparison
"""

import logging
from pathlib import Path

import numpy as np
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

import config
from utils.audio import load_audio, chunk_audio, get_audio_duration

logger = logging.getLogger(__name__)


class ArabicASR:
    """Arabic speech-to-text with lazy model loading and memory management."""

    def __init__(self, model_name: str = "whisper", device: str | None = None):
        """
        Args:
            model_name: "whisper" or "wav2vec2"
            device: torch device string; auto-detected if None
        """
        self.model_name = model_name
        self.device = device or config.DEVICE
        self.model = None
        self.processor = None
        self._loaded = False

    def _load_whisper(self) -> None:
        logger.info("Loading Whisper model: %s", config.WHISPER_MODEL_ID)
        self.processor = WhisperProcessor.from_pretrained(
            config.WHISPER_MODEL_ID, language="Arabic", task="transcribe"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            config.WHISPER_MODEL_ID,
            torch_dtype=torch.float32,
        )
        self.model.generation_config.language = "arabic"
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Whisper loaded on %s", self.device)

    def _load_wav2vec2(self) -> None:
        logger.info("Loading Wav2Vec2 model: %s", config.WAV2VEC2_MODEL_ID)
        self.processor = Wav2Vec2Processor.from_pretrained(config.WAV2VEC2_MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(config.WAV2VEC2_MODEL_ID)
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Wav2Vec2 loaded on %s", self.device)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.model_name == "whisper":
            self._load_whisper()
        elif self.model_name == "wav2vec2":
            self._load_wav2vec2()
        else:
            raise ValueError(f"Unknown model: {self.model_name}. Use 'whisper' or 'wav2vec2'.")

    def _transcribe_whisper_chunk(self, waveform: np.ndarray) -> str:
        input_features = self.processor(
            waveform, sampling_rate=config.SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language="arabic",
                task="transcribe",
                max_new_tokens=440,
            )

        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()

    def _transcribe_wav2vec2_chunk(self, waveform: np.ndarray) -> str:
        inputs = self.processor(
            waveform, sampling_rate=config.SAMPLE_RATE, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(predicted_ids)[0]
        return text.strip()

    def transcribe(self, audio_path: str | Path) -> dict:
        """Transcribe an audio file to Arabic text.

        Returns dict with keys: text, segments, duration_sec, model.
        """
        self._ensure_loaded()
        audio_path = Path(audio_path)

        waveform, sr = load_audio(audio_path)
        duration = len(waveform) / sr

        chunks = chunk_audio(waveform, sr)
        segments = []
        all_text_parts = []

        transcribe_fn = (
            self._transcribe_whisper_chunk
            if self.model_name == "whisper"
            else self._transcribe_wav2vec2_chunk
        )

        for chunk in chunks:
            text = transcribe_fn(chunk["waveform"])
            if text:
                segments.append({
                    "start": chunk["start_sec"],
                    "end": chunk["end_sec"],
                    "text": text,
                })
                all_text_parts.append(text)

        full_text = " ".join(all_text_parts)

        return {
            "text": full_text,
            "segments": segments,
            "duration_sec": duration,
            "model": self.model_name,
        }

    def transcribe_with_timestamps(self, audio_path: str | Path) -> list[dict]:
        """Transcribe with word-level timestamps (Whisper only).

        Returns list of dicts with start, end, text.
        """
        if self.model_name != "whisper":
            result = self.transcribe(audio_path)
            return result["segments"]

        self._ensure_loaded()
        waveform, sr = load_audio(audio_path)

        input_features = self.processor(
            waveform, sampling_rate=config.SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_features,
                language="arabic",
                task="transcribe",
                max_new_tokens=440,
                return_timestamps=True,
            )

        result = self.processor.batch_decode(output, skip_special_tokens=True, output_offsets=True)

        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], str):
            return self.transcribe(audio_path)["segments"]

        return result

    def unload(self) -> None:
        """Free model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        logger.info("ASR model unloaded")
