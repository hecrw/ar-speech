"""Emotion / sentiment detection on the Arabic transcript.

Approach: text-based sentiment classification using CAMeL-Lab's Arabic BERT,
which has been fine-tuned on Arabic sentiment data. Outputs probabilities for
{positive, negative, neutral}, which we map onto a coarse emotion label.

This is more reliable for Arabic than acoustic SER models — the publicly
available Arabic SER checkpoints are scarce and unstable, while Arabic text
sentiment is well-supported.
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config

logger = logging.getLogger(__name__)


# Map sentiment labels → human-friendly emotion bucket + emoji
LABEL_TO_EMOTION = {
    "positive": ("Positive", "😊"),
    "negative": ("Negative", "😠"),
    "neutral":  ("Neutral",  "😐"),
}


class ArabicEmotionDetector:
    """Sentiment / emotion classifier over Arabic text."""

    def __init__(self, device: str | None = None):
        self.device = device or config.DEVICE
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def _load(self) -> None:
        model_id = config.EMOTION_MODEL_ID
        logger.info("Loading emotion model: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device).eval()
        self._loaded = True
        logger.info("Emotion model loaded on %s", self.device)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    def detect(self, text: str) -> dict:
        """Classify a single piece of text. Returns label, emoji, scores."""
        self._ensure_loaded()
        if not text or not text.strip():
            return {"label": "Neutral", "emoji": "😐", "scores": {}}

        # CAMeL BERT max length 512
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits[0], dim=-1).cpu().tolist()
        id2label = self.model.config.id2label
        scores = {id2label[i].lower(): probs[i] for i in range(len(probs))}

        top_label_raw = max(scores, key=scores.get)
        label, emoji = LABEL_TO_EMOTION.get(top_label_raw, (top_label_raw.title(), "❓"))

        return {
            "label": label,
            "emoji": emoji,
            "confidence": scores[top_label_raw],
            "scores": scores,
        }

    def detect_segments(self, segments: list[dict]) -> list[dict]:
        """Classify each ASR segment individually for a per-segment timeline."""
        return [
            {**seg, "emotion": self.detect(seg["text"])}
            for seg in segments
        ]

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Emotion model unloaded")
