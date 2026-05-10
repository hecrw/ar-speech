"""Keyword spotting over an ASR transcript.

Two flavors:
  1) Exact / substring keyword matching with timestamps from Whisper segments.
     Fast, deterministic, useful for known-vocabulary alerts ("emergency",
     "deadline", "exam") which is exactly what the project brief calls for.

  2) Semantic keyword matching: each segment is embedded; keywords are
     embedded as queries; a match is reported if cosine similarity exceeds
     a threshold. Catches paraphrases (e.g. "اجتماع طارئ" matches "emergency").

The Gradio app exposes both — exact for precision, semantic for recall.
"""

import logging
import re

import numpy as np

from utils.text import normalize_arabic

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def _tokenize(text: str) -> list[str]:
    return _WS.split(normalize_arabic(text).strip())


def find_keywords_exact(
    segments: list[dict],
    keywords: list[str],
) -> list[dict]:
    """Find exact (substring) keyword occurrences in each segment.

    Args:
        segments: list of {start, end, text} from ArabicASR.transcribe.
        keywords: list of keyword strings (Arabic or English).

    Returns:
        list of hits: [{keyword, segment_text, start, end, count}, ...]
    """
    hits = []
    for kw_raw in keywords:
        kw = normalize_arabic(kw_raw).strip()
        if not kw:
            continue
        for seg in segments:
            seg_norm = normalize_arabic(seg["text"])
            count = seg_norm.count(kw)
            if count > 0:
                hits.append({
                    "keyword": kw_raw,
                    "segment_text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "count": count,
                })
    return hits


def find_keywords_semantic(
    segments: list[dict],
    keywords: list[str],
    embedder,
    threshold: float = 0.55,
    top_k_per_keyword: int = 3,
) -> list[dict]:
    """Find semantically similar segments to each keyword via cosine similarity.

    Args:
        segments: list of {start, end, text}.
        keywords: list of keyword/phrase strings.
        embedder: an ArabicEmbedder instance (used for both segments & queries).
        threshold: minimum cosine similarity to count as a hit.
        top_k_per_keyword: cap on hits per keyword.

    Returns:
        list of hits sorted by score descending, with the original segment text.
    """
    if not segments or not keywords:
        return []

    seg_texts = [s["text"] for s in segments]
    seg_embeddings = embedder.embed_documents(seg_texts, show_progress=False)

    hits = []
    for kw in keywords:
        if not kw.strip():
            continue
        q = embedder.embed_query(kw)
        # Both are L2-normalized → dot product is cosine similarity
        sims = seg_embeddings @ q
        order = np.argsort(-sims)
        for idx in order[:top_k_per_keyword]:
            score = float(sims[idx])
            if score < threshold:
                break
            seg = segments[int(idx)]
            hits.append({
                "keyword": kw,
                "segment_text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
                "score": score,
            })
    hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    return hits


def parse_keyword_input(text: str) -> list[str]:
    """User types comma- or newline-separated keywords; we split & strip."""
    if not text:
        return []
    parts = re.split(r"[,\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def format_timestamp(seconds: float) -> str:
    """Pretty mm:ss formatter."""
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
