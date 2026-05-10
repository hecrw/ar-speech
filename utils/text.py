"""Arabic text normalization and preprocessing utilities."""

import re
import unicodedata


# Unicode ranges for Arabic diacritics (tashkeel)
_DIACRITICS_RE = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

# Tatweel (kashida)
_TATWEEL = "\u0640"

# Alef variants → bare alef
_ALEF_VARIANTS = {
    "\u0622": "\u0627",  # alef with madda
    "\u0623": "\u0627",  # alef with hamza above
    "\u0625": "\u0627",  # alef with hamza below
    "\u0671": "\u0627",  # alef wasla
}
_ALEF_RE = re.compile("[" + "".join(_ALEF_VARIANTS.keys()) + "]")

# Multiple whitespace
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text by removing diacritics and standardizing characters."""
    # Remove diacritics
    text = _DIACRITICS_RE.sub("", text)

    # Remove tatweel
    text = text.replace(_TATWEEL, "")

    # Normalize alef variants
    text = _ALEF_RE.sub("\u0627", text)

    # Normalize whitespace
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return text


def clean_transcript(text: str) -> str:
    """Clean ASR transcript output for downstream processing."""
    text = normalize_arabic(text)

    # Remove repeated consecutive words (common ASR artifact)
    words = text.split()
    cleaned = []
    for i, word in enumerate(words):
        if i == 0 or word != words[i - 1]:
            cleaned.append(word)
    text = " ".join(cleaned)

    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split Arabic text into sentences."""
    # Split on Arabic full stop, regular period, question mark, exclamation
    sentences = re.split(r"[.\u06D4?!؟]\s*", text)
    return [s.strip() for s in sentences if s.strip()]


def count_words(text: str) -> int:
    """Count words in Arabic text."""
    return len(text.split())
