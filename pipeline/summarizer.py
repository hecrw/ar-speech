"""Arabic text summarization pipeline.

Primary backend: an Ollama-served LLM (aya-expanse:8b by default) accessed via
its HTTP API. Aya is Cohere's multilingual model with excellent Arabic. This
gives much higher-quality, more coherent summaries than a 580M-param seq2seq.

Fallback: a local mT5_multilingual_XLSum (or our fine-tuned variant) if Ollama
is unreachable, so the app still functions when the LLM server is down.
"""

import json
import logging
import re

import httpx

import config
from utils.text import normalize_arabic

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text.replace("\n", " ")).strip()


SYSTEM_PROMPT = (
    "أنت مساعد متخصص في تلخيص النصوص العربية. "
    "قم بتلخيص النص المعطى في فقرة واحدة موجزة ودقيقة (2-4 جمل). "
    "لا تضف معلومات غير موجودة في النص الأصلي. "
    "اكتب الملخص فقط دون مقدمات أو شروحات."
)


class ArabicSummarizer:
    """Arabic text summarizer using an Ollama LLM, with local seq2seq fallback."""

    def __init__(self, device: str | None = None):
        self.device = device or config.DEVICE
        self.ollama_url = config.OLLAMA_URL
        self.ollama_model = config.OLLAMA_SUMMARIZER_MODEL
        self._fallback = None  # Lazily initialized mT5 fallback

    # ── Backend selection ──────────────────────────────────────────────

    def _ollama_available(self) -> bool:
        try:
            r = httpx.get(f"{self.ollama_url}/api/tags", timeout=2.0)
            r.raise_for_status()
            tags = r.json().get("models", [])
            names = {m.get("name", "") for m in tags}
            if self.ollama_model in names:
                return True
            logger.warning(
                "Ollama is up but model %s not pulled. Available: %s",
                self.ollama_model,
                sorted(names),
            )
            return False
        except Exception as e:
            logger.warning("Ollama unreachable at %s: %s", self.ollama_url, e)
            return False

    def _ensure_fallback(self):
        """Lazy-load the local mT5 fallback only if needed."""
        if self._fallback is not None:
            return
        logger.info("Loading mT5 fallback summarizer: %s", config.SUMMARIZER_FALLBACK_ID)
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tok = AutoTokenizer.from_pretrained(config.SUMMARIZER_FALLBACK_ID)
        mod = AutoModelForSeq2SeqLM.from_pretrained(config.SUMMARIZER_FALLBACK_ID)
        mod.to(self.device).eval()
        self._fallback = (tok, mod, torch)

    # ── Public API ─────────────────────────────────────────────────────

    def summarize(self, text: str, max_length: int = config.MAX_SUMMARY_LENGTH) -> str:
        text = _clean(normalize_arabic(text))
        if not text.strip():
            return ""

        if self._ollama_available():
            try:
                return self._summarize_ollama(text, max_length)
            except Exception as e:
                logger.exception("Ollama summarization failed, falling back: %s", e)

        return self._summarize_fallback(text, max_length)

    def summarize_segments(self, segments: list[dict]) -> dict:
        if not segments:
            return {"overall_summary": "", "section_summaries": []}

        # Group segments into ~5-chunk sections
        section_size = 5
        sections = []
        for i in range(0, len(segments), section_size):
            chunk = segments[i : i + section_size]
            sections.append({
                "text": " ".join(s["text"] for s in chunk),
                "start": chunk[0]["start"],
                "end": chunk[-1]["end"],
            })

        section_summaries = [
            {
                "summary": self.summarize(s["text"], max_length=96),
                "start": s["start"],
                "end": s["end"],
            }
            for s in sections
        ]

        overall = self.summarize(" ".join(s["text"] for s in segments))

        return {"overall_summary": overall, "section_summaries": section_summaries}

    # ── Backends ───────────────────────────────────────────────────────

    def _summarize_ollama(self, text: str, max_length: int) -> str:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"النص:\n{text}\n\n"
            f"الملخص:"
        )
        r = httpx.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": max_length * 2,  # tokens, generous
                    "stop": ["النص:", "\n\n\n"],
                },
            },
            timeout=120.0,
        )
        r.raise_for_status()
        out = r.json()["response"].strip()
        # Trim any echoed prompt or label
        for prefix in ("الملخص:", "الملخص :", "الملخّص:"):
            if out.startswith(prefix):
                out = out[len(prefix):].strip()
        return out

    def _summarize_fallback(self, text: str, max_length: int) -> str:
        self._ensure_fallback()
        tok, mod, torch = self._fallback

        # Hierarchical for long inputs
        token_count = len(tok.encode(text, add_special_tokens=False))
        if token_count > config.MAX_INPUT_LENGTH:
            words = text.split()
            chunks = [" ".join(words[i:i + 200]) for i in range(0, len(words), 200)]
            summaries = [self._fallback_generate(c, max_length // 2) for c in chunks if c.strip()]
            return self._fallback_generate(" ".join(summaries), max_length)

        return self._fallback_generate(text, max_length)

    def _fallback_generate(self, text: str, max_length: int) -> str:
        tok, mod, torch = self._fallback
        ids = tok(text, return_tensors="pt", max_length=config.MAX_INPUT_LENGTH, truncation=True)
        ids = {k: v.to(self.device) for k, v in ids.items()}

        with torch.no_grad():
            out = mod.generate(
                input_ids=ids["input_ids"],
                max_length=max_length,
                num_beams=4,
                no_repeat_ngram_size=2,
            )[0]
        return tok.decode(out, skip_special_tokens=True).strip()

    def unload(self) -> None:
        if self._fallback is not None:
            tok, mod, _ = self._fallback
            del tok, mod
            self._fallback = None
        logger.info("Summarizer unloaded")
