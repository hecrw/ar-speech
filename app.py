"""Gradio demo for the Arabic Audio Understanding system."""

import logging
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from pipeline.asr import ArabicASR
from pipeline.summarizer import ArabicSummarizer
from pipeline.embedder import ArabicEmbedder
from pipeline.emotion import ArabicEmotionDetector
from pipeline import keywords as kw
from utils.audio import validate_audio
from utils.text import clean_transcript

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_asr: ArabicASR | None = None
_asr_wav2vec: ArabicASR | None = None
_summarizer: ArabicSummarizer | None = None
_embedder: ArabicEmbedder | None = None
_emotion: ArabicEmotionDetector | None = None


def get_asr(model: str = "whisper") -> ArabicASR:
    global _asr, _asr_wav2vec
    if model == "wav2vec2":
        if _asr_wav2vec is None:
            _asr_wav2vec = ArabicASR(model_name="wav2vec2")
        return _asr_wav2vec
    if _asr is None:
        _asr = ArabicASR(model_name="whisper")
    return _asr


def get_summarizer() -> ArabicSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = ArabicSummarizer()
    return _summarizer


def get_embedder() -> ArabicEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ArabicEmbedder()
    return _embedder


def get_emotion() -> ArabicEmotionDetector:
    global _emotion
    if _emotion is None:
        _emotion = ArabicEmotionDetector()
    return _emotion


_LAST_SEGMENTS: list[dict] = []


KEYWORD_COLS = ["Match Type", "Keyword", "Time", "Count", "Score", "Segment"]
SEARCH_COLS = ["Score", "Time", "Segment"]


def _empty_keywords_df() -> pd.DataFrame:
    return pd.DataFrame(columns=KEYWORD_COLS)


def _empty_search_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SEARCH_COLS)


def process_audio(audio_path: str, keyword_input: str) -> tuple:
    global _LAST_SEGMENTS

    if audio_path is None:
        return "", "", "", _empty_keywords_df(), "Please upload an audio file."

    is_valid, msg = validate_audio(audio_path)
    if not is_valid:
        return "", "", "", _empty_keywords_df(), f"Invalid audio: {msg}"

    try:
        asr = get_asr("whisper")
        result = asr.transcribe(audio_path)
        transcript = clean_transcript(result["text"])
        if not transcript.strip():
            return "", "", "", _empty_keywords_df(), "No speech detected in the audio."

        _LAST_SEGMENTS = result["segments"]

        summarizer = get_summarizer()
        summary = summarizer.summarize(transcript)

        emotion_detector = get_emotion()
        emotion = emotion_detector.detect(transcript)
        emotion_html = _format_emotion(emotion)

        keyword_df = _empty_keywords_df()
        keyword_list = kw.parse_keyword_input(keyword_input)
        if keyword_list:
            keyword_df = _run_keyword_spotting(result["segments"], keyword_list)

        backend = "Ollama LLM" if summarizer._ollama_available() else "mT5 (fallback)"
        status = (
            f"Transcribed {result['duration_sec']:.1f}s with Whisper. "
            f"Summarized via {backend}. "
            f"Sentiment: {emotion['label']} ({emotion['confidence']:.0%})."
        )
        return transcript, summary, emotion_html, keyword_df, status

    except Exception as e:
        logger.exception("Pipeline error")
        return "", "", "", _empty_keywords_df(), f"Error: {e}"


def rerun_keywords(keyword_input: str) -> pd.DataFrame:
    if not _LAST_SEGMENTS:
        return _empty_keywords_df()
    keyword_list = kw.parse_keyword_input(keyword_input)
    if not keyword_list:
        return _empty_keywords_df()
    return _run_keyword_spotting(_LAST_SEGMENTS, keyword_list)


def _run_keyword_spotting(segments: list[dict], keyword_list: list[str]) -> pd.DataFrame:
    exact = kw.find_keywords_exact(segments, keyword_list)
    embedder = get_embedder()
    semantic = kw.find_keywords_semantic(segments, keyword_list, embedder)

    rows = []
    for h in exact:
        rows.append({
            "Match Type": "Exact",
            "Keyword": h["keyword"],
            "Time": f"{kw.format_timestamp(h['start'])} - {kw.format_timestamp(h['end'])}",
            "Count": str(h["count"]),
            "Score": "1.000",
            "Segment": h["segment_text"],
        })
    seen = {(h["keyword"], h["start"]) for h in exact}
    for h in semantic:
        key = (h["keyword"], h["start"])
        if key in seen:
            continue
        rows.append({
            "Match Type": "Semantic",
            "Keyword": h["keyword"],
            "Time": f"{kw.format_timestamp(h['start'])} - {kw.format_timestamp(h['end'])}",
            "Count": "-",
            "Score": f"{h['score']:.3f}",
            "Segment": h["segment_text"],
        })

    if not rows:
        return _empty_keywords_df()
    return pd.DataFrame(rows, columns=KEYWORD_COLS)


def _format_emotion(emotion: dict) -> str:
    if not emotion or not emotion.get("scores"):
        return ""
    rows = []
    for label, score in sorted(emotion["scores"].items(), key=lambda x: -x[1]):
        pct = int(round(score * 100))
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        rows.append(
            f"<tr><td style='padding:4px 12px'><b>{label.title()}</b></td>"
            f"<td style='padding:4px 12px;font-family:monospace'>{bar}</td>"
            f"<td style='padding:4px 12px'>{pct}%</td></tr>"
        )
    table = "<table style='border-collapse:collapse'>" + "".join(rows) + "</table>"
    return (
        f"<div style='font-size:1.1em'>"
        f"<div style='font-size:1.6em'><b>{emotion['label']}</b></div>"
        f"<div style='color:#666;margin-bottom:8px'>"
        f"Confidence: {emotion['confidence']:.0%}</div>"
        f"{table}</div>"
    )


def search_transcript(query: str, top_k: int = 5) -> tuple[str, pd.DataFrame]:
    if not _LAST_SEGMENTS:
        return "Process an audio file first.", _empty_search_df()
    if not query or not query.strip():
        return "Type a query.", _empty_search_df()

    try:
        embedder = get_embedder()
        seg_texts = [s["text"] for s in _LAST_SEGMENTS]
        seg_embeddings = embedder.embed_documents(seg_texts, show_progress=False)
        q = embedder.embed_query(query)
        sims = seg_embeddings @ q
        order = np.argsort(-sims)[: int(top_k)]

        rows = []
        for idx in order:
            seg = _LAST_SEGMENTS[int(idx)]
            rows.append({
                "Score": f"{float(sims[idx]):.3f}",
                "Time": f"{kw.format_timestamp(seg['start'])} - {kw.format_timestamp(seg['end'])}",
                "Segment": seg["text"],
            })

        df = pd.DataFrame(rows, columns=SEARCH_COLS) if rows else _empty_search_df()
        status = f"Found top {len(rows)} segments for: {query}"
        return status, df

    except Exception as e:
        logger.exception("Search failed")
        return f"Search error: {e}", _empty_search_df()


def compare_asr(audio_path: str) -> tuple[str, str]:
    if audio_path is None:
        return "Please upload an audio file.", ""
    is_valid, msg = validate_audio(audio_path)
    if not is_valid:
        return f"Invalid: {msg}", ""

    try:
        whisper_text = clean_transcript(get_asr("whisper").transcribe(audio_path)["text"])
    except Exception as e:
        whisper_text = f"Whisper error: {e}"

    try:
        wav2vec_text = clean_transcript(get_asr("wav2vec2").transcribe(audio_path)["text"])
    except Exception as e:
        wav2vec_text = f"Wav2Vec2 error: {e}"

    return whisper_text, wav2vec_text


EXAMPLE_KEYWORDS = "اجتماع, امتحان, طارئ, deadline"


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Arabic Audio Understanding System",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    ) as demo:
        gr.Markdown(
            "# Arabic Audio Understanding System\n"
            "**Speech -> Transcript -> Summary + Sentiment + Keywords**\n\n"
            "End-to-end deep learning pipeline for Arabic audio analysis."
        )

        with gr.Tab("Full Pipeline"):
            gr.Markdown(
                "Upload Arabic audio. Get transcript, AI summary, sentiment, and keyword timestamps."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    audio_input = gr.Audio(
                        type="filepath",
                        label="Arabic Audio (upload or record)",
                        sources=["upload", "microphone"],
                    )
                with gr.Column(scale=1):
                    keyword_box = gr.Textbox(
                        label="Keywords to spot (comma-separated)",
                        placeholder=EXAMPLE_KEYWORDS,
                        lines=2,
                    )
                    process_btn = gr.Button("Process Audio", variant="primary", size="lg")

            status_output = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                with gr.Column(scale=2):
                    transcript_output = gr.Textbox(
                        label="Transcript",
                        lines=10,
                        rtl=True,
                        interactive=False,
                    )
                    summary_output = gr.Textbox(
                        label="Summary (LLM)",
                        lines=5,
                        rtl=True,
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    emotion_output = gr.HTML(label="Sentiment / Emotion")

            gr.Markdown("### Keyword Hits (with timestamps)")
            keyword_output = gr.Dataframe(
                headers=KEYWORD_COLS,
                value=_empty_keywords_df(),
                wrap=True,
                interactive=False,
            )
            rerun_kw_btn = gr.Button("Re-run keyword search on last audio")

            process_btn.click(
                fn=process_audio,
                inputs=[audio_input, keyword_box],
                outputs=[transcript_output, summary_output, emotion_output,
                         keyword_output, status_output],
            )
            rerun_kw_btn.click(
                fn=rerun_keywords,
                inputs=[keyword_box],
                outputs=[keyword_output],
            )

            gr.Markdown("### Search Inside Transcript")
            gr.Markdown(
                "Type an Arabic query. The embedder ranks segments from the "
                "**last processed audio** by semantic similarity."
            )
            with gr.Row():
                query_box = gr.Textbox(
                    label="Query (Arabic)",
                    placeholder="ابحث في النص...",
                    rtl=True,
                    scale=3,
                )
                query_top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K", scale=1)
            query_btn = gr.Button("Search Transcript", variant="primary")
            query_status = gr.Textbox(label="Search Status", interactive=False)
            query_results = gr.Dataframe(
                headers=SEARCH_COLS,
                value=_empty_search_df(),
                wrap=True,
                interactive=False,
            )
            query_btn.click(
                fn=search_transcript,
                inputs=[query_box, query_top_k],
                outputs=[query_status, query_results],
            )

        with gr.Tab("ASR Comparison"):
            gr.Markdown("Compare **fine-tuned Whisper** vs **Wav2Vec2 XLSR-53 Arabic**.")
            compare_audio = gr.Audio(type="filepath", label="Audio")
            compare_btn = gr.Button("Compare Models", variant="primary")
            with gr.Row():
                whisper_output = gr.Textbox(
                    label="Whisper (fine-tuned turbo)", lines=8, rtl=True, interactive=False,
                )
                wav2vec_output = gr.Textbox(
                    label="Wav2Vec2 (XLSR-53 Arabic)", lines=8, rtl=True, interactive=False,
                )
            compare_btn.click(compare_asr, [compare_audio], [whisper_output, wav2vec_output])

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
