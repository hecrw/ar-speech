"""Gradio demo for the Arabic Audio Understanding & Retrieval System.

Pipeline: Speech → Transcript → Summary + Emotion + Keywords → Semantic Search
"""

import logging
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from pipeline.asr import ArabicASR
from pipeline.summarizer import ArabicSummarizer
from pipeline.embedder import ArabicEmbedder
from pipeline.search import SemanticSearchEngine
from pipeline.emotion import ArabicEmotionDetector
from pipeline import keywords as kw
from utils.audio import validate_audio
from utils.text import clean_transcript, split_sentences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lazy-loaded singletons ─────────────────────────────────────────────

_asr: ArabicASR | None = None
_asr_wav2vec: ArabicASR | None = None
_summarizer: ArabicSummarizer | None = None
_embedder: ArabicEmbedder | None = None
_search_engine: SemanticSearchEngine | None = None
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


def get_search_engine() -> SemanticSearchEngine:
    global _search_engine
    if _search_engine is None:
        embedder = get_embedder()
        _search_engine = SemanticSearchEngine(
            embedding_dim=embedder.get_embedding_dim()
        )
        index_path = config.INDEX_DIR
        if (index_path / "index.faiss").exists():
            _search_engine.load_index(index_path)
            logger.info("Loaded existing search index")
    return _search_engine


# ── Per-session segment cache (for keyword spotting after transcribe) ──
_LAST_SEGMENTS: list[dict] = []
_LAST_AUDIO_PATH: str | None = None


# ── Pipeline Functions ─────────────────────────────────────────────────

def process_audio(audio_path: str, keyword_input: str) -> tuple:
    """Full pipeline. Returns transcript, summary, emotion HTML, keyword DF, status."""
    global _LAST_SEGMENTS, _LAST_AUDIO_PATH

    if audio_path is None:
        return "", "", "", pd.DataFrame(), "Please upload an audio file."

    is_valid, msg = validate_audio(audio_path)
    if not is_valid:
        return "", "", "", pd.DataFrame(), f"Invalid audio: {msg}"

    try:
        # 1. Transcribe
        asr = get_asr("whisper")
        result = asr.transcribe(audio_path)
        transcript = clean_transcript(result["text"])
        if not transcript.strip():
            return "", "", "", pd.DataFrame(), "No speech detected in the audio."

        _LAST_SEGMENTS = result["segments"]
        _LAST_AUDIO_PATH = audio_path

        # 2. Summarize (LLM via Ollama, falls back to mT5 automatically)
        summarizer = get_summarizer()
        summary = summarizer.summarize(transcript)

        # 3. Emotion / sentiment over the full transcript
        emotion_detector = get_emotion()
        emotion = emotion_detector.detect(transcript)
        emotion_html = _format_emotion(emotion)

        # 4. Keyword spotting (exact + semantic) if user provided keywords
        keyword_df = pd.DataFrame()
        keyword_list = kw.parse_keyword_input(keyword_input)
        if keyword_list:
            keyword_df = _run_keyword_spotting(result["segments"], keyword_list)

        # 5. Index sentences for semantic search
        embedder = get_embedder()
        engine = get_search_engine()
        sentences = split_sentences(transcript)
        if sentences:
            embeddings = embedder.embed_documents(sentences, show_progress=False)
            metadata = [
                {
                    "audio_file": str(Path(audio_path).name),
                    "segment_id": i,
                    "start_sec": 0.0,
                    "end_sec": result["duration_sec"],
                    "text": sent,
                }
                for i, sent in enumerate(sentences)
            ]
            engine.add_documents(embeddings, metadata)
            config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
            engine.save_index(config.INDEX_DIR)

        backend = "Ollama LLM" if summarizer._ollama_available() else "mT5 (fallback)"
        status = (
            f"✓ Transcribed {result['duration_sec']:.1f}s with Whisper • "
            f"Summarized via {backend} • "
            f"Sentiment: {emotion['label']} ({emotion['confidence']:.0%}) • "
            f"Indexed {len(sentences)} sentences for search"
        )
        return transcript, summary, emotion_html, keyword_df, status

    except Exception as e:
        logger.exception("Pipeline error")
        return "", "", "", pd.DataFrame(), f"Error: {e}"


def rerun_keywords(keyword_input: str) -> pd.DataFrame:
    """Re-run keyword spotting on the last transcribed audio without re-transcribing."""
    if not _LAST_SEGMENTS:
        return pd.DataFrame({"info": ["Run the Full Pipeline on an audio file first."]})
    keyword_list = kw.parse_keyword_input(keyword_input)
    if not keyword_list:
        return pd.DataFrame()
    return _run_keyword_spotting(_LAST_SEGMENTS, keyword_list)


def _run_keyword_spotting(segments: list[dict], keyword_list: list[str]) -> pd.DataFrame:
    """Combine exact + semantic matches into a single ranked DataFrame."""
    exact = kw.find_keywords_exact(segments, keyword_list)
    embedder = get_embedder()
    semantic = kw.find_keywords_semantic(segments, keyword_list, embedder)

    rows = []
    for h in exact:
        rows.append({
            "Match Type": "Exact",
            "Keyword": h["keyword"],
            "Time": f"{kw.format_timestamp(h['start'])} – {kw.format_timestamp(h['end'])}",
            "Count": h["count"],
            "Score": "1.000",
            "Segment": h["segment_text"],
        })
    seen_segments = {(h["keyword"], h["start"]) for h in exact}
    for h in semantic:
        key = (h["keyword"], h["start"])
        if key in seen_segments:
            continue
        rows.append({
            "Match Type": "Semantic",
            "Keyword": h["keyword"],
            "Time": f"{kw.format_timestamp(h['start'])} – {kw.format_timestamp(h['end'])}",
            "Count": "—",
            "Score": f"{h['score']:.3f}",
            "Segment": h["segment_text"],
        })

    if not rows:
        return pd.DataFrame({"info": [f"No matches found for: {', '.join(keyword_list)}"]})
    return pd.DataFrame(rows)


def _format_emotion(emotion: dict) -> str:
    """HTML widget showing sentiment label, emoji, and a probability bar chart."""
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
        f"<div style='font-size:2em'>{emotion['emoji']} <b>{emotion['label']}</b></div>"
        f"<div style='color:#666;margin-bottom:8px'>"
        f"Confidence: {emotion['confidence']:.0%}</div>"
        f"{table}</div>"
    )


def load_arcd_corpus(num_passages: int = 500) -> str:
    """Load Arabic Reading Comprehension contexts into the FAISS index.

    Gives the user something to search against without first having to
    process audio. Also doubles as a sanity check that the embedder works.
    """
    try:
        from datasets import load_dataset
        from utils.text import normalize_arabic

        raw = load_dataset("hsseinmz/arcd", split="train")
        # Unique passages, preserve insertion order
        seen = {}
        for item in raw:
            ctx = item["context"].strip()
            if ctx and ctx not in seen:
                seen[ctx] = len(seen)
            if len(seen) >= num_passages:
                break
        passages = list(seen.keys())

        embedder = get_embedder()
        engine = get_search_engine()
        engine.clear()

        embeddings = embedder.embed_documents(passages, show_progress=False)
        metadata = [
            {
                "audio_file": "ARCD",
                "segment_id": i,
                "start_sec": 0.0,
                "end_sec": 0.0,
                "text": text,
            }
            for i, text in enumerate(passages)
        ]
        engine.add_documents(embeddings, metadata)

        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        engine.save_index(config.INDEX_DIR)

        return f"✓ Indexed {len(passages)} ARCD passages. Try queries like 'متى ولد' or 'ما هي العاصمة'."
    except Exception as e:
        logger.exception("ARCD load failed")
        return f"Error loading ARCD: {e}"


def evaluate_search_metrics(num_samples: int = 50) -> pd.DataFrame:
    """Run Precision@K / Recall@K / MRR evaluation on ARCD held-out queries."""
    try:
        from evaluation.evaluate_search import (
            build_evaluation_dataset,
            evaluate_search as _eval,
        )
        embedder = get_embedder()
        engine, test_queries = build_evaluation_dataset(embedder, num_samples)
        results = _eval(engine, embedder, test_queries, k_values=[1, 3, 5, 10])

        rows = []
        for k in results["k_values"]:
            rows.append({
                "K": k,
                "Precision@K": f"{results['metrics'][f'precision@{k}']['mean']:.4f}",
                "Recall@K":    f"{results['metrics'][f'recall@{k}']['mean']:.4f}",
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.exception("Search eval failed")
        return pd.DataFrame({"error": [str(e)]})


def search_query(query: str, top_k: int = 5) -> pd.DataFrame:
    if not query.strip():
        return pd.DataFrame(columns=["Score", "Text", "Audio File"])

    engine = get_search_engine()
    if engine.get_stats()["num_vectors"] == 0:
        return pd.DataFrame({"info": ["Index is empty. Process some audio first."]})

    embedder = get_embedder()
    query_embedding = embedder.embed_query(query)
    results = engine.search(query_embedding, top_k=int(top_k))
    if not results:
        return pd.DataFrame(columns=["Score", "Text", "Audio File"])

    return pd.DataFrame([
        {
            "Score": f"{r['score']:.4f}",
            "Text": r.get("text", ""),
            "Audio File": r.get("audio_file", ""),
        }
        for r in results
    ])


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


# ── Gradio UI ──────────────────────────────────────────────────────────

EXAMPLE_KEYWORDS = "اجتماع, امتحان, طارئ, deadline"


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Arabic Audio Understanding System",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    ) as demo:
        gr.Markdown(
            "# 🎙️ Arabic Audio Understanding & Retrieval System\n"
            "**Speech → Transcript → Summary + Sentiment + Keywords → Semantic Search**\n\n"
            "End-to-end deep learning pipeline for Arabic audio analysis."
        )

        # ── Tab 1: Full Pipeline ──
        with gr.Tab("🚀 Full Pipeline"):
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
                        label="📝 Transcript",
                        lines=10,
                        rtl=True,
                        interactive=False,
                    )
                    summary_output = gr.Textbox(
                        label="📋 Summary (LLM)",
                        lines=5,
                        rtl=True,
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    emotion_output = gr.HTML(label="💭 Sentiment / Emotion")

            gr.Markdown("### 🔑 Keyword Hits (with timestamps)")
            keyword_output = gr.Dataframe(
                headers=["Match Type", "Keyword", "Time", "Count", "Score", "Segment"],
                wrap=True,
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

        # ── Tab 2: Semantic Search ──
        with gr.Tab("🔍 Semantic Search"):
            gr.Markdown(
                "Search through indexed audio segments — or load the **ARCD** "
                "(Arabic Reading Comprehension) corpus to demo without audio."
            )
            with gr.Row():
                load_arcd_btn = gr.Button("Load ARCD demo corpus (500 passages)")
                arcd_status = gr.Textbox(label="Status", interactive=False)
            load_arcd_btn.click(load_arcd_corpus, [], [arcd_status])

            with gr.Row():
                search_input = gr.Textbox(
                    label="Search Query (Arabic)",
                    placeholder="ابحث هنا...",
                    rtl=True,
                )
                top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Top-K")
            search_btn = gr.Button("Search", variant="primary")
            search_results = gr.Dataframe(
                headers=["Score", "Text", "Audio File"],
                wrap=True,
            )
            search_btn.click(search_query, [search_input, top_k_slider], [search_results])

            gr.Markdown("### 📊 Search Quality Evaluation (ARCD held-out)")
            gr.Markdown(
                "Quantitative metrics on real ARCD question→passage pairs. "
                "Precision@K = how often the correct passage appears in top-K. "
                "Recall@K = same thing weighted by total relevant passages."
            )
            with gr.Row():
                eval_samples = gr.Slider(10, 200, value=50, step=10, label="Eval queries")
                eval_btn = gr.Button("Run Evaluation", variant="secondary")
            eval_results = gr.Dataframe(headers=["K", "Precision@K", "Recall@K"])
            eval_btn.click(evaluate_search_metrics, [eval_samples], [eval_results])

        # ── Tab 3: ASR Comparison ──
        with gr.Tab("⚖️ ASR Comparison"):
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

        # ── Tab 4: System Info ──
        with gr.Tab("ℹ️ System Info"):
            gr.Markdown("### Models")
            gr.Markdown(
                f"| Component | Model |\n"
                f"|-----------|-------|\n"
                f"| **ASR (primary)** | `{config.WHISPER_MODEL_ID}` |\n"
                f"| **ASR (compare)** | `{config.WAV2VEC2_MODEL_ID}` |\n"
                f"| **Summarizer (primary)** | `{config.OLLAMA_SUMMARIZER_MODEL}` via Ollama |\n"
                f"| **Summarizer (fallback)** | `{config.SUMMARIZER_FALLBACK_ID}` |\n"
                f"| **Embeddings** | `{config.EMBEDDING_MODEL_ID}` |\n"
                f"| **Sentiment** | `{config.EMOTION_MODEL_ID}` |\n"
                f"| **Device** | `{config.DEVICE}` |\n"
            )

            gr.Markdown("### Evaluation Metrics")
            gr.Markdown(
                "| Task | Metric | Test Set |\n"
                "|------|--------|----------|\n"
                "| Speech Recognition | Word Error Rate (WER, Arabic-normalized) | Common Voice AR test |\n"
                "| Summarization | ROUGE-1 / ROUGE-2 / ROUGE-L | WikiLingua AR held-out |\n"
                "| Semantic Search | Precision@K / Recall@K / MRR | ARCD held-out |\n"
                "| Sentiment | Top-1 accuracy | CAMeL benchmark |\n"
            )

            gr.Markdown("### Datasets")
            gr.Markdown(
                "- **ASR fine-tune:** Mozilla Common Voice (Arabic)\n"
                "- **Summarization fine-tune:** WikiLingua (Arabic)\n"
                "- **Embedding fine-tune:** ARCD (Arabic Reading Comprehension)\n"
            )

            gr.Markdown("### Architecture")
            gr.Markdown(
                "```\n"
                "                       Audio Input\n"
                "                            │\n"
                "                            ▼\n"
                "                ┌─────────────────────────┐\n"
                "                │  Whisper-large-v3-turbo │\n"
                "                │   (fine-tuned, Arabic)  │\n"
                "                └───────────┬─────────────┘\n"
                "                            │ transcript + segments\n"
                "         ┌──────────────────┼──────────────────┬─────────────────┐\n"
                "         ▼                  ▼                  ▼                 ▼\n"
                "  ┌───────────────┐  ┌────────────┐    ┌──────────────┐  ┌──────────────┐\n"
                "  │  Aya-8B LLM   │  │ CAMeL BERT │    │  Nomic-Embed │  │   Keyword    │\n"
                "  │   (Ollama)    │  │ (sentiment)│    │   v2-MoE     │  │  (exact +    │\n"
                "  │ summarization │  │            │    │  embeddings  │  │  semantic)   │\n"
                "  └───────┬───────┘  └─────┬──────┘    └──────┬───────┘  └──────┬───────┘\n"
                "          ▼                ▼                  ▼                 ▼\n"
                "      Summary          Emotion             FAISS Index      Hits + Times\n"
                "                                                │\n"
                "                                                ▼\n"
                "                                          Search Engine\n"
                "```"
            )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
