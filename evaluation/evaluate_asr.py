"""Evaluate ASR models using Word Error Rate (WER) on Arabic speech datasets.

Usage:
    python -m evaluation.evaluate_asr --model whisper --num-samples 50
    python -m evaluation.evaluate_asr --model wav2vec2 --num-samples 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from jiwer import wer as compute_jiwer_wer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.asr import ArabicASR
from utils.text import normalize_arabic

logger = logging.getLogger(__name__)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis.

    Both texts are normalized (diacritics removed, alef standardized)
    before comparison for fair evaluation.
    """
    ref = normalize_arabic(reference).strip()
    hyp = normalize_arabic(hypothesis).strip()

    if not ref:
        return 1.0 if hyp else 0.0

    return compute_jiwer_wer(ref, hyp)


def evaluate_on_common_voice(
    model: ArabicASR,
    num_samples: int = 100,
    split: str = "test",
) -> dict:
    """Evaluate ASR model on Mozilla Common Voice Arabic.

    Uses streaming to avoid downloading the full dataset.
    """
    from datasets import load_dataset

    logger.info("Loading Common Voice Arabic (%s split, streaming)...", split)
    dataset = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "ar",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    results = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        reference = sample["sentence"]
        audio = sample["audio"]

        # Save audio temporarily for processing
        import soundfile as sf
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio["array"], audio["sampling_rate"])
            tmp_path = tmp.name

        try:
            result = model.transcribe(tmp_path)
            hypothesis = result["text"]
        except Exception as e:
            logger.warning("Failed to transcribe sample %d: %s", i, e)
            hypothesis = ""
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        sample_wer = compute_wer(reference, hypothesis)
        results.append({
            "sample_id": i,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": sample_wer,
        })

        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d samples", i + 1, num_samples)

    wer_values = [r["wer"] for r in results]

    return {
        "model": model.model_name,
        "dataset": "common_voice_11_0_ar",
        "split": split,
        "num_samples": len(results),
        "mean_wer": float(np.mean(wer_values)),
        "median_wer": float(np.median(wer_values)),
        "std_wer": float(np.std(wer_values)),
        "min_wer": float(np.min(wer_values)),
        "max_wer": float(np.max(wer_values)),
        "per_sample": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Arabic ASR models")
    parser.add_argument(
        "--model", choices=["whisper", "wav2vec2"], default="whisper",
        help="ASR model to evaluate",
    )
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Number of test samples to evaluate",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    asr = ArabicASR(model_name=args.model)
    results = evaluate_on_common_voice(asr, num_samples=args.num_samples)

    print(f"\n{'='*50}")
    print(f"ASR Evaluation Results ({args.model})")
    print(f"{'='*50}")
    print(f"Samples:    {results['num_samples']}")
    print(f"Mean WER:   {results['mean_wer']:.4f}")
    print(f"Median WER: {results['median_wer']:.4f}")
    print(f"Std WER:    {results['std_wer']:.4f}")
    print(f"Min WER:    {results['min_wer']:.4f}")
    print(f"Max WER:    {results['max_wer']:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")

    asr.unload()


if __name__ == "__main__":
    main()
