"""Evaluate summarization models using ROUGE scores on Arabic text.

Usage:
    python -m evaluation.evaluate_summary --model arat5 --num-samples 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.summarizer import ArabicSummarizer
from utils.text import normalize_arabic

logger = logging.getLogger(__name__)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Both texts are normalized before scoring.
    """
    ref = normalize_arabic(reference).strip()
    hyp = normalize_arabic(hypothesis).strip()

    if not ref or not hyp:
        return {
            "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        }

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    scores = scorer.score(ref, hyp)

    return {
        metric: {
            "precision": score.precision,
            "recall": score.recall,
            "fmeasure": score.fmeasure,
        }
        for metric, score in scores.items()
    }


def evaluate_on_dataset(
    model: ArabicSummarizer,
    num_samples: int = 50,
) -> dict:
    """Evaluate summarization model on an Arabic summarization dataset."""
    from datasets import load_dataset

    logger.info("Loading Arabic summarization dataset...")

    # Try loading an Arabic news summarization dataset
    try:
        dataset = load_dataset(
            "csebuetnlp/xlsum", "arabic",
            split="test",
            streaming=True,
            trust_remote_code=True,
        )
        text_key = "text"
        summary_key = "summary"
    except Exception:
        logger.warning("XL-Sum not available, using synthetic evaluation")
        return _synthetic_evaluation(model, num_samples)

    results = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        source_text = sample[text_key]
        reference_summary = sample[summary_key]

        try:
            hypothesis = model.summarize(source_text)
        except Exception as e:
            logger.warning("Failed to summarize sample %d: %s", i, e)
            hypothesis = ""

        scores = compute_rouge(reference_summary, hypothesis)
        results.append({
            "sample_id": i,
            "reference": reference_summary,
            "hypothesis": hypothesis,
            "rouge_scores": scores,
        })

        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d samples", i + 1, num_samples)

    return _aggregate_results(results, "silma-9b")


def _synthetic_evaluation(model: ArabicSummarizer, num_samples: int) -> dict:
    """Fallback evaluation using sample Arabic texts."""
    sample_texts = [
        "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهتم بإنشاء آلات ذكية قادرة على أداء المهام التي تتطلب عادة ذكاء بشريا. يشمل ذلك التعلم الآلي ومعالجة اللغة الطبيعية والرؤية الحاسوبية والروبوتات. تطورت تقنيات الذكاء الاصطناعي بشكل كبير في السنوات الأخيرة وأصبحت تستخدم في العديد من المجالات مثل الطب والتعليم والأعمال.",
        "تعتبر اللغة العربية من أكثر اللغات انتشارا في العالم حيث يتحدث بها أكثر من ثلاثمائة مليون شخص. وهي اللغة الرسمية في اثنتين وعشرين دولة عربية. تتميز اللغة العربية بغناها اللغوي وتنوع مفرداتها وقواعدها النحوية المعقدة.",
    ]

    results = []
    for i, text in enumerate(sample_texts[:num_samples]):
        hypothesis = model.summarize(text)
        # Use first sentence as pseudo-reference
        ref = text.split(".")[0] + "."
        scores = compute_rouge(ref, hypothesis)
        results.append({
            "sample_id": i,
            "reference": ref,
            "hypothesis": hypothesis,
            "rouge_scores": scores,
        })

    return _aggregate_results(results, "silma-9b")


def _aggregate_results(results: list[dict], model_name: str) -> dict:
    """Aggregate per-sample ROUGE scores into summary statistics."""
    metrics = {}
    for metric in ["rouge1", "rouge2", "rougeL"]:
        fmeasures = [r["rouge_scores"][metric]["fmeasure"] for r in results]
        precisions = [r["rouge_scores"][metric]["precision"] for r in results]
        recalls = [r["rouge_scores"][metric]["recall"] for r in results]
        metrics[metric] = {
            "mean_fmeasure": float(np.mean(fmeasures)),
            "mean_precision": float(np.mean(precisions)),
            "mean_recall": float(np.mean(recalls)),
            "std_fmeasure": float(np.std(fmeasures)),
        }

    return {
        "model": model_name,
        "num_samples": len(results),
        "metrics": metrics,
        "per_sample": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Arabic summarization")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    summarizer = ArabicSummarizer()
    results = evaluate_on_dataset(summarizer, num_samples=args.num_samples)

    print(f"\n{'='*50}")
    print("Summarization Evaluation (SILMA-9B)")
    print(f"{'='*50}")
    for metric, values in results["metrics"].items():
        print(f"{metric}: F1={values['mean_fmeasure']:.4f} "
              f"P={values['mean_precision']:.4f} R={values['mean_recall']:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")

    summarizer.unload()


if __name__ == "__main__":
    main()
