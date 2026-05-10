"""Evaluate semantic search using Precision@K and Recall@K.

Uses Arabic reading comprehension data as ground truth:
questions → queries, context paragraphs → documents.

Usage:
    python -m evaluation.evaluate_search --num-samples 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.embedder import ArabicEmbedder
from pipeline.search import SemanticSearchEngine
from utils.text import normalize_arabic

logger = logging.getLogger(__name__)


def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Compute Precision@K."""
    if k == 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    return len(set(retrieved_at_k) & relevant) / k


def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Compute Recall@K."""
    if not relevant:
        return 0.0
    retrieved_at_k = retrieved[:k]
    return len(set(retrieved_at_k) & relevant) / len(relevant)


def build_evaluation_dataset(
    embedder: ArabicEmbedder,
    num_samples: int = 50,
) -> tuple[SemanticSearchEngine, list[dict]]:
    """Build a search index from Arabic QA data and return test queries.

    Uses the ARCD or similar Arabic reading comprehension dataset.
    """
    from datasets import load_dataset

    logger.info("Loading Arabic QA dataset for search evaluation...")

    try:
        dataset = load_dataset("hsseinmz/arcd", split="train")
    except Exception as e:
        logger.warning("ARCD not available (%s), using synthetic data", e)
        return _build_synthetic_dataset(embedder)

    # Extract unique contexts and build document index
    contexts = {}  # context_text -> doc_id
    test_queries = []

    for item in dataset:
        context = normalize_arabic(item["context"])
        if context not in contexts:
            contexts[context] = len(contexts)

        for qa in item.get("qas", [{"question": item.get("question", ""), "answers": item.get("answers", {"text": []})}]):
            question = qa.get("question", item.get("question", ""))
            if question and len(test_queries) < num_samples:
                test_queries.append({
                    "query": normalize_arabic(question),
                    "relevant_ids": {contexts[context]},
                })

    if not contexts:
        return _build_synthetic_dataset(embedder)

    # Build search index
    doc_texts = sorted(contexts.keys(), key=lambda x: contexts[x])
    engine = SemanticSearchEngine(embedding_dim=embedder.get_embedding_dim())

    logger.info("Embedding %d documents...", len(doc_texts))
    embeddings = embedder.embed_documents(doc_texts)

    metadata = [{"doc_id": i, "text": text} for i, text in enumerate(doc_texts)]
    engine.add_documents(embeddings, metadata)

    return engine, test_queries[:num_samples]


def _build_synthetic_dataset(
    embedder: ArabicEmbedder,
) -> tuple[SemanticSearchEngine, list[dict]]:
    """Fallback: build a small synthetic dataset for testing."""
    documents = [
        "الذكاء الاصطناعي يغير عالم التكنولوجيا بشكل جذري",
        "التعلم العميق هو فرع من التعلم الآلي يستخدم الشبكات العصبية",
        "معالجة اللغة الطبيعية تساعد الحواسيب على فهم اللغة البشرية",
        "الرؤية الحاسوبية تمكن الآلات من تفسير الصور والفيديو",
        "الروبوتات الذكية تستخدم في المصانع والمستشفيات",
        "البيانات الضخمة تلعب دورا محوريا في تطوير الذكاء الاصطناعي",
        "الحوسبة السحابية توفر موارد حاسوبية عبر الإنترنت",
        "أمن المعلومات يحمي البيانات من الاختراقات والهجمات",
        "إنترنت الأشياء يربط الأجهزة المنزلية بالشبكة",
        "الواقع الافتراضي يقدم تجارب غامرة للمستخدمين",
    ]

    queries = [
        {"query": "ما هو الذكاء الاصطناعي", "relevant_ids": {0, 5}},
        {"query": "كيف تعمل الشبكات العصبية", "relevant_ids": {1}},
        {"query": "فهم اللغة البشرية بالحاسوب", "relevant_ids": {2}},
        {"query": "تحليل الصور بالذكاء الاصطناعي", "relevant_ids": {3}},
        {"query": "حماية البيانات والأمن السيبراني", "relevant_ids": {7}},
    ]

    engine = SemanticSearchEngine(embedding_dim=embedder.get_embedding_dim())
    embeddings = embedder.embed_documents(documents)
    metadata = [{"doc_id": i, "text": doc} for i, doc in enumerate(documents)]
    engine.add_documents(embeddings, metadata)

    return engine, queries


def evaluate_search(
    engine: SemanticSearchEngine,
    embedder: ArabicEmbedder,
    test_queries: list[dict],
    k_values: list[int] | None = None,
) -> dict:
    """Evaluate search quality across multiple K values."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    all_results = {f"precision@{k}": [] for k in k_values}
    all_results.update({f"recall@{k}": [] for k in k_values})
    per_query = []

    for query_data in test_queries:
        query_text = query_data["query"]
        relevant = query_data["relevant_ids"]

        # Convert set elements to ints if needed
        relevant = {int(r) for r in relevant}

        query_embedding = embedder.embed_query(query_text)
        results = engine.search(query_embedding, top_k=max(k_values))
        retrieved_ids = [r["doc_id"] for r in results]

        query_metrics = {"query": query_text}
        for k in k_values:
            p = precision_at_k(relevant, retrieved_ids, k)
            r = recall_at_k(relevant, retrieved_ids, k)
            all_results[f"precision@{k}"].append(p)
            all_results[f"recall@{k}"].append(r)
            query_metrics[f"P@{k}"] = p
            query_metrics[f"R@{k}"] = r

        per_query.append(query_metrics)

    # Aggregate
    summary = {}
    for metric, values in all_results.items():
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return {
        "num_queries": len(test_queries),
        "k_values": k_values,
        "metrics": summary,
        "per_query": per_query,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic search")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    embedder = ArabicEmbedder()
    engine, test_queries = build_evaluation_dataset(embedder, args.num_samples)
    results = evaluate_search(engine, embedder, test_queries)

    print(f"\n{'='*50}")
    print("Semantic Search Evaluation")
    print(f"{'='*50}")
    print(f"Queries: {results['num_queries']}")
    for metric, values in results["metrics"].items():
        print(f"{metric}: {values['mean']:.4f} (+/- {values['std']:.4f})")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")

    embedder.unload()


if __name__ == "__main__":
    main()
