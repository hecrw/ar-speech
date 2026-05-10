"""FAISS-based semantic search engine for Arabic text segments."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

import config

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    def __init__(self, embedding_dim: int = config.EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata: list[dict] = []

    def add_documents(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) length mismatch"
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata)

        logger.info("Added %d documents (total: %d)", len(metadata), self.index.ntotal)

    def search(self, query_embedding: np.ndarray, top_k: int = config.TOP_K_DEFAULT) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        top_k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query, top_k)

        results = []
        for j in range(top_k):
            idx = int(indices[0][j])
            if idx < 0:
                continue
            result = {**self.metadata[idx], "score": float(distances[0][j])}
            results.append(result)

        return results

    def save_index(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info("Index saved to %s (%d vectors)", path, self.index.ntotal)

    def load_index(self, path: str | Path) -> None:
        path = Path(path)

        self.index = faiss.read_index(str(path / "index.faiss"))
        self.embedding_dim = self.index.d

        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info("Index loaded from %s (%d vectors)", path, self.index.ntotal)

    def clear(self) -> None:
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []

    def get_stats(self) -> dict:
        return {
            "num_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "num_metadata": len(self.metadata),
        }
