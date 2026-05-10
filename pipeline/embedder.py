"""Arabic text embedding pipeline using Nomic Embed Text v2 MoE.

nomic-embed-text-v2-moe produces 768-dim embeddings and supports
task-specific prefixes via prompt_name ("query" vs "passage").
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


class ArabicEmbedder:
    """Sentence embedding model using Nomic Embed Text v2 MoE."""

    def __init__(self, device: str | None = None):
        """
        Args:
            device: torch device string; auto-detected if None
        """
        self.device = device or config.DEVICE
        self.model = None
        self._loaded = False

    def _load_model(self) -> None:
        model_id = config.EMBEDDING_MODEL_ID
        logger.info("Loading embedding model: %s", model_id)
        self.model = SentenceTransformer(
            model_id,
            trust_remote_code=True,
            device=self.device,
        )
        self._loaded = True
        logger.info(
            "Embedder loaded on %s (dim=%d)",
            self.device,
            self.model.get_sentence_embedding_dimension(),
        )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load_model()

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        """Embed a list of texts into dense vectors.

        Args:
            texts: list of strings to embed
            batch_size: encoding batch size
            show_progress: show progress bar
            is_query: if True, use "query" prefix; otherwise "passage" prefix.
                      Nomic uses search_query / search_document prefixes.

        Returns:
            numpy array of shape (len(texts), 768), L2-normalized.
        """
        self._ensure_loaded()

        prompt_name = "query" if is_query else "passage"

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
            prompt_name=prompt_name,
        )

        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Returns 1-D array of shape (768,)."""
        return self.embed_texts([text], show_progress=False, is_query=True)[0]

    def embed_documents(self, texts: list[str], **kwargs) -> np.ndarray:
        """Embed documents for indexing. Returns (n, 768) array."""
        return self.embed_texts(texts, is_query=False, **kwargs)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text as a document. Returns 1-D array."""
        return self.embed_texts([text], show_progress=False, is_query=False)[0]

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension of the loaded model."""
        self._ensure_loaded()
        return self.model.get_sentence_embedding_dimension()

    def unload(self) -> None:
        """Free model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        logger.info("Embedder model unloaded")
