import threading
from typing import Iterable, List, Optional


class EmbeddingUnavailableError(RuntimeError):
    pass


class FastEmbedEngine:
    """Lazy CPU/local embedding adapter; model weights live in the provider's shared model cache."""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from fastembed import TextEmbedding
                except ImportError as exc:
                    raise EmbeddingUnavailableError(
                        "FastEmbed is not installed. Install the 'fastembed' package or disable retrieval."
                    ) from exc
                options = {"model_name": self.model_name}
                if self.cache_dir:
                    options["cache_dir"] = self.cache_dir
                try:
                    self._model = TextEmbedding(**options)
                except Exception as exc:
                    raise EmbeddingUnavailableError(
                        f"Unable to initialize the local embedding model ({type(exc).__name__})."
                    ) from exc
        return self._model

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        values = list(texts)
        if not values:
            return []
        try:
            return [list(map(float, vector)) for vector in self._load_model().embed(values)]
        except EmbeddingUnavailableError:
            raise
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"Local embedding generation failed ({type(exc).__name__})."
            ) from exc
