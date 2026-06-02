"""RAG embedder compute-lock wiring (no GPU / real model needed)."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))


class _FakeModel:
    """Records whether the shared compute lock was held during each call."""

    def __init__(self, lock) -> None:
        self._lock = lock
        self.encode_held: bool | None = None
        self.tokenize_held: bool | None = None

    def encode(self, texts, **_kw):
        self.encode_held = self._lock.locked()
        return np.zeros((len(texts), 3), dtype = "float32")

    def tokenize(self, _texts):
        self.tokenize_held = self._lock.locked()
        return {"input_ids": np.zeros((1, 5), dtype = "int64")}


def test_encode_holds_compute_lock_then_releases(monkeypatch):
    from core.rag import embeddings

    fake = _FakeModel(embeddings._compute_lock)
    monkeypatch.setattr(embeddings, "get_embedder", lambda *a, **k: fake)

    out = embeddings.encode(["hello"])

    assert fake.encode_held is True
    assert embeddings._compute_lock.locked() is False
    assert out.shape == (1, 3)


def test_token_counter_holds_compute_lock_then_releases(monkeypatch):
    from core.rag import embeddings

    fake = _FakeModel(embeddings._compute_lock)
    monkeypatch.setattr(embeddings, "get_embedder", lambda *a, **k: fake)

    counter = embeddings.token_counter()
    n = counter("hello world")

    assert fake.tokenize_held is True
    assert n == 5
    assert embeddings._compute_lock.locked() is False
