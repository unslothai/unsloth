# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Compatibility contract for GET /kv-cache-estimate.

The route already shipped, so an existing caller must keep working across an
upgrade. Two directions matter:

* An OLD client still sends n_ctx and reads kv_bytes / weights_bytes /
  native_context. Making n_ctx optional is a widening, and the new spec_bytes
  and n_ctx fields are additions, so nothing it relies on may move.

* A NEW client may omit n_ctx to ask for the model's native length. That is the
  only request shape the previous version would have rejected, so it is the one
  worth pinning.

No GPU, no network.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from test_kv_cache_estimation import _make_gguf_bytes  # noqa: E402

import routes.models as models_routes  # noqa: E402

_FIELDS = {
    "context_length": 8192,
    "block_count": 32,
    "attention.head_count": 32,
    "attention.head_count_kv": 8,
    "embedding_length": 4096,
    "attention.key_length": 128,
    "attention.value_length": 128,
}

# What a client written against the previous version reads back.
_LEGACY_KEYS = {"kv_bytes", "weights_bytes", "native_context"}


def _gguf(tmp_path: Path) -> Path:
    kv = {"general.architecture": "llama"}
    for k, v in _FIELDS.items():
        kv[f"llama.{k}"] = v
    p = tmp_path / "model-Q4_K_M.gguf"
    p.write_bytes(_make_gguf_bytes("llama", kv))
    return p


def _call(monkeypatch, path: Path | None, **overrides):
    # path=None leaves whatever the caller already patched in place, for the
    # cases that are about the route's answer when nothing resolves.
    if path is not None:
        monkeypatch.setattr(
            models_routes,
            "_resolve_quant_gguf",
            lambda *_a, **_k: (str(path), 4_000_000_000),
        )
    kwargs = dict(
        repo_id = "org/repo",
        quant = "Q4_K_M",
        n_ctx = 4096,
        cache_type_kv = None,
        n_parallel = None,
        speculative_type = None,
        spec_draft_n_max = None,
        spec_draft_cache_type = None,
        ctx_checkpoints = None,
        disable_vision = False,
        n_batch = None,
        n_ubatch = None,
        tensor_parallel = False,
        request = None,
        current_subject = "test",
    )
    kwargs.update(overrides)
    return asyncio.run(models_routes.get_kv_cache_estimate(**kwargs))


def test_every_parameter_an_old_caller_sent_is_still_accepted(tmp_path):
    """Signature check: nothing an existing client passes may have been removed
    or made required in a way it does not satisfy."""
    sig = inspect.signature(models_routes.get_kv_cache_estimate)
    for name in ("repo_id", "quant", "n_ctx", "cache_type_kv"):
        assert name in sig.parameters, f"{name} was removed from the route"
    # The new ones must be optional, or an old caller's request 422s.
    for name in ("n_parallel", "speculative_type"):
        assert sig.parameters[name].default is not inspect.Parameter.empty


def test_an_old_callers_request_still_answers_the_old_keys(monkeypatch, tmp_path):
    out = _call(monkeypatch, _gguf(tmp_path), n_ctx = 4096)
    assert _LEGACY_KEYS <= set(out), f"missing legacy keys: {_LEGACY_KEYS - set(out)}"
    assert out["kv_bytes"] and out["kv_bytes"] > 0
    assert out["weights_bytes"] == 4_000_000_000
    assert out["native_context"] == 8192


def test_the_answer_for_a_pinned_context_did_not_move(monkeypatch, tmp_path):
    """The added parameters default to what the previous version implied, so an
    unchanged request must produce an unchanged number."""
    gguf = _gguf(tmp_path)
    before = _call(monkeypatch, gguf, n_ctx = 4096)
    # Exactly what an old client sends: no n_parallel, no speculative_type.
    after = _call(monkeypatch, gguf, n_ctx = 4096, n_parallel = None, speculative_type = None)
    assert before["kv_bytes"] == after["kv_bytes"]


def test_omitting_the_context_sizes_at_the_models_native_length(monkeypatch, tmp_path):
    """The new shape: no n_ctx. The response says which length it used."""
    gguf = _gguf(tmp_path)
    native = _call(monkeypatch, gguf, n_ctx = None)
    assert native["n_ctx"] == 8192
    assert native["native_context"] == 8192
    explicit = _call(monkeypatch, gguf, n_ctx = 8192)
    assert native["kv_bytes"] == explicit["kv_bytes"]


def test_the_failure_answer_carries_every_key(monkeypatch):
    """A row that cannot be sized still has to be readable by both clients,
    rather than arriving as a short dict that KeyErrors in the caller."""
    monkeypatch.setattr(models_routes, "_resolve_quant_gguf", lambda *_a, **_k: (None, 0))
    out = _call(monkeypatch, None)
    assert _LEGACY_KEYS <= set(out)
    assert {"spec_bytes", "n_ctx", "projector_bytes", "spec_unpriced"} <= set(out)
    # Every byte figure is absent. spec_unpriced is a flag, not a measurement:
    # False is its correct value here, since nothing was left unpriced.
    assert all(v is None for k, v in out.items() if k != "spec_unpriced")
    assert out["spec_unpriced"] is False


def test_speculative_modes_that_cost_nothing_report_none(monkeypatch, tmp_path):
    gguf = _gguf(tmp_path)
    for mode in (None, "", "off", "ngram"):
        out = _call(monkeypatch, gguf, speculative_type = mode)
        assert out["spec_bytes"] is None, f"{mode!r} reserved memory"
