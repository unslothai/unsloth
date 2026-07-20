# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Effective-offline handling for the embedding preflight.

``huggingface_hub`` honors only ``HF_HUB_OFFLINE``; ``TRANSFORMERS_OFFLINE`` expresses the
same intent but does not stop a fetch. Studio treats either as offline (``hf_env_offline``)
and makes it real by passing ``local_files_only`` to the loader, which lets the Hub security
scan skip to its fail-open instead of burning both timeouts. That skip is sound only while
the loader is pinned to the local cache, so the coupling is pinned here as an invariant.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _maybe_stub(name: str, builder):
    # Stub only if the real module is unavailable, so this file never shadows real packages.
    try:
        importlib.import_module(name)
    except ImportError:
        sys.modules[name] = builder()


def _build_loggers_stub():
    m = types.ModuleType("loggers")
    m.get_logger = lambda name: __import__("logging").getLogger(name)
    return m


def _build_structlog_stub():
    m = types.ModuleType("structlog")
    m.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    return m


_maybe_stub("loggers", _build_loggers_stub)
_maybe_stub("structlog", _build_structlog_stub)

from utils.utils import hf_env_offline  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    yield


# ── flag semantics ──


def test_neither_flag_is_online():
    assert hf_env_offline() is False


@pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_either_flag_means_offline(monkeypatch, var):
    monkeypatch.setenv(var, "1")
    assert hf_env_offline() is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "  1  "])
def test_truthy_values_parse(monkeypatch, value):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert hf_env_offline() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_non_truthy_values_do_not_parse(monkeypatch, value):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert hf_env_offline() is False


# ── the offline security short-circuit, and the invariant it rests on ──


def _fake_hub(monkeypatch, calls):
    fake = types.ModuleType("huggingface_hub")

    def _model_info(*a, **k):
        calls.append(1)
        raise RuntimeError("network unreachable")

    fake.model_info = _model_info
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


@pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_shared_gate_still_scans_when_offline_by_default(monkeypatch, var):
    # Shared gate: an offline env var alone must never disable the scan for a fetching loader.
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    monkeypatch.setenv(var, "1")
    assert fs._fetch_security_status("org/model", None) is None
    assert calls, f"{var} alone must NOT bypass the shared malware gate"


def test_local_only_load_fails_closed_offline_never_hitting_the_hub(monkeypatch, tmp_path):
    # A local-only (offline) load never hits the Hub; it is evaluated fail-CLOSED against the
    # cached files: a pickle-free (safetensors) cache is allowed, a cached pickle is blocked.
    import utils.models.model_config as mc
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)

    def _evaluate(dir_path):
        monkeypatch.setattr(mc, "_active_snapshot_dir", lambda name: dir_path)
        decision = fs.evaluate_file_security("org/model", None, local_only_load = True)
        assert calls == [], "a local-only load must not hit the Hub"
        return decision.blocked

    def _snap(name, files: dict):
        d = tmp_path / name / "aaa"
        d.mkdir(parents = True)
        for fname, body in files.items():
            (d / fname).write_bytes(body if isinstance(body, bytes) else body.encode())
        return d

    # A genuinely inert (unsharded safetensors) cache is allowed.
    assert _evaluate(_snap("safe", {"model.safetensors": b"\0"})) is False

    # A cached pickle weight with no loadable safetensors alternative is blocked.
    assert _evaluate(_snap("bad", {"pytorch_model.bin": b"\0"})) is True

    # A pickle beside a bare ADAPTER (not a base weight) still loads the pickle -> blocked:
    # from_pretrained cannot use adapter_model.safetensors as the base checkpoint.
    assert _evaluate(_snap("adapter", {
        "pytorch_model.bin": b"\0", "adapter_model.safetensors": b"\0",
    })) is True

    # A pickle beside a lone ORPHAN shard (no index) still loads the pickle -> blocked:
    # a sharded safetensors load needs the index the loader reads to locate every shard.
    assert _evaluate(_snap("orphan", {
        "pytorch_model.bin": b"\0", "model-00001-of-00002.safetensors": b"\0",
    })) is True

    # A COMPLETE indexed safetensors shard set is what the loader picks instead of the
    # pickle -> allowed.
    index = '{"weight_map": {"a": "model-00001-of-00002.safetensors", ' \
            '"b": "model-00002-of-00002.safetensors"}}'
    assert _evaluate(_snap("sharded", {
        "pytorch_model.bin": b"\0",
        "model-00001-of-00002.safetensors": b"\0",
        "model-00002-of-00002.safetensors": b"\0",
        "model.safetensors.index.json": index,
    })) is False


def test_security_scan_runs_when_online(monkeypatch):
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    assert fs._fetch_security_status("org/model", None) is None
    assert calls, "online must attempt the Hub"


def _read_backend(rel: str) -> str:
    return (Path(__file__).resolve().parents[1] / rel).read_text(encoding = "utf-8")


def test_embedding_loader_forces_local_only_when_offline():
    """The invariant the RAG opt-in rests on: embeddings.py may pass local_only_load
    because its loader is pinned to the local cache by the SAME value, read ONCE and shared
    (two hf_env_offline() reads can disagree since _hf_offline_if_dns_dead() flips the vars).
    Checked at source level because importing the loader drags in sentence_transformers.
    """
    src = _read_backend("core/rag/embeddings.py")
    assert (
        "local_only = hf_env_offline()" in src
    ), "core/rag/embeddings.py must capture the offline state once in _get()"
    assert (
        "_guard_model_security(load_name, local_only)" in src
    ), "the security guard must receive the captured value, not re-read the env"
    assert "local_files_only = local_only" in src, (
        "SentenceTransformer must be pinned with the SAME captured value; a second "
        "hf_env_offline() read can flip to False and fetch the unscanned repo"
    )
    guard = src.split("def _guard_model_security", 1)[1].split("\ndef ", 1)[0]
    assert "hf_env_offline()" not in guard, (
        "_guard_model_security must take local_only_load as an argument so it "
        "cannot observe a different offline state than the loader"
    )


def test_only_the_rag_embedding_path_opts_into_the_bypass():
    """No other loader may claim local-only without constraining its loader: the
    MLX/inference, training and export gates call from_pretrained without a local-only
    argument, so passing local_only_load there would disable the gate for a fetching path.
    """
    allowed = {"core/rag/embeddings.py", "routes/settings.py"}
    callers = [
        "core/inference/worker.py",
        "core/training/worker.py",
        "core/export/worker.py",
        "routes/models.py",
        "routes/inference.py",
    ]
    for rel in callers:
        assert rel not in allowed
        assert "local_only_load" not in _read_backend(rel), (
            f"{rel} passes local_only_load but does not pin its loader to the "
            "local cache; that would disable the malware gate for a fetching path"
        )
