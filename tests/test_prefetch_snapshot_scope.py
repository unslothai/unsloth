"""Pure-CPU, no-network unit tests for the prefetch snapshot scoping in
unsloth/models/_utils.py.

maybe_prefetch_hf_snapshot warms the HF cache before the in-process load so the load is a cache
hit and cannot hang on a stalled Xet transfer. The warm must download AT LEAST what the load
reads (else the missing file falls to an unprotected in-process Xet fetch) but should not pull
weights the load never reads. These tests lock the allow_patterns / ignore_patterns each mode
hands snapshot_download_with_xet_fallback (Codex #6638: adapter-only, weights-at-root, subfolder).
No network, no subprocess: the zoo downloader is monkeypatched to capture its kwargs.
"""

import fnmatch
import sys
import types

import pytest

from unsloth.models import _utils as U


def _filter(names, allow_patterns, ignore_patterns):
    """Mirror Hugging Face filter_repo_objects: keep a name if it matches any allow pattern
    (or allow is None), then drop it if it matches any ignore pattern. fnmatch '*' spans '/'
    exactly as HF's matcher does, so this reproduces the real selection over a sample file list."""
    kept = []
    for name in names:
        if allow_patterns is not None and not any(fnmatch.fnmatch(name, p) for p in allow_patterns):
            continue
        if ignore_patterns and any(fnmatch.fnmatch(name, p) for p in ignore_patterns):
            continue
        kept.append(name)
    return kept


@pytest.fixture
def capture(monkeypatch):
    """Call maybe_prefetch_hf_snapshot with a fake repo id and capture the allow/ignore patterns
    it forwards to the zoo downloader. A fake unsloth_zoo.hf_xet_fallback module is injected into
    sys.modules so the test is independent of the installed unsloth_zoo version (the published
    package may predate the helper, which maybe_prefetch_hf_snapshot then imports lazily). Offline
    env vars are cleared so the warm is not skipped."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)

    state = {}

    def fake_download(repo_id, **kw):
        state["repo_id"] = repo_id
        state["allow_patterns"] = kw.get("allow_patterns")
        state["ignore_patterns"] = kw.get("ignore_patterns")
        return "/tmp/fake-snapshot"

    fake_module = types.ModuleType("unsloth_zoo.hf_xet_fallback")
    fake_module.snapshot_download_with_xet_fallback = fake_download
    fake_module.DownloadStallError = type("DownloadStallError", (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, "unsloth_zoo.hf_xet_fallback", fake_module)

    def run(**call_kwargs):
        state.clear()
        ok = U.maybe_prefetch_hf_snapshot("some-org/some-repo", **call_kwargs)
        return ok, state

    return run


# A representative repo file listing: root weights + tokenizer/config, plus an alternate-precision
# subdir, an adapter, a checkpoint dir, and merged full-model weights an adapter repo might ship.
_SAMPLE_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "fp16/model.safetensors",
    "experimental/model-00001-of-00002.safetensors",
    "checkpoint-500/model.safetensors",
    "adapter_config.json",
    "adapter_model.safetensors",
]


def test_weights_at_root_excludes_subdir_weights(capture):
    """A bare root load reads only root weight files, so weights nested in subdirs (fp16/,
    experimental/, checkpoint-500/) must be ignored while the root weights stay warmed. An
    explicit use_safetensors avoids the auto branch's model_info network call."""
    ok, st = capture(weights_at_root = True, use_safetensors = True)
    assert ok is True
    assert st["allow_patterns"] is None          # the warm stays otherwise unfiltered
    ig = st["ignore_patterns"]
    assert "*/*.safetensors" in ig and "*/*.bin" in ig
    kept = _filter(_SAMPLE_FILES, st["allow_patterns"], ig)
    # Root weights + config/tokenizer survive; subdir weights are dropped.
    assert "model-00001-of-00002.safetensors" in kept
    assert "model.safetensors.index.json" in kept
    assert "config.json" in kept
    assert "fp16/model.safetensors" not in kept
    assert "experimental/model-00001-of-00002.safetensors" not in kept
    assert "checkpoint-500/model.safetensors" not in kept


def test_adapter_only_excludes_merged_weights(capture):
    """An adapter warm reads only adapter_config.json + adapter_model.* (plus root tokenizer /
    config); a repo that also ships merged full-model weights must not pull them."""
    ok, st = capture(adapter_only = True)
    assert ok is True
    assert st["ignore_patterns"] is None          # the exact allowlist makes the format filter moot
    allow = st["allow_patterns"]
    assert "adapter_config.json" in allow and "adapter_model*" in allow
    kept = _filter(_SAMPLE_FILES, allow, st["ignore_patterns"])
    # The adapter's own files + the root aux files are warmed.
    assert "adapter_config.json" in kept
    assert "adapter_model.safetensors" in kept
    assert "config.json" in kept and "tokenizer.json" in kept
    # The merged / full-model weights are NOT pulled.
    assert "model-00001-of-00002.safetensors" not in kept
    assert "pytorch_model.bin" not in kept
    assert "fp16/model.safetensors" not in kept


def test_adapter_only_warms_sharded_adapter(capture):
    """A sharded adapter (adapter_model-00001-of-00002.safetensors) is still covered by the
    adapter_model* glob, so a large adapter is not left to an in-process Xet fetch."""
    _, st = capture(adapter_only = True)
    sharded = [
        "adapter_config.json",
        "adapter_model-00001-of-00002.safetensors",
        "adapter_model-00002-of-00002.safetensors",
        "adapter_model.safetensors.index.json",
    ]
    kept = _filter(sharded, st["allow_patterns"], st["ignore_patterns"])
    assert set(kept) == set(sharded)


def test_tokenizer_only_warms_only_aux_files(capture):
    """A distinct tokenizer repo warms only its tokenizer / config / vocab files, never weights."""
    _, st = capture(tokenizer_only = True)
    assert st["ignore_patterns"] is None
    assert st["allow_patterns"] == list(U._ROOT_AUX_PREFETCH_PATTERNS)
    kept = _filter(_SAMPLE_FILES, st["allow_patterns"], st["ignore_patterns"])
    assert "tokenizer.json" in kept and "config.json" in kept
    assert "model-00001-of-00002.safetensors" not in kept
    assert "adapter_model.safetensors" not in kept


def test_subfolder_warms_subfolder_plus_root_aux(capture):
    """A subfolder load warms that subfolder's weights plus the root tokenizer / config; the
    root weights and OTHER subfolders are skipped."""
    _, st = capture(subfolder = "fp16")
    allow = st["allow_patterns"]
    assert "fp16/*" in allow
    assert all(p in allow for p in U._ROOT_AUX_PREFETCH_PATTERNS)
    kept = _filter(_SAMPLE_FILES, allow, st["ignore_patterns"])
    assert "fp16/model.safetensors" in kept
    assert "config.json" in kept
    assert "experimental/model-00001-of-00002.safetensors" not in kept


def test_subfolder_takes_precedence_over_weights_at_root(capture):
    """weights_at_root is a root-load assertion; when a subfolder IS requested the subfolder
    branch wins (the load reads that subfolder), so the warm is the subfolder, not a
    root-with-subdir-weights-excluded warm."""
    _, st = capture(subfolder = "fp16", weights_at_root = True)
    assert "fp16/*" in st["allow_patterns"]
    kept = _filter(_SAMPLE_FILES, st["allow_patterns"], st["ignore_patterns"])
    assert "fp16/model.safetensors" in kept


def test_local_dir_is_not_warmed(capture, tmp_path):
    """A local directory path has nothing to download: the warm is skipped (returns False)."""
    d = tmp_path / "local-model"
    d.mkdir()
    ok = U.maybe_prefetch_hf_snapshot(str(d), weights_at_root = True)
    assert ok is False
