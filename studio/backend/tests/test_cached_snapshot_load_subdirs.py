# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cached snapshot that loads from a subdirectory must still resolve.

``unsloth/Spark-TTS-0.5B`` keeps everything trainable under ``LLM/``; its snapshot root
holds only ``README.md`` and ``config.yaml`` (verified against the Hub file listing), so a
resolver that insists on a root-level ``config.json`` plus root-level weights finds
nothing. The remote preflight already expands those load roots through
``load_scan_target``, and ``security_load_subdirs`` reports ``("LLM",)`` for BiCodec, so
the cached path has to agree or a perfectly good cache is reported as absent:
``_apply_model_cache_pin`` warns "not found on disk; downloading" and, offline, the start
route turns the same ``None`` into a 409 ``hf_model_not_cached_offline``.
"""

import json

import pytest

from core.training import training as training_mod


_REPO = "unsloth/Spark-TTS-0.5B"
_PLAIN_REPO = "unsloth/Llama-3.2-1B-Instruct"


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """A tmp dir registered as an HF cache root, as validated_repo_cache_path requires."""
    from hub.utils import hf_cache_state

    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda **kw: [root])
    return root


def _snapshot(
    cache_root,
    repo_id: str,
    revision: str = "a" * 40,
):
    """Build a real models--org--name/snapshots/<rev> cache layout."""
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents = True)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text(revision, encoding = "utf-8")
    return repo_dir, snapshot


def _write_model(directory, *, weights: bool = True):
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    if weights:
        (directory / "model.safetensors").write_bytes(b"\x00" * 512)


@pytest.fixture
def bicodec_subdirs(monkeypatch):
    """Report LLM/ for the BiCodec repo without touching the network."""
    import utils.security as security_pkg

    def fake_subdirs(model_name, hf_token = None, local_files_only = False):
        return ("LLM",) if model_name == _REPO else ()

    monkeypatch.setattr(security_pkg, "security_load_subdirs", fake_subdirs)
    return fake_subdirs


def test_a_cached_bicodec_snapshot_resolves_from_its_llm_load_root(cache_root, bicodec_subdirs):
    _, snapshot = _snapshot(cache_root, _REPO)
    # Exactly the real layout: nothing loadable at the root, everything under LLM/.
    (snapshot / "config.yaml").write_text("sample_rate: 16000\n")
    _write_model(snapshot / "LLM")

    resolved = training_mod._resolve_model_snapshot(_REPO, str(snapshot))

    assert resolved is not None, (
        "a cached Spark-TTS snapshot read as absent: the start route turns this None "
        "into a 409 hf_model_not_cached_offline"
    )
    assert str(snapshot) == resolved


def test_the_pin_helper_expands_only_for_a_subdir_loading_repo(bicodec_subdirs):
    names = ("config.json",)

    assert training_mod._with_load_subdirs(_REPO, names) == ("config.json", "LLM/config.json")
    assert training_mod._with_load_subdirs(_PLAIN_REPO, names) == names


def test_an_ordinary_root_loading_snapshot_is_unaffected(cache_root, bicodec_subdirs):
    _, snapshot = _snapshot(cache_root, _PLAIN_REPO)
    _write_model(snapshot)

    assert training_mod._resolve_model_snapshot(_PLAIN_REPO, str(snapshot)) == str(snapshot)


def test_a_snapshot_with_neither_root_nor_subdir_weights_still_fails(cache_root, bicodec_subdirs):
    """The widening must not turn "nothing usable here" into a false positive."""
    _, snapshot = _snapshot(cache_root, _REPO)
    (snapshot / "config.yaml").write_text("sample_rate: 16000\n")
    (snapshot / "LLM").mkdir()

    assert training_mod._resolve_model_snapshot(_REPO, str(snapshot)) is None


def test_a_subdir_snapshot_survives_the_metadata_only_second_pass(cache_root, bicodec_subdirs):
    """Pass 2 keeps caches that never held weights resolvable; subdirs count there too."""
    _, snapshot = _snapshot(cache_root, _REPO)
    _write_model(snapshot / "LLM", weights = False)

    assert training_mod._resolve_model_snapshot(_REPO, str(snapshot)) == str(snapshot)


def test_load_subdir_lookup_failure_degrades_to_root_only(cache_root, monkeypatch):
    """Detection can raise offline or for a gated repo; that must not break resolution."""
    import utils.security as security_pkg

    def boom(model_name, hf_token = None, local_files_only = False):
        raise RuntimeError("hub unreachable")

    monkeypatch.setattr(security_pkg, "security_load_subdirs", boom)

    assert training_mod._with_load_subdirs(_REPO, ("config.json",)) == ("config.json",)

    _, snapshot = _snapshot(cache_root, _PLAIN_REPO)
    _write_model(snapshot)
    assert training_mod._resolve_model_snapshot(_PLAIN_REPO, str(snapshot)) == str(snapshot)
