# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The provenance attester and the worker's revalidation need the load subdirs too.

``ca7c72e75`` taught three sites about subdirectory-loading repos. An audit of the full
backend suite found only one of them was detectable: reverting the subdir expansion in
``core/training/provenance.py`` or in ``core/training/worker.py`` left all 17,204 passing
tests green, with a byte-identical failure set. Both were shipped unguarded.

They are not decorative. For ``unsloth/Spark-TTS-0.5B`` -- snapshot root holds only
``README.md`` and ``config.yaml``, everything trainable under ``LLM/`` -- the provenance
site turns a snapshot sitting on disk into "The exact model snapshot for this run is no
longer available." and refuses the resume, and the worker site either errors with "The
cached model snapshot selected during preflight is no longer available." or silently
drops the pin and goes back to the Hub.
"""

import json

import pytest


_BICODEC = "unsloth/Spark-TTS-0.5B"
_PLAIN = "unsloth/Llama-3.2-1B-Instruct"
_REVISION = "d" * 40


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    from hub.utils import hf_cache_state

    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda **kw: [root])
    return root


@pytest.fixture
def bicodec_subdirs(monkeypatch):
    import utils.security as security_pkg
    monkeypatch.setattr(
        security_pkg,
        "security_load_subdirs",
        lambda model_name, hf_token = None, local_files_only = False: (
            ("LLM",) if model_name == _BICODEC else ()
        ),
    )


def _snapshot(
    cache_root,
    repo_id,
    revision = _REVISION,
):
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents = True)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text(revision, encoding = "utf-8")
    return snapshot


def _write_model(directory):
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (directory / "model.safetensors").write_bytes(b"\x00" * 512)


def _bicodec_snapshot(cache_root):
    snapshot = _snapshot(cache_root, _BICODEC)
    # The real layout: nothing loadable at the snapshot root.
    (snapshot / "config.yaml").write_text("sample_rate: 16000\n")
    _write_model(snapshot / "LLM")
    return snapshot


def test_the_attester_accepts_a_subdir_snapshot(cache_root, bicodec_subdirs):
    """provenance.py: without the expansion this returns None and resume is refused."""
    from core.training.provenance import exact_model_snapshot_path

    snapshot = _bicodec_snapshot(cache_root)

    assert exact_model_snapshot_path(str(snapshot), _BICODEC) == str(snapshot)


def test_the_attester_is_unchanged_for_a_root_loading_snapshot(cache_root, bicodec_subdirs):
    from core.training.provenance import exact_model_snapshot_path

    snapshot = _snapshot(cache_root, _PLAIN)
    _write_model(snapshot)

    assert exact_model_snapshot_path(str(snapshot), _PLAIN) == str(snapshot)


def test_the_attester_still_rejects_a_snapshot_with_nothing_loadable(cache_root, bicodec_subdirs):
    """The widening must not turn an empty cache into a false positive."""
    from core.training.provenance import exact_model_snapshot_path

    snapshot = _snapshot(cache_root, _BICODEC)
    (snapshot / "config.yaml").write_text("sample_rate: 16000\n")
    (snapshot / "LLM").mkdir()

    assert exact_model_snapshot_path(str(snapshot), _BICODEC) is None


def test_the_resume_gate_allows_a_subdir_snapshot(cache_root, bicodec_subdirs):
    """The user-visible end of the same site: no spurious refusal message."""
    from core.training.provenance import (
        RESOURCE_PROVENANCE_KEY,
        resource_provenance_resume_blocker,
    )

    snapshot = _bicodec_snapshot(cache_root)
    config = {
        "model_name": _BICODEC,
        "model_snapshot_path": str(snapshot),
        "model_revision": _REVISION,
        RESOURCE_PROVENANCE_KEY: {
            "version": 1,
            "status": "complete",
            "model_status": "attested",
            "model_repo_id": _BICODEC,
            "model_revision": _REVISION,
        },
    }

    blocker = resource_provenance_resume_blocker(config)

    assert (
        blocker is None or "no longer available" not in blocker
    ), f"a snapshot present on disk was reported as gone: {blocker!r}"


def test_the_worker_keeps_a_subdir_pin_under_strict_resume(cache_root, bicodec_subdirs):
    """worker.py: strict resume must not error out on a cache that is present."""
    import queue

    from core.training.worker import _verify_config_pins

    snapshot = _bicodec_snapshot(cache_root)
    events: queue.Queue = queue.Queue()
    config = {
        "model_name": _BICODEC,
        "model_snapshot_path": str(snapshot),
        "model_revision": _REVISION,
        "require_exact_model_resource": True,
    }

    ok = _verify_config_pins(config, events)

    assert ok is True, (
        "strict resume rejected a cached subdir snapshot; the user sees "
        "'The cached model snapshot selected during preflight is no longer available.'"
    )
    assert config["model_snapshot_path"] == str(snapshot), "the pin was dropped"


def test_the_worker_keeps_a_subdir_pin_without_strict_resume(cache_root, bicodec_subdirs):
    """The non-strict branch is the quieter failure: the pin just disappears."""
    import queue

    from core.training.worker import _verify_config_pins

    snapshot = _bicodec_snapshot(cache_root)
    events: queue.Queue = queue.Queue()
    config = {
        "model_name": _BICODEC,
        "model_snapshot_path": str(snapshot),
        "model_revision": _REVISION,
    }

    assert _verify_config_pins(config, events) is True
    assert config.get("model_snapshot_path") == str(snapshot), (
        "the pin was silently dropped, so the load goes back to the Hub instead of the "
        "snapshot the user selected"
    )
