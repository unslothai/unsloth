# SPDX-License-Identifier: AGPL-3.0-only
"""The `load_in_fp8` artifact is reused by name and loaded with the caller's trust_remote_code,
so on a shared /tmp another account could plant it first. Nothing is downloaded or loaded."""

from __future__ import annotations

import hashlib
import os
import types

import pytest

loader_utils = pytest.importorskip("unsloth.models.loader_utils")

posix_only = pytest.mark.skipif(not hasattr(os, "getuid"), reason = "POSIX ownership")


class _Quantizing(Exception):
    pass


@pytest.fixture
def shared_tmp(monkeypatch, tmp_path):
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: types.SimpleNamespace(architectures = ["LlamaForCausalLM"])),
    )

    def reached(*a, **k):
        raise _Quantizing

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", reached)
    monkeypatch.setattr(loader_utils, "_get_torchao_fp8_config", lambda mode: None)
    monkeypatch.setattr(transformers, "TorchAoConfig", lambda cfg: None)
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setattr(loader_utils.tempfile, "gettempdir", lambda: str(shared))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    return shared


def _quantize(name = "org/model"):
    try:
        return loader_utils._offline_quantize_to_fp8(name, "row")
    except _Quantizing:
        return None


def test_a_directory_planted_at_the_shared_name_is_not_loaded(shared_tmp):
    name = "org/model"
    planted = shared_tmp / (
        "model-fp8-row-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    )
    planted.mkdir()
    (planted / "config.json").write_text('{"auto_map": {"AutoModelForCausalLM": "evil.Model"}}')
    result = _quantize(name)
    assert result is None or os.path.realpath(result) != os.path.realpath(planted)


@posix_only
def test_the_cache_root_is_private(shared_tmp):
    _quantize()
    root = loader_utils._private_fp8_cache_dir()
    info = os.lstat(root)
    assert info.st_uid == os.getuid()
    assert info.st_mode & 0o777 == 0o700


@posix_only
def test_a_loose_root_of_ours_is_tightened(shared_tmp):
    root = shared_tmp / ("unsloth-fp8-%d" % os.getuid())
    root.mkdir()
    os.chmod(root, 0o777)
    loader_utils._private_fp8_cache_dir()
    assert os.lstat(root).st_mode & 0o777 == 0o700


@posix_only
def test_a_root_owned_by_another_account_is_refused(shared_tmp, monkeypatch):
    other = os.getuid() + 1
    (shared_tmp / ("unsloth-fp8-%d" % other)).mkdir(mode = 0o700)
    monkeypatch.setattr(loader_utils.os, "getuid", lambda: other)
    with pytest.raises(RuntimeError, match = "symlink or owned"):
        _quantize()


@posix_only
def test_a_symlinked_root_is_refused(shared_tmp):
    target = shared_tmp / "elsewhere"
    target.mkdir(mode = 0o700)
    os.symlink(target, shared_tmp / ("unsloth-fp8-%d" % os.getuid()))
    with pytest.raises(RuntimeError, match = "symlink or owned"):
        _quantize()


@posix_only
def test_a_writable_artifact_inside_our_root_is_refused(shared_tmp):
    _quantize()
    root = loader_utils._private_fp8_cache_dir()
    name = "org/model"
    artifact = os.path.join(
        root, "model-fp8-row-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    )
    os.mkdir(artifact)
    os.chmod(artifact, 0o777)
    with pytest.raises(RuntimeError, match = "world-writable"):
        _quantize(name)


def test_our_own_artifact_is_still_reused(shared_tmp):
    root = loader_utils._private_fp8_cache_dir()
    name = "org/model"
    artifact = os.path.join(
        root, "model-fp8-row-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    )
    os.mkdir(artifact, 0o700)
    first = _quantize(name)
    assert first == artifact


def test_without_posix_owners_the_root_is_not_in_a_shared_temp(shared_tmp, monkeypatch):
    # TEMP can be shared on Windows and there is no owner check, so a name derived from
    # USERNAME there could be pre-created by another account.
    monkeypatch.delattr(loader_utils.os, "getuid", raising = False)
    monkeypatch.setenv("USERNAME", "victim")
    (shared_tmp / "unsloth-fp8-victim").mkdir()
    root = os.path.realpath(loader_utils._private_fp8_cache_dir())
    assert not root.startswith(os.path.realpath(shared_tmp))
    assert root.startswith(os.path.realpath(os.environ["LOCALAPPDATA"]))
