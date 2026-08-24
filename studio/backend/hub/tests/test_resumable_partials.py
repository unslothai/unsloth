# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Restoring huggingface_hub's resumable HTTP partials.

The patched writer has to be a faithful 1.17 caller and nothing more: a stable name, opened for
append, told how far it got, and left alone on failure. It also has to stand down wherever it
cannot prove that is safe, which is the whole reason 1.18 removed it.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from hub.utils import resumable_partials as rp


@pytest.fixture(autouse = True)
def _fresh_probe():
    rp.reset_probe_cache_for_tests()
    yield
    rp.reset_probe_cache_for_tests()


def _fake_file_download(monkeypatch, *, xet_available = False):
    """A stand-in for huggingface_hub.file_download that records what the writer did."""
    calls = {"http_get": [], "stock": [], "moved": []}

    def http_get(
        url,
        handle,
        *,
        resume_size = 0,
        headers = None,
        expected_size = None,
        tqdm_class = None,
    ):
        calls["http_get"].append({"resume_size": resume_size, "mode": handle.mode})
        handle.write(b"x" * 10)

    def stock(**kwargs):
        calls["stock"].append(kwargs)

    module = types.ModuleType("huggingface_hub.file_download")
    module._download_to_tmp_and_move = stock
    module.http_get = http_get
    module._chmod_and_move = lambda src, dst: calls["moved"].append((src, dst))
    module._check_disk_space = lambda size, path: None
    module.is_xet_available = lambda: xet_available

    hub = types.ModuleType("huggingface_hub")
    hub.__version__ = "1.28.0"
    hub.file_download = module
    hub.constants = types.SimpleNamespace(HF_HUB_CACHE = None)

    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.file_download", module)
    monkeypatch.setattr(rp, "_lock_is_honoured", lambda: True)
    return module, calls


def _patched_writer(module):
    return module._download_to_tmp_and_move


# ---------------------------------------------------------------------------------------------
# When it engages
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version, expected",
    [("0.36.2", False), ("1.17.0", False), ("1.18.0", True), ("1.28.0", True), ("2.0.0", False)],
)
def test_only_the_versions_that_need_it_and_that_it_has_been_read_against(
    monkeypatch, version, expected
):
    module, _ = _fake_file_download(monkeypatch)
    sys.modules["huggingface_hub"].__version__ = version
    assert rp.can_restore_partials() is expected


def test_a_filesystem_that_grants_the_lock_twice_keeps_the_stock_writer(monkeypatch):
    _fake_file_download(monkeypatch)
    monkeypatch.setattr(rp, "_lock_is_honoured", lambda: False)
    assert rp.can_restore_partials() is False
    assert rp.restore_resumable_partials() is False


def test_a_hub_missing_the_pieces_is_left_alone(monkeypatch):
    module, _ = _fake_file_download(monkeypatch)
    del module._chmod_and_move
    assert rp.can_restore_partials() is False


def test_the_lock_probe_reports_a_working_lock(tmp_path, monkeypatch):
    """The real probe, on the real filesystem the tests run on."""
    monkeypatch.setattr(rp, "_probe_dir", lambda: tmp_path)
    assert rp._lock_is_honoured() is True
    assert not list(tmp_path.iterdir()), "the probe left its file behind"


def test_the_probe_follows_the_cache_studio_is_using_now(tmp_path, monkeypatch):
    """Not the one this process booted with.

    huggingface_hub resolves HF_HUB_CACHE at import and moving the cache in Settings does not
    rewrite the live process, so probing the constant would judge a different filesystem than the
    one a freshly spawned worker writes its partial to.
    """
    live = tmp_path / "moved-cache"
    fake = types.ModuleType("utils.hf_cache_settings")
    fake.active_hf_hub_cache = lambda: str(live)
    monkeypatch.setitem(sys.modules, "utils.hf_cache_settings", fake)

    assert rp._probe_dir() == live
    assert live.is_dir(), "the probe did not create the cache root"


def test_moving_the_cache_re_probes_rather_than_reusing_the_old_verdict(tmp_path, monkeypatch):
    """Two roots can sit on filesystems that disagree about flock, so the verdict is per root."""
    probed: list[str] = []
    monkeypatch.setattr(rp, "_lock_is_honoured_at", lambda directory: probed.append(directory) or True)

    first, second = tmp_path / "one", tmp_path / "two"
    monkeypatch.setattr(rp, "_probe_dir", lambda: first)
    rp._lock_is_honoured()
    monkeypatch.setattr(rp, "_probe_dir", lambda: second)
    rp._lock_is_honoured()

    assert probed == [str(first), str(second)]


def test_the_verdict_is_cached_per_directory(tmp_path):
    """The hot path asks per blob, so a repeat on the same root must not re-probe."""
    rp.invalidate_probe_cache()
    root = tmp_path / "cache"
    root.mkdir()
    assert rp._lock_is_honoured_at(str(root)) is True
    before = rp._lock_is_honoured_at.cache_info()
    assert rp._lock_is_honoured_at(str(root)) is True
    assert rp._lock_is_honoured_at.cache_info().hits == before.hits + 1


def test_changing_the_cache_home_invalidates_the_verdict(monkeypatch):
    """The one place the root can move at runtime must drop both cached answers."""
    from hub.utils import hf_cache_state

    monkeypatch.setattr(rp, "can_restore_partials", lambda: True)
    monkeypatch.setattr("huggingface_hub.__version__", "1.28.0", raising = False)
    hf_cache_state.hf_partials_are_resumable.cache_clear()
    assert hf_cache_state.hf_partials_are_resumable() is True

    monkeypatch.setattr(rp, "can_restore_partials", lambda: False)
    assert hf_cache_state.hf_partials_are_resumable() is True, "cached, as the hot path needs"

    hf_cache_state.invalidate_partial_resumability()
    assert hf_cache_state.hf_partials_are_resumable() is False
    hf_cache_state.hf_partials_are_resumable.cache_clear()


# ---------------------------------------------------------------------------------------------
# What it does once it has
# ---------------------------------------------------------------------------------------------


def test_it_appends_to_the_stable_name_and_says_how_far_it_got(monkeypatch, tmp_path):
    module, calls = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True

    partial = tmp_path / "abc.incomplete"
    partial.write_bytes(b"y" * 40)
    destination = tmp_path / "abc"

    _patched_writer(module)(
        incomplete_path = partial,
        destination_path = destination,
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )

    assert calls["http_get"] == [{"resume_size": 40, "mode": "ab"}]
    assert calls["moved"] == [(partial, destination)]
    assert calls["stock"] == []


def test_a_failed_download_leaves_the_partial_for_the_next_attempt(monkeypatch, tmp_path):
    module, _ = _fake_file_download(monkeypatch)
    rp.restore_resumable_partials()

    def boom(*_args, **_kwargs):
        raise OSError("connection reset")

    module.http_get = boom
    partial = tmp_path / "abc.incomplete"
    partial.write_bytes(b"y" * 40)

    with pytest.raises(OSError):
        _patched_writer(module)(
            incomplete_path = partial,
            destination_path = tmp_path / "abc",
            url_to_download = "https://example/f",
            headers = {},
            expected_size = 50,
            filename = "f",
        )
    assert partial.exists() and partial.stat().st_size == 40


def test_xet_keeps_its_own_writer(monkeypatch, tmp_path):
    module, calls = _fake_file_download(monkeypatch, xet_available = True)
    rp.restore_resumable_partials()

    _patched_writer(module)(
        incomplete_path = tmp_path / "abc.incomplete",
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
        xet_file_data = object(),
    )
    assert calls["http_get"] == [] and len(calls["stock"]) == 1


def test_a_xet_backed_repo_downloading_over_http_still_resumes(monkeypatch, tmp_path):
    """xet_file_data is set for any XET-backed repo, including when hf_xet is off.

    Gating on the metadata rather than on whether XET will actually run silently disables this for
    most of the Hub.
    """
    module, calls = _fake_file_download(monkeypatch, xet_available = False)
    rp.restore_resumable_partials()

    _patched_writer(module)(
        incomplete_path = tmp_path / "abc.incomplete",
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
        xet_file_data = object(),
    )
    assert len(calls["http_get"]) == 1 and calls["stock"] == []


def test_force_download_defers_to_stock(monkeypatch, tmp_path):
    module, calls = _fake_file_download(monkeypatch)
    rp.restore_resumable_partials()

    _patched_writer(module)(
        incomplete_path = tmp_path / "abc.incomplete",
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
        force_download = True,
    )
    assert calls["http_get"] == [] and len(calls["stock"]) == 1


def test_an_already_downloaded_blob_is_not_fetched_again(monkeypatch, tmp_path):
    module, calls = _fake_file_download(monkeypatch)
    rp.restore_resumable_partials()

    destination = tmp_path / "abc"
    destination.write_bytes(b"done")
    _patched_writer(module)(
        incomplete_path = tmp_path / "abc.incomplete",
        destination_path = destination,
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )
    assert calls["http_get"] == [] and calls["stock"] == []


def test_patching_twice_keeps_one_layer(monkeypatch):
    module, _ = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True
    first = module._download_to_tmp_and_move
    assert rp.restore_resumable_partials() is True
    assert module._download_to_tmp_and_move is first


# ---------------------------------------------------------------------------------------------
# The capability the UI reads
# ---------------------------------------------------------------------------------------------


def test_the_ui_is_told_partials_are_resumable_again(monkeypatch):
    from hub.utils import hf_cache_state

    _fake_file_download(monkeypatch)
    hf_cache_state.hf_partials_are_resumable.cache_clear()
    monkeypatch.setattr(rp, "can_restore_partials", lambda: True)
    assert hf_cache_state.hf_partials_are_resumable() is True

    hf_cache_state.hf_partials_are_resumable.cache_clear()
    monkeypatch.setattr(rp, "can_restore_partials", lambda: False)
    assert hf_cache_state.hf_partials_are_resumable() is False
    hf_cache_state.hf_partials_are_resumable.cache_clear()


def test_the_worker_restores_it_on_import():
    """A structural check: moving the call out of the worker fails here, not in the field."""
    import ast

    source = (Path(rp.__file__).parent.parent / "workers" / "hf_download.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "restore_resumable_partials" in calls
