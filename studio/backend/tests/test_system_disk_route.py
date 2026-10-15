# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/system/disk must be one cheap syscall on the download volume, not the filesystem root."""

from __future__ import annotations

import ast
import os
import shutil
import sys
import time
import types
from pathlib import Path

import pytest

ROUTE_SOURCE = Path(__file__).resolve().parents[1] / "main.py"


def _route_source() -> str:
    src = ROUTE_SOURCE.read_text(encoding = "utf-8")
    start = src.index('@app.get("/api/system/disk")')
    return src[start : src.index('@app.get("/api/system/gpu-visibility")', start)]


def _route_body() -> ast.FunctionDef:
    """The route function, parsed. Structure, not text: a comment naming psutil is fine, a call
    to it is not, and the two are indistinguishable to a grep."""
    module = ast.parse(_route_source().split("\n", 1)[1])
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    return function


def test_the_route_walks_nothing():
    """The disk route must not walk directories: cache_inventory's walks take seconds on large caches."""
    called = {
        node.func.attr
        for node in ast.walk(_route_body())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "disk_usage" in called
    for banned in ("walk", "scandir", "rglob", "iterdir", "glob"):
        assert banned not in called, f"the disk route calls {banned}"

    names = {node.id for node in ast.walk(_route_body()) if isinstance(node, ast.Name)}
    assert "psutil" not in names
    imported = {
        alias.name
        for node in ast.walk(_route_body())
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert (
        "hf_default_cache_dir" in imported
    ), "the route reports the filesystem root, not the models volume"
    assert (
        "get_hf_cache_paths" in imported
    ), "the route reports the default cache, not the configured one"


def _real_redactor():
    """Test the shipped redact_inventory_host_paths; a stub hid that the route leaked the path field."""
    from hub.utils.host_paths import redact_inventory_host_paths
    return redact_inventory_host_paths


def _load_route(
    monkeypatch,
    *,
    hub_cache,
    default_cache,
    studio,
    xet_cache = None,
    via_api_key = False,
    is_owner = True,
):
    """main.py imports the whole backend, so the route is compiled from its source with stubs."""
    namespace = {
        "app": types.SimpleNamespace(get = lambda _path: (lambda fn: fn)),
        "Depends": lambda _dep: None,
        "get_current_subject": lambda: "alice",
        "shutil": shutil,
        "os": os,
        "Path": Path,
        "logger": types.SimpleNamespace(debug = lambda *_a, **_k: None),
        "authenticated_via_api_key": lambda: via_api_key,
        "redact_inventory_host_paths": _real_redactor(),
    }
    storage = types.ModuleType("utils.paths.storage_roots")
    storage.hf_default_cache_dir = lambda: default_cache
    storage.studio_root = lambda: studio
    settings = types.ModuleType("utils.hf_cache_settings")
    settings.get_hf_cache_paths = lambda: types.SimpleNamespace(
        hub_cache = hub_cache, xet_cache = hub_cache if xet_cache is None else xet_cache
    )
    accounts = types.ModuleType("utils.account_context")
    accounts.is_owner_context = lambda: is_owner
    monkeypatch.setitem(sys.modules, "utils.paths.storage_roots", storage)
    monkeypatch.setitem(sys.modules, "utils.hf_cache_settings", settings)
    monkeypatch.setitem(sys.modules, "utils.account_context", accounts)
    exec(compile(_route_source(), "<route>", "exec"), namespace)
    return namespace["get_disk_space"]


def test_the_route_follows_the_configured_models_folder(monkeypatch, tmp_path):
    """Follow the configured Models Folder; hf_default_cache_dir ignores HF_HUB_CACHE and the setting."""
    configured = tmp_path / "elsewhere" / "hub"
    configured.mkdir(parents = True)
    default = tmp_path / "home" / ".cache" / "huggingface" / "hub"
    default.mkdir(parents = True)

    route = _load_route(
        monkeypatch, hub_cache = configured, default_cache = default, studio = tmp_path / "s"
    )
    assert route(current_subject = "alice")["path"] == str(configured)


def test_the_route_still_answers_when_the_settings_read_fails(monkeypatch, tmp_path):
    """The settings read touches SQLite, and a reading is worth less than a broken download."""
    broken = types.ModuleType("utils.hf_cache_settings")

    def _raise():
        raise RuntimeError("no database")

    broken.get_hf_cache_paths = _raise
    default = tmp_path / "default"
    default.mkdir()
    monkeypatch.setitem(sys.modules, "utils.hf_cache_settings", broken)

    storage = types.ModuleType("utils.paths.storage_roots")
    storage.hf_default_cache_dir = lambda: default
    storage.studio_root = lambda: tmp_path / "s"
    monkeypatch.setitem(sys.modules, "utils.paths.storage_roots", storage)
    namespace = {
        "app": types.SimpleNamespace(get = lambda _path: (lambda fn: fn)),
        "Depends": lambda _dep: None,
        "get_current_subject": lambda: "alice",
        "shutil": shutil,
        "os": os,
        "Path": Path,
        "logger": types.SimpleNamespace(debug = lambda *_a, **_k: None),
        "authenticated_via_api_key": lambda: False,
        "redact_inventory_host_paths": _real_redactor(),
    }
    exec(compile(_route_source(), "<route>", "exec"), namespace)
    assert namespace["get_disk_space"](current_subject = "alice")["path"] == str(default)


def test_a_disk_reading_is_microseconds(tmp_path):
    """disk_usage must stay microseconds; a loose budget fails only on a stalled mount, not a busy
    runner."""
    shutil.disk_usage(tmp_path)
    started = time.perf_counter()
    for _ in range(200):
        shutil.disk_usage(tmp_path)
    per_call_ms = (time.perf_counter() - started) / 200 * 1000
    assert per_call_ms < 5.0, f"disk_usage took {per_call_ms:.2f} ms per call"


def test_a_missing_cache_dir_falls_back_to_a_real_ancestor(tmp_path):
    """disk_usage raises on a path that does not exist, and the model cache legitimately does
    not exist yet on a fresh install. The route walks up to the first ancestor that does."""
    missing = tmp_path / "not" / "created" / "yet"
    with pytest.raises(OSError):
        shutil.disk_usage(missing)

    for candidate in (missing, *missing.parents):
        try:
            usage = shutil.disk_usage(candidate)
        except (OSError, ValueError):
            continue
        assert usage.total > 0
        assert candidate in missing.parents
        break
    else:
        pytest.fail("no ancestor of a tmp_path was readable")


def test_an_unreadable_host_reports_null_not_zero():
    """Unknown is not zero: an unreadable host reports null, since the frontend reads zero as a failure."""
    source = _route_source()
    assert '"total_gb": None' in source
    assert '"free_gb": None' in source


def test_the_route_reports_the_tighter_of_the_hub_and_xet_volumes(monkeypatch, tmp_path):
    """Report the tighter of the hub and Xet volumes; an Xet download can exhaust its own disk."""
    hub = tmp_path / "roomy" / "hub"
    hub.mkdir(parents = True)
    xet = tmp_path / "cramped" / "xet"
    xet.mkdir(parents = True)

    roomy = shutil._ntuple_diskusage(1_000_000_000_000, 100_000_000_000, 900_000_000_000)
    cramped = shutil._ntuple_diskusage(1_000_000_000_000, 998_000_000_000, 2_000_000_000)

    def fake_usage(path):
        return cramped if str(xet).startswith(str(path)) or path == xet else roomy

    monkeypatch.setattr(shutil, "disk_usage", fake_usage)
    # Distinct devices, so the two readings are not deduplicated onto one volume.
    real_stat = os.stat
    monkeypatch.setattr(
        os,
        "stat",
        lambda p, *a, **k: types.SimpleNamespace(
            st_dev = 2 if str(p).startswith(str(tmp_path / "cramped")) else 1
        )
        if str(p).startswith(str(tmp_path))
        else real_stat(p, *a, **k),
    )

    route = _load_route(
        monkeypatch,
        hub_cache = hub,
        xet_cache = xet,
        default_cache = tmp_path / "default",
        studio = tmp_path / "s",
    )
    reading = route(current_subject = "alice")

    assert reading["free_gb"] == 2.0, "the roomy hub volume masked the full Xet volume"
    assert reading["path"] == str(xet)


def test_one_volume_is_read_once(monkeypatch, tmp_path):
    """One volume must be read once: resolve the device before disk_usage, not after deduplicating."""
    both = tmp_path / "cache"
    (both / "hub").mkdir(parents = True)
    (both / "xet").mkdir(parents = True)

    calls = []
    real_usage = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda p: (calls.append(str(p)), real_usage(p))[1])

    route = _load_route(
        monkeypatch,
        hub_cache = both / "hub",
        xet_cache = both / "xet",
        default_cache = tmp_path / "default",
        studio = tmp_path / "s",
    )
    route(current_subject = "alice")

    assert (
        len(calls) == 1
    ), f"hub and xet share a volume, so one disk_usage should answer for both; got {calls}"


def test_an_api_key_caller_is_not_told_the_host_path(monkeypatch, tmp_path):
    """An API-key caller must not get the host path, which exposes the service account's home layout."""
    hub = tmp_path / "cache" / "hub"
    hub.mkdir(parents = True)

    route = _load_route(
        monkeypatch,
        hub_cache = hub,
        default_cache = tmp_path / "d",
        studio = tmp_path / "s",
        via_api_key = True,
    )
    reading = route(current_subject = "alice", via_api_key = True)

    assert not reading.get("path"), "the raw host path went out to an API-key caller"
    assert reading["free_gb"] is not None, "the capacity fields must survive redaction"


def test_a_ui_session_still_sees_the_path(monkeypatch, tmp_path):
    """The control. Redacting for everyone would take the path off the Resources tab, which is
    where a user checks WHICH volume the reading is about."""
    hub = tmp_path / "cache" / "hub"
    hub.mkdir(parents = True)

    route = _load_route(
        monkeypatch,
        hub_cache = hub,
        default_cache = tmp_path / "d",
        studio = tmp_path / "s",
    )
    reading = route(current_subject = "alice", via_api_key = False)

    assert reading["path"], "a UI session lost the path it needs to identify the volume"


def test_a_managed_account_is_not_told_the_host_path(monkeypatch, tmp_path):
    """A managed account's session JWT must not get the host path either; the path is owner-only."""
    hub = tmp_path / "cache" / "hub"
    hub.mkdir(parents = True)

    route = _load_route(
        monkeypatch,
        hub_cache = hub,
        default_cache = tmp_path / "d",
        studio = tmp_path / "s",
        is_owner = False,
    )
    reading = route(current_subject = "bob", via_api_key = False)

    assert not reading.get("path"), "the raw host path went out to a managed account"
    assert reading["free_gb"] is not None, "the capacity fields must survive redaction"


def test_a_single_user_install_still_sees_the_path(monkeypatch, tmp_path):
    """A single-user install still sees the path; redaction covers API-key and managed callers."""
    hub = tmp_path / "cache" / "hub"
    hub.mkdir(parents = True)

    route = _load_route(
        monkeypatch,
        hub_cache = hub,
        default_cache = tmp_path / "d",
        studio = tmp_path / "s",
        is_owner = True,
    )
    reading = route(current_subject = "alice", via_api_key = False)

    assert reading["path"], "the owner lost the path on a single-user install"


def test_an_unreadable_cache_volume_is_not_reported_as_its_parent(monkeypatch, tmp_path):
    """An unreadable cache volume must not be reported as its nearest existing parent's free space."""
    roomy = tmp_path / "roomy"
    roomy.mkdir()
    cache = roomy / "cache"
    cache.mkdir()

    real_stat = os.stat
    real_usage = shutil.disk_usage

    def deny_stat(path, *args, **kwargs):
        if str(path) == str(cache):
            raise PermissionError(13, "Permission denied")
        return real_stat(path, *args, **kwargs)

    def deny_usage(path, *args, **kwargs):
        # Deny both calls: the old and new route versions fail at different ones.
        if str(path) == str(cache):
            raise PermissionError(13, "Permission denied")
        return real_usage(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", deny_stat)
    monkeypatch.setattr(shutil, "disk_usage", deny_usage)

    route = _load_route(
        monkeypatch,
        hub_cache = cache,
        xet_cache = cache,
        default_cache = tmp_path / "missing-default",
        studio = tmp_path / "missing-studio",
    )
    reading = route(current_subject = "alice", via_api_key = False)

    assert reading["path"] != str(
        roomy
    ), "an unreadable cache reported its parent volume's free space"


def test_one_unreadable_root_does_not_let_the_other_answer_for_it(monkeypatch, tmp_path):
    """An unavailable hub volume must not borrow the Xet volume's free space; report unknown."""
    hub = tmp_path / "mounted" / "hub"
    hub.mkdir(parents = True)
    xet = tmp_path / "local" / "xet"
    xet.mkdir(parents = True)

    real_stat = os.stat
    real_usage = shutil.disk_usage

    def deny_stat(path, *args, **kwargs):
        if str(path) == str(hub):
            raise OSError(5, "Input/output error")
        return real_stat(path, *args, **kwargs)

    def deny_usage(path, *args, **kwargs):
        if str(path) == str(hub):
            raise OSError(5, "Input/output error")
        return real_usage(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", deny_stat)
    monkeypatch.setattr(shutil, "disk_usage", deny_usage)

    route = _load_route(
        monkeypatch,
        hub_cache = hub,
        xet_cache = xet,
        default_cache = tmp_path / "d",
        studio = tmp_path / "s",
    )
    reading = route(current_subject = "alice", via_api_key = False)

    assert (
        reading["free_gb"] is None
    ), "the readable Xet volume answered for an unreadable hub cache"
    assert reading["path"] is None
