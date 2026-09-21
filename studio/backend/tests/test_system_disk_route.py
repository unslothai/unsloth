# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/system/disk: one syscall, at the volume the downloads land on.

The low-disk notice used to poll /api/system every 60 s from the app shell. That route
enumerates GPUs, reads package metadata and samples CPU, so it ran that work forever in every
open tab to answer a question whose answer only changes when something writes to the disk. The
notice now asks for this route instead, once on mount and once on the way into a download.

Which means this route has to be cheap enough to sit in front of a download, and has to report
the volume the bytes are actually going to, which is not the filesystem root whenever the model
cache lives on another disk.
"""

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
    """A directory walk here would put seconds in front of every download start.

    cache_inventory measures cache sizes by walking, and its own comments call that "seconds on
    a large uv or triton cache". None of that may be reachable from this route.
    """
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
    # Imported inside the function, so the resolver is not pulled in at module import time.
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
    """The SHIPPED redactor, not a stand-in.

    A stub here is what let the first version of this guard pass while the route was still
    handing out the path: it implemented the behaviour I assumed rather than the one the repo
    has. redact_host_paths runs _redact with redact_ambiguous_path=False and leaves a field
    named "path" untouched; only the inventory redactor treats it as a host path. Importing
    the real thing is what makes this test able to fail.
    """
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
):
    """Run the real route body against stub resolvers, without importing main.py.

    main.py pulls in the whole backend, which no unit test can afford, so the function is
    compiled from its own source with a no-op decorator and the three modules it imports
    inside itself replaced. What runs is the shipped code, not a paraphrase of it.
    """
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
    monkeypatch.setitem(sys.modules, "utils.paths.storage_roots", storage)
    monkeypatch.setitem(sys.modules, "utils.hf_cache_settings", settings)
    exec(compile(_route_source(), "<route>", "exec"), namespace)
    return namespace["get_disk_space"]


def test_the_route_follows_the_configured_models_folder(monkeypatch, tmp_path):
    """A user who moved the Models Folder to another volume is asking about THAT volume.

    hf_default_cache_dir() is documented to ignore HF_HUB_CACHE and the Studio setting, and its
    home ancestor always exists, so a route that started there would answer confidently about a
    disk the download is not going to touch.
    """
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
    """The claim the design rests on, measured rather than asserted.

    shutil.disk_usage is statvfs on Linux and macOS and GetDiskFreeSpaceExW on Windows. The
    budget is deliberately loose, three orders of magnitude above what it costs here, so this
    fails on a genuinely slow path (a stalled network mount) and not on a busy runner.
    """
    shutil.disk_usage(tmp_path)  # warm any lazy loading, so the first call is not the sample
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
    """Zero is a different claim from unknown.

    The frontend's diskPressure() reads a zero total as the host having failed and a zero free
    as a full disk. A route that answered 0 for "I could not tell" would warn every user on a
    platform where the probe fails.
    """
    source = _route_source()
    assert '"total_gb": None' in source
    assert '"free_gb": None' in source


def test_the_route_reports_the_tighter_of_the_hub_and_xet_volumes(monkeypatch, tmp_path):
    """HF_XET_CACHE is resolved independently of HF_HUB_CACHE, and every download now streams
    its chunks through the Xet cache, so the two can sit on different volumes.

    Reading only the hub volume lets an Xet download exhaust its own disk while the route keeps
    reporting ample space, which is the one answer this endpoint exists to get right. The
    tighter reading wins: the volume that runs out first is the one that stops the download.
    """
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
    """The negative control for the test above, and the cost claim in the docstring.

    Hub and Xet share a volume on an ordinary install. The first version of this deduplicated
    AFTER reading, so it still paid a disk_usage per root: two syscalls to answer about one
    volume on every ordinary machine, which is what the docstring promises not to do and what
    costs most on a network mount. The device is resolved first now.
    """
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
    """get_current_subject accepts an sk-unsloth key, and `path` is a raw host path naming the
    service account and its home layout. hub/utils/host_paths draws that boundary for the Hub
    inventory routes already; a capacity reading is not a reason to cross it, and the low-disk
    client reads only the numbers."""
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


def test_an_unreadable_cache_volume_is_not_reported_as_its_parent(monkeypatch, tmp_path):
    """Missing and unreadable are different answers.

    A cache directory that does not exist yet is ordinary, and the volume it would live on is
    its nearest existing parent. A permission error, an I/O error or a network mount that is
    not answering is not missing: climbing past it reports the parent filesystem's free space
    for a disk nothing could read, which suppresses the warning with a confidently wrong
    number. This route already refuses to turn an unreadable host into a zero for the same
    reason.
    """
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
        # Both calls, because an unreadable volume refuses both and the two versions of this
        # route fail at different ones: denying only os.stat left the old code reading the
        # cache happily and the test passing against the bug.
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
    """Hub and Xet can be on different volumes, so the readable one is not a stand-in.

    An unavailable network-mounted hub cache beside a local Xet cache would otherwise return
    the Xet volume's free space: a confident number about a disk the download is not filling,
    which is the same mistake as climbing past an unreadable root, one level up. Unknown is
    the honest answer, and the client treats it as neither a warning nor a block.
    """
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
