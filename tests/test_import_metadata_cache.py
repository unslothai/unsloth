# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""`import unsloth` scans the installed distributions once, not once per library that asks."""

import ast
import importlib.metadata as metadata
import importlib.util
import os
import pathlib
import subprocess
import sys
import textwrap
import threading
import time

import pytest

_UNSLOTH = pathlib.Path(__file__).resolve().parents[1] / "unsloth"
_REPO = _UNSLOTH.parent

pytestmark = pytest.mark.skipif(
    not hasattr(metadata, "packages_distributions"),
    reason = "importlib.metadata.packages_distributions needs Python 3.10",
)


def _load(name = "_unsloth_import_metadata_cache_under_test"):
    spec = importlib.util.spec_from_file_location(name, _UNSLOTH / "_import_metadata_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache_module():
    original = metadata.packages_distributions
    module = _load()
    try:
        yield module
    finally:
        metadata.packages_distributions = original


@pytest.fixture
def counted(monkeypatch):
    """The real scan, counted; installed before the cache so the cache wraps it."""
    real = metadata.packages_distributions
    calls = []

    def scan():
        calls.append(1)
        return real()

    monkeypatch.setattr(metadata, "packages_distributions", scan)
    return calls


def _write_dist(site, name, top_level):
    dist_info = site / f"{name}-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
    (dist_info / "top_level.txt").write_text(top_level + "\n")


def _bump_mtime(path):
    # Some filesystems keep coarse mtimes; force a visible change.
    stat = os.stat(path)
    os.utime(path, ns = (stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))


def test_repeat_call_reuses_the_scan_and_matches_it(cache_module, counted):
    token = cache_module.install()
    first = metadata.packages_distributions()
    second = metadata.packages_distributions()
    assert len(counted) == 1
    cache_module.uninstall(token)
    reference = metadata.packages_distributions()
    assert first == second == reference
    assert list(first) == list(second) == list(reference)  # key order too


def test_callers_get_independent_copies(cache_module, counted):
    cache_module.install()
    first = metadata.packages_distributions()
    key = next(iter(first))
    first[key].append("mutated")
    first["__unsloth_injected__"] = ["x"]
    second = metadata.packages_distributions()
    assert "mutated" not in second[key]
    assert "__unsloth_injected__" not in second
    assert len(counted) == 1


def test_new_distribution_on_sys_path_is_seen(cache_module, counted, tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    cache_module.install()
    assert "unsloth_cache_probe_pkg" not in metadata.packages_distributions()
    _write_dist(site, "unsloth-cache-probe", "unsloth_cache_probe_pkg")
    _bump_mtime(site)
    after = metadata.packages_distributions()
    assert after["unsloth_cache_probe_pkg"] == ["unsloth-cache-probe"]
    assert len(counted) == 2


def test_removed_distribution_is_dropped(cache_module, counted, tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    _write_dist(site, "unsloth-cache-gone", "unsloth_cache_gone_pkg")
    monkeypatch.syspath_prepend(str(site))
    cache_module.install()
    assert "unsloth_cache_gone_pkg" in metadata.packages_distributions()
    for child in (site / "unsloth-cache-gone-1.0.dist-info").iterdir():
        child.unlink()
    (site / "unsloth-cache-gone-1.0.dist-info").rmdir()
    _bump_mtime(site)
    assert "unsloth_cache_gone_pkg" not in metadata.packages_distributions()
    assert len(counted) == 2


def test_sys_path_change_rescans(cache_module, counted, tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    _write_dist(site, "unsloth-cache-late", "unsloth_cache_late_pkg")
    cache_module.install()
    assert "unsloth_cache_late_pkg" not in metadata.packages_distributions()
    monkeypatch.syspath_prepend(str(site))
    assert metadata.packages_distributions()["unsloth_cache_late_pkg"] == ["unsloth-cache-late"]
    assert len(counted) == 2


def test_new_distribution_finder_rescans(cache_module, counted, monkeypatch):
    class Finder:
        def find_spec(self, *args, **kwargs):
            return None

        def find_distributions(self, context = None):
            return iter(())

    cache_module.install()
    metadata.packages_distributions()
    monkeypatch.setattr(sys, "meta_path", sys.meta_path + [Finder()])
    metadata.packages_distributions()
    assert len(counted) == 2


def test_a_plain_import_hook_does_not_rescan(cache_module, counted, monkeypatch):
    # unsloth_zoo adds a modeling-module finder between the transformers and TRL scans.
    class ImportOnly:
        def find_spec(self, *args, **kwargs):
            return None

    cache_module.install()
    metadata.packages_distributions()
    monkeypatch.setattr(sys, "meta_path", [ImportOnly()] + sys.meta_path)
    metadata.packages_distributions()
    assert len(counted) == 1


def test_uninstall_restores_and_forgets(cache_module, counted):
    before = metadata.packages_distributions
    token = cache_module.install()
    assert metadata.packages_distributions is token
    metadata.packages_distributions()
    cache_module.uninstall(token)
    assert metadata.packages_distributions is before
    assert token._unsloth_metadata_cache[0] is None
    cache_module.uninstall(None)  # a failed install is a no-op


def test_uninstall_leaves_a_later_wrapper_in_place(cache_module, counted):
    token = cache_module.install()

    def outer():
        return token()

    metadata.packages_distributions = outer
    cache_module.uninstall(token)
    assert metadata.packages_distributions is outer
    # Still reachable, but the window is over: every call is the real scan, nothing memoised.
    metadata.packages_distributions()
    metadata.packages_distributions()
    assert len(counted) == 2


def test_a_reference_kept_past_the_window_reaches_the_original(cache_module, counted):
    token = cache_module.install()
    bound = metadata.packages_distributions  # e.g. `from importlib.metadata import ...` meanwhile
    bound()
    cache_module.uninstall(token)
    bound()
    bound()
    assert len(counted) == 3
    assert token._unsloth_metadata_cache[0] is None


def _fake_owner(monkeypatch, name = "_unsloth_metadata_owner_under_test"):
    import types

    owner = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, owner)
    return owner


def test_a_failed_import_does_not_leave_the_wrapper(cache_module, counted, monkeypatch):
    # `import unsloth` raised midway: uninstall never ran and the module left sys.modules.
    real = metadata.packages_distributions
    owner = _fake_owner(monkeypatch)
    token = cache_module.install(owner)
    metadata.packages_distributions()
    metadata.packages_distributions()
    assert len(counted) == 1
    del sys.modules[owner.__name__]
    assert metadata.packages_distributions is token
    metadata.packages_distributions()  # the first call after the failure restores the original
    assert metadata.packages_distributions is real
    assert len(counted) == 2
    token()  # a kept reference passes through
    assert len(counted) == 3


def test_retrying_a_failed_import_does_not_stack(cache_module, counted, monkeypatch):
    real = metadata.packages_distributions
    first = _fake_owner(monkeypatch)
    cache_module.install(first)
    del sys.modules[first.__name__]  # failed, nobody called the wrapper since
    second = _fake_owner(monkeypatch)
    token = cache_module.install(second)
    assert token.__wrapped__ is real
    metadata.packages_distributions()
    metadata.packages_distributions()
    assert len(counted) == 1
    cache_module.uninstall(token)
    assert metadata.packages_distributions is real


def test_inner_failure_inside_an_outer_window(cache_module, counted, monkeypatch):
    # Studio's worker window is open; unsloth's import inside it raises; the worker's finally runs.
    real = metadata.packages_distributions
    outer = cache_module.install()
    inner_module = _load("_unsloth_import_metadata_cache_inner_fail")
    owner = _fake_owner(monkeypatch)
    inner = inner_module.install(owner)
    assert inner.__wrapped__ is outer
    del sys.modules[owner.__name__]
    cache_module.uninstall(outer)  # inner is still on top: left in place
    assert metadata.packages_distributions is inner
    metadata.packages_distributions()  # inner closes, finds outer closed too, unwraps both
    assert metadata.packages_distributions is real
    assert len(counted) == 1


def test_nested_scopes_share_the_outer_scan(cache_module, counted):
    # Studio's worker installs its copy before transformers; unsloth/__init__.py installs another.
    outer = cache_module.install()
    metadata.packages_distributions()  # transformers
    inner_module = _load("_unsloth_import_metadata_cache_inner")
    inner = inner_module.install()
    metadata.packages_distributions()  # TRL, during import unsloth
    inner_module.uninstall(inner)
    assert metadata.packages_distributions is outer
    cache_module.uninstall(outer)
    assert len(counted) == 1


def test_concurrent_callers_agree(cache_module, counted):
    cache_module.install()
    reference = metadata.packages_distributions()
    results, errors = [], []

    def worker():
        try:
            for _ in range(5):
                results.append(metadata.packages_distributions())
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target = worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert all(result == reference for result in results)
    assert len(counted) == 1


def test_unreadable_entries_do_not_break_it(cache_module, counted, monkeypatch):
    monkeypatch.setattr(
        sys, "path", sys.path + [os.path.join(os.sep, "nonexistent-unsloth-dir"), ""]
    )
    cache_module.install()
    assert metadata.packages_distributions() == metadata.packages_distributions()
    assert len(counted) == 1


def test_helper_is_stdlib_only():
    tree = ast.parse((_UNSLOTH / "_import_metadata_cache.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "Studio loads this file by path: no relative imports"
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names)


def test_init_installs_first_and_removes_last():
    source = (_UNSLOTH / "__init__.py").read_text()
    tree = ast.parse(source)
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "_install_metadata_cache" in calls and "_uninstall_metadata_cache" in calls
    lines = source.splitlines()
    install = next(i for i, l in enumerate(lines) if "_install_metadata_cache(sys.modules" in l)
    uninstall = next(i for i, l in enumerate(lines) if "_uninstall_metadata_cache(" in l)
    first_import = next(
        i
        for i, l in enumerate(lines)
        if l.startswith(("from .", "import "))
        and "_import_gc" not in l
        and "_import_metadata_cache" not in l
        and not l.startswith("import os")
    )
    assert install < first_import, "must be in place before transformers / TRL can be imported"
    assert uninstall > len(lines) - 20, "must stay in place for the whole import"


_COUNT_SCRIPT = textwrap.dedent(
    """
    import importlib.metadata as md, json, sys
    real = md.packages_distributions
    calls = []
    def scan():
        calls.append(1)
        return real()
    md.packages_distributions = scan
    import unsloth
    restored = md.packages_distributions is scan
    import importlib.util
    print("R12COUNT " + json.dumps({"scans": len(calls), "restored": restored,
        "trl": "trl" in sys.modules, "transformers": "transformers" in sys.modules}))
    """
)


_FAILED_IMPORT_SCRIPT = textwrap.dedent(
    """
    import importlib.metadata as md, json, sys
    real = md.packages_distributions
    sys.modules["unsloth_zoo"] = None  # any failure partway through unsloth/__init__.py
    try:
        import unsloth
        failed = False
    except ImportError:
        failed = True
    md.packages_distributions()
    print("R12FAIL " + json.dumps({"failed": failed, "restored": md.packages_distributions is real}))
    """
)


def test_import_unsloth_that_raises_gives_the_function_back():
    env = dict(os.environ, PYTHONWARNINGS = "ignore", UNSLOTH_DISABLE_AUTO_UPDATES = "1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _FAILED_IMPORT_SCRIPT],
        env = env,
        cwd = str(_REPO),
        capture_output = True,
        text = True,
        timeout = 900,
    )
    lines = [l for l in result.stdout.splitlines() if l.startswith("R12FAIL ")]
    assert lines, (result.returncode, result.stderr[-4000:])
    import json

    report = json.loads(lines[-1].split(" ", 1)[1])
    assert report == {"failed": True, "restored": True}, report


def _scan_callers_at_import():
    """How many of transformers / TRL call packages_distributions() at import (versions differ)."""
    found = 0
    for name in ("transformers.utils.import_utils", "trl.import_utils"):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.origin or not os.path.isfile(spec.origin):
            continue
        with open(spec.origin, encoding = "utf-8") as fh:
            for line in fh:
                if not line[:1].isspace() and "packages_distributions()" in line:
                    found += 1
                    break
    return found


def _has_accelerator():
    try:
        import torch
        return bool(
            torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())
        )
    except Exception:
        return False


def test_import_unsloth_scans_distributions_at_most_once():
    if _scan_callers_at_import() < 2:
        pytest.skip("installed transformers / TRL do not both scan at import")
    env = dict(os.environ, PYTHONWARNINGS = "ignore", UNSLOTH_DISABLE_AUTO_UPDATES = "1")
    # tests/conftest.py sets this for in-process tests; a GPU-less child still imports through the CPU path.
    if not _has_accelerator():
        env.pop("UNSLOTH_DISABLE_CPU_FALLBACK", None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO), env.get("PYTHONPATH")]))
    started = time.time()
    result = subprocess.run(
        [sys.executable, "-c", _COUNT_SCRIPT],
        env = env,
        cwd = str(_REPO),
        capture_output = True,
        text = True,
        timeout = 900,
    )
    lines = [l for l in result.stdout.splitlines() if l.startswith("R12COUNT ")]
    assert lines, (result.returncode, result.stderr[-4000:], time.time() - started)
    import json

    report = json.loads(lines[-1].split(" ", 1)[1])
    assert report["transformers"] and report["trl"], report
    assert report["scans"] == 1, report
    assert report["restored"], report  # the caller's function is back after the import
