# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Verification tests for PR #6269 (namespace-shadow guard).

`ensure_real_packages` (core/import_guards.py) drops namespace-package
shadow dirs (a `unsloth`/`unsloth_zoo` dir with no __init__.py on sys.path)
before `from unsloth import ...`. Order matters: `unsloth.__init__` runs its
ROCm/Windows bnb fixes before importing unsloth_zoo, so the guard must import
unsloth first. Each test runs the real guard (ast-extracted from source, no
GPU/torch) in a subprocess, with fake packages reachable only via a meta path
finder to mimic an editable/PEP 660 install where the shadow wins.

Originally defined in core/training/trainer.py; extracted to the shared
core/import_guards.py so the inference, export and embedding workers reuse it.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

GUARD_PY = Path(__file__).resolve().parents[1] / "core" / "import_guards.py"


# ── fake package bodies ──────────────────────────────────────────────
# Real `unsloth` sets a sentinel (mimics _gpu_init's pre-zoo fixes) then
# imports unsloth_zoo, which records whether that sentinel was set first.

_REAL_UNSLOTH_INIT = textwrap.dedent(
    """
    import os
    with open(os.environ["GUARD_ORDER_FILE"], "a") as _f:
        _f.write("unsloth\\n")
    # mimic _gpu_init: pre-zoo ROCm/Windows fixes run before importing zoo
    os.environ["UNSLOTH_GPU_INIT_RAN"] = "1"
    import unsloth_zoo  # noqa: F401
    REAL = True
    """
)

_REAL_ZOO_INIT = textwrap.dedent(
    """
    import os
    with open(os.environ["GUARD_ORDER_FILE"], "a") as _f:
        _f.write("unsloth_zoo\\n")
    with open(os.environ["GUARD_SENTINEL_FILE"], "w") as _f:
        _f.write(os.environ.get("UNSLOTH_GPU_INIT_RAN", "0"))
    REAL = True
    """
)


# ── subprocess driver ────────────────────────────────────────────────
# Reads a JSON config, rebuilds sys.path / sys.meta_path to model the
# scenario, runs the real guard, and writes the observed result as JSON.

_DRIVER = textwrap.dedent(
    """
    import ast, importlib.abc, importlib.util, json, os, sys

    cfg = json.load(open(sys.argv[1]))

    # Extract the real ensure_real_packages from import_guards.py source
    # without importing the heavy module or its `from unsloth import ...` line.
    src = open(cfg["guard_py"]).read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "ensure_real_packages")
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"os": os, "sys": sys}
    exec(compile(mod, cfg["guard_py"], "exec"), ns)
    _ensure_real_packages = ns["ensure_real_packages"]

    # Under -S site-packages is off, so a shadow root placed first wins the
    # path finder; the real packages come only from the meta finder below.
    for root in reversed(cfg["shadow_roots"]):
        sys.path.insert(0, root)

    # Real packages are reachable only via a meta path finder appended AFTER
    # the standard PathFinder -- the editable / PEP 660 install shape.
    real_root = cfg.get("real_root")
    if real_root:
        class _RealFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] not in ("unsloth", "unsloth_zoo"):
                    return None
                parts = fullname.split(".")
                init = os.path.join(real_root, *parts, "__init__.py")
                if os.path.isfile(init):
                    return importlib.util.spec_from_file_location(
                        fullname, init,
                        submodule_search_locations=[os.path.dirname(init)],
                    )
                return None
        sys.meta_path.append(_RealFinder())

    # optionally force importlib.invalidate_caches to raise, to prove the guard
    # still restores sys.path in that case
    if cfg.get("raise_on_invalidate"):
        def _boom():
            raise RuntimeError("invalidate_caches failed")
        importlib.invalidate_caches = _boom

    path_before = list(sys.path)
    result = {"error": None}
    try:
        _ensure_real_packages(*cfg["names"])
    except Exception as e:
        result["error"] = type(e).__name__

    result["path_restored"] = list(sys.path) == path_before
    of = os.environ["GUARD_ORDER_FILE"]
    result["order"] = open(of).read().split() if os.path.isfile(of) else []
    sf = os.environ["GUARD_SENTINEL_FILE"]
    result["sentinel_when_zoo_imported"] = (
        open(sf).read().strip() if os.path.isfile(sf) else None
    )

    def _real(name):
        m = sys.modules.get(name)
        return bool(m and getattr(m, "REAL", False))
    result["unsloth_real"] = _real("unsloth")
    result["unsloth_zoo_real"] = _real("unsloth_zoo")

    json.dump(result, open(cfg["out"], "w"))
    """
)


def _make_namespace_shadow(root: Path, name: str) -> None:
    """A directory named `name` with no __init__.py -> namespace portion."""
    (root / name).mkdir(parents = True, exist_ok = True)


def _make_real_pkg(root: Path) -> None:
    (root / "unsloth").mkdir(parents = True, exist_ok = True)
    (root / "unsloth" / "__init__.py").write_text(_REAL_UNSLOTH_INIT)
    (root / "unsloth_zoo").mkdir(parents = True, exist_ok = True)
    (root / "unsloth_zoo" / "__init__.py").write_text(_REAL_ZOO_INIT)


def _run(
    tmp_path: Path,
    *,
    shadow_roots,
    real: bool,
    names = ("unsloth_zoo", "unsloth"),
    guard_py: Path = GUARD_PY,
    raise_on_invalidate: bool = False,
):
    order_file = tmp_path / "order.txt"
    sentinel_file = tmp_path / "sentinel.txt"
    out = tmp_path / "result.json"
    real_root = tmp_path / "real"
    if real:
        _make_real_pkg(real_root)

    cfg = {
        "guard_py": str(guard_py),
        "shadow_roots": [str(r) for r in shadow_roots],
        "real_root": str(real_root) if real else None,
        "names": list(names),
        "out": str(out),
        "raise_on_invalidate": raise_on_invalidate,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)

    env = dict(os.environ)
    env["GUARD_ORDER_FILE"] = str(order_file)
    env["GUARD_SENTINEL_FILE"] = str(sentinel_file)
    env.pop("UNSLOTH_GPU_INIT_RAN", None)
    # Drop PYTHONPATH too so nothing re-introduces the real packages onto the
    # path; -S already keeps site-packages off.
    env.pop("PYTHONPATH", None)

    proc = subprocess.run(
        # -S: skip site-packages so the installed unsloth_zoo can't shadow-beat
        # the namespace portion; the real package is served by the meta finder.
        [sys.executable, "-S", str(driver), str(cfg_path)],
        env = env,
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert out.is_file(), f"driver did not produce a result\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return json.loads(out.read_text())


# ── scenarios ────────────────────────────────────────────────────────


def test_only_zoo_shadowed_imports_unsloth_first(tmp_path):
    """Core concern: a unsloth_zoo shadow must not make the guard import
    unsloth_zoo before unsloth -- that would skip _gpu_init's pre-zoo fixes."""
    shadow = tmp_path / "shadow"
    _make_namespace_shadow(shadow, "unsloth_zoo")
    res = _run(tmp_path, shadow_roots = [shadow], real = True)

    assert res["error"] is None
    assert res["order"] == ["unsloth", "unsloth_zoo"], res
    assert res["sentinel_when_zoo_imported"] == "1", res
    assert res["unsloth_real"] and res["unsloth_zoo_real"], res
    assert res["path_restored"], res


def test_both_shadowed_imports_unsloth_first(tmp_path):
    shadow = tmp_path / "shadow"
    _make_namespace_shadow(shadow, "unsloth")
    _make_namespace_shadow(shadow, "unsloth_zoo")
    res = _run(tmp_path, shadow_roots = [shadow], real = True)

    assert res["error"] is None
    assert res["order"] == ["unsloth", "unsloth_zoo"], res
    assert res["sentinel_when_zoo_imported"] == "1", res
    assert res["unsloth_real"] and res["unsloth_zoo_real"], res
    assert res["path_restored"], res


def test_only_unsloth_shadowed_recovers(tmp_path):
    shadow = tmp_path / "shadow"
    _make_namespace_shadow(shadow, "unsloth")
    res = _run(tmp_path, shadow_roots = [shadow], real = True)

    assert res["error"] is None
    assert res["order"] == ["unsloth", "unsloth_zoo"], res
    assert res["unsloth_real"] and res["unsloth_zoo_real"], res
    assert res["path_restored"], res


def test_healthy_install_is_noop(tmp_path):
    """No shadow on sys.path: the guard imports nothing and leaves sys.path."""
    res = _run(tmp_path, shadow_roots = [], real = True)

    assert res["error"] is None
    assert res["order"] == [], res  # guard short-circuits before importing
    assert res["path_restored"], res


def test_real_package_absent_surfaces_module_not_found(tmp_path):
    shadow = tmp_path / "shadow"
    _make_namespace_shadow(shadow, "unsloth_zoo")
    res = _run(tmp_path, shadow_roots = [shadow], real = False)

    assert res["error"] == "ModuleNotFoundError", res
    assert res["path_restored"], res  # restored even on failure


def test_multiple_shadow_entries_all_removed(tmp_path):
    shadow_a = tmp_path / "shadow_a"
    shadow_b = tmp_path / "shadow_b"
    _make_namespace_shadow(shadow_a, "unsloth_zoo")
    _make_namespace_shadow(shadow_b, "unsloth_zoo")
    res = _run(tmp_path, shadow_roots = [shadow_a, shadow_b], real = True)

    assert res["error"] is None
    # if only one offending entry were dropped, zoo would still resolve to a
    # namespace shadow and unsloth_zoo_real would be False
    assert res["unsloth_zoo_real"], res
    assert res["order"] == ["unsloth", "unsloth_zoo"], res
    assert res["path_restored"], res


def test_sys_path_restored_if_invalidate_caches_raises(tmp_path):
    """sys.path is restored even if importlib.invalidate_caches raises."""
    shadow = tmp_path / "shadow"
    _make_namespace_shadow(shadow, "unsloth_zoo")
    res = _run(tmp_path, shadow_roots = [shadow], real = True, raise_on_invalidate = True)

    assert res["error"] == "RuntimeError", res
    assert res["path_restored"], res


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
