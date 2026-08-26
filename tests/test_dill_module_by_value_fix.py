# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An off-prefix install makes dill pickle whole modules by value.

`dill._dill._is_builtin_module` pickles a module by REFERENCE only if its
`__file__` starts with a sys prefix, ends with an extension suffix, or contains
the literal string `site-packages`. `pip install --target <dir>`, a PYTHONPATH
overlay and a Lambda-style layer satisfy none of the three, so every package
there is pickled BY VALUE.

`datasets` fingerprints through dill, so on such an install
`Dataset.from_dict({"text": ["a", "b"]})` walks
`datasets/utils/_dill.py:_save_arrowTable` -> `create_arrowTable` -> that
function's globals -> the pyarrow MODULE, and dies on pyarrow's Cython
`MonthDayNano`, whose `__module__` is `builtins`:

    PicklingError: Can't pickle <class 'MonthDayNano'>:
        it's not found as builtins.MonthDayNano

Measured against a byte-identical package tree with the DIRECTORY NAME as the
only variable, on dill 0.3.8 and 0.4.1 alike: the plain `--target` directory
raised, the copy named `site-packages` returned a fingerprint. datasets 4.3.0
never reached that path and is unaffected either way; 5.0.1 fails 100% of the
time.

The tests below reproduce the MECHANISM rather than the package: a two-module
tree carrying a class that claims `__module__ = "builtins"`, which is the one
property of `MonthDayNano` that matters here. Real dill, real import machinery,
real subprocess -- stubbing `sys.modules` would test the stub, and the whole
bug is about where a file lives on disk.
"""

import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytest.importorskip("dill")


# --------------------------------------------------------------------------
# The real thing, in a subprocess, on a real off-prefix tree
# --------------------------------------------------------------------------

_HOSTILE_TREE = {
    # The pyarrow stand-in. The fix's gate asks whether `datasets` or `pyarrow`
    # resolve to somewhere dill pickles by value, and answers from the SPEC, so
    # this file is never executed and needs no content.
    "pyarrow.py": "VERSION = '0'\n",
    # The class that cannot be pickled by reference. Two properties, both taken
    # from the real `MonthDayNano` and both necessary:
    #
    # * `__module__ = "builtins"`, where the class is not found, which is what
    #   Cython types in pyarrow do; and
    # * a self-reference, which puts the class in dill's postproc list, so the
    #   second encounter takes dill's `save_global` branch instead of writing
    #   the class out by value. Without it dill pickles the class by value and
    #   succeeds -- the real run says so in its own warning, "has recursive
    #   self-references that trigger a RecursionError".
    "ovmod.py": textwrap.dedent(
        """
        class Sneaky:
            pass

        Sneaky.self_ref = Sneaky
        Sneaky.__module__ = "builtins"
        """
    ),
    # The function dill has to save, in the shape datasets uses.
    #
    # NESTED, and that is the whole reason this reproduces:
    # `datasets/utils/_dill.py:_save_arrowTable` defines `create_arrowTable`
    # INSIDE itself and hands it to `save_reduce`. dill's `_locate_function`
    # cannot find a `<locals>` qualname at module level, so it falls to saving
    # the function BY VALUE, walks its globals with `recurse=True`, and reaches
    # the module. A module-level function would be saved by reference and
    # nothing would ever look at pyarrow -- which is exactly why datasets 4.3.0,
    # whose table reducer does not take this path, is unaffected.
    "ovuser.py": textwrap.dedent(
        """
        import ovmod

        def outer():
            def create_arrowTable():
                return ovmod.Sneaky
            return create_arrowTable
        """
    ),
}

_DRIVER = textwrap.dedent(
    """
    import importlib.util, json, os, sys

    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes", os.environ["IMPORT_FIXES"])
    fixes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixes)

    out = {"applied": None, "second_call": None, "error": None}
    # Imported BEFORE the patch: `dill.session` does
    # `from ._dill import _is_builtin_module`, so it holds its own binding and
    # patching the defining module alone leaves this copy on the old function.
    import dill.session as _session
    import dill._dill as _core
    if os.environ.get("APPLY") == "1":
        out["applied"] = fixes.fix_dill_module_by_value_pickling()
        out["second_call"] = fixes.fix_dill_module_by_value_pickling()
    out["affected"] = fixes._dill_environment_is_affected()
    out["session_binding_patched"] = (
        _session._is_builtin_module is _core._is_builtin_module)

    import dill, ovuser
    try:
        dill.dumps(ovuser.outer(), recurse=True)
        out["dumps"] = "ok"
    except Exception as exc:
        out["dumps"] = "%s: %s" % (type(exc).__name__, exc)
    print("RESULT " + json.dumps(out))
    """
)


def _child_python(tmp_path):
    """An interpreter whose `sys.prefix` is INSIDE tmp_path.

    Without this the test is at the mercy of where tmp lives: on a box whose
    virtualenv root is an ancestor of tmp (ours is), the tree would sit under
    `sys.prefix`, dill would be perfectly happy with it, and all three
    subprocess tests would pass while reproducing nothing. A throwaway venv
    beside the overlay makes the overlay off-prefix everywhere, and
    `--system-site-packages` means dill is still importable without an install.
    """
    root = tmp_path / "venv"
    try:
        import venv as _venv
        _venv.EnvBuilder(system_site_packages = True, with_pip = False).create(root)
    except Exception as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"cannot build a venv to host the child interpreter: {exc}")
    for candidate in (root / "bin" / "python", root / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    pytest.skip("the venv produced no interpreter on this platform")


def _run_on_hostile_tree(
    tmp_path,
    *,
    apply,
    extra_env = None,
):
    """Build the tree OUTSIDE any sys prefix and run the driver against it."""
    overlay = tmp_path / "overlay_leg"  # deliberately not "site-packages"
    overlay.mkdir()
    for name, body in _HOSTILE_TREE.items():
        (overlay / name).write_text(body, encoding = "utf-8")
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding = "utf-8")

    # The child venv inherits the BASE prefix's site-packages, not this
    # interpreter's, so dill would be missing when the tests run from a venv.
    # Appended AFTER the overlay, and it is a real site-packages directory, so
    # dill itself stays on the by-reference side of its own rule.
    import sysconfig

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(overlay), sysconfig.get_paths()["purelib"], os.environ.get("PYTHONPATH", "")]
    )
    env["IMPORT_FIXES"] = str(REPO / "unsloth" / "import_fixes.py")
    env["APPLY"] = "1" if apply else "0"
    env.pop("UNSLOTH_DISABLE_DILL_FIX", None)
    env.update(extra_env or {})
    proc = subprocess.run(
        [_child_python(tmp_path), str(driver)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 300,
        cwd = str(tmp_path),
    )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, f"driver produced no result\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return json.loads(line[0][len("RESULT ") :])


def test_an_off_prefix_install_breaks_dill_without_the_fix(tmp_path):
    """The negative control, and the reason the fix exists at all.

    If this ever passes, dill has changed its own rule and the patch below is
    dead weight -- re-measure before deleting it.
    """
    got = _run_on_hostile_tree(tmp_path, apply = False)
    assert got["affected"] is True, (
        "the gate does not recognise this layout, so the fix would never install itself here"
    )
    assert got["dumps"].startswith("PicklingError"), (
        "dill pickled the off-prefix module by reference unaided; the bug this "
        f"guards is gone or moved: {got['dumps']}"
    )
    assert "builtins.Sneaky" in got["dumps"]


def test_the_fix_makes_the_same_tree_picklable(tmp_path):
    got = _run_on_hostile_tree(tmp_path, apply = True)
    assert got["applied"] is True
    assert got["dumps"] == "ok", got["dumps"]


def test_it_is_idempotent(tmp_path):
    """Applied twice, dill must not end up wrapping the wrapper: a second layer
    is invisible until something recurses."""
    got = _run_on_hostile_tree(tmp_path, apply = True)
    assert got["second_call"] is False, "the patch re-applied itself"


def test_the_env_switch_turns_it_off(tmp_path):
    """A user whose environment this misjudges needs a way out that does not
    involve editing site-packages."""
    got = _run_on_hostile_tree(tmp_path, apply = True, extra_env = {"UNSLOTH_DISABLE_DILL_FIX": "1"})
    assert got["applied"] is False
    assert got["dumps"].startswith("PicklingError")


# --------------------------------------------------------------------------
# The gate: an ordinary install must be untouched
# --------------------------------------------------------------------------


def test_an_ordinary_site_packages_install_is_a_no_op():
    """dill's behaviour, fingerprints included, has to be identical where it
    already works. The gate is what guarantees that, so it is asserted against
    the environment this test suite itself runs in."""
    from unsloth.import_fixes import _dill_path_pickles_by_value

    assert not _dill_path_pickles_by_value("/usr/lib/python3.12/site-packages/pyarrow/__init__.py")
    assert not _dill_path_pickles_by_value(os.path.join(sys.prefix, "x", "y.py"))
    assert not _dill_path_pickles_by_value(None)
    assert _dill_path_pickles_by_value("/opt/layer/python/pyarrow/__init__.py")


def test_the_widening_only_covers_modules_that_import_back():
    """Pickling by reference is valid exactly when the unpickler can `import
    <name>` and get the same object. Everything else keeps dill's by-value
    behaviour, and `__main__` most of all: `python -m pkg` gives it a real
    `__spec__`, so a rule reading only `__spec__` would quietly change how
    dill treats the user's own script."""
    from unsloth.import_fixes import _dill_module_is_importable_by_name

    real = sys.modules["json"]
    assert _dill_module_is_importable_by_name(real)

    orphan = types.ModuleType("not_in_sys_modules")
    orphan.__spec__ = types.SimpleNamespace(name = "not_in_sys_modules", origin = "/x.py")
    assert not _dill_module_is_importable_by_name(orphan)

    no_spec = types.ModuleType("json_lookalike")
    sys.modules["json_lookalike"] = no_spec
    try:
        assert not _dill_module_is_importable_by_name(no_spec)
    finally:
        del sys.modules["json_lookalike"]

    # `__main__`, and it has to be SYNTHESISED rather than read off this
    # process. Under pytest `sys.modules["__main__"]` is a console script whose
    # `__spec__` is None, so it is already refused by the rule above and an
    # assertion on it passes without ever exercising the exclusion -- which is
    # exactly how a mutation that deleted the exclusion survived this file once.
    # `python -m pkg` gives `__main__` a real spec, and that is the case that
    # matters.
    for hostile in ("__main__", "__mp_main__"):
        fake = types.ModuleType(hostile)
        fake.__spec__ = types.SimpleNamespace(name = hostile, origin = f"/somewhere/pkg/{hostile}.py")
        previous = sys.modules.get(hostile)
        sys.modules[hostile] = fake
        try:
            assert not _dill_module_is_importable_by_name(fake), (
                f"{hostile} would be pickled by reference, which changes dill's "
                "contract for the user's own code"
            )
        finally:
            if previous is None:
                sys.modules.pop(hostile, None)
            else:
                sys.modules[hostile] = previous

    namespace_like = types.ModuleType("namespace_like")
    namespace_like.__spec__ = types.SimpleNamespace(name = "namespace_like", origin = None)
    sys.modules["namespace_like"] = namespace_like
    try:
        assert not _dill_module_is_importable_by_name(
            namespace_like
        ), "a module with no file backing it is not safely importable by name"
    finally:
        del sys.modules["namespace_like"]


def test_the_fix_is_called_on_every_import_path():
    """It lives outside the MLX/GPU branch on purpose: the layout that triggers
    this is a property of the install, not of the accelerator, and `unsloth`'s
    `__init__` picks one of those two branches and never both."""
    import ast

    source = (REPO / "unsloth" / "__init__.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)

    def _calls_it(nodes):
        return any(
            isinstance(node, ast.ImportFrom)
            and any(a.name == "fix_dill_module_by_value_pickling" for a in node.names)
            for parent in nodes
            for node in ast.walk(parent)
        )

    # MODULE BODY ONLY. A `try` at column zero is fine; an `if _IS_MLX` is not,
    # because the other branch would then never apply the fix.
    assert _calls_it(tree.body), (
        "the fix is not reached from the module body, so one of the two import "
        "paths runs without it"
    )
    called = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_fix_dill"
    ]
    assert called, "the fix is imported and never called"


def test_the_dill_session_binding_is_patched_too(tmp_path):
    """`dill.session` binds the name at import time. A patch that only touches
    `dill._dill` leaves session pickling on the old predicate, which is the
    kind of half-applied fix that works until someone calls `dump_module`."""
    got = _run_on_hostile_tree(tmp_path, apply = True)
    assert got["applied"] is True
    assert got["session_binding_patched"] is True
