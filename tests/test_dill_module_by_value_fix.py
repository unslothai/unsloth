# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

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
import shutil
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
    # The pyarrow stand-in.
    # The fix's gate asks whether `datasets` or `pyarrow` resolve to somewhere dill pickles by value, and answers from
    # the SPEC, so this file is never executed and needs no content.
    "pyarrow.py": "VERSION = '0'\n",
    # The class that cannot be pickled by reference.
    # and a self-reference, which puts it in dill's postproc list so the second encounter takes `save_global` rather
    # than writing the class out by value and succeeding.
    "ovmod.py": textwrap.dedent(
        """
        class Sneaky:
            pass

        Sneaky.self_ref = Sneaky
        Sneaky.__module__ = "builtins"
        """
    ),
    # The function dill has to save, in the shape datasets uses.
    # NESTED, and that is the whole reason this reproduces: `_save_arrowTable` defines `create_arrowTable` inside
    # itself, dill's `_locate_function` cannot find a `<locals>` qualname at module level, so it saves BY VALUE, walks
    # the globals with `recurse=True` and reaches the module.
    # which is why datasets 4.3.0, whose reducer skips this path, is unaffected.
    "ovuser.py": textwrap.dedent(
        """
        import ovmod

        def outer():
            def create_arrowTable():
                return ovmod.Sneaky
            return create_arrowTable
        """
    ),
    # The user's OWN module, in the same directory, as `pip install --target .` and a Lambda bundle produce.
    "projcfg.py": "VALUE = 1\n",
    # What pip writes beside the packages it installs.
    # Two distributions so both readers are exercised: `top_level.txt`, and RECORD, the only metadata a modern wheel is
    # guaranteed to carry.
    "pyarrow-0.0.dist-info/RECORD": "pyarrow.py,,\npyarrow-0.0.dist-info/RECORD,,\n",
    "ovdep-0.0.dist-info/top_level.txt": "ovmod\novuser\n",
    "ovdep-0.0.dist-info/RECORD": "ovmod.py,,\novuser.py,,\n",
}

# A SECOND off-prefix layer, holding a recorded dependency and nothing the gate looks for, so the roots search has to
# find it from sys.path rather than only from wherever `datasets` or `pyarrow` happens to live.
_SECOND_LAYER = {
    "secondlayer.py": "V = 0\n",
    "secondproj.py": "VALUE = 1\n",
    "seconddep-0.0.dist-info/RECORD": "secondlayer.py,,\n",
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

    # Asked of dill's LIVE predicate, so it reports what dill will really do.
    # `projcfg` is the user's own module sitting in the same directory as the
    # dependencies, which is what `pip install --target .` produces: it must
    # stay by value or its mutable state drops out of every fingerprint.
    import pyarrow, ovmod, projcfg, secondlayer, secondproj
    out["by_reference"] = {
        name: bool(_core._is_builtin_module(sys.modules[name]))
        for name in ("pyarrow", "ovmod", "projcfg", "secondlayer", "secondproj")
    }

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
    omit_metadata = False,
):
    """Build the tree OUTSIDE any sys prefix and run the driver against it."""
    overlay = tmp_path / "overlay_leg"  # deliberately not "site-packages"
    overlay.mkdir()
    for name, body in _HOSTILE_TREE.items():
        target = overlay / name
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_text(body, encoding = "utf-8")
    second = tmp_path / "overlay_second"
    second.mkdir()
    for name, body in _SECOND_LAYER.items():
        target = second / name
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_text(body, encoding = "utf-8")
    if omit_metadata:
        # Both layers: leaving the second one's metadata would keep the patch
        # alive and the "no metadata anywhere" case would never be exercised.
        for layer in (overlay, second):
            for meta in layer.glob("*.dist-info"):
                shutil.rmtree(meta)
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding = "utf-8")

    # The child venv inherits the BASE prefix's site-packages, not this interpreter's, so dill would otherwise be
    # missing when the tests run from a venv. Appended AFTER the overlay, and a real site-packages directory, so dill
    # itself stays on the by-reference side of its own rule.
    import sysconfig

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(overlay),
            str(second),
            sysconfig.get_paths()["purelib"],
            os.environ.get("PYTHONPATH", ""),
        ]
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
    assert (
        got["affected"] is True
    ), "the gate does not recognise this layout, so the fix would never install itself here"
    assert got["dumps"].startswith("PicklingError"), (
        "dill pickled the off-prefix module by reference unaided; the bug this "
        f"guards is gone or moved: {got['dumps']}"
    )
    assert "builtins.Sneaky" in got["dumps"]


def test_the_fix_makes_the_same_tree_picklable(tmp_path):
    got = _run_on_hostile_tree(tmp_path, apply = True)
    assert got["applied"] is True
    assert got["dumps"] == "ok", got["dumps"]


def test_a_co_located_project_module_keeps_its_by_value_state(tmp_path):
    """P1 from review, executed rather than reasoned about.

    `pip install --target .` and a Lambda deployment bundle put dependencies
    into the application's OWN directory, so the install root is shared with
    the user's code. Root containment alone would move `projcfg` to
    by-reference along with the libraries, its mutable state would leave the
    `recurse=True` fingerprint, and `datasets` would serve a stale cached
    result after `projcfg.VALUE` changed. Installed metadata is what tells the
    two apart, and this asks dill's live predicate which side each landed on.
    """
    got = _run_on_hostile_tree(tmp_path, apply = True)
    assert got["applied"] is True
    assert got["by_reference"] == {
        "pyarrow": True,
        "ovmod": True,
        "projcfg": False,
        # A recorded dependency in a SECOND off-prefix layer, found from sys.path rather than from wherever pyarrow
        # happens to live.
        "secondlayer": True,
        # And that layer's own unrecorded project module is untouched.
        "secondproj": False,
    }, got["by_reference"]


def test_a_root_with_no_installed_metadata_is_left_alone(tmp_path):
    """Nothing there says which files are dependencies, so nothing is widened.

    A hand-assembled vendor directory reaches the same crash, and the honest
    answer is to decline: the crash is loud and immediate, while guessing would
    silently pin fingerprints on whatever the user keeps beside it.
    """
    got = _run_on_hostile_tree(tmp_path, apply = True, omit_metadata = True)
    assert got["affected"] is True, "the layout is still the hostile one"
    assert (
        got["applied"] is False
    ), "the patch installed itself with no way to tell a dependency from the user's own module"
    assert got["dumps"].startswith("PicklingError"), got["dumps"]


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

    # Every call now carries the install ROOTS and the names installed there, because the widening is scoped to both;
    # `json` stands in for a library that landed in one.
    package_dir = os.path.dirname(os.path.realpath(sys.modules["json"].__file__ or ""))
    roots = (os.path.dirname(package_dir),)  # the package's install root
    installed = frozenset({os.path.realpath(sys.modules["json"].__file__ or ""), "/x.py"})
    real = sys.modules["json"]
    assert _dill_module_is_importable_by_name(real, installed)

    orphan = types.ModuleType("not_in_sys_modules")
    orphan.__spec__ = types.SimpleNamespace(name = "not_in_sys_modules", origin = "/x.py")
    assert not _dill_module_is_importable_by_name(orphan, installed)

    no_spec = types.ModuleType("json_lookalike")
    sys.modules["json_lookalike"] = no_spec
    try:
        assert not _dill_module_is_importable_by_name(no_spec, installed)
    finally:
        del sys.modules["json_lookalike"]

    # `__main__`, SYNTHESISED rather than read off this process: under pytest `sys.modules["__main__"]` is a console
    # script whose `__spec__` is None, so it is already refused above and an assertion on it passes without exercising
    # the exclusion -- which is how a mutation deleting the exclusion survived this file once. `python -m pkg` gives
    # it a real spec.
    for hostile in ("__main__", "__mp_main__"):
        fake = types.ModuleType(hostile)
        fake.__spec__ = types.SimpleNamespace(name = hostile, origin = f"/somewhere/pkg/{hostile}.py")
        previous = sys.modules.get(hostile)
        sys.modules[hostile] = fake
        try:
            assert not _dill_module_is_importable_by_name(fake, installed), (
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
            namespace_like, installed
        ), "a module with no file backing it is not safely importable by name"
    finally:
        del sys.modules["namespace_like"]


def _unconditional(body):
    """Statements that run on EVERY import, one level of `try` included.

    Shared by the rule below and by its negative control on purpose: a control
    that carries its own copy of the walker passes when the real one regresses,
    which is the shape of a guard that guards nothing.
    """
    import ast
    for node in body:
        if isinstance(node, ast.Try):
            yield from _unconditional(node.body)
        elif not isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
            yield node


def test_the_fix_is_called_on_every_import_path():
    """It lives outside the MLX/GPU branch on purpose: the layout that triggers
    this is a property of the install, not of the accelerator, and `unsloth`'s
    `__init__` picks one of those two branches and never both.

    The rule walks only UNCONDITIONAL top-level statements, plus the body of a
    top-level `try`, which is how every other fix in `__init__` is guarded. An
    earlier version walked every descendant of each module-body node, so moving
    the import inside `if _IS_MLX:` still passed -- the exact placement this
    exists to reject.
    """
    import ast

    source = (REPO / "unsloth" / "__init__.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)

    top = list(_unconditional(tree.body))
    imports = [
        node
        for node in top
        if isinstance(node, ast.ImportFrom)
        and any(a.name == "fix_dill_module_by_value_pickling" for a in node.names)
    ]
    assert imports, (
        "the fix is not imported from an unconditional top-level statement, so "
        "one of the two import paths runs without it"
    )
    called = [
        node
        for node in top
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_fix_dill"
    ]
    assert called, "the fix is imported and never called unconditionally"


def test_that_rule_rejects_a_one_sided_conditional():
    """The negative control for the rule above, because a walk that recurses
    into `if` bodies passes on the very placement being rejected."""
    import ast

    hidden = ast.parse(
        "if _IS_MLX:\n"
        "    try:\n"
        "        from .import_fixes import fix_dill_module_by_value_pickling as _fix_dill\n"
        "        _fix_dill()\n"
        "    except Exception:\n"
        "        pass\n"
    )

    top = list(_unconditional(hidden.body))
    assert not [
        node
        for node in top
        if isinstance(node, ast.ImportFrom)
        and any(a.name == "fix_dill_module_by_value_pickling" for a in node.names)
    ], "an import inside `if _IS_MLX:` is being counted as unconditional"


def test_a_project_module_outside_the_install_root_keeps_its_by_value_state(tmp_path):
    """P1 from review, and it is a fingerprint-correctness rule rather than a
    tidiness one.

    A user's own project module normally sits outside site-packages, so dill
    pickles it BY VALUE and its mutable state participates in a `recurse=True`
    fingerprint. Widening the predicate for every live module would flip that,
    and `config.VALUE = 2` would stop changing the fingerprint while `datasets`
    served a stale cached result. Only modules inside the install root that made
    the environment dill-hostile are moved back to by-reference.
    """
    from unsloth.import_fixes import (
        _dill_install_root,
        _dill_module_is_importable_by_name,
    )

    # Native paths, not POSIX literals: on Windows `os.path.realpath("/opt")` is `D:\\opt`, which took this test red on
    # the cross-platform lane while the code under it was fine.
    layer = tmp_path / "layer"
    elsewhere = tmp_path / "project"
    root = _dill_install_root(str(layer / "pyarrow" / "__init__.py"))
    assert root == os.path.realpath(str(layer))
    assert _dill_install_root(str(layer / "dill.py")) == os.path.realpath(str(layer))
    assert _dill_install_root(None) is None

    library = types.ModuleType("pretend_library")
    library.__spec__ = types.SimpleNamespace(
        name = "pretend_library", origin = str(layer / "pretend_library.py")
    )
    project = types.ModuleType("pretend_project")
    project.__spec__ = types.SimpleNamespace(
        name = "pretend_project", origin = str(elsewhere / "pretend_project.py")
    )
    # The user's own module in the SAME directory as the dependencies. Root
    # containment cannot separate it from `library`; installed metadata can.
    colocated = types.ModuleType("pretend_colocated")
    colocated.__spec__ = types.SimpleNamespace(
        name = "pretend_colocated", origin = str(layer / "pretend_colocated.py")
    )
    sys.modules["pretend_library"] = library
    sys.modules["pretend_project"] = project
    sys.modules["pretend_colocated"] = colocated
    try:
        installed = frozenset({os.path.realpath(str(layer / "pretend_library.py"))})
        assert _dill_module_is_importable_by_name(library, installed)
        assert not _dill_module_is_importable_by_name(project, installed), (
            "a project module outside the install root would be pickled by "
            "reference, so its mutable state would drop out of the fingerprint"
        )
        assert not _dill_module_is_importable_by_name(colocated, installed), (
            "a co-located project module no distribution recorded would be "
            "pickled by reference, so `config.VALUE = 2` would stop changing "
            "the fingerprint and datasets would serve a stale cached result"
        )
        # Nothing installed anywhere means no widening whatsoever.
        assert not _dill_module_is_importable_by_name(library)
    finally:
        del sys.modules["pretend_library"]
        del sys.modules["pretend_project"]
        del sys.modules["pretend_colocated"]


def test_only_recorded_files_are_treated_as_dependency_owned(tmp_path):
    """`_dill_distribution_paths`, driven against a real directory.

    Recorded PATHS, not top-level names. A name cannot separate an installed
    `google` distribution from a co-located `google/myconfig.py` that nothing
    installed, and it forces a guess about leading underscores that discards
    `_soundfile` along with `__pycache__`.
    """
    from unsloth.import_fixes import _dill_distribution_paths

    root = tmp_path / "target"
    (root / "withtop-1.0.dist-info").mkdir(parents = True)
    (root / "withtop-1.0.dist-info" / "top_level.txt").write_text(
        "pkgone\n\n# comment\n", encoding = "utf-8"
    )
    # The single module that name resolves to: a name is honoured only where it really is one file on disk.
    (root / "pkgone.py").write_text("X = 1\n", encoding = "utf-8")
    (root / "onlyrecord-1.0.dist-info").mkdir()
    (root / "onlyrecord-1.0.dist-info" / "RECORD").write_text(
        "ns/cloud/__init__.py,sha256=x,10\n"
        "_soundfile.py,sha256=u,9\n"
        "singlemod.py,sha256=y,4\n"
        "sourceless/__init__.pyc,sha256=z,8\n"
        "onlyrecord-1.0.dist-info/RECORD,,\n"
        "onlyrecord-1.0.data/scripts/thing,,\n"
        "__pycache__/singlemod.cpython-312.pyc,,\n",
        encoding = "utf-8",
    )
    # Both files, which is ordinary.
    # RECORD wins: running the name fallback too would claim the whole `bothns` directory and put a co-located
    # `bothns/myconfig.py` back on the dependency side.
    (root / "both-1.0.dist-info").mkdir()
    (root / "both-1.0.dist-info" / "RECORD").write_text("bothns/cloud.py,,\n", encoding = "utf-8")
    (root / "both-1.0.dist-info" / "top_level.txt").write_text("bothns\n", encoding = "utf-8")
    (root / "eggy.egg-info").mkdir()
    (root / "eggy.egg-info" / "installed-files.txt").write_text("../eggmod.py\n", encoding = "utf-8")
    (root / "myproj.py").write_text("VALUE = 1\n", encoding = "utf-8")

    files = _dill_distribution_paths(str(root))
    rel = {os.path.relpath(f, str(root)) for f in files}

    assert "ns/cloud/__init__.py".replace("/", os.sep) in rel
    assert "_soundfile.py" in rel, (
        "a leading underscore is not metadata: _soundfile and _multiprocess "
        "are real distributions' real modules, and dropping them leaves them "
        "pickled by value with the original PicklingError intact"
    )
    assert (
        "sourceless/__init__.pyc".replace("/", os.sep) in rel
    ), "a bytecode-only deployment records .pyc, and it is just as installed"
    assert "singlemod.py" in rel and "eggmod.py" in rel
    assert "myproj.py" not in rel, "a file no distribution recorded is claimed"
    assert not any("dist-info" in r or ".data" in r or "__pycache__" in r for r in rel)

    # The fallback is honoured only where the name resolves to ONE file: a
    # package name cannot say which of the directory's contents were installed,
    # so it is declined rather than guessed.
    assert "pkgone.py" in rel
    assert not any(r == "pkgone" or r.startswith("pkgone" + os.sep) for r in rel), (
        "a top_level.txt package name claimed the whole directory, so a "
        "co-located module inside it counts as dependency-owned"
    )

    assert "bothns/cloud.py".replace("/", os.sep) in rel
    assert not any(
        r == "bothns"
        or (r.startswith("bothns" + os.sep) and r != os.path.join("bothns", "cloud.py"))
        for r in rel
    ), (
        "a distribution that ships both RECORD and top_level.txt had its name "
        "fallback applied too, so the whole directory is claimed and a "
        "co-located module inside it counts as dependency-owned"
    )

    # A shared namespace is exactly what the name-based version could not do.
    assert not any(r == "ns" for r in rel), (
        "the namespace directory is claimed wholesale, so a co-located "
        "ns/myconfig.py would be treated as dependency-owned"
    )

    assert _dill_distribution_paths(str(tmp_path / "absent")) == set()


def test_stripped_bytecode_answers_to_its_recorded_source(tmp_path):
    """A sourceless install keeps RECORD naming the `.py` it built from.

    `compileall` then deleting the sources leaves `pkg/__init__.pyc` live while
    the retained wheel RECORD still says `pkg/__init__.py`. An exact match then
    leaves the installed package by value and the original PicklingError
    stands, on exactly the stripped layers this patch is for.
    """
    from unsloth.import_fixes import _dill_module_is_importable_by_name

    layer = tmp_path / "layer"
    recorded = os.path.realpath(str(layer / "pkg" / "__init__.py"))
    files = frozenset({recorded})

    module = types.ModuleType("pkg")
    module.__spec__ = types.SimpleNamespace(name = "pkg", origin = str(layer / "pkg" / "__init__.pyc"))
    unrecorded = types.ModuleType("otherpkg")
    unrecorded.__spec__ = types.SimpleNamespace(
        name = "otherpkg", origin = str(layer / "otherpkg" / "__init__.pyc")
    )
    sys.modules["pkg"] = module
    sys.modules["otherpkg"] = unrecorded
    try:
        assert _dill_module_is_importable_by_name(module, files), (
            "the live .pyc is not matched to the .py its own metadata "
            "recorded, so a stripped layer keeps the crash"
        )
        assert not _dill_module_is_importable_by_name(
            unrecorded, files
        ), "a .pyc whose source was never recorded is claimed anyway"
    finally:
        del sys.modules["pkg"]
        del sys.modules["otherpkg"]


def test_a_top_level_package_name_alone_claims_nothing(tmp_path):
    """`top_level.txt` with no file list can only be honoured for one file.

    `dill` -> `dill.py` is unambiguous. `google` names a directory whose
    contents the metadata cannot account for, so honouring it would put a
    co-located `google/myconfig.py` on the dependency side -- the same hole a
    top-level name leaves anywhere else, arriving through the fallback. The
    package case is declined, so the worst outcome is the original loud
    PicklingError rather than a silently pinned fingerprint.
    """
    from unsloth.import_fixes import _dill_distribution_paths

    root = tmp_path / "layer"
    (root / "google").mkdir(parents = True)
    (root / "google" / "cloud.py").write_text("X = 1\n", encoding = "utf-8")
    (root / "google" / "myconfig.py").write_text("VALUE = 1\n", encoding = "utf-8")
    (root / "single.py").write_text("X = 1\n", encoding = "utf-8")
    (root / "legacy-1.0.egg-info").mkdir()
    (root / "legacy-1.0.egg-info" / "top_level.txt").write_text(
        "google\nsingle\n", encoding = "utf-8"
    )

    files = _dill_distribution_paths(str(root))
    rel = {os.path.relpath(f, str(root)) for f in files}
    assert "single.py" in rel, "an unambiguous single-module name was dropped"
    assert not any(r.startswith("google") for r in rel), (
        "the package name was honoured, so google/myconfig.py is claimed by "
        "metadata that never mentioned it"
    )


def test_a_project_module_under_a_shared_namespace_stays_by_value(tmp_path):
    """The namespace case, end to end through the ownership test."""
    from unsloth.import_fixes import (
        _dill_distribution_paths,
        _dill_module_is_importable_by_name,
    )

    root = tmp_path / "layer"
    (root / "ns").mkdir(parents = True)
    (root / "ns" / "cloud.py").write_text("X = 1\n", encoding = "utf-8")
    (root / "ns" / "myconfig.py").write_text("VALUE = 1\n", encoding = "utf-8")
    (root / "nsdist-1.0.dist-info").mkdir()
    (root / "nsdist-1.0.dist-info" / "RECORD").write_text("ns/cloud.py,,\n", encoding = "utf-8")
    installed = _dill_distribution_paths(str(root))

    for name, filename, expected in (
        ("ns.cloud", "cloud.py", True),
        ("ns.myconfig", "myconfig.py", False),
    ):
        module = types.ModuleType(name)
        module.__spec__ = types.SimpleNamespace(name = name, origin = str(root / "ns" / filename))
        sys.modules[name] = module
        try:
            assert _dill_module_is_importable_by_name(module, installed) is expected, (
                f"{name} landed on the wrong side; a shared namespace's first "
                "component says nothing about who installed the submodule"
            )
        finally:
            del sys.modules[name]


def test_metadata_in_one_root_cannot_vouch_for_a_file_in_another(tmp_path):
    """Two off-prefix layers, each with its own `config`.

    Unioning the two roots' ownership lets layer A's installed `config`
    distribution certify layer B's project `config.py`, whose mutable state
    then leaves the fingerprint.
    """
    from unsloth.import_fixes import (
        _dill_distribution_paths,
        _dill_module_is_importable_by_name,
    )

    a, b = tmp_path / "a", tmp_path / "b"
    (a / "config-1.0.dist-info").mkdir(parents = True)
    (a / "config-1.0.dist-info" / "RECORD").write_text("config.py,,\n", encoding = "utf-8")
    (a / "config.py").write_text("X = 1\n", encoding = "utf-8")
    b.mkdir()
    (b / "other-1.0.dist-info").mkdir()
    (b / "other-1.0.dist-info" / "RECORD").write_text("other.py,,\n", encoding = "utf-8")
    (b / "other.py").write_text("X = 1\n", encoding = "utf-8")
    (b / "config.py").write_text("VALUE = 1\n", encoding = "utf-8")

    installed = set()
    for root in (a, b):
        installed |= _dill_distribution_paths(str(root))

    module = types.ModuleType("config")
    module.__spec__ = types.SimpleNamespace(name = "config", origin = str(b / "config.py"))
    sys.modules["config"] = module
    try:
        assert not _dill_module_is_importable_by_name(module, installed), (
            "the project config.py in layer B is claimed by layer A's config "
            "distribution, so changes to it stop changing the fingerprint"
        )
    finally:
        del sys.modules["config"]


def test_a_bytecode_only_package_still_finds_its_metadata(tmp_path):
    """`find_spec(...).origin` ends in `__init__.pyc` on a sourceless install.

    Matching `__init__.py` exactly left the root one level too deep, no sibling
    metadata was found, and the fix declined on exactly the deployments -- a
    stripped Lambda layer -- it was written for.
    """
    from unsloth.import_fixes import _dill_install_root

    # Built with the platform's own separators; a POSIX literal compares
    # against a drive-qualified path on Windows and fails for the wrong reason.
    layer = tmp_path / "layer"
    expected = os.path.realpath(str(layer))
    assert _dill_install_root(str(layer / "pyarrow" / "__init__.pyc")) == expected
    assert _dill_install_root(str(layer / "pyarrow" / "__init__.py")) == expected
    assert _dill_install_root(str(layer / "dill.py")) == expected


def test_the_gate_reads_the_literal_path_the_way_dill_does(tmp_path):
    """dill tests `'site-packages' in module.__file__`, the literal one.

    It resolves the path only for the sys-prefix comparisons. Searching the
    RESOLVED path here too answered "not affected" for a PYTHONPATH entry that
    is a symlink into a directory whose name contains site-packages, while dill
    went on pickling it by value -- and since this gate decides whether dill's
    own predicate is ever consulted, the crash simply stood.
    """
    from unsloth.import_fixes import _dill_path_pickles_by_value

    target = tmp_path / "a-site-packages-cache" / "libs"
    target.mkdir(parents = True)
    (target / "pyarrow.py").write_text("V = 0\n", encoding = "utf-8")
    link = tmp_path / "layer"
    try:
        link.symlink_to(target, target_is_directory = True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("this platform cannot create the symlink this needs")

    literal = str(link / "pyarrow.py")
    assert "site-packages" not in literal
    assert "site-packages" in os.path.realpath(literal)

    # This box's virtualenv root is an ancestor of tmp, so without moving the sys prefixes aside the gate answers False
    # on the prefix rule and the site-packages rule is never reached -- a green test measuring nothing.
    names = ("base_prefix", "base_exec_prefix", "exec_prefix", "prefix", "real_prefix")
    saved = {n: getattr(sys, n) for n in names if hasattr(sys, n)}
    elsewhere = str(tmp_path / "not-a-prefix")
    try:
        for n in names:
            setattr(sys, n, elsewhere)
        assert _dill_path_pickles_by_value(literal) is True, (
            "the gate resolves the path before searching for site-packages, "
            "so it reports unaffected where dill still pickles by value"
        )
        assert _dill_path_pickles_by_value(str(target / "pyarrow.py")) is False
        assert _dill_path_pickles_by_value(os.path.join(elsewhere, "x.py")) is False
    finally:
        for n in names:
            if n in saved:
                setattr(sys, n, saved[n])
            else:
                delattr(sys, n)
