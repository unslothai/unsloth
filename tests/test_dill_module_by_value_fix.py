# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Off-prefix installs make dill pickle modules by value, which fails on pyarrow's MonthDayNano."""

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


_HOSTILE_TREE = {
    # The gate answers from the module spec, so this file is never executed.
    "pyarrow.py": "VERSION = '0'\n",
    # Mimics pyarrow's `MonthDayNano`: `__module__ = "builtins"` plus a self-reference, which
    # puts it in dill's postproc list so the second encounter takes `save_global`.
    "ovmod.py": textwrap.dedent(
        """
        class Sneaky:
            pass

        Sneaky.self_ref = Sneaky
        Sneaky.__module__ = "builtins"
        """
    ),
    # Nested like datasets' `create_arrowTable`, so dill saves it by value and walks the globals;
    # a module-level function would be saved by reference and never reach pyarrow.
    "ovuser.py": textwrap.dedent(
        """
        import ovmod

        def outer():
            def create_arrowTable():
                return ovmod.Sneaky
            return create_arrowTable
        """
    ),
    # The user's own module beside the deps (`pip install --target .`): no distribution claims it.
    "projcfg.py": "VALUE = 1\n",
    # Two distributions so both `top_level.txt` and RECORD readers are exercised.
    "pyarrow-0.0.dist-info/RECORD": "pyarrow.py,,\npyarrow-0.0.dist-info/RECORD,,\n",
    "ovdep-0.0.dist-info/top_level.txt": "ovmod\novuser\n",
    "ovdep-0.0.dist-info/RECORD": "ovmod.py,,\novuser.py,,\n",
}

# A second off-prefix layer, so roots must be found from sys.path.
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
    """Child runs in a throwaway venv beside the overlay, so the overlay is off sys.prefix on any host."""
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
        # Leaving the second layer's metadata would keep the patch alive.
        for layer in (overlay, second):
            for meta in layer.glob("*.dist-info"):
                shutil.rmtree(meta)
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding = "utf-8")

    # The child venv inherits the base prefix's site-packages, so add this one for dill; after the
    # overlay, and a real site-packages dir, so dill itself stays pickled by reference.
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
    """Negative control: dill must still break on an off-prefix install, or the fix is dead weight."""
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
    """Project code sharing the install root stays by value; installed metadata decides which side."""
    got = _run_on_hostile_tree(tmp_path, apply = True)
    assert got["applied"] is True
    assert got["by_reference"] == {
        "pyarrow": True,
        "ovmod": True,
        "projcfg": False,
        "secondlayer": True,
        "secondproj": False,
    }, got["by_reference"]


def test_a_root_with_no_installed_metadata_is_left_alone(tmp_path):
    """With no installed metadata, decline rather than guess which files are dependencies."""
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
    """Widen only to modules that import back by name; reading __spec__ alone also catches __main__."""
    from unsloth.import_fixes import _dill_module_is_importable_by_name

    # Every call carries the install roots and their names; `json` stands in for a library.
    package_dir = os.path.dirname(os.path.realpath(sys.modules["json"].__file__ or ""))
    roots = (os.path.dirname(package_dir),)
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

    # `__main__` is synthesised: under pytest it is a console script with no spec, which would
    # pass without exercising the exclusion.
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
    """Statements that run on every import, one try level included; shared with the control on purpose."""
    import ast
    for node in body:
        if isinstance(node, ast.Try):
            yield from _unconditional(node.body)
        elif not isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
            yield node


def test_the_fix_is_called_on_every_import_path():
    """Checks the call is unconditional at top level; moving it under `if _IS_MLX:` must fail."""
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
    """Project modules outside the install root stay by value, keeping their state in fingerprints."""
    from unsloth.import_fixes import (
        _dill_install_root,
        _dill_module_is_importable_by_name,
    )

    # Native paths: on Windows `os.path.realpath("/opt")` is drive-qualified.
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
    # Root containment cannot separate a co-located user module from `library`; metadata can.
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
        assert not _dill_module_is_importable_by_name(library)
    finally:
        del sys.modules["pretend_library"]
        del sys.modules["pretend_project"]
        del sys.modules["pretend_colocated"]


def test_only_recorded_files_are_treated_as_dependency_owned(tmp_path):
    """Only recorded paths are dependency-owned; a top-level name would also claim co-located files."""
    from unsloth.import_fixes import _dill_distribution_paths

    root = tmp_path / "target"
    (root / "withtop-1.0.dist-info").mkdir(parents = True)
    (root / "withtop-1.0.dist-info" / "top_level.txt").write_text(
        "pkgone\n\n# comment\n", encoding = "utf-8"
    )
    # A name is honoured only where it resolves to exactly one file on disk.
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
    # RECORD wins: the name fallback would claim the whole `bothns` directory.
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

    # A package name cannot say which of a directory's contents were installed, so it is declined.
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

    assert not any(r == "ns" for r in rel), (
        "the namespace directory is claimed wholesale, so a co-located "
        "ns/myconfig.py would be treated as dependency-owned"
    )

    assert _dill_distribution_paths(str(tmp_path / "absent")) == set()


def test_stripped_bytecode_answers_to_its_recorded_source(tmp_path):
    """A bytecode-only install must match the .py its RECORD names, or the package stays by value."""
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
    """A package named only in top_level.txt claims nothing; a single-module name maps to its one .py."""
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
    """On bytecode-only installs the origin ends in __init__.pyc, not __init__.py; match both."""
    from unsloth.import_fixes import _dill_install_root

    # Platform separators: a POSIX literal fails against a drive-qualified Windows path.
    layer = tmp_path / "layer"
    expected = os.path.realpath(str(layer))
    assert _dill_install_root(str(layer / "pyarrow" / "__init__.pyc")) == expected
    assert _dill_install_root(str(layer / "pyarrow" / "__init__.py")) == expected
    assert _dill_install_root(str(layer / "dill.py")) == expected


def test_the_gate_reads_the_literal_path_the_way_dill_does(tmp_path):
    """Match 'site-packages' against the literal __file__, as dill does; a resolved symlink hides it."""
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

    # The venv root is an ancestor of tmp, so move the sys prefixes aside or the prefix rule answers.
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
