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
"""A broken TensorFlow / Flax install must not break importing Unsloth. Transformers
4.x imports either backend merely because it is installed, via `processing_utils`
-> `image_transforms`."""

import ast
import functools
import os
import pathlib
import subprocess
import sys
import textwrap
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_INIT = _ROOT / "unsloth" / "__init__.py"
_SOURCE = _INIT.read_text(encoding = "utf-8")

_BROKEN_TF = "raise ImportError(\"cannot import name 'runtime_version' from 'google.protobuf'\")\n"


def _fake_tensorflow(tmp_path):
    """A `tensorflow` Transformers detects but cannot import. `_tf_available` needs
    a `find_spec` hit *and* an installed version >= 2, hence the
    `.dist-info/METADATA`. Never touches site-packages."""
    site = tmp_path / "fakesite"
    package = site / "tensorflow"
    package.mkdir(parents = True)
    (package / "__init__.py").write_text(_BROKEN_TF, encoding = "utf-8")
    dist = site / "tensorflow-2.20.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: tensorflow\nVersion: 2.20.0\n",
        encoding = "utf-8",
    )
    return site


def _working_tensorflow(tmp_path):
    """A `tensorflow` that Transformers detects *and* imports cleanly."""
    site = tmp_path / "worksite"
    package = site / "tensorflow"
    package.mkdir(parents = True)
    (package / "__init__.py").write_text('__version__ = "2.20.0"\n', encoding = "utf-8")
    dist = site / "tensorflow-2.20.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: tensorflow\nVersion: 2.20.0\n",
        encoding = "utf-8",
    )
    return site


# Every variable Transformers reads to pick a backend.
# `USE_TORCH` belongs here too: `_tf_available` is gated on `USE_TORCH not in ENV_VARS_TRUE_VALUES`, so an inherited
# `USE_TORCH=1` forces it False whatever is on the path.
_BACKEND_ENV = ("USE_TF", "USE_FLAX", "USE_TORCH", "FORCE_TF_AVAILABLE")


def _run(
    code,
    site = None,
    **env,
):
    """Run `code` in a fresh interpreter, so no module state leaks between cases."""
    path = [str(_ROOT)] + ([str(site)] if site is not None else [])
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    # Importing Unsloth sets USE_TF/USE_FLAX here; each test says its own.
    clean = {k: v for k, v in os.environ.items() if k not in _BACKEND_ENV}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output = True,
        text = True,
        env = dict(clean, PYTHONPATH = os.pathsep.join(path), **env),
        timeout = 900,
    )


@functools.cache
def _unsloth_is_importable():
    return _run("import unsloth").returncode == 0


def _needs_unsloth():
    if not _unsloth_is_importable():
        pytest.skip("unsloth is not importable in this environment")


_V4_ONLY = ("_tf_available", "_flax_available", "USE_TF")


@functools.cache
def _v4_names():
    """Which v4-only `import_utils` names the installed Transformers still has.
    5.x dropped TF/Flax and these names with them, so reading one there is an
    `AttributeError` rather than a failing assertion."""
    out = _run(
        """
        from transformers.utils import import_utils
        for name in {names!r}:
            print("HAS", name, hasattr(import_utils, name))
        """.format(names = _V4_ONLY),
    )
    if out.returncode != 0:
        return {}
    found = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "HAS" and parts[1] in _V4_ONLY:
            found[parts[1]] = parts[2] == "True"
    return found


def _needs_v4_flag(name):
    if not _v4_names().get(name, False):
        pytest.skip(f"transformers here has no import_utils.{name} (5.x dropped TF/Flax)")


def _exec_guard(modules, environ):
    """Execute the opt-out block against a synthetic `sys.modules` / environment."""
    scope = {
        "os": types.SimpleNamespace(environ = environ),
        "sys": types.SimpleNamespace(modules = modules),
    }
    exec(ast.unparse(_guard_block()), scope)


def _guard_block():
    """The `if "transformers" not in sys.modules:` block, or None."""
    for node in ast.parse(_SOURCE).body:
        if not isinstance(node, ast.If):
            continue
        if "transformers" in ast.unparse(node.test) and "sys.modules" in ast.unparse(node.test):
            return node
    return None


def test_the_backends_are_opted_out_of_before_transformers_loads():
    block = _guard_block()
    assert block is not None, "the opt-out block is gone"
    # Run the block rather than grep its source: grepping only tracked the spelling.
    environ = {}
    _exec_guard({}, environ)
    assert environ.get("USE_TF") == "0"
    assert environ.get("USE_FLAX") == "0"
    # It has to sit above every `transformers` import here, or it is a no-op.
    first_import = min(
        (
            node.lineno
            for node in ast.walk(ast.parse(_SOURCE))
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("transformers")
        ),
        default = 10**9,
    )
    assert block.lineno < first_import


@pytest.mark.parametrize("value", ["1", "true", "YES", "On"])
def test_an_explicit_choice_is_never_overwritten(value):
    """Someone who wants TF in-process keeps it, in any spelling Transformers
    accepts as true."""
    environ = {"USE_TF": value, "USE_FLAX": value}
    _exec_guard({}, environ)
    assert environ["USE_TF"] == value
    assert environ["USE_FLAX"] == value


@pytest.mark.parametrize("value", ["AUTO", "auto", "Auto"])
def test_auto_is_overwritten_because_transformers_reads_it_as_enabled(value):
    """`AUTO` is what an unset variable means to Transformers: enable if installed,
    the exact state this guard prevents and the one `setdefault` used to keep."""
    environ = {"USE_TF": value, "USE_FLAX": value}
    _exec_guard({}, environ)
    assert environ["USE_TF"] == "0"
    assert environ["USE_FLAX"] == "0"


def test_force_tf_available_alone_counts_as_an_opt_in():
    """`FORCE_TF_AVAILABLE=1` asks for TensorFlow without also asking Transformers
    to disable PyTorch, so it is the spelling a real user reaches for."""
    environ = {"FORCE_TF_AVAILABLE": "1"}
    _exec_guard({}, environ)
    assert environ.get("USE_TF") != "0", environ


def test_a_value_that_means_off_is_normalised_rather_than_preserved():
    """Anything Transformers does not read as true already means off, so
    rewriting it to "0" changes no behaviour."""
    environ = {"USE_TF": "0", "USE_FLAX": "false"}
    _exec_guard({}, environ)
    assert environ["USE_TF"] == "0"
    assert environ["USE_FLAX"] == "0"


@pytest.mark.parametrize("value", ["0", "1"])
def test_transformers_reads_the_variable_from_the_environment(value):
    """The half in Transformers: read once at import, so ours must land first."""
    _needs_v4_flag("USE_TF")
    env = dict(os.environ, USE_TF = value)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from transformers.utils import import_utils; print(import_utils.USE_TF)",
        ],
        capture_output = True,
        text = True,
        env = env,
        timeout = 300,
    )
    if out.returncode != 0:
        pytest.skip(f"transformers not importable here: {out.stderr.strip()[:200]}")
    assert out.stdout.strip() == value


def test_the_variables_are_written_even_once_transformers_is_loaded():
    """Inert against a fully imported Transformers, but the partly-imported case
    below needs them, and this branch cannot tell the two apart."""
    environ = {}
    _exec_guard({"transformers": object()}, environ)
    assert environ == {"USE_TF": "0", "USE_FLAX": "0"}


def test_the_environment_branch_honours_an_already_imported_backend():
    """Nothing imported: opt both out. One imported: leave that one to its user."""
    for modules, expected in (
        ({}, {"USE_TF": "0", "USE_FLAX": "0"}),
        ({"tensorflow": object()}, {"USE_FLAX": "0"}),
        ({"jax": object()}, {"USE_TF": "0"}),
        ({"flax": object()}, {"USE_TF": "0"}),
        ({"tensorflow": object(), "flax": object()}, {}),
    ):
        environ = {}
        _exec_guard(dict(modules), environ)
        assert environ == expected, modules


def test_a_broken_backend_still_loses_when_transformers_came_first(tmp_path):
    """The regression: `_tf_available` was cached True before Unsloth got a say."""
    _needs_unsloth()
    out = _run(
        """
        import transformers
        from transformers.utils import import_utils
        assert getattr(import_utils, "_tf_available", None) is not False, \\
            "the fake tensorflow was not detected"
        import sys, unsloth
        print("TF_LOADED", "tensorflow" in sys.modules)
        print("TF_AVAILABLE", getattr(import_utils, "_tf_available", "ABSENT"))
        """,
        site = _fake_tensorflow(tmp_path),
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "TF_LOADED False" in out.stdout, out.stdout
    if _v4_names().get("_tf_available"):
        assert "TF_AVAILABLE False" in out.stdout, out.stdout
    else:
        assert "TF_AVAILABLE ABSENT" in out.stdout, out.stdout


def test_the_environment_path_still_covers_the_transformers_not_loaded_case(tmp_path):
    _needs_unsloth()
    # `getattr`, because 5.x has no such flag; "TF never loads" still asserts.
    out = _run(
        """
        import unsloth, os, sys
        from transformers.utils import import_utils
        print("ENV_USE_TF", os.environ.get("USE_TF"))
        print("USE_TF", getattr(import_utils, "USE_TF", "ABSENT"))
        print("TF_LOADED", "tensorflow" in sys.modules)
        """,
        site = _fake_tensorflow(tmp_path),
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "ENV_USE_TF 0" in out.stdout, out.stdout
    assert "TF_LOADED False" in out.stdout, out.stdout
    if _v4_names().get("USE_TF"):
        assert "USE_TF 0" in out.stdout, out.stdout


def _run_env_branch(tmp_path, preamble, site, **env):
    """Run the real opt-out block with Transformers not yet imported. The
    `import tensorflow; import unsloth` order cannot be tested end to end here:
    leaving TF enabled makes Transformers import `TFPreTrainedModel`, which needs
    a genuine `tf.keras` (and h5py), not a stub."""
    guard = tmp_path / "env_guard.py"
    guard.write_text(ast.unparse(_guard_block()), encoding = "utf-8")
    return _run(
        f"""
        import os, sys
        {preamble}
        assert "transformers" not in sys.modules, "the env-var branch needs it absent"
        exec(open({str(guard)!r}).read())
        print("ENV_USE_TF", os.environ.get("USE_TF"))
        print("ENV_USE_FLAX", os.environ.get("USE_FLAX"))
        """,
        site = site,
        **env,
    )


def test_an_imported_backend_is_not_opted_out_when_transformers_comes_later(tmp_path):
    """The env-var branch has to honour an in-use backend too, and `setdefault`
    cannot: nothing set USE_TF, so there is no explicit value to defer to."""
    site = _working_tensorflow(tmp_path)
    out = _run_env_branch(tmp_path, "import tensorflow", site)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "ENV_USE_TF None" in out.stdout, out.stdout
    assert "ENV_USE_FLAX 0" in out.stdout, out.stdout
    # Only 4.x has a flag to read, and `_v4_names()` is empty without Transformers.
    if _v4_names().get("_tf_available"):
        probe = _run(
            """
            from transformers.utils import import_utils
            print("TF_AVAILABLE", getattr(import_utils, "_tf_available", "ABSENT"))
            """,
            site = site,
        )
        assert probe.returncode == 0, probe.stderr[-3000:]
        assert "TF_AVAILABLE True" in probe.stdout, probe.stdout


def test_a_broken_uninvolved_backend_is_still_opted_out(tmp_path):
    """The protection this file exists for, in the same real-process harness."""
    out = _run_env_branch(tmp_path, "", _fake_tensorflow(tmp_path))
    assert out.returncode == 0, out.stderr[-3000:]
    assert "ENV_USE_TF 0" in out.stdout, out.stdout
    # The backend nobody is using still gets opted out.
    assert "ENV_USE_FLAX 0" in out.stdout, out.stdout


def _run_guard(tmp_path, preamble, **env):
    """Run the block against a real, already-imported Transformers (v4 only)."""
    _needs_v4_flag("_tf_available")
    guard = tmp_path / "guard.py"
    guard.write_text(ast.unparse(_guard_block()), encoding = "utf-8")
    return _run(
        f"""
        import os, sys, types, transformers
        from transformers.utils import import_utils
        print("BEFORE", import_utils._tf_available)
        {preamble}
        exec(open({str(guard)!r}).read())
        print("AFTER", import_utils._tf_available)
        """,
        site = _fake_tensorflow(tmp_path),
        **env,
    )


def test_an_explicit_opt_in_keeps_the_backend(tmp_path):
    """FORCE_TF_AVAILABLE=1 means the user wants TensorFlow; never sabotage that.
    Not USE_TF=1, which Transformers also reads as "disable PyTorch"."""
    out = _run_guard(tmp_path, "", FORCE_TF_AVAILABLE = "1")
    assert out.returncode == 0, out.stderr[-3000:]
    assert "BEFORE True" in out.stdout and "AFTER True" in out.stdout, out.stdout


def test_a_backend_already_in_use_is_left_alone(tmp_path):
    """`tensorflow` imported already: the user is using it, hands off."""
    out = _run_guard(tmp_path, 'sys.modules["tensorflow"] = types.ModuleType("tensorflow")')
    assert out.returncode == 0, out.stderr[-3000:]
    assert "BEFORE True" in out.stdout and "AFTER True" in out.stdout, out.stdout


def test_the_cached_flag_is_cleared_against_a_real_transformers(tmp_path):
    """Same harness, nothing opted in: the flag flips."""
    out = _run_guard(tmp_path, "")
    assert out.returncode == 0, out.stderr[-3000:]
    assert "BEFORE True" in out.stdout and "AFTER False" in out.stdout, out.stdout


def test_an_opt_in_that_was_consumed_and_restored_still_counts(tmp_path):
    """Transformers reads these once, at its own import, so a variable that was set,
    consumed and restored is still an opt-in `os.environ` no longer shows."""
    out = _run_guard(
        tmp_path,
        'del os.environ["FORCE_TF_AVAILABLE"]',
        FORCE_TF_AVAILABLE = "1",
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "BEFORE True" in out.stdout and "AFTER True" in out.stdout, out.stdout


def test_transformers_5x_has_neither_flag_and_nothing_raises():
    """5.x dropped both backends: no attribute to clear, no exception either."""
    import_utils = types.ModuleType("transformers.utils.import_utils")
    modules = {
        "transformers": types.ModuleType("transformers"),
        "transformers.utils.import_utils": import_utils,
    }
    _exec_guard(modules, {})
    assert not hasattr(import_utils, "_tf_available")
    assert not hasattr(import_utils, "_flax_available")


@pytest.mark.parametrize(
    "case",
    [
        test_an_explicit_opt_in_keeps_the_backend,
        test_a_backend_already_in_use_is_left_alone,
        test_the_cached_flag_is_cleared_against_a_real_transformers,
        test_an_opt_in_that_was_consumed_and_restored_still_counts,
        test_transformers_reads_the_variable_from_the_environment,
    ],
)
def test_the_v4_only_cases_skip_on_transformers_5x(monkeypatch, tmp_path, case):
    """With the flags gone, these read a name that no longer exists: skip, not error."""
    monkeypatch.setattr(sys.modules[__name__], "_v4_names", dict)
    kwargs = {"tmp_path": tmp_path} if "tmp_path" in case.__code__.co_varnames else {"value": "0"}
    with pytest.raises(pytest.skip.Exception) as caught:
        case(**kwargs)
    assert "5.x dropped TF/Flax" in str(caught.value)


def test_a_partly_imported_transformers_still_gets_the_variables():
    """`"transformers" in sys.modules` does not mean Transformers is ready: Python
    publishes a module object before executing its body, so a thread part-way
    through `import transformers` reaches the `else` branch with `import_utils`
    still absent. Nothing cached to clear there, so the environment is the lever."""
    environ = {}
    _exec_guard({"transformers": types.ModuleType("transformers")}, environ)
    assert environ == {"USE_TF": "0", "USE_FLAX": "0"}


@pytest.mark.parametrize(
    "case",
    [
        ({"USE_TF": "1"}, {"USE_TF": "1", "USE_FLAX": "0"}),
        ({"FORCE_TF_AVAILABLE": "yes"}, {"FORCE_TF_AVAILABLE": "yes", "USE_FLAX": "0"}),
        ({"USE_FLAX": "ON"}, {"USE_FLAX": "ON", "USE_TF": "0"}),
        ({"USE_TF": "AUTO"}, {"USE_TF": "0", "USE_FLAX": "0"}),
    ],
)
def test_the_partial_window_write_still_obeys_every_opt_in(case):
    """The new write is the same decision as the other branch, not a blunter one."""
    environ, expected = dict(case[0]), case[1]
    _exec_guard({"transformers": types.ModuleType("transformers")}, environ)
    assert environ == expected


def test_the_partial_window_write_leaves_an_imported_backend_alone():
    """A backend already in `sys.modules` is one in use, in this branch too."""
    for modules, expected in (
        ({"tensorflow": object()}, {"USE_FLAX": "0"}),
        ({"jax": object()}, {"USE_TF": "0"}),
        ({"tensorflow": object(), "flax": object()}, {}),
    ):
        environ = {}
        _exec_guard(dict(modules, transformers = types.ModuleType("transformers")), environ)
        assert environ == expected, modules


def test_a_cached_opt_in_also_blocks_the_partial_window_write():
    """`import_utils` present and opted in: neither the flag nor the variable moves."""
    import_utils = types.ModuleType("transformers.utils.import_utils")
    import_utils.USE_JAX = "1"
    import_utils.FORCE_TF_AVAILABLE = "1"
    environ = {}
    _exec_guard(
        {"transformers": object(), "transformers.utils.import_utils": import_utils},
        environ,
    )
    assert environ == {}


def test_the_subprocess_environment_drops_every_backend_variable(monkeypatch):
    """A runner that exports one of these must not decide the cases for us:
    `USE_TORCH=1` in the parent makes every "BEFORE True" case fail."""
    # Spelled out, so shortening `_BACKEND_ENV` fails here instead of narrowing.
    names = ("USE_TF", "USE_FLAX", "USE_TORCH", "FORCE_TF_AVAILABLE")
    for name in names:
        monkeypatch.setenv(name, "1")
    out = _run(
        """
        import os
        for name in {names!r}:
            print("ENV", name, os.environ.get(name))
        """.format(names = names),
    )
    assert out.returncode == 0, out.stderr[-3000:]
    for name in names:
        assert f"ENV {name} None" in out.stdout, out.stdout


def test_the_flags_are_cleared_only_when_the_backend_is_unused():
    import_utils = types.ModuleType("transformers.utils.import_utils")
    import_utils._tf_available = True
    import_utils._flax_available = True
    modules = {"transformers": object(), "transformers.utils.import_utils": import_utils}
    _exec_guard(modules, {})
    assert import_utils._tf_available is False
    assert import_utils._flax_available is False
    import_utils._flax_available = True
    _exec_guard(dict(modules, jax = object()), {})
    assert import_utils._flax_available is True
    # jax in play means Flax is genuinely in use.
    import_utils._flax_available = True
    _exec_guard(modules, {"USE_FLAX": "yes"})
    assert import_utils._flax_available is True
    for _var in ("USE_TF", "FORCE_TF_AVAILABLE"):
        import_utils._tf_available = True
        _exec_guard(modules, {_var: "1"})
        assert import_utils._tf_available is True, _var
    import_utils._tf_available = True
    _exec_guard(dict(modules, tensorflow = object()), {})
    assert import_utils._tf_available is True


def test_the_snapshot_transformers_kept_counts_as_an_opt_in():
    """Each variable in the name Transformers files it under: env `USE_FLAX` is
    stored as `USE_JAX`, so looking for a cached `USE_FLAX` finds nothing."""
    import_utils = types.ModuleType("transformers.utils.import_utils")
    modules = {"transformers": object(), "transformers.utils.import_utils": import_utils}
    for flag, cached in (
        ("_tf_available", "USE_TF"),
        ("_tf_available", "FORCE_TF_AVAILABLE"),
        ("_flax_available", "USE_JAX"),
    ):
        setattr(import_utils, flag, True)
        setattr(import_utils, cached, "1")
        _exec_guard(modules, {})
        assert getattr(import_utils, flag) is True, cached
        delattr(import_utils, cached)


def test_the_default_snapshot_is_not_an_opt_in():
    """All three default to `"AUTO"` when unset, which Transformers reads as "enable
    if installed": accepting it would make the guard a no-op on most machines."""
    import_utils = types.ModuleType("transformers.utils.import_utils")
    # An imported TensorFlow is one in use.
    import_utils._tf_available = True
    import_utils._flax_available = True
    import_utils.USE_TF = "AUTO"
    import_utils.FORCE_TF_AVAILABLE = "AUTO"
    import_utils.USE_JAX = "AUTO"
    _exec_guard({"transformers": object(), "transformers.utils.import_utils": import_utils}, {})
    assert import_utils._tf_available is False
    assert import_utils._flax_available is False


def test_the_snapshot_is_overwritten_while_import_utils_is_mid_body():
    """The window between `import_utils` copying the environment into `USE_TF` /
    `USE_JAX` (its lines 102-104) and deriving the flags (264 / 355)."""
    import_utils = types.ModuleType("transformers.utils.import_utils")
    import_utils.USE_TF = "AUTO"
    import_utils.FORCE_TF_AVAILABLE = "AUTO"
    import_utils.USE_JAX = "AUTO"
    _exec_guard({"transformers": object(), "transformers.utils.import_utils": import_utils}, {})
    assert import_utils.USE_TF == "0"
    assert import_utils.USE_JAX == "0"
    # Not a blunter write than the flag clearing: the same opt-outs still hold.
    for modules, environ, kept in (
        ({"tensorflow": object()}, {}, "USE_TF"),
        ({"jax": object()}, {}, "USE_JAX"),
        ({}, {"USE_TF": "1"}, "USE_TF"),
        ({}, {"FORCE_TF_AVAILABLE": "1"}, "USE_TF"),
        ({}, {"USE_FLAX": "1"}, "USE_JAX"),
    ):
        import_utils.USE_TF = import_utils.USE_JAX = "AUTO"
        _exec_guard(
            dict(
                modules,
                **{"transformers": object(), "transformers.utils.import_utils": import_utils},
            ),
            dict(environ),
        )
        assert getattr(import_utils, kept) == "AUTO", (modules, environ)


def test_a_broken_backend_loses_inside_the_real_import_utils_window(tmp_path):
    """The same window against the real `import_utils.py`, run in two halves with the
    guard between them. Without the constant write this ends `_tf_available` True."""
    _needs_v4_flag("USE_TF")
    out = _run(
        """
        import ast, pathlib, sys, types
        from transformers.utils import import_utils as real

        source = pathlib.Path(real.__file__).read_text(encoding = "utf-8")
        head, tail = source.split("\\n_torch_available = False", 1)
        tail = "\\n_torch_available = False" + tail
        assert "USE_TF = os.environ" in head and "_tf_available = False" in tail

        # Republish Transformers as a package that has only got as far as the top
        # of `import_utils`, keeping the real __path__ so its own imports resolve.
        package = pathlib.Path(real.__file__).parent
        for name in [n for n in sys.modules if n == "transformers" or n.startswith("transformers.")]:
            del sys.modules[name]
        for name, path in (("transformers", package.parent), ("transformers.utils", package)):
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
        window = types.ModuleType("transformers.utils.import_utils")
        window.__file__ = str(real.__file__)
        window.__package__ = "transformers.utils"
        sys.modules["transformers.utils.import_utils"] = window
        exec(compile(head, real.__file__, "exec"), window.__dict__)
        print("WINDOW", window.USE_TF, hasattr(window, "_tf_available"))

        block = None
        for node in ast.parse(pathlib.Path({root!r}, "unsloth", "__init__.py").read_text()).body:
            if isinstance(node, ast.If) and "sys.modules" in ast.unparse(node.test):
                block = node
                break
        import os
        exec(compile(ast.unparse(block), "<guard>", "exec"), {{"os": os, "sys": sys}})

        exec(compile(tail, real.__file__, "exec"), window.__dict__)
        print("TF", window.__dict__["_tf_available"])
        """.format(root = str(_ROOT)),
        site = _fake_tensorflow(tmp_path),
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "WINDOW AUTO False" in out.stdout, out.stdout
    assert "TF False" in out.stdout, out.stdout
