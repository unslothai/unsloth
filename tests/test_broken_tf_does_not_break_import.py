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
"""A broken TensorFlow / Flax install must not break importing Unsloth.

Transformers 4.x imports either backend merely because it is installed, via
`processing_utils` -> `image_transforms`; on Colab a protobuf-moving install cell
then leaves TF raising `cannot import name 'runtime_version'`.
"""

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

_BROKEN_TF = 'raise ImportError("cannot import name \'runtime_version\' from \'google.protobuf\'")\n'


def _fake_tensorflow(tmp_path):
    """A `tensorflow` on `sys.path` that Transformers detects but cannot import.

    `_tf_available` needs a `find_spec` hit *and* an installed version >= 2, so the
    directory carries a matching `.dist-info/METADATA`. Never touches site-packages.
    """
    site = tmp_path / "fakesite"
    package = site / "tensorflow"
    package.mkdir(parents = True)
    (package / "__init__.py").write_text(_BROKEN_TF, encoding = "utf-8")
    dist = site / "tensorflow-2.20.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: tensorflow\nVersion: 2.20.0\n", encoding = "utf-8",
    )
    return site


def _run(code, site = None, **env):
    """Run `code` in a fresh interpreter, so no module state leaks between cases."""
    path = [str(_ROOT)] + ([str(site)] if site is not None else [])
    if os.environ.get("PYTHONPATH"): path.append(os.environ["PYTHONPATH"])
    # Importing Unsloth sets USE_TF/USE_FLAX in this process, so an inherited
    # value would decide the case before the child starts. Each test says.
    clean = {
        k: v for k, v in os.environ.items()
        if k not in ("USE_TF", "USE_FLAX", "FORCE_TF_AVAILABLE")
    }
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
    if not _unsloth_is_importable(): pytest.skip("unsloth is not importable in this environment")


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
    body = ast.unparse(block)
    for name in ("USE_TF", "USE_FLAX"):
        assert f'"{name}", "0"' in body or f"'{name}', '0'" in body, name
    # The guard has to sit above every `transformers` import in this file, or a
    # module that already read the variables makes it a no-op.
    first_import = min(
        (
            node.lineno
            for node in ast.walk(ast.parse(_SOURCE))
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("transformers")
        ),
        default = 10**9,
    )
    assert block.lineno < first_import


def test_an_explicit_choice_is_never_overwritten():
    """`setdefault`, not assignment: someone who wants TF in-process keeps it."""
    body = ast.unparse(_guard_block())
    assert "os.environ.setdefault" in body
    assert "os.environ[" not in body, "an assignment would override the user"


@pytest.mark.parametrize("value", ["0", "1"])
def test_transformers_reads_the_variable_from_the_environment(value):
    """The half in Transformers: read once at import, so ours must land first."""
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


def test_the_variables_are_left_alone_once_transformers_is_loaded():
    """They are spent by then, so the cached flags are what gets cleared instead."""
    environ = {}
    _exec_guard({"transformers": object()}, environ)
    assert environ == {}


def test_a_broken_backend_still_loses_when_transformers_came_first(tmp_path):
    """The regression: `_tf_available` was cached True before Unsloth got a say."""
    _needs_unsloth()
    out = _run(
        """
        import transformers
        from transformers.utils import import_utils
        assert import_utils._tf_available, "the fake tensorflow was not detected"
        import sys, unsloth
        print("TF_LOADED", "tensorflow" in sys.modules)
        print("TF_AVAILABLE", import_utils._tf_available)
        """,
        site = _fake_tensorflow(tmp_path),
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "TF_LOADED False" in out.stdout, out.stdout
    assert "TF_AVAILABLE False" in out.stdout, out.stdout


def test_the_environment_path_still_covers_the_transformers_not_loaded_case(tmp_path):
    _needs_unsloth()
    out = _run(
        """
        import unsloth, sys
        from transformers.utils import import_utils
        print("USE_TF", import_utils.USE_TF)
        print("TF_LOADED", "tensorflow" in sys.modules)
        """,
        site = _fake_tensorflow(tmp_path),
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "USE_TF 0" in out.stdout, out.stdout
    assert "TF_LOADED False" in out.stdout, out.stdout


def _run_guard(tmp_path, preamble, **env):
    """Run the real opt-out block against a real, already-imported Transformers."""
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

    Spelled with FORCE_TF_AVAILABLE rather than USE_TF because Transformers reads
    USE_TF=1 as "and disable PyTorch", which Unsloth needs.
    """
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


def test_a_transformers_without_import_utils_loaded_is_a_no_op():
    _exec_guard({"transformers": types.ModuleType("transformers")}, {})


def test_the_flags_are_cleared_only_when_the_backend_is_unused():
    import_utils = types.ModuleType("transformers.utils.import_utils")
    import_utils._tf_available = True
    import_utils._flax_available = True
    modules = {"transformers": object(), "transformers.utils.import_utils": import_utils}
    _exec_guard(modules, {})
    assert import_utils._tf_available is False
    assert import_utils._flax_available is False
    # jax in play means Flax is genuinely in use.
    import_utils._flax_available = True
    _exec_guard(dict(modules, jax = object()), {})
    assert import_utils._flax_available is True
    import_utils._flax_available = True
    _exec_guard(modules, {"USE_FLAX": "yes"})
    assert import_utils._flax_available is True
    for _var in ("USE_TF", "FORCE_TF_AVAILABLE"):
        import_utils._tf_available = True
        _exec_guard(modules, {_var: "1"})
        assert import_utils._tf_available is True, _var
    # An imported TensorFlow is one in use.
    import_utils._tf_available = True
    _exec_guard(dict(modules, tensorflow = object()), {})
    assert import_utils._tf_available is True
