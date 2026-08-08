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
import os
import pathlib
import subprocess
import sys

import pytest

_INIT = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "__init__.py"
_SOURCE = _INIT.read_text(encoding = "utf-8")


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


def test_the_guard_is_skipped_once_transformers_is_loaded():
    """Nothing to fix then: an already-imported Transformers settled it."""
    block = _guard_block()
    scope = {"os": os, "sys": type("S", (), {"modules": {"transformers": object()}})()}
    before = os.environ.get("USE_TF")
    try:
        os.environ.pop("USE_TF", None)
        exec(ast.unparse(block), scope)
        assert "USE_TF" not in os.environ
    finally:
        if before is None:
            os.environ.pop("USE_TF", None)
        else:
            os.environ["USE_TF"] = before
