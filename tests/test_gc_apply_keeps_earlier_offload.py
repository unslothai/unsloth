# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A <512 seq model must not unpatch smart offloading an earlier model enabled. Stub torch, no GPU."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"
NAMES = ("_smart_gradient_checkpointing_patched", "apply_unsloth_gradient_checkpointing")


def _load():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    found = {n.name for n in nodes}
    if "apply_unsloth_gradient_checkpointing" not in found:
        pytest.fail("apply_unsloth_gradient_checkpointing not found")

    class Plain:
        pass

    class UnslothCheckpointFunction:
        pass

    ckpt = SimpleNamespace(CheckpointFunction = Plain)
    torch = SimpleNamespace(utils = SimpleNamespace(checkpoint = ckpt))
    log = []

    def patch(dtype = None):
        ckpt.CheckpointFunction = UnslothCheckpointFunction
        log.append("patch")

    def unpatch():
        ckpt.CheckpointFunction = Plain
        log.append("unpatch")

    ns = {
        "torch": torch,
        "patch_unsloth_smart_gradient_checkpointing": patch,
        "unpatch_unsloth_smart_gradient_checkpointing": unpatch,
    }
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE), "exec"), ns)
    patched = lambda: ckpt.CheckpointFunction is UnslothCheckpointFunction  # noqa: E731
    return ns["apply_unsloth_gradient_checkpointing"], patched


def test_short_sequence_model_keeps_an_earlier_models_offload():
    apply, patched = _load()
    assert apply("unsloth", 2048, None) == "unsloth" and patched()
    assert apply("unsloth", 256, None) == "unsloth"
    assert patched(), "a <512 model turned smart offloading off for the model loaded before it"


def test_short_sequence_alone_still_uses_plain_checkpointing():
    apply, patched = _load()
    assert apply("unsloth", 256, None) is True
    assert not patched()
    assert apply("unsloth", 4096, None) == "unsloth" and patched()


@pytest.mark.parametrize("value", [True, False])
def test_explicit_true_or_false_turns_it_off_and_says_so(value, capsys):
    apply, patched = _load()
    apply("unsloth", 2048, None)
    capsys.readouterr()
    assert apply(value, 2048, None) is value
    assert not patched()
    assert "turns off smart offloading" in capsys.readouterr().out


def test_explicit_true_without_offload_is_quiet(capsys):
    apply, patched = _load()
    assert apply(True, 2048, None) is True and not patched()
    assert capsys.readouterr().out == ""
