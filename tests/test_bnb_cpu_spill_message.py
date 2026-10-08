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

"""A 4-bit / 8-bit load that does not fit the GPU should say what to do in Unsloth (#1629).

transformers' bitsandbytes quantizers refuse a map that spills to the CPU and suggest
`llm_int8_enable_fp32_cpu_offload`, which is no route to training. Unsloth's
`offload_layers = "auto"` is, so the error points there instead.
"""

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unsloth.models.loader_utils import (  # noqa: E402
    _BNB_CPU_SPILL_PREFIX,
    raise_if_bnb_cpu_spill,
)


@pytest.mark.parametrize("module", ["quantizer_bnb_4bit", "quantizer_bnb_8bit"])
def test_prefix_matches_the_installed_transformers(module):
    quantizer = pytest.importorskip(f"transformers.quantizers.{module}")
    assert _BNB_CPU_SPILL_PREFIX in inspect.getsource(quantizer)


def _spill_error():
    return ValueError(
        _BNB_CPU_SPILL_PREFIX + ". Make sure you have enough GPU RAM to fit the quantized model."
    )


def test_spill_points_at_offload_layers():
    original = _spill_error()
    with pytest.raises(ValueError) as info:
        raise_if_bnb_cpu_spill(original, "unsloth/Mistral-Small-3.2-24B-Instruct-2506")
    message = str(info.value)
    assert message.startswith("Unsloth: unsloth/Mistral-Small-3.2-24B-Instruct-2506 does not fit")
    assert 'offload_layers = "auto"' in message
    assert "llm_int8_enable_fp32_cpu_offload" not in message
    assert info.value.__cause__ is original


@pytest.mark.parametrize("offload_layers", ["auto", 4])
def test_spill_with_offload_already_requested(offload_layers):
    with pytest.raises(ValueError) as info:
        raise_if_bnb_cpu_spill(_spill_error(), "m", offload_layers)
    assert "offload_layers" not in str(info.value)
    assert "max_seq_length" in str(info.value)


@pytest.mark.parametrize(
    "error",
    [ValueError("Unrecognized configuration class"), RuntimeError(_BNB_CPU_SPILL_PREFIX)],
)
def test_other_errors_are_left_alone(error):
    assert raise_if_bnb_cpu_spill(error, "m") is None


@pytest.mark.parametrize(
    "path, cls",
    [("unsloth/models/llama.py", "FastLlamaModel"), ("unsloth/models/vision.py", "FastBaseModel")],
)
def test_both_loaders_route_load_errors_through_it(path, cls):
    import ast

    tree = ast.parse((ROOT / path).read_text(encoding = "utf-8"))
    method = next(
        node
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == cls
        for node in c.body
        if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"
    )
    handlers = [
        h
        for node in ast.walk(method)
        if isinstance(node, ast.Try)
        for h in node.handlers
        if isinstance(h.type, ast.Name) and h.type.id == "ValueError"
    ]
    routed = [
        h
        for h in handlers
        if any(
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Call)
            and getattr(s.value.func, "id", None) == "raise_if_bnb_cpu_spill"
            for s in h.body
        )
        and isinstance(h.body[-1], ast.Raise)
        and h.body[-1].exc is None
    ]
    assert len(routed) == 1
