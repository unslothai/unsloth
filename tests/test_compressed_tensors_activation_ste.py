# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

"""A LoRA finetune of a W8A8 compressed-tensors checkpoint must get gradients through the
quantized activations.

compressed-tensors' `fake_quantize` runs under `@torch.no_grad()`, so with a frozen base weight
the Linear output of a dynamic FP8 activation scheme has no path back to its input
(ibm-granite/granite-4.1-30b-FP8: q_proj LoRA gradients 1000x too small, loss drifting up).
`fix_compressed_tensors_activation_quant_gradient` applies a straight-through estimator: the
forward is the quantized one, bit for bit, and the input gradient is the unquantized one.
Each case runs in a subprocess because the patch changes compressed-tensors module state.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("compressed_tensors")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_RUNNER = textwrap.dedent(
    """
    import importlib.util, json, sys
    mode = sys.argv[1]
    if mode != "base":
        spec = importlib.util.spec_from_file_location("unsloth_import_fixes_under_test", sys.argv[2])
        fixes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixes)
        if mode == "lazy":
            assert "compressed_tensors.quantization.lifecycle.forward" not in sys.modules
            fixes.fix_compressed_tensors_activation_quant_gradient()
        import compressed_tensors  # noqa: F401
        if mode != "lazy":
            fixes.fix_compressed_tensors_activation_quant_gradient()
            fixes.fix_compressed_tensors_activation_quant_gradient()  # idempotent
    import torch
    from compressed_tensors.quantization import (
        QuantizationArgs, QuantizationConfig, QuantizationScheme, apply_quantization_config,
    )
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias = False))
    scheme = QuantizationScheme(
        targets = ["Linear"],
        input_activations = QuantizationArgs(
            num_bits = 8, type = "float", strategy = "token", dynamic = True, symmetric = True,
        ),
    )
    apply_quantization_config(model, QuantizationConfig(config_groups = {"group_0": scheme}, quantization_status = "frozen"))
    lin = model[0]
    lin.weight.requires_grad_(False)          # the frozen base layer under a LoRA adapter
    x = torch.randn(4, 64, requires_grad = True)
    y = lin(x)
    with torch.no_grad():
        y_nograd = lin(x)
    out = {"has_grad_fn": y.grad_fn is not None, "same_forward": bool(torch.equal(y.detach(), y_nograd))}
    if y.grad_fn is not None:
        (gx,) = torch.autograd.grad(y.sum(), x)
        ref = torch.ones(4, 32) @ lin.weight
        out["grad_err"] = float((gx - ref).abs().max())
    import compressed_tensors.quantization.lifecycle.forward as f
    out["patched"] = bool(getattr(f.forward_quantize, "_unsloth_activation_ste", False))
    print(json.dumps(out))
    """
)


def _run(mode):
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, mode, os.path.join(_ROOT, "unsloth", "import_fixes.py")],
        capture_output = True,
        text = True,
        timeout = 600,
        env = dict(os.environ, CUDA_VISIBLE_DEVICES = ""),
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    import json

    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_without_the_fix_the_input_gradient_is_lost():
    """The defect itself, so a compressed-tensors release that fixes it upstream shows up here."""
    out = _run("base")
    assert not out["has_grad_fn"]


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_activation_quantization_passes_the_gradient_straight_through(mode):
    out = _run(mode)
    assert out["patched"]
    assert out["has_grad_fn"]
    assert out["same_forward"]  # the quantized forward is untouched
    assert out["grad_err"] < 1e-5  # dX = dY @ W, as if the activation were not quantized
