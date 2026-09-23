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

"""Gradients pass straight through compressed-tensors' W8A8 activation fake quantization.

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
    static = len(sys.argv) > 3 and sys.argv[3] == "static"
    if static:  # INT8 W8A8 with a calibrated per-tensor input scale that saturates
        activations = QuantizationArgs(
            num_bits = 8, type = "int", strategy = "tensor", dynamic = False, symmetric = True,
        )
    else:
        activations = QuantizationArgs(
            num_bits = 8, type = "float", strategy = "token", dynamic = True, symmetric = True,
        )
    scheme = QuantizationScheme(targets = ["Linear"], input_activations = activations)
    apply_quantization_config(model, QuantizationConfig(config_groups = {"group_0": scheme}, quantization_status = "frozen"))
    lin = model[0]
    lin.weight.requires_grad_(False)          # the frozen base layer under a LoRA adapter
    if static:
        model.to(torch.bfloat16)
        lin.input_scale.data.fill_(0.01)      # clips at 1.27, so most of randn * 4 saturates
    x = (torch.randn(4, 64) * (4 if static else 1)).to(lin.weight.dtype).requires_grad_(True)
    y = lin(x)
    with torch.no_grad():
        y_nograd = lin(x)
    out = {"has_grad_fn": y.grad_fn is not None, "same_forward": bool(torch.equal(y.detach(), y_nograd))}
    if y.grad_fn is not None:
        (gx,) = torch.autograd.grad(y.sum(), x)
        ref = torch.ones(4, 32, dtype = lin.weight.dtype) @ lin.weight
        out["grad_err"] = float((gx - ref).abs().max().float())
    import compressed_tensors.quantization.lifecycle.forward as f
    out["patched"] = bool(getattr(f.forward_quantize, "_unsloth_activation_ste", False))
    print(json.dumps(out))
    """
)


def _run(mode, scheme = "dynamic"):
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNNER,
            mode,
            os.path.join(_ROOT, "unsloth", "import_fixes.py"),
            scheme,
        ],
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


def test_a_saturating_static_scale_keeps_the_forward_bit_identical():
    """`x + (q - x).detach()` rounds wherever a static scale clips; the forward must stay `q` exactly."""
    out = _run("eager", "static")
    assert out["patched"]
    assert out["has_grad_fn"]
    assert out["same_forward"]
