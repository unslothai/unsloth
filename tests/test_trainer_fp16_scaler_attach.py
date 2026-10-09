# SPDX-License-Identifier: AGPL-3.0-only
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
"""A plain transformers.Trainer under fp16 must hand its GradScaler to the model as
`accelerator_scaler`, which the fused CE loss reads; without it the saved logit gradients
underflow in fp16 and LoRA barely trains (#3529). `import unsloth` needs CUDA or XPU, so
the helper is executed from source.
"""

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")


def _functions():
    tree = ast.parse(_UTILS.read_text(encoding = "utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _load_helper():
    ns = {
        "torch": torch,
        "_UNSLOTH_WRAPPED_MODULE_ATTRS": ("module", "_orig_mod", "_fsdp_wrapped_module"),
    }
    exec(
        compile(
            ast.Module([_functions()["_unsloth_attach_accelerator_scaler"]], []),
            str(_UTILS),
            "exec",
        ),
        ns,
    )
    return ns["_unsloth_attach_accelerator_scaler"]


def _peft_llama():
    cfg = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
    )
    causal = transformers.LlamaForCausalLM(cfg)
    model = peft.get_peft_model(causal, peft.LoraConfig(r = 4, target_modules = ["q_proj"]))
    return causal, model


class _Wrapper(torch.nn.Module):  # DDP-style `module` wrapper, as accelerate.prepare returns
    def __init__(self, module):
        super().__init__()
        self.module = module


def test_compute_loss_patch_attaches_scaler_before_loss():
    body = _functions()["_unsloth_pre_compute_loss"].body
    calls = [
        ast.unparse(n.value.func)
        for n in body
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
    ]
    assert "_unsloth_attach_accelerator_scaler" in calls
    src = ast.unparse(ast.Module(body, []))
    assert src.index("_unsloth_attach_accelerator_scaler") < src.index("self._old_compute_loss")


def test_scaler_reaches_causal_lm_through_peft_and_wrapper():
    attach = _load_helper()
    causal, model = _peft_llama()
    scaler = object()
    trainer = SimpleNamespace(accelerator = SimpleNamespace(scaler = scaler))
    attach(trainer, _Wrapper(model))
    # The fused CE call site reads `getattr(self, "accelerator_scaler", None)` on the CausalLM.
    assert causal.__dict__.get("accelerator_scaler") is scaler
    attach(trainer, _Wrapper(model))  # idempotent
    assert causal.__dict__.get("accelerator_scaler") is scaler


def test_no_scaler_is_a_no_op():
    attach = _load_helper()
    causal, model = _peft_llama()
    for trainer in (SimpleNamespace(accelerator = SimpleNamespace(scaler = None)), SimpleNamespace()):
        attach(trainer, model)
    assert "accelerator_scaler" not in causal.__dict__
