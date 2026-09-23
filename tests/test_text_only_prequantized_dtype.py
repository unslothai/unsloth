# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A text-only load of a pre-quantized VLM repo must end up in the requested dtype.

transformers >= 5 keeps the checkpoint dtype for every key a key_mapping renamed on a
pre-quantized load (core_model_loading: `hf_quantizer.pre_quantized and original_key !=
renamed_key` -> `_dtype = None`). text_only renames `model.language_model.*` onto the
decoder's `model.*`, so `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` (saved in float16) loaded with
dtype=bfloat16 kept embed_tokens and the unquantized linear-attention projections in float16,
while the unrenamed `lm_head.weight` became bfloat16. A plain forward outside autocast then
died at the lm_head: "expected mat1 and mat2 to have the same dtype, Half != BFloat16".

CPU only: the helper is lifted out of unsloth/models/_utils.py by AST so no accelerator and
no particular transformers version is needed, and the call site in vision.py is checked
structurally.
"""

import ast
import os
import re
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(ROOT, "unsloth", "models", "_utils.py")
VISION = os.path.join(ROOT, "unsloth", "models", "vision.py")
HELPER = "_cast_text_only_prequantized_params"


def _load_helper():
    src = open(UTILS, encoding = "utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == HELPER:
            ns = {"torch": torch, "re": re}
            exec(ast.get_source_segment(src, node), ns)
            return ns[HELPER]
    raise AssertionError(f"{HELPER} not found in unsloth/models/_utils.py")


class _QuantStorage(nn.Parameter):
    """Stands in for bnb Params4bit, which may carry bf16 / fp16 quant_storage."""


class _TinyDecoder(nn.Module):
    """The float16 leftovers of a Qwen3.5 text-only load, in miniature."""

    def __init__(self, tie = False):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8).half()
        self.in_proj_qkv = nn.Linear(8, 8, bias = False).half()
        self.A_log = nn.Parameter(torch.zeros(4, dtype = torch.float16))
        self.norm = nn.LayerNorm(8)  # float32, upcast on purpose
        self.quantized = nn.Linear(8, 8, bias = False)
        self.quantized.weight = _QuantStorage(
            torch.zeros(32, 1, dtype = torch.float16), requires_grad = False
        )
        self.lm_head = nn.Linear(8, 16, bias = False).to(torch.bfloat16)
        if tie:
            self.lm_head.weight = self.embed_tokens.weight
        self.hf_quantizer = types.SimpleNamespace(pre_quantized = True)

    def forward(self, ids):
        h = self.in_proj_qkv(self.embed_tokens(ids))
        return self.lm_head(h)


def test_prequantized_float16_leftovers_follow_requested_dtype():
    cast = _load_helper()
    model = _TinyDecoder()
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(RuntimeError):
        # The bug: bf16 lm_head fed by an fp16 residual stream.
        model(ids)
    n = cast(model, torch.bfloat16)
    assert n == 3
    assert model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.in_proj_qkv.weight.dtype == torch.bfloat16
    assert model.A_log.dtype == torch.bfloat16
    assert model.norm.weight.dtype == torch.float32
    assert model.quantized.weight.dtype == torch.float16
    assert isinstance(model.quantized.weight, _QuantStorage)
    with torch.no_grad():
        assert model(ids).dtype == torch.bfloat16


def test_float16_request_is_a_no_op_for_a_float16_checkpoint():
    # T4 / V100: float16 is the right dtype, nothing may move.
    cast = _load_helper()
    model = _TinyDecoder()
    model.lm_head.to(torch.float16)
    before = {k: (v.dtype, v.data_ptr()) for k, v in model.named_parameters()}
    assert cast(model, torch.float16) == 0
    assert before == {k: (v.dtype, v.data_ptr()) for k, v in model.named_parameters()}


def test_bfloat16_checkpoint_on_a_float16_device_is_cast_down():
    cast = _load_helper()
    model = _TinyDecoder()
    model.embed_tokens.to(torch.bfloat16)
    cast(model, torch.float16)
    assert model.embed_tokens.weight.dtype == torch.float16
    assert model.lm_head.weight.dtype == torch.float16


def test_tied_weights_stay_tied():
    cast = _load_helper()
    model = _TinyDecoder(tie = True)
    cast(model, torch.bfloat16)
    assert model.lm_head.weight is model.embed_tokens.weight
    assert model.embed_tokens.weight.dtype == torch.bfloat16


def test_not_prequantized_or_unknown_dtype_is_left_alone():
    cast = _load_helper()
    model = _TinyDecoder()
    model.hf_quantizer = None
    assert cast(model, torch.bfloat16) == 0
    assert model.embed_tokens.weight.dtype == torch.float16
    model.hf_quantizer = types.SimpleNamespace(pre_quantized = False)
    assert cast(model, torch.bfloat16) == 0
    model.hf_quantizer = types.SimpleNamespace(pre_quantized = True)
    assert cast(model, None) == 0
    assert cast(model, "auto") == 0
    assert model.embed_tokens.weight.dtype == torch.float16


def test_keep_in_fp32_plan_is_honoured():
    cast = _load_helper()
    model = _TinyDecoder()
    model._get_dtype_plan = lambda dtype: {"A_log": torch.float32}
    cast(model, torch.bfloat16)
    assert model.A_log.dtype == torch.float32
    assert model.embed_tokens.weight.dtype == torch.bfloat16


def test_real_bnb_params4bit_is_untouched():
    bnb = pytest.importorskip("bitsandbytes")
    cast = _load_helper()
    model = _TinyDecoder()
    storage = torch.zeros(32, 1, dtype = torch.bfloat16)
    model.quantized.weight = bnb.nn.Params4bit(storage, requires_grad = False)
    cast(model, torch.float16)
    assert type(model.quantized.weight) is bnb.nn.Params4bit
    assert model.quantized.weight.dtype == torch.bfloat16


def _fast_base_from_pretrained():
    tree = ast.parse(open(VISION, encoding = "utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FastBaseModel":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "from_pretrained":
                    return item
    raise AssertionError("FastBaseModel.from_pretrained not found")


def test_vision_loader_casts_right_after_the_text_only_load():
    fn = _fast_base_from_pretrained()
    load_line = None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "from_pretrained"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "auto_model"
        ):
            load_line = node.lineno
    assert load_line is not None
    calls = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test_names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if "text_only_decoder" not in test_names:
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == HELPER
            ):
                args = [a.id for a in call.args if isinstance(a, ast.Name)]
                calls.append((call.lineno, args))
    assert calls, f"{HELPER} is not called under a text_only_decoder guard"
    line, args = min(calls)
    assert line > load_line
    assert args == ["model", "torch_dtype"]
    # Before offload moves embed_tokens to CPU or hooks capture the weights.
    src_lines = open(VISION, encoding = "utf-8").read().splitlines()
    tail = "\n".join(src_lines[load_line:line])
    assert "_attach_bnb_multidevice_hooks(" not in tail
    assert 'embed_tokens.to("cpu")' not in tail


@pytest.mark.parametrize(
    "requested, resolved",
    [(torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16)],
    ids = ["awq_bf16_to_fp16", "fbgemm_fp8_fp16_to_bf16"],
)
def test_quantizer_resolved_dtype_wins_over_the_request(requested, resolved):
    # AWQ turns a bf16 request into fp16 on CUDA and FBGEMM FP8 forces bf16; from_pretrained
    # stores the quantizer's choice on config.dtype and the repair must not undo it.
    cast = _load_helper()
    model = _TinyDecoder()
    model.embed_tokens.to(torch.bfloat16 if resolved == torch.float16 else torch.float16)
    model.lm_head.to(resolved)
    model.config = types.SimpleNamespace(dtype = resolved)
    cast(model, requested)
    assert model.embed_tokens.weight.dtype == resolved
    assert model.in_proj_qkv.weight.dtype == resolved
    assert model.lm_head.weight.dtype == resolved
    assert model.quantized.weight.dtype == torch.float16


@pytest.mark.parametrize("offload", ["cpu", "disk"])
def test_offloaded_leftovers_materialize_in_the_requested_dtype(offload, tmp_path):
    # device_map with "cpu" / "disk" leaves meta placeholders plus accelerate hooks whose weights
    # map still holds the float16 checkpoint tensors. accelerate casts that stored value to the
    # placeholder's dtype on every pre_forward, so the placeholders must be recast too.
    accelerate = pytest.importorskip("accelerate")
    cast = _load_helper()
    model = _TinyDecoder()
    model.quantized = nn.Identity()
    model.lm_head.to(torch.float32)
    model.norm.to(torch.float16)
    ids = torch.tensor([[1, 2, 3]])
    if offload == "cpu":
        accelerate.cpu_offload(model, execution_device = torch.device("cpu"))
    else:
        accelerate.disk_offload(model, offload_dir = str(tmp_path), execution_device = torch.device("cpu"))
    assert model.embed_tokens.weight.device.type == "meta"
    assert model.embed_tokens.weight.dtype == torch.float16
    cast(model, torch.float32)
    assert model.embed_tokens.weight.dtype == torch.float32
    assert model.in_proj_qkv.weight.dtype == torch.float32
    with torch.no_grad():
        out = model(ids)
    assert out.dtype == torch.float32
