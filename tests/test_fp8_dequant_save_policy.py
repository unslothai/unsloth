# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A dequantized fp8 load saves 16bit weights in the plain layout: every submodel of a composite
model, and every dequantizing route (Unsloth's 16bit flag, a caller's config, transformers' own)."""

import ast
import inspect
import json
from pathlib import Path

import pytest
import torch

try:
    from transformers import FineGrainedFP8Config
    _HAS_DEQUANTIZE = "dequantize" in inspect.signature(FineGrainedFP8Config).parameters
except Exception:
    _HAS_DEQUANTIZE = False

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 8


@pytest.fixture(autouse = True)
def _restore_fp8_integration(monkeypatch):
    # A real fp8 load patches Fp8Dequantize process-wide; later test files check the unpatched class.
    try:
        import transformers.integrations.finegrained_fp8 as fp8_integration
    except Exception:
        return
    cls = getattr(fp8_integration, "Fp8Dequantize", None)
    if cls is not None:
        for name, value in list(vars(cls).items()):
            if not name.startswith("__"):
                monkeypatch.setattr(cls, name, value)
    if hasattr(fp8_integration, "replace_with_fp8_linear"):
        monkeypatch.setattr(
            fp8_integration, "replace_with_fp8_linear", fp8_integration.replace_with_fp8_linear
        )


def _helper():
    # Exact source, without importing unsloth (which needs an accelerator).
    path = ROOT / "unsloth" / "models" / "loader_utils.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_drop_fp8_dequantize_conversions"
    ]
    ns = {}
    exec(compile(ast.Module(fn, []), str(path), "exec"), ns)
    return ns["_drop_fp8_dequantize_conversions"]


def _pair_classes():
    from transformers import PreTrainedModel, Qwen3MoeConfig, Qwen3MoeForCausalLM

    try:
        from transformers import PreTrainedConfig
    except ImportError:
        from transformers import PretrainedConfig as PreTrainedConfig

    class PairConfig(PreTrainedConfig):
        model_type = "unsloth_test_fp8_pair"

    class Pair(PreTrainedModel):
        config_class = PairConfig

        def __init__(self, config):
            super().__init__(config)
            sub = Qwen3MoeConfig(
                hidden_size = 16,
                intermediate_size = 32,
                moe_intermediate_size = 8,
                num_hidden_layers = 1,
                num_attention_heads = 2,
                num_key_value_heads = 1,
                head_dim = 8,
                num_experts = 2,
                num_experts_per_tok = 1,
                vocab_size = 32,
            )
            self.left = Qwen3MoeForCausalLM(sub)
            self.right = Qwen3MoeForCausalLM(sub)
            self.post_init()

    return PairConfig, Pair


def _quantize_checkpoint(src):
    from safetensors.torch import load_file, save_file

    weights = load_file(src / "model.safetensors")
    out = {}
    for key, v in weights.items():
        if key.endswith("proj.weight") and v.ndim == 2:
            n, k = v.shape
            b = v.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
            s = b.abs().amax(dim = (1, 3)).clamp_min(1e-12) / 448
            out[key] = (b / s[:, None, :, None]).reshape(n, k).to(torch.float8_e4m3fn)
            out[key.replace(".weight", ".weight_scale_inv")] = s
        else:
            out[key] = v
    save_file(out, src / "model.safetensors")
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [BLOCK, BLOCK],
    }
    (src / "config.json").write_text(json.dumps(cfg))


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
def test_composite_model_keeps_every_scoped_counterpart(tmp_path):
    from safetensors.torch import load_file
    from transformers.conversion_mapping import get_model_conversion_mapping

    PairConfig, Pair = _pair_classes()
    torch.manual_seed(0)
    src = tmp_path / "src"
    Pair(PairConfig()).to(torch.bfloat16).save_pretrained(src)
    _quantize_checkpoint(src)
    qc = FineGrainedFP8Config(dequantize = True, weight_block_size = [BLOCK, BLOCK])
    model = Pair.from_pretrained(
        src, device_map = "cpu", dtype = torch.bfloat16, quantization_config = qc
    )
    assert not any(p.dtype == torch.float8_e4m3fn for p in model.parameters())
    assert _helper()(model) > 0

    model.save_pretrained(tmp_path / "saved")
    model._weight_conversions = get_model_conversion_mapping(model, None, None)
    model.save_pretrained(tmp_path / "canonical")
    saved = load_file(tmp_path / "saved" / "model.safetensors")
    canonical = load_file(tmp_path / "canonical" / "model.safetensors")
    # Same per-expert layout as a plain 16bit save of both submodels, so older transformers
    # (ModuleList experts) reload it too.
    assert sorted(saved) == sorted(canonical)
    assert "right.model.layers.0.mlp.experts.0.gate_proj.weight" in saved
    assert all(torch.equal(saved[k], canonical[k]) for k in saved)


def _calls_and_guards(path, cls_name):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")
    found = []

    def visit(node, guards):
        for child in ast.iter_child_nodes(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "_drop_fp8_dequantize_conversions"
            ):
                found.append(guards)
            g = guards + [ast.unparse(child.test)] if isinstance(child, ast.If) else guards
            visit(child, g)

    visit(fn, [])
    return found


@pytest.mark.parametrize(
    "rel, cls_name",
    [
        ("unsloth/models/loader.py", "FastLanguageModel"),
        ("unsloth/models/vision.py", "FastBaseModel"),
    ],
)
def test_cleanup_runs_for_every_dequantizing_route(rel, cls_name):
    # A caller's FineGrainedFP8Config(dequantize = True) and transformers' automatic dequantize
    # (pre-sm89, CPU) leave the same Fp8Dequantize conversions as Unsloth's 16bit flag.
    found = _calls_and_guards(ROOT / rel, cls_name)
    assert found, "cleanup call missing"
    for guards in found:
        text = " ".join(guards)
        assert "_fp8_dequantize_config" not in text and "load_in_16bit" not in text, guards


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_and_automatic_dequantize_reload(tmp_path, explicit, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM

    if not explicit and torch.cuda.is_available():
        # transformers dequantizes by itself only without CUDA or below sm89; on an sm89+ card
        # (L4, H100, B200) a device_map="cpu" load keeps fp8 weights. Report a pre-sm89 card.
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))

    torch.manual_seed(0)
    src = tmp_path / "src"
    config = LlamaConfig(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        vocab_size = 64,
    )
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(src)
    _quantize_checkpoint(src)
    kwargs = {}
    if explicit:
        kwargs["quantization_config"] = FineGrainedFP8Config(
            dequantize = True, weight_block_size = [BLOCK, BLOCK]
        )
    # On CPU transformers dequantizes by itself, as on pre-sm89 GPUs.
    model = LlamaForCausalLM.from_pretrained(src, device_map = "cpu", dtype = torch.bfloat16, **kwargs)
    assert not any(p.dtype == torch.float8_e4m3fn for p in model.parameters())
    assert _helper()(model) > 0
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.save_pretrained(tmp_path / "saved")
    after = LlamaForCausalLM.from_pretrained(tmp_path / "saved", dtype = torch.bfloat16).state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
def test_heterogeneous_composite_keeps_one_copy_of_an_unwrapped_converter(monkeypatch):
    # Two submodels whose converters share a type and target but differ in sources: only the
    # `.weight`-sourced one gets the fp8 dequant prepended. The other one is kept as loaded and must
    # not gain a second copy, or save_pretrained would reverse it twice.
    import transformers.conversion_mapping as cm
    from types import SimpleNamespace
    from transformers.core_model_loading import (
        Concatenate,
        MergeModulelist,
        Transpose,
        WeightConverter,
    )
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    target = "mlp.experts.gate_up_proj"

    def plain():
        merge = WeightConverter(
            source_patterns = ["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"],
            target_patterns = target,
            operations = [MergeModulelist(dim = 0), Concatenate(dim = 1)],
        )
        merge.scope_prefix = "left"
        transpose = WeightConverter(
            source_patterns = target, target_patterns = target, operations = [Transpose(1, 2)]
        )
        transpose.scope_prefix = "right"
        return [merge, transpose]

    quantizer = FineGrainedFP8HfQuantizer(
        FineGrainedFP8Config(dequantize = True, weight_block_size = [BLOCK, BLOCK])
    )
    quantizer.pre_quantized = True
    loaded = quantizer.update_weight_conversions(plain())
    model = SimpleNamespace(_weight_conversions = loaded)
    monkeypatch.setattr(cm, "get_model_conversion_mapping", lambda *a, **k: plain())
    assert _helper()(model) > 0

    convs = [c for c in model._weight_conversions if isinstance(c, WeightConverter)]
    transposes = [c for c in convs if any(isinstance(op, Transpose) for op in c.operations)]
    merges = [c for c in convs if any(isinstance(op, MergeModulelist) for op in c.operations)]
    assert len(transposes) == 1, [(c.scope_prefix, c._original_source_patterns) for c in transposes]
    assert len(merges) == 1 and merges[0].scope_prefix == "left"
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    assert not any(isinstance(op, Fp8Dequantize) for c in convs for op in c.operations)
