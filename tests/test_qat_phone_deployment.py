# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""QAT scheme construction on tiny CPU models."""

from __future__ import annotations

import importlib.util
import warnings

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None, reason = "require torchao"
)

from unsloth.models._utils import (
    TorchAOConfig,
    _convert_torchao_model,
    _prepare_model_for_qat,
)


def _tiny_gemma3():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size = 256,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        tie_word_embeddings = True,
    )
    return Gemma3ForCausalLM(config).eval()


def _tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 1,
        head_dim = 32,
    )
    return Qwen3ForCausalLM(config).eval()


def _is_fake_quantized(module):
    return hasattr(module, "weight_fake_quantizer")


def test_phone_deployment_alias_in_prepare():
    model = _prepare_model_for_qat(_tiny_qwen3(), "phone-deployment")
    assert model._torchao_config.qat_scheme == "int8-int4"
    assert _is_fake_quantized(model.model.layers[0].self_attn.q_proj)
    assert _is_fake_quantized(model.model.embed_tokens)


def test_gemma3_int8_int4_keeps_phone_scheme():
    from torchao.quantization import Int8DynamicActivationIntxWeightConfig

    model = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    assert model._torchao_config.qat_scheme == "int8-int4"
    configs = [c for c, _ in model._torchao_config.base_config_and_filter_fns]
    assert any(isinstance(c, Int8DynamicActivationIntxWeightConfig) for c in configs)
    assert _is_fake_quantized(model.model.layers[0].self_attn.q_proj)


def _int8_embedding_config():
    from torchao.quantization import IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerAxis
    return TorchAOConfig(
        qat_scheme = None,
        base_config_and_filter_fns = [
            (
                IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0)),
                lambda m, _: isinstance(m, torch.nn.Embedding),
            )
        ],
    )


@pytest.mark.parametrize("scheme", ["int8-int4", "custom"])
@pytest.mark.parametrize("convert", ["none", "unsloth", "manual"])
def test_gemma3_fake_quantized_embedding_keeps_embed_scale(scheme, convert):
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding

    model = _tiny_gemma3()
    ids = torch.randint(0, model.config.vocab_size, (2, 12))
    with torch.no_grad():
        ref = model.get_input_embeddings()(ids)
    assert float(model.get_input_embeddings().embed_scale) == 8.0

    model = _prepare_model_for_qat(
        model, "int8-int4" if scheme == "int8-int4" else _int8_embedding_config()
    )
    emb = model.get_input_embeddings()
    assert _is_fake_quantized(emb)
    if convert == "unsloth":
        _convert_torchao_model(model)
    elif convert == "manual":
        quantize_(model, QATConfig(step = "convert"))
    emb = model.get_input_embeddings()
    assert isinstance(emb, Gemma3TextScaledWordEmbedding)
    with torch.no_grad():
        out = emb(ids)
    # int8 per-row error is ~0.4% of the row max; a lost sqrt(hidden_size) = 8x scale is not.
    rel = (out - ref).abs().max() / ref.abs().max()
    assert rel < 0.02, float(rel)


def test_int4_skips_linears_not_divisible_by_group_size():
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ok = torch.nn.Linear(256, 64, bias = False)
            self.odd = torch.nn.Linear(192, 64, bias = False)

        def forward(self, x):
            return self.ok(torch.nn.functional.pad(x, (0, 64))) + self.odd(x)

    model = _M().to(torch.bfloat16)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, "int4")
    assert _is_fake_quantized(model.ok)
    assert not _is_fake_quantized(model.odd)
    assert any("odd" in str(w.message) for w in caught)
    model(torch.randn(2, 192, dtype = torch.bfloat16))


@pytest.mark.parametrize("qat_scheme", ["int8-int4", "int4", "int8", "fp8-fp8"])
def test_lora_adapters_are_not_fake_quantized(qat_scheme, monkeypatch):
    from peft import LoraConfig, get_peft_model

    monkeypatch.delenv("UNSLOTH_QAT_FAKE_QUANTIZE_LORA", raising = False)
    model = get_peft_model(
        _tiny_qwen3(),
        LoraConfig(r = 32, lora_alpha = 32, target_modules = ["q_proj", "o_proj"]),
    )
    model = _prepare_model_for_qat(model, qat_scheme)
    q_proj = model.base_model.model.model.layers[0].self_attn.q_proj
    assert _is_fake_quantized(q_proj.base_layer)
    assert not _is_fake_quantized(q_proj.lora_A["default"])
    assert not _is_fake_quantized(q_proj.lora_B["default"])


def test_lora_fake_quant_opt_out(monkeypatch):
    from peft import LoraConfig, get_peft_model

    monkeypatch.setenv("UNSLOTH_QAT_FAKE_QUANTIZE_LORA", "1")
    model = get_peft_model(
        _tiny_qwen3(), LoraConfig(r = 32, lora_alpha = 32, target_modules = ["q_proj"])
    )
    model = _prepare_model_for_qat(model, "int8-int4")
    q_proj = model.base_model.model.model.layers[0].self_attn.q_proj
    assert _is_fake_quantized(q_proj.lora_A["default"])


class _EoiEmbedding(torch.nn.Embedding):
    # T5Gemma2TextScaledWordEmbedding pattern: an extra registered parameter read in forward.
    def __init__(self, num, dim):
        super().__init__(num, dim)
        self.eoi_embedding = torch.nn.Parameter(torch.ones(dim))

    def forward(self, input_ids):
        out = super().forward(input_ids)
        out[input_ids == 0] = self.eoi_embedding.to(out.dtype)
        return out


@pytest.mark.parametrize("convert", [False, True])
def test_embedding_subclass_keeps_extra_parameters(convert):
    model = _tiny_gemma3()
    model.model.embed_tokens = _EoiEmbedding(model.config.vocab_size, model.config.hidden_size)
    eoi = model.model.embed_tokens.eoi_embedding
    model = _prepare_model_for_qat(model, "int8-int4")
    if convert:
        _convert_torchao_model(model)
    emb = model.get_input_embeddings()
    assert isinstance(emb, _EoiEmbedding)
    assert emb.eoi_embedding is eoi
    with torch.no_grad():
        out = emb(torch.tensor([[0, 1, 2]]))
    assert torch.equal(out[0, 0], eoi)


def test_compiled_fake_quantized_embedding_pickles():
    import pickle

    model = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    emb = model.get_input_embeddings()
    emb.compile()
    restored = pickle.loads(pickle.dumps(emb))
    assert type(restored) is type(emb)
    assert "_compiled_call_impl" not in vars(restored) or restored._compiled_call_impl is None


def test_fake_quantized_embedding_subclass_pickles():
    import io
    import pickle

    model = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    emb = model.get_input_embeddings()
    assert _is_fake_quantized(emb)
    restored = pickle.loads(pickle.dumps(emb))
    assert type(restored) is type(emb)
    buf = io.BytesIO()
    torch.save(emb, buf)
    buf.seek(0)
    assert type(torch.load(buf, weights_only = False)) is type(emb)


def test_fake_quantized_embedding_unpickles_in_fresh_process(tmp_path):
    import os
    import pickle
    import subprocess
    import sys

    model = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    emb = model.get_input_embeddings()
    path = tmp_path / "emb.pkl"
    path.write_bytes(pickle.dumps(emb))
    ids = torch.tensor([[1, 2, 3]])
    expected = emb(ids)
    # Unpickling imports unsloth (the class lives in unsloth.models._utils), which refuses a
    # CPU-only torch. The child has no conftest, so on a GPU-less runner it gets the same spoof.
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    code = (
        "import pickle, sys, torch\n"
        f"sys.path.insert(0, {tests_dir!r})\n"
        "if not torch.cuda.is_available():\n"
        "    import _zoo_aggressive_cuda_spoof\n"
        "    _zoo_aggressive_cuda_spoof.apply()\n"
        f"emb = pickle.loads(open({str(path)!r}, 'rb').read())\n"
        "print(type(emb).__name__, float(emb.embed_scale))\n"
        "torch.save(emb(torch.tensor([[1, 2, 3]])), sys.argv[1])\n"
    )
    out_path = tmp_path / "out.pt"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.getcwd(), env.get("PYTHONPATH", "")])
    res = subprocess.run(
        [sys.executable, "-c", code, str(out_path)],
        capture_output = True,
        text = True,
        timeout = 600,
        env = env,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    assert res.stdout.split()[-2] == type(emb).__name__
    assert float(res.stdout.split()[-1]) == float(emb.embed_scale)
    assert torch.equal(torch.load(out_path), expected)


def test_builtin_recipe_flag_only_for_scheme_names():
    model = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    assert model._torchao_config._unsloth_builtin_recipe is True
    custom = _int8_embedding_config()
    model = _prepare_model_for_qat(_tiny_gemma3(), custom)
    assert not getattr(model._torchao_config, "_unsloth_builtin_recipe", False)


def _count_fake_quant_calls(emb):
    calls = []
    emb.weight_fake_quantizer.register_forward_hook(lambda *args: calls.append(1))
    return calls


def _expected_fake_quantized_lookup(emb, ids):
    # FakeQuantizedEmbedding's own lookup, then Gemma3's float32 embed_scale multiply.
    with torch.no_grad():
        w = emb.weight_fake_quantizer(emb.weight)
        return torch.nn.functional.embedding(ids, w).float() * emb.embed_scale


def _direct_weight_forward(self, input_ids):
    # unsloth_zoo's UNSLOTH_FORCE_FLOAT32 Gemma3 / Gemma4 embedding forward: reads self.weight, no super().
    embeds = torch.nn.functional.embedding(
        input_ids, weight = self.weight, padding_idx = self.padding_idx
    )
    return embeds.to(torch.float32) * self.embed_scale


@pytest.mark.parametrize("forward", ["upstream", "direct_weight"])
def test_gemma3_embedding_forward_is_fake_quantized_once(forward, monkeypatch):
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding

    if forward == "direct_weight":
        monkeypatch.setattr(Gemma3TextScaledWordEmbedding, "forward", _direct_weight_forward)
    model = _tiny_gemma3()
    ids = torch.randint(0, model.config.vocab_size, (2, 12))
    with torch.no_grad():
        unquantized = model.get_input_embeddings()(ids).float()
    model = _prepare_model_for_qat(model, "int8-int4")
    emb = model.get_input_embeddings()
    calls = _count_fake_quant_calls(emb)
    with torch.no_grad():
        out = emb(ids).float()
    assert len(calls) == 1
    assert not torch.equal(out, unquantized)
    torch.testing.assert_close(out, _expected_fake_quantized_lookup(emb, ids))
    # The swapped-in weight is scoped to the call.
    assert isinstance(emb.weight, torch.nn.Parameter)


def test_accelerate_hooked_embedding_runs_fake_quantized_forward():
    accelerate_hooks = pytest.importorskip("accelerate.hooks")

    model = _tiny_gemma3()
    original = model.get_input_embeddings()
    ids = torch.randint(0, model.config.vocab_size, (2, 12))
    with torch.no_grad():
        unquantized = original(ids).float()
    accelerate_hooks.add_hook_to_module(
        original, accelerate_hooks.AlignDevicesHook(execution_device = "cpu", io_same_device = True)
    )
    model = _prepare_model_for_qat(model, "int8-int4")
    emb = model.get_input_embeddings()
    assert emb is not original
    # The hook moved onto the replacement and wraps its forward, not the old module's.
    assert hasattr(emb, "_hf_hook")
    assert getattr(emb.forward, "args", ()) != (original,)
    calls = _count_fake_quant_calls(emb)
    with torch.no_grad():
        out = emb(ids).float()
    assert len(calls) == 1
    assert not torch.equal(out, unquantized)
    torch.testing.assert_close(out, _expected_fake_quantized_lookup(emb, ids))


def test_fake_quantized_embedding_forward_compiles_fullgraph():
    model = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    emb = model.get_input_embeddings()
    ids = torch.randint(0, model.config.vocab_size, (2, 12))
    torch._dynamo.reset()
    compiled = torch.compile(emb, backend = "eager", fullgraph = True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(ids), emb(ids))


def _expected_lookup_with(emb, weight, ids):
    with torch.no_grad():
        w = emb.weight_fake_quantizer(weight)
        return torch.nn.functional.embedding(ids, w).float() * emb.embed_scale


def test_functional_call_plain_tensor_weight_is_fake_quantized():
    emb = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4").get_input_embeddings()
    ids = torch.randint(0, 256, (2, 12))
    weight = emb.weight.detach().clone().requires_grad_()
    calls = _count_fake_quant_calls(emb)
    out = torch.func.functional_call(emb, {"weight": weight}, (ids,)).float()
    assert len(calls) == 1
    torch.testing.assert_close(out.detach(), _expected_lookup_with(emb, weight, ids))
    out.sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert isinstance(emb.weight, torch.nn.Parameter)


def test_nested_fake_quantized_weight_call_quantizes_once():
    from unsloth.models._utils import _call_with_fake_quantized_weight

    emb = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4").get_input_embeddings()
    ids = torch.randint(0, 256, (2, 12))
    calls = _count_fake_quant_calls(emb)

    def lookup():
        return torch.nn.functional.embedding(ids, emb.weight).float() * emb.embed_scale

    with torch.no_grad():
        out = _call_with_fake_quantized_weight(emb, _call_with_fake_quantized_weight, emb, lookup)
    assert len(calls) == 1
    torch.testing.assert_close(out, _expected_fake_quantized_lookup(emb, ids))
    assert "_unsloth_weight_fake_quantized" not in vars(emb)


@pytest.mark.parametrize("use_orig_params", [False, True])
def test_fsdp1_embedding_is_fake_quantized(use_orig_params, tmp_path):
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires torch.distributed with gloo")
    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    emb = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4").get_input_embeddings()
    ids = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        expected = emb(ids).float()
    dist.init_process_group(
        "gloo", store = dist.FileStore(str(tmp_path / "store"), 1), rank = 0, world_size = 1
    )
    try:
        wrapped = FSDP(emb, device_id = torch.device("cpu"), use_orig_params = use_orig_params)
        calls = _count_fake_quant_calls(emb)
        seen = []
        emb.register_forward_pre_hook(
            lambda m, a: seen.append(isinstance(m.weight, torch.nn.Parameter))
        )
        out = wrapped(ids).float()
        out.sum().backward()
        grads = [p.grad for p in wrapped.parameters() if p.requires_grad]
    finally:
        dist.destroy_process_group()
    assert seen == [False]
    assert len(calls) == 1
    torch.testing.assert_close(out.detach(), expected)
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def _offload_embedding(model, **hook_kwargs):
    # What cpu_offload / dispatch_model install: the real weight in weights_map, the parameter on meta between calls.
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from accelerate.utils import PrefixedDataset

    prefix = "model.embed_tokens."
    weights = {k: v.clone() for k, v in model.state_dict().items() if k.startswith(prefix)}
    add_hook_to_module(
        model.get_input_embeddings(),
        AlignDevicesHook(
            execution_device = "cpu",
            offload = True,
            weights_map = PrefixedDataset(weights, prefix),
            **hook_kwargs,
        ),
    )
    assert model.get_input_embeddings().weight.device.type == "meta"
    return model


def test_offloaded_embedding_is_fake_quantized():
    pytest.importorskip("accelerate.hooks")
    ids = torch.randint(0, 256, (2, 12))
    reference = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4").get_input_embeddings()
    with torch.no_grad():
        expected = reference(ids).float()

    model = _prepare_model_for_qat(_offload_embedding(_tiny_gemma3()), "int8-int4")
    emb = model.get_input_embeddings()
    assert getattr(emb._hf_hook, "offload", False)
    calls = _count_fake_quant_calls(emb)
    with torch.no_grad():
        out = emb(ids).float()
    assert out.device.type == "cpu"
    assert len(calls) == 1
    torch.testing.assert_close(out, expected)
    assert emb.weight.device.type == "meta"  # offloaded again after the call


def test_offloaded_embedding_hook_that_cannot_move_is_refused():
    pytest.importorskip("accelerate.hooks")
    model = _offload_embedding(_tiny_gemma3(), place_submodules = True)
    original = model.get_input_embeddings()
    with pytest.raises(RuntimeError, match = "offloaded embedding"):
        _prepare_model_for_qat(model, "int8-int4")
    assert model.get_input_embeddings() is original


@pytest.mark.parametrize("convert", ["unsloth", "strip"])
def test_offloaded_embedding_converts_to_a_real_embedding(convert):
    pytest.importorskip("accelerate.hooks")
    from torchao.utils import TorchAOBaseTensor
    from unsloth.models._utils import _strip_torchao_fake_quantization

    ids = torch.randint(0, 256, (2, 12))
    reference = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    model = _prepare_model_for_qat(_offload_embedding(_tiny_gemma3()), "int8-int4")
    if convert == "unsloth":
        _convert_torchao_model(reference)
        _convert_torchao_model(model)
    else:
        _strip_torchao_fake_quantization(reference)
        _strip_torchao_fake_quantization(model)
    emb = model.get_input_embeddings()
    assert emb.weight.device.type == "cpu"
    assert not hasattr(emb, "_hf_hook")
    assert isinstance(emb.weight, TorchAOBaseTensor) == (convert == "unsloth")
    with torch.no_grad():
        torch.testing.assert_close(emb(ids), reference.get_input_embeddings()(ids))


def test_offloaded_embedding_converts_on_cpu_not_the_execution_device():
    pytest.importorskip("accelerate.hooks")
    model = _prepare_model_for_qat(_offload_embedding(_tiny_gemma3()), "int8-int4")
    emb = model.get_input_embeddings()
    # A real offload runs on a GPU; conversion must not pull the whole table onto it.
    emb._hf_hook.execution_device = torch.device("cuda", 7)
    _convert_torchao_model(model)
    emb = model.get_input_embeddings()
    assert emb.weight.device.type == "cpu"
    assert all(b.device.type == "cpu" for b in emb.buffers())


@pytest.mark.parametrize("qat_scheme", ["int4", "int8"])
def test_offload_hook_on_unselected_embedding_is_not_refused(qat_scheme):
    pytest.importorskip("accelerate.hooks")
    model = _offload_embedding(_tiny_gemma3(), place_submodules = True)
    original = model.get_input_embeddings()
    model = _prepare_model_for_qat(
        model, qat_scheme
    )  # Linear-only recipes never touch the embedding
    assert model.get_input_embeddings() is original
    assert _is_fake_quantized(
        model.model.layers[0].mlp.down_proj
    )  # in_features 128 fits int4's group


@pytest.mark.parametrize("convert", ["unsloth", "strip"])
def test_offloaded_embedding_with_offloaded_buffers_converts(convert):
    # offload_buffers = True on a meta-initialized module: weight AND embed_scale live only in weights_map.
    pytest.importorskip("accelerate.hooks")
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from accelerate.utils import PrefixedDataset
    from unsloth.models._utils import _strip_torchao_fake_quantization

    ids = torch.randint(0, 256, (2, 12))
    reference = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    model = _tiny_gemma3()
    emb = model.get_input_embeddings()
    scale = emb.embed_scale.detach().clone()
    emb.register_buffer("embed_scale", scale.clone(), persistent = True)
    prefix = "model.embed_tokens."
    weights = {
        prefix + "weight": emb.weight.detach().clone(),
        prefix + "embed_scale": scale.clone(),
    }
    emb.to("meta")
    add_hook_to_module(
        emb,
        AlignDevicesHook(
            execution_device = "cpu",
            offload = True,
            offload_buffers = True,
            weights_map = PrefixedDataset(weights, prefix),
        ),
    )
    assert emb.embed_scale.device.type == "meta"
    model = _prepare_model_for_qat(model, "int8-int4")
    with torch.no_grad():  # the hook still serves forward from weights_map
        torch.testing.assert_close(
            model.get_input_embeddings()(ids), reference.get_input_embeddings()(ids)
        )
    if convert == "unsloth":
        _convert_torchao_model(reference)
        _convert_torchao_model(model)
    else:
        _strip_torchao_fake_quantization(reference)
        _strip_torchao_fake_quantization(model)
    emb = model.get_input_embeddings()
    assert not hasattr(emb, "_hf_hook")
    assert emb.weight.device.type == "cpu"
    assert emb.embed_scale.device.type == "cpu"
    assert float(emb.embed_scale) == float(scale)
    with torch.no_grad():
        torch.testing.assert_close(emb(ids), reference.get_input_embeddings()(ids))


def _direct_weight_override(self, input_ids):
    return torch.nn.functional.embedding(input_ids, self.weight).float() * self.embed_scale


def _delegating_override(self, input_ids):
    return type(self).forward(self, input_ids)


def _install_override(emb, kind):
    import types
    if kind == "direct_weight":
        emb.forward = types.MethodType(_direct_weight_override, emb)
    elif kind == "delegating":
        emb.forward = types.MethodType(_delegating_override, emb)
    elif kind == "accelerate_over_override":
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module
        emb.forward = types.MethodType(_delegating_override, emb)
        add_hook_to_module(emb, AlignDevicesHook(execution_device = "cpu", io_same_device = True))
    elif kind == "forward_hook":
        return emb.register_forward_hook(lambda m, a, out: out * 2)
    elif kind == "unsloth_offload_hooks":
        from unsloth.models.vision import _install_offload_embedding_hooks
        assert _install_offload_embedding_hooks(emb, None, torch.device("cpu"))
    return None


@pytest.mark.parametrize(
    "kind",
    [
        "direct_weight",
        "delegating",
        "accelerate_over_override",
        "forward_hook",
        "unsloth_offload_hooks",
    ],
)
@pytest.mark.parametrize("convert", ["unsloth", "strip"])
def test_embedding_with_instance_forward_or_hooks_is_left_out_of_qat(kind, convert):
    # Nothing is rebound: the embedding keeps its module, override and hooks, and stays full precision throughout.
    pytest.importorskip("accelerate.hooks")
    from torchao.utils import TorchAOBaseTensor
    from unsloth.models._utils import _strip_torchao_fake_quantization

    model = _tiny_gemma3()
    original = model.get_input_embeddings()
    handle = _install_override(original, kind)
    ids = torch.randint(0, model.config.vocab_size, (2, 12))
    with torch.no_grad():
        before = original(ids)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, "int8-int4")
    emb = model.get_input_embeddings()
    assert emb is original and not _is_fake_quantized(emb)
    assert sum("leaves these embeddings in full precision" in str(w.message) for w in caught) == 1
    assert _is_fake_quantized(
        model.model.layers[0].self_attn.q_proj
    )  # the rest of the recipe still applies
    with torch.no_grad():
        torch.testing.assert_close(emb(ids), before)
    if handle is not None:
        handle.remove()  # the caller's handle still controls the live module
        with torch.no_grad():
            torch.testing.assert_close(emb(ids) * 2, before)
    if convert == "unsloth":
        _convert_torchao_model(model)
    else:
        _strip_torchao_fake_quantization(model)
    assert model.get_input_embeddings() is original
    assert not isinstance(original.weight, TorchAOBaseTensor)  # saved as trained: unquantized


@pytest.mark.parametrize("offload", [False, True])
def test_sequential_hook_with_align_devices_hook_is_supported(offload):
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    from accelerate.utils import PrefixedDataset
    from torchao.utils import TorchAOBaseTensor

    ids = torch.randint(0, 256, (2, 12))
    reference = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    with torch.no_grad():
        expected = reference.get_input_embeddings()(ids)
    model = _tiny_gemma3()
    emb = model.get_input_embeddings()
    kwargs = {"execution_device": "cpu", "offload": offload}
    if offload:
        prefix = "model.embed_tokens."
        kwargs["weights_map"] = PrefixedDataset(
            {prefix + "weight": emb.weight.detach().clone()}, prefix
        )
    accelerate_hooks.add_hook_to_module(emb, accelerate_hooks.AlignDevicesHook(**kwargs))
    accelerate_hooks.add_hook_to_module(emb, accelerate_hooks.ModelHook(), append = True)
    assert type(emb._hf_hook).__name__ == "SequentialHook"
    model = _prepare_model_for_qat(model, "int8-int4")
    emb = model.get_input_embeddings()
    assert _is_fake_quantized(emb) and type(emb._hf_hook).__name__ == "SequentialHook"
    calls = _count_fake_quant_calls(emb)
    with torch.no_grad():
        torch.testing.assert_close(emb(ids), expected)
    assert len(calls) == 1
    _convert_torchao_model(reference)
    _convert_torchao_model(model)
    emb = model.get_input_embeddings()
    assert emb.weight.device.type == "cpu" and isinstance(emb.weight, TorchAOBaseTensor)
    with torch.no_grad():
        torch.testing.assert_close(emb(ids), reference.get_input_embeddings()(ids))


def test_two_offload_hooks_are_left_out_of_qat():
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    model = _tiny_gemma3()
    emb = model.get_input_embeddings()
    from accelerate.utils import PrefixedDataset

    weights = {"w.weight": emb.weight.detach().clone()}
    for append in (False, True):
        hook = accelerate_hooks.AlignDevicesHook(
            execution_device = "cpu", offload = True, weights_map = PrefixedDataset(weights, "w.")
        )
        accelerate_hooks.add_hook_to_module(emb, hook, append = append)
    with pytest.warns(UserWarning, match = "2 AlignDevicesHooks"):
        model = _prepare_model_for_qat(model, "int8-int4")
    assert model.get_input_embeddings() is emb and not _is_fake_quantized(emb)


@pytest.mark.parametrize("offload", [False, True])
def test_plain_embedding_keeps_its_accelerate_hook(offload):
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    from torchao.quantization.qat.embedding import FakeQuantizedEmbedding
    from torchao.utils import TorchAOBaseTensor

    ids = torch.randint(0, 256, (2, 12))
    reference = _prepare_model_for_qat(_tiny_qwen3(), "int8-int4")
    with torch.no_grad():
        expected = reference.get_input_embeddings()(ids)
    model = _tiny_qwen3()
    emb = model.get_input_embeddings()
    assert type(emb) is torch.nn.Embedding
    accelerate_hooks.add_hook_to_module(
        emb,
        accelerate_hooks.AlignDevicesHook(
            execution_device = "cpu", offload = offload, io_same_device = True
        ),
    )
    model = _prepare_model_for_qat(model, "int8-int4")
    emb = model.get_input_embeddings()
    assert type(emb) is FakeQuantizedEmbedding and hasattr(emb, "_hf_hook")
    calls = _count_fake_quant_calls(emb)
    with torch.no_grad():
        torch.testing.assert_close(emb(ids), expected)
    assert len(calls) == 1
    _convert_torchao_model(reference)
    _convert_torchao_model(model)
    emb = model.get_input_embeddings()
    assert emb.weight.device.type == "cpu" and isinstance(emb.weight, TorchAOBaseTensor)


def test_input_require_grads_hook_follows_the_qat_embedding():
    # transformers' enable_input_require_grads hook is not a reason to skip QAT; it is re-registered on the swap.
    model = _tiny_gemma3()
    model.enable_input_require_grads()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, "int8-int4")
    assert not any("leaves these embeddings" in str(w.message) for w in caught)
    emb = model.get_input_embeddings()
    assert _is_fake_quantized(emb)
    ids = torch.randint(0, model.config.vocab_size, (2, 12))
    for p in emb.parameters():
        p.requires_grad_(False)
    assert emb(ids).requires_grad
    model.disable_input_require_grads()  # the owner's handles control the live embedding
    assert not emb(ids).requires_grad


def test_peft_require_grad_hook_follows_the_qat_embedding():
    # PEFT's fallback hook (no handle kept) on a model without enable_input_require_grads.
    model = _tiny_gemma3()
    emb = model.get_input_embeddings()

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    emb.register_forward_hook(make_inputs_require_grad)
    model = _prepare_model_for_qat(model, "int8-int4")
    emb = model.get_input_embeddings()
    assert _is_fake_quantized(emb)
    assert list(emb._forward_hooks.values()) == [make_inputs_require_grad]
    for p in emb.parameters():
        p.requires_grad_(False)
    assert emb(torch.tensor([[1, 2, 3]])).requires_grad
