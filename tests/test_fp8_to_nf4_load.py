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

"""`load_in_4bit = True` on a block-FP8 checkpoint must give NF4 bytes identical to loading the
same checkpoint dequantized to bf16 by an independent formula."""

import json
import os
import shutil
from types import SimpleNamespace

import pytest
import torch

import unsloth  # noqa: F401  (patches before transformers)
from unsloth import FastLanguageModel, FastModel
from unsloth.models.loader_utils import check_and_disable_bitsandbytes_loading

_BLOCK = 128
_TOKENIZER_REPO = "trl-internal-testing/tiny-Qwen3ForCausalLM"
_KEEP_16BIT = "model.layers.1.self_attn.o_proj"

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "bitsandbytes 4bit needs a GPU"
)


def _feature_unavailable():
    try:
        from unsloth.models.fp8_to_nf4 import fp8_to_nf4_unavailable_reason
    except ImportError:
        return None  # main: no module, the tests must fail there
    return fp8_to_nf4_unavailable_reason()


needs_feature = pytest.mark.skipif(
    _feature_unavailable() is not None,
    reason = f"fp8 -> 4bit loading unavailable: {_feature_unavailable()}",
)
needs_fp8_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason = "fp8 checkpoints load natively only on sm89+",
)


def _quantize_block_fp8(weight):
    rows, cols = weight.shape
    grid_r, grid_c = -(-rows // _BLOCK), -(-cols // _BLOCK)
    padded = torch.zeros(grid_r * _BLOCK, grid_c * _BLOCK, dtype = torch.float32)
    padded[:rows, :cols] = weight.float()
    blocks = padded.view(grid_r, _BLOCK, grid_c, _BLOCK)
    scale = blocks.abs().amax(dim = (1, 3)).clamp(min = 1e-12) / 448.0
    quant = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    quant = quant.view(grid_r * _BLOCK, grid_c * _BLOCK)[:rows, :cols].contiguous()
    return quant, scale.contiguous()


def _dequantize_reference(quant, scale):
    rows, cols = quant.shape
    expanded = (
        scale.float().repeat_interleave(_BLOCK, 0)[:rows].repeat_interleave(_BLOCK, 1)[:, :cols]
    )
    return (quant.float() * expanded).to(torch.bfloat16)


def _is_fp8_candidate(key, tensor):
    if not key.endswith(".weight") or tensor.dim() != 2:
        return False
    if ".layers." not in key or "norm" in key or ".mlp.gate.weight" in key:
        return False
    return not key.startswith(_KEEP_16BIT + ".")


def _write_pair(bf16_dir, root):
    """bf16 save -> (fp8 dir, dequantized-bf16 dir)."""
    from safetensors.torch import load_file, save_file

    fp8_dir, deq_dir = os.path.join(root, "fp8"), os.path.join(root, "deq")
    for d in (fp8_dir, deq_dir):
        os.makedirs(d, exist_ok = True)
        for name in os.listdir(bf16_dir):
            if not name.endswith(".safetensors") and not name.endswith(".index.json"):
                shutil.copy2(os.path.join(bf16_dir, name), os.path.join(d, name))
    state = {}
    for name in sorted(os.listdir(bf16_dir)):
        if name.endswith(".safetensors"):
            state.update(load_file(os.path.join(bf16_dir, name)))
    fp8_state, deq_state = {}, {}
    for key, tensor in state.items():
        if _is_fp8_candidate(key, tensor):
            quant, scale = _quantize_block_fp8(tensor)
            fp8_state[key] = quant
            fp8_state[key + "_scale_inv"] = scale
            deq_state[key] = _dequantize_reference(quant, scale)
        else:
            fp8_state[key] = tensor
            deq_state[key] = tensor
    save_file(fp8_state, os.path.join(fp8_dir, "model.safetensors"), metadata = {"format": "pt"})
    save_file(deq_state, os.path.join(deq_dir, "model.safetensors"), metadata = {"format": "pt"})
    with open(os.path.join(fp8_dir, "config.json")) as f:
        config = json.load(f)
    config["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [_BLOCK, _BLOCK],
        "modules_to_not_convert": ["lm_head", "model.embed_tokens", _KEEP_16BIT],
    }
    with open(os.path.join(fp8_dir, "config.json"), "w") as f:
        json.dump(config, f, indent = 2)
    return fp8_dir, deq_dir


def _build(root, moe):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_REPO)
    common = dict(
        vocab_size = len(tokenizer),
        hidden_size = 256,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 64,
        max_position_embeddings = 512,
        tie_word_embeddings = False,
        torch_dtype = "bfloat16",
    )
    torch.manual_seed(0)
    if moe:
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        config = Qwen3MoeConfig(
            intermediate_size = 320,
            moe_intermediate_size = 192,  # ragged: 1.5 blocks
            num_experts = 4,
            num_experts_per_tok = 2,
            decoder_sparse_step = 1,
            **common,
        )
        model = Qwen3MoeForCausalLM(config)
    else:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(intermediate_size = 320, **common)  # ragged: 2.5 blocks
        model = Qwen3ForCausalLM(config)
    model = model.to(torch.bfloat16)
    bf16_dir = os.path.join(root, "bf16")
    model.save_pretrained(bf16_dir, safe_serialization = True)
    tokenizer.save_pretrained(bf16_dir)
    return _write_pair(bf16_dir, root)


@pytest.fixture(scope = "module")
def dense_pair(tmp_path_factory):
    return _build(str(tmp_path_factory.mktemp("fp8_dense")), moe = False)


@pytest.fixture(scope = "module")
def moe_pair(tmp_path_factory):
    return _build(str(tmp_path_factory.mktemp("fp8_moe")), moe = True)


def _fingerprint(model):
    """name -> tensors that define the loaded weight: packed NF4 bytes and the full quant state."""
    out = {}
    for name, param in model.named_parameters():
        state = getattr(param, "quant_state", None)
        entry = {"data": param.data.detach().clone().cpu()}
        if state is not None:
            entry["absmax"] = state.absmax.detach().cpu()
            entry["shape"] = tuple(state.shape)
            entry["dtype"] = state.dtype
            entry["blocksize"] = state.blocksize
            entry["quant_type"] = state.quant_type
            if getattr(state, "nested", False):
                entry["offset"] = state.offset.detach().cpu()
                entry["absmax2"] = state.state2.absmax.detach().cpu()
                entry["code2"] = state.state2.code.detach().cpu()
        out[name] = entry
    return out


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for name in a:
        ea, eb = a[name], b[name]
        assert ea.keys() == eb.keys(), name
        for key in ea:
            va, vb = ea[key], eb[key]
            if isinstance(va, torch.Tensor):
                assert va.dtype == vb.dtype and va.shape == vb.shape, (name, key)
                assert torch.equal(
                    va.reshape(-1).view(torch.uint8), vb.reshape(-1).view(torch.uint8)
                ), (name, key)
            else:
                assert va == vb, (name, key)


def _count_4bit(model):
    return sum(1 for _, p in model.named_parameters() if type(p).__name__ == "Params4bit")


def _count_fp8(model):
    return sum(1 for _, p in model.named_parameters() if p.dtype == torch.float8_e4m3fn)


def _load(loader, path, **kwargs):
    model, _ = loader.from_pretrained(path, max_seq_length = 64, dtype = torch.bfloat16, **kwargs)
    return model


def _reference_config():
    from transformers import BitsAndBytesConfig
    from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES
    return BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16,
        llm_int8_skip_modules = SKIP_QUANTIZATION_MODULES + [_KEEP_16BIT],
    )


def _free(*models):
    for model in models:
        del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()


@needs_cuda
@needs_feature
def test_dense_fp8_loads_nf4_bit_identical_to_dequantized_bf16(dense_pair, capsys):
    fp8_dir, deq_dir = dense_pair
    model = _load(FastLanguageModel, fp8_dir, load_in_4bit = True)
    printed = capsys.readouterr().out
    assert _count_4bit(model) == 13  # 2 x (q, k, v, o, gate, up, down) minus the 16-bit o_proj
    assert _count_fp8(model) == 0
    assert "dequantizing each fp8 tensor" in printed
    assert not hasattr(model.config, "_unsloth_fp8_to_nf4")
    ours = _fingerprint(model)
    _free(model)
    reference = _load(FastLanguageModel, deq_dir, quantization_config = _reference_config())
    theirs = _fingerprint(reference)
    _free(reference)
    _assert_same(ours, theirs)


@needs_cuda
@needs_feature
def test_moe_fp8_loads_nf4_experts_bit_identical(moe_pair):
    fp8_dir, deq_dir = moe_pair
    model = _load(FastModel, fp8_dir, load_in_4bit = True)
    experts = [
        name
        for name, p in model.named_parameters()
        if ".experts." in name and type(p).__name__ == "Params4bit"
    ]
    # gate_up_proj and down_proj stacks on both layers go through Unsloth's bnb expert path.
    assert len(experts) == 4, experts
    assert _count_fp8(model) == 0
    ours = _fingerprint(model)
    _free(model)
    reference = _load(FastModel, deq_dir, quantization_config = _reference_config())
    theirs = _fingerprint(reference)
    _free(reference)
    _assert_same(ours, theirs)


@needs_cuda
@needs_feature
def test_checkpoint_16bit_modules_stay_16bit(dense_pair):
    from safetensors.torch import load_file

    fp8_dir, _ = dense_pair
    model = _load(FastLanguageModel, fp8_dir, load_in_4bit = True)
    module = model.get_submodule(_KEEP_16BIT)
    assert type(module) is torch.nn.Linear
    stored = load_file(os.path.join(fp8_dir, "model.safetensors"))[_KEEP_16BIT + ".weight"]
    assert stored.dtype == torch.bfloat16
    assert torch.equal(module.weight.detach().cpu(), stored)
    quantization_config = model.config.quantization_config
    if not isinstance(quantization_config, dict):
        quantization_config = quantization_config.to_dict()
    # Saved with the model, so a reload of the 4bit save keeps it in 16bit too.
    assert any(
        p.replace("\\", "").rstrip("$") == _KEEP_16BIT
        for p in quantization_config["llm_int8_skip_modules"]
    )
    _free(model)


@needs_fp8_gpu
@needs_feature
def test_default_load_in_4bit_and_kill_switch_keep_fp8(dense_pair, monkeypatch):
    fp8_dir, _ = dense_pair
    # load_in_4bit left at its default is not a request: unchanged fp8 load.
    model = _load(FastLanguageModel, fp8_dir)
    assert _count_4bit(model) == 0 and _count_fp8(model) > 0
    _free(model)
    monkeypatch.setenv("UNSLOTH_FP8_TO_NF4", "0")
    model = _load(FastLanguageModel, fp8_dir, load_in_4bit = True)
    assert _count_4bit(model) == 0 and _count_fp8(model) > 0
    _free(model)
    monkeypatch.setenv("UNSLOTH_FP8_TO_NF4", "1")
    model = _load(FastLanguageModel, fp8_dir)
    assert _count_4bit(model) == 13 and _count_fp8(model) == 0
    _free(model)


@needs_cuda
@needs_feature
def test_fp8_keys_the_model_does_not_have_are_ignored(dense_pair, tmp_path):
    """MTP layers ship in fp8 but transformers drops them as unexpected (DeepSeek-V3, GLM-4 MoE,
    Qwen3-Next): they must not count as unapplied scales."""
    from safetensors.torch import load_file, save_file

    fp8_dir, _ = dense_pair
    extra_dir = str(tmp_path / "fp8_mtp")
    shutil.copytree(fp8_dir, extra_dir)
    state = load_file(os.path.join(extra_dir, "model.safetensors"))
    quant, scale = _quantize_block_fp8(torch.randn(320, 256) * 0.02)
    state["mtp.layers.0.mlp.gate_proj.weight"] = quant
    state["mtp.layers.0.mlp.gate_proj.weight_scale_inv"] = scale
    save_file(state, os.path.join(extra_dir, "model.safetensors"), metadata = {"format": "pt"})
    model = _load(FastLanguageModel, extra_dir, load_in_4bit = True)
    assert _count_4bit(model) == 13
    ours = _fingerprint(model)
    _free(model)
    plain = _load(FastLanguageModel, fp8_dir, load_in_4bit = True)
    theirs = _fingerprint(plain)
    _free(plain)
    _assert_same(ours, theirs)


@needs_cuda
@needs_feature
def test_quantize_16bit_switch_matches_plain_4bit_skip_list(dense_pair, monkeypatch):
    fp8_dir, deq_dir = dense_pair
    monkeypatch.setenv("UNSLOTH_FP8_TO_NF4_QUANTIZE_16BIT", "1")
    model = _load(FastLanguageModel, fp8_dir, load_in_4bit = True)
    assert _count_4bit(model) == 14
    ours = _fingerprint(model)
    _free(model)
    reference = _load(FastLanguageModel, deq_dir, load_in_4bit = True)
    theirs = _fingerprint(reference)
    _free(reference)
    _assert_same(ours, theirs)


def _zoo_reads_fp8_block_size_from_disk():
    try:
        from unsloth_zoo import saving_utils
    except Exception:
        return False
    return hasattr(saving_utils, "_fp8_block_size_on_disk")


@needs_cuda
@needs_feature
@pytest.mark.skipif(
    not _zoo_reads_fp8_block_size_from_disk(),
    reason = "unsloth_zoo takes the fp8 block size of a 16bit merge from the in-memory config only",
)
def test_moe_merged_16bit_save_equals_dequantized_checkpoint(moe_pair, tmp_path):
    from safetensors.torch import load_file

    fp8_dir, deq_dir = moe_pair
    model, tokenizer = FastModel.from_pretrained(
        fp8_dir, max_seq_length = 64, dtype = torch.bfloat16, load_in_4bit = True
    )
    assert _count_4bit(model) > 0
    model = FastModel.get_peft_model(
        model, r = 8, target_modules = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # lora_B starts at zero, so the merge must give back the dequantized fp8 weights exactly,
    # read from the fp8 shards on disk rather than the NF4 copy in memory.
    model.save_pretrained_merged(str(tmp_path / "merged"), tokenizer, save_method = "merged_16bit")
    _free(model)
    merged = {}
    for name in os.listdir(tmp_path / "merged"):
        if name.endswith(".safetensors"):
            merged.update(load_file(str(tmp_path / "merged" / name)))
    reference = load_file(os.path.join(deq_dir, "model.safetensors"))
    assert merged.keys() == reference.keys()
    for key, tensor in reference.items():
        assert torch.equal(merged[key], tensor), key


@needs_fp8_gpu
@pytest.mark.skipif(_feature_unavailable() is None, reason = "only where fp8 -> 4bit cannot run")
def test_unavailable_keeps_todays_fp8_load(dense_pair, capsys):
    fp8_dir, _ = dense_pair
    model = _load(FastLanguageModel, fp8_dir, load_in_4bit = True)
    assert _count_4bit(model) == 0 and _count_fp8(model) > 0
    assert "cannot quantize to 4bit here" in capsys.readouterr().out
    _free(model)


def _fp8_config():
    return SimpleNamespace(
        model_type = "qwen3",
        quantization_config = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head"],
        },
    )


@needs_feature
def test_arming_needs_an_explicit_request(monkeypatch):
    from unsloth.models import fp8_to_nf4

    monkeypatch.delenv("UNSLOTH_FP8_TO_NF4", raising = False)
    config = _fp8_config()
    assert check_and_disable_bitsandbytes_loading(config, load_in_4bit = True, verbose = False)[:2] == (
        False,
        False,
    )
    assert config.quantization_config["quant_method"] == "fp8"

    inside = {}

    @fp8_to_nf4.track_explicit_4bit_request
    def from_pretrained(
        model_name = None,
        load_in_4bit = True,
        **kwargs,
    ):
        flags = check_and_disable_bitsandbytes_loading(
            config, load_in_4bit = load_in_4bit, verbose = False
        )
        inside["armed"] = fp8_to_nf4.fp8_to_nf4_armed(config)
        inside["stripped"] = not hasattr(config, "quantization_config")
        return flags

    assert from_pretrained("x")[:2] == (False, False)
    config = _fp8_config()
    assert from_pretrained("x", load_in_4bit = True)[:2] == (True, False)
    assert inside == {"armed": True, "stripped": True}
    # The outermost from_pretrained hands the caller's config back as it was.
    assert config.quantization_config["quant_method"] == "fp8"
    assert not fp8_to_nf4.fp8_to_nf4_armed(config)

    config = _fp8_config()
    monkeypatch.setenv("UNSLOTH_FP8_TO_NF4", "0")
    assert from_pretrained("x", load_in_4bit = True)[:2] == (False, False)
    monkeypatch.setenv("UNSLOTH_FP8_TO_NF4", "1")
    assert from_pretrained("x")[:2] == (True, False)


def test_non_block_fp8_and_8bit_are_not_armed(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FP8_TO_NF4", "1")
    config = _fp8_config()
    config.quantization_config.pop("weight_block_size")
    assert check_and_disable_bitsandbytes_loading(config, load_in_4bit = True, verbose = False)[:2] == (
        False,
        False,
    )
    config = _fp8_config()
    assert check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = False, load_in_8bit = True, verbose = False
    )[:2] == (False, False)
    assert config.quantization_config["quant_method"] == "fp8"


@needs_feature
def test_scaled_weights_stored_as_integers_are_refused():
    from unsloth.models import fp8_to_nf4
    fp8_to_nf4._refuse_packed_scaled_weights(
        {"a.weight": "F8_E4M3", "a.weight_scale_inv": "F32", "b.weight": "BF16", "n.scale": "F32"}
    )
    with pytest.raises(RuntimeError, match = "cannot dequantize"):
        fp8_to_nf4._refuse_packed_scaled_weights({"e.weight": "I8", "e.scale": "F8_E8M0"})


@needs_feature
def test_packed_fp4_expert_checkpoints_are_not_armed(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FP8_TO_NF4", "1")
    config = _fp8_config()
    config.quantization_config["expert_dtype"] = "fp4"
    assert check_and_disable_bitsandbytes_loading(config, load_in_4bit = True, verbose = False)[:2] == (
        False,
        False,
    )
    assert config.quantization_config["quant_method"] == "fp8"


@pytest.mark.parametrize("shape", [(256, 384), (320, 256), (192, 200), (3, 320, 192)])
def test_block_dequant_matches_reference_formula(shape):
    torch.manual_seed(1)
    weight = torch.randn(*shape) * 0.02
    if len(shape) == 2:
        quant, scale = _quantize_block_fp8(weight)
        expected = _dequantize_reference(quant, scale)
    else:
        pairs = [_quantize_block_fp8(w) for w in weight]
        quant = torch.stack([q for q, _ in pairs])
        scale = torch.stack([s for _, s in pairs])
        expected = torch.stack([_dequantize_reference(q, s) for q, s in pairs])
    from unsloth.models import fp8_to_nf4

    got = fp8_to_nf4._dequantize_block_fp8(quant, scale, [128, 128], torch.bfloat16)
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


@needs_feature
def test_stacked_16bit_weight_with_a_scale_is_copied_into_the_stack():
    import unsloth.models.fp8_to_nf4 as fp8_to_nf4

    dequantize_op, _ = fp8_to_nf4._build_classes()
    torch.manual_seed(2)
    quant, scale = _quantize_block_fp8(torch.randn(128, 128) * 0.02)
    stored_16bit = (torch.randn(128, 128) * 0.02).to(torch.bfloat16)
    fp8_to_nf4._STATE = SimpleNamespace(
        block = [128, 128], dtype = torch.bfloat16, kept_fp8 = 0, dequantized = 0
    )
    try:
        result = dequantize_op(stack = True).convert(
            {
                "mlp.experts.*.down_proj.weight": [quant, stored_16bit],
                "mlp.experts.*.down_proj.weight_scale_inv": [scale, torch.ones(1, 1)],
            },
            target_patterns = ["mlp.experts.down_proj"],
        )
    finally:
        fp8_to_nf4._STATE = None
    stack = result["mlp.experts.*.down_proj.weight"]
    assert torch.equal(stack[0], _dequantize_reference(quant, scale))
    assert torch.equal(stack[1], stored_16bit)


@needs_feature
def test_a_load_that_raises_before_loading_hands_the_config_back(monkeypatch):
    # Prefetch or device-map planning can raise after arming, before the load restores the config.
    from unsloth.models import fp8_to_nf4

    monkeypatch.delenv("UNSLOTH_FP8_TO_NF4", raising = False)
    config = _fp8_config()
    original = dict(config.quantization_config)

    @fp8_to_nf4.track_explicit_4bit_request
    def from_pretrained(model_name = None, load_in_4bit = True, **kwargs):
        assert check_and_disable_bitsandbytes_loading(config, load_in_4bit = load_in_4bit, verbose = False)[0]
        assert fp8_to_nf4.fp8_to_nf4_armed(config)
        raise RuntimeError("prefetch stalled")

    with pytest.raises(RuntimeError, match = "prefetch stalled"):
        from_pretrained("x", load_in_4bit = True)
    assert config.quantization_config == original
    assert not fp8_to_nf4.fp8_to_nf4_armed(config)
