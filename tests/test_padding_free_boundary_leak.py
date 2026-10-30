# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

try:
    import unsloth.trainer as trainer_module
except ImportError:
    import unsloth
    if getattr(unsloth, "DEVICE_TYPE", None) != "mlx":
        raise
    pytest.skip("unsloth.trainer is the MLX shim here", allow_module_level = True)


@pytest.fixture(autouse = True)
def _return_logits_unset(monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    monkeypatch.delenv("UNSLOTH_PADDING_CHOICE", raising = False)


def _tiny(model_type):
    config = transformers.AutoConfig.for_model(model_type)
    common = dict(
        hidden_size = 64,
        intermediate_size = 128,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        vocab_size = 256,
        max_position_embeddings = 512,
    )
    if model_type == "deepseek_v4":
        common.update(
            num_key_value_heads = 1,
            head_dim = 32,
            moe_intermediate_size = 64,
            n_routed_experts = 4,
            num_experts_per_tok = 2,
            q_lora_rank = 32,
            o_lora_rank = 32,
            o_groups = 2,
            qk_rope_head_dim = 16,
            index_n_heads = 2,
            index_head_dim = 16,
            index_topk = 4096,
            sliding_window = 16,
            compress_rates = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
            num_hidden_layers = 3,
            layer_types = [
                "sliding_attention",
                "compressed_sparse_attention",
                "heavily_compressed_attention",
            ],
            mlp_layer_types = ["moe", "moe", "moe"],
        )
    else:
        common.update(num_hidden_layers = 2)
    for key, value in common.items():
        setattr(config, key, value)
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32).eval()


def _tiny_or_skip(model_type):
    try:
        return _tiny(model_type)
    except (KeyError, ValueError) as exc:
        pytest.skip(f"transformers {transformers.__version__} cannot build {model_type}: {exc}")


def _second_document_error(model):
    g = torch.Generator().manual_seed(1)
    a = torch.randint(3, 250, (1, 48), generator = g)
    b = torch.randint(3, 250, (1, 40), generator = g)
    with torch.no_grad():
        alone = model(input_ids = b, use_cache = False).logits
        position_ids = torch.cat([torch.arange(a.shape[1]), torch.arange(b.shape[1])])[None]
        packed = model(
            input_ids = torch.cat([a, b], 1), position_ids = position_ids, use_cache = False
        ).logits[:, a.shape[1] :]
    return ((packed - alone).norm() / alone.norm()).item()


class _StubTrainer:
    def __init__(
        self,
        model = None,
        args = None,
        **kwargs,
    ):
        self.model = model
        self.args = args


def _sft_args(**overrides):
    args = SimpleNamespace(
        packing = False,
        padding_free = None,
        max_length = 64,
        eval_packing = None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _build_sft(monkeypatch, model, args):
    for name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, name, lambda model, trainer: None)
    module = SimpleNamespace(SFTTrainer = type("SFTTrainer", (_StubTrainer,), {}))
    trainer_module._patch_sft_trainer_auto_packing(module)
    module.SFTTrainer(model = model, args = args)
    return args


def test_deepseek_v4_packed_row_leaks_and_dense_does_not():
    v4 = _tiny_or_skip("deepseek_v4")
    assert _second_document_error(v4) > 1e-2
    assert _second_document_error(_tiny("llama")) < 1e-5


def test_sft_keeps_padding_free_off_for_deepseek_v4(monkeypatch, capsys):
    v4 = _tiny_or_skip("deepseek_v4")
    args = _build_sft(monkeypatch, v4, _sft_args())
    assert args.padding_free is False and args.packing is False
    assert "mixes tokens across packed documents" in capsys.readouterr().out
    args = _build_sft(monkeypatch, v4, _sft_args(padding_free = True, eval_packing = True))
    assert args.padding_free is False and args.eval_packing is False
    args = _build_sft(monkeypatch, v4, _sft_args(packing = True))
    assert args.packing is False and args.padding_free is False


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
def test_sft_still_auto_enables_padding_free_for_dense(monkeypatch, capsys, model_type):
    model = _tiny_or_skip(model_type)
    args = _build_sft(monkeypatch, model, _sft_args())
    assert args.padding_free is True
    assert "mixes tokens across packed documents" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "name",
    [
        "DeepseekV4CSACompressor",
        "DeepseekV4HCACompressor",
        "ZayaCCAProjection",
        "RecurrentGemmaRglru",
    ],
)
def test_boundary_blocker_matches_modules_and_string_model_types(name):
    from unsloth.utils.packing import packing_boundary_blocker

    inner = type(name, (torch.nn.Module,), {})()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sequential(inner))
    assert packing_boundary_blocker(model) == name
    assert packing_boundary_blocker(torch.nn.Linear(2, 2)) is None
    assert packing_boundary_blocker("org/repo", ["deepseek_v4"]) == "deepseek_v4"
    assert packing_boundary_blocker("org/repo", ["deepseek_v3"]) is None
