# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""REAP expert pruning (unsloth/models/reap.py) against a brute-force oracle, on CPU tiny MoEs."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip(
        reason = "REAP prunes transformers 5 fused MoE experts (3D expert weights); 4.x keeps a ModuleList",
        allow_module_level = True,
    )

_MODULE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "reap.py"
_spec = importlib.util.spec_from_file_location("_unsloth_reap", _MODULE)
reap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reap)


@pytest.fixture(autouse = True)
def _one_thread():
    # Tiny CPU ops thrash on a loaded many-core host; one thread is faster and deterministic.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _qwen3_moe(
    num_experts = 16,
    top_k = 4,
    seed = 0,
):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size = 96,
        hidden_size = 32,
        intermediate_size = 48,
        moe_intermediate_size = 16,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        num_experts = num_experts,
        num_experts_per_tok = top_k,
        norm_topk_prob = True,
        mlp_only_layers = [0],  # a dense layer in front, like GLM / DeepSeek
        max_position_embeddings = 64,
    )
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM._from_config(
        config, dtype = torch.float64, experts_implementation = "eager"
    )
    return model.eval()


def _deepseek_v3(
    n_group = 4,
    topk_group = 2,
    seed = 0,
):
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

    config = DeepseekV3Config(
        vocab_size = 96,
        hidden_size = 32,
        intermediate_size = 48,
        moe_intermediate_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        n_routed_experts = 16,
        n_shared_experts = 1,
        num_experts_per_tok = 4,
        n_group = n_group,
        topk_group = topk_group,
        first_k_dense_replace = 0,
        q_lora_rank = 16,
        kv_lora_rank = 16,
        qk_rope_head_dim = 8,
        qk_nope_head_dim = 8,
        v_head_dim = 8,
        max_position_embeddings = 64,
    )
    torch.manual_seed(seed)
    model = DeepseekV3ForCausalLM._from_config(
        config, dtype = torch.float64, experts_implementation = "eager"
    )
    with torch.no_grad():
        for module in model.modules():
            bias = getattr(module, "e_score_correction_bias", None)
            if bias is not None:
                bias.copy_(torch.randn_like(bias) * 0.1)
    return model.eval()


def _samples(n = 7, seed = 1):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randint(1, 96, (int(torch.randint(3, 19, (1,), generator = g)),), generator = g).tolist()
        for _ in range(n)
    ]


def _oracle(model, samples, n_drop):
    """Sequential REAP written from the paper: per sample (no padding), per (token, slot) eager
    expert matmuls in float64, each layer pruned before the next one is scored."""
    model = copy.deepcopy(model)
    kept, scores = {}, {}
    for index, layer in enumerate(model.model.layers):
        block = layer.mlp
        if not hasattr(block, "experts"):
            continue
        experts = block.experts
        num_experts = experts.gate_up_proj.shape[0]
        total = torch.zeros(num_experts, dtype = torch.float64)
        count = torch.zeros(num_experts, dtype = torch.float64)
        seen = {}
        hook = experts.register_forward_hook(
            lambda m, args, out: seen.update(x = args[0], idx = args[1], w = args[2])
        )
        with torch.no_grad():
            for sample in samples:
                model.model(input_ids = torch.tensor([sample]), use_cache = False)
                for t in range(seen["x"].shape[0]):
                    for s in range(seen["idx"].shape[1]):
                        e = int(seen["idx"][t, s])
                        gate, up = torch.nn.functional.linear(
                            seen["x"][t], experts.gate_up_proj[e]
                        ).chunk(2)
                        f = torch.nn.functional.linear(
                            experts.act_fn(gate) * up, experts.down_proj[e]
                        )
                        total[e] += seen["w"][t, s].double() * f.double().norm()
                        count[e] += 1
        hook.remove()
        saliency = total / count.clamp(min = 1)
        drop = set(torch.topk(saliency, n_drop, largest = False).indices.tolist())
        keep = [i for i in range(num_experts) if i not in drop]
        name = f"model.layers.{index}.mlp"
        kept[name], scores[name] = keep, saliency
        idx = torch.tensor(keep)
        experts.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.data[idx])
        experts.down_proj = torch.nn.Parameter(experts.down_proj.data[idx])
        experts.num_experts = len(keep)
        block.gate.weight = torch.nn.Parameter(block.gate.weight.data[idx])
        block.gate.num_experts = len(keep)
    return kept, scores


def test_matches_sequential_oracle_with_padded_batches():
    model = _qwen3_moe()
    samples = _samples()
    kept, scores = _oracle(model, samples, n_drop = 4)
    grad_modes = []
    hook = model.model.layers[1].register_forward_hook(
        lambda *_: grad_modes.append(torch.is_grad_enabled())
    )
    # A small token budget forces several right-padded batches.
    reap.reap_prune(model, None, sparsity = 0.25, dataset = samples, max_batch_tokens = 40, device = "cpu")
    hook.remove()
    # No autograd graph may keep calibration activations alive, and weights stay trainable.
    assert grad_modes and not any(grad_modes)
    assert model.model.layers[1].mlp.experts.gate_up_proj.requires_grad
    assert model._unsloth_reap_report == kept
    for name, oracle in scores.items():
        torch.testing.assert_close(
            torch.tensor(model._unsloth_reap_saliency[name], dtype = torch.float64),
            oracle,
            rtol = 1e-9,
            atol = 1e-12,
        )
    assert model.config.num_experts == 12
    for layer in model.model.layers[1:]:
        assert layer.mlp.experts.gate_up_proj.shape[0] == 12
        assert layer.mlp.gate.weight.shape[0] == 12
        assert layer.mlp.experts.num_experts == 12


def test_batching_does_not_change_the_result():
    samples = _samples(n = 9, seed = 3)
    a, b = _qwen3_moe(seed = 5), _qwen3_moe(seed = 5)
    reap.reap_prune(a, None, sparsity = 0.5, dataset = samples, max_batch_tokens = 1, device = "cpu")
    reap.reap_prune(b, None, sparsity = 0.5, dataset = samples, max_batch_tokens = 10_000, device = "cpu")
    assert a._unsloth_reap_report == b._unsloth_reap_report
    for name in a._unsloth_reap_saliency:
        torch.testing.assert_close(
            torch.tensor(a._unsloth_reap_saliency[name]),
            torch.tensor(b._unsloth_reap_saliency[name]),
            rtol = 1e-9,
            atol = 1e-12,
        )


def test_pruned_model_saves_reloads_and_writes_report(tmp_path):
    from transformers import AutoModelForCausalLM

    model = _qwen3_moe()
    report = tmp_path / "sub" / "reap.json"
    reap.reap_prune(
        model, None, sparsity = 0.25, dataset = _samples(), device = "cpu", report_path = str(report)
    )
    assert json.loads(report.read_text()) == model._unsloth_reap_report
    model.save_pretrained(tmp_path / "pruned")
    reloaded = AutoModelForCausalLM.from_pretrained(
        tmp_path / "pruned", dtype = torch.float64, experts_implementation = "eager"
    ).eval()
    assert reloaded.config.num_experts == 12
    ids = torch.tensor([_samples(n = 1, seed = 9)[0]])
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids).logits, model(ids).logits)


def test_group_limited_router_drops_per_group():
    model = _deepseek_v3()
    biases = [layer.mlp.gate.e_score_correction_bias.clone() for layer in model.model.layers]
    reap.reap_prune(model, None, sparsity = 0.25, dataset = _samples(), device = "cpu")
    for index, (layer, bias) in enumerate(zip(model.model.layers, biases)):
        kept = model._unsloth_reap_report[f"model.layers.{index}.mlp"]
        # 16 experts in 4 groups of 4, one dropped per group.
        assert [sum(1 for e in kept if e // 4 == g) for g in range(4)] == [3, 3, 3, 3]
        torch.testing.assert_close(layer.mlp.gate.e_score_correction_bias, bias[kept])
        assert layer.mlp.experts.gate_up_proj.shape[0] == 12
    assert model.config.n_routed_experts == 12
    with torch.no_grad():
        model(torch.tensor([_samples(n = 1)[0]]))


def test_rejects_bad_requests():
    model = _qwen3_moe()
    samples = _samples()
    with pytest.raises(ValueError, match = "drops 0"):
        reap.reap_prune(model, None, sparsity = 0.01, dataset = samples, device = "cpu")
    with pytest.raises(ValueError, match = "too aggressive"):
        reap.reap_prune(model, None, sparsity = 0.9, dataset = samples, device = "cpu")
    with pytest.raises(ValueError, match = r"in \(0, 1\)"):
        reap.reap_prune(model, None, sparsity = 1.0, dataset = samples, device = "cpu")
    model.peft_config = {}
    with pytest.raises(RuntimeError, match = "get_peft_model"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = samples, device = "cpu")
    # Nothing was touched by the refused calls.
    del model.peft_config
    assert model.model.layers[1].mlp.experts.gate_up_proj.shape[0] == 16


def test_full_layer_rerun_matches_residual_shortcut(monkeypatch):
    samples = _samples(n = 5, seed = 4)
    a, b = _qwen3_moe(seed = 2), _qwen3_moe(seed = 2)
    reap.reap_prune(a, None, sparsity = 0.25, dataset = samples, device = "cpu")
    monkeypatch.setattr(reap, "_residual_split", lambda layer, block: None)
    reap.reap_prune(b, None, sparsity = 0.25, dataset = samples, device = "cpu")
    assert a._unsloth_reap_report == b._unsloth_reap_report
    ids = torch.tensor([samples[0]])
    with torch.no_grad():
        torch.testing.assert_close(a(ids).logits, b(ids).logits)


def test_save_directory_becomes_the_merge_base(tmp_path):
    model = _qwen3_moe()
    reap.reap_prune(
        model,
        None,
        sparsity = 0.25,
        dataset = _samples(),
        device = "cpu",
        save_directory = str(tmp_path / "pruned"),
    )
    # Later merged / GGUF saves of a LoRA on top read the base checkpoint from config._name_or_path.
    assert model.config._name_or_path == str((tmp_path / "pruned").resolve())
    assert model._unsloth_reap_saved == model.config._name_or_path == model.name_or_path
    saved = json.loads((tmp_path / "pruned" / "config.json").read_text())
    assert saved.get("num_experts", saved.get("num_local_experts")) == 12


@pytest.mark.parametrize(
    "build",
    [
        # Router weight in a child module (`gate.wg`).
        lambda T, base: T.HunYuanMoEV1Config(**base, num_experts = 8, moe_topk = 2),
        # [1, E] score bias in `gate.moe_statics`.
        lambda T, base: T.Ernie4_5_MoeConfig(
            **base, moe_num_experts = 8, moe_layer_start_index = 0, moe_k = 2
        ),
    ],
    ids = ["hunyuan_v1_moe", "ernie4_5_moe"],
)
def test_per_expert_router_tensors_in_child_modules_are_sliced(build, tmp_path):
    from transformers import AutoModelForCausalLM

    base = dict(
        vocab_size = 96,
        hidden_size = 32,
        intermediate_size = 48,
        moe_intermediate_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        head_dim = 8,
        max_position_embeddings = 64,
        num_experts_per_tok = 2,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(build(transformers, base), dtype = torch.float64).eval()
    reap.reap_prune(model, None, sparsity = 0.25, dataset = _samples(), device = "cpu")
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if ".mlp." in name and "shared" not in name:
            assert 8 not in tuple(tensor.shape), (name, tuple(tensor.shape))
    model.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype = torch.float64).eval()
    ids = torch.tensor([_samples(n = 1, seed = 9)[0]])
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids).logits, model(ids).logits)


def test_unsupported_weights_refuse_before_any_layer_is_pruned():
    model = _qwen3_moe()
    experts = model.model.layers[2].mlp.experts
    experts.down_proj = torch.nn.Parameter(
        torch.zeros(experts.down_proj.shape, dtype = torch.int8), requires_grad = False
    )
    with pytest.raises(RuntimeError, match = "16-bit or 32-bit"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = _samples(), device = "cpu")
    assert model.model.layers[1].mlp.experts.gate_up_proj.shape[0] == 16
