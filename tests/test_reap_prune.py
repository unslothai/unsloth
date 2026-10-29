# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""REAP expert pruning (unsloth/models/reap.py) against a brute-force oracle, on CPU tiny MoEs."""

import copy
import importlib.util
import json
import sys
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


@pytest.fixture(autouse = True)
def _float64_moe_backend(monkeypatch):
    # These models are float64; Unsloth's grouped_mm MoE backend (picked when a GPU is visible) has no
    # float64 kernel, so pin the looped backend. The backend choice is lru_cached per module copy.
    monkeypatch.setenv("UNSLOTH_MOE_BACKEND", "native_torch")
    caches = [
        getattr(vars(module).get("select_moe_backend"), "cache_clear", None)
        for module in list(sys.modules.values())
        if module is not None and hasattr(module, "__dict__")
    ]
    caches = [clear for clear in caches if callable(clear)]
    for clear in caches:
        clear()
    yield
    for clear in caches:
        clear()


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
    assert json.loads(report.read_text(encoding = "utf-8")) == model._unsloth_reap_report
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
    model.vllm_engine = object()
    with pytest.raises(RuntimeError, match = "vLLM engine"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = samples, device = "cpu")
    del model.vllm_engine
    object.__setattr__(model, "_unsloth_mtp_head", torch.nn.Linear(1, 1))
    with pytest.raises(RuntimeError, match = "MTP head"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = samples, device = "cpu")
    del model.__dict__["_unsloth_mtp_head"]
    for layout, match in (
        ({"enable_expert_parallel": True}, "expert parallel"),
        ({"tp_size": 2}, "tensor parallel"),
        ({"pp_size": 2}, "pipeline parallel"),
        ({"fsdp_size": 2}, "FSDP"),
        ({"enable_sequence_parallel": True}, "sequence parallel"),
    ):
        model.config.distributed_config = layout
        with pytest.raises(RuntimeError, match = match):
            reap.reap_prune(model, None, sparsity = 0.25, dataset = samples, device = "cpu")
    # Size 1 is not model parallel.
    model.config.distributed_config = {"tp_size": 1, "pp_size": 1}
    assert reap._model_parallel_layout(model) is None
    del model.config.distributed_config
    model.config.use_bidirectional_attention = "all"
    with pytest.raises(ValueError, match = "causal"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = samples, device = "cpu")
    model.config.use_bidirectional_attention = None
    model.model.layers[1].self_attn.is_causal = False
    with pytest.raises(ValueError, match = "causal"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = samples, device = "cpu")
    model.model.layers[1].self_attn.is_causal = True
    layer0 = model.model.layers[0]
    original = layer0.forward
    layer0.forward = lambda hidden_states, prev_topk_indices = None, **kwargs: original(
        hidden_states, **kwargs
    )  # GLM-5 / HY-V4 hand top-k indices from layer to layer
    with pytest.raises(ValueError, match = "extra state to the next layer"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = samples, device = "cpu")
    del layer0.forward
    peft = pytest.importorskip("peft")
    lora = peft.get_peft_model(model, peft.LoraConfig(r = 2, target_modules = ["q_proj"]))
    with pytest.raises(RuntimeError, match = "get_peft_model"):
        reap.reap_prune(lora, None, sparsity = 0.25, dataset = samples, device = "cpu")
    with pytest.raises(RuntimeError, match = "get_peft_model"):
        reap.reap_prune(lora.base_model.model, None, sparsity = 0.25, dataset = samples, device = "cpu")
    # Adapters that are not tuner layers (adaption prompt wraps attention in its own module).
    wrapper = type(
        "AdaptedAttention", (torch.nn.Module,), {"__module__": "peft.tuners.adaption_prompt.layer"}
    )
    lora.unload()
    model.model.layers[0].adapted = wrapper()
    with pytest.raises(RuntimeError, match = "get_peft_model"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = samples, device = "cpu")
    del model.model.layers[0].adapted
    # Custom LoRA layers registered from user code subclass PEFT's layer classes.
    custom = type("CustomLoraLayer", (wrapper,), {"__module__": "my_package.layers"})
    model.model.layers[0].adapted = custom()
    with pytest.raises(RuntimeError, match = "get_peft_model"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = samples, device = "cpu")
    del model.model.layers[0].adapted
    # Nothing was touched by the refused calls.
    assert model.model.layers[1].mlp.experts.gate_up_proj.shape[0] == 16


def test_merged_lora_with_leftover_peft_config_prunes():
    peft = pytest.importorskip("peft")
    samples = _samples(n = 4, seed = 5)
    lora = peft.get_peft_model(_qwen3_moe(), peft.LoraConfig(r = 2, target_modules = ["q_proj"]))
    merged = lora.merge_and_unload()
    merged.peft_config = lora.peft_config  # PEFT 0.18 keeps it after merge_and_unload
    reap.reap_prune(merged, None, sparsity = 0.25, dataset = samples, device = "cpu")
    assert merged.model.layers[1].mlp.experts.gate_up_proj.shape[0] == 12


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
    saved = json.loads((tmp_path / "pruned" / "config.json").read_text(encoding = "utf-8"))
    assert saved.get("num_experts", saved.get("num_local_experts")) == 12


@pytest.mark.parametrize(
    "build",
    [
        # Router weight in a child module (`gate.wg`); top_k per layer, as in Hunyuan-A13B.
        lambda T, base: T.HunYuanMoEV1Config(
            **{**base, "num_experts_per_tok": [2] * base["num_hidden_layers"]}, num_experts = 8
        ),
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
    model = AutoModelForCausalLM.from_config(
        build(transformers, base), dtype = torch.float64, experts_implementation = "eager"
    ).eval()
    reap.reap_prune(model, None, sparsity = 0.25, dataset = _samples(), device = "cpu")
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if ".mlp." in name and "shared" not in name:
            assert 8 not in tuple(tensor.shape), (name, tuple(tensor.shape))
    model.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(
        tmp_path, dtype = torch.float64, experts_implementation = "eager"
    ).eval()
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


def test_offloaded_activations_give_the_same_result():
    samples = _samples(n = 6, seed = 7)
    a, b = _qwen3_moe(seed = 3), _qwen3_moe(seed = 3)
    reap.reap_prune(a, None, sparsity = 0.25, dataset = samples, max_batch_tokens = 30, device = "cpu")
    reap.reap_prune(
        b,
        None,
        sparsity = 0.25,
        dataset = samples,
        max_batch_tokens = 30,
        device = "cpu",
        offload_hidden_states = True,
    )
    assert a._unsloth_reap_report == b._unsloth_reap_report
    assert a._unsloth_reap_saliency == b._unsloth_reap_saliency


def test_pretokenized_padding_is_dropped_via_attention_mask():
    samples = _samples(n = 5, seed = 11)
    width = max(len(x) for x in samples)
    padded = [
        {
            "input_ids": [0] * (width - len(x)) + x,
            "attention_mask": [0] * (width - len(x)) + [1] * len(x),
        }
        for x in samples
    ]
    a, b = _qwen3_moe(seed = 4), _qwen3_moe(seed = 4)
    reap.reap_prune(a, None, sparsity = 0.25, dataset = samples, device = "cpu")
    reap.reap_prune(b, None, sparsity = 0.25, dataset = padded, device = "cpu")
    assert a._unsloth_reap_saliency == b._unsloth_reap_saliency


def test_latent_moe_scores_expert_outputs_in_hidden_space():
    # Nemotron-H with moe_latent_size: experts run in a latent space and fc2_latent_proj maps
    # their summed output back, so ||f_j|| is taken after that projection.
    NemotronHConfig = getattr(transformers, "NemotronHConfig", None)
    if NemotronHConfig is None:
        pytest.skip("this transformers has no Nemotron-H")
    from transformers import NemotronHForCausalLM

    config = NemotronHConfig(
        vocab_size = 96,
        hidden_size = 32,
        layers_block_type = ["moe", "attention", "moe"],
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        intermediate_size = 48,
        n_routed_experts = 8,
        n_shared_experts = 1,
        moe_intermediate_size = 16,
        moe_shared_expert_intermediate_size = 16,
        moe_latent_size = 12,
        num_experts_per_tok = 2,
        n_group = 1,
        topk_group = 1,
        mamba_num_heads = 4,
        mamba_head_dim = 8,
        n_groups = 1,
    )
    torch.manual_seed(0)
    model = NemotronHForCausalLM._from_config(
        config, dtype = torch.float64, experts_implementation = "eager"
    ).eval()
    samples = _samples()
    block = model.model.layers[0].mixer
    # The experts call's own inputs (latent states, routing): the router's return tuple varies by version.
    calls = []
    hook = block.experts.register_forward_pre_hook(lambda module, args: calls.append(args[:3]))
    with torch.no_grad():
        for x in samples:
            model(torch.tensor([x]))
    hook.remove()
    total = torch.zeros(8, dtype = torch.float64)
    count = torch.zeros(8, dtype = torch.float64)
    with torch.no_grad():
        for latent, index, weights in calls:
            for t in range(latent.shape[0]):
                for s in range(index.shape[1]):
                    e = int(index[t, s])
                    f = block.experts(
                        latent[t : t + 1], index[t : t + 1, s : s + 1], torch.ones(1, 1)
                    )
                    total[e] += weights[t, s] * (f @ block.fc2_latent_proj.weight.T).norm()
                    count[e] += 1
    expected = total / count.clamp(min = 1)

    reap.reap_prune(model, None, sparsity = 0.5, dataset = samples, device = "cpu")
    torch.testing.assert_close(
        torch.tensor(model._unsloth_reap_saliency["model.layers.0.mixer"], dtype = torch.float64),
        expected,
        rtol = 1e-9,
        atol = 1e-12,
    )
    assert model.model.layers[0].mixer.experts.up_proj.shape[0] == 4


def test_refuses_layers_with_several_moe_blocks_and_keeps_training_modes():
    model = _qwen3_moe()
    samples = _samples()
    layer = model.model.layers[1]
    layer.vision_mlp = copy.deepcopy(layer.mlp)  # e.g. Ernie 4.5 VL's text_moe + vision_moe
    with pytest.raises(ValueError, match = "one MoE block per decoder layer"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = samples, device = "cpu")
    assert layer.mlp.experts.gate_up_proj.shape[0] == 16
    del layer.vision_mlp

    model.train()
    model.model.layers[0].eval()  # e.g. a frozen tower kept in eval mode
    reap.reap_prune(model, None, sparsity = 0.25, dataset = samples, device = "cpu")
    assert model.training and model.model.layers[1].training
    assert not model.model.layers[0].training


def test_chunked_scoring_matches_one_pass(monkeypatch):
    samples = _samples(n = 6, seed = 5)
    a, b = _qwen3_moe(seed = 6), _qwen3_moe(seed = 6)
    reap.reap_prune(a, None, sparsity = 0.25, dataset = samples, device = "cpu")
    monkeypatch.setattr(reap, "_TAP_CHUNK_BYTES", 1)  # one token per experts call
    reap.reap_prune(b, None, sparsity = 0.25, dataset = samples, device = "cpu")
    assert a._unsloth_reap_report == b._unsloth_reap_report
    for name in a._unsloth_reap_saliency:
        torch.testing.assert_close(
            torch.tensor(a._unsloth_reap_saliency[name], dtype = torch.float64),
            torch.tensor(b._unsloth_reap_saliency[name], dtype = torch.float64),
            rtol = 1e-9,
            atol = 1e-12,
        )


def _saliency_rank(rank, init_file, shards, out):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = 2)
    try:
        saliency = reap._Saliency(4)
        ids, weights, norms = shards[rank]
        if ids is not None:
            saliency.add(ids, weights, norms)
        saliency.all_reduce()
        out.put((rank, saliency.total.tolist(), saliency.count.tolist()))
    finally:
        dist.destroy_process_group()


def test_saliency_is_summed_across_ranks(tmp_path):
    # Two gloo ranks with different calibration shards (rank 1 routed nothing) must both end with
    # the one-process sums, so every rank of a DDP job prunes the same experts.
    import multiprocessing

    ids = torch.tensor([0, 1, 1, 3])
    weights = torch.tensor([0.5, 1.0, 0.25, 2.0])
    norms = torch.tensor([2.0, 1.0, 4.0, 0.5])
    one = reap._Saliency(4)
    one.add(ids, weights, norms)
    shards = [(ids, weights, norms), (None, None, None)]
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    procs = [
        ctx.Process(target = _saliency_rank, args = (r, str(tmp_path / "pg"), shards, out))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    results = sorted(out.get(timeout = 120) for _ in procs)
    for p in procs:
        p.join(timeout = 60)
        assert p.exitcode == 0
    for _, total, count in results:
        assert total == one.total.tolist() and count == one.count.tolist()
    # Single process (no process group): unchanged.
    before = one.total.clone()
    one.all_reduce()
    assert torch.equal(one.total, before)


def test_saliency_reduces_on_the_accelerator_for_xccl_and_hccl(monkeypatch):
    # Accelerator-only backends cannot reduce a CPU tensor, and their devices have no float64.
    dist = torch.distributed
    seen = []
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 2)
    monkeypatch.setattr(dist, "get_backend", lambda *a, **k: "xccl")
    monkeypatch.setattr(
        dist, "all_reduce", lambda t, *a, **k: seen.append((t.device.type, t.dtype))
    )
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda *a, **k: torch.device("meta")
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0, raising = False)
    saliency = reap._Saliency(4)
    saliency.all_reduce()
    assert seen == [("meta", torch.float32)]


def test_dataset_dict_uses_a_split_not_its_names():
    # load_dataset(...) without split= returns a DatasetDict; its iteration yields split names.
    datasets = pytest.importorskip("datasets")
    samples = _samples(n = 6, seed = 9)
    split = datasets.Dataset.from_list([{"input_ids": ids} for ids in samples])
    model = _qwen3_moe()
    a = copy.deepcopy(model)
    b = copy.deepcopy(model)
    reap.reap_prune(a, None, sparsity = 0.25, dataset = samples, device = "cpu")
    reap.reap_prune(b, None, sparsity = 0.25, dataset = datasets.DatasetDict(train = split), device = "cpu")
    assert a._unsloth_reap_report == b._unsloth_reap_report
    two = datasets.DatasetDict(validation = split, test = split)
    with pytest.raises(ValueError, match = "splits"):
        reap.reap_prune(copy.deepcopy(model), None, sparsity = 0.25, dataset = two, device = "cpu")


def _prune_rank(
    rank,
    init_file,
    save_dir,
    out,
    fail_save = False,
    bad_shard = False,
    report = False,
):
    import json
    import os
    import torch.distributed as dist

    os.environ["UNSLOTH_MOE_BACKEND"] = "native_torch"
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = 2)
    try:
        model = _qwen3_moe(seed = 0)
        original = model.save_pretrained

        def save(*args, **kwargs):
            if fail_save:
                raise OSError("No space left on device")
            with open(os.path.join(os.path.dirname(save_dir), f"saved_by_{rank}"), "w") as f:
                f.write("1")
            return original(*args, **kwargs)

        model.save_pretrained = save
        report_path = None
        if report:
            report_path = os.path.join(os.path.dirname(save_dir), "report.json")
            original_dump = json.dump

            def dump(obj, f, *args, **kwargs):
                if getattr(f, "name", "") == report_path:
                    if report == "fail":
                        raise OSError("Read-only file system")
                    with open(
                        os.path.join(os.path.dirname(save_dir), f"report_by_{rank}"), "w"
                    ) as m:
                        m.write("1")
                return original_dump(obj, f, *args, **kwargs)

            json.dump = dump
        if bad_shard == "move" and rank == 1:

            def fail_move(*args, **kwargs):
                raise RuntimeError("CUDA out of memory while streaming the layer in")

            reap._stream_in = fail_move
        if bad_shard == "slice" and rank == 1:

            def fail_slice(*args, **kwargs):
                raise RuntimeError("out of memory while slicing experts")

            reap._prune_block = fail_slice
        if bad_shard == "post" and rank == 1:
            # Rank 1 fails in the post-prune pass (an OOM on a larger shard, say).
            original_prune = reap._prune_block

            def prune_then_fail(block, *args, **kwargs):
                original_prune(block, *args, **kwargs)

                def boom(*_):
                    raise RuntimeError("out of memory in the post-prune pass")

                block.register_forward_hook(boom)

            reap._prune_block = prune_then_fail
        # Each rank calibrates on its own shard, as a DDP job would.
        try:
            reap.reap_prune(
                model,
                None,
                sparsity = 0.25,
                dataset = (
                    ([[10**6] * 6] if bad_shard == "out_of_range" else [])
                    if bad_shard in (True, "out_of_range") and rank == 1
                    else _samples(n = 5, seed = 20 + rank)
                ),
                device = "cpu",
                save_directory = save_dir,
                report_path = report_path,
            )
        except Exception as e:
            out.put((rank, "error", str(e)))
            return
        out.put((rank, model._unsloth_reap_report, model.config._name_or_path))
    finally:
        dist.destroy_process_group()


def test_ddp_ranks_keep_the_same_experts_and_one_writes(tmp_path):
    import multiprocessing

    save_dir = str(tmp_path / "pruned")
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    procs = [
        ctx.Process(target = _prune_rank, args = (r, str(tmp_path / "pg"), save_dir, out))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    results = sorted(out.get(timeout = 300) for _ in procs)
    for p in procs:
        p.join(timeout = 60)
        assert p.exitcode == 0
    (_, report0, path0), (_, report1, path1) = results
    assert report0 == report1 and path0 == path1 == save_dir
    assert (tmp_path / "saved_by_0").exists() and not (tmp_path / "saved_by_1").exists()
    # The one shared-data answer: both shards together, in one process.
    one = _qwen3_moe(seed = 0)
    reap.reap_prune(
        one,
        None,
        sparsity = 0.25,
        dataset = _samples(n = 5, seed = 20) + _samples(n = 5, seed = 21),
        device = "cpu",
    )
    assert one._unsloth_reap_report == report0


def test_a_failed_rank_zero_save_stops_every_rank(tmp_path):
    # Rank 0's write fails (disk full): every rank must raise, none may wait forever.
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    args = (str(tmp_path / "pg"), str(tmp_path / "pruned"), out, True)
    procs = [ctx.Process(target = _prune_rank, args = (r, *args)) for r in range(2)]
    for p in procs:
        p.start()
    results = sorted(out.get(timeout = 300) for _ in procs)
    for p in procs:
        p.join(timeout = 60)
        assert p.exitcode == 0
    assert results[0][1] == "error" and "No space left" in results[0][2]
    assert results[1][1] == "error" and "another rank failed to save" in results[1][2]


def _run_ranks(tmp_path, **kwargs):
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    args = (str(tmp_path / "pg"), str(tmp_path / "pruned"), out)
    procs = [ctx.Process(target = _prune_rank, args = (r, *args), kwargs = kwargs) for r in range(2)]
    for p in procs:
        p.start()
    results = sorted(out.get(timeout = 300) for _ in procs)
    for p in procs:
        p.join(timeout = 60)
        assert p.exitcode == 0
    return results


def test_ddp_report_is_written_by_rank_zero_only(tmp_path):
    (_, report0, _), (_, report1, _) = _run_ranks(tmp_path, report = True)
    assert report0 == report1
    assert (tmp_path / "report_by_0").exists() and not (tmp_path / "report_by_1").exists()
    assert json.loads((tmp_path / "report.json").read_text(encoding = "utf-8")) == report0


def test_a_failed_rank_zero_report_write_stops_every_rank(tmp_path):
    results = _run_ranks(tmp_path, report = "fail")
    assert results[0][1] == "error" and "Read-only file system" in results[0][2]
    assert (
        results[1][1] == "error" and "another rank failed to write the REAP report" in results[1][2]
    )


def test_one_bad_calibration_shard_stops_every_rank(tmp_path):
    # Rank 1's shard is empty: it must not leave rank 0 waiting in the first per-layer collective.
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    args = (str(tmp_path / "pg"), str(tmp_path / "pruned"), out, False, True)
    procs = [ctx.Process(target = _prune_rank, args = (r, *args)) for r in range(2)]
    for p in procs:
        p.start()
    results = sorted(out.get(timeout = 300) for _ in procs)
    for p in procs:
        p.join(timeout = 60)
        assert p.exitcode == 0
    assert results[0][1] == "error" and "another rank failed to prepare" in results[0][2]
    assert results[1][1] == "error"


def test_group_metadata_must_be_complete():
    model = _qwen3_moe()
    model.config.n_group = 2
    with pytest.raises(ValueError, match = "no topk_group"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = _samples(), device = "cpu")
    # n_group 1 (or a lone topk_group) routes ungrouped and prunes globally.
    model.config.n_group = 1
    reap.reap_prune(model, None, sparsity = 0.25, dataset = _samples(), device = "cpu")
    assert model._unsloth_reap_report


def test_one_rank_failing_at_capture_stops_every_rank(tmp_path):
    # Rank 1's shard has an out-of-range token id: its embedding lookup fails while capturing.
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    args = (str(tmp_path / "pg"), str(tmp_path / "pruned"), out, False, "out_of_range")
    procs = [ctx.Process(target = _prune_rank, args = (r, *args)) for r in range(2)]
    for p in procs:
        p.start()
    results = sorted(out.get(timeout = 300) for _ in procs)
    for p in procs:
        p.join(timeout = 60)
        assert p.exitcode == 0
    assert results[0][1] == "error" and "another rank failed to capture" in results[0][2]
    assert results[1][1] == "error"


def test_metadata_free_streaming_token_rows_need_no_tokenizer():
    datasets = pytest.importorskip("datasets")
    samples = _samples(n = 6, seed = 13)

    def rows():
        for ids in samples:
            yield {"input_ids": ids}

    stream = datasets.IterableDataset.from_generator(rows)
    model = _qwen3_moe()
    a, b = copy.deepcopy(model), copy.deepcopy(model)
    reap.reap_prune(a, None, sparsity = 0.25, dataset = samples, device = "cpu")
    reap.reap_prune(b, None, sparsity = 0.25, dataset = stream, device = "cpu")
    assert a._unsloth_reap_report == b._unsloth_reap_report
    # Text rows still need one, now reported when the first text row is read.
    with pytest.raises(ValueError, match = "needs a tokenizer"):
        reap.reap_prune(
            copy.deepcopy(model), None, sparsity = 0.25, dataset = [{"text": "hi"}], device = "cpu"
        )


def test_failure_after_pruning_a_layer_marks_the_model_unsaved(monkeypatch):
    model = _qwen3_moe()
    model._unsloth_reap_saved = "/stale/earlier/save"
    original = reap._prune_block
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("out of memory")
        return original(*args, **kwargs)

    monkeypatch.setattr(reap, "_prune_block", flaky)
    with pytest.raises(RuntimeError, match = "out of memory"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = _samples(), device = "cpu")
    # The completed layer keeps its experts; the one cut short is recorded too, both unsaved.
    done, cut = list(model._unsloth_reap_report.values())
    assert done and cut == [] and model._unsloth_reap_saved is None


def test_one_rank_failing_after_pruning_stops_every_rank(tmp_path):
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    args = (str(tmp_path / "pg"), str(tmp_path / "pruned"), out, False, "post")
    procs = [ctx.Process(target = _prune_rank, args = (r, *args)) for r in range(2)]
    for p in procs:
        p.start()
    results = sorted(out.get(timeout = 300) for _ in procs)
    for p in procs:
        p.join(timeout = 60)
        assert p.exitcode == 0
    assert results[0][1] == "error" and "another rank failed to run decoder layer" in results[0][2]
    assert results[1][1] == "error" and "post-prune" in results[1][2]


def test_failure_inside_the_first_slice_still_marks_the_model(monkeypatch):
    model = _qwen3_moe()
    model._unsloth_reap_saved = "/stale/earlier/save"

    def half_sliced(block, experts, router, retained, num_experts):
        experts.gate_up_proj.data = experts.gate_up_proj.data[retained]
        raise RuntimeError("failed after slicing one tensor")

    monkeypatch.setattr(reap, "_prune_block", half_sliced)
    with pytest.raises(RuntimeError, match = "one tensor"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = _samples(), device = "cpu")
    assert model._unsloth_reap_report and model._unsloth_reap_saved is None


def test_one_rank_failing_to_slice_stops_every_rank(tmp_path):
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    args = (str(tmp_path / "pg"), str(tmp_path / "pruned"), out, False, "slice")
    procs = [ctx.Process(target = _prune_rank, args = (r, *args)) for r in range(2)]
    for p in procs:
        p.start()
    results = sorted(out.get(timeout = 300) for _ in procs)
    for p in procs:
        p.join(timeout = 60)
        assert p.exitcode == 0
    assert results[0][1] == "error" and "another rank failed to run decoder layer" in results[0][2]
    assert results[1][1] == "error" and "slicing experts" in results[1][2]


def test_several_moe_decoder_stacks_are_refused():
    class TwoStacks(torch.nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.config = a.config
            self.text = a.model
            self.other = b.model

    a, b = _qwen3_moe(seed = 0), _qwen3_moe(seed = 1)
    model = TwoStacks(a, b)
    experts = next(m for n, m in model.other.named_modules() if n.endswith("mlp.experts"))
    before = experts.gate_up_proj.shape
    with pytest.raises(ValueError, match = "several decoder stacks"):
        reap.reap_prune(model, None, sparsity = 0.25, dataset = _samples(), device = "cpu")
    # Refused before anything changed.
    assert experts.gate_up_proj.shape == before
    assert model.config.num_experts == before[0]


def test_one_rank_failing_to_stream_a_layer_in_stops_every_rank(tmp_path):
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    args = (str(tmp_path / "pg"), str(tmp_path / "pruned"), out, False, "move")
    procs = [ctx.Process(target = _prune_rank, args = (r, *args)) for r in range(2)]
    for p in procs:
        p.start()
    results = sorted(out.get(timeout = 300) for _ in procs)
    for p in procs:
        p.join(timeout = 60)
        assert p.exitcode == 0
    assert (
        results[0][1] == "error" and "another rank failed to run decoder layer 1" in results[0][2]
    )
    assert results[1][1] == "error" and "streaming the layer in" in results[1][2]


def test_shared_layer_state_follows_each_layer_without_offloading(monkeypatch):
    # Gemma 4 passes one mutable `shared_kv_states` mapping to every layer. Under a device_map the
    # layers sit on different GPUs, so it must be moved to each layer's device even when the
    # activations stay put (no offload store).
    from collections import UserDict

    model = _qwen3_moe(seed = 0)
    shared = UserDict({"kv": torch.zeros(1)})
    capture = reap._capture_inputs

    def capture_with_shared(*args, **kwargs):
        h, per_layer = capture(*args, **kwargs)
        return h, [(a, {**kw, "shared_kv_states": shared}) for a, kw in per_layer]

    moved = []
    to = reap._to

    def recording_to(obj, device, *rest):
        if isinstance(obj, dict) and obj.keys() == {"kv"}:
            moved.append(torch.device(device))
        return to(obj, device, *rest)

    monkeypatch.setattr(reap, "_capture_inputs", capture_with_shared)
    monkeypatch.setattr(reap, "_to", recording_to)
    reap.reap_prune(
        model,
        None,
        sparsity = 0.5,
        dataset = _samples(),
        device = "cpu",
        offload_hidden_states = False,
    )
    assert moved and set(moved) == {torch.device("cpu")}


def test_mistral_common_backend_is_not_unwrapped():
    # MistralCommonBackend is an HF tokenizer whose `.tokenizer` is the raw mistral_common one,
    # which has neither __call__ nor apply_chat_template. Only processors are unwrapped.
    from transformers import ProcessorMixin

    class Raw:
        pass

    class Backend:
        tokenizer = Raw()
        chat_template = "{{ messages[0]['content'] }}"
        pad_token_id = 0

        def __call__(
            self,
            text,
            truncation = True,
            max_length = None,
        ):
            return {"input_ids": [5, 6, 7][:max_length]}

        def apply_chat_template(
            self,
            messages,
            tokenize = False,
            **kwargs,
        ):
            return messages[0]["content"]

    backend = Backend()
    assert reap._text_tokenizer(backend) is backend
    ids = reap._calibration_ids([{"messages": [{"role": "user", "content": "hi"}]}], backend, 4, 8)
    assert ids == [[5, 6, 7]]
    processor = ProcessorMixin.__new__(ProcessorMixin)
    processor.tokenizer = backend
    assert reap._text_tokenizer(processor) is backend


def test_saved_pruned_checkpoint_carries_its_provenance(tmp_path):
    import json

    model = _qwen3_moe(seed = 0)
    source = model.config._name_or_path
    reap.reap_prune(
        model, None, sparsity = 0.5, dataset = _samples(), device = "cpu", save_directory = str(tmp_path)
    )
    data = json.loads((tmp_path / reap.REAP_PROVENANCE_FILE).read_text(encoding = "utf-8"))
    assert data["source_model"] == (source or None)
    assert data["kept_experts"] == {k: list(v) for k, v in model._unsloth_reap_report.items()}


def test_provenance_names_a_local_source_without_its_path(tmp_path):
    import json

    model = _qwen3_moe(seed = 0)
    local = tmp_path / "home" / "me" / "my-moe"
    local.mkdir(parents = True)
    model.config._name_or_path = str(local)
    out = tmp_path / "pruned"
    reap.reap_prune(
        model, None, sparsity = 0.5, dataset = _samples(), device = "cpu", save_directory = str(out)
    )
    data = json.loads((out / reap.REAP_PROVENANCE_FILE).read_text(encoding = "utf-8"))
    assert data["source_model"] == "my-moe"
