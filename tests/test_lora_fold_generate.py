# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""16-bit LoRA generate through an exact in-place fold (unsloth/models/_lora_fold_generate.py).

CPU: the zoo's Triton kernel needs CUDA, so a pure-torch fold with the same contract (in place, fix-up
store of the elements whose round trip is inexact, checksum on unfold) stands in for it. The kernel itself
is covered by the zoo's self-check and the GPU benchmark."""

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path

import time

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "unsloth" / "models" / "_lora_fold_generate.py"

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402

MODELS = [
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "trl-internal-testing/tiny-Qwen3ForCausalLM",
]


def _load_module():
    # By path: `import unsloth` needs an accelerator, this module needs only torch (+ peft lazily).
    spec = importlib.util.spec_from_file_location(
        "_unsloth_lora_fold_generate_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fold = _load_module()


class _RefTarget:
    def __init__(self, w_view, a, b, scale):
        assert w_view.dim() == 3 and w_view.dtype in (torch.bfloat16, torch.float16)
        self.w, self.a, self.b, self.scale = w_view, a, b, float(scale)
        self.folded = False

    def delta(self):
        return (self.b[0].double() @ self.a[0].double()).float() * self.scale


class _RefApi:
    """Same contract as unsloth_zoo.merged_rollout_exact, in torch."""

    def __init__(self, available = True):
        self.available = available
        self.folds = self.unfolds = 0
        self.ExactFoldTarget = _RefTarget

    def exact_merge_available(
        self,
        device,
        dtype = torch.bfloat16,
    ):
        return self.available

    def exact_fold(
        self,
        targets,
        stream = None,
    ):
        self.folds += 1
        bad_total = 0
        for t in targets:
            w = t.w[0]
            d = t.delta()
            new = (w.float() + d).to(w.dtype)
            back = (new.float() - d).to(w.dtype)
            bad = back.view(torch.int16) != w.view(torch.int16)
            t.bad, t.val = bad, w[bad].clone()
            # Host fix-up store as the zoo keeps it: int32 flat indices + original 16-bit values.
            t.host_idx = bad.flatten().nonzero().flatten().to(torch.int32)
            t.host_val = t.val.cpu()
            t.checksum = w.view(torch.int16).long().sum()
            w.copy_(new)
            t.folded = True
            bad_total += int(bad.sum())
        return {"bad": bad_total, "elements": sum(t.w.numel() for t in targets)}

    def exact_unfold(
        self,
        targets,
        stream = None,
    ):
        self.unfolds += 1
        targets = [t for t in targets if t.folded]
        wrong = 0
        for t in targets:
            w = t.w[0]
            w.copy_((w.float() - t.delta()).to(w.dtype))
            w[t.bad] = t.val
            wrong += int(w.view(torch.int16).long().sum() != t.checksum)
            t.folded = False
        if wrong:
            raise RuntimeError(f"checksum mismatch on {wrong} weights")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for name in (
        "UNSLOTH_LORA_FOLD_GENERATE",
        "UNSLOTH_LORA_FOLD_GENERATE_MIN_TOKENS",
        "ACCELERATE_USE_FSDP",
        "FSDP_VERSION",
    ):
        monkeypatch.delenv(name, raising = False)
    # Tiny generates: test the mechanism at 16 tokens; the default (128) is checked on its own.
    monkeypatch.setenv("UNSLOTH_LORA_FOLD_GENERATE_MIN_TOKENS", "16")
    api = _RefApi()
    monkeypatch.setattr(fold, "_exact_api", lambda: api)
    monkeypatch.setattr(fold, "_POISONED", None)
    fold.reset_stats()
    yield api


@pytest.fixture(scope = "module", params = MODELS)
def model_id(request):
    return request.param


def _make(
    model_id,
    dtype = torch.bfloat16,
    seed = 0,
    **lora,
):
    torch.manual_seed(seed)
    base = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = dtype)
    config = dict(r = 8, lora_alpha = 16, target_modules = "all-linear", lora_dropout = 0.0)
    config.update(lora)
    model = get_peft_model(base, LoraConfig(**config))
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator = g) * 0.05)
    model.eval()
    return model


def _bits(model):
    out = {}
    for name, p in model.named_parameters():
        out[name] = (
            p.detach().view(torch.int16).clone() if p.element_size() == 2 else p.detach().clone()
        )
    return out


def _same_bits(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _merged_state(model):
    return {
        name: (id(m.merged_adapters), list(m.merged_adapters))
        for name, m in model.named_modules()
        if hasattr(m, "merged_adapters")
    }


def _inputs(bsz = 2, length = 5):
    g = torch.Generator().manual_seed(7)
    return torch.randint(3, 100, (bsz, length), generator = g)


def _generate(
    model,
    new_tokens = 20,
    **kwargs,
):
    kw = dict(
        input_ids = _inputs(), max_new_tokens = new_tokens, min_new_tokens = new_tokens, do_sample = False
    )
    kw.update(kwargs)
    with torch.inference_mode(), fold.lora_fold_generate(model, (), kw) as engaged:
        out = model.generate(**kw)
    return engaged, out


def _count_lora_calls(model):
    calls = [0]

    def hook(*_):
        calls[0] += 1

    handles = [
        m.register_forward_hook(hook)
        for n, m in model.named_modules()
        if n.endswith("lora_A.default")
    ]
    return calls, handles


def test_loads_this_tree():
    assert Path(fold.__file__).resolve() == MODULE_PATH
    assert str(MODULE_PATH).startswith(str(ROOT))


def test_requested_new_tokens():
    model = types.SimpleNamespace(generation_config = transformers.GenerationConfig(max_length = 30))
    ids = torch.zeros(1, 10, dtype = torch.long)
    assert fold.requested_new_tokens(model, (), {"input_ids": ids, "max_new_tokens": 5}) == 5
    assert fold.requested_new_tokens(model, (ids,), {}) == 20
    assert fold.requested_new_tokens(model, (), {"input_ids": ids, "max_length": 12}) == 2
    assert fold.requested_new_tokens(model, (), {"inputs": ids, "max_length": 12}) == 2
    config = transformers.GenerationConfig(max_new_tokens = 64)
    assert (
        fold.requested_new_tokens(model, (), {"input_ids": ids, "generation_config": config}) == 64
    )


def test_fold_engages_skips_lora_and_restores_bits(model_id, _clean):
    model = _make(model_id)
    before, merged = _bits(model), _merged_state(model)
    versions = {n: p._version for n, p in model.named_parameters()}
    calls, handles = _count_lora_calls(model)
    engaged, _ = _generate(model)
    for h in handles:
        h.remove()
    assert engaged is True
    assert calls[0] == 0, "LoRA ran on top of folded weights"
    assert _same_bits(before, _bits(model))
    assert _merged_state(model) == merged
    assert {n: p._version for n, p in model.named_parameters()} == versions
    s = fold.stats()
    assert s["engaged"] == 1 and s["restored"] == 1 and s["refused"] == {}
    assert s["last"]["targets"] > 0 and s["last"]["elements"] > 0
    # 4 bytes of index + 2 of value per fix-up element; CPU tensors here are not pinned.
    assert s["last"]["host_store_bytes"] == 6 * s["last"]["bad"]
    assert s["last"]["host_store_pinned"] in (False, None)
    assert _clean.folds == 1 and _clean.unfolds == 1
    # The unfolded path does run LoRA: the counter above can fail.
    calls, handles = _count_lora_calls(model)
    with torch.inference_mode():
        model(_inputs())
    for h in handles:
        h.remove()
    assert calls[0] > 0


def test_folded_logits_match_unfolded(model_id, monkeypatch):
    model = _make(model_id)
    oracle = _make(model_id, dtype = torch.float32)
    ids = _inputs()
    new = 20
    _, folded = _generate(model, new, output_logits = True, return_dict_in_generate = True)
    assert fold.stats()["engaged"] == 1
    monkeypatch.setenv("UNSLOTH_LORA_FOLD_GENERATE", "0")
    _, plain = _generate(model, new, output_logits = True, return_dict_in_generate = True)
    assert fold.stats()["engaged"] == 1 and fold.stats()["refused"] == {"kill_switch": 1}
    monkeypatch.delenv("UNSLOTH_LORA_FOLD_GENERATE")
    # Teacher-force the unfolded tokens through every arm; fp32 LoRA (same seed) is the oracle.
    seq = plain.sequences
    with torch.inference_mode():
        ref = oracle(seq).logits[:, ids.shape[1] - 1 : -1].float()
        unfolded = model(seq).logits[:, ids.shape[1] - 1 : -1].float()
        with fold.lora_fold_generate(model, (), {"max_new_tokens": new}) as engaged:
            assert engaged
            folded_tf = model(seq).logits[:, ids.shape[1] - 1 : -1].float()
    err_unfolded = (unfolded - ref).abs().max().item()
    err_folded = (folded_tf - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err_folded <= 2 * err_unfolded + 1e-2 * scale, (err_folded, err_unfolded, scale)
    # Generated logits step by step agree with the teacher-forced ones up to bf16 noise.
    gen = torch.stack(folded.logits, 1).float()
    agree = (folded.sequences == seq).all(1)
    if agree.any():
        assert (gen[agree] - folded_tf[agree]).abs().max().item() <= 0.05 * scale + 1e-2


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_restored_when_generate_raises(error, _clean):
    model = _make(MODELS[0])
    before, merged = _bits(model), _merged_state(model)
    layer = model.base_model.model.model.layers[0]
    seen = [0]

    def boom(*_):
        seen[0] += 1
        if seen[0] == 5:
            raise error("injected mid-generate")

    handle = layer.register_forward_pre_hook(boom)
    with pytest.raises(error):
        _generate(model, 20)
    handle.remove()
    assert seen[0] == 5
    assert _same_bits(before, _bits(model))
    assert _merged_state(model) == merged
    s = fold.stats()
    assert s["engaged"] == 1 and s["restored"] == 1
    assert _clean.unfolds == 1
    assert fold._ACTIVE == 0
    # Usable again afterwards.
    assert _generate(model, 16)[0] is True


@pytest.mark.parametrize("error", [MemoryError, KeyboardInterrupt])
def test_fold_failure_part_way_is_undone(error, _clean, monkeypatch):
    model = _make(MODELS[0])
    before, merged = _bits(model), _merged_state(model)
    real = _clean.exact_fold

    def half_then_fail(targets, stream = None):
        real(targets[: len(targets) // 2])
        raise error("pinned buffer")

    monkeypatch.setattr(_clean, "exact_fold", half_then_fail)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            _generate(model, 20)
    else:
        # A failed fold falls back to the normal LoRA path.
        engaged, out = _generate(model, 20)
        assert engaged is False and out.shape[1] > 20
        assert fold.stats()["refused"] == {"fold_failed": 1}
    assert _clean.unfolds == 1
    assert _same_bits(before, _bits(model)) and _merged_state(model) == merged
    assert fold.stats()["engaged"] == 0 and fold._ACTIVE == 0 and fold._POISONED is None


def test_interrupt_after_fold_restores(monkeypatch, _clean):
    model = _make(MODELS[0])
    before = _bits(model)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(fold, "_record_fold", interrupted)
    with pytest.raises(KeyboardInterrupt):
        with torch.inference_mode(), fold.lora_fold_generate(model, (), {"max_new_tokens": 32}):
            pass
    assert _clean.folds == 1 and _clean.unfolds == 1
    assert _same_bits(before, _bits(model))
    assert all(not m.merged_adapters for m in model.modules() if hasattr(m, "merged_adapters"))


def test_checksum_mismatch_raises_and_poisons():
    model = _make(MODELS[0])
    W = model.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight
    with pytest.raises(RuntimeError, match = "Reload the model"):
        with (
            torch.inference_mode(),
            fold.lora_fold_generate(model, (), {"max_new_tokens": 32}) as engaged,
        ):
            assert engaged
            W.data[3].add_(1.0)
    assert fold._POISONED is not None
    assert all(not m.merged_adapters for m in model.modules() if hasattr(m, "merged_adapters"))
    # This model's weights are unknown: every later generate raises, short ones too.
    for n in (32, 1):
        with pytest.raises(RuntimeError, match = "Reload the model"):
            _generate(model, n)
    # Other models generate normally but never fold again in this process.
    assert _generate(_make(MODELS[0]), 32)[0] is False
    assert fold.stats()["refused"] == {"poisoned": 1}


def test_short_calls_and_threshold(monkeypatch):
    model = _make(MODELS[0])
    assert _generate(model, 4)[0] is False
    assert _generate(model, 15)[0] is False
    assert _generate(model, 16)[0] is True
    assert fold.stats()["refused"] == {"short": 2}
    monkeypatch.setenv("UNSLOTH_LORA_FOLD_GENERATE_MIN_TOKENS", "2")
    assert _generate(model, 4)[0] is True
    assert fold.stats()["engaged"] == 2
    monkeypatch.delenv("UNSLOTH_LORA_FOLD_GENERATE_MIN_TOKENS")
    assert fold._min_tokens() == 128
    assert fold.requested_new_tokens(model, (), {"max_new_tokens": 127}) < fold._min_tokens()
    monkeypatch.setenv("UNSLOTH_LORA_FOLD_GENERATE_MIN_TOKENS", "bad")
    assert fold._min_tokens() == 128
    # Decode CUDA graphs already remove most of the per-token LoRA cost: folding pays back later.
    monkeypatch.delenv("UNSLOTH_LORA_FOLD_GENERATE_MIN_TOKENS")
    monkeypatch.setattr(fold, "_graphs_enabled", lambda: True)
    assert fold._min_tokens() == 512
    assert _generate(model, 128)[0] is False
    monkeypatch.setenv("UNSLOTH_LORA_FOLD_GENERATE_MIN_TOKENS", "64")
    assert fold._min_tokens() == 64


def test_other_threads_wait_for_restore(_clean):
    import threading

    model = _make(MODELS[0])
    kw = {"max_new_tokens": 32}
    inside, release, order = threading.Event(), threading.Event(), []

    def first():
        with torch.inference_mode(), fold.lora_fold_generate(model, (), kw) as engaged:
            order.append(("first", engaged))
            inside.set()
            release.wait(30)
        order.append("first_restored")

    def second():
        inside.wait(30)
        with torch.inference_mode(), fold.lora_fold_generate(model, (), kw) as engaged:
            order.append(("second", engaged))

    threads = [threading.Thread(target = first), threading.Thread(target = second)]
    for t in threads:
        t.start()
    inside.wait(30)
    time.sleep(0.3)
    assert order == [("first", True)]  # the second call waits while weights are folded
    release.set()
    for t in threads:
        t.join(30)
    assert order == [("first", True), "first_restored", ("second", True)]
    assert fold._ACTIVE == 0 and fold._OWNER is None


def test_nested_call_keeps_the_outer_ownership(_clean):
    import threading

    model = _make(MODELS[0])
    kw = {"max_new_tokens": 32}
    inside, release, order = threading.Event(), threading.Event(), []

    def first():
        with torch.inference_mode(), fold.lora_fold_generate(model, (), kw) as engaged:
            with fold.lora_fold_generate(model, (), kw) as nested:
                order.append(("nested", nested))
            inside.set()
            release.wait(30)
        order.append("first_restored")

    def second():
        inside.wait(30)
        with torch.inference_mode(), fold.lora_fold_generate(model, (), kw) as engaged:
            order.append(("second", engaged))

    threads = [threading.Thread(target = first), threading.Thread(target = second)]
    for t in threads:
        t.start()
    inside.wait(30)
    time.sleep(0.3)
    assert order == [("nested", False)]  # still waiting: the outer fold owns the weights
    release.set()
    for t in threads:
        t.join(30)
    assert order == [("nested", False), "first_restored", ("second", True)]
    assert fold._ACTIVE == 0 and fold._OWNER is None


def test_nested_call_does_not_fold_twice(_clean):
    model = _make(MODELS[0])
    before = _bits(model)
    kw = {"max_new_tokens": 32}
    with torch.inference_mode(), fold.lora_fold_generate(model, (), kw) as outer:
        with fold.lora_fold_generate(model, (), kw) as inner:
            assert outer is True and inner is False
    assert _clean.folds == 1 and _clean.unfolds == 1
    assert fold.stats()["refused"] == {"nested": 1}
    assert _same_bits(before, _bits(model))


def _refusal_cases():
    def plain(m):
        return m

    def tp(m):
        m._unsloth_tp = True

    def device_map(m):
        m.base_model.model.hf_device_map = {"model.embed_tokens": 0, "lm_head": 1}

    def vllm(m):
        m.base_model.model.vllm_engine = object()

    def quantized(m):
        layer = m.base_model.model.model.layers[0].mlp.up_proj.base_layer

        class Packed(torch.nn.Parameter):
            pass

        layer.weight = Packed(layer.weight.data, requires_grad = False)

    def trainable(m):
        m.base_model.model.model.layers[0].mlp.up_proj.base_layer.weight.requires_grad_(True)

    def tied(m):
        attn = m.base_model.model.model.layers[0].self_attn
        attn.v_proj.base_layer.weight = attn.k_proj.base_layer.weight

    def second_adapter(m):
        m.add_adapter("other", LoraConfig(r = 4, target_modules = "all-linear"))
        m.base_model.set_adapter(["default", "other"])

    def merged(m):
        m.merge_adapter()

    def dropout_train(m):
        m.train()

    def qat(m):
        m.base_model.model.model.layers[
            0
        ].mlp.up_proj.base_layer.weight_fake_quantizer = torch.nn.Identity()

    def linear_subclass(m):
        layer = m.base_model.model.model.layers[0].mlp.up_proj.base_layer

        class Custom(torch.nn.Linear):
            pass

        layer.__class__ = Custom

    def adapter_bias(m):
        lora_a = m.base_model.model.model.layers[0].self_attn.q_proj.lora_A["default"]
        lora_a.bias = torch.nn.Parameter(
            torch.zeros(lora_a.out_features, dtype = lora_a.weight.dtype)
        )

    def adapter_param_subclass(m):
        lora_b = m.base_model.model.model.layers[0].self_attn.q_proj.lora_B["default"]

        class Packed(torch.nn.Parameter):
            pass

        lora_b.weight = Packed(lora_b.weight.data)

    def moe_config(m):
        m.base_model.model.config.num_experts = 4

    def mixed_hooks(m):
        from functools import partial
        from peft.tuners.lora.model import _adapter_names_pre_forward_hook

        hook = partial(_adapter_names_pre_forward_hook, adapter_names = ["default", "__base__"])
        m.base_model.model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
            hook, with_kwargs = True
        )

    return [
        ("no_lora", {}, "unwrap", None),
        ("distributed", {}, None, tp),
        ("multi_device", {}, None, device_map),
        ("vllm", {}, None, vllm),
        ("quantized", {}, None, quantized),
        ("base_trainable", {}, None, trainable),
        ("tied_weight", {}, None, tied),
        ("multiple_adapters", {}, None, second_adapter),
        ("already_merged", {}, None, merged),
        ("dora", {"use_dora": True}, None, None),
        ("lora_bias", {"lora_bias": True}, None, None),
        ("unsupported_layer", {"target_modules": ["q_proj", "embed_tokens"]}, None, None),
        ("dropout_active", {"lora_dropout": 0.1}, None, dropout_train),
        ("not_16bit", {}, "fp32", None),
        ("qat", {}, None, qat),
        ("mixed_adapters", {}, None, mixed_hooks),
        ("unsupported_layer", {}, None, linear_subclass),
        ("lora_bias", {}, None, adapter_bias),
        ("quantized", {}, None, adapter_param_subclass),
        ("moe", {}, None, moe_config),
    ]


@pytest.mark.parametrize(
    "reason,lora,kind,mutate", _refusal_cases(), ids = [c[0] for c in _refusal_cases()]
)
def test_refusals_leave_model_untouched(reason, lora, kind, mutate, _clean):
    if reason == "lora_bias" and "lora_bias" not in LoraConfig.__dataclass_fields__:
        pytest.skip("peft without lora_bias")
    if kind == "unwrap":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODELS[0], torch_dtype = torch.bfloat16
        )
    else:
        model = _make(MODELS[0], dtype = torch.float32 if kind == "fp32" else torch.bfloat16, **lora)
    if mutate is not None:
        mutate(model)
    before, merged = _bits(model), _merged_state(model)
    kw = {"max_new_tokens": 32}
    with torch.inference_mode(), fold.lora_fold_generate(model, (), kw) as engaged:
        assert engaged is False
    assert fold.stats()["refused"] == {reason: 1}
    assert fold.stats()["engaged"] == 0
    assert _clean.folds == 0 and _clean.unfolds == 0
    assert _same_bits(before, _bits(model)) and _merged_state(model) == merged


def test_environment_refusals(monkeypatch, _clean):
    model = _make(MODELS[0])
    kw = {"max_new_tokens": 32}
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    with torch.inference_mode(), fold.lora_fold_generate(model, (), kw) as engaged:
        assert engaged is False
    monkeypatch.delenv("ACCELERATE_USE_FSDP")
    monkeypatch.setenv("UNSLOTH_LORA_FOLD_GENERATE", "0")
    with torch.inference_mode(), fold.lora_fold_generate(model, (), kw) as engaged:
        assert engaged is False
    monkeypatch.delenv("UNSLOTH_LORA_FOLD_GENERATE")
    # Mixed-adapter batches: PEFT refuses adapter_names on merged layers.
    with (
        torch.inference_mode(),
        fold.lora_fold_generate(
            model, (), {**kw, "adapter_names": ["default", "__base__"]}
        ) as engaged,
    ):
        assert engaged is False
    # Grad on: refused (the Unsloth wrappers enter it inside their inference-mode scope).
    with fold.lora_fold_generate(model, (), kw) as engaged:
        assert engaged is False
    _clean.available = False
    with torch.inference_mode(), fold.lora_fold_generate(model, (), kw) as engaged:
        assert engaged is False
    assert fold.stats()["refused"] == {
        "distributed": 1,
        "kill_switch": 1,
        "mixed_adapters": 1,
        "grad_enabled": 1,
        "exact_unavailable": 1,
    }
    assert _clean.folds == 0


def test_real_zoo_api_refuses_on_cpu(monkeypatch):
    zoo = pytest.importorskip("unsloth_zoo.merged_rollout_exact")
    zoo_root = os.environ.get("UNSLOTH_ZOO_TREE")
    if zoo_root:
        assert Path(zoo.__file__).resolve().is_relative_to(Path(zoo_root).resolve()), zoo.__file__
    monkeypatch.setattr(fold, "_exact_api", lambda: zoo)
    model = _make(MODELS[0])
    before = _bits(model)
    with (
        torch.inference_mode(),
        fold.lora_fold_generate(model, (), {"max_new_tokens": 32}) as engaged,
    ):
        assert engaged is False
    assert fold.stats()["refused"] == {"exact_unavailable": 1}
    assert _same_bits(before, _bits(model))


def _train_step(model, opt, ids):
    model.train()
    out = model(ids, labels = ids)
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    opt.step()
    opt.zero_grad(set_to_none = True)
    model.eval()
    return out.loss.detach(), grads


@pytest.mark.parametrize("pending_graph", [False, True])
def test_training_after_generate_is_bitwise_unchanged(pending_graph):
    ids = _inputs(2, 12)
    results = []
    for do_generate in (False, True):
        model = _make(MODELS[1])
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr = 1e-3)
        _train_step(model, opt, ids)
        if pending_graph:
            # A loss computed before generate() backpropagates through the restored weights.
            model.train()
            out = model(ids, labels = ids)
            model.eval()
            if do_generate:
                assert _generate(model, 20)[0] is True
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none = True)
        elif do_generate:
            assert _generate(model, 20)[0] is True
        loss, grads = _train_step(model, opt, ids)
        results.append((loss, grads, _bits(model)))
    (l0, g0, b0), (l1, g1, b1) = results
    assert torch.equal(l0, l1)
    assert g0.keys() == g1.keys() and all(torch.equal(g0[k], g1[k]) for k in g0)
    assert _same_bits(b0, b1)
    assert fold.stats()["engaged"] == 1 and fold.stats()["restored"] == 1


def _with_blocks(path, name):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_old_generate"
    ]
    withs = [n for n in ast.walk(fn) if isinstance(n, ast.With)]
    return calls, withs


@pytest.mark.parametrize(
    "path,name",
    [
        (ROOT / "unsloth" / "models" / "llama.py", "unsloth_fast_generate"),
        (ROOT / "unsloth" / "models" / "vision.py", "unsloth_base_fast_generate"),
    ],
)
def test_every_generate_path_is_wrapped(path, name):
    calls, withs = _with_blocks(path, name)
    assert calls
    for call in calls:
        owner = [w for w in withs if any(c is call for c in ast.walk(w))]
        assert owner, f"{name}: _old_generate outside a with block"
        items = [ast.unparse(item.context_expr) for item in owner[-1].items]
        assert "_lora_fold_generate(self, args, kwargs)" in items, items
        # Inside the inference-mode scope, so grad is off when it decides.
        assert items.index("_lora_fold_generate(self, args, kwargs)") > items.index(
            "_get_inference_mode_context_manager(self)"
        )
