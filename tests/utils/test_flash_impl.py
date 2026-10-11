# SPDX-License-Identifier: AGPL-3.0-only
"""FA4 / FA3 activation gates (CPU, faked registry) and FA4 parity on real sm90 / sm100 hardware."""

import importlib.util
import os
import types

import pytest
import torch
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

from unsloth.utils import flash_impl as fi


class FakeRegistry:
    def __init__(
        self,
        impls = ("FA3", "FA4"),
        current = None,
    ):
        self.impls, self.current_impl, self.calls = list(impls), current, []

    def activate(self, impl):
        self.calls.append(("activate", impl))
        self.current_impl = impl

    def restore(self, _raise_warn = True):
        self.calls.append(("restore",))
        self.current_impl = None

    def tuple(self):
        return (self.activate, self.restore, lambda: sorted(self.impls), lambda: self.current_impl)


def _config(**kwargs):
    base = dict(
        num_attention_heads = 32,
        num_key_value_heads = 8,
        hidden_size = 4096,
        head_dim = 128,
        attention_dropout = 0.0,
    )
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def _model(**kwargs):
    return types.SimpleNamespace(config = _config(**kwargs), dtype = torch.bfloat16)


@pytest.fixture
def fake(monkeypatch):
    registry = FakeRegistry()
    monkeypatch.setattr(fi, "_registry", registry.tuple)
    monkeypatch.setattr(fi, "_device_majors", lambda: {10})
    monkeypatch.setattr(fi, "_package_error", lambda impl: None)
    probes = []

    def probe(impl, cases, activate, restore):
        probes.append((impl, cases))
        activate(impl)
        return None

    monkeypatch.setattr(fi, "_probe_error", probe)
    monkeypatch.setitem(fi.FLASH_IMPL_STATE, "active", None)
    monkeypatch.setitem(fi.FLASH_IMPL_STATE, "verdicts", {})
    monkeypatch.setenv(fi._ENV, "auto")
    registry.probes = probes
    return registry


def test_off_by_default(fake, monkeypatch):
    monkeypatch.delenv(fi._ENV, raising = False)
    assert fi.maybe_activate_flash_attention_impl(_model()) is None
    assert fake.calls == [] and fake.probes == []


def test_auto_activates_fa4_on_sm100(fake):
    assert fi.maybe_activate_flash_attention_impl(_model()) == "FA4"
    assert fake.current_impl == "FA4" and len(fake.probes) == 1
    # Decided once: a second call with the same shapes does not re-probe.
    assert fi.maybe_activate_flash_attention_impl(_model()) == "FA4"
    assert len(fake.probes) == 1
    fi.deactivate_flash_attention_impl()
    assert fake.current_impl is None and fi.FLASH_IMPL_STATE["active"] is None


@pytest.mark.parametrize("value", ["off", "0", "false", "", "bogus"])
def test_kill_switch_and_unknown_value_stay_off(fake, monkeypatch, value):
    monkeypatch.setenv(fi._ENV, value)
    assert fi.maybe_activate_flash_attention_impl(_model()) is None
    assert fake.calls == [] and fake.probes == []


def test_no_registry_is_a_no_op(fake, monkeypatch):
    monkeypatch.setattr(fi, "_registry", lambda: None)
    assert fi.maybe_activate_flash_attention_impl(_model()) is None
    assert fake.calls == []


@pytest.mark.parametrize("majors", [None, {8}, {12}, {11}, {9, 10}, set()])
def test_unsupported_or_mixed_devices_stay_off(fake, monkeypatch, majors):
    monkeypatch.setattr(fi, "_device_majors", lambda: majors)
    assert fi.maybe_activate_flash_attention_impl(_model()) is None
    assert fake.probes == [] and fake.current_impl is None


def test_hopper_is_opt_in(fake, monkeypatch):
    monkeypatch.setattr(fi, "_device_majors", lambda: {9})
    assert fi.maybe_activate_flash_attention_impl(_model()) is None
    assert fake.probes == []
    monkeypatch.setenv(fi._ENV, "fa3")
    assert fi.maybe_activate_flash_attention_impl(_model()) == "FA3"
    fi.deactivate_flash_attention_impl()
    monkeypatch.setenv(fi._ENV, "fa4")
    assert fi.maybe_activate_flash_attention_impl(_model()) == "FA4"
    fi.deactivate_flash_attention_impl()
    monkeypatch.setattr(fi, "_device_majors", lambda: {10})
    monkeypatch.setenv(fi._ENV, "fa3")
    assert fi.maybe_activate_flash_attention_impl(_model()) is None


def test_impl_not_registered_or_package_missing_stays_off(fake, monkeypatch):
    fake.impls = ["FA3"]
    assert fi.maybe_activate_flash_attention_impl(_model()) is None
    fake.impls = ["FA4"]
    monkeypatch.setattr(fi, "_package_error", lambda impl: "flash_attn.cute not importable")
    assert fi.maybe_activate_flash_attention_impl(_model()) is None
    assert fake.probes == []


@pytest.mark.parametrize(
    "model",
    [
        _model(attention_dropout = 0.1),
        types.SimpleNamespace(
            config = types.SimpleNamespace(
                text_config = _config(), vision_config = _config(attention_dropout = 0.05)
            )
        ),
        _model(attn_pdrop = 0.1),
    ],
)
def test_attention_dropout_stays_off(fake, model):
    assert fi.maybe_activate_flash_attention_impl(model) is None
    assert "dropout" in fi.FLASH_IMPL_STATE["reason"]
    assert fake.probes == []


def test_attention_module_dropout_stays_off(fake):
    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_dropout = 0.1

    model = torch.nn.Sequential(FakeAttention())
    model.config = _config()
    assert fi.maybe_activate_flash_attention_impl(model) is None


@pytest.mark.parametrize("head_dim", [160, 192, 136])
def test_head_dim_fa4_lacks_on_sm100_stays_off(fake, head_dim):
    assert fi.maybe_activate_flash_attention_impl(_model(head_dim = head_dim)) is None
    assert "head_dim" in fi.FLASH_IMPL_STATE["reason"]


def test_head_dims_that_never_reach_flash_do_not_block(fake):
    # MLA (192 / 128) and Gemma-4 global 512 never reach torch's flash op with unequal / >256 heads.
    mla = _model(qk_nope_head_dim = 128, qk_rope_head_dim = 64, v_head_dim = 128)
    assert fi.maybe_activate_flash_attention_impl(mla) == "FA4"
    fi.deactivate_flash_attention_impl()
    assert (
        fi.maybe_activate_flash_attention_impl(_model(head_dim = 256, global_head_dim = 512)) == "FA4"
    )
    fi.deactivate_flash_attention_impl()


def test_probe_failure_restores_and_stays_off(fake, monkeypatch):
    def failing(impl, cases, activate, restore):
        activate(impl)
        restore(_raise_warn = False)
        return "probe failed (mismatch)"

    monkeypatch.setattr(fi, "_probe_error", failing)
    assert fi.maybe_activate_flash_attention_impl(_model()) is None
    assert fake.current_impl is None and fi.FLASH_IMPL_STATE["active"] is None
    # The failed verdict is remembered: no second probe, still off.
    monkeypatch.setattr(fi, "_probe_error", lambda *a: pytest.fail("re-probed"))
    assert fi.maybe_activate_flash_attention_impl(_model()) is None


def test_user_activated_impl_left_alone(fake):
    fake.current_impl = "FA4"
    assert fi.maybe_activate_flash_attention_impl(_model()) is None
    assert fake.calls == [] and fake.current_impl == "FA4"
    fi.deactivate_flash_attention_impl()
    assert fake.current_impl == "FA4"


def test_later_model_failing_gates_restores(fake):
    assert fi.maybe_activate_flash_attention_impl(_model()) == "FA4"
    assert fi.maybe_activate_flash_attention_impl(_model(attention_dropout = 0.1)) is None
    assert fake.current_impl is None


def test_scope_to_train_activates_only_inside_train(fake):
    seen = []

    class Trainer:
        def train(self, *args, **kwargs):
            seen.append(fake.current_impl)
            return "done"

    trainer = Trainer()
    trainer.model = _model()
    fi.scope_flash_attention_impl_to_train(trainer, trainer.model)
    fi.scope_flash_attention_impl_to_train(trainer, trainer.model)  # idempotent
    assert fake.current_impl is None
    assert trainer.train() == "done" and seen == ["FA4"]
    assert fake.current_impl is None
    assert trainer.train() == "done" and seen == ["FA4", "FA4"]
    assert len(fake.probes) == 1


def test_scope_restores_when_train_raises(fake):
    class Trainer:
        def train(self):
            raise RuntimeError("boom")

    trainer = Trainer()
    trainer.model = _model()
    fi.scope_flash_attention_impl_to_train(trainer, trainer.model)
    with pytest.raises(RuntimeError):
        trainer.train()
    assert fake.current_impl is None


def test_reused_padding_free_collator_still_scopes_the_new_trainer(fake):
    from unsloth.utils import packing

    class Trainer:
        def train(self):
            return fake.current_impl

    collator = types.SimpleNamespace(padding_free = True, _unsloth_padding_free_lengths_wrapped = True)
    trainer = Trainer()
    trainer.model, trainer.data_collator = _model(), collator
    packing.enable_padding_free_metadata(trainer.model, trainer)
    assert trainer.train() == "FA4" and fake.current_impl is None


@pytest.mark.parametrize("native_gqa", [True, False])
def test_probe_repeats_kv_heads_without_enable_gqa(monkeypatch, native_gqa):
    import torch.nn.attention.varlen as varlen

    seen = []

    def fake_with_gqa(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        window_size = (-1, -1),
        enable_gqa = False,
    ):
        seen.append((k.shape[1], enable_gqa))
        return q * (k.float().mean() + v.float().mean()).to(q.dtype)

    def fake_2_11(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        window_size = (-1, -1),
    ):
        seen.append((k.shape[1], None))
        return q * (k.float().mean() + v.float().mean()).to(q.dtype)

    monkeypatch.setattr(varlen, "varlen_attn", fake_with_gqa if native_gqa else fake_2_11)
    out, dq, dk, dv = fi._run_case(("varlen", 64, 4, None, torch.float32), "cpu")
    assert seen == [(2, True)] if native_gqa else seen == [(8, None)]
    assert dk.shape == dv.shape == (sum([200, 1, 333, 64, 130]), 2, 64)


def test_scope_skipped_when_off(fake, monkeypatch):
    monkeypatch.setenv(fi._ENV, "off")

    class Trainer:
        def train(self):
            return 1

    trainer = Trainer()
    fi.scope_flash_attention_impl_to_train(trainer, None)
    assert "train" not in vars(trainer)


@pytest.mark.parametrize(
    "impl,major,head_dim,window,ok",
    [
        ("FA4", 10, 64, None, True),
        ("FA4", 10, 128, 4096, True),
        ("FA4", 10, 136, None, False),
        ("FA4", 10, 160, None, False),
        ("FA4", 10, 192, None, False),
        ("FA4", 10, 256, None, True),
        ("FA4", 10, 256, 4096, False),
        ("FA4", 10, 72, None, True),
        ("FA4", 10, 100, None, False),
        ("FA4", 9, 160, None, True),
        ("FA4", 9, 256, 4096, True),
        ("FA3", 9, 192, None, True),
        ("FA3", 9, 264, None, False),
    ],
)
def test_head_dim_supported(impl, major, head_dim, window, ok):
    assert fi.head_dim_supported(impl, major, head_dim, window = window) is ok


def test_flash_impl_allows_reads_torch_active_impl(monkeypatch):
    registry = types.SimpleNamespace(_FLASH_ATTENTION_ACTIVE = None)
    monkeypatch.setattr(fi, "_TORCH_FLASH_REGISTRY", registry)
    assert fi.flash_impl_allows(160, None, torch.device("cuda", 0))
    registry._FLASH_ATTENTION_ACTIVE = ("FA4", object())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (10, 0))
    assert fi.flash_impl_allows(128, None, torch.device("cuda", 0))
    assert not fi.flash_impl_allows(160, None, torch.device("cuda", 0))
    assert not fi.flash_impl_allows(256, 4096, torch.device("cuda", 0))
    assert fi.flash_impl_allows(256, None, torch.device("cuda", 0))


def _fa4_ready():
    if not has_real_cuda() or torch.version.hip is not None:
        return False
    if fi._registry() is None or "FA4" not in fi._registry()[2]():
        return False
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        return False
    return importlib.util.find_spec("flash_attn") is not None and fi._package_error("FA4") is None


needs_fa4 = pytest.mark.skipif(
    not _fa4_ready(),
    reason = "needs sm90 / sm100 CUDA, torch >= 2.11 flash impl registry and flash_attn.cute (FA4)",
)


@pytest.fixture
def real_fa4(monkeypatch):
    monkeypatch.setenv(fi._ENV, "fa4")
    monkeypatch.setitem(fi.FLASH_IMPL_STATE, "active", None)
    yield
    fi.deactivate_flash_attention_impl()
    assert fi._current_impl() is None
    # Restored for real: the dispatcher no longer reaches torch's FA4 runner.
    with fi._count_impl_calls("FA4") as calls:
        _varlen(64, torch.bfloat16, None, lengths = (64, 32))
    assert calls == {"fwd": 0, "bwd": 0}, calls


def _varlen(
    head_dim,
    dtype,
    window,
    lengths = (300, 17, 1000, 1, 731),
    hq = 16,
    hk = 4,
    seed = 0,
):
    from torch.nn.attention.varlen import varlen_attn

    g = torch.Generator(device = "cuda").manual_seed(seed)
    total = sum(lengths)

    def randn(*shape):
        return torch.randn(*shape, device = "cuda", dtype = dtype, generator = g)

    q, k, v = (randn(total, h, head_dim).requires_grad_() for h in (hq, hk, hk))
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device = "cuda", dtype = torch.int32
    )
    out = varlen_attn(
        q,
        k,
        v,
        cu,
        cu,
        max(lengths),
        max(lengths),
        window_size = (window - 1, 0) if window else (-1, 0),
        enable_gqa = True,
    )
    out.backward(randn(*out.shape))
    return out.detach(), q.grad, k.grad, v.grad


@needs_fa4
def test_fa4_varlen_matches_bundled_flash(real_fa4):
    cases = [
        (64, torch.bfloat16, None),
        (64, torch.bfloat16, 128),
        (128, torch.bfloat16, None),
        (128, torch.bfloat16, 128),
        (256, torch.bfloat16, None),
        (128, torch.float16, None),
        (128, torch.float16, 128),
    ]
    if torch.cuda.get_device_capability()[0] == 9:
        cases.append((256, torch.bfloat16, 128))
    refs = [_varlen(*case) for case in cases]
    model = _model(num_attention_heads = 16, num_key_value_heads = 4)
    assert fi.maybe_activate_flash_attention_impl(model) == "FA4", fi.FLASH_IMPL_STATE["reason"]
    for case, ref in zip(cases, refs):
        got = _varlen(*case)
        for a, b in zip(got, ref):
            err = (a.float() - b.float()).abs().max().item() / b.float().abs().max().item()
            assert err < 2e-2, (case, err)


@needs_fa4
def test_fa4_backward_deterministic_mode(real_fa4):
    assert fi.maybe_activate_flash_attention_impl(_model()) == "FA4"
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only = True)
    try:
        a = _varlen(128, torch.bfloat16, None, lengths = (3000, 17, 1000, 1, 731) * 2)
        b = _varlen(128, torch.bfloat16, None, lengths = (3000, 17, 1000, 1, 731) * 2)
    finally:
        torch.use_deterministic_algorithms(previous)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@needs_fa4
def test_fa4_run_attention_compiles_without_breaks(real_fa4):
    import torch._dynamo
    from torch._dynamo.utils import counters

    from unsloth.utils import attention_dispatch as ad
    from unsloth.utils import packing

    if ad._TORCH_VARLEN_ATTN is None:
        pytest.skip("torch varlen_attn with window_size + enable_gqa unavailable")
    assert fi.maybe_activate_flash_attention_impl(_model()) == "FA4"
    lengths = [300, 700, 24, 500, 524]
    total = sum(lengths)
    lens = torch.tensor(lengths, dtype = torch.int32, device = "cuda")
    cu = torch.zeros(len(lengths) + 1, dtype = torch.int32, device = "cuda")
    cu[1:] = lens.cumsum(0)
    config = ad.AttentionConfig(backend = ad.SDPA, n_kv_heads = 8, n_groups = 4)
    ctx = ad.AttentionContext(
        bsz = 1,
        q_len = total,
        kv_seq_len = total,
        n_heads = 32,
        head_dim = 128,
        requires_grad = True,
        seq_info = (lens, cu, max(lengths)),
        attention_mask = None,
        causal_mask = None,
    )
    q = torch.randn(1, 32, total, 128, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    k = torch.randn(1, 8, total, 128, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    calls = []
    real = ad._TORCH_VARLEN_ATTN

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    def f(q, k, v):
        return ad.run_attention(config = config, context = ctx, Q = q, K = k, V = v)

    try:
        ad._TORCH_VARLEN_ATTN = counted
        eager = f(q, k, k)
        assert calls, "packed row did not take the varlen route"
        torch._dynamo.reset()
        counters.clear()
        packing.clear_packed_caches()
        out = torch.compile(f)(q, k, k)
        out.float().sum().backward()
    finally:
        ad._TORCH_VARLEN_ATTN = real
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    assert (out.float() - eager.float()).abs().max().item() < 2e-2


@needs_fa4
def test_fa4_hd256_window_keeps_segments_route(real_fa4):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("sm100 hd256 backward has no local attention; sm90 does")
    from unsloth.utils import attention_dispatch as ad

    assert fi.maybe_activate_flash_attention_impl(_model(head_dim = 256)) == "FA4"
    q = torch.randn(1, 8, 512, 256, device = "cuda", dtype = torch.bfloat16)
    assert not ad._torch_varlen_takes(q, q[:, :2], q[:, :2], {}, True, 128)
    assert ad._torch_varlen_takes(q, q[:, :2], q[:, :2], {}, True, None) == (
        ad._TORCH_VARLEN_ATTN is not None
    )


@pytest.mark.skipif(
    not has_real_cuda(), reason = "needs CUDA tensors (the wrapper only takes CUDA rows)"
)
@pytest.mark.skipif(torch.version.hip is not None, reason = "torch varlen is off on ROCm")
@pytest.mark.parametrize("window", [None, 2])
@pytest.mark.parametrize("allowed", [True, False])
def test_hf_wrapper_skips_torch_varlen_for_shapes_the_impl_lacks(monkeypatch, allowed, window):
    from unsloth.utils import hf_packed_attention as hpa
    from unsloth.utils import attention_dispatch as ad

    seen, used, asked = [], [], []

    def orig(
        module,
        q,
        k,
        v,
        mask,
        dropout = 0.0,
        scaling = None,
        is_causal = None,
        **kwargs,
    ):
        seen.append(1)
        return "orig", None

    def fake_varlen(q, k, v, *args, **kwargs):
        used.append(1)
        return q

    saved = hpa._ORIG_SDPA[0]
    hpa._ORIG_SDPA[0] = orig
    hpa._MASK_CHECK.clear()
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    monkeypatch.setattr(hpa, "_TORCH_VARLEN", [fake_varlen])
    monkeypatch.setattr(hpa, "_TORCH_VARLEN_DEVICES", {})
    monkeypatch.setattr(hpa, "_sdpa_requested", lambda q, k: False)
    # No FA2 / xformers either, so a refused torch varlen goes back to the wrapped sdpa.
    monkeypatch.setattr(ad, "select_attention_backend", lambda use_varlen = False: ad.SDPA)
    monkeypatch.setattr(
        fi, "flash_impl_allows", lambda head_dim, window, device: asked.append(window) or allowed
    )
    lengths = [3, 3]
    q = torch.randn(1, 4, 6, 64, device = "cuda", dtype = torch.bfloat16)
    segment = torch.repeat_interleave(torch.arange(2), torch.tensor(lengths)).cuda()
    mask = (segment[:, None] == segment[None, :]).tril()[None, None]
    if window is not None:
        position = torch.arange(6, device = "cuda")
        mask = mask & (position[None, :] > position[:, None] - window)
    try:
        out = hpa._sdpa_packed_varlen(
            types.SimpleNamespace(is_causal = True, sliding_window = window),
            q,
            q,
            q,
            mask,
            packed_seq_lengths = torch.tensor(lengths, dtype = torch.int32),
        )
    finally:
        hpa._ORIG_SDPA[0] = saved
        hpa._MASK_CHECK.clear()
    assert asked == [window]
    if allowed:
        assert used and not seen
    else:
        assert out == ("orig", None) and not used
