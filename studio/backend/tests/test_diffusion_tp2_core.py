# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-process TP-2 (diffusion_tp2) sharding, setup, placement and guard rules on a stub Qwen-Image-2.1 that needs
no diffusers, so the default backend CI runs them (the model-level equivalence tests are test_diffusion_tp2.py)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from core.inference import diffusion_tp2 as tp2  # noqa: E402

CPU2 = [torch.device("cpu"), torch.device("cpu")]
DIM, HEADS = 64, 4


def dispatch_attention_fn(*a, **k):  # the name the INT8 dispatch discovery looks for
    raise AssertionError("not called in these tests")


class QwenImage21AttnProcessor:
    _attention_backend = None

    def __call__(self, attn, hidden_states, **kw):
        return dispatch_attention_fn(hidden_states)


class QwenImage21FlexAttnProcessor(QwenImage21AttnProcessor):
    pass


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads, self.inner_dim = HEADS, DIM
        self.to_q, self.to_k, self.to_v = (nn.Linear(DIM, DIM, bias = False) for _ in range(3))
        self.to_out = nn.ModuleList([nn.Linear(DIM, DIM, bias = False), nn.Dropout(0.0)])
        self.processor = QwenImage21AttnProcessor()


class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_layer, self.proj = (
            nn.Linear(DIM, 3 * DIM, bias = False),
            nn.Linear(DIM, 3 * DIM, bias = False),
        )
        self.out = nn.Linear(3 * DIM, DIM, bias = False)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.img_mlp = Attn(), FF()


class QwenImage21Transformer2DModel(nn.Module):
    def __init__(self, layers = 2):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads = HEADS, attention_head_dim = DIM // HEADS, mlp_ratio = 3
        )
        self.img_in = nn.Linear(16, DIM, bias = False)
        self.transformer_blocks = nn.ModuleList(Block() for _ in range(layers))

    @property
    def dtype(self):
        return self.img_in.weight.dtype


def _stub(layers = 2):
    torch.manual_seed(0)
    return QwenImage21Transformer2DModel(layers)


def _weights(m):
    return [p.detach().clone() for p in m.parameters()]


@dataclasses.dataclass(frozen = True)
class Plan:
    offload_policy: str = "group"
    stream_transformer: bool = True
    stream_text_encoders: bool = False
    resident_transformer_mib: object = 100
    secondary_ordinal: object = None
    secondary_components: object = None
    layer_split: object = None
    fallback_plan: object = None
    reasons: tuple = ("planned",)


@pytest.fixture
def two_cards(monkeypatch):
    from core.inference import diffusion_multigpu as mg

    monkeypatch.setenv(tp2.TP_ENV, "1")
    monkeypatch.setattr(
        mg, "current_selection", lambda: mg.MultiGpuSelection(primary = 0, secondaries = (1, 2))
    )
    monkeypatch.setattr(tp2, "refusal", lambda tr, devs: None)
    monkeypatch.setattr(tp2, "_card_free_total_mib", lambda d: (15000, 15360))
    calls = []
    monkeypatch.setattr(
        tp2,
        "install",
        lambda tr, devs, comm = None, logger = None: calls.append(devs)
        or {"devices": [str(d) for d in devs]},
    )
    return calls


# ------------------------------------------------------------------------------------------ type / attention gates


def test_only_exact_linear_types_and_plain_weights_shard():
    from core.inference.diffusion_convrot import _install_rotation

    assert tp2.refusal(_stub(), CPU2) is None
    m = _stub()
    _install_rotation(m.transformer_blocks[0].attn.to_q, 64)
    assert "ConvRotLinear" in tp2.refusal(m, CPU2)

    class Sub(nn.Linear):
        pass

    m = _stub()
    m.transformer_blocks[1].img_mlp.out = Sub(3 * DIM, DIM, bias = False)
    assert "Sub" in tp2.refusal(m, CPU2)
    m = _stub()
    k = m.transformer_blocks[0].attn.to_k
    k.weight = nn.Parameter(k.weight.data.to(torch.int8), requires_grad = False)
    assert "weight" in tp2.refusal(m, CPU2)


def test_flex_attention_is_refused_and_a_block_mask_never_reaches_card_two():
    m = _stub()
    m.transformer_blocks[1].attn.processor = QwenImage21FlexAttnProcessor()
    why = tp2.refusal(m, CPU2)
    assert why is not None and "flex" in why.lower()
    m = _stub()
    tp2.install(m, CPU2)
    stack = m.transformer_blocks[0]
    with pytest.raises(RuntimeError, match = "BlockMask"):
        stack(
            torch.zeros(1, 4, DIM),
            torch.zeros(2, 4 * DIM),
            attention_mask = SimpleNamespace(mask_mod = None),
        )


# ------------------------------------------------------------------------------------------------ setup / rollback


@pytest.mark.parametrize("where", ["comm", "shard", "place"])
def test_failed_install_keeps_the_original_blocks(monkeypatch, where):
    m = _stub(3)
    before = _weights(m)
    blocks = list(m.transformer_blocks)
    if where == "comm":
        monkeypatch.setattr(
            tp2, "make_all_reduce", lambda *a, **k: (_ for _ in ()).throw(MemoryError("probe"))
        )
    elif where == "shard":
        real, n = tp2.shard_block, []

        def flaky(*a, **k):
            n.append(1)
            if len(n) == 4:
                raise MemoryError("shard")
            return real(*a, **k)

        monkeypatch.setattr(tp2, "shard_block", flaky)
    else:
        real_to = nn.Module.to

        def failing(self, *a, **k):
            if self is m and any(isinstance(x, torch.device) for x in a):
                raise MemoryError("place")
            return real_to(self, *a, **k)

        monkeypatch.setattr(nn.Module, "to", failing)
    with pytest.raises(MemoryError):
        tp2.install(m, CPU2)
    monkeypatch.undo()
    assert not tp2.is_installed(m) and list(m.transformer_blocks) == blocks
    assert all(p.device.type == "cpu" for p in m.parameters())
    assert all(torch.equal(a, b) for a, b in zip(_weights(m), before))


def test_failure_retiring_the_originals_leaves_tp_complete_and_marked(monkeypatch):
    m = _stub()
    real_to = nn.Module.to

    def no_meta(self, *a, **k):
        if a and a[0] == "meta":
            raise RuntimeError("cannot free")
        return real_to(self, *a, **k)

    monkeypatch.setattr(nn.Module, "to", no_meta)
    rec = tp2.install(m, CPU2)
    assert tp2.is_installed(m) and "originals_kept" in rec
    assert type(m.transformer_blocks[0]).__name__ == "TPStack"


def test_a_pinned_backend_is_probed_before_success(monkeypatch):
    m = _stub()
    blocks = list(m.transformer_blocks)

    def broken(*xs):
        raise RuntimeError("NCCL is not available in this build")

    monkeypatch.setattr(tp2, "make_all_reduce", lambda kind, devs: broken)
    with pytest.raises(RuntimeError, match = "no all-reduce backend works"):
        tp2.install(m, CPU2, comm = "nccl")
    assert not tp2.is_installed(m) and list(m.transformer_blocks) == blocks


def test_shards_own_their_storage_and_shard_mib_counts_storage():
    m = _stub()
    full = m.transformer_blocks[0].attn.to_q.weight
    sb = tp2.shard_block(m.transformer_blocks[0], 0, 2, "cpu")  # same device as the source
    w = sb.attn.to_q.weight
    assert w.untyped_storage().nbytes() == w.numel() * w.element_size()
    assert w.untyped_storage().data_ptr() != full.untyped_storage().data_ptr()
    # shard accounting counts storage: a view of the whole weight would report the whole weight
    view = full[: DIM // 2]
    assert tp2._storage_mib([view]) == tp2._storage_mib([full])


# ------------------------------------------------------------------------------------------- placement / fit / plan


def test_engage_prices_per_card_headroom(two_cards, monkeypatch):
    pipe, plan = SimpleNamespace(transformer = _stub()), Plan()
    monkeypatch.setattr(tp2, "_card_free_total_mib", lambda d: (100, 15360))  # nothing fits
    assert tp2.engage_for_plan(pipe, plan) is plan and two_cards == []
    # only cuda:2 has room next to cuda:0: TP takes it, not the first secondary
    free = {0: 15000, 1: 100, 2: 15000}
    monkeypatch.setattr(tp2, "_card_free_total_mib", lambda d: (free[d.index], 15360))
    out = tp2.engage_for_plan(pipe, plan)
    assert out is not plan and [str(d) for d in two_cards[0]] == ["cuda:0", "cuda:2"]


@pytest.mark.parametrize("policy", ["streaming", "model", "sequential", "group"])
def test_tp_plan_never_offloads_the_denoiser_through_apply_memory_plan(
    two_cards, monkeypatch, policy
):
    from core.inference import diffusion_memory as dm

    plan = dm.MemoryPlan(
        requested_mode = "auto",
        offload_policy = policy,
        vae_tiling = False,
        vae_slicing = False,
        device_memory = dm.DeviceMemory(backend = "cuda", device = "cuda", memory_kind = "discrete_vram"),
        estimates = {},
        resident_transformer_mib = 512,
    )
    out = tp2.engage_for_plan(SimpleNamespace(transformer = _stub()), plan)
    assert (
        out.offload_policy == "group"
        and out.stream_transformer is False
        and out.stream_text_encoders
    )
    seen = []
    monkeypatch.setattr(
        dm,
        "_apply_group_offload",
        lambda pipe, placement, logger, **kw: seen.append(("group", kw)) or True,
    )
    monkeypatch.setattr(
        dm, "_apply_streaming_offload", lambda *a, **k: seen.append(("streaming", k))
    )
    pipe = SimpleNamespace(
        transformer = object(),
        enable_model_cpu_offload = lambda **k: seen.append(("model", k)),
        enable_sequential_cpu_offload = lambda **k: seen.append(("sequential", k)),
    )
    dm.apply_memory_plan(pipe, out, device = "cuda", placement_device = "cuda:0")
    assert [k for k, _ in seen] == ["group"]
    assert (
        seen[0][1].get("stream_transformer") is False
        and "resident_transformer_mib" not in seen[0][1]
    )


def test_resident_plan_stays_resident(two_cards):
    out = tp2.engage_for_plan(SimpleNamespace(transformer = _stub()), Plan(offload_policy = "none"))
    assert out.offload_policy == "none" and out.stream_transformer is False


def test_hook_is_a_no_op_when_off_or_on_one_card(two_cards, monkeypatch):
    from core.inference import diffusion_multigpu as mg

    pipe, plan = SimpleNamespace(transformer = _stub()), Plan()
    for off in ("0", "off", ""):
        monkeypatch.setenv(tp2.TP_ENV, off)
        assert tp2.engage_for_plan(pipe, plan) is plan
    monkeypatch.setenv(tp2.TP_ENV, "1")
    for sel in (None, mg.MultiGpuSelection(primary = 0, secondaries = ())):
        monkeypatch.setattr(mg, "current_selection", lambda s = sel: s)
        assert tp2.engage_for_plan(pipe, plan) is plan
    assert two_cards == []


def test_engage_falls_back_when_setup_fails(two_cards, monkeypatch):
    monkeypatch.setattr(
        tp2, "install", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NCCL init failed"))
    )
    plan, pipe = Plan(), SimpleNamespace(transformer = _stub())
    assert tp2.engage_for_plan(pipe, plan) is plan and not hasattr(pipe, "_unsloth_tp2")


def test_engage_declines_a_companion_split_and_load_time_loras(two_cards):
    pipe = SimpleNamespace(transformer = _stub())
    split = Plan(offload_policy = "none", secondary_components = ("text_encoder",), secondary_ordinal = 1)
    assert tp2.engage_for_plan(pipe, split) is split
    plan = Plan()
    assert tp2.engage_for_plan(pipe, plan, loras = [("style", 0.8)]) is plan and two_cards == []


# ---------------------------------------------------------------------------------------- numerics / comm / compile


def test_fp16_row_partials_do_not_overflow_into_nan():
    lin = nn.Linear(4, 1, bias = False).half()
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[4.0, 4.0, -4.0, -4.0]]))
    x = torch.tensor([[1.5e4, 1.5e4, 1.5e4, 1.5e4]], dtype = torch.float16)
    rows = [tp2._row_parallel_class()(tp2.shard_linear(lin, 1, i, 2, "cpu")) for i in range(2)]
    parts = tp2.make_all_reduce("cpu", CPU2)([rows[0](x[:, :2]), rows[1](x[:, 2:])])
    out = parts[0].to(torch.float16)
    assert torch.isfinite(out).all() and float(out) == 0.0


def test_fp16_gemm_oom_propagates_and_is_not_cached(monkeypatch):
    tp2._MM_OUT_DTYPE.clear()
    real = torch.mm

    def mm(a, b, **kw):
        if "out_dtype" in kw:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real(a, b)

    monkeypatch.setattr(torch, "mm", mm)
    x, w = torch.ones(2, 4, dtype = torch.float16), torch.ones(3, 4, dtype = torch.float16)
    with pytest.raises(torch.OutOfMemoryError):
        tp2._mm_fp32(x, w)
    assert tp2._MM_OUT_DTYPE.get(("cpu", torch.float16), True) is True


def test_comm_choice_prefers_nccl_within_the_margin():
    assert tp2.choose_comm({"nccl": 4.4, "swap": 4.3, "peer": 7.0}) == "nccl"
    assert tp2.choose_comm({"nccl": 5.7, "swap": 4.4, "peer": 7.5}) == "swap"
    assert tp2.choose_comm({"nccl": "failed: x"}) is None


def test_hostswap_buffers_are_bounded():
    from collections import OrderedDict

    host, tmp = OrderedDict(), OrderedDict()
    for n in (32, 64, 128):
        tp2._evict(host, tmp, tp2.HOSTSWAP_SIZES - 1)
        host[n], tmp[n] = [n], [n]
    assert list(host) == [64, 128] and list(tmp) == [64, 128]


def test_compiled_stage_build_failure_falls_back_to_eager(monkeypatch):
    from torch._dynamo.exc import BackendCompilerFailed

    m = _stub()
    tp2.install(m, CPU2)
    stack = m.transformer_blocks[0]
    monkeypatch.setenv(tp2.TP_COMPILE_ENV, "1")
    exc = BackendCompilerFailed.__new__(BackendCompilerFailed)
    Exception.__init__(exc, "inductor lowering failed")

    def failing(*a, **k):
        raise exc

    monkeypatch.setattr(torch, "compile", lambda fn, *a, **k: failing)
    stack.compile(fullgraph = True)
    assert stack.compiled and stack._stage_attn is not tp2._attn_half
    guarded = stack._guarded(failing, lambda *a, **k: "eager")
    assert guarded("blk") == "eager"
    assert stack.compiled is False and "lowering" in stack.compile_error
    assert stack._stage_attn is tp2._attn_half and stack._stage_mlp is tp2._mlp_half

    def runtime_error(*a, **k):
        raise ValueError("a kernel bug, not a compile failure")

    stack.compiled = True
    with pytest.raises(ValueError):
        stack._guarded(runtime_error, lambda *a, **k: "eager")("blk")


# ----------------------------------------------------------------------------------- CFG / LoRA / INT8 attention


def test_cfg_replica_and_lora_decline_a_tensor_parallel_denoiser():
    from core.inference import diffusion_multigpu as mg

    m = _stub()
    tp2.install(m, CPU2)
    why = mg.cfg_parallel_decline(
        SimpleNamespace(transformer = m),
        offload_policy = "none",
        selection = mg.MultiGpuSelection(primary = 0, secondaries = (1,)),
    )
    assert why is not None and "tensor parallel" in why
    with pytest.raises(RuntimeError, match = "tensor-parallel"):
        mg.build_replica(m, "cpu")
    assert tp2.TP_ENV in tp2.lora_block_reason(SimpleNamespace(transformer = m))
    assert tp2.lora_block_reason(SimpleNamespace(transformer = _stub())) is None


def test_int8_dispatch_discovery_reaches_the_shards_at_their_head_count():
    from core.inference import diffusion_int8_attention_dispatch as disp

    m = _stub()
    assert [mod.heads for mod in disp.routable_modules(m)] == [HEADS, HEADS]
    tp2.install(m, CPU2)
    found = disp.routable_modules(m)
    # one shard per block (both cards share each block's processor), raced at the per-card head count
    assert [mod.heads for mod in found] == [HEADS // 2, HEADS // 2]


def test_release_kv_frees_the_per_card_caches():
    m = _stub()
    tp2.install(m, CPU2)
    stack = m.transformer_blocks[0]

    class Key:
        pass

    key = Key()
    layers = [
        [SimpleNamespace(k = torch.ones(4), v = torch.ones(4)) for _ in range(2)] for _ in range(2)
    ]
    stack._kv[key] = layers
    assert tp2.release_kv(m) == 8 * 4 * 4
    assert all(layer.k is None and layer.v is None for card in layers for layer in card)
