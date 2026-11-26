# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""TP-2 for Krea 2 (diffusion_tp2): a tiny real Krea2Transformer2DModel (GQA, sigmoid output gate, SwiGLU) split over
two CPU "cards" matches the unsplit model, with the stock processor, the prefix K/V processor and the small-host
int8 Linears; an Automatic plan that streams the denoiser engages TP with the env unset, everything else keeps it."""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "Krea2Transformer2DModel"):
    pytest.skip("diffusers without Krea 2", allow_module_level = True)

from core.inference import diffusion_krea2_prefix_kv as pkv  # noqa: E402
from core.inference import diffusion_small_host as small_host  # noqa: E402
from core.inference import diffusion_tp2 as tp2  # noqa: E402

CPU2 = [torch.device("cpu"), torch.device("cpu")]
TEXT, GRID, LAYERS, DIM, BLOCKS = 12, 4, 3, 32, 3


def _model():
    torch.manual_seed(0)
    m = diffusers.Krea2Transformer2DModel(
        in_channels = 8,
        num_layers = BLOCKS,
        attention_head_dim = 16,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        intermediate_size = 96,
        timestep_embed_dim = 16,
        text_hidden_dim = DIM,
        num_text_layers = LAYERS,
        text_num_attention_heads = 2,
        text_num_key_value_heads = 2,
        text_intermediate_size = 32,
        num_layerwise_text_blocks = 1,
        num_refiner_text_blocks = 1,
        axes_dims_rope = (4, 6, 6),
    ).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    return m


def _positions(grid = GRID):
    """``Krea2Pipeline.prepare_position_ids``' layout: text rows at the origin, image rows ``(0, h, w)``."""
    hh, ww = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing = "ij")
    img = torch.stack([torch.zeros_like(hh), hh, ww], dim = -1).reshape(-1, 3)
    return torch.cat([torch.zeros(TEXT, 3, dtype = img.dtype), img])


def _inputs(valid = 7, grid = None):
    grid = GRID if grid is None else grid
    g = torch.Generator().manual_seed(1)
    mask = torch.zeros(1, TEXT, dtype = torch.bool)
    mask[:, :valid] = True
    return {
        "hidden_states": torch.randn(1, grid * grid, 8, generator = g),
        "encoder_hidden_states": torch.randn(1, TEXT, LAYERS, DIM, generator = g),
        "timestep": torch.full((1,), 0.5),
        "position_ids": _positions(grid),
        "encoder_attention_mask": mask,
        "return_dict": False,
    }


def _run(m, **kw):
    with torch.no_grad():
        return m(**_inputs(**kw))[0]


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(tp2.TP_ENV, raising = False)
    monkeypatch.delenv(pkv.PREFIX_ENV, raising = False)


def test_krea2_tp2_matches_one_card():
    ref = _model()
    tp = copy.deepcopy(ref)
    assert tp2.refusal(tp, CPU2) is None
    rec = tp2.install(tp, CPU2)
    assert rec["blocks"] == BLOCKS and rec["comm"] == "cpu"
    shard = tp.transformer_blocks[0].shards()[1][0]
    a = shard.attn
    assert (a.num_heads, a.num_kv_heads, a.head_dim) == (2, 1, 16)
    assert a.to_q.out_features == 32 and a.to_k.out_features == 16 and a.to_gate.out_features == 32
    assert a.to_out[0].in_features == 32 and shard.ff.down.in_features == 48
    for valid in (7, TEXT):
        torch.testing.assert_close(
            _run(tp, valid = valid), _run(ref, valid = valid), rtol = 1e-5, atol = 1e-5
        )
    assert tp2.status(tp)["calls"] == 2


def test_krea2_tp2_with_prefix_kv_processor():
    ref = _model()
    assert pkv.install(ref) == BLOCKS
    tp = copy.deepcopy(ref)
    assert tp2.refusal(tp, CPU2) is None
    tp2.install(tp, CPU2)
    # both cards' shards keep the shared prefix processor
    procs = {id(c[0].attn.processor) for c in tp.transformer_blocks[0].shards()}
    assert len(procs) == 1
    assert type(tp.transformer_blocks[0].shards()[0][0].attn.processor).__name__ == (
        "Krea2PrefixKVAttnProcessor"
    )
    torch.testing.assert_close(_run(tp), _run(ref), rtol = 1e-5, atol = 1e-5)


def test_krea2_tp2_small_host_int8_linears_shard_exactly(monkeypatch):
    monkeypatch.setattr(small_host, "_INT8_MIN_ELEMENTS", 1)
    ref = _model()
    small_host.quantize_int8_weight_(
        ref, compute_dtype = torch.float32, work_device = "cpu", keep_device = "cpu"
    )
    cls = small_host.int8_linear_class()
    blk = ref.transformer_blocks[0]
    assert isinstance(blk.attn.to_q, cls) and isinstance(blk.ff.down, cls)
    tp = copy.deepcopy(ref)
    assert tp2.refusal(tp, CPU2) is None
    tp2.install(tp, CPU2)
    s0, s1 = tp.transformer_blocks[0].shards()[0][0], tp.transformer_blocks[0].shards()[1][0]
    # column split: rows and their scales; row split: input channels, every row's scale kept
    torch.testing.assert_close(
        torch.cat([s0.ff.up.qweight, s1.ff.up.qweight]), blk.ff.up.qweight, rtol = 0, atol = 0
    )
    torch.testing.assert_close(
        torch.cat([s0.ff.up.scale, s1.ff.up.scale]), blk.ff.up.scale, rtol = 0, atol = 0
    )
    down0, down1 = s0.ff.down.inner, s1.ff.down.inner
    assert isinstance(down0, cls) and down0.in_features == 48
    torch.testing.assert_close(
        torch.cat([down0.qweight, down1.qweight], dim = 1), blk.ff.down.qweight, rtol = 0, atol = 0
    )
    torch.testing.assert_close(down1.scale, blk.ff.down.scale, rtol = 0, atol = 0)
    torch.testing.assert_close(_run(tp), _run(ref), rtol = 1e-5, atol = 1e-5)


def test_int8_linears_still_refused_for_qwen_image_21():
    """The small-host int8 shard is verified for Krea 2 only; Qwen-Image-2.1 keeps its refusal."""
    lin = small_host.int8_linear_class()(
        torch.zeros(4, 4, dtype = torch.int8), torch.ones(4, 1), None, 4, 4
    )
    assert "Int8WeightLinear" in tp2._linear_refusal(lin, "attn.to_q")
    assert tp2._linear_refusal(lin, "attn.to_q", tp2._KREA2_ROW, int8w_ok = True) is None


def test_krea2_refusals():
    m = _model()
    m.transformer_blocks[1].attn.processor = object()
    assert "not a verified Krea 2 attention processor" in tp2.refusal(m, CPU2)
    m = _model()
    m.transformer_blocks[0].ff.down = torch.nn.Conv1d(2, 2, 1)
    assert "Conv1d" in tp2.refusal(m, CPU2)
    m = _model()
    m.register_to_config(num_key_value_heads = 3)
    assert "do not split" in tp2.refusal(m, CPU2)


@dataclasses.dataclass(frozen = True)
class _Plan:
    requested_mode: str = "auto"
    offload_policy: str = "group"
    stream_transformer: bool = True
    stream_text_encoders: bool = True
    resident_transformer_mib: int = 3262
    secondary_ordinal: object = None
    secondary_components: object = None
    layer_split: object = None
    fallback_plan: object = None
    reasons: tuple = ("planned",)


def _two_cards(
    monkeypatch,
    seen,
    denoiser_mib = 12245,
):
    from core.inference import diffusion_multigpu as mg

    # the Kaggle T4 load: 12245 MiB of int8 denoiser, 3262 MiB of it resident, the rest streamed every step
    monkeypatch.setattr(tp2, "_denoiser_mib", lambda tr: denoiser_mib, raising = False)

    monkeypatch.setattr(
        mg, "current_selection", lambda: mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    )
    monkeypatch.setattr(tp2, "refusal", lambda tr, devs: None)
    monkeypatch.setattr(tp2, "fit_refusal", lambda tr, devs: None)
    monkeypatch.setattr(
        tp2,
        "install",
        lambda tr, devs, comm = None, logger = None, retire = True: seen.append(devs)
        or {"devices": [str(d) for d in devs]},
    )


def test_auto_engages_krea2_when_the_automatic_plan_streams_the_denoiser(monkeypatch):
    seen = []
    _two_cards(monkeypatch, seen)
    pipe = SimpleNamespace(transformer = _model())
    out = tp2.engage_for_plan(pipe, _Plan())
    assert (
        len(seen) == 1 and out.stream_transformer is False and out.resident_transformer_mib is None
    )
    assert "tensor parallel" in out.reasons[-1]
    assert tp2.engage_for_plan(
        SimpleNamespace(transformer = _model()), _Plan(offload_policy = "streaming")
    )
    assert len(seen) == 2


@pytest.mark.parametrize(
    "env,plan",
    [
        ("0", _Plan()),  # turned off
        ("off", _Plan()),
        (None, _Plan(offload_policy = "none", stream_transformer = True)),  # fits one card
        (None, _Plan(stream_transformer = False)),  # denoiser resident, encoders stream
        (None, _Plan(layer_split = "split")),  # the layer split already places it
        (None, _Plan(offload_policy = "model")),  # whole-module offload
        (None, _Plan(requested_mode = "low_vram")),  # an explicit memory mode
    ],
)
def test_auto_keeps_the_plan_otherwise(monkeypatch, env, plan):
    seen = []
    _two_cards(monkeypatch, seen)
    if env is not None:
        monkeypatch.setenv(tp2.TP_ENV, env)
    assert tp2.engage_for_plan(SimpleNamespace(transformer = _model()), plan) is plan
    assert seen == []


def test_auto_is_krea2_only(monkeypatch):
    seen = []
    _two_cards(monkeypatch, seen)

    class QwenImage21Transformer2DModel(torch.nn.Module):
        pass

    plan = _Plan()
    assert (
        tp2.engage_for_plan(SimpleNamespace(transformer = QwenImage21Transformer2DModel()), plan)
        is plan
    )
    assert seen == []


def test_auto_on_one_card_keeps_the_plan(monkeypatch):
    from core.inference import diffusion_multigpu as mg

    calls = []
    monkeypatch.setattr(tp2, "install", lambda *a, **k: calls.append(a))
    pipe, plan = SimpleNamespace(transformer = _model()), _Plan()
    for one_card in (None, mg.MultiGpuSelection(primary = 0, secondaries = ())):
        monkeypatch.setattr(mg, "current_selection", lambda sel = one_card: sel)
        assert tp2.engage_for_plan(pipe, plan) is plan
    assert calls == [] and not tp2.is_installed(pipe.transformer)


def test_decode_lend_is_a_no_op_without_two_cuda_cards_or_for_other_dits(monkeypatch):
    m = _model()
    assert tp2.lend_for_decode(SimpleNamespace(transformer = m)) is None  # not tensor parallel
    tp2.install(m, CPU2)
    assert tp2.lend_for_decode(SimpleNamespace(transformer = m)) is None  # CPU "cards"
    assert tp2.lend_for_decode(SimpleNamespace(transformer = None)) is None
    assert tp2.status(m)["decode_lends"] == 0


def test_a_lend_left_open_is_returned_by_the_next_denoiser_call():
    ref = _model()
    tp = copy.deepcopy(ref)
    tp2.install(tp, CPU2)
    stack = tp.transformer_blocks[0]
    entries = tp2._shard_tensors(stack.shards()[0])
    assert entries and all(
        (m._parameters[n] if k == "p" else m._buffers[n]) is not None for m, k, n in entries
    )
    stack.__dict__[tp2._LENT_ATTR] = (entries, torch.device("cpu"))
    torch.testing.assert_close(_run(tp), _run(ref), rtol = 1e-5, atol = 1e-5)
    assert tp2._LENT_ATTR not in stack.__dict__


@pytest.mark.parametrize(
    "plan,denoiser_mib",
    [
        (_Plan(resident_transformer_mib = 12245), 12245),  # group policy, whole denoiser pinned
        (_Plan(resident_transformer_mib = 12200), 12245),  # within the planner-vs-storage slack
        (_Plan(offload_policy = "streaming", resident_transformer_mib = 12245), 12245),
        (_Plan(), None),  # a resident share with an unreadable denoiser size: treated as resident
    ],
)
def test_auto_keeps_a_plan_that_holds_the_whole_denoiser_resident(monkeypatch, plan, denoiser_mib):
    seen = []
    _two_cards(monkeypatch, seen, denoiser_mib)
    assert tp2.engage_for_plan(SimpleNamespace(transformer = _model()), plan) is plan
    assert seen == []


def test_auto_engages_when_every_group_streams(monkeypatch):
    seen = []
    _two_cards(monkeypatch, seen, None)
    for plan in (
        _Plan(resident_transformer_mib = None),
        _Plan(offload_policy = "streaming", resident_transformer_mib = None),
    ):
        assert tp2.engage_for_plan(SimpleNamespace(transformer = _model()), plan) is not plan
    assert len(seen) == 2


def test_int8_row_split_under_fp16_autocast_stays_finite():
    """Reported repro: int8 row [100, 100, -100, -100], unit scale, inputs of 400 under fp16 autocast. Unsplit the
    sum is 0; each half alone is +-80000, past fp16's range, so the partials must leave in fp32."""
    host = small_host
    layer = host.int8_linear_class()(
        torch.tensor([[100, 100, -100, -100]], dtype = torch.int8), torch.ones(1, 1), None, 4, 1
    )
    parts = [tp2._row_parallel_class()(tp2.shard_linear(layer, 1, i, 2, "cpu")) for i in range(2)]
    x = torch.full((1, 4), 400.0)
    with torch.inference_mode(), torch.autocast("cpu", dtype = torch.float16):
        reference = layer(x)
        partials = [part(chunk) for part, chunk in zip(parts, x.chunk(2, dim = 1))]
        result = partials[0] + partials[1]
    assert all(p.dtype == torch.float32 for p in partials)
    assert torch.isfinite(result).all() and float(result) == float(reference) == 0.0


def _lend_setup():
    tp = _model()
    tp2.install(tp, CPU2)
    stack = tp.transformer_blocks[0]
    entries = tp2._shard_tensors(stack.shards()[0])
    return tp, stack, entries


HOME, AWAY = "card0", "card1"


class _Mover:
    """Stands in for ``_move_card`` on CPU: records where each tensor would be, failing on chosen calls (the
    ``fail`` schedule: per destination, the 1-based call numbers that raise after moving ``partial`` tensors)."""

    def __init__(
        self,
        fail,
        partial = 1,
    ):
        self.where, self.calls, self.fail, self.partial = {}, {HOME: 0, AWAY: 0}, fail, partial

    def __call__(self, ents, device):
        self.calls[device] += 1
        if self.calls[device] in self.fail.get(device, ()):
            for e in ents[: self.partial if device == AWAY else 0]:
                self.where[id(e[0]), e[1], e[2]] = device
            raise torch.OutOfMemoryError(f"injected OOM moving to {device}")
        for e in ents:
            self.where[id(e[0]), e[1], e[2]] = device

    def places(self, entries):
        return {self.where.get((id(m), k, n), HOME) for m, k, n in entries}


def test_a_failed_lend_with_a_failed_rollback_is_recovered_by_the_next_call(monkeypatch):
    """Fault injection: the lend OOMs after moving one tensor, then the first rollback OOMs too. The lend stays
    recorded (never a silent mixed placement) and the next denoiser call brings every tensor home."""
    ref = _model()
    tp, stack, entries = _lend_setup()
    mover = _Mover({AWAY: (1,), HOME: (1,)})
    monkeypatch.setattr(tp2, "_move_card", mover)
    assert tp2._lend_card(stack, entries, HOME, AWAY) is False
    assert mover.places(entries) == {HOME, AWAY}  # mixed after the failed rollback ...
    assert stack.__dict__[tp2._LENT_ATTR] == (entries, HOME)  # ... but recorded
    torch.testing.assert_close(_run(tp), _run(ref), rtol = 1e-5, atol = 1e-5)
    assert mover.places(entries) == {HOME} and tp2._LENT_ATTR not in stack.__dict__


def test_a_lend_whose_return_keeps_failing_raises_instead_of_running_mixed(monkeypatch):
    tp, stack, entries = _lend_setup()
    mover = _Mover({AWAY: (1,), HOME: (1, 2)})
    monkeypatch.setattr(tp2, "_move_card", mover)
    assert tp2._lend_card(stack, entries, HOME, AWAY) is False
    with pytest.raises(torch.OutOfMemoryError):
        _run(
            tp
        )  # the second return attempt fails loudly instead of running half the blocks off their card
    assert tp2._LENT_ATTR in stack.__dict__
    _run(tp)  # the third attempt works: home again
    assert mover.places(entries) == {HOME} and tp2._LENT_ATTR not in stack.__dict__


def test_a_failed_lend_with_a_working_rollback_leaves_nothing_recorded(monkeypatch):
    tp, stack, entries = _lend_setup()
    mover = _Mover({AWAY: (1,)}, partial = 2)
    monkeypatch.setattr(tp2, "_move_card", mover)
    assert tp2._lend_card(stack, entries, HOME, AWAY) is False
    assert mover.places(entries) == {HOME} and tp2._LENT_ATTR not in stack.__dict__
    assert tp2._lend_card(stack, entries, HOME, AWAY) is True  # a clean lend records it
    assert mover.places(entries) == {AWAY} and stack.__dict__[tp2._LENT_ATTR] == (entries, HOME)


def test_outer_compile_of_a_tp_model_adds_no_recompiles():
    """Whole-model torch.compile (eager backend), 4 calls at 2 resolutions: TP compiles as many frames as one card."""
    from torch._dynamo.testing import CompileCounter

    def frames(model):
        torch._dynamo.reset()
        cnt = CompileCounter()
        fn = torch.compile(model, backend = cnt)
        with torch.no_grad():
            for grid in (GRID, GRID, GRID + 2, GRID + 2):
                fn(**_inputs(grid = grid))
        return cnt.frame_count

    ref = _model()
    tp = copy.deepcopy(ref)
    tp2.install(tp, CPU2)
    base = frames(ref)
    assert frames(tp) == base
    torch._dynamo.reset()
