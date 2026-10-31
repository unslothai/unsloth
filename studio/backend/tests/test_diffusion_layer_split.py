# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer split of one denoiser across cards (core/inference/diffusion_layer_split.py), CPU only.

The planner runs on real module sizes. The boundary is exercised with the blocks assigned to ``cuda:0`` / ``cuda:1``
while their weights stay on the CPU: the cross-card copy is replaced by a recorder, which checks where every block's
inputs and outputs would go without a second card."""

from __future__ import annotations

import contextlib
import gc

import pytest
import torch
from torch import nn

from core.inference import diffusion_layer_split as ls
from core.inference import diffusion_multigpu as mg


class FakeBlock(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states,
        temb,
        rotary = None,
    ):
        out = self.lin(hidden_states) * temb
        if rotary is not None:
            out = out + rotary[0]
        return out


class FakeSingle(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.inner = FakeBlock(dim)  # a repeated class nested in a block stays with its block

    def forward(self, hidden_states, temb):
        return self.inner(self.lin(hidden_states), temb)


class FakeDiT(nn.Module):
    _repeated_blocks = ["FakeBlock", "FakeSingle"]

    def __init__(
        self,
        dim: int = 8,
        double: int = 3,
        single: int = 3,
    ):
        super().__init__()
        self.x_embedder = nn.Linear(dim, dim)
        self.transformer_blocks = nn.ModuleList(FakeBlock(dim) for _ in range(double))
        self.single_transformer_blocks = nn.ModuleList(FakeSingle(dim) for _ in range(single))
        self.proj_out = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states,
        controlnet_block_samples = None,
    ):
        h = self.x_embedder(hidden_states)
        temb = torch.ones_like(h[:1])
        rotary = (torch.zeros_like(h), torch.zeros_like(h))
        for i, block in enumerate(self.transformer_blocks):
            h = block(h, temb, rotary)
            if controlnet_block_samples is not None:
                h = h + controlnet_block_samples[i]
        for block in self.single_transformer_blocks:
            h = block(h, temb)
        return self.proj_out(h)


LazyDiT = type("Flux2Transformer2DModel", (FakeDiT,), {})


class Pipe:
    def __init__(self, transformer):
        self.transformer = transformer
        self.vae = nn.Linear(2, 2)

    @property
    def components(self):
        return {"transformer": self.transformer, "vae": self.vae}


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(ls.LAYER_SPLIT_ENV, raising = False)
    monkeypatch.delenv(mg.MULTIGPU_ENV, raising = False)
    monkeypatch.delenv(mg.CFG_PARALLEL_ENV, raising = False)
    yield


def _mib(module) -> int:
    return ls.module_bytes(module) / (1 << 20)


def test_mode_parsing(monkeypatch):
    assert ls.layer_split_mode() == "auto"
    for raw in ("0", "off", "False"):
        monkeypatch.setenv(ls.LAYER_SPLIT_ENV, raw)
        assert ls.layer_split_mode() == "off"
    monkeypatch.setenv(ls.LAYER_SPLIT_ENV, "force")
    assert ls.layer_split_mode() == "force"
    monkeypatch.setenv(mg.MULTIGPU_ENV, "0")
    assert ls.layer_split_mode() == "off"


def test_find_blocks_in_registration_order_and_skips_nested():
    dit = FakeDiT()
    found = ls.find_blocks(dit)
    assert [q for q, _, _ in found] == [
        "transformer_blocks.0",
        "transformer_blocks.1",
        "transformer_blocks.2",
        "single_transformer_blocks.0",
        "single_transformer_blocks.1",
        "single_transformer_blocks.2",
    ]
    assert {g for _, g, _ in found} == {"transformer_blocks", "single_transformer_blocks"}


def test_find_blocks_follows_the_forward_order_of_a_listed_class(monkeypatch):
    class ModeDiT(nn.Module):
        _repeated_blocks = ["FakeBlock"]

        def __init__(self):
            super().__init__()
            self.mode_blocks = nn.ModuleDict(
                {"image": nn.ModuleList(FakeBlock() for _ in range(2))}
            )
            self.txt_mode_blocks = nn.ModuleList(FakeBlock() for _ in range(2))

    monkeypatch.setitem(ls.FORWARD_ORDER, "ModeDiT", ("txt_mode_blocks", "mode_blocks.image"))
    assert [q for q, _, _ in ls.find_blocks(ModeDiT())] == [
        "txt_mode_blocks.0",
        "txt_mode_blocks.1",
        "mode_blocks.image.0",
        "mode_blocks.image.1",
    ]


def test_flux3_forward_order_names_its_real_lists():
    try:
        from diffusers.models.transformers import transformer_flux3
    except Exception:  # noqa: BLE001 - this diffusers has no FLUX.3
        pytest.skip("diffusers without FLUX.3")
    import inspect

    src = inspect.getsource(transformer_flux3)
    for group in ls.FORWARD_ORDER["Flux3Transformer2DModel"]:
        head = group.split(".")[0]
        assert f"self.{head}" in src


def test_a_wrapped_list_entry_groups_with_its_list():
    assert ls._block_list_name("double_stream_blocks.3.block") == "double_stream_blocks"
    assert ls._block_list_name("transformer_blocks.7") == "transformer_blocks"
    assert ls._block_list_name("mode_blocks.image.2") == "mode_blocks.image"
    assert ls._block_list_name("stages.1.blocks.0") == "stages.1.blocks"

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = FakeBlock(8)

    class Wrapped(nn.Module):
        _repeated_blocks = ["FakeBlock"]

        def __init__(self):
            super().__init__()
            self.double = nn.ModuleList([Wrap() for _ in range(3)])
            self.single = nn.ModuleList([Wrap() for _ in range(2)])

    groups = [g for _q, g, _b in ls.find_blocks(Wrapped())]
    assert groups == ["double"] * 3 + ["single"] * 2


def test_module_bytes_counts_parameters_and_buffers_once():
    lin = nn.Linear(4, 4)
    lin.register_buffer("scale", torch.ones(3, dtype = torch.float16))
    lin.alias = lin.weight  # the same tensor twice
    assert ls.module_bytes(lin) == (16 + 4) * 4 + 3 * 2


def test_plan_fill_packs_the_primary_first():
    dit = FakeDiT(dim = 1024, double = 4, single = 4)
    block = _mib(dit.transformer_blocks[0])
    single = _mib(dit.single_transformer_blocks[0])
    rest = _mib(dit) - 4 * block - 4 * single
    budget0 = int(rest + 3 * block + 0.5 * block + 1)
    plan = ls.plan_layer_split(dit, primary = 0, cards = [1], budgets_mib = {0: budget0, 1: 100})
    assert plan.primary == "cuda:0"
    assert [r.device for r in plan.ranges] == ["cuda:0", "cuda:1"]
    assert plan.ranges[0].blocks == [f"transformer_blocks.{i}" for i in range(3)]
    assert plan.ranges[1].blocks[0] == "transformer_blocks.3"
    assert len(plan.ranges[1].blocks) == 5
    assert set(plan.assignment.values()) == {"cuda:0", "cuda:1"}
    plan = ls.plan_layer_split(
        dit, primary = "cuda:0", cards = ["cuda:1"], budgets_mib = {"cuda:0": 100, "cuda:1": 100}
    )
    assert [r.device for r in plan.ranges] == ["cuda:0"]


def test_plan_balance_spreads_by_budget():
    dit = FakeDiT(dim = 1024, double = 4, single = 4)
    plan = ls.plan_layer_split(
        dit, primary = 0, cards = [1], budgets_mib = {0: 100, 1: 100}, strategy = "balance"
    )
    counts = [len(r.blocks) for r in plan.ranges]
    assert counts and abs(counts[0] - counts[1]) <= 2 and sum(counts) == 8


def test_plan_refusals():
    dit = FakeDiT(dim = 1024, double = 4, single = 4)
    with pytest.raises(ls.LayerSplitError, match = "two cards"):
        ls.plan_layer_split(dit, primary = 0, cards = [0], budgets_mib = {0: 100})
    with pytest.raises(ls.LayerSplitError, match = "no budget"):
        ls.plan_layer_split(dit, primary = 0, cards = [1], budgets_mib = {0: 100})
    with pytest.raises(ls.LayerSplitError, match = "non-block"):
        ls.plan_layer_split(dit, primary = 0, cards = [1], budgets_mib = {0: 0, 1: 100})
    with pytest.raises(ls.LayerSplitError, match = "blocks need"):
        ls.plan_layer_split(dit, primary = 0, cards = [1], budgets_mib = {0: 10, 1: 1})
    with pytest.raises(ls.LayerSplitError, match = "strategy"):
        ls.plan_layer_split(dit, primary = 0, cards = [1], budgets_mib = {0: 100, 1: 100}, strategy = "x")


def test_refusal_reasons():
    assert ls.layer_split_refusal(None) == "no denoiser module"
    dit = FakeDiT()
    assert ls.layer_split_refusal(dit) is None
    dit.transformer_blocks[0]._hf_hook = object()
    assert "accelerate" in ls.layer_split_refusal(dit)
    dit = FakeDiT()
    dit.compile()
    assert ls.layer_split_refusal(dit) == "denoiser is compiled whole"
    assert "fewer than two" in ls.layer_split_refusal(FakeDiT(double = 1, single = 0))


def test_off_switch_refuses(monkeypatch):
    monkeypatch.setenv(ls.LAYER_SPLIT_ENV, "0")
    with pytest.raises(ls.LayerSplitError, match = ls.LAYER_SPLIT_ENV):
        ls.place_layer_split(Pipe(FakeDiT()), primary = 0, cards = [1], budgets_mib = {0: 9, 1: 9})


class _Recorder:
    """Stands in for the cross-card copies: records (what, device) and leaves the CPU tensors where they are."""

    def __init__(
        self,
        monkeypatch,
        simulate_cards = True,
    ):
        self.out_moves: list = []
        self.in_moves: list = []
        real = mg.move_to_device

        def fake_move(
            obj,
            device,
            _depth = 0,
            *,
            tensor_fn = None,
            objects = True,
        ):
            assert (
                objects is False
            )  # a split never copies attribute holders (KV caches keep their identity)
            if tensor_fn is None:
                self.out_moves.append(str(device))
            else:

                def counted(t):
                    self.in_moves.append(str(device))
                    return t

                return real(obj, torch.device("meta"), _depth, tensor_fn = counted, objects = False)
            return obj

        monkeypatch.setattr(mg, "move_to_device", fake_move)
        monkeypatch.setattr(mg, "_device_context", lambda device: contextlib.nullcontext())
        if simulate_cards:
            monkeypatch.setattr(ls, "_probe_device", lambda probe, default: default)


def _install(dit, budgets):
    plan = ls.plan_layer_split(dit, primary = 0, cards = [1], budgets_mib = budgets)
    split = ls._Split(dit, torch.device("cuda", 0), type(dit).__name__ in ls.LAZY_CLASSES)
    ls._install_boundary(dit, plan, ls.find_blocks(dit), split)
    return plan, split


def _two_card_budget(dit):
    block = _mib(dit.transformer_blocks[0])
    rest = _mib(dit) - sum(_mib(b) for _, _, b in ls.find_blocks(dit))
    return {0: int(rest + 2 * block + 0.5 * block + 1), 1: 100}


def test_lazy_routing_moves_only_at_card_and_list_boundaries(monkeypatch):
    torch.manual_seed(0)
    dit = LazyDiT(dim = 1024, double = 4, single = 3)
    x = torch.randn(2, 1024)
    with torch.no_grad():
        reference = dit(x)
    plan, split = _install(dit, _two_card_budget(dit))
    assert plan.ranges[0].blocks == ["transformer_blocks.0", "transformer_blocks.1"]
    rec = _Recorder(monkeypatch)
    with torch.no_grad():
        out = dit(x)
    assert torch.equal(out, reference)
    assert rec.out_moves == ["cuda:1", "cuda:0", "cuda:0"]
    assert split.lazy_calls == 1 and split.eager_calls == 0
    assert not split.lazy_now  # reset after the forward


def test_eager_routing_with_a_residual_kwarg(monkeypatch):
    torch.manual_seed(0)
    dit = LazyDiT(dim = 1024, double = 4, single = 3)
    x = torch.randn(2, 1024)
    residual = [torch.randn(2, 1024) for _ in range(4)]
    with torch.no_grad():
        reference = dit(x, controlnet_block_samples = residual)
    _, split = _install(dit, _two_card_budget(dit))
    rec = _Recorder(monkeypatch)
    with torch.no_grad():
        out = dit(x, controlnet_block_samples = residual)
    assert torch.equal(out, reference)
    assert rec.out_moves == ["cuda:0"] * 5
    assert split.eager_calls == 1


def test_unlisted_class_is_always_eager(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    _, split = _install(dit, _two_card_budget(dit))
    rec = _Recorder(monkeypatch)
    with torch.no_grad():
        dit(torch.randn(2, 1024))
    assert rec.out_moves == ["cuda:0"] * 5
    assert split.lazy_calls == 0


def test_class_swap_keeps_names_and_undo_restores(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe = Pipe(dit)
    budgets = _two_card_budget(dit)
    monkeypatch.setattr(ls, "_move_weights", lambda *a, **k: None)
    info = ls.place_layer_split(pipe, primary = 0, cards = [1], budgets_mib = budgets)
    assert [r["count"] for r in info["ranges"]] == [2, 5]
    block = dit.transformer_blocks[3]
    assert type(block).__name__ == "FakeBlock" and type(block) is not FakeBlock
    assert isinstance(block, FakeBlock)
    from core.inference.diffusion_block_graph import repeated_blocks

    assert len(repeated_blocks(dit)) == len(
        list(m for m in dit.modules() if isinstance(m, (FakeBlock, FakeSingle)))
    )
    assert ls.is_layer_split(dit) and mg.is_split(pipe)
    assert dit._unsloth_split_home == torch.device("cuda", 0)
    status = mg.placement_status(pipe)
    assert status["placement"]["transformer"] == "cuda:0+cuda:1"
    assert status["layer_split"]["per_card_mib"].keys() == {"cuda:0", "cuda:1"}
    assert status["layer_split"]["calls"] == 0
    ls.place_layer_split(pipe, primary = 0, cards = [1], budgets_mib = budgets)
    assert type(block).__mro__[1] is FakeBlock

    assert ls.undo_layer_split(pipe)
    assert all(type(b) in (FakeBlock, FakeSingle) for _, _, b in ls.find_blocks(dit))
    assert not dit._forward_pre_hooks and not dit._forward_hooks
    assert "_unsloth_split_home" not in dit.__dict__
    assert not ls.is_layer_split(dit) and not mg.is_split(pipe)
    assert ls.undo_layer_split(pipe) is False


def test_undo_keeps_a_component_split_record(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe = Pipe(dit)
    monkeypatch.setattr(ls, "_move_weights", lambda *a, **k: None)
    mg._PLACEMENTS[pipe] = {
        "placement": {"vae": "cuda:1", "transformer": "cuda:0"},
        "moved": ["vae"],
    }
    dit._unsloth_split_home = torch.device("cuda", 0)
    ls.place_layer_split(pipe, primary = 0, cards = [1], budgets_mib = _two_card_budget(dit))
    assert mg.placement_status(pipe)["placement"]["vae"] == "cuda:1"
    ls.undo_layer_split(pipe)
    record = mg._PLACEMENTS[pipe]
    assert "layer_split" not in record and record["moved"] == ["vae"]
    assert dit._unsloth_split_home == torch.device("cuda", 0)  # the component split's marker stays


def test_a_split_denoiser_is_freed_without_the_cycle_collector(monkeypatch):
    import weakref

    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe = Pipe(dit)
    monkeypatch.setattr(ls, "_move_weights", lambda *a, **k: None)
    ls.place_layer_split(pipe, primary = 0, cards = [1], budgets_mib = _two_card_budget(dit))
    refs = [weakref.ref(dit), weakref.ref(dit.single_transformer_blocks[2])]
    gc.disable()
    try:
        del dit, pipe
        assert all(
            r() is None for r in refs
        )  # the blocks' weights go with the pipeline, as on one card
    finally:
        gc.enable()


def test_memo_copies_once_and_drops_stale_entries():
    memo = ls._Memo()
    t = torch.ones(4)
    a = memo.get(t, torch.float64)
    assert a.dtype == torch.float64
    assert memo.get(t, torch.float64) is a
    t.add_(1)  # an in-place write invalidates the copy
    b = memo.get(t, torch.float64)
    assert b is not a and torch.equal(b, torch.full((4,), 2.0, dtype = torch.float64))
    assert len(memo.entries) == 1
    del t, a, b
    gc.collect()
    assert not memo.entries  # the source died: its copies go with it
    with torch.inference_mode():
        u = torch.ones(2)
        assert memo.get(u, torch.float64) is memo.get(u, torch.float64)


def test_move_to_device_tensor_fn_reaches_nested_tensors():
    seen = []

    def fn(t):
        seen.append(t)
        return t * 2

    t = torch.ones(1)
    out = mg.move_to_device({"a": (t, [t])}, torch.device("meta"), tensor_fn = fn)
    assert len(seen) == 2 and torch.equal(out["a"][0], torch.full((1,), 2.0))


def test_move_to_device_keeps_attribute_holders_when_asked():
    class LayerCache:
        def __init__(self):
            self.k = torch.ones(1)

    cache = LayerCache()
    moved = mg.move_to_device(
        {"c": cache}, torch.device("meta"), tensor_fn = lambda t: t, objects = False
    )
    assert moved["c"] is cache
    assert mg.move_to_device(cache, torch.device("meta"), tensor_fn = lambda t: t) is not cache


def test_cfg_parallel_declines_on_a_split(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe = Pipe(dit)
    monkeypatch.setattr(ls, "_move_weights", lambda *a, **k: None)
    ls.place_layer_split(pipe, primary = 0, cards = [1], budgets_mib = _two_card_budget(dit))
    sel = mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    monkeypatch.setenv(mg.CFG_PARALLEL_ENV, "force")
    assert "layer split" in mg.cfg_parallel_decline(pipe, offload_policy = "none", selection = sel)
    from core.inference.diffusion_block_graph import install_block_graphs

    handle, reason = install_block_graphs(dit)
    assert handle is None and "weights off the GPU" in reason
    ls.undo_layer_split(pipe)


def test_a_whole_step_graph_armed_before_the_split_stays_eager(monkeypatch):
    from core.inference import diffusion_cuda_graph as cg

    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe = Pipe(dit)
    monkeypatch.setattr(ls, "_move_weights", lambda *a, **k: None)
    monkeypatch.setattr(cg, "_denoiser_modules", lambda pipe: [pipe.transformer])
    early = cg.GraphedForward(dit).enable()
    ls.place_layer_split(pipe, primary = 0, cards = [1], budgets_mib = _two_card_budget(dit))
    seen = []
    monkeypatch.setattr(early, "_eager", lambda args, kwargs: seen.append(1) or "eager")
    assert early(torch.zeros(1), return_dict = False) == "eager" and seen
    assert not early.poisoned and not early.stats.get("refused_host_tensor")
    early.disable()
    handles, reason = cg.arm_after_placement(pipe)
    assert handles == () and "split across cards" in reason
    ls.undo_layer_split(pipe)


def test_a_positional_residual_runs_eager(monkeypatch):
    torch.manual_seed(0)
    dit = LazyDiT(dim = 1024, double = 4, single = 3)
    x = torch.randn(2, 1024)
    residual = [torch.randn(2, 1024) for _ in range(4)]
    with torch.no_grad():
        reference = dit(x, residual)
    _, split = _install(dit, _two_card_budget(dit))
    rec = _Recorder(monkeypatch)
    with torch.no_grad():
        out = dit(x, residual)
    assert torch.equal(out, reference)
    assert rec.out_moves == ["cuda:0"] * 5
    assert split.eager_calls == 1 and split.lazy_calls == 0
    with torch.no_grad():
        dit(x, None)
    assert split.lazy_calls == 1


def test_routes_follow_a_relocated_denoiser(monkeypatch):
    torch.manual_seed(0)
    dit = LazyDiT(dim = 1024, double = 4, single = 3)
    pipe = Pipe(dit)
    x = torch.randn(2, 1024)
    with torch.no_grad():
        reference = dit(x)
    monkeypatch.setattr(ls, "_move_weights", lambda *a, **k: None)
    ls.place_layer_split(pipe, primary = 0, cards = [1], budgets_mib = _two_card_budget(dit))
    real = mg.move_to_device
    moves: list = []

    def spy(
        obj,
        device,
        _depth = 0,
        *,
        tensor_fn = None,
        objects = True,
    ):
        moves.append(str(device))
        return real(obj, device, _depth, tensor_fn = tensor_fn, objects = objects)

    monkeypatch.setattr(mg, "move_to_device", spy)
    with torch.no_grad():
        out = dit(x)  # on the CPU: a route still aimed at cuda:1 would fail here
    assert torch.equal(out, reference)
    assert set(moves) <= {"cpu"}
    assert dit._unsloth_split_home == torch.device("cuda", 0)
    assert mg._split_home(pipe) == torch.device("cpu")
    ls.undo_layer_split(pipe)


def test_routes_follow_one_relocated_block(monkeypatch):
    torch.manual_seed(0)
    dit = LazyDiT(dim = 1024, double = 4, single = 3)
    _install(dit, _two_card_budget(dit))
    moved = dit.transformer_blocks[3]
    where = {id(moved): torch.device("cuda", 0)}

    def probe_device(probe, default):
        owner = probe[0]() if probe is not None else None
        for module in (owner,):
            for block in dit.modules():
                if (
                    block is moved
                    and owner is not None
                    and any(m is owner for m in block.modules())
                ):
                    return where[id(moved)]
        return default

    rec = _Recorder(monkeypatch)
    monkeypatch.setattr(ls, "_probe_device", probe_device)
    with torch.no_grad():
        dit(torch.randn(2, 1024))
    assert rec.out_moves == ["cuda:1", "cuda:0", "cuda:0"]


def test_generic_undo_split_removes_a_layer_split(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe = Pipe(dit)
    monkeypatch.setattr(ls, "_move_weights", lambda *a, **k: None)
    ls.place_layer_split(pipe, primary = 0, cards = [1], budgets_mib = _two_card_budget(dit))
    assert mg.is_split(pipe)
    mg.undo_split(pipe)
    assert not ls.is_layer_split(dit) and not mg.is_split(pipe)
    assert all(type(b) in (FakeBlock, FakeSingle) for _, _, b in ls.find_blocks(dit))
    assert not dit._forward_pre_hooks and not dit._forward_hooks
    assert pipe not in mg._PLACEMENTS


def test_eager_calls_keep_no_copy_across_blocks(monkeypatch):
    # inference tensors carry no version: an eager call must not reuse an earlier block's copy; a lazy call shares one per card
    class InPlaceDiT(FakeDiT):
        def forward(
            self,
            hidden_states,
            controlnet_block_samples = None,
        ):
            temb = torch.ones_like(hidden_states[:1])
            h = hidden_states
            for block in self.transformer_blocks:
                h = block(h, temb)
                temb.add_(1)  # written in place on the primary between blocks
            for block in self.single_transformer_blocks:
                h = block(h, temb)
            return h

    hits: list = []

    def fake_move(
        obj,
        device,
        _depth = 0,
        *,
        tensor_fn = None,
        objects = True,
    ):
        if tensor_fn is None:
            return obj

        def walk(o):
            if isinstance(o, torch.Tensor):
                return tensor_fn(o)
            if isinstance(o, (list, tuple)):
                return type(o)(walk(v) for v in o)
            if isinstance(o, dict):
                return {k: walk(v) for k, v in o.items()}
            return o

        return walk(obj)

    def fake_get(self, t, device):
        hits.append(id(t) in self.entries)  # a copy made for an earlier block would be served again
        self.entries[id(t)] = (None, 0, None, {device: t})
        return t

    for cls, lazy in (
        (InPlaceDiT, False),
        (type("Flux2Transformer2DModel", (InPlaceDiT,), {}), True),
    ):
        torch.manual_seed(0)
        dit = cls(dim = 64, double = 4, single = 3)
        x = torch.randn(2, 64)
        with torch.inference_mode():
            reference = dit(x)
        _, split = _install(dit, {0: 10**6, 1: 10**6})
        monkeypatch.setattr(mg, "move_to_device", fake_move)
        monkeypatch.setattr(mg, "_device_context", lambda device: contextlib.nullcontext())
        monkeypatch.setattr(ls, "_probe_device", lambda probe, default: default)
        monkeypatch.setattr(split, "home", torch.device("cpu"))
        for qual, _, block in ls.find_blocks(dit):
            block.__dict__[ls._SPLIT_ATTR].device = torch.device("cuda", 1)
        monkeypatch.setattr(ls._Memo, "get", fake_get)
        hits.clear()
        with torch.inference_mode():
            out = dit(x)
        assert torch.equal(out, reference)
        if lazy:
            assert any(hits)  # temb is copied once per card and shared by the blocks
        else:
            assert hits and not any(hits)  # every block re-copies what it is given
        monkeypatch.undo()


def test_flex_processor_refuses_and_block_mask_moves():
    class FlexAttnProcessor:
        pass

    dit = FakeDiT()
    dit.attn_processors = {"transformer_blocks.0.attn.processor": FlexAttnProcessor()}
    assert "flex attention" in ls.layer_split_refusal(dit)

    class BlockMask:
        def __init__(self, device):
            self.device = device

        def to(self, device):
            return BlockMask(device)

    mask = BlockMask("cpu")
    moved = mg.move_to_device({"attention_mask": mask}, torch.device("meta"), objects = False)
    assert moved["attention_mask"].device == torch.device("meta")

    class KVCache:  # an attribute holder the callee writes into keeps its identity
        pass

    cache = KVCache()
    assert (
        mg.move_to_device({"cache": cache}, torch.device("meta"), objects = False)["cache"] is cache
    )


def test_graph_eligibility_refuses_a_split_unet(monkeypatch):
    from core.inference import diffusion_cuda_graph as cg

    unet = FakeDiT(dim = 1024, double = 4, single = 3)
    monkeypatch.setattr(cg, "_denoiser_modules", lambda pipe: [unet])
    from core.inference import diffusion_speed

    monkeypatch.setattr(diffusion_speed, "_denoiser_dits", lambda pipe: [])
    monkeypatch.setattr(diffusion_speed, "_denoiser_unet", lambda pipe: unet)
    pipe = Pipe(unet)
    monkeypatch.setattr(ls, "_move_weights", lambda *a, **k: None)
    ls.place_layer_split(pipe, primary = 0, cards = [1], budgets_mib = _two_card_budget(unet))
    target = type("T", (), {"device": "cuda", "backend": "cuda"})()
    monkeypatch.setattr(cg, "cuda_graph_disabled", lambda: False)
    ok, reason = cg.graph_eligible(
        target,
        family = None,
        pipe = pipe,
        offload_active = False,
        cache_active = False,
        speed_mode = "default",
    )
    assert not ok and "split" in reason
    ls.undo_layer_split(pipe)
