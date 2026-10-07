# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""offload_layers = "auto" re-planned at Trainer init for the real batch size; helpers extracted
from _utils.py with ast, so no GPU or model is needed."""

import ast, os, types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")
NAMES = (
    "_training_reserve_bytes",
    "_auto_block_swap_indices",
    "install_block_swap",
    "_attach_block_swap",
    "_layer_devices",
    "_replan_auto_offload_safely",
    "_trainer_offload_replan_skip",
    "replan_auto_offload_for_trainer",
)
GIB = 2**30
SRC = open(UTILS, encoding = "utf-8").read()


def _load(
    *,
    free = 100 * GIB,
    per_row = GIB,
    cuda = True,
    pick = None,
):
    """`free`: bytes the card has; the fake reserve is `per_row` per batch row and every swapped
    layer frees 1 GiB, so the planner picks ceil((reserve - free) / 1 GiB) layers."""
    mod = ast.parse(SRC)
    nodes = {n.name: n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in NAMES}
    assert set(nodes) == set(NAMES), f"missing from _utils.py: {set(NAMES) - set(nodes)}"
    state = types.SimpleNamespace(
        free = free,
        free_by = {},
        homes = {},
        estimates = [],
        swaps = [],
        removed = [],
        fail = 0,
        layer_devices = {"cuda:0"},
        resident = False,
        idle_slot_bytes = 0,
    )

    def estimate(
        config,
        seq_len,
        batch_size = 1,
        extra_bytes = 0,
    ):
        state.estimates.append((seq_len, batch_size))
        return batch_size * per_row

    def auto(layers, reserve, depth):
        if pick is not None:
            return pick, 0
        need = reserve - state.free
        n = 0 if need <= 0 else min(len(layers) - 1, -(-need // GIB))
        return list(range(len(layers) - n, len(layers))), 0

    class FakeSwap:
        device = "cuda:0"

        def __init__(
            self,
            layers,
            n,
            depth,
            placement = "tail",
        ):
            if state.fail:
                state.fail -= 1
                raise RuntimeError("pinned allocation failed")
            self.indices = list(range(n)) if isinstance(n, int) else list(n)
            self.depth = depth
            state.swaps.append(self)
            state.free += len(self.indices) * GIB

        @property
        def blocks(self):
            return [
                types.SimpleNamespace(
                    home = state.homes.get(i, "cuda:0"), nbytes = lambda: GIB, resident = state.resident
                )
                for i in self.indices
            ]

        @property
        def free(self):
            buf = types.SimpleNamespace(numel = lambda: state.idle_slot_bytes, element_size = lambda: 1)
            return {"sig": [({"cuda:0": buf}, None)]} if state.idle_slot_bytes else {}

        def host_bytes(self):
            return len(self.indices) * GIB

        def remove(self):
            state.removed.append(self)
            state.free -= self.host_bytes()
            self.indices = []

    ns = {
        "is_moe_model": lambda m: False,
        "is_integrated_unified_memory_gpu": lambda: False,
        "BlockSwap": FakeSwap,
        "build_host_layers": object(),
        "find_decoder_layers": lambda m: m.layers,
        "auto_swap_indices": auto,
        "_offload_embedding_for_room": lambda model: False,
        "_check_block_swap": lambda model: None,
        "_new_block_swap": lambda layers, n, depth, placement: FakeSwap(
            layers, n, depth, placement
        ),
        "estimate_training_reserve_bytes": estimate,
        "usable_cuda_bytes": lambda device: state.free_by.get(device, state.free),
        "_AUTO_OFFLOAD_BATCH_SIZE": 2,
        "torch": types.SimpleNamespace(cuda = types.SimpleNamespace(is_available = lambda: cuda)),
    }
    for name in NAMES:
        exec(ast.get_source_segment(SRC, nodes[name]), ns)
    ns["_layer_devices"] = lambda layers: set(state.layer_devices)
    exec("_REPLAN_FAILED_PRINTED = False", ns)
    exec('_PAIRED_FORWARD_TRAINERS = ("DPOTrainer", "ORPOTrainer", "CPOTrainer")', ns)
    return ns, state


class _Layers(list):
    pass


class _Model:
    config = None
    max_seq_length = 2048

    def __init__(self, n = 8):
        self.layers = _Layers(types.SimpleNamespace(name = f"L{i}") for i in range(n))

    def parameters(self):
        return []


def _trainer(
    model,
    batch_size = 8,
    **kw,
):
    args = types.SimpleNamespace(
        per_device_train_batch_size = batch_size,
        max_length = kw.pop("max_length", None),
        world_size = kw.pop("world_size", 1),
    )
    return types.SimpleNamespace(model = model, args = args, optimizer = kw.pop("optimizer", None), **kw)


def test_batch_size_reaches_the_estimate():
    ns, state = _load()
    reserve, seq_len = ns["_training_reserve_bytes"](_Model(), 512, batch_size = 4)
    assert (reserve, seq_len) == (4 * GIB, 512)
    assert state.estimates[-1] == (512, 4)
    ns["_training_reserve_bytes"](_Model())
    assert state.estimates[-1] == (2048, 1)


def test_attach_time_auto_plans_for_batch_two():
    ns, state = _load(free = 10 * GIB)
    ns["install_block_swap"](_Model(), "auto")
    assert state.estimates[-1] == (2048, 2)


def test_only_attach_time_auto_is_marked():
    ns, state = _load(free = 10 * GIB)
    fits = _Model()
    ns["install_block_swap"](fits, "auto", prefetch_depth = 3)
    assert fits._unsloth_offload_layers_auto == 3 and not state.swaps
    counted = _Model()
    ns["install_block_swap"](counted, 2)
    assert not hasattr(counted, "_unsloth_offload_layers_auto")
    loaded = _Model()
    loaded._unsloth_block_swap = object()
    ns["install_block_swap"](loaded, "auto")
    assert not hasattr(loaded, "_unsloth_offload_layers_auto")


def test_unmarked_model_is_untouched():
    ns, state = _load(free = GIB)
    assert ns["replan_auto_offload_for_trainer"](_trainer(_Model())) is None
    assert not state.estimates and not state.swaps


def test_no_op_when_the_real_batch_fits(capsys):
    ns, state = _load(free = 10 * GIB)
    model = _Model()
    ns["install_block_swap"](model, "auto")
    ns["replan_auto_offload_for_trainer"](_trainer(model, batch_size = 8))
    assert not state.swaps and "re-planned" not in capsys.readouterr().out
    assert state.estimates[-1] == (2048, 8)


def test_installs_when_short_and_nothing_was_swapped(capsys):
    ns, state = _load(free = 5 * GIB)
    model = _Model()
    ns["install_block_swap"](model, "auto")
    assert not state.swaps
    swapper = ns["replan_auto_offload_for_trainer"](_trainer(model, batch_size = 8, max_length = 1024))
    assert swapper is model._unsloth_block_swap is model.layers._unsloth_block_swap
    assert len(swapper.indices) == 3
    assert state.estimates[-1] == (1024, 8)
    assert "0 -> 3 decoder layers" in capsys.readouterr().out


def test_rebuilds_a_smaller_plan_when_the_restore_fits(capsys):
    ns, state = _load(free = 1 * GIB)
    model = _Model()
    first = ns["install_block_swap"](model, "auto")
    assert len(first.indices) == 1
    state.free = 3 * GIB
    swapper = ns["replan_auto_offload_for_trainer"](_trainer(model, batch_size = 6))
    assert state.removed == [first] and swapper is not first
    # 3 GiB free with layer 1 out -> 2 GiB after restore, so 4 layers cover a 6 GiB reserve.
    assert len(swapper.indices) == 4
    assert "1 -> 4 decoder layers" in capsys.readouterr().out


def test_rebuild_is_refused_when_the_restore_does_not_fit(capsys):
    ns, state = _load(free = 0)
    model = _Model()
    first = ns["install_block_swap"](model, "auto")
    assert len(first.indices) == 2
    state.free = GIB // 2  # less than the 2 GiB the old layers would bring back
    swapper = ns["replan_auto_offload_for_trainer"](_trainer(model, batch_size = 4))
    assert swapper is first and not state.removed and len(state.swaps) == 1
    out = capsys.readouterr().out
    assert "offload_layers = 6" in out and "lower per_device_train_batch_size" in out


def test_never_shrinks_an_existing_plan():
    ns, state = _load(free = 0, pick = [7])
    model = _Model()
    ns["install_block_swap"](model, "auto")
    model._unsloth_block_swap.indices = [5, 6, 7]
    state.free = 3 * GIB
    swapper = ns["replan_auto_offload_for_trainer"](_trainer(model, batch_size = 5))
    assert swapper.indices == [5, 6, 7]


def test_skips_with_an_optimizer_distributed_or_no_cuda():
    for kw, load in (
        ({"optimizer": object()}, {}),
        ({"world_size": 2}, {}),
        ({"is_fsdp_enabled": True}, {}),
        ({"is_deepspeed_enabled": True}, {}),
        ({}, {"cuda": False}),
    ):
        ns, state = _load(free = 5 * GIB, **load)
        model = _Model()
        model._unsloth_offload_layers_auto = 2
        assert ns["replan_auto_offload_for_trainer"](_trainer(model, **kw)) is None, kw
        assert not state.swaps and not state.estimates


def test_errors_never_break_trainer_init(capsys):
    ns, state = _load(free = GIB)
    model = _Model()
    model._unsloth_offload_layers_auto = 2

    def boom(*a, **k):
        raise RuntimeError("probe failed")

    ns["estimate_training_reserve_bytes"] = boom
    assert ns["_replan_auto_offload_safely"](_trainer(model)) is None
    assert ns["_replan_auto_offload_safely"](_trainer(model)) is None
    assert capsys.readouterr().out.count("could not re-plan") == 1


def test_trainer_init_wrapper_calls_the_safe_replan():
    mod = ast.parse(SRC)
    wrapper = next(
        n
        for n in ast.walk(mod)
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_trainer_init"
    )
    calls = [
        c.func.id
        for c in ast.walk(wrapper)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    ]
    assert "_replan_auto_offload_safely" in calls
    assert calls.index("_replan_auto_offload_safely") > calls.index("_original_trainer_init")


def test_a_short_second_card_still_rebuilds():
    ns, state = _load(free = 1 * GIB)
    model = _Model()
    first = ns["install_block_swap"](model, "auto")
    state.homes = {i: "cuda:1" for i in range(8)}
    state.layer_devices = {"cuda:0", "cuda:1"}
    state.free = 3 * GIB
    state.free_by = {"cuda:0": 100 * GIB}
    swapper = ns["replan_auto_offload_for_trainer"](_trainer(model, batch_size = 6))
    assert state.removed == [first] and swapper is not first


def test_a_failed_rebuild_puts_the_old_plan_back(capsys):
    ns, state = _load(free = 1 * GIB)
    model = _Model()
    first = ns["install_block_swap"](model, "auto")
    old = list(first.indices)
    state.free = 3 * GIB
    state.fail = 1  # the replacement plan's allocation fails, the old plan's re-attach does not
    assert ns["_replan_auto_offload_safely"](_trainer(model, batch_size = 6)) is None
    assert model._unsloth_block_swap is not first and model._unsloth_block_swap.indices == old
    assert "could not re-plan" in capsys.readouterr().out


def test_newly_swapped_layers_recompute_in_backward():
    ns, state = _load(free = 1 * GIB)
    model = _Model()
    ns["install_block_swap"](model, "auto")
    for layer in model.layers:
        layer.__dict__["_unsloth_skip_checkpoint"] = True
    state.free = 3 * GIB
    swapper = ns["replan_auto_offload_for_trainer"](_trainer(model, batch_size = 6))
    for i, layer in enumerate(model.layers):
        assert layer.__dict__.get("_unsloth_skip_checkpoint", False) == (i not in swapper.indices)


def test_a_card_without_swapped_layers_is_checked():
    ns, state = _load(free = 1 * GIB)
    model = _Model()
    first = ns["install_block_swap"](model, "auto")
    state.layer_devices = {"cuda:0", "cuda:1"}
    state.free = 3 * GIB
    state.free_by = {"cuda:0": 100 * GIB}
    swapper = ns["replan_auto_offload_for_trainer"](_trainer(model, batch_size = 6))
    assert state.removed == [first] and swapper is not first


def test_resident_blocks_and_idle_slots_need_no_restore():
    ns, state = _load(free = 0)
    model = _Model()
    first = ns["install_block_swap"](model, "auto")
    assert len(first.indices) == 2
    state.resident = True
    state.idle_slot_bytes = GIB
    state.free = GIB // 2
    swapper = ns["replan_auto_offload_for_trainer"](_trainer(model, batch_size = 4))
    assert state.removed == [first] and swapper is not first


def test_a_rebuild_keeps_the_old_plans_layers_swapped():
    ns, state = _load(free = 0, pick = [3, 7])
    model = _Model()
    first = ns["install_block_swap"](model, "auto")
    assert first.indices == [3, 7]
    state.free = 3 * GIB
    ns["auto_swap_indices"] = lambda layers, reserve, depth: ([1, 4, 7], 0)
    swapper = ns["replan_auto_offload_for_trainer"](_trainer(model, batch_size = 6))
    assert swapper.indices == [1, 3, 4, 7]


def test_preference_trainers_plan_for_both_rows_of_a_pair():
    ns, state = _load(free = 100 * GIB)
    model = _Model()
    model._unsloth_offload_layers_auto = 2

    # Unsloth's generated trainer copies TRL's source, so TRL's DPOTrainer is not in the MRO.
    class _UnslothDPOTrainer:
        pass

    class UnslothDPOTrainer(_UnslothDPOTrainer):
        pass

    trainer = UnslothDPOTrainer()
    trainer.__dict__.update(vars(_trainer(model, batch_size = 3)))
    ns["replan_auto_offload_for_trainer"](trainer)
    assert state.estimates[-1][1] == 6
    ns["replan_auto_offload_for_trainer"](_trainer(model, batch_size = 3))
    assert state.estimates[-1][1] == 3
