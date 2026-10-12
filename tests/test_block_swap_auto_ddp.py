# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""from_pretrained(offload_layers = "auto") under a DDP launch. No GPU needed.

Each rank of a torchrun launch loads one whole model onto its own card, so "auto" can plan that
card exactly like a single-GPU load. FSDP / DeepSpeed and multi-GPU splits stay refused, and the
embedding stays on the card (DDP refuses a module with CPU parameters). Ranks may plan different
counts and never talk to each other while loading, so the frozen decoder weights are kept out of
DDP's init-time broadcast on every rank alike: after get_peft_model (custom loops) and at Trainer init.
A Trainer that turns out to run FSDP / DeepSpeed refuses the swap. Extracted with ast so nothing imports torch's CUDA stack.
"""

import ast
import importlib.util
import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "unsloth", "models")
LOADER_UTILS = os.path.join(MODELS, "loader_utils.py")
UTILS = os.path.join(MODELS, "_utils.py")
_FAKE_PKG = "_block_swap_auto_ddp_pkg"


def _extract(
    path,
    functions,
    assigns = (),
):
    src = open(path, encoding = "utf-8").read()
    out = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            out.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in assigns:
            out.append(ast.get_source_segment(src, node))
    return "\n\n".join(out)


class _FakeCuda:
    def __init__(
        self,
        count = 2,
        free = 8 << 30,
    ):
        self.count, self.free = count, free

    def is_available(self):
        return True

    def device_count(self):
        return self.count

    def current_device(self):
        return 0


class _Plan:
    def __init__(
        self,
        layers,
        offload_embedding = False,
    ):
        self.layers, self.offload_embedding, self.device_plan = layers, offload_embedding, None

    def describe(self):
        return "block swap: fabricated plan"


@pytest.fixture
def resolve(monkeypatch):
    """resolve_auto_block_swap over a fabricated CUDA, planner and launcher env."""
    for key in (
        "ACCELERATE_USE_FSDP",
        "FSDP_VERSION",
        "ACCELERATE_USE_DEEPSPEED",
        "UNSLOTH_FORCE_FUSED_LORA",
    ):
        monkeypatch.delenv(key, raising = False)
    state = types.SimpleNamespace(distributed = False, calls = [], layers = 6)

    def plan_block_swap(
        model_name,
        max_memory,
        seq_len,
        offload_embedding = None,
        **options,
    ):
        call = dict(max_memory = max_memory, **options)
        if offload_embedding is not None:
            call["offload_embedding"] = offload_embedding
        state.calls.append(call)
        return _Plan(state.layers, bool(offload_embedding))

    planner = types.ModuleType("unsloth_zoo.device_map_planner")
    planner.plan_block_swap = plan_block_swap
    monkeypatch.setitem(sys.modules, "unsloth_zoo.device_map_planner", planner)
    package = types.ModuleType(_FAKE_PKG)
    package.__path__ = []
    utils = types.ModuleType(f"{_FAKE_PKG}._utils")
    utils.usable_cuda_bytes = lambda device: 8 << 30
    utils.auto_plan_depth = lambda depth: depth
    monkeypatch.setitem(sys.modules, _FAKE_PKG, package)
    monkeypatch.setitem(sys.modules, f"{_FAKE_PKG}._utils", utils)
    fake_torch = types.SimpleNamespace(
        cuda = _FakeCuda(), device = torch.device, tensor = torch.tensor, int64 = torch.int64
    )
    ns = {
        "__name__": f"{_FAKE_PKG}.loader_utils",
        "__package__": _FAKE_PKG,
        "os": os,
        "inspect": __import__("inspect"),
        "torch": fake_torch,
        "DEVICE_TYPE_TORCH": "cuda",
        "is_distributed": lambda: state.distributed,
    }
    exec(
        _extract(
            LOADER_UTILS,
            (
                "resolve_auto_block_swap",
                "_sharded_data_parallel_launch",
                "_single_card_device_map",
                "_single_device_index",
                "ddp_auto_block_swap_launch",
                "_as_bytes",
            ),
            (
                "_TRUTHY",
                "_SIZE_UNITS",
                "UNSLOTH_DEVICE_MAP",
                "UNSLOTH_BALANCED_DEVICE_MAP",
                "_PLANNED_DEVICE_MAPS",
                "TRANSFORMERS_PLACEMENT_STRATEGIES",
                "_BLOCK_SWAP_PLANNER_KEYS",
            ),
        ),
        ns,
    )
    state.fn = lambda device_map, **kw: ns["resolve_auto_block_swap"](
        device_map, "m", max_seq_length = 2048, **kw
    )
    state.ns = ns
    return state


def test_a_ddp_rank_plans_its_own_card(resolve):
    """torchrun gives each rank {"": "cuda:LOCAL_RANK"}: plan that card, keep the embedding on it."""
    resolve.distributed = True
    layers, device_map, embedding = resolve.fn({"": "cuda:1"}, offload_embedding = True)
    assert (layers, device_map, embedding) == (6, {"": 1}, False)
    assert len(resolve.calls) == 1
    assert list(resolve.calls[0]["max_memory"]) == [1]
    assert "offload_embedding" not in resolve.calls[0]


@pytest.mark.parametrize("device_map, card", [(1, 1), ("cuda:1", 1)])
def test_a_ddp_rank_plans_a_bare_card_map(resolve, device_map, card):
    """device_map = 1 or "cuda:1" (no dict) is the same single-card load as {"": "cuda:1"}."""
    resolve.distributed = True
    layers, out_map, embedding = resolve.fn(device_map)
    assert (layers, out_map, embedding) == (6, {"": card}, False)
    assert list(resolve.calls[0]["max_memory"]) == [card]


def test_a_ddp_rank_with_room_swaps_nothing(resolve):
    resolve.distributed, resolve.layers = True, 0
    assert resolve.fn({"": "cuda:0"}) == (0, {"": "cuda:0"}, False)


def test_a_ddp_rank_honours_a_max_memory_for_its_card(resolve):
    resolve.distributed = True
    resolve.fn({"": "cuda:1"}, planner_kwargs = {"max_memory": {1: "2GiB"}})
    assert resolve.calls[0]["max_memory"] == {1: 2 << 30}


@pytest.mark.parametrize(
    "env",
    [
        {"ACCELERATE_USE_FSDP": "true"},
        {"FSDP_VERSION": "2"},
        {"ACCELERATE_USE_DEEPSPEED": "true"},
    ],
)
def test_fsdp_and_deepspeed_launches_stay_refused(resolve, monkeypatch, capsys, env):
    resolve.distributed = True
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert resolve.fn({"": "cuda:0"}) == (0, {"": "cuda:0"}, None)
    assert resolve.calls == []
    assert "shards the weights" in capsys.readouterr().out


@pytest.mark.parametrize(
    "device_map",
    ["unsloth", "unsloth_balanced", "auto", "sequential", None, {"a": 0, "b": 1}, {"": "cpu"}],
)
def test_ddp_never_plans_a_multi_gpu_split_or_a_non_cuda_map(resolve, device_map):
    resolve.distributed = True
    layers, returned, embedding = resolve.fn(device_map)
    assert (layers, embedding) == (0, None)
    assert returned == device_map
    assert resolve.calls == []


def test_a_single_process_still_offers_the_embedding(resolve):
    """Unchanged outside a distributed launch."""
    resolve.fn({"": "cuda:0"}, offload_embedding = True)
    assert resolve.calls[0]["offload_embedding"] is True


def test_planning_never_runs_a_collective(resolve, monkeypatch):
    """Even with a process group up, from_pretrained must not wait on the other ranks: a loader only
    some ranks reach would hang. Each rank keeps its own plan."""
    import torch.distributed as dist

    calls = []

    def collective(*args, **kwargs):
        calls.append(args)
        raise AssertionError("collective inside from_pretrained")

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda *a, **k: "gloo")
    for name in (
        "all_reduce",
        "all_gather",
        "all_gather_object",
        "broadcast",
        "broadcast_object_list",
        "barrier",
    ):
        monkeypatch.setattr(dist, name, collective)
    resolve.distributed = True
    assert resolve.fn({"": "cuda:1"}) == (6, {"": 1}, False)
    assert calls == []


# Trainer side: the DDP ignore list.


def _ignore_helpers(monkeypatch, distributed = True):
    zoo_root = importlib.util.find_spec("unsloth_zoo").submodule_search_locations[0]
    spec = importlib.util.spec_from_file_location(
        "_block_swap_for_ddp_test", os.path.join(zoo_root, "block_swap.py")
    )
    block_swap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(block_swap)
    package = types.ModuleType(_FAKE_PKG)
    package.__path__ = []
    loader_utils = types.ModuleType(f"{_FAKE_PKG}.loader_utils")
    loader_utils.is_distributed = lambda: distributed
    monkeypatch.setitem(sys.modules, _FAKE_PKG, package)
    monkeypatch.setitem(sys.modules, f"{_FAKE_PKG}.loader_utils", loader_utils)
    ns = {
        "__name__": f"{_FAKE_PKG}._utils",
        "__package__": _FAKE_PKG,
        "torch": torch,
        "find_decoder_layers": block_swap.find_decoder_layers,
    }
    exec(
        _extract(
            UTILS,
            (
                "mark_auto_offload_for_ddp",
                "_ignore_block_swap_params_for_ddp",
                "_block_swap_for_trainer_backend",
            ),
        ),
        ns,
    )
    return ns


def _tiny_peft_model(mark = True):
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 2,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config)
    if mark:
        model._unsloth_offload_layers_auto_ddp = True
    # What _exclude_rope_inv_freq_from_ddp leaves on the base model before get_peft_model.
    model._ddp_params_and_buffers_to_ignore = ["model.rotary_emb.inv_freq"]
    lora = peft.LoraConfig(r = 4, target_modules = ["q_proj", "v_proj", "down_proj"])
    return peft.get_peft_model(model, lora)


def test_ignore_list_names_the_frozen_decoder_weights_of_the_wrapped_model(monkeypatch):
    ns = _ignore_helpers(monkeypatch)
    model = _tiny_peft_model()
    # A swapped block between uses: 0-element weights, same Parameter objects.
    layer = model.base_model.model.model.layers[2]
    for p in layer.parameters():
        if not p.requires_grad:
            p.data = torch.empty(0, dtype = p.dtype)
    added = ns["_ignore_block_swap_params_for_ddp"](model)
    params = dict(model.named_parameters())
    assert added and all(name in params for name in added)
    assert not any(params[name].requires_grad for name in added)
    assert not any("lora_" in name for name in added)
    frozen_layer_params = {
        name for name, p in params.items() if ".layers." in name and not p.requires_grad
    }
    assert set(added) == frozen_layer_params
    # Embedding, final norm and lm_head stay synced; the rotary entry survives the merge.
    assert not any(
        "embed_tokens" in n or "lm_head" in n or n.endswith("model.norm.weight") for n in added
    )
    ignored = model._ddp_params_and_buffers_to_ignore
    assert "model.rotary_emb.inv_freq" in ignored and set(added) <= set(ignored)
    # Idempotent: a second Trainer on the same model adds nothing twice.
    assert ns["_ignore_block_swap_params_for_ddp"](model) == []
    assert len(model._ddp_params_and_buffers_to_ignore) == len(set(ignored))


def test_ignore_list_is_the_same_whatever_this_rank_swapped(monkeypatch):
    """Rank 0 swapped layer 2, rank 1 nothing: both must hand DDP the same list."""
    ns = _ignore_helpers(monkeypatch)
    swapped, resident = _tiny_peft_model(), _tiny_peft_model()
    for p in swapped.base_model.model.model.layers[2].parameters():
        if not p.requires_grad:
            p.data = torch.empty(0, dtype = p.dtype)
    assert ns["_ignore_block_swap_params_for_ddp"](swapped) == ns[
        "_ignore_block_swap_params_for_ddp"
    ](resident)


def test_ignore_list_untouched_without_auto_or_outside_ddp(monkeypatch):
    ns = _ignore_helpers(monkeypatch)
    model = _tiny_peft_model(mark = False)
    assert ns["_ignore_block_swap_params_for_ddp"](model) == []
    ns = _ignore_helpers(monkeypatch, distributed = False)
    assert ns["_ignore_block_swap_params_for_ddp"](_tiny_peft_model()) == []


def test_mark_follows_the_loaders_ddp_decision(monkeypatch):
    ns = _ignore_helpers(monkeypatch)
    for ddp_auto in (True, False):
        model = types.SimpleNamespace()
        ns["mark_auto_offload_for_ddp"](model, ddp_auto)
        assert getattr(model, "_unsloth_offload_layers_auto_ddp", False) is ddp_auto


def test_ddp_auto_launch_needs_ddp_and_one_card(resolve, monkeypatch):
    """The loaders' switch for the DDP plan, the embedding veto and the marker. A launch that already
    shows FSDP / DeepSpeed keeps main's fallback (load every layer, no marker, no Trainer refusal)."""
    launch = resolve.ns["ddp_auto_block_swap_launch"]
    assert not launch({"": "cuda:0"})
    resolve.distributed = True
    assert launch({"": "cuda:0"}) and launch(0) and launch("cuda:1")
    assert not launch("auto") and not launch({"a": 0, "b": 1})
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    assert not launch({"": "cuda:0"})


def test_loaders_keep_the_embedding_on_the_card_for_a_ddp_plan():
    """resolve_auto_block_swap only returns embedding = False; the loaders hold the caller's
    offload_embedding, so they must drop it themselves before offload_input_embedding runs."""
    for name, var, veto in (
        ("llama.py", "offload_layers", "offload_embedding = _offload_embedding_mode = False"),
        ("vision.py", "_offload_layers", "offload_embedding = False"),
    ):
        src = open(os.path.join(MODELS, name), encoding = "utf-8").read()
        call = src.index(f"{var}, device_map, _embedding_needed = resolve_auto_block_swap(")
        before, after = src[call - 200 : call], src[call : call + 4000]
        assert (
            "_ddp_auto_swap = ddp_auto_block_swap_launch(requested_device_map(device_map))"
            in before
        )
        assert (
            "if _ddp_auto_swap:" in after and veto in after.split("if _ddp_auto_swap:", 1)[1][:200]
        )
        assert "mark_auto_offload_for_ddp(model, _ddp_auto_swap)" in src


def test_wiring_trainer_init_and_both_loaders():
    utils = open(UTILS, encoding = "utf-8").read()
    init = next(
        n
        for n in ast.walk(ast.parse(utils))
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_trainer_init"
    )
    assert "_block_swap_for_trainer_backend(self)" in ast.get_source_segment(utils, init)
    for name in ("llama.py", "vision.py"):
        src = open(os.path.join(MODELS, name), encoding = "utf-8").read()
        assert "mark_auto_offload_for_ddp(model, _ddp_auto_swap)" in src, name
    # The pre-wrapped PEFT branch installs the swap after its rope pass, so it applies the list itself.
    llama = open(os.path.join(MODELS, "llama.py"), encoding = "utf-8").read()
    install = llama.index("install_block_swap(\n                    model,")
    assert "_ignore_block_swap_params_for_ddp(model)" in llama[install : install + 600]


def _swapped_model(
    fsdp = False,
    deepspeed = False,
    swapped = (2,),
    mark = True,
):
    model = types.SimpleNamespace(_unsloth_block_swap = types.SimpleNamespace(indices = list(swapped)))
    if mark:
        model._unsloth_offload_layers_auto_ddp = True
    return types.SimpleNamespace(model = model, is_fsdp_enabled = fsdp, is_deepspeed_enabled = deepspeed)


@pytest.mark.parametrize("backend", ["fsdp", "deepspeed"])
def test_a_sharded_trainer_refuses_an_auto_swap(monkeypatch, backend):
    """TrainingArguments(fsdp / deepspeed = ...) under torchrun leaves no trace at from_pretrained, so
    "auto" planned for DDP; the sharding backend must not wrap 0-element placeholders."""
    ns = _ignore_helpers(monkeypatch)
    check = ns["_block_swap_for_trainer_backend"]
    with pytest.raises(ValueError, match = "offload_layers = 0"):
        check(_swapped_model(**{backend: True}))
    # A rank whose own plan swapped nothing refuses too: a peer may have swapped, and a one-sided
    # refusal would leave the other ranks hanging in the backend's collectives.
    with pytest.raises(ValueError, match = "swapped 0 decoder layers"):
        check(_swapped_model(swapped = (), **{backend: True}))
    # No "auto": nothing to refuse.
    assert check(_swapped_model(mark = False, **{backend: True})) == []
    # get_peft_model(offload_layers = "auto") alone: refuse where it swapped, not where it did not.
    peft_auto = _swapped_model(mark = False, **{backend: True})
    peft_auto.model._unsloth_offload_layers_auto = 1
    with pytest.raises(ValueError, match = "swapped 1 decoder layers"):
        check(peft_auto)
    peft_auto.model._unsloth_block_swap.indices = []
    assert check(peft_auto) == []


def test_ignore_list_is_applied_at_the_end_of_get_peft_model():
    """A custom loop hands the PeftModel to DistributedDataParallel / accelerator.prepare directly, so
    both get_peft_model paths apply the list on return. Not at load: before get_peft_model only the
    swapped host-built layers are frozen, so the list would differ between ranks."""
    for name, owner in (("llama.py", "FastLlamaModel"), ("vision.py", "FastBaseModel")):
        src = open(os.path.join(MODELS, name), encoding = "utf-8").read()
        cls = next(
            n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef) and n.name == owner
        )
        fn = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_peft_model"
        )
        body = ast.get_source_segment(src, fn)
        tail = body[body.rindex("_exclude_rope_inv_freq_from_ddp(model)") :]
        assert "_ignore_block_swap_params_for_ddp(model)" in tail.split("return model")[0], name
    rope = _extract(LOADER_UTILS, ("_exclude_rope_inv_freq_from_ddp",))
    assert "_ignore_block_swap_params_for_ddp" not in rope


def test_load_time_frozen_set_is_rank_dependent(monkeypatch):
    """Why the list is never built at load: there only swapped layers are frozen."""
    ns = _ignore_helpers(monkeypatch)
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 2,
    )
    frozen = []
    for swapped in (1, 2):
        model = transformers.LlamaForCausalLM(config)
        model._unsloth_offload_layers_auto_ddp = True
        for layer in model.model.layers[3 - swapped :]:
            layer.requires_grad_(False)
        frozen.append({n for n, p in model.named_parameters() if not p.requires_grad})
        # So the helper leaves a model PEFT has not wrapped alone.
        assert ns["_ignore_block_swap_params_for_ddp"](model) == []
        assert not getattr(model, "_ddp_params_and_buffers_to_ignore", None)
    assert frozen[0] != frozen[1]


def test_ignore_list_is_applied_when_from_pretrained_loads_an_adapter():
    """An existing adapter returns through patch_peft_model / post_patch_model, not get_peft_model:
    both loader return paths apply the list after the rope pass."""
    src = open(os.path.join(MODELS, "loader.py"), encoding = "utf-8").read()
    rope = "model = _exclude_rope_inv_freq_from_ddp(model)\n"
    assert src.count(rope) == 2
    for at in [i for i in range(len(src)) if src.startswith(rope, i)]:
        assert "_ignore_block_swap_params_for_ddp(model)" in src[at : at + 300]


def test_full_finetuning_never_ignores_a_trainable_weight(monkeypatch):
    """Every weight trainable: nothing to keep out of DDP (zoo's BlockSwap only swaps frozen weights,
    and the loaders refuse offload_layers with full_finetuning)."""
    ns = _ignore_helpers(monkeypatch)
    model = _tiny_peft_model()
    for p in model.parameters():
        p.requires_grad_(True)
    assert ns["_ignore_block_swap_params_for_ddp"](model) == []
