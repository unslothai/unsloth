# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DeepSeek-V4.1 under TP / PP / FSDP2 on a gloo CPU world vs one process.

The engram gate sigmoid(sign(dot) sqrt|dot|) amplifies fp32 rounding near dot = 0 (~4000x), so both arms clamp
it at 1e-2 and the parity tolerance measures parallelism only.
"""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _packages():
    """unsloth / unsloth.models as plain namespace packages (importing unsloth itself needs a GPU)."""
    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = types.ModuleType("unsloth.models")
    models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    for package in (pkg, models):
        package.__spec__ = importlib.machinery.ModuleSpec(package.__name__, None, is_package = True)
        package.__spec__.submodule_search_locations = package.__path__
    sys.modules.setdefault("unsloth", pkg)
    sys.modules.setdefault("unsloth.models", models)


def _load(name):
    full = f"unsloth.models.{name}"
    if full in sys.modules:
        return sys.modules[full]
    _packages()
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(REPO, "unsloth", "models", f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


if not os.path.exists(os.path.join(REPO, "unsloth", "models", "deepseek_v41", "modeling.py")):
    pytest.skip("this tree has no native DeepSeek-V4.1", allow_module_level = True)

tpn = _load("tp_native")
ppn = _load("pp_native")
_packages()
from unsloth.models.deepseek_v41 import modeling as dsv41  # noqa: E402
from unsloth.models.deepseek_v41.configuration import DeepseekV41Config  # noqa: E402

VOCAB = 512
SEQ = 24  # compress ratio 2 -> 12 compressed positions > index_topk: the indexer selects


def _config(**over):
    text = dict(
        vocab_size = VOCAB,
        hidden_size = 64,
        moe_intermediate_size = 32,
        num_hidden_layers = 6,
        num_attention_heads = 4,
        head_dim = 32,
        qk_rope_head_dim = 8,
        q_lora_rank = 32,
        o_groups = 4,
        o_lora_rank = 16,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        sliding_window = 8,
        compress_ratios = [0, 0, 2, 2, 1, 1, 0],
        kv_source_layer_ids = [2, 4],
        index_source_layer_ids = [2, 3, 4, 5],
        candidate_source_layer_id = 4,
        candidate_topk_blocks = 2,
        candidate_block_size = 2,
        index_n_heads = 2,
        index_head_dim = 16,
        index_topk = 3,
        engram_layer_ids = [1, 4],
        engram_vocab_size = 101,
        engram_max_ngram_size = 3,
        engram_n_heads = 2,
        engram_head_dim = 32,
        engram_compressed_vocab_size = 97,
        num_nextn_predict_layers = 1,
        max_position_embeddings = 1024,
        rope_scaling = {
            "rope_type": "yarn",
            "factor": 4,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 64,
        },
    )
    text.update(over)
    primes, offsets = dsv41.engram_layout(
        tuple(text["engram_layer_ids"]),
        text["engram_max_ngram_size"],
        text["engram_n_heads"],
        text["engram_vocab_size"],
    )
    text["engram_num_embeddings"] = [o[-1] + p[-1][-1] for o, p in zip(offsets, primes)]
    return DeepseekV41Config(text_config = text)


def build(
    seed = 0,
    lora = True,
    dropout = 0.0,
    **over,
):
    from peft import LoraConfig, get_peft_model

    from unsloth.models.grouped_linear_lora import register_grouped_linear_lora

    torch.manual_seed(seed)
    model = dsv41.DeepseekV41ForCausalLM(_config(**over)).float()
    with torch.no_grad():  # random (not identity) mixing and norms, so every path carries signal
        for name, p in model.named_parameters():
            if p.ndim == 1 or name.endswith(("q_weight", "k_weight")):
                p.add_(torch.randn_like(p) * 0.1)
    model.bind_token_map([i % 97 for i in range(VOCAB)])
    text = model.config.text_config
    g = torch.Generator().manual_seed(seed + 2)
    for rows, layer_id in zip(text.engram_num_embeddings, text.engram_layer_ids):
        model.model.set_engram_table(
            layer_id, torch.randn(rows, text.engram_head_dim, generator = g) * 0.5
        )
    for layer in model.model.layers:
        if layer.engram is not None:
            layer.engram.clamp_value = 1e-2
    model.config._attn_implementation = "eager"
    if not lora:
        return model
    names = {
        n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    }
    names.discard("lm_head")
    # The indexer selects (no gradient) and the router stays frozen.
    regex = r"^(?!.*\.indexer\.)(?!.*\.gate\.).*\.(" + "|".join(sorted(names)) + r")$"
    config = LoraConfig(
        r = 4, lora_alpha = 8, target_modules = regex, init_lora_weights = False, lora_dropout = dropout
    )
    assert register_grouped_linear_lora(config, model)
    torch.manual_seed(seed + 1)
    return get_peft_model(model, config)


def batch(bsz = 2):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1, VOCAB - 1, (bsz, SEQ), generator = g)
    return ids, ids.clone()


def _lora(model, full = lambda g, p: g):
    return {
        n.replace(".default.", "."): full(p.grad, p).detach().clone().double()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }


def _reference(
    mode,
    ids,
    labels,
    parts,
    steps,
    dropout = 0.0,
):
    from peft import get_peft_model_state_dict

    model = build(dropout = dropout)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 1e-2)
    per = ids.shape[0] // parts
    chunks = [(ids[i * per : (i + 1) * per], labels[i * per : (i + 1) * per]) for i in range(parts)]
    n_items = int((labels[:, 1:] != -100).sum())
    losses, grads = [], None
    for step in range(steps):
        if mode == "tp":
            loss = model(input_ids = ids, labels = labels).loss
        elif mode == "fsdp2":
            loss = sum(model(input_ids = a, labels = b).loss for a, b in chunks) / parts
        else:
            loss = sum(
                model(input_ids = a, labels = b, num_items_in_batch = n_items).loss for a, b in chunks
            )
        loss.backward()
        if step == 0:
            grads = _lora(model)
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
    trainable = {
        n.replace(".default.", ".") for n, p in model.named_parameters() if p.requires_grad
    }
    state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    return {"losses": losses, "grads": grads, "trainable": trainable, "state": state}


def _local_shape(t):
    return tuple(t.to_local().shape) if hasattr(t, "to_local") else tuple(t.shape)


def _case_parity(
    rank,
    world,
    tmp,
    mode,
    schedule = "gpipe",
    n_mb = 2,
    steps = 2,
    fault = None,
    layers_per_stage = None,
    dropout = 0.0,
):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    bsz = max(2, world) if mode == "fsdp2" else (max(2, n_mb) if mode == "pp" else 2)
    ids, labels = batch(bsz)
    parts = world if mode == "fsdp2" else n_mb
    ref = _reference(mode, ids, labels, parts, steps, dropout)
    model = build(
        dropout = dropout
    )  # LoRA dropout: the same RNG draws as the reference, in the same order
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = (mode,))[mode]
    save_dir = os.path.join(tmp, "adapter")
    info = {}
    if fault == "no_cut_points":
        ppn._cut_points = lambda parts: None  # the stage boundary may cut the shared group state
    if fault == "no_replicated_grad_sum":
        derive = tpn.derive_tp_plan

        def unsummed(*args, **kwargs):
            plan = derive(*args, **kwargs)
            for region in plan.regions:
                region.replicated = []  # q latent, key / value head, compressor: gradients left partial
            return plan

        tpn.derive_tp_plan = unsummed
    if mode == "tp":
        model = tpn.apply_tensor_parallel(model, mesh)
        attn = model.get_base_model().model.layers[2].self_attn
        info["shapes"] = {
            "q_b_proj": _local_shape(attn.q_b_proj.base_layer.weight),
            "sinks": _local_shape(attn._parameters["sinks"]),
            "o_a_proj": _local_shape(attn.o_a_proj.base_layer.weight),
            "o_a_proj.lora_B": _local_shape(attn.o_a_proj.lora_B["default"].weight),
            "o_b_proj": _local_shape(attn.o_b_proj.base_layer.weight),
            "kv_proj": _local_shape(attn.kv_proj.base_layer.weight),
        }
        params = [p for p in model.parameters() if p.requires_grad]
        step = lambda: model(input_ids = ids, labels = labels).loss
        full = tpn._full
    elif mode == "fsdp2":
        native = _load("fsdp2_native")
        native.shard_model(
            model, native.FSDP2Config(device = "cpu", param_dtype = torch.float32, mesh = mesh)
        )
        per = bsz // world
        mine = slice(rank * per, (rank + 1) * per)
        params = [p for p in model.parameters() if p.requires_grad]
        step = lambda: model(input_ids = ids[mine], labels = labels[mine]).loss
        full = lambda g, p: g.full_tensor() if hasattr(g, "full_tensor") else g
    else:
        config = ppn.PPConfig(layers_per_stage = layers_per_stage)
        pp = ppn.apply_pipeline_parallel(
            model, mesh, n_microbatches = n_mb, schedule = schedule, config = config
        )
        params = pp.parameters()
        step = lambda: pp.step(ids, labels)
        full = lambda g, p: g
        info["bounds"] = pp.bounds
        tables = model.get_base_model().model.engram_tables
        info["tables"] = {k: type(v).__name__ for k, v in tables.items()}
        layers = model.get_base_model().model.layers
        info["owned"] = [i for i, layer in enumerate(layers) if "_pp_index" not in layer.__dict__]
        info["stage_forward"] = pp.ctx.stage_forward is not None
    opt = torch.optim.SGD(params, lr = 1e-2)
    losses, grads = [], None
    for s in range(steps):
        loss = step()
        if mode != "pp":
            loss.backward()
        if s == 0:
            grads = _lora(model, full)
        opt.step()
        opt.zero_grad(set_to_none = True)
        if mode == "fsdp2":
            value = torch.tensor([float(loss)], dtype = torch.float64)
            dist.all_reduce(value)
            losses.append(float(value) / world)
        else:
            losses.append(None if loss is None else float(loss))
    if mode == "pp":
        pp.save_adapter(save_dir)
    elif mode == "tp":
        tpn.save_tp_adapter(model, save_dir)
    else:
        with _load("fsdp2_save").fsdp2_gathered(model, trainable_only = True) as writer:
            if writer:
                model.save_pretrained(save_dir)
        dist.barrier()
    saved = None
    if rank == 0:
        from safetensors.torch import load_file
        saved = load_file(os.path.join(save_dir, "adapter_model.safetensors"))
    return {"ref": ref, "losses": losses, "grads": grads, "saved": saved, "info": info}


def _case_attention_units(rank, world, tmp):
    """FSDP2 on each self_attn by hand: its input cast rebuilds the group state dict, dropping layer writes."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    model = build()
    mesh = init_device_mesh("cpu", (world,))
    policy = MixedPrecisionPolicy(param_dtype = torch.float32)
    for layer in model.get_base_model().model.layers:
        fully_shard(layer.self_attn, mesh = mesh, mp_policy = policy)
    fully_shard(model, mesh = mesh, mp_policy = policy)
    ids, labels = batch(2)
    try:
        model(input_ids = ids, labels = labels).loss.backward()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"error": None}


CASES = {"parity": _case_parity, "attention_units": _case_attention_units}


def _worker(rank, world, port, case, tmp, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 300)
    )
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    except Exception as exc:  # recorded for flip tests that expect a failure
        chain, cause = [], exc
        while cause is not None and len(chain) < 6:
            chain.append(f"{type(cause).__name__}: {cause}"[:400])
            cause = cause.__cause__ or cause.__context__
        torch.save({"error": " <- ".join(chain)}, os.path.join(tmp, f"result_{rank}.pt"))
        raise
    finally:
        dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
    allow_failure = False,
    **kwargs,
):
    if kwargs.get("mode") == "pp":
        try:
            ppn._check_torch_pipelining()
        except RuntimeError as exc:
            pytest.skip(str(exc))
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        # Windows gloo does not see a failed rank exit: stop the peers after a grace period.
        deadline, failed_at = time.monotonic() + 900, None
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            if failed_at is None and any(p.exitcode not in (None, 0) for p in procs):
                failed_at = time.monotonic()
            if failed_at is not None and time.monotonic() - failed_at > 30:
                break
            time.sleep(0.2)
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(30)
        codes = [p.exitcode for p in procs]
        if not allow_failure:
            assert codes == [0] * world, f"rank exit codes {codes}"
        out = []
        for r in range(world):
            path = os.path.join(tmp, f"result_{r}.pt")
            out.append(torch.load(path, weights_only = False) if os.path.exists(path) else None)
        return out


def _parity_problems(results, cell):
    """What differs from one process ([] when the parallel run equals it)."""
    problems = []
    ref = results[0]["ref"]
    if results[0]["losses"] != pytest.approx(ref["losses"], rel = 1e-6):
        problems.append(f"losses {results[0]['losses']} vs {ref['losses']}")
    merged = {}
    for res in results:
        for name, grad in res["grads"].items():
            merged.setdefault(name, grad)
    if set(ref["grads"]) != set(merged):
        problems.append(f"gradient sets differ: {sorted(set(ref['grads']) ^ set(merged))[:6]}")
    for name, want in ref["grads"].items():
        got = merged.get(name)
        if got is None:
            continue
        scale = float(want.abs().max())
        if not scale > 0:
            problems.append(f"{name}: zero reference gradient")
        elif float((got - want).abs().max()) > 1e-7 + 1e-4 * scale:
            problems.append(
                f"{name}: gradient off by {float((got - want).abs().max()) / scale:.2e}"
            )
    saved = results[0]["saved"]
    if set(saved) != set(ref["state"]):
        problems.append(f"saved keys differ: {sorted(set(saved) ^ set(ref['state']))[:6]}")
    for key, want in ref["state"].items():
        if key in saved:
            diff = float((saved[key].double() - want.double()).abs().max())
            if diff > 1e-7 + 1e-5 * float(want.abs().max()):
                problems.append(f"saved {key} off by {diff:.2e}")
    return problems


CELLS = {
    "tp2": dict(world = 2, mode = "tp"),
    "tp4": dict(world = 4, mode = "tp"),
    # LoRA dropout on every linear: each rank's masks are its slice of the one-process masks (grouped o_a_proj:
    # its own groups' slice)
    "tp2_lora_dropout": dict(world = 2, mode = "tp", dropout = 0.3),
    "fsdp2_2": dict(world = 2, mode = "fsdp2"),
    "pp2_gpipe": dict(world = 2, mode = "pp"),
    "pp2_1f1b": dict(world = 2, mode = "pp", schedule = "1f1b"),
    "pp4_gpipe_mb4": dict(world = 4, mode = "pp", n_mb = 4),
    "pp2x2_interleaved1f1b": dict(world = 2, mode = "pp", schedule = "interleaved1f1b", n_mb = 4),
    "pp2x2_zbv": dict(world = 2, mode = "pp", schedule = "zbvzerobubble", n_mb = 4),
}


def _v_schedules_supported():
    # V placement (two adjacent stages on one rank) needs PipelineStage._is_same_rank, torch >= 2.12.
    try:
        from torch.distributed.pipelining import PipelineStage
    except ImportError:
        return False
    return hasattr(PipelineStage, "_is_same_rank")


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_deepseek_v41_parallel_matches_one_process(cell):
    if (
        CELLS[cell].get("schedule") in ("zbvzerobubble", "dualpipev")
        and not _v_schedules_supported()
    ):
        pytest.skip(f"V schedules need torch >= 2.12 (refused on {torch.__version__})")
    kwargs = dict(CELLS[cell])
    world = kwargs.pop("world")
    results = _spawn("parity", world, **kwargs)
    assert _parity_problems(results, cell) == []
    ref = results[0]["ref"]
    assert any(".o_a_proj.lora_B" in n for n in ref["trainable"])  # the grouped LoRA trains too
    assert not any(".indexer." in n for n in ref["trainable"])
    infos = [r["info"] for r in results]
    if kwargs["mode"] == "tp":
        tp = world
        # query heads, their sinks and whole output groups split; the key / value head stays whole
        heads, head_dim, groups, o_rank, hidden = 4, 32, 4, 16, 64
        for info in infos:
            assert info["shapes"] == {
                "q_b_proj": (heads * head_dim // tp, 32),
                "sinks": (heads // tp,),
                "o_a_proj": (groups * o_rank // tp, heads * head_dim // groups),
                "o_a_proj.lora_B": (groups * o_rank // tp, 4),
                "o_b_proj": (hidden, groups * o_rank // tp),
                "kv_proj": (head_dim, hidden),
            }
    if kwargs["mode"] == "pp":
        cuts = dsv41.pipeline_safe_cut_points(_config())
        for info in infos:
            assert all(start in cuts for start, _ in info["bounds"]), info["bounds"]
            # each engram table lives on the stage owning its layer only
            for key, kind in info["tables"].items():
                owner = int(key) in info["owned"]
                assert (kind != "_PPTableStub") == owner, (key, kind, info["owned"])
            # the stages ran through the model's own pipeline_stage_forward, not the pass-through fallback
            assert info["stage_forward"]


def test_pp_cut_at_an_unsafe_layer_breaks_parity():
    """Flip: a stage starting at layer 3 diverges once the cut-point rule is off."""
    results = _spawn(
        "parity",
        2,
        allow_failure = True,
        mode = "pp",
        layers_per_stage = [3, 3],
        fault = "no_cut_points",
    )
    errors = [r["error"] for r in results if r is not None and "error" in r]
    # the stage starting at layer 3 finds no indexer keys / compressed KV / top-k of layer 2 in its group state
    assert any(
        "KeyError" in e and any(k in e for k in ("index_k", "topk_idxs", "compress_kv"))
        for e in errors
    ), errors


def test_tp_without_the_replicated_gradient_sum_breaks_parity():
    """Flip: without the tp sum the shared latent / kv head / compressor gradients stay partial."""
    results = _spawn("parity", 2, mode = "tp", fault = "no_replicated_grad_sum")
    problems = _parity_problems(results, "tp_unsummed")
    assert any(".kv_proj." in p for p in problems) and any(
        ".q_a_proj." in p for p in problems
    ), problems


def test_pp_refuses_layers_per_stage_at_an_unsafe_cut():
    model = build(lora = False)
    with pytest.raises(ValueError, match = "not safe cut points"):
        ppn.derive_pp_split(model, 2, ppn.PPConfig(layers_per_stage = [3, 3]))
    bounds = ppn.derive_pp_split(model, 4)
    assert [start for start, _ in bounds] == [0, 1, 2, 4]
    with pytest.raises(ValueError, match = "safe cut points"):
        ppn.derive_pp_split(model, 5)


def test_pp_carries_streams_and_pre_mix():
    model = build(lora = False)
    parts = ppn._find_parts(model)
    assert ppn._layer_call_format(parts.decoder, parts.layers_name) == "carry:2"
    assert ppn._carry_layout(model, 2) == [((4, 64), None), ((4,), torch.float32)]
    assert [
        name for name, _ in ppn._layer_tables(parts.decoder, parts.layers, parts.layers_name)
    ] == ["engram_tables"]
    assert ppn._refuse_unsupported(model, parts) is None  # MoE without a router auxiliary loss


def test_tp_plan_shards_heads_sinks_and_groups():
    model = build(lora = False)
    plan = tpn.derive_tp_plan(model, 4)
    attn = [r for r in plan.regions if r.kind == "attention"]
    assert len(attn) == 6 and not [r for r in plan.regions if r.kind == "mlp"]
    region = attn[2]
    assert region.latent and region.kv_replicated and region.head_params == ["sinks"]
    assert (region.heads, region.head_dim) == (4, 32)
    styles = {p.name: p.style for p in region.projs}
    assert styles == {"q_b_proj": tpn.COLWISE, "o_a_proj": tpn.COLWISE, "o_b_proj": tpn.ROWWISE}
    assert {name for name, _ in region.replicated} == {
        "q_a_proj",
        "q_a_norm",
        "kv_proj",
        "kv_norm",
        "compressor",
        "indexer",
    }
    assert len(plan.moe_blocks) == 6


def test_tp_refuses_a_size_that_splits_an_output_group():
    with pytest.raises(ValueError, match = "o_groups = 2"):
        tpn.derive_tp_plan(build(lora = False, o_groups = 2), 4)


def test_fsdp2_shards_whole_decoder_layers():
    """An FSDP2 unit at or below self_attn would rebuild the group state dict, so it is refused."""
    native = _load("fsdp2_native")
    model = build()
    layers = list(model.get_base_model().model.layers)
    units = native.find_shard_units(model)
    assert [id(u) for u in units] == [id(layer) for layer in layers]
    with pytest.raises(RuntimeError, match = "its own"):
        native.check_shard_units(model, [layers[0].self_attn])


def test_fsdp2_on_the_attention_alone_loses_the_group_state():
    """Flip for the rule above: wrapping self_attn by hand fails on the group state the next layer reads."""
    (res,) = _spawn("attention_units", 1)
    assert res["error"] is not None and "KeyError" in res["error"], res


def test_tp_still_refuses_mla_key_value_expansion():
    model = build(lora = False)
    attn = model.model.layers[0].self_attn
    attn.kv_b_proj = torch.nn.Linear(
        8, 8, bias = False
    )  # per-head keys / values from a latent: real MLA
    with pytest.raises(ValueError, match = "multi-head latent attention"):
        tpn.derive_tp_plan(model, 2)


def _fake_stage(carry = None, segments = None):
    """What _PPStageModule._wire / _model_stage_forward read of a stage, for one call outside a pipeline."""
    h, pre_mix = torch.randn(2, SEQ, 4, 64), torch.randn(2, SEQ, 4)
    decoder = types.SimpleNamespace(
        pipeline_stage_forward = lambda start, end, **kw: (h, pre_mix, None)
    )
    ctx = types.SimpleNamespace(
        carry = carry or [((4, 64), None), ((4,), torch.float32)],
        stage_forward = (decoder, None, segments),
        is_first = True,
        is_last = False,
        start = 0,
        end = 2,
    )
    return types.SimpleNamespace(
        _pp_ctx = ctx, _pp_dtype = torch.float32, _pp_warned = False, _wire = lambda t: t
    )


def test_pp_boundary_compiles_in_one_graph():
    """The view check is skipped while compiling (Dynamo cannot trace _is_view)."""
    stage = _fake_stage()

    def boundary(h, pre_mix):
        return ppn._PPStageModule._wire(stage, (h * 2, pre_mix.transpose(0, 1).transpose(0, 1)))

    h, pre_mix = torch.randn(2, 3, 4, 64), torch.randn(2, 3, 4)
    out = torch.compile(boundary, fullgraph = True, backend = "eager")(h, pre_mix)
    assert [tuple(t.shape) for t in out] == [(2, 3, 256), (2, 3, 4)]
    assert not any(t._is_view() for t in out)
    assert torch.equal(out[1], pre_mix)


def test_pp_stage_forward_takes_ordinary_position_ids():
    """Sequential position_ids run; packed rows and left padding are refused."""
    ids = torch.zeros(2, SEQ, dtype = torch.long)
    stage = _fake_stage(segments = dsv41.sequence_segments)
    run = lambda **kw: ppn._PPStageModule._model_stage_forward(stage, ids, kw)
    positions = torch.arange(SEQ).expand(2, -1)
    assert len(run(position_ids = positions)) == 2
    assert len(run(position_ids = positions[:1], attention_mask = torch.ones(2, SEQ))) == 2
    packed = torch.cat([torch.arange(SEQ // 2), torch.arange(SEQ // 2)]).expand(2, -1)
    with pytest.raises(ValueError, match = "packed or left-padded"):
        run(position_ids = packed)
    left = torch.ones(2, SEQ, dtype = torch.long)
    left[0, :3] = 0
    with pytest.raises(ValueError, match = "packed or left-padded"):
        run(attention_mask = left)
    with pytest.raises(
        ValueError, match = "packed or left-padded"
    ):  # a model that cannot classify positions
        ppn._PPStageModule._model_stage_forward(_fake_stage(), ids, {"position_ids": positions})


def test_fsdp2_over_its_own_pipeline_stage_forward_is_refused():
    """That forward runs outside the FSDP root's hooks, so root-owned parameters would stay sharded."""
    model = torch.nn.Linear(2, 2)
    ctx = types.SimpleNamespace(moe = None, stage_forward = (None, None, None))
    model.__dict__["_unsloth_pp_stage"] = types.SimpleNamespace(
        model = model, dp_mesh = None, stages_per_rank = 1, ctx = ctx
    )
    with pytest.raises(ValueError, match = "own pipeline_stage_forward"):
        ppn.check_pp_dp_composition(model, None)
    ctx.stage_forward = None
    with pytest.raises(ValueError, match = r"FSDP2Config\(mesh"):
        ppn.check_pp_dp_composition(model, None)
