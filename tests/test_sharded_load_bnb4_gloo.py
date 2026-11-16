# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Sharded loading of a pre-quantized bitsandbytes 4-bit checkpoint on a gloo CPU world (2 ranks), real
Params4bit: each rank's packed bytes and QuantState equal the slice a full load + split / shard cuts, bitwise,
the sharded-loaded model trains like the full load, and each rank holds ~1/N of the 4-bit bytes."""

import json
import multiprocessing
import os

if os.environ.get("UNSLOTH_TEST_RANKS_ON_GPU") != "1" and (
    multiprocessing.parent_process() is not None
    or getattr(multiprocessing.current_process(), "_inheriting", False)
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("bitsandbytes")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pp_native_gloo import _free_port, _load as _load_module, _torch_pipelining_error  # noqa: E402
from test_tp_native_oracle import LLAMA, QWEN3, build  # noqa: E402

SIZES = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "num_hidden_layers": 4,
}
ARCHS = {"llama": LLAMA, "qwen3": QWEN3}


def _device():
    return (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def make_checkpoint(
    arch,
    directory,
    nested = True,
    dtype = torch.float32,
    device = "cpu",
    quant_storage = None,
):
    """A tiny random model quantized by transformers (NF4) and saved pre-quantized in several files."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    overrides = dict(SIZES)
    if arch == "qwen3":
        overrides["layer_types"] = ["full_attention"] * SIZES["num_hidden_layers"]
    model = build(ARCHS[arch], overrides, lora = False)
    model.generation_config.pad_token_id = None
    src = os.path.join(directory, "src")
    model.save_pretrained(src, safe_serialization = True)
    q = AutoModelForCausalLM.from_pretrained(
        src,
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_use_double_quant = nested,
            bnb_4bit_compute_dtype = dtype,
            **({"bnb_4bit_quant_storage": quant_storage} if quant_storage is not None else {}),
        ),
        device_map = device,
        dtype = dtype,
    )
    out = os.path.join(directory, "q4")
    q.save_pretrained(out, max_shard_size = "1MB", safe_serialization = True)
    return out


def _load(ckpt, dtype = torch.float32):
    import transformers

    device = _device()
    config = transformers.AutoConfig.from_pretrained(ckpt)
    auto = (
        transformers.AutoModelForImageTextToText
        if getattr(config, "vision_config", None) is not None
        else transformers.AutoModelForCausalLM
    )
    return auto.from_pretrained(
        ckpt,
        dtype = dtype,
        attn_implementation = "eager",
        device_map = {"": device.index if device.type == "cuda" else "cpu"},
    )


def _saved_storage(ckpt):
    with open(os.path.join(ckpt, "config.json")) as f:
        storage = json.load(f).get("quantization_config", {}).get("bnb_4bit_quant_storage")
    dtype = getattr(torch, storage or "uint8")
    return dtype if dtype.is_floating_point else None


def _peft(model):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    config = LoraConfig(
        r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False, lora_dropout = 0.0
    )
    return get_peft_model(model, config).train()


def _is4(p):
    return type(p).__name__ == "Params4bit"


def _quant_fields(state, prefix = ""):
    if state is None:
        return {}
    out = {
        f"{prefix}absmax": state.absmax,
        f"{prefix}code": state.code,
        f"{prefix}shape": tuple(state.shape) if state.shape is not None else None,
        f"{prefix}blocksize": state.blocksize,
        f"{prefix}dtype": state.dtype,
        f"{prefix}quant_type": state.quant_type,
        f"{prefix}offset": state.offset,
    }
    out.update(_quant_fields(getattr(state, "state2", None), prefix + "state2."))
    return out


def _local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def _snapshot(model):
    """{name: (bytes / tensor, QuantState fields, tensor parallel 1-GPU QuantState, layout)} on the CPU."""
    out = {}
    for name, p in model.named_parameters():
        # A plain view: Params4bit.cpu() would move the shared QuantState off the GPU in place.
        local = _local(p).detach().as_subclass(torch.Tensor).contiguous()
        # Raw bytes: float-stored packed 4-bit weights hold NaN bit patterns.
        entry = {
            "data": local.reshape(-1).view(torch.uint8).cpu().clone(),
            "meta": (local.dtype, tuple(local.shape)),
        }
        if _is4(p) or getattr(p, "quant_state", None) is not None:
            entry["quant"] = {
                k: (v.cpu().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in _quant_fields(p.quant_state).items()
            }
            entry["tp_quant"] = {
                k: (v.cpu().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in _quant_fields(getattr(p, "_unsloth_tp_quant_state", None)).items()
            }
            entry["layout"] = getattr(p, "_unsloth_tp_layout", None)
            entry["class"] = type(p).__name__
        out[name] = entry
    return out


def _diff(a, b):
    def same(x, y):
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            return (
                isinstance(x, torch.Tensor)
                and isinstance(y, torch.Tensor)
                and x.dtype == y.dtype
                and x.shape == y.shape
                and torch.equal(x, y)
            )
        if isinstance(x, dict):
            return isinstance(y, dict) and set(x) == set(y) and all(same(x[k], y[k]) for k in x)
        return x == y

    return sorted(set(a) ^ set(b)) + [k for k in a if k in b and not same(a[k], b[k])]


def _held_4bit_bytes(model):
    """Bytes of the 4-bit weights this rank holds: packed storage plus QuantState tensors (zero-stride = 0)."""
    seen, total = set(), 0

    def add(t):
        nonlocal total
        if not isinstance(t, torch.Tensor) or id(t) in seen:
            return
        seen.add(id(t))
        local = _local(t)
        if local.numel() and not any(local.stride()) and local.numel() > 1:
            return
        total += local.numel() * local.element_size()

    for p in model.parameters():
        if not (_is4(p) or getattr(p, "quant_state", None) is not None):
            continue
        add(p)
        for v in _quant_fields(p.quant_state).values():
            add(v)
    return total


def _case_pp(
    rank,
    world,
    tmp,
    ckpt,
    schedule = "gpipe",
):
    from torch.distributed.device_mesh import init_device_mesh

    ppn = _load_module("pp_native")
    ppl = _load_module("pp_load")
    from test_pp_load_gloo import _train
    from test_pp_native_gloo import batch

    mesh = init_device_mesh(_device().type, (world,), mesh_dim_names = ("pp",))["pp"]
    full = _peft(_load(ckpt))
    full_bytes = _held_4bit_bytes(full)
    ids, labels = batch(full)
    ids, labels = ids.to(_device()), labels.to(_device())
    pp_full = ppn.apply_pipeline_parallel(full, mesh, n_microbatches = 2, schedule = schedule)
    with ppl.pipeline_sharded_loading(mesh, schedule = schedule) as state:
        model = _load(ckpt)
    stats = dict(model._unsloth_pp_preload)
    loaded_bytes = _held_4bit_bytes(model)
    placeholders4 = sum(1 for p in model.parameters() if _is4(p) and ppl.is_placeholder(p))
    model = _peft(model)
    pp = ppn.apply_pipeline_parallel(model, mesh, n_microbatches = 2, schedule = schedule)
    diffs = _diff(_snapshot(full), _snapshot(model))
    return {
        "models": len(state["models"]),
        "stats": stats,
        "bytes": (full_bytes, loaded_bytes, _held_4bit_bytes(full), _held_4bit_bytes(model)),
        "placeholders4": placeholders4,
        "diffs": diffs,
        "params": len(_snapshot(model)),
        "full": _train(full, pp_full, ids, labels),
        "sharded": _train(model, pp, ids, labels),
    }


def _case_tp(rank, world, tmp, ckpt):
    from torch.distributed.device_mesh import init_device_mesh

    tpn = _load_module("tp_native")
    from test_tp_native_gloo import _train
    from test_tp_native_oracle import _batch

    mesh = init_device_mesh(_device().type, (world,), mesh_dim_names = ("tp",))["tp"]
    ref = _peft(_load(ckpt))
    full = _peft(_load(ckpt))
    full_loaded = _held_4bit_bytes(full)
    tpn.apply_tensor_parallel(full, mesh)
    with tpn.sharded_loading(mesh) as state:
        model = _load(ckpt)
    stats = dict(model._unsloth_tp_load_stats)
    loaded_bytes = _held_4bit_bytes(model)
    unapplied = None
    ids = _batch(ref).to(_device())
    try:
        model(input_ids = ids)
    except RuntimeError as exc:
        unapplied = str(exc)
    model = _peft(model)
    tpn.apply_tensor_parallel(model, mesh)
    diffs = _diff(_snapshot(full), _snapshot(model))
    return {
        "models": len(state["models"]),
        "stats": stats,
        "bytes": (full_loaded, loaded_bytes, _held_4bit_bytes(full), _held_4bit_bytes(model)),
        "unapplied": unapplied,
        "diffs": diffs,
        "params": len(_snapshot(model)),
        "ref": _train(ref, ids, 3, torch.nn.utils.clip_grad_norm_)[2],
        "full": _train(full, ids, 3, tpn.tp_clip_grad_norm_)[2],
        "sharded": _train(model, ids, 3, tpn.tp_clip_grad_norm_)[2],
    }


def _case_fsdp2(
    rank,
    world,
    tmp,
    ckpt,
    fit = False,
    storage = None,
):
    """storage: repack the 4-bit bytes as that float dtype right after loading, as FastModel does under FSDP2."""
    import importlib

    from test_fsdp2_native_sharded_load_gloo import _train
    from test_fsdp2_native_units import _load_native

    native = _load_native()
    loading = importlib.import_module("unsloth.models.fsdp2_load")
    device = _device()
    config = native.FSDP2Config(param_dtype = torch.float32, fit_layout = fit, device = device.type)

    def load():
        # transformers value-casts a float quant storage to `dtype`, so load in the saved storage dtype.
        saved = _saved_storage(ckpt)
        model = _load(ckpt, dtype = saved or torch.float32)
        if storage is not None:
            from unsloth.models.fsdp2_qlora import params4bit_to_float_storage
            assert params4bit_to_float_storage(model, getattr(torch, storage)) > 0
        return model

    full = _peft(load())
    full_loaded = _held_4bit_bytes(full)
    native.shard_model(full, config)
    with loading.fsdp2_sharded_loading() as state:
        model = load()
    stats = dict(model._unsloth_fsdp2_load_stats)
    loaded_bytes = _held_4bit_bytes(model)
    chunks4 = sum(1 for p in model.parameters() if _is4(p) and getattr(p, loading._PRELOAD, None))
    model = _peft(model)
    native.shard_model(model, config)
    diffs = _diff(_snapshot(full), _snapshot(model))
    ids = torch.randint(0, 100, (2, 7), generator = torch.Generator().manual_seed(1)).to(device)
    return {
        "models": len(state["models"]),
        "stats": stats,
        "bytes": (full_loaded, loaded_bytes, _held_4bit_bytes(full), _held_4bit_bytes(model)),
        "chunks4": chunks4,
        "diffs": diffs,
        "params": len(_snapshot(model)),
        "full": _train(native, full, ids)[0],
        "sharded": _train(native, model, ids)[0],
    }


def _case_env_auto(rank, world, tmp, ckpt):
    """Native FSDP2 from the environment without UNSLOTH_FSDP2_SHARDED_LOAD: a checkpoint larger than the GPU
    loads sharded, one that fits loads whole (the decision stubbed: no GPU here)."""
    import importlib

    from test_fsdp2_native_units import _load_native

    _load_native()
    loading = importlib.import_module("unsloth.models.fsdp2_load")
    tpn = importlib.import_module("unsloth.models.tp_native")
    os.environ["UNSLOTH_FSDP2"] = "1"
    os.environ.pop("UNSLOTH_FSDP2_SHARDED_LOAD", None)
    out = {}
    original = tpn._whole_load_exceeds_device
    try:
        for too_big in (True, False):
            tpn._whole_load_exceeds_device = lambda *args, big = too_big: big
            model = loading.sharded_load_env(_load)(ckpt)
            out[too_big] = "_unsloth_fsdp2_load_stats" in model.__dict__
        os.environ["UNSLOTH_FSDP2_SHARDED_LOAD"] = "0"
        tpn._whole_load_exceeds_device = lambda *args: True
        out["opted_out"] = (
            "_unsloth_fsdp2_load_stats" in loading.sharded_load_env(_load)(ckpt).__dict__
        )
    finally:
        tpn._whole_load_exceeds_device = original
    return out


def _case_env_pp_auto(rank, world, tmp, ckpt):
    """UNSLOTH_PP_SIZE: a checkpoint larger than the GPU loads only this rank's stages and the Trainer split keeps
    no replica; one that fits loads whole (the decision stubbed: no GPU here)."""
    import importlib

    _load_module("pp_native")
    tp_env = importlib.import_module("unsloth.models.tp_env")
    pp_env = importlib.import_module("unsloth.models.pp_env")
    tpn = importlib.import_module("unsloth.models.tp_native")
    os.environ["UNSLOTH_PP_SIZE"] = str(world)
    tp_env._reset_env_layout()
    out = {}
    original = tpn._whole_load_exceeds_device

    def load(name, load_in_4bit = True):
        return _load(name)

    try:
        for too_big in (True, False):
            tpn._whole_load_exceeds_device = lambda *args, big = too_big: big
            model = _peft(tp_env.tensor_parallel_env_load(load)(ckpt))
            stage = pp_env.pipeline_parallel_env_apply(model, tp_env.env_layout())
            out[too_big] = (
                "_unsloth_pp_preload" in model.base_model.model.__dict__,
                stage._kept is not None,
            )
    finally:
        tpn._whole_load_exceeds_device = original
        os.environ.pop("UNSLOTH_PP_SIZE", None)
        tp_env._reset_env_layout()
    return out


CASES = {
    "pp": _case_pp,
    "tp": _case_tp,
    "fsdp2": _case_fsdp2,
    "env_auto": _case_env_auto,
    "env_pp_auto": _case_env_pp_auto,
}


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
    backend = "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        backend = "cuda:gloo,cpu:gloo"  # both ranks share one GPU: NCCL refuses duplicate devices
    dist.init_process_group(
        backend, rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 300)
    )
    finished = False
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
        finished = True
    finally:
        dist.destroy_process_group()
    if finished:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _spawn(
    case,
    world = 2,
    **kwargs,
):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(900)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


_CKPTS = {}


def _checkpoint(
    arch,
    nested = True,
    quant_storage = None,
):
    key = (arch, nested, quant_storage)
    if key not in _CKPTS:
        directory = tempfile.mkdtemp(prefix = f"bnb4_{arch}_")
        _CKPTS[key] = make_checkpoint(arch, directory, nested, quant_storage = quant_storage)
    return _CKPTS[key]


def _check_bytes(results, world):
    for res in results:
        full, loaded, _, _ = res["bytes"]
        # Each rank holds about 1/world of the 4-bit bytes (plus the whole small QuantState maps).
        assert loaded < full / world * 1.15, (loaded, full)


@pytest.mark.parametrize("arch,schedule", [("llama", "gpipe"), ("qwen3", "1f1b")])
def test_pipeline_sharded_loading_reads_only_its_stages_4bit(arch, schedule):
    if _torch_pipelining_error():
        pytest.skip(_torch_pipelining_error())
    results = _spawn("pp", 2, ckpt = _checkpoint(arch), schedule = schedule)
    for rank, res in enumerate(results):
        assert res["models"] == 1
        assert res["placeholders4"] > 0, rank
        assert res["diffs"] == [], (rank, res["diffs"][:6])
        assert res["sharded"][0] == res["full"][0], (rank, res["sharded"][0], res["full"][0])
        full, held, _, kept = res["bytes"]
        start, stop = res["stats"]["bounds"][res["stats"]["local_stages"][0]]
        # This rank's layers' 4-bit bytes; a placeholder keeps only its code maps and offset (~1.1 KB).
        assert kept == full * (stop - start) // SIZES["num_hidden_layers"], (rank, res["bytes"])
        assert kept <= held <= kept + 1100 * res["placeholders4"], (rank, res["bytes"])


@pytest.mark.parametrize("arch,nested", [("llama", True), ("qwen3", False)])
def test_tensor_parallel_sharded_loading_reads_its_4bit_slices(arch, nested):
    results = _spawn("tp", 2, ckpt = _checkpoint(arch, nested))
    for rank, res in enumerate(results):
        assert res["models"] == 1
        assert "apply_tensor_parallel" in (res["unapplied"] or ""), res["unapplied"]
        assert res["diffs"] == [], (rank, res["diffs"][:6])
        assert res["sharded"] == res["full"], (rank, res["sharded"], res["full"])
        assert res["sharded"] == pytest.approx(res["ref"], rel = 1e-5, abs = 1e-6)
        assert res["stats"]["read_bytes_sliced"] > 0
    _check_bytes(results, 2)


@pytest.mark.parametrize(
    "fit, storage, saved",
    [(False, None, None), (True, None, None), (False, "float16", None), (False, None, "float16")],
)
def test_fsdp2_sharded_loading_reads_its_4bit_byte_chunks(fit, storage, saved):
    """saved: the checkpoint itself stores the 4-bit bytes as that float dtype (bnb_4bit_quant_storage)."""
    ckpt = _checkpoint("llama", quant_storage = getattr(torch, saved) if saved else None)
    results = _spawn("fsdp2", 2, ckpt = ckpt, fit = fit, storage = storage)
    for rank, res in enumerate(results):
        assert res["models"] == 1
        assert res["chunks4"] > 0, rank
        assert res["diffs"] == [], (rank, res["diffs"][:6])
        assert res["sharded"] == res["full"], (rank, res["sharded"], res["full"])
    _check_bytes(results, 2)


def test_native_fsdp2_env_loads_sharded_only_when_the_checkpoint_does_not_fit():
    for res in _spawn("env_auto", 2, ckpt = _checkpoint("llama")):
        assert res == {True: True, False: False, "opted_out": False}, res


def test_pipeline_env_loads_per_stage_only_when_the_checkpoint_does_not_fit():
    if _torch_pipelining_error():
        pytest.skip(_torch_pipelining_error())
    for res in _spawn("env_pp_auto", 2, ckpt = _checkpoint("llama")):
        # Larger than the GPU: per stage, no replica kept; fits: whole, replica kept for generate / exports.
        assert res == {True: (True, False), False: (False, True)}, res
