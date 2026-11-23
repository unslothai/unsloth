# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 sharded loading while quantizing a 16-bit checkpoint to bitsandbytes 4-bit on load (QLoRA of a
model whose 4-bit form does not fit one GPU), on a gloo CPU world (2 ranks) with real Params4bit: each rank keeps
only its chunk of every quantized weight (about 1/N of the 4-bit bytes), the chunks and QuantStates equal what a
full quantize-on-load + shard_model holds, bitwise, and the model trains exactly like the full load."""

import multiprocessing
import os

if os.environ.get("UNSLOTH_TEST_RANKS_ON_GPU") != "1" and (
    multiprocessing.parent_process() is not None
    or getattr(multiprocessing.current_process(), "_inheriting", False)
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("bitsandbytes")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sharded_load_bnb4_gloo as bnb4  # noqa: E402
from test_tp_native_oracle import build  # noqa: E402

if not hasattr(
    __import__("importlib").import_module("transformers.core_model_loading"), "set_param_for_module"
):
    pytest.skip(
        "transformers without core_model_loading.set_param_for_module (4.x) keeps the full load",
        allow_module_level = True,
    )


def _source_checkpoint(arch, directory):
    """A tiny random 16-bit (float32 here) model saved in several safetensors files."""
    overrides = dict(bnb4.SIZES)
    if arch == "qwen3":
        overrides["layer_types"] = ["full_attention"] * bnb4.SIZES["num_hidden_layers"]
    model = build(bnb4.ARCHS[arch], overrides, lora = False)
    model.generation_config.pad_token_id = None
    out = os.path.join(directory, "src")
    model.save_pretrained(out, max_shard_size = "1MB", safe_serialization = True)
    return out


def _quantize_on_load(ckpt, nested = True):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    device = bnb4._device()
    return AutoModelForCausalLM.from_pretrained(
        ckpt,
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_use_double_quant = nested,
            bnb_4bit_compute_dtype = torch.float32,
        ),
        dtype = torch.float32,
        attn_implementation = "eager",
        device_map = {"": device.index if device.type == "cuda" else "cpu"},
    )


def _case_on_load(
    rank,
    world,
    tmp,
    ckpt,
    nested = True,
    fit = False,
):
    import importlib

    from test_fsdp2_native_sharded_load_gloo import _train
    from test_fsdp2_native_units import _load_native

    native = _load_native()
    loading = importlib.import_module("unsloth.models.fsdp2_load")
    device = bnb4._device()
    config = native.FSDP2Config(param_dtype = torch.float32, fit_layout = fit, device = device.type)
    full = bnb4._peft(_quantize_on_load(ckpt, nested))
    full_loaded = bnb4._held_4bit_bytes(full)
    native.shard_model(full, config)
    with loading.fsdp2_sharded_loading() as state:
        model = _quantize_on_load(ckpt, nested)
    stats = dict(model._unsloth_fsdp2_load_stats)
    loaded_bytes = bnb4._held_4bit_bytes(model)
    chunks4 = sum(
        1 for p in model.parameters() if bnb4._is4(p) and getattr(p, loading._PRELOAD, None)
    )
    unsharded = None
    try:
        model(input_ids = torch.zeros((1, 3), dtype = torch.long, device = device))
    except RuntimeError as exc:
        unsharded = str(exc)
    model = bnb4._peft(model)
    native.shard_model(model, config)
    diffs = bnb4._diff(bnb4._snapshot(full), bnb4._snapshot(model))
    ids = torch.randint(0, 100, (2, 7), generator = torch.Generator().manual_seed(1)).to(device)
    return {
        "models": len(state["models"]),
        "stats": stats,
        "bytes": (full_loaded, loaded_bytes),
        "chunks4": chunks4,
        "unsharded": unsharded,
        "diffs": diffs,
        "params": len(bnb4._snapshot(model)),
        "full": _train(native, full, ids)[0],
        "sharded": _train(native, model, ids)[0],
    }


def _case_env_auto(rank, world, tmp, ckpt):
    """UNSLOTH_FSDP2=1 without UNSLOTH_FSDP2_SHARDED_LOAD: a 4-bit quantize-on-load whose 4-bit size does not fit
    the GPU keeps chunks; one that fits loads whole; UNSLOTH_FSDP2_SHARDED_LOAD=0 opts out (no GPU here: the
    size decision is stubbed and must receive the 4-bit estimate, not the 16-bit file size)."""
    import importlib

    from test_fsdp2_native_units import _load_native

    _load_native()
    loading = importlib.import_module("unsloth.models.fsdp2_load")
    tpn = importlib.import_module("unsloth.models.tp_native")
    os.environ["UNSLOTH_FSDP2"] = "1"
    os.environ.pop("UNSLOTH_FSDP2_SHARDED_LOAD", None)
    out, seen = {}, []
    original = tpn._whole_load_exceeds_device
    try:
        for too_big in (True, False):

            def decide(
                *args,
                big = too_big,
                **kwargs,
            ):
                seen.append(kwargs.get("nbytes"))
                return big

            tpn._whole_load_exceeds_device = decide
            model = loading.sharded_load_env(_quantize_on_load)(ckpt)
            out[too_big] = "_unsloth_fsdp2_load_stats" in model.__dict__
        os.environ["UNSLOTH_FSDP2_SHARDED_LOAD"] = "0"
        tpn._whole_load_exceeds_device = lambda *args, **kwargs: True
        out["opted_out"] = (
            "_unsloth_fsdp2_load_stats"
            in loading.sharded_load_env(_quantize_on_load)(ckpt).__dict__
        )
    finally:
        tpn._whole_load_exceeds_device = original
        os.environ.pop("UNSLOTH_FSDP2_SHARDED_LOAD", None)
    files = sum(os.path.getsize(p) for p in set(tpn._checkpoint_index(ckpt, {}).values()))
    out["estimate"] = seen
    out["files"] = files
    from safetensors import safe_open

    expected = 0
    for path in set(tpn._checkpoint_index(ckpt, {}).values()):
        with safe_open(path, framework = "pt") as f:
            for key in f.keys():
                t = f.get_slice(key)
                numel = 1
                for d in t.get_shape():
                    numel *= d
                linear = len(t.get_shape()) == 2 and not any(x in key for x in ("embed", "lm_head"))
                expected += int(numel * 0.53) if linear else numel * 4
    out["expected"] = expected
    return out


CASES = {"on_load": _case_on_load, "on_load_env_auto": _case_env_auto}


def _worker(rank, world, port, case, tmp, kwargs):
    bnb4.CASES.update(CASES)
    bnb4._worker(rank, world, port, case, tmp, kwargs)


def _spawn(
    case,
    world = 2,
    **kwargs,
):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    port = bnb4._free_port()
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


_DIRS = []


@pytest.fixture(scope = "module")
def checkpoints():
    made = {}

    def get(arch):
        if arch not in made:
            directory = tempfile.mkdtemp(prefix = f"q4onload_{arch}_")
            _DIRS.append(directory)
            made[arch] = _source_checkpoint(arch, directory)
        return made[arch]

    yield get
    while _DIRS:
        shutil.rmtree(_DIRS.pop(), ignore_errors = True)


@pytest.mark.parametrize(
    "arch, nested, fit", [("llama", True, False), ("qwen3", False, False), ("llama", True, True)]
)
def test_fsdp2_sharded_loading_quantizing_on_load_keeps_its_4bit_chunks(
    checkpoints, arch, nested, fit
):
    results = _spawn("on_load", 2, ckpt = checkpoints(arch), nested = nested, fit = fit)
    for rank, res in enumerate(results):
        assert res["models"] == 1
        assert res["stats"]["quantized_on_load"] is True
        assert res["chunks4"] > 0, rank
        assert "shard it first" in (res["unsharded"] or ""), res["unsharded"]
        assert res["diffs"] == [], (rank, res["diffs"][:6])
        assert res["sharded"] == res["full"], (rank, res["sharded"], res["full"])
        full, loaded = res["bytes"]
        # Each rank holds about 1/world of the 4-bit bytes (plus the whole small QuantState maps).
        assert loaded < full / 2 * 1.15, (rank, loaded, full)


def test_native_fsdp2_env_quantizing_on_load_shards_only_when_the_4bit_load_does_not_fit(
    checkpoints,
):
    for res in _spawn("on_load_env_auto", 2, ckpt = checkpoints("llama")):
        assert {k: res[k] for k in (True, False, "opted_out")} == {
            True: True,
            False: False,
            "opted_out": False,
        }, res
        # Sized as the 4-bit load: the float32 linears at ~0.53 bytes per parameter, the (here dominant)
        # embedding and output head dense; not as the float32 files.
        assert len(res["estimate"]) == 2 and all(e for e in res["estimate"]), res["estimate"]
        assert all(e == res["expected"] < res["files"] for e in res["estimate"]), res
