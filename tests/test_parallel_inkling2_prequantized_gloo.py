# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tensor / expert parallel and native FSDP2 on a PRE-QUANTIZED bitsandbytes 4-bit Inkling-2-Mini (gloo CPU, 2 ranks, fp32).

The fake is saved as unsloth/Inkling-2-Mini-unsloth-bnb-4bit is: quantized on load (attention Linear4bit, the
3-D routed expert stacks as bare Params4bit), then save_pretrained under Unsloth's save fixes, so the
projections are in the original `model.llm.*` layout and the fused expert stacks in model layout (packed [n, 1]
bytes + QuantState entries). Ranks reload it with zoo's pre-quantized expert loaders.

- tp=2, full load + LoRA + apply_tensor_parallel: the 4-bit experts shard by expert index; losses / LoRA grads
  match one process.
- tp=2 under sharded_loading: each rank's 4-bit tensors bitwise equal to the full load + apply, same losses.
- native FSDP2, full load + shard_model, and fsdp2_sharded_loading: same 4-bit chunks bitwise, same losses.
- ep=2 under expert_parallel_loading: each rank keeps only its 4-bit experts, bitwise the full load + apply."""

import functools
import importlib.util
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile


def test_pp_prequantized_inkling_sharded_load(ckpt):
    """pipeline_sharded_loading of the pre-quantized checkpoint (the default 4-bit Inkling-2-Mini repo): each rank
    keeps its stages' 4-bit tensors, bitwise those of a full load, and placeholders for the rest."""
    results = _run("_case_pp", ckpt)
    for r in results:
        assert os.path.realpath(r["pp_load"]).startswith(os.path.realpath(REPO) + os.sep), r[
            "pp_load"
        ]
        assert "sharded_error" not in r, r["sharded_error"]
        assert r["models"] == 1
        assert r["stats"].get("prequantized") and r["stats"]["placeholder_tensors"] > 0, r["stats"]
        assert 0 < r["kept"] < r["want"] and r["stacks_kept"] > 0, r
        assert not r["diffs"], r["diffs"][:5]


if __name__ == "__main__" and sys.argv[1:2] == ["--make"]:
    import unsloth  # noqa: F401  (first, as in user code: before torch / transformers / bitsandbytes)

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("bitsandbytes")
pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")
if importlib.util.find_spec("unsloth_zoo") is None:
    pytest.skip("unsloth_zoo is not installed", allow_module_level = True)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import test_tp_inkling2_gloo as ink  # noqa: E402
import test_tp_inkling2_qlora_experts_gloo as q  # noqa: E402

# The released checkpoint's llm_int8_skip_modules (vision, router, message gates stay 16-bit).
SKIP = [
    "lm_head",
    "multi_modal_projector",
    "merger",
    "router",
    "mlp.gate",
    "vision_tower",
    "visual",
] + [
    f"{g}_message_gate.{p}"
    for g in ("q", "k", "v", "mlp")
    for p in ("down_proj", "forget_gate_proj", "message_gate_proj")
]
_ZOO_PATCHES = (
    "patch_bnb4bit_quantizer_weight_conversions",
    "patch_bnb4bit_model_conversion_mapping",
    "patch_bnb4bit_quantizer_param_needs_quantization",
    "patch_bnb4bit_quantizer_process_model",
    "patch_bnb4bit_quantize_convert",
    "patch_transformers_weight_converter_kwargs",
    "patch_bnb4bit_prequantized_plain_weight_dtype",
)


def _make_checkpoint(directory):
    """Run in a fresh process (imports the unsloth package): a 16-bit fake, quantized on load, saved pre-quantized."""
    import unsloth
    from unsloth.import_fixes import fix_transformers_bnb_prequantized_save
    from unsloth.save import _save_bare_params4bit_quant_state

    assert os.path.realpath(unsloth.__file__).startswith(
        os.path.realpath(REPO) + os.sep
    ), unsloth.__file__
    src, dst = os.path.join(directory, "src"), os.path.join(directory, "prequantized")
    source = ink.build(True, lora = False, text_overrides = q.TEXT)
    source.generation_config.pad_token_id = None
    source.save_pretrained(src, safe_serialization = True)
    q._quantize_stacks_on_load()
    model = q._load_4bit(src, skip_modules = SKIP)
    fix_transformers_bnb_prequantized_save()
    with _save_bare_params4bit_quant_state(model):
        model.save_pretrained(dst, safe_serialization = True)
    print(json.dumps({"unsloth": unsloth.__file__}))


@pytest.fixture(scope = "module")
def ckpt(tmp_path_factory):
    directory = str(tmp_path_factory.mktemp("ink_prequant"))
    # conftest disables the CPU fallback for the test process; the checkpoint writer trains nothing.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_DISABLE_CPU_FALLBACK = "0")
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    done = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--make", directory],
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    made = json.loads(done.stdout.strip().splitlines()[-1])
    print("checkpoint written by unsloth at", made["unsloth"])
    path = os.path.join(directory, "prequantized")
    from safetensors import safe_open

    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        keys = set(f.keys())
        # The released layout: original names for projections, model layout for the fused 4-bit stacks.
        assert "model.llm.layers.0.attn.wq_du.weight.quant_state.bitsandbytes__nf4" in keys
        stack = "model.llm.layers.0.mlp.experts.gate_up_proj"
        assert stack in keys and f.get_slice(stack).get_shape()[1] == 1
        assert f"{stack}.quant_state.bitsandbytes__nf4" in keys
    return path


def _zoo():
    """zoo's pre-quantized bnb 4-bit loaders, as Unsloth installs them (CPU ranks: zoo's CPU fallback allowed)."""
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_DISABLE_CPU_FALLBACK"] = "0"
    import unsloth_zoo.temporary_patches.moe_utils_bnb4bit as m

    for name in _ZOO_PATCHES:
        getattr(m, name)()


def _load(path):
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        path, dtype = torch.float32, attn_implementation = "eager", device_map = {"": "cpu"}
    )
    cls = type(q._experts(model)[0])
    if not cls.__dict__.get("_test_4bit"):
        prior = cls.forward
        q._patch_experts_forward(model)
        # Keep transformers' experts-implementation wrapper visible: zoo only packs the stacks of an experts
        # forward it routes (the next load would otherwise keep them 16-bit and refuse the 4-bit checkpoint).
        functools.update_wrapper(cls.forward, prior)
    return model


def _stacks_4bit(model):
    return all(
        type(e._parameters[n]).__name__ == "Params4bit" for e in q._experts(model) for n in q.STACKS
    )


def _diffs(want, have):
    return sorted(
        n
        for n in set(want) | set(have)
        if n not in want
        or n not in have
        or want[n][2] != have[n][2]
        or not torch.equal(want[n][0], have[n][0])
        or not torch.equal(want[n][1], have[n][1])
    )


def _case_tp(rank, world, tmp, path):
    from torch.distributed.device_mesh import init_device_mesh

    _zoo()
    tpn = ink._load("tp_native")
    ink._load("tp_replica")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]
    ids, labels = ink.batch()
    out = {"tp_native": tpn.__file__}

    # (a) full load + LoRA + apply_tensor_parallel against one process.
    ref = ink._train(q._lora(_load(path)), ids, labels)
    model = _load(path)
    out["stacks_4bit"] = _stacks_4bit(model)
    whole = [{n: q._dense(e._parameters[n]) for n in q.STACKS} for e in q._experts(model)]
    model = q._lora(model)
    tpn.apply_tensor_parallel(model, mesh)
    out["moe_sharded"] = tpn._unwrap(model).__dict__["_unsloth_tp_plan"].sharded_moe
    sliced, per = True, None
    for experts, want in zip(q._experts(model), whole):
        for name in q.STACKS:
            p = experts._parameters[name]
            per = p.quant_state.shape[0]
            sliced &= type(p).__name__ == "Params4bit" and torch.equal(
                q._dense(p), want[name][rank * per : (rank + 1) * per]
            )
    out["per_rank"], out["sliced_equal"] = per, sliced
    want = q._snapshot(model)
    out["ref"], out["full"] = ref, ink._train(model, ids, labels, full = tpn._full)

    # (b) sharded_loading of the pre-quantized checkpoint.
    try:
        with tpn.sharded_loading(mesh):
            model = _load(path)
    except Exception as exc:
        out["sharded_error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["stats"] = dict(model.__dict__.get("_unsloth_tp_load_stats") or {})
    out["sharded_stacks_4bit"] = _stacks_4bit(model)
    model = q._lora(model)
    tpn.apply_tensor_parallel(model, mesh)
    have = q._snapshot(model)
    out["compared"], out["diffs"] = len(want), _diffs(want, have)
    out["sharded"] = ink._train(model, ids, labels, full = tpn._full)
    return out


def _case_ep(rank, world, tmp, path):
    from torch.distributed.device_mesh import init_device_mesh

    import test_ep_inkling2_gloo as epi

    _zoo()
    ink._load("tp_native")
    epn = ink._load("ep_native")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    out = {"ep_native": epn.__file__}

    # (a) full load + LoRA + apply_expert_parallel against one process.
    out["ref"] = epi._ref(q._lora(_load(path)), world)
    model = _load(path)
    out["full_bytes"] = q._held_4bit(model)
    model = q._lora(model)
    epn.apply_expert_parallel(model, mesh)
    want = q._snapshot(model)
    out["full"] = epi._ep_train(epn, model, rank, world)

    # (b) expert_parallel_loading of the pre-quantized checkpoint.
    with epn.expert_parallel_loading(mesh) as state:
        model = _load(path)
    out["models"] = len(state["models"])
    out["stats"] = dict(model.__dict__.get("_unsloth_tp_load_stats") or {})
    out["held"] = q._held_4bit(model)
    model = q._lora(model)
    epn.apply_expert_parallel(model, mesh)
    have = q._snapshot(model)
    out["compared"], out["diffs"] = len(want), _diffs(want, have)
    out["sharded"] = epi._ep_train(epn, model, rank, world)
    return out


def _case_fsdp2(rank, world, tmp, path):
    import importlib

    from test_fsdp2_native_units import _load_native
    import test_sharded_load_bnb4_gloo as bnb4

    _zoo()
    native = _load_native()
    loading = importlib.import_module("unsloth.models.fsdp2_load")
    config = native.FSDP2Config(param_dtype = torch.float32, device = "cpu")
    ids, labels = ink.batch()
    out = {"fsdp2_load": loading.__file__}

    def train(model):
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 0.05)
        losses = []
        for _ in range(2):
            loss = model(input_ids = ids, labels = labels, use_cache = False).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none = True)
            losses.append(float(loss.detach()))
        return losses

    out["ref"] = train(q._lora(_load(path)))
    full = q._lora(_load(path))
    native.shard_model(full, config)
    want = bnb4._snapshot(full)
    out["full"] = train(full)
    try:
        with loading.fsdp2_sharded_loading() as state:
            model = _load(path)
    except Exception as exc:
        out["sharded_error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["models"] = len(state["models"])
    out["stats"] = dict(getattr(model, "_unsloth_fsdp2_load_stats", {}) or {})
    out["sharded_stacks_4bit"] = _stacks_4bit(model)
    model = q._lora(model)
    native.shard_model(model, config)
    have = bnb4._snapshot(model)
    out["compared"], out["diffs"] = len(want), bnb4._diff(want, have)
    out["sharded"] = train(model)
    return out


def _case_pp(rank, world, tmp, path):
    import importlib

    from torch.distributed.device_mesh import init_device_mesh

    _zoo()
    ppl = importlib.import_module("unsloth.models.pp_load")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    out = {"pp_load": ppl.__file__}
    want = q._snapshot(_load(path))
    try:
        with ppl.pipeline_sharded_loading(mesh) as state:
            model = _load(path)
    except Exception as exc:
        out["sharded_error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["models"] = len(state["models"])
    out["stats"] = dict(model.__dict__.get("_unsloth_pp_preload") or {})
    kept = {n for n, p in model.named_parameters() if not ppl.is_placeholder(p)}
    have = {n: v for n, v in q._snapshot(model).items() if n in kept}
    out["want"], out["kept"] = len(want), len(have)
    out["diffs"] = _diffs({n: want[n] for n in have if n in want}, have)
    out["stacks_kept"] = sum(1 for n in have if n.endswith(q.STACKS))
    return out


def _worker(rank, world, tmp, case, kwargs):
    q._worker.__globals__.setdefault(
        case, globals()[case]
    )  # q's gloo worker, running this file's case
    q._worker(rank, world, tmp, case, kwargs)


def _run(case, path):
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, 2, tmp, case, {"path": path})) for r in range(2)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(900)
            if p.is_alive():
                p.terminate()
                p.join()  # before reading its result file and removing the directory
        results = [
            torch.load(f, weights_only = False) if os.path.exists(f) else None
            for f in (os.path.join(tmp, f"result_{r}.pt") for r in range(2))
        ]
    assert all(r is not None and "error" not in r for r in results), results
    return results


def _loss_close(want, got):
    assert got[0] == pytest.approx(want[0], rel = 1e-6, abs = 1e-6), (want, got)
    assert got[1] == pytest.approx(want[1], rel = 1e-3), (want, got)


def test_tp_prequantized_inkling_full_and_sharded_load(ckpt):
    results = _run("_case_tp", ckpt)
    for r in results:
        assert os.path.realpath(r["tp_native"]).startswith(os.path.realpath(REPO) + os.sep), r[
            "tp_native"
        ]
        # (a) the reloaded stacks are 4-bit; 12 MoE blocks shard 2 of 4 experts per rank, bitwise slices.
        assert r["stacks_4bit"]
        assert r["moe_sharded"] == 12 and r["per_rank"] == 2 and r["sliced_equal"], r
        ink._check(r["ref"], r["full"], 1e-4)
        # (b) sharded_loading: bitwise the full load + apply, same losses.
        assert "sharded_error" not in r, r["sharded_error"]
        assert r["sharded_stacks_4bit"]
        # Inkling's converted tensors: transformers' own load keeps each rank's shards as they are set.
        assert r["stats"].get("prequantized") and r["stats"].get("sharded_expert_stacks") == 24, r[
            "stats"
        ]
        assert r["stats"]["sharded_weights"] >= 60, r["stats"]
        assert r["compared"] > 0 and not r["diffs"], r["diffs"][:5]
        assert r["sharded"]["losses"] == r["full"]["losses"]


def test_ep_prequantized_inkling_sharded_load_keeps_only_local_experts(ckpt):
    import test_ep_inkling2_gloo as epi
    results = _run("_case_ep", ckpt)
    for r in results:
        assert os.path.realpath(r["ep_native"]).startswith(os.path.realpath(REPO) + os.sep), r[
            "ep_native"
        ]
        epi._check(r["ref"], r["full"], 5e-4)
        # Each rank reads only its 4-bit experts (12 blocks x 2 stacks); attention and the rest load whole.
        assert r["models"] == 1, r
        assert r["stats"].get("prequantized") and r["stats"]["sharded_expert_stacks"] == 24, r[
            "stats"
        ]
        assert r["stats"]["sharded_weights"] == 0, r["stats"]
        assert r["held"] < r["full_bytes"], (r["held"], r["full_bytes"])
        assert r["compared"] > 0 and not r["diffs"], r["diffs"][:5]
        assert r["sharded"]["losses"] == r["full"]["losses"]


def test_fsdp2_prequantized_inkling_full_and_sharded_load(ckpt):
    results = _run("_case_fsdp2", ckpt)
    for r in results:
        assert os.path.realpath(r["fsdp2_load"]).startswith(os.path.realpath(REPO) + os.sep), r[
            "fsdp2_load"
        ]
        _loss_close(r["ref"], r["full"])
        assert "sharded_error" not in r, r["sharded_error"]
        assert r["models"] == 1 and r["sharded_stacks_4bit"]
        assert r["stats"].get("prequantized") and r["stats"]["sharded_tensors"] > 0, r["stats"]
        assert r["compared"] > 0 and not r["diffs"], r["diffs"][:5]
        assert r["sharded"] == r["full"]


if __name__ == "__main__" and sys.argv[1:2] == ["--make"]:
    _make_checkpoint(sys.argv[2])
