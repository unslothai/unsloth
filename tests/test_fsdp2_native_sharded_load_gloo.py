# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 sharded loading (fsdp2_sharded_loading) on a gloo CPU world of 2 vs the full load + shard_model:
every FSDP2 local shard and 3 training steps are bitwise equal, and each rank reads only its rows."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import dataclasses  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_tp_native_oracle import ARCHS, build  # noqa: E402
from test_tp_native_gloo import _free_port  # noqa: E402

WORLD = 2


def _load_module():
    native = _load_native()
    import importlib
    return native, importlib.import_module("unsloth.models.fsdp2_load")


VLM = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"


def _auto(arch):
    import transformers
    return (
        transformers.AutoModelForImageTextToText
        if arch == "qwen2_5_vl"
        else transformers.AutoModelForCausalLM
    )


def _save(
    arch,
    directory,
    overrides = None,
    remap = False,
):
    if arch == "qwen2_5_vl":
        import transformers

        config = transformers.AutoConfig.from_pretrained(VLM)
        torch.manual_seed(0)
        model = transformers.AutoModelForImageTextToText.from_config(
            config, torch_dtype = torch.float32
        )
    else:
        repo, base = ARCHS[arch]
        model = build(repo, {**base, **(overrides or {})}, lora = False)
    model.generation_config.pad_token_id = None
    model.save_pretrained(directory, max_shard_size = "1MB", safe_serialization = True)
    if remap:
        _prefix_checkpoint_keys(directory)
    return directory


# A VLM-style checkpoint of a text decoder, loaded through from_pretrained(key_mapping = ...) as Unsloth's
# text-only loads do (_apply_text_only_key_mapping).
REMAP = {r"^language_model\.model\.": "model.", r"^language_model\.lm_head\.": "lm_head."}


def _prefix_checkpoint_keys(directory):
    import glob
    import json

    from safetensors.torch import load_file, save_file

    rename = lambda k: "language_model." + k
    for path in glob.glob(os.path.join(directory, "*.safetensors")):
        save_file(
            {rename(k): v for k, v in load_file(path).items()}, path, metadata = {"format": "pt"}
        )
    index = os.path.join(directory, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding = "utf-8") as f:
            data = json.load(f)
        data["weight_map"] = {rename(k): v for k, v in data["weight_map"].items()}
        with open(index, "w", encoding = "utf-8") as f:
            json.dump(data, f)


def _raised(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return str(e)
    return None


def _prep(model, mode):
    if mode in ("lora", "lora_mts"):
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(1)
        if mode == "lora":
            config = LoraConfig(
                r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False
            )
        else:
            # modules_to_save deep-copies a sharded-loaded linear (its chunk and its forward guard).
            config = LoraConfig(
                r = 4,
                lora_alpha = 8,
                target_modules = ["q_proj"],
                modules_to_save = ["down_proj"],
                init_lora_weights = False,
            )
        return get_peft_model(model, config)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def _local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def _shards(model):
    """name -> (type, local tensor, padded FSDP2 storage or None)."""
    padded = {}
    for module in model.modules():
        get = getattr(module, "_get_fsdp_state", None)
        if get is None:
            continue
        group = getattr(get(), "_fsdp_param_group", None)
        for fp in getattr(group, "fsdp_params", None) or ():
            padded[id(fp.sharded_param)] = fp._sharded_param_data.detach().clone()
    return {
        n: (type(p).__name__, _local(p).detach().clone(), padded.get(id(p)))
        for n, p in model.named_parameters()
    }


def _train(
    native,
    model,
    ids,
    steps = 3,
):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-2)
    losses, norms = [], []
    for _ in range(steps):
        loss = model(input_ids = ids, labels = ids).loss
        loss.backward()
        norms.append(float(native.native_clip_grad_norm_(model.parameters(), 1.0)))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
    return losses, norms


IGNORED = ("layers.0.mlp.down_proj.weight", "layers.0.mlp.down_proj.base_layer.weight")


def _ignoring(config, model, ignore):
    """`config` with FSDP2Config.ignored_params = {the weight of `model` named `ignore`} (None: as is)."""
    if not ignore:
        return config
    picked = {p for n, p in model.named_parameters() if n.endswith(ignore)}
    assert len(picked) == 1, ignore
    return dataclasses.replace(config, ignored_params = picked)


def _ignored_state(model, config):
    """Each FSDP2Config.ignored_params entry: (is a live parameter of `model`, kept a plain whole tensor)."""
    live = {id(p): n for n, p in model.named_parameters()}
    return [
        (id(p) in live, type(p).__name__ != "DTensor" and p.device.type != "meta")
        for p in config.ignored_params or ()
    ]


def _case(
    rank,
    world,
    tmp,
    ckpt,
    mode,
    fit,
    arch,
    ignore = None,
    remap = False,
):
    native, loading = _load_module()
    AutoModelForCausalLM = _auto(arch)

    kw = dict(torch_dtype = torch.float32, attn_implementation = "eager")
    if remap:
        kw["key_mapping"] = REMAP
    base_config = native.FSDP2Config(param_dtype = torch.float32, fit_layout = fit, device = "cpu")

    full = _prep(AutoModelForCausalLM.from_pretrained(ckpt, **kw), mode)
    config = _ignoring(base_config, full, ignore)
    native.shard_model(full, config)
    ignored_full = _ignored_state(full, config)

    with loading.fsdp2_sharded_loading() as state:
        model = AutoModelForCausalLM.from_pretrained(ckpt, **kw)
    stats = dict(model._unsloth_fsdp2_load_stats)
    loaded_numel = sum(p.numel() for p in model.parameters())
    full_numel = sum(
        p.numel() for p in AutoModelForCausalLM.from_pretrained(ckpt, **kw).parameters()
    )
    expect_sliced = 0
    for p in model.parameters():
        info = getattr(p, loading._PRELOAD, None)
        if info is not None:
            start, stop = info["rows"]
            assert tuple(p.shape) == (stop - start,) + info["shape"][1:]
            expect_sliced += p.numel() * 4
    ids = torch.randint(0, 100, (2, 7), generator = torch.Generator().manual_seed(1))
    unsharded = _raised(lambda: model(input_ids = ids))
    model = _prep(model, mode)
    config = _ignoring(base_config, model, ignore)
    native.shard_model(model, config)
    ignored_sharded = _ignored_state(model, config)

    a, b = _shards(full), _shards(model)
    diffs = sorted(set(a) ^ set(b))
    for name in a:
        if name not in b:
            continue
        (ta, la, pa), (tb, lb, pb) = a[name], b[name]
        if ta != tb or la.shape != lb.shape or not torch.equal(la, lb):
            diffs.append(name)
        elif (pa is None) != (pb is None) or (pa is not None and not torch.equal(pa, pb)):
            diffs.append(name + " (padded storage)")
    ref = _train(native, full, ids)
    out = _train(native, model, ids)
    a, b = _shards(full), _shards(model)
    trained_diffs = [n for n in a if n in b and not torch.equal(a[n][1], b[n][1])]
    return {
        "diffs": diffs,
        "trained_diffs": trained_diffs,
        "ref": ref,
        "out": out,
        "stats": stats,
        "expect_sliced": expect_sliced,
        "numel": (loaded_numel, full_numel),
        "unsharded": unsharded,
        "models": len(state["models"]),
        "dtensors": sum(1 for v in b.values() if v[0] == "DTensor"),
        "ignored": (ignored_full, ignored_sharded),
    }


def _case_refusals(rank, world, tmp, ckpt):
    native, loading = _load_module()
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoModelForCausalLM

    kw = dict(torch_dtype = torch.float32, attn_implementation = "eager")
    out = {}
    with loading.fsdp2_sharded_loading():
        out["4bit"] = _raised(
            lambda: AutoModelForCausalLM.from_pretrained(ckpt, load_in_4bit = True, **kw)
        )
    with loading.fsdp2_sharded_loading():
        model = AutoModelForCausalLM.from_pretrained(ckpt, **kw)
    mesh = init_device_mesh("cpu", (1, world), mesh_dim_names = ("r", "s"))
    out["2d"] = _raised(
        lambda: native.shard_model(
            model, native.FSDP2Config(param_dtype = torch.float32, mesh = mesh, device = "cpu")
        )
    )
    out["offload"] = _raised(
        lambda: native.shard_model(
            model, native.FSDP2Config(param_dtype = torch.float32, cpu_offload = True, device = "cpu")
        )
    )
    return out


def _case_dora(rank, world, tmp, ckpt):
    """DoRA's magnitude init reads the whole base weight: refused with sharded loading, before PEFT fails on
    this rank's rows."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    native, loading = _load_module()
    kw = dict(torch_dtype = torch.float32, attn_implementation = "eager")
    with loading.fsdp2_sharded_loading():
        model = AutoModelForCausalLM.from_pretrained(ckpt, **kw)
    dora = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", use_dora = True)
    out = {"dora": _raised(lambda: get_peft_model(model, dora))}
    # The full load still takes DoRA (the refusal is scoped to sharded-loaded weights).
    full = AutoModelForCausalLM.from_pretrained(ckpt, **kw)
    out["full"] = _raised(lambda: get_peft_model(full, dora))
    return out


CASES = {"parity": _case, "refusals": _case_refusals, "dora": _case_dora}


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
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(
    case,
    world = WORLD,
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
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


@pytest.mark.parametrize(
    "arch,overrides,mode,fit",
    [
        (
            "llama",
            {"intermediate_size": 37},
            "lora",
            False,
        ),  # odd rows: uneven chunks, FSDP2 padding
        (
            "llama",
            {"intermediate_size": 37},
            "lora",
            True,
        ),  # fit layout: frozen chunks gathered back
        ("llama", {"intermediate_size": 37}, "full", False),
        ("qwen3", None, "lora", False),
        ("qwen3", None, "full", False),
        ("qwen2_5_vl", None, "lora", False),  # vision tower + projector linears sliced too
        # FSDP2Config.ignored_params naming a sharded-loaded weight: gathered whole and kept replicated.
        ("llama", {"intermediate_size": 37}, "full", "ignore"),
        ("llama", {"intermediate_size": 37}, "lora", "ignore"),
        ("llama", {"intermediate_size": 37}, "lora_mts", False),
        # A load-specific key_mapping (VLM-prefixed checkpoint of a text decoder).
        ("llama", {"intermediate_size": 37}, "lora", "remap"),
    ],
)
def test_sharded_loading_matches_full_load_shard_model(arch, overrides, mode, fit):
    ignore, remap = None, fit == "remap"
    if fit == "ignore":
        fit, ignore = False, IGNORED[mode == "lora"]
    if remap:
        fit = False
    with tempfile.TemporaryDirectory() as ckpt:
        _save(arch, ckpt, overrides, remap = remap)
        results = _spawn(
            "parity", ckpt = ckpt, mode = mode, fit = fit, arch = arch, ignore = ignore, remap = remap
        )
    sliced = [r["stats"]["read_bytes_sliced"] for r in results]
    for rank, res in enumerate(results):
        assert res["models"] == 1
        ignored_full, ignored_sharded = res["ignored"]
        assert ignored_sharded == ignored_full == ([(True, True)] if ignore else []), res["ignored"]
        assert res["diffs"] == [], (rank, res["diffs"][:6])
        assert res["dtensors"] > 0
        # 3 steps bitwise equal to the full load (losses, clipped grad norms, final local shards).
        assert res["out"] == res["ref"], (rank, res["out"], res["ref"])
        assert res["trained_diffs"] == [], (rank, res["trained_diffs"][:6])
        assert res["unsharded"] and "shard it first" in res["unsharded"], res["unsharded"]
        stats = res["stats"]
        # This rank read only its own rows of the sliced weights, everything else whole, nothing twice.
        assert stats["read_bytes_sliced"] == res["expect_sliced"] > 0, stats
        assert stats["read_bytes_full"] == stats["replicated_numel"] * 4, stats
        loaded, full = res["numel"]
        assert loaded == full - stats["sharded_numel"] + res["expect_sliced"] // 4 < full
    assert sum(sliced) == results[0]["stats"]["sharded_numel"] * 4


def test_sharded_loading_refusals():
    with tempfile.TemporaryDirectory() as ckpt:
        _save("llama", ckpt, None)
        results = _spawn("refusals", ckpt = ckpt)
    for res in results:
        assert "16-bit safetensors checkpoints only" in (res["4bit"] or ""), res["4bit"]
        assert "QLoRA" in res["4bit"]
        assert "1-D shard mesh" in (res["2d"] or ""), res["2d"]
        assert "cpu_offload" in (res["offload"] or ""), res["offload"]


def test_sharded_loading_refuses_dora():
    with tempfile.TemporaryDirectory() as ckpt:
        _save("llama", ckpt, {"intermediate_size": 37})
        results = _spawn("dora", ckpt = ckpt)
    for res in results:
        assert res["full"] is None, res["full"]
        assert "use_dora" in (res["dora"] or "") and "full" in res["dora"], res["dora"]


def test_chunk_rows_match_fully_shard():
    _, loading = _load_module()
    for rows in (1, 2, 5, 7, 37, 64):
        for world in (1, 2, 3, 4, 8):
            chunks = list(torch.arange(rows).chunk(world))
            chunks += [torch.empty(0, dtype = torch.long)] * (world - len(chunks))
            for rank in range(world):
                start, stop = loading._chunk_rows(rows, world, rank)
                assert torch.equal(torch.arange(rows)[start:stop], chunks[rank]), (
                    rows,
                    world,
                    rank,
                )


def test_env_wrapper_is_inert_without_the_flag(monkeypatch):
    _, loading = _load_module()
    monkeypatch.delenv("UNSLOTH_FSDP2_SHARDED_LOAD", raising = False)
    calls = []
    wrapped = loading.sharded_load_env(lambda *a, **k: calls.append((a, k)) or "ok")
    assert wrapped(1, x = 2) == "ok" and calls == [((1,), {"x": 2})]
    assert loading.fsdp2_sharded_load_requested() is False
    monkeypatch.setenv("UNSLOTH_FSDP2_SHARDED_LOAD", "1")
    monkeypatch.setenv("UNSLOTH_FSDP2", "0")
    assert loading.fsdp2_sharded_load_requested() is False
