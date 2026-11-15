# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Shared harness for the frozen lookup table checks under FSDP2 / TP / PP; `fault` injects a break per test."""

import contextlib
import importlib.machinery
import importlib.util
import inspect
import json
import multiprocessing
import os
import struct
import sys
import tempfile
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = 512
SEQ = 48
TABLE = "ple.ple_embedding.ngram_embedding"
_TABLE_KEYS = ("ngram_embedding", "engram_tables.")  # Qwen4Exp PLE, DeepSeek-V4.1 engram


def _is_table_key(k):
    return any(t in k for t in _TABLE_KEYS)


CONFIG = dict(
    vocab_size = VOCAB,
    bos_token_id = VOCAB - 1,
    eos_token_id = VOCAB - 1,
    hidden_size = 64,
    num_hidden_layers = 4,
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    full_attention_interval = 4,
    num_attention_heads = 4,
    num_key_value_heads = 2,
    head_dim = 16,
    hc_count = 4,
    hc_lowrank = 8,
    indexer_budget = 32,
    indexer_compress_ratio = 4,
    indexer_head_dim = 16,
    indexer_kv_heads = 1,
    indexer_n_heads = 2,
    linear_conv_kernel_dim = 4,
    linear_key_head_dim = 16,
    linear_num_key_heads = 2,
    linear_num_value_heads = 4,
    linear_value_head_dim = 16,
    num_experts = 4,
    num_experts_per_tok = 2,
    moe_intermediate_size = 32,
    shared_expert_intermediate_size = 32,
    ple_layer_ids = [2],  # layer 1 holds the table
    ple_embed_dim = 64,
    heads_per_ngram = 8,
    ngram_size = 3,
    ngram_vocab_size_base = 509,
    split_ngram_parts = 4,
    make_ngram_vocab_size_divisible_by = 8,
    ple_conv_kernel_size = 4,
    mtp_num_hidden_layers = 0,
    max_position_embeddings = 512,
    tie_word_embeddings = False,
)


def load_module(name):
    """unsloth/models/<name>.py (or unsloth/<name>.py) without importing the unsloth package (it needs a GPU)."""
    full = f"unsloth.models.{name}"
    if full in sys.modules:
        return sys.modules[full]
    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = types.ModuleType("unsloth.models")
    models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    for package in (pkg, models):
        package.__spec__ = importlib.machinery.ModuleSpec(package.__name__, None, is_package = True)
        package.__spec__.submodule_search_locations = package.__path__
    sys.modules.setdefault("unsloth", pkg)
    sys.modules.setdefault("unsloth.models", models)
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(REPO, "unsloth", "models", f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


def available():
    """Why these tests cannot run here, or None."""
    if importlib.util.find_spec("transformers.models.qwen4_exp") is None:
        return "transformers without qwen4_exp (5.22+)"
    try:
        from unsloth_zoo.frozen_table_store import UnslothFrozenTableEmbedding  # noqa: F401
        from unsloth_zoo.temporary_patches.qwen4_exp import patch_qwen4_exp  # noqa: F401
    except Exception as e:
        return f"unsloth_zoo without the frozen table store ({type(e).__name__})"
    return None


def make_checkpoint(
    path,
    big = False,
    table = True,
):
    """Save a tiny Qwen4Exp in bf16, the table as row shards over several files. big: a ~128 MiB table."""
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    cfg = dict(CONFIG)
    if big:
        cfg.update(ngram_vocab_size_base = 524287, ple_embed_dim = 128)
    if not table:
        cfg["ple_layer_ids"] = []
    torch.manual_seed(0)
    model = Qwen4ExpForCausalLM(Qwen4ExpTextConfig(**cfg))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("gate.weight") and p.ndim == 2:
                p.normal_(0, 0.5)
            elif "ngram_embedding" in name:
                p.normal_(0, 1.0)
    model.to(torch.bfloat16).save_pretrained(path, max_shard_size = "8MB" if big else "200KB")
    return path


def _headers(path):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            full = os.path.join(path, f)
            with open(full, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            for k in header:
                if k != "__metadata__":
                    out[k] = full
    return out


def table_files(path):
    """{shard key: file} of the n-gram table."""
    return {
        k: f
        for k, f in _headers(path).items()
        if ".ngram_embedding." in k and k.endswith(".weight")
    }


def reference_table(path):
    """The whole table read straight from the files (row shards in order)."""
    from safetensors import safe_open

    files = table_files(path)
    keys = sorted(
        files, key = lambda k: int(k.split("shard_")[1].split(".")[0]) if "shard_" in k else 0
    )
    parts = []
    for k in keys:
        with safe_open(files[k], framework = "pt") as f:
            parts.append(f.get_tensor(k))
    return torch.cat(parts)


class _ReadRecorder:
    """Wraps transformers' safe_open: records every checkpoint key whose tensor is materialized."""

    def __init__(self):
        self.read = []

    def install(self):
        import transformers.modeling_utils as mu

        original = mu.safe_open
        recorder = self

        class _Slice:
            def __init__(self, s, key):
                self._s, self._key = s, key

            def __getitem__(self, item):
                recorder.read.append(self._key)
                return self._s[item]

            def __getattr__(self, name):
                return getattr(self._s, name)

        class _File:
            def __init__(self, f):
                self._f = f

            def __enter__(self):
                self._f.__enter__()
                return self

            def __exit__(self, *a):
                return self._f.__exit__(*a)

            def get_slice(self, key):
                return _Slice(self._f.get_slice(key), key)

            def get_tensor(self, key):
                recorder.read.append(key)
                return self._f.get_tensor(key)

            def __getattr__(self, name):
                return getattr(self._f, name)

        self._original = original
        mu.safe_open = lambda *a, **k: _File(original(*a, **k))
        return self

    def uninstall(self):
        import transformers.modeling_utils as mu
        mu.safe_open = self._original


def load(path, mode):
    """The checkpoint with UNSLOTH_FROZEN_TABLE = mode, LoRA on Unsloth's default targets (QSA indexer out)."""
    os.environ["UNSLOTH_FROZEN_TABLE"] = mode
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    if not torch.cuda.is_available():
        os.environ.setdefault(
            "UNSLOTH_ZOO_DISABLE_GPU_INIT", "1"
        )  # CPU gloo ranks: no accelerator probe
    from unsloth_zoo.temporary_patches.qwen4_exp import patch_qwen4_exp
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, Qwen4ExpForCausalLM

    patch_qwen4_exp()
    recorder = _ReadRecorder().install()
    cfg = AutoConfig.from_pretrained(path)
    cfg._attn_implementation = "eager"
    try:
        model = Qwen4ExpForCausalLM.from_pretrained(path, config = cfg, dtype = torch.float32)
    finally:
        recorder.uninstall()
    names = {
        n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    }
    names.discard("lm_head")
    regex = r"^(?!.*\.indexer\.).*\.(" + "|".join(sorted(names)) + r")$"
    torch.manual_seed(1)
    model = get_peft_model(
        model, LoraConfig(r = 4, lora_alpha = 8, target_modules = regex, init_lora_weights = False)
    )
    return model, recorder.read


def table_modules(model):
    return [(n, m) for n, m in model.named_modules() if n.endswith("ngram_embedding")]


def _is_dtensor(t):
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return isinstance(t, DTensor)


# /proc is Linux only: elsewhere RssAnon / fd checks read None and are skipped.
_PROC = os.path.isfile("/proc/self/status")


@contextlib.contextmanager
def cpu_forwards():
    """Force torch paths: FLA / causal-conv1d and the MoE backend are picked per process, not device; Triton rejects CPU."""
    undo = []
    try:
        import transformers.models.qwen4_exp.modeling_qwen4_exp as module
    except ImportError:
        module = None
    for fn in (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        if module is not None and callable(getattr(module, fn, None)):
            undo.append((fn, getattr(module, fn)))
            setattr(module, fn, inspect.unwrap(getattr(module, fn)))
    backend = os.environ.get("UNSLOTH_MOE_BACKEND")
    os.environ["UNSLOTH_MOE_BACKEND"] = "native_torch"
    _clear_moe_backend()
    try:
        yield
    finally:
        for fn, original in undo:
            setattr(module, fn, original)
        if backend is None:
            os.environ.pop("UNSLOTH_MOE_BACKEND", None)
        else:
            os.environ["UNSLOTH_MOE_BACKEND"] = backend
        _clear_moe_backend()


def _clear_moe_backend():
    # The compiled cache holds its own moe_utils copy with its own cache.
    for name, mod in list(sys.modules.items()):
        if "moe_utils" not in name or not isinstance(getattr(mod, "__dict__", None), dict):
            continue
        clear = getattr(mod.__dict__.get("select_moe_backend"), "cache_clear", None)
        if callable(clear):
            clear()


def _rss_anon_kb():
    if not _PROC:
        return None
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1])
    return None


def _open_fds(paths):
    if not _PROC:
        return None
    paths = {os.path.realpath(p) for p in paths}
    n = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.path.realpath(os.readlink(f"/proc/self/fd/{fd}")) in paths:
                n += 1
        except OSError:
            pass
    return n


def batch(bsz):
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, VOCAB - 1, (bsz, SEQ), generator = g)
    return ids, ids.clone()


def _capture(model, sink):
    hooks = []
    for name, module in table_modules(model):
        hooks.append(
            module.register_forward_hook(
                lambda m, args, out, name = name: sink.append(
                    (name, args[0].detach().cpu().clone(), out.detach().cpu().clone())
                )
            )
        )
    return hooks


def _reference_ids(path, mode, chunks):
    """The table ids one process computes for each chunk of tokens (same checkpoint, same mode)."""
    out = []
    with cpu_forwards():
        model, _ = load(path, mode)
        with torch.no_grad():
            for ids in chunks:
                seen = []
                hooks = _capture(model, seen)
                model(input_ids = ids, use_cache = False)
                for h in hooks:
                    h.remove()
                out.append([s[1] for s in seen])
    return out


def worker_case(
    rank,
    world,
    tmp,
    mode,
    table_mode,
    path,
    schedule = "gpipe",
    n_mb = 2,
    gc = False,
    fault = None,
    steps = 2,
    sharded_load = False,
    second_adapter = False,
    layers_per_stage = None,
    no_bitsandbytes = False,
    install_dcp = True,
):
    import gc as gc_module

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    rec = {"rank": rank}
    bsz = max(2, world) if mode == "fsdp2" else max(2, n_mb)
    ids, labels = batch(bsz)
    rss0 = _rss_anon_kb()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = (mode,))[mode]
    loading = contextlib.nullcontext()
    if sharded_load:  # meta-device init, each rank reads its own part of the checkpoint
        if mode == "fsdp2":
            loading = load_module("fsdp2_load").fsdp2_sharded_loading(mesh)
        elif mode == "tp":
            loading = load_module("tp_native").sharded_loading(mesh)
        else:
            loading = load_module("pp_load").pipeline_sharded_loading(mesh, schedule = schedule)
    try:
        with loading:
            model, materialized = load(path, mode = table_mode)
    except (ValueError, RuntimeError, NotImplementedError) as e:
        rec["load_refused"] = f"{type(e).__name__}: {e}"[:400]
        return rec
    if no_bitsandbytes:  # from here on bitsandbytes cannot be imported (an install without it)
        sys.modules["bitsandbytes.nn"] = None
    if second_adapter:  # a named adapter: PEFT saves it in <directory>/second
        from peft import LoraConfig
        model.add_adapter("second", LoraConfig(r = 2, target_modules = ["o_proj"]))
    files = set(table_files(path).values())
    table_keys = set(table_files(path))
    rec["table_keys_materialized"] = sorted(set(materialized) & table_keys)
    zoo_store = importlib.import_module("unsloth_zoo.frozen_table_store")
    if fault == "memo":
        # Broken memo: any lookup of the same shape on the same step reuses the previous rows.
        def _bad_lookup(
            self,
            ids_,
            _orig = zoo_store.UnslothFrozenTableEmbedding._lookup,
        ):
            memo = self._memo
            if isinstance(memo, tuple):  # an unsloth_zoo with a single-entry memo
                memo = [memo]
            if memo and memo[0][2][2] == tuple(ids_.shape):
                return memo[0][3]
            return _orig(self, ids_)

        zoo_store.UnslothFrozenTableEmbedding._lookup = _bad_lookup
    if fault == "rows" and rank == world - 1:
        orig_gather = zoo_store.FrozenTableStore.gather
        zoo_store.FrozenTableStore.gather = (
            lambda self, i, dtype = None, **kw: orig_gather(self, i, dtype, **kw) + 1
        )
    if gc:
        # GradientCheckpointingLayer checkpoints in training mode only (every dropout here is 0).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
        model.train()
    if mode == "fsdp2":
        native = load_module("fsdp2_native")
        if fault == "shard_table":
            native._frozen_table_params = lambda m: set()
        native.shard_model(
            model,
            native.FSDP2Config(
                device = "cpu", param_dtype = torch.float32, mesh = mesh, shard_frozen = True
            ),
        )
        per = bsz // world
        mine = [ids[rank * per : (rank + 1) * per]]
        step = lambda: model(
            input_ids = mine[0], labels = labels[rank * per : (rank + 1) * per], use_cache = False
        ).loss
        params = [p for p in model.parameters() if p.requires_grad]
    elif mode == "tp":
        tpn = load_module("tp_native")
        model = tpn.apply_tensor_parallel(model, mesh)
        mine = [ids]
        step = lambda: model(input_ids = ids, labels = labels, use_cache = False).loss
        params = load_module("fsdp2_native").split_param_groups(
            [p for p in model.parameters() if p.requires_grad]
        )
    else:
        ppn = load_module("pp_native")
        # leak_table: a reference kept to the table module, so a stage without its layer would still hold it.
        rec_leak = table_modules(model) if fault == "leak_table" else None
        pp = ppn.apply_pipeline_parallel(
            model,
            mesh,
            n_microbatches = n_mb,
            schedule = schedule,
            config = ppn.PPConfig(layers_per_stage = layers_per_stage),
        )
        mine = list(ids.chunk(n_mb))
        step = lambda: pp.step(ids, labels)
        params = pp.parameters()
        rec["owns_table"] = pp.start <= 1 < pp.end
    gc_module.collect()
    tables = table_modules(model)
    if mode == "pp" and fault == "leak_table":
        tables = tables or rec_leak
    rec["table_kinds"] = sorted({type(m).__name__ for _, m in tables})
    rec["table_params"] = sorted(n for n, _ in model.named_parameters() if _is_table_key(n))
    rec["table_buffers"] = sorted(n for n, _ in model.named_buffers() if _is_table_key(n))
    tensors = [t for _, m in tables for t in list(m.parameters()) + list(m.buffers())]
    for _, m in tables:
        store = getattr(m, "frozen_table_store", None)
        if store is not None:
            tensors += list(store._device_planes.values())
    rec["table_dtensor"] = any(_is_dtensor(t) for t in tensors)
    rec["fds"] = _open_fds(files)
    rec["table_bytes"] = sum(os.path.getsize(f) for f in files)
    calls = []
    hooks = _capture(model, calls)
    if zoo_store is not None:
        zoo_store.reset_frozen_table_stats()
    opt = torch.optim.SGD(params, lr = 1e-2)
    rec["table_memo"] = []
    for _ in range(steps):
        loss = step()
        if mode != "pp":
            loss.backward()
        else:
            rec["table_memo"].append(pp.last_table_memo)
        opt.step()
        opt.zero_grad(set_to_none = True)
        if zoo_store is not None:
            zoo_store.frozen_table_next_step()
    for h in hooks:
        h.remove()
    rec["stats"] = dict(zoo_store.FROZEN_TABLE_STATS) if zoo_store is not None else {}
    rec["rss_anon_delta_kb"] = None if rss0 is None else _rss_anon_kb() - rss0
    rec["fds_after"] = _open_fds(files)
    rec["calls"] = calls
    rec["chunks"] = mine
    # DCP state dicts (Trainer checkpoints under FSDP2 / TP: accelerate's get_state_dict, Unsloth's adapter gather)
    # through Unsloth's wrappers: no table key, so DCP never writes or loads it, and no AttributeError on the
    # table's checkpoint names. The unwrapped torch call shows what the wrappers prevent.
    import torch.distributed.checkpoint.state_dict as dsd

    fsave = load_module("fsdp2_save")
    if install_dcp:  # else only what the parallel setup itself installed
        fsave.install_dcp_bnb_state_dict()
    native = load_module("fsdp2_native")
    for key, options in (
        ("adapter", dsd.StateDictOptions(ignore_frozen_params = True)),
        ("full", None),
    ):
        try:
            sd = (
                dsd.get_model_state_dict(model, options = options)
                if options
                else dsd.get_model_state_dict(model)
            )
            rec[f"dcp_{key}_table_keys"] = sum(_is_table_key(k) for k in sd)
        except Exception as e:
            rec[f"dcp_{key}_error"] = f"{type(e).__name__}: {e}"[:300]
        sd = None
    raw = getattr(dsd.get_model_state_dict, "__wrapped__", None) and inspect.unwrap(
        dsd.get_model_state_dict
    )
    if raw is not None:
        try:
            raw(model)
            rec["raw_dcp_error"] = None
        except Exception as e:
            rec["raw_dcp_error"] = type(e).__name__
    if mode == "fsdp2":
        full = native.native_full_state_dict(model, trainable_only = False)
        keys = [k for k in full if _is_table_key(k)]
        rec["full_save_table_keys"] = len(keys)
        rec["full_save_table_dtensor"] = any(_is_dtensor(full[k]) for k in keys)
        full = None
    save_dir = os.path.join(tmp, f"adapter_{rank}")
    if mode == "pp":
        pp.save_adapter(save_dir)
    elif mode == "tp":
        load_module("tp_native").save_tp_adapter(model, save_dir)
    else:
        with load_module("fsdp2_save").fsdp2_gathered(model, trainable_only = True) as writer:
            if writer:
                model.save_pretrained(save_dir)
        dist.barrier()
    for key, directory in (
        ("provenance", save_dir),
        ("second_provenance", os.path.join(save_dir, "second")),
    ):
        record = os.path.join(directory, "frozen_tables.json")
        if os.path.exists(record):
            with open(record) as f:
                rec[key] = json.load(f)
    saved = os.path.join(save_dir, "adapter_model.safetensors")
    if os.path.exists(saved):
        from safetensors import safe_open
        with safe_open(saved, framework = "pt") as f:
            rec["saved_table_keys"] = [k for k in f.keys() if _is_table_key(k)]
    return rec


def _worker(rank, world, port, tmp, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        CUDA_VISIBLE_DEVICES = "",
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 300)
    )
    try:
        with cpu_forwards():
            res = worker_case(rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


def run(world, **kwargs):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, kwargs)) for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(900)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [torch.load(os.path.join(tmp, f"r{r}.pt"), weights_only = False) for r in range(world)]


def violations(results, mode, table_mode, path):
    """Every broken property, as strings (empty = all hold)."""
    out = []
    table = reference_table(path)
    if table_mode in ("disk", "ram"):
        # The adapter leaves the table out: rank 0 writes frozen_tables.json next to it, naming exactly the
        # checkpoint's table tensors (every stage's under pipeline parallelism).
        record = results[0].get("provenance")
        if not record:
            out.append("rank 0: no frozen_tables.json next to the adapter")
        else:
            names = {t["name"] for p in record["tables"].values() for t in p["tensors"]}
            if names != set(table_files(path)):
                out.append(
                    f"rank 0: frozen_tables.json lists {sorted(names)[:3]}, not the checkpoint's table"
                )
    for res in results:
        r = res["rank"]
        if res["table_keys_materialized"] and table_mode in ("disk", "ram"):
            out.append(f"rank {r}: loader read table keys {res['table_keys_materialized'][:2]}")
        if res["table_dtensor"]:
            out.append(f"rank {r}: a table tensor is a DTensor (sharded / all-gathered)")
        if table_mode in ("disk", "ram") and (res["table_params"] or res["table_buffers"]):
            out.append(
                f"rank {r}: {table_mode} table holds parameters / buffers {res['table_params'][:2]}"
            )
        for key in ("adapter", "full"):
            if res.get(f"dcp_{key}_error"):
                out.append(f"rank {r}: {key} DCP state dict failed: {res[f'dcp_{key}_error']}")
            elif res.get(f"dcp_{key}_table_keys") and (
                key == "adapter" or table_mode in ("disk", "ram")
            ):
                out.append(
                    f"rank {r}: the {key} DCP state dict holds {res[f'dcp_{key}_table_keys']} table keys"
                )
        if res.get("full_save_table_dtensor"):
            out.append(f"rank {r}: the full save gathered the table")
        if res.get("saved_table_keys"):
            out.append(f"rank {r}: the adapter save holds table keys {res['saved_table_keys'][:2]}")
        owns = res.get("owns_table", True)
        if mode == "pp" and not owns:
            if res["table_kinds"] or res["calls"] or res["fds_after"]:
                out.append(
                    f"rank {r}: a stage without the table layer holds it ({res['table_kinds']}, "
                    f"{len(res['calls'])} lookups, {res['fds_after']} open fds)"
                )
            continue
        if table_mode in ("disk", "ram") and res["fds_after"] == 0:
            out.append(f"rank {r}: the {table_mode} table has no open file (not streaming)")
        if not res["calls"]:
            out.append(f"rank {r}: no table lookup")
        for name, ids, rows in res["calls"]:
            want = table.to(rows.dtype)[ids.long()]
            if not torch.equal(rows, want):
                out.append(f"rank {r}: {name} rows differ from the file rows for its ids")
                break
    return out


def check_ids(
    results,
    path,
    table_mode,
    mode,
    n_mb = 2,
):
    """Each rank's lookup ids equal those one process computes for the same tokens (per micro-batch for PP)."""
    out = []
    for res in results:
        if mode == "pp" and not res.get("owns_table", True):
            continue
        ref = _reference_ids(path, table_mode, res["chunks"])
        got = [ids for _, ids, _ in res["calls"]]
        # Every forward lookup (and a checkpoint recompute) must be one of the reference lookups of its chunk, in
        # order of the chunks for the first appearance of each.
        flat = [x for per in ref for x in per]
        for ids in got:
            if not any(torch.equal(ids, x) for x in flat):
                out.append(f"rank {res['rank']}: a lookup's ids match no reference chunk")
                break
        firsts = []
        for ids in got:
            if not any(torch.equal(ids, f) for f in firsts):
                firsts.append(ids)
        if len(firsts) < len(flat):
            out.append(
                f"rank {res['rank']}: {len(firsts)} distinct lookups, expected {len(flat)} (chunks merged)"
            )
    return out


def big_checkpoints(tmp_path_factory):
    if not _PROC:
        import pytest
        pytest.skip("anonymous RSS (RssAnon) is read from /proc/self/status: Linux only")
    d = tmp_path_factory.mktemp("big_table")
    return make_checkpoint(str(d / "with"), big = True), make_checkpoint(
        str(d / "without"), big = True, table = False
    )


def memory_violations(with_table, without_table, table_mode, path):
    """Per-rank anonymous RSS growth with the table minus without it (ram / disk tables stay page cache)."""
    table_bytes = reference_table(path).numel() * 2  # bf16 on disk
    out, rows = [], []
    for a, b in zip(with_table, without_table):
        diff = (a["rss_anon_delta_kb"] - b["rss_anon_delta_kb"]) * 1024
        rows.append((a["rank"], a.get("owns_table", True), diff))
        owns = a.get("owns_table", True)
        if table_mode in ("disk", "ram") or not owns:
            if diff > 0.25 * table_bytes:
                out.append(
                    f"rank {a['rank']}: {table_mode} table run holds {diff / 2**20:.1f} MiB more anonymous "
                    f"memory than without it (table {table_bytes / 2**20:.1f} MiB)"
                )
        elif diff < 0.75 * table_bytes:
            out.append(
                f"rank {a['rank']}: gpu table not resident ({diff / 2**20:.1f} MiB for {table_bytes / 2**20:.1f})"
            )
    print(
        "MEMORY",
        table_mode,
        f"table {table_bytes / 2**20:.1f} MiB",
        [(r, o, round(d / 2**20, 1)) for r, o, d in rows],
    )
    return out
