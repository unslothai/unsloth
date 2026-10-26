# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""apply_tensor_parallel / apply_pipeline_parallel across architectures, gloo CPU, 2 ranks, fp32, vs one
process: every architecture either trains like one process or is refused before the model is touched.

The configs are tiny shrinks of each model type's default config, built offline."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    """unsloth/models/<name>.py without importing the unsloth package (which needs a GPU)."""
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


tpn = _load("tp_native")
ppn = _load("pp_native")

_SMALL = dict(vocab_size = 512, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 4)
_HEADS = dict(num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16)
# arch -> (model_type, config kwargs)
CONFIGS = {
    "llama": ("llama", {**_SMALL, **_HEADS}),
    # GPT-J: parallel attention + MLP, fc_in / fc_out, partial rotary
    "gptj": (
        "gptj",
        dict(vocab_size = 512, n_embd = 64, n_layer = 4, n_head = 4, rotary_dim = 8, n_inner = 128),
    ),
    # per-head query_layernorm / key_layernorm
    "hunyuan_v1_dense": (
        "hunyuan_v1_dense",
        {
            **_SMALL,
            **_HEADS,
            "use_qk_norm": True,
            "attention_dropout": 0.0,
            "tie_word_embeddings": False,
        },
    ),
    # short-conv layers (replicated) between attention layers with q_layernorm / k_layernorm
    "lfm2": (
        "lfm2",
        {
            **_SMALL,
            **_HEADS,
            "layer_types": ["conv", "full_attention", "conv", "full_attention"],
            "full_attn_idxs": [1, 3],
        },
    ),
    # xIELU activation with scalar parameters inside the MLP
    "apertus": ("apertus", {**_SMALL, "num_attention_heads": 4, "num_key_value_heads": 2}),
    # activation sparsity: mean / std over the intermediate features
    "gemma3n_text": (
        "gemma3n_text",
        {
            **_SMALL,
            **_HEADS,
            "intermediate_size": 128,
            "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
            "hidden_size_per_layer_input": 16,
            "num_kv_shared_layers": 0,
            "activation_sparsity_pattern": [0.95, 0.95, 0.0, 0.0],
            "laurel_rank": 8,
        },
    ),
    # attention output reshaped with self.embed_dim / the input width
    "xglm": (
        "xglm",
        dict(vocab_size = 512, d_model = 64, ffn_dim = 128, num_layers = 4, attention_heads = 4),
    ),
    "trocr": (
        "trocr",
        dict(
            vocab_size = 512,
            d_model = 64,
            decoder_layers = 4,
            decoder_attention_heads = 4,
            decoder_ffn_dim = 128,
        ),
    ),
    # fused qkv_proj in mp_num groups, not [q, k, v] blocks
    "codegen": (
        "codegen",
        dict(vocab_size = 512, n_embd = 64, n_layer = 4, n_head = 4, rotary_dim = 8, n_inner = 128),
    ),
    # learned position tables before the layers
    "gpt2": ("gpt2", dict(vocab_size = 512, n_embd = 64, n_layer = 4, n_head = 4, n_inner = 128)),
    "opt": (
        "opt",
        dict(
            vocab_size = 512,
            hidden_size = 64,
            num_hidden_layers = 4,
            ffn_dim = 128,
            word_embed_proj_dim = 64,
            num_attention_heads = 4,
        ),
    ),
    # CrossEntropyLoss mean per call (no num_items_in_batch)
    "bart": (
        "bart",
        dict(
            vocab_size = 512,
            d_model = 64,
            decoder_layers = 4,
            decoder_ffn_dim = 128,
            decoder_attention_heads = 4,
            encoder_layers = 1,
            encoder_ffn_dim = 128,
            encoder_attention_heads = 4,
        ),
    ),
}


def _zero_dropout(config):
    for key in list(config.to_dict()):
        value = getattr(config, key, None)
        if isinstance(value, float) and ("dropout" in key or "pdrop" in key):
            setattr(config, key, 0.0)


def build(
    arch,
    lora = True,
    seed = 0,
    share = None,
):
    from transformers import AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    model_type, kwargs = CONFIGS[arch]
    if model_type not in CONFIG_MAPPING:
        pytest.skip(f"transformers has no {model_type}")
    config = CONFIG_MAPPING[model_type](**kwargs)
    _zero_dropout(config)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    try:
        model = AutoModelForCausalLM.from_config(config, dtype = torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.normal_(0, 0.1)
            elif "norm" in name and p.ndim == 1:
                p.add_(torch.randn_like(p) * 0.1)
    if (
        share is not None
    ):  # layers[j] is layers[i]: a block reused by several layers (Zamba2 / ALBERT style)
        i, j = share
        layers = tpn._decoder_layers(model)[1]
        layers[j] = layers[i]
    if not lora:
        return model.train()
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed + 1)
    kw = dict(
        r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False, lora_dropout = 0.0
    )
    return get_peft_model(model, LoraConfig(**kw)).train()


def batch(
    model,
    bsz = 4,
    seq = 8,
):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 500, (bsz, seq), generator = g)
    labels = ids.clone()
    labels[0, :3] = -100
    labels[-1, -2:] = -100
    return ids, labels


def _clip_fp64(params, max_norm):
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.stack([g.double().square().sum() for g in grads]).sum().sqrt()
    coef = torch.clamp(max_norm / (total + 1e-6), max = 1.0)
    for g in grads:
        g.mul_(coef.to(g.dtype))
    return float(total)


def _reference(
    model,
    ids,
    labels,
    steps = 2,
):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-2, foreach = False)
    losses, norms, grads = [], [], None
    for step in range(steps):
        out = model(input_ids = ids, labels = labels, use_cache = False)
        out.loss.backward()
        if step == 0:
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        norms.append(_clip_fp64(params, 1.0))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss))
    return {"losses": losses, "norms": norms, "grads": grads}


def _fingerprint(model):
    """Everything a refusal must leave as it was."""
    return {
        "params": {
            n: (type(p).__name__, p.detach().clone(), p.requires_grad)
            for n, p in model.named_parameters()
        },
        "modules": [(n, type(m).__name__) for n, m in model.named_modules()],
        "marks": sorted(
            (n, k)
            for n, m in model.named_modules()
            for k in m.__dict__
            if k.startswith("_unsloth") or k in ("forward", "generate")
        ),
        "hooks": sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()),
    }


def _unchanged(before, after):
    if before["modules"] != after["modules"] or before["marks"] != after["marks"]:
        return False
    if before["hooks"] != after["hooks"] or set(before["params"]) != set(after["params"]):
        return False
    for n, (cls, value, grad) in before["params"].items():
        cls2, value2, grad2 = after["params"][n]
        if cls != cls2 or grad != grad2 or not torch.equal(value, value2):
            return False
    return True


def _refusal(apply, model):
    before = _fingerprint(model)
    try:
        apply(model)
    except ValueError as e:
        return {"error": str(e), "unchanged": _unchanged(before, _fingerprint(model))}
    except Exception as e:
        return {
            "error": f"crash {type(e).__name__}: {e}",
            "unchanged": _unchanged(before, _fingerprint(model)),
        }
    return {"error": None, "unchanged": None}


# ---------------------------------------------------------------------------------------------- cases

TP_PASS = {
    "gptj": None,
    "hunyuan_v1_dense": None,
    "lfm2": None,
    "apertus": None,
    "llama_shared": (0, 1),
}
TP_REFUSE = {
    "gemma3n_text": "reduces over its features",
    "xglm": "the global width",
    "trocr": "keeps the input width",
    "codegen": "does not slice the fused 'qkv_proj'",
}


def _case_tp(rank, world, tmp):
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]
    out = {}
    for name, share in TP_PASS.items():
        try:
            out[name] = _tp_parity(mesh, name.replace("_shared", ""), share)
        except (
            Exception
        ) as e:  # recorded per architecture: one failing case must not hide the others
            out[name] = {"crash": f"{type(e).__name__}: {e}"}
    for arch in TP_REFUSE:
        out[arch] = _refusal(lambda m: tpn.apply_tensor_parallel(m, mesh), build(arch))
    return out


def _tp_parity(mesh, arch, share):
    ref = build(arch, share = share)
    ids, labels = batch(ref, bsz = 2)
    model = build(arch, share = share)
    tpn.apply_tensor_parallel(model, mesh)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-2, foreach = False)
    losses, norms, grads = [], [], None
    for step in range(2):
        loss = model(input_ids = ids, labels = labels, use_cache = False).loss
        loss.backward()
        if step == 0:
            grads = {
                n: tpn._full(p.grad, p).detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        norms.append(float(tpn.tp_clip_grad_norm_(params, 1.0)))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
    return {
        "tp": {"losses": losses, "norms": norms, "grads": grads},
        "ref": _reference(ref, ids, labels),
        "regions": len(model._unsloth_tp_plan.regions),
    }


PP_PASS = ("gpt2", "opt")
PP_REFUSE = {
    "bart": ("bart", None, "needs the BartForCausalLM loss to take num_items_in_batch"),
    "llama_shared": ("llama", (0, 3), "share trainable parameters across stages"),
}


def _case_pp(rank, world, tmp):
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    out = {}
    for arch in PP_PASS:
        try:
            out[arch] = _pp_parity(mesh, arch)
        except Exception as e:
            out[arch] = {"crash": f"{type(e).__name__}: {e}"}
    for name, (arch, share, _) in PP_REFUSE.items():
        model = build(arch, share = share)
        out[name] = _refusal(
            lambda m: ppn.apply_pipeline_parallel(m, mesh, n_microbatches = 2), model
        )
    return out


def _pp_parity(mesh, arch):
    ref = build(arch)
    ids, labels = batch(ref)
    model = build(arch)
    pp = ppn.apply_pipeline_parallel(model, mesh, n_microbatches = 2)
    params = pp.parameters()
    opt = torch.optim.AdamW(params, lr = 1e-2, foreach = False)
    losses, norms, grads = [], [], None
    for step in range(2):
        loss = pp.step(ids, labels)
        if step == 0:
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        norms.append(float(pp.clip_grad_norm_(1.0)))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
    return {
        "pp": {"losses": losses, "norms": norms, "grads": grads},
        "ref": _reference(ref, ids, labels),
        "bounds": pp.bounds,
    }


def _import_checkout_unsloth():
    """`import unsloth` from this checkout (conftest's device spoof first); None, or why it failed."""
    for name in [n for n in sys.modules if n == "unsloth" or n.startswith("unsloth.")]:
        del sys.modules[name]
    sys.path.insert(0, REPO)
    try:
        spec = importlib.util.spec_from_file_location(
            "_pg_conftest", os.path.join(REPO, "tests", "conftest.py")
        )
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        import unsloth  # noqa: F401
    except Exception as e:  # a CPU host that refuses `import unsloth`
        return repr(e)
    return None


def _no_host_sync():
    """Patches making any device -> host read of a tensor (float(), .item(), bool()) raise."""
    import contextlib

    @contextlib.contextmanager
    def guard():
        saved = {k: getattr(torch.Tensor, k) for k in ("__float__", "item", "__bool__")}

        def refuse(self, *args, **kwargs):
            raise AssertionError("host sync")

        try:
            for k in saved:
                setattr(torch.Tensor, k, refuse)
            yield
        finally:
            for k, v in saved.items():
                setattr(torch.Tensor, k, v)

    return guard()


def _case_clip(rank, world, tmp):
    """tp_clip_grad_norm_ / _clip_over_world: no host sync, a tensor norm, grads clipped to max_norm."""
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]
    model = build("llama")
    tpn.apply_tensor_parallel(model, mesh)
    ids, labels = batch(model, bsz = 2)
    params = [p for p in model.parameters() if p.requires_grad]
    out = {}
    for name, clip in (
        ("tp_clip_grad_norm_", lambda: tpn.tp_clip_grad_norm_(params, 1e-3)),
        (
            "_clip_over_world",
            lambda: tpn._clip_over_world([p for p in params if p.grad is not None], 1e-3, 1e-6),
        ),
        ("no_clip", lambda: tpn.tp_clip_grad_norm_(params, 1e9)),
    ):
        model.zero_grad(set_to_none = True)
        model(input_ids = ids, labels = labels, use_cache = False).loss.backward()
        before = {
            n: tpn._full(p.grad, p).clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        try:
            with _no_host_sync():
                norm = clip()
            error = None
        except AssertionError as e:
            norm, error = None, str(e)
        after = {
            n: tpn._full(p.grad, p).clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        total = float(torch.stack([g.double().square().sum() for g in after.values()]).sum().sqrt())
        out[name] = {
            "error": error,
            "is_tensor": isinstance(norm, torch.Tensor),
            "norm": None if norm is None else float(norm),
            "before": float(
                torch.stack([g.double().square().sum() for g in before.values()]).sum().sqrt()
            ),
            "after": total,
            "unchanged": all(torch.equal(before[k], after[k]) for k in before),
        }
    return out


EXPORTS = (
    "unsloth_save_model",
    "unsloth_save_model:lora",
    "unsloth_generic_save_pretrained_merged",
    "unsloth_save_pretrained_merged",
)


def _case_export(rank, world, tmp):
    """Merged / LoRA exports refuse a TP or PP model on every rank before writing anything."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth import save
    from unsloth.models import _utils, pp_native, tp_native

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]
    guard = getattr(_utils, "_refuse_model_parallel_export", None)
    if guard is not None:
        guard(build("llama"), "save_pretrained_merged")  # not parallel: passes through
    tp_model = build("llama")
    tp_native.apply_tensor_parallel(tp_model, mesh)
    pp_model = build("llama")
    pp_native.apply_pipeline_parallel(pp_model, mesh, n_microbatches = 2)
    calls = {
        "unsloth_save_model": lambda m, d: save.unsloth_save_model(
            m, None, d, save_method = "merged_16bit"
        ),
        "unsloth_save_model:lora": lambda m, d: save.unsloth_save_model(
            m, None, d, save_method = "lora"
        ),
        "unsloth_generic_save_pretrained_merged": lambda m,
        d: save.unsloth_generic_save_pretrained_merged(
            m, d, tokenizer = None, save_method = "merged_16bit"
        ),
        "unsloth_save_pretrained_merged": lambda m, d: save.unsloth_save_pretrained_merged(
            m, d, tokenizer = None
        ),
    }
    out = {}
    for kind, model in (("tp", tp_model), ("pp", pp_model)):
        for name in EXPORTS:
            directory = os.path.join(tmp, f"{kind}_{name.replace(':', '_')}_{rank}")
            try:
                calls[name](model, directory)
                error = None
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            wrote = os.path.exists(directory) and bool(os.listdir(directory))
            out[f"{kind}:{name}"] = {"error": error, "wrote": wrote}
    return out


CASES = {"tp": _case_tp, "pp": _case_pp, "clip": _case_clip, "export": _case_export}


def _worker(rank, world, port, case, tmp):
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
        torch.save(CASES[case](rank, world, tmp), os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(case, world = 2):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp)) for r in range(world)
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


_SPAWNED = {}


def _spawn_cached(case):
    if case not in _SPAWNED:
        _SPAWNED[case] = _spawn(case)
    return _SPAWNED[case]


def _close(a, b, rtol):
    return abs(a - b) <= 1e-6 + rtol * abs(b)


def _check_run(
    run,
    ref,
    name,
    union = None,
):
    # step 0 at 1e-4; step 1 norms at 1e-3 (after an AdamW step, near-zero grads flip update signs)
    for i, (a, b) in enumerate(zip(run["losses"], ref["losses"])):
        assert _close(a, b, 1e-4), (name, "loss", i, a, b)
    for i, (a, b) in enumerate(zip(run["norms"], ref["norms"])):
        assert _close(a, b, 1e-4 if i == 0 else 1e-3), (name, "grad norm", i, a, b)
    if union is None:
        assert set(run["grads"]) == set(ref["grads"]), (
            name,
            sorted(set(run["grads"]) ^ set(ref["grads"]))[:4],
        )
    for key, g in run["grads"].items():
        r = ref["grads"][key]
        atol = max(
            1e-7, 1e-4 * float(r.abs().max())
        )  # fp32 reduction order: ~1e-6..1e-5 of the max
        torch.testing.assert_close(g, r, rtol = 1e-4, atol = atol, msg = f"{name}: {key}")


@pytest.fixture(scope = "module")
def tp_results():
    return _spawn("tp")


@pytest.fixture(scope = "module")
def pp_results():
    return _spawn("pp")


@pytest.mark.parametrize("name", sorted(TP_PASS))
def test_tp_trains_like_one_process(tp_results, name):
    for res in tp_results:
        assert "crash" not in res[name], res[name]
        _check_run(res[name]["tp"], res[name]["ref"], name)
        assert res[name]["regions"] > 0
    if name == "llama_shared":
        # a block shared by two layers is sharded once (it used to be re-sharded: DTensor of a DTensor)
        assert tp_results[0][name]["regions"] == 6


@pytest.mark.parametrize("arch", sorted(TP_REFUSE))
def test_tp_refuses_before_mutation(tp_results, arch):
    for res in tp_results:
        assert res[arch]["error"] and TP_REFUSE[arch] in res[arch]["error"], res[arch]
        assert res[arch]["unchanged"], arch


@pytest.mark.parametrize("arch", PP_PASS)
def test_pp_trains_like_one_process(pp_results, arch):
    for res in pp_results:
        assert "crash" not in res[arch], res[arch]
        _check_run(res[arch]["pp"], res[arch]["ref"], arch, union = True)
    keys = set(pp_results[0][arch]["pp"]["grads"]) | set(pp_results[1][arch]["pp"]["grads"])
    assert keys == set(pp_results[0][arch]["ref"]["grads"])
    assert not set(pp_results[0][arch]["pp"]["grads"]) & set(pp_results[1][arch]["pp"]["grads"])


@pytest.mark.parametrize("name", sorted(PP_REFUSE))
def test_pp_refuses_before_mutation(pp_results, name):
    for res in pp_results:
        assert res[name]["error"] and PP_REFUSE[name][2] in res[name]["error"], res[name]
        assert res[name]["unchanged"], name


def test_tp_head_norm_names_need_one_head():
    # q_layernorm over all heads (not one) is still refused: a norm across heads needs the full projection
    model = build("hunyuan_v1_dense", lora = False)
    attn = model.model.layers[0].self_attn
    attn.query_layernorm = type(attn.query_layernorm)(64)
    with pytest.raises(ValueError, match = "normalizes 64 features, not one head"):
        tpn.derive_tp_plan(model, 2)


@pytest.mark.parametrize("name", ["tp_clip_grad_norm_", "_clip_over_world", "no_clip"])
def test_tp_clip_has_no_host_sync(name):
    for res in _spawn_cached("clip"):
        r = res[name]
        assert r["error"] is None and r["is_tensor"], r
        assert r["norm"] == pytest.approx(r["before"], rel = 1e-5), r
        if name == "no_clip":
            assert r[
                "unchanged"
            ], r  # a coefficient clamped at 1 leaves the grads bitwise unchanged
        else:
            assert r["after"] == pytest.approx(1e-3, rel = 1e-4), r


def test_tp_clip_coefficient_is_not_narrowed_to_fp16():
    # A coefficient of ~1.7e-8 rounds to 0 in fp16; the clipped grads (~1e-3) are representable.
    p = torch.nn.Parameter(torch.zeros(4, dtype = torch.float16))
    p.grad = torch.full((4,), 60000.0, dtype = torch.float16)
    tpn._scale_grads_([p], torch.tensor(6e7), 1.0, 1e-6)
    assert torch.all(p.grad > 0), p.grad
    assert p.grad.float() == pytest.approx(torch.full((4,), 1e-3), rel = 1e-2), p.grad


@pytest.mark.parametrize("name", EXPORTS)
def test_exports_refuse_pipeline_parallel_before_writing(name):
    results = _spawn_cached("export")
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    for res in results:
        r = res[f"pp:{name}"]
        assert (
            r["error"] and "RuntimeError" in r["error"] and "pipeline parallel model" in r["error"]
        ), r
        assert not r["wrote"], r


@pytest.mark.parametrize("name", EXPORTS)
def test_exports_of_a_tensor_parallel_model_run_on_rank0_only(name):
    """tp_replica.py: rank 0 runs the one-process export on the gathered model, the other ranks wait and
    share its outcome (here a merged save with no base checkpoint fails the same way one process does)."""
    results = _spawn_cached("export")
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    rank0, rank1 = results[0][f"tp:{name}"], results[1][f"tp:{name}"]
    assert not rank1["wrote"], rank1  # rank 1's own directory: only rank 0 writes
    if rank0["error"] is None:
        assert rank0["wrote"] and rank1["error"] is None, (rank0, rank1)
    else:
        assert (
            "tensor parallel" not in rank0["error"]
        ), rank0  # the export's own error, not a refusal
        assert rank1["error"] and "failed on rank 0" in rank1["error"], rank1


def test_every_whole_model_export_is_guarded():
    """Static: every export patch_saving_functions binds, and every writer behind them, refuses TP / PP first."""
    import ast

    with open(os.path.join(REPO, "unsloth", "save.py"), encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    bound = set()
    for node in ast.walk(functions["patch_saving_functions"]):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_bind_saving_method":
            bound.add(node.args[0].id)
    adapter_or_tokenizer = {
        "unsloth_model_save_pretrained",
        "unsloth_push_to_hub",
        "unsloth_tokenizer_save_pretrained",
    }
    writers = {
        "unsloth_save_model",
        "unsloth_generic_save",
        "unsloth_save_pretrained_merged",
        "unsloth_push_to_hub_merged",
        "save_to_gguf_generic",
        "_unsloth_save_lora_gguf",
        "save_lora_to_custom_dir",
        "_unsloth_save_compressed_tensors",
        "_unsloth_save_torchao",
        "_unsloth_save_openvino",
    }
    targets = (bound - adapter_or_tokenizer) | writers
    assert len(targets) >= 18, sorted(targets)
    unguarded = []
    for name in sorted(targets):
        decorators = functions[
            name
        ].decorator_list  # bound inside patch_saving_functions: defined at module level
        first = decorators[0] if decorators else None
        if not (
            isinstance(first, ast.Call)
            and getattr(first.func, "id", None) == "_model_parallel_refuses_save"
        ):
            unguarded.append(name)
    assert not unguarded, unguarded
