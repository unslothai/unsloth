# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""apply_pipeline_parallel on a gloo CPU world (2 and 4 stages) vs the same model in one process, fp32."""

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
import weakref  # noqa: E402
import gc as gc_module  # noqa: E402

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


ppn = _load("pp_native")

LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
QWEN3 = "trl-internal-testing/tiny-Qwen3ForCausalLM"
GEMMA2 = "trl-internal-testing/tiny-Gemma2ForCausalLM"
GEMMA3 = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"
QWEN3_MOE = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
DEEPSEEK_V3 = "trl-internal-testing/tiny-DeepseekV3ForCausalLM"

ARCHS = {"llama": LLAMA, "qwen3": QWEN3, "gemma2": GEMMA2, "gemma3": GEMMA3, "mistral": MISTRAL}
BLOOM = "trl-internal-testing/tiny-BloomForCausalLM"  # a norm before the layer list (word_embeddings_layernorm)
N_LAYERS = 6


def build(
    repo,
    layers = N_LAYERS,
    lora = True,
    seed = 0,
    tie = None,
    **lora_kwargs,
):
    from transformers import AutoConfig, AutoModelForCausalLM

    probe = AutoConfig.from_pretrained(repo)
    values = probe.to_dict()
    values["num_hidden_layers"] = layers
    values.pop("layer_types", None)  # rebuilt for the new depth (sliding / full pattern)
    if tie is not None:
        values["tie_word_embeddings"] = tie
    config = type(probe).from_dict(values)
    for key in ("attention_dropout", "hidden_dropout", "dropout"):
        if isinstance(getattr(config, key, None), float):
            setattr(config, key, 0.0)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.add_(torch.randn_like(p) * 0.1)
    if not lora:
        return model.train()
    from peft import LoraConfig, get_peft_model

    kw = dict(
        r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False, lora_dropout = 0.0
    )
    kw.update(lora_kwargs)
    torch.manual_seed(seed + 1)
    return get_peft_model(model, LoraConfig(**kw)).train()


def batch(
    model,
    bsz = 4,
    seq = 8,
    seed = 1,
):
    vocab = ppn._text_config(model).vocab_size
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (bsz, seq), generator = g)
    labels = ids.clone()
    labels[0, :3] = -100  # uneven token counts per micro-batch: num_items normalisation matters
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
    steps,
    clip = 1.0,
):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-2, foreach = False)
    losses, norms, grads, hidden = [], [], None, None
    for step in range(steps):
        out = model(input_ids = ids, labels = labels, use_cache = False, output_hidden_states = step == 0)
        out.loss.backward()
        if step == 0:
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            hidden = [h.detach().clone() for h in out.hidden_states]
        norms.append(_clip_fp64(params, clip))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss))
    return {"losses": losses, "norms": norms, "grads": grads, "hidden": hidden}


def _pp_run(
    model,
    pp,
    ids,
    labels,
    steps,
    clip = 1.0,
):
    params = pp.parameters()
    opt = torch.optim.AdamW(params, lr = 1e-2, foreach = False)
    outs = []
    handle = pp.module.register_forward_hook(
        lambda m, a, k, o: outs.append(o.detach().clone()), with_kwargs = True
    )
    losses, norms, grads = [], [], None
    for step in range(steps):
        loss = pp.step(ids, labels)
        if step == 0:
            grads = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            handle.remove()
        norms.append(float(pp.clip_grad_norm_(1.0 if clip is None else clip)))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
    return {"losses": losses, "norms": norms, "grads": grads, "outs": outs}


def _read_adapter(path):
    from safetensors.torch import load_file
    return load_file(os.path.join(path, "adapter_model.safetensors"))


def _enable_reentrant_gc(model):
    """Reentrant checkpointing only back-propagates into a segment whose input requires grad; with the
    embeddings frozen (LoRA) that needs enable_input_require_grads, which transformers 4.x
    gradient_checkpointing_enable does not do for a PEFT model (5.x does)."""
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.enable_input_require_grads()


def _case_parity(
    rank,
    world,
    tmp,
    arch,
    schedule = "gpipe",
    n_mb = 2,
    gc = False,
    steps = 3,
    tie = None,
    output_attentions = False,
    trainable_norms = False,
):
    from torch.distributed.device_mesh import init_device_mesh

    repo = ARCHS.get(arch) or {"bloom": BLOOM}[arch]

    extra = {"modules_to_save": ["word_embeddings_layernorm", "ln_f"]} if trainable_norms else {}
    ref = build(repo, tie = tie, **extra)
    ids, labels = batch(ref)
    if gc:
        _enable_reentrant_gc(ref)
    ref_res = _reference(ref, ids, labels, steps)
    if rank == 0:
        ref.save_pretrained(os.path.join(tmp, "ref"))
    model = build(repo, tie = tie, **extra)
    if gc:
        _enable_reentrant_gc(model)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    originals = {
        "embed": weakref.ref(model.get_input_embeddings()),
        "head": weakref.ref(model.get_output_embeddings()),
    }
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = n_mb, schedule = schedule)
    gc_module.collect()
    alive = {k: r() is not None for k, r in originals.items()}
    seen = []
    if output_attentions:
        # Unsloth's tuple-format decoder reads layer_outputs[1] when attentions are requested; a pass-through
        # layer returns hidden states only, so the stage forward must turn them off.
        model.config.output_attentions = True
        hook = model.get_base_model().register_forward_pre_hook(
            lambda m, a, k: seen.append(k.get("output_attentions")), with_kwargs = True
        )
    pp_res = _pp_run(model, pp, ids, labels, steps)
    if output_attentions:
        hook.remove()
    pp.save_adapter(os.path.join(tmp, "pp"))
    generate_error = None
    try:
        model.generate(ids[:1, :2], max_new_tokens = 1)
    except RuntimeError as exc:
        generate_error = str(exc)
    direct_error = None
    try:
        model(input_ids = ids, labels = labels)
    except RuntimeError as exc:
        direct_error = str(exc)
    owned = {n for n, p in model.named_parameters() if p.requires_grad}
    trained = {
        n.replace(".default.", ".").replace(".modules_to_save.", "."): p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    return {
        "ref": {k: v for k, v in ref_res.items() if k != "hidden"},
        "hidden": ref_res["hidden"],
        "pp": pp_res,
        "bounds": pp.bounds,
        "owned": owned,
        "trained": trained,
        "flag": ppn.is_pipeline_parallel(model),
        "generate_error": generate_error,
        "direct_error": direct_error,
        "tied": pp.parts.tied,
        "alive": alive,
        "seen_output_attentions": seen,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def _case_ga(rank, world, tmp, arch, n_mb):
    """n_microbatches = GA steps of one process with num_items_in_batch over the whole batch."""
    from torch.distributed.device_mesh import init_device_mesh

    repo = ARCHS[arch]
    ref = build(repo)
    ids, labels = batch(ref, bsz = 2 * n_mb)
    n_items = int((labels[:, 1:] != -100).sum())
    total = 0.0
    for chunk_ids, chunk_labels in zip(ids.chunk(n_mb), labels.chunk(n_mb)):
        out = ref(
            input_ids = chunk_ids, labels = chunk_labels, num_items_in_batch = n_items, use_cache = False
        )
        out.loss.backward()
        total += float(out.loss)
    ga_grads = {n: p.grad.detach().clone() for n, p in ref.named_parameters() if p.grad is not None}
    model = build(repo)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = n_mb)
    # [1, seq] position_ids (the default positions) are broadcast to every micro-batch.
    loss = float(pp.step(ids, labels, position_ids = torch.arange(ids.shape[1])[None]))
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    # Unscaled-mean mistake check: what scale_grads=True (divide by n_mb) would have produced.
    return {"ga_loss": total, "ga_grads": ga_grads, "loss": loss, "grads": grads}


def _case_scaler(rank, world, tmp):
    """found_inf is MAX over stages: an inf on one stage skips the step on every stage."""
    from torch.distributed.device_mesh import init_device_mesh

    model = build(LLAMA)
    ids, labels = batch(model)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 2)
    scaler = torch.amp.GradScaler("cpu", init_scale = 1024.0)
    opt = torch.optim.SGD(pp.parameters(), lr = 1.0)
    loss = pp.step(ids, labels, scaler = scaler)
    if rank == world - 1:
        pp.parameters()[0].grad.view(-1)[0] = float("inf")
    before = [p.detach().clone() for p in pp.parameters()]
    pp.unscale_(scaler, opt)
    scaler.step(opt)
    scaler.update()
    after = [p.detach().clone() for p in pp.parameters()]
    unchanged = all(torch.equal(a, b) for a, b in zip(before, after))
    return {"loss": float(loss), "unchanged": unchanged, "scale": scaler.get_scale()}


def _case_refusals(rank, world, tmp):
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    out = {}

    def catch(key, fn):
        try:
            fn()
            out[key] = None
        except (ValueError, RuntimeError) as exc:
            out[key] = str(exc)

    catch("moe", lambda: ppn.apply_pipeline_parallel(build(QWEN3_MOE, layers = 2), mesh))
    catch("mla", lambda: ppn.apply_pipeline_parallel(build(DEEPSEEK_V3, layers = 2), mesh))

    def tied_trainable():
        model = build(GEMMA3)
        model.get_input_embeddings().weight.requires_grad_(True)
        ppn.apply_pipeline_parallel(model, mesh)

    catch("tied_trainable", tied_trainable)
    catch("too_many_stages", lambda: ppn.apply_pipeline_parallel(build(LLAMA, layers = 1), mesh))
    catch(
        "mesh_2d",
        lambda: ppn.apply_pipeline_parallel(
            build(LLAMA), init_device_mesh("cpu", (1, world), mesh_dim_names = ("dp", "pp"))
        ),
    )

    def old_zoo_1f1b():
        fake = types.ModuleType("unsloth_zoo.gradient_checkpointing")
        saved = sys.modules.get("unsloth_zoo.gradient_checkpointing")
        sys.modules["unsloth_zoo.gradient_checkpointing"] = fake
        try:
            model = build(LLAMA)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs = {"use_reentrant": True}
            )
            ppn.apply_pipeline_parallel(model, mesh, n_microbatches = 2, schedule = "1f1b")
        finally:
            if saved is None:
                sys.modules.pop("unsloth_zoo.gradient_checkpointing", None)
            else:
                sys.modules["unsloth_zoo.gradient_checkpointing"] = saved

    catch("old_zoo_1f1b", old_zoo_1f1b)

    def new_zoo_1f1b():
        fake = types.ModuleType("unsloth_zoo.gradient_checkpointing")
        fake._CpuSlotLease = object
        saved = sys.modules.get("unsloth_zoo.gradient_checkpointing")
        sys.modules["unsloth_zoo.gradient_checkpointing"] = fake
        try:
            model = build(LLAMA)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs = {"use_reentrant": True}
            )
            pp = ppn.apply_pipeline_parallel(model, mesh, n_microbatches = 2, schedule = "1f1b")
            assert pp.schedule == "1f1b"
        finally:
            if saved is None:
                sys.modules.pop("unsloth_zoo.gradient_checkpointing", None)
            else:
                sys.modules["unsloth_zoo.gradient_checkpointing"] = saved

    catch("new_zoo_1f1b", new_zoo_1f1b)

    def step_errors():
        model = build(LLAMA)
        pp = ppn.apply_pipeline_parallel(model, mesh, n_microbatches = 2)
        ids, labels = batch(model, bsz = 3)
        errors = []
        for kwargs in ({"packed_seq_lengths": torch.tensor([3])},):
            try:
                pp.step(ids, labels, **kwargs)
            except ValueError as exc:
                errors.append(str(exc))
        try:
            pp.step(ids, labels)
        except ValueError as exc:
            errors.append(str(exc))
        raise ValueError(" | ".join(errors))

    catch("step_errors", step_errors)
    return out


def _import_checkout_unsloth():
    """`import unsloth` from this checkout (not the stub pp_native is loaded under); None, or why it failed."""
    import pathlib

    for name in [n for n in sys.modules if n == "unsloth" or n.startswith("unsloth.")]:
        del sys.modules[name]
    sys.path.insert(0, REPO)
    try:
        spec = importlib.util.spec_from_file_location(
            "_pp_conftest", os.path.join(REPO, "tests", "conftest.py")
        )
        spec.loader.exec_module(importlib.util.module_from_spec(spec))  # conftest's device spoof
        import unsloth
    except Exception as e:  # a CPU host that refuses `import unsloth`
        return repr(e)
    root = pathlib.Path(REPO).resolve()
    assert root in pathlib.Path(unsloth.__file__).resolve().parents, unsloth.__file__
    return None


def _raised(fn):
    try:
        fn()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def _case_guards(rank, world, tmp):
    """Entry points that cannot run one stage alone are refused on every wrapper level."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import transformers
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth import FastLanguageModel, FastModel
    from unsloth.models import fsdp2_native, pp_native, tp_native
    from unsloth.models._utils import patch_gradient_accumulation_fix
    from unsloth.models.llama import LlamaModel_fast_forward

    patch_gradient_accumulation_fix(transformers.Trainer)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    args = transformers.TrainingArguments(output_dir = tmp, report_to = [], use_cpu = True)
    model = build(LLAMA)
    pp = pp_native.apply_pipeline_parallel(model, mesh["pp"])
    assert pp_native.apply_pipeline_parallel(model, mesh["pp"]) is pp  # idempotent
    levels = {
        "peft": model,
        "lora": model.base_model,
        "causal": model.base_model.model,
        "decoder": model.base_model.model.model,
    }
    res = {}
    for name, level in levels.items():
        res[f"is_pp:{name}"] = pp_native.is_pipeline_parallel(level)
        res[f"fsdp2:{name}"] = _raised(lambda: fsdp2_native._refuse_tensor_parallel_model(level))
        res[f"tp:{name}"] = _raised(lambda: tp_native.apply_tensor_parallel(level, mesh["pp"]))
        res[f"for_inference:{name}"] = _raised(lambda: FastLanguageModel.for_inference(level))
        res[f"FastModel.for_inference:{name}"] = _raised(lambda: FastModel.for_inference(level))
        if name != "decoder":
            res[f"generate:{name}"] = _raised(
                lambda: level.generate(input_ids = torch.zeros(1, 2, dtype = torch.long))
            )
            res[f"trainer:{name}"] = _raised(lambda: transformers.Trainer(model = level, args = args))
            res[f"model_init:{name}"] = _raised(
                lambda: transformers.Trainer(model_init = lambda: level, args = args)
            )

    class FastDecoder(torch.nn.Module):
        forward = LlamaModel_fast_forward

    res["unsloth_format"] = pp_native._layer_call_format(FastDecoder(), "layers")
    return res


CASES = {
    "guards": _case_guards,
    "parity": _case_parity,
    "ga": _case_ga,
    "scaler": _case_scaler,
    "refusals": _case_refusals,
}


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 180)
    )
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _torch_pipelining_error():
    try:
        ppn._check_torch_pipelining()
    except RuntimeError as exc:
        return str(exc)
    return None


def _spawn(
    case,
    world = 2,
    keep = None,
    **kwargs,
):
    if _torch_pipelining_error():
        pytest.skip(_torch_pipelining_error())
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
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
        if keep is not None:
            keep["ref"] = _read_adapter(os.path.join(tmp, "ref"))
            keep["pp"] = _read_adapter(os.path.join(tmp, "pp"))
        return results


def _check_parity(results, saved, world):
    ref = results[0]["ref"]
    bounds = results[0]["bounds"]
    assert len(bounds) == world and bounds[0][0] == 0 and bounds[-1][1] == N_LAYERS
    union = set()
    for rank, res in enumerate(results):
        pp = res["pp"]
        assert pp["losses"] == pytest.approx(ref["losses"], rel = 1e-5, abs = 1e-6), (
            rank,
            pp["losses"],
            ref["losses"],
        )
        # Step 0 exact to fp32; later steps carry micro-batch summation order through AdamW (lr 1e-2).
        assert pp["norms"][0] == pytest.approx(ref["norms"][0], rel = 1e-5), (
            rank,
            pp["norms"],
            ref["norms"],
        )
        assert pp["norms"] == pytest.approx(ref["norms"], rel = 1e-4), (
            rank,
            pp["norms"],
            ref["norms"],
        )
        assert set(pp["grads"]) == res["owned"], sorted(set(pp["grads"]) ^ res["owned"])[:6]
        assert not union & set(pp["grads"]), "a trainable parameter lives on two stages"
        union |= set(pp["grads"])
        for name, grad in pp["grads"].items():
            torch.testing.assert_close(grad, ref["grads"][name], rtol = 1e-4, atol = 1e-6, msg = name)
        start, end = res["bounds"][rank]
        if rank < world - 1:
            # Stage output = the 1-process hidden state entering layer `end`, micro-batch by micro-batch.
            torch.testing.assert_close(
                torch.cat(pp["outs"]), res["hidden"][end], rtol = 1e-5, atol = 1e-5
            )
        else:
            assert all(o.dim() == 0 for o in pp["outs"])
        # A stage keeps the embedding / lm_head module only if it owns it.
        assert res["alive"] == {"embed": rank == 0, "head": rank == world - 1}, (rank, res["alive"])
        assert res["flag"]
        assert "generation is not supported on a pipeline parallel model" in (
            res["generate_error"] or ""
        )
        assert "split for pipeline parallelism" in (res["direct_error"] or "")
    assert union == set(ref["grads"]), sorted(union ^ set(ref["grads"]))[:6]
    assert set(saved["pp"]) == set(saved["ref"]), sorted(set(saved["pp"]) ^ set(saved["ref"]))[:6]
    trained = {}
    for res in results:
        trained.update(res["trained"])
    assert set(trained) == set(saved["pp"]), sorted(set(trained) ^ set(saved["pp"]))[:6]
    for key, value in saved["pp"].items():
        assert torch.equal(value, trained[key]), key  # the file holds exactly every stage's weights
    for key, value in saved["ref"].items():
        # 3 AdamW steps (lr 1e-2) amplify fp32 micro-batch summation order (1e-6 grads) up to ~4e-4.
        diff = float((saved["pp"][key] - value).abs().max())
        torch.testing.assert_close(
            saved["pp"][key], value, rtol = 1e-3, atol = 2e-3, msg = f"{key}: {diff}"
        )


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_pp2_matches_one_process(arch):
    saved = {}
    _check_parity(_spawn("parity", 2, keep = saved, arch = arch), saved, 2)


@pytest.mark.parametrize("arch", ["llama", "gemma3"])
def test_pp4_matches_one_process(arch):
    saved = {}
    _check_parity(_spawn("parity", 4, keep = saved, arch = arch, n_mb = 4), saved, 4)


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_pp2_gradient_checkpointing_matches_one_process(schedule):
    saved = {}
    results = _spawn("parity", 2, keep = saved, arch = "qwen3", schedule = schedule, n_mb = 4, gc = True)
    _check_parity(results, saved, 2)


def test_pp4_1f1b_matches_one_process():
    saved = {}
    _check_parity(
        _spawn("parity", 4, keep = saved, arch = "mistral", schedule = "1f1b", n_mb = 4), saved, 4
    )


def test_pp2_stage_forward_turns_off_attention_outputs():
    saved = {}
    results = _spawn("parity", 2, keep = saved, arch = "llama", output_attentions = True)
    _check_parity(results, saved, 2)
    for res in results:
        assert res["seen_output_attentions"], res
        assert all(v is False for v in res["seen_output_attentions"]), res["seen_output_attentions"]


def test_pp2_trainable_input_and_final_norms_train_on_their_owner():
    # Bloom: word_embeddings_layernorm before the layer list (first stage), ln_f after it (last stage).
    saved = {}
    results = _spawn("parity", 2, keep = saved, arch = "bloom", trainable_norms = True)
    _check_parity(results, saved, 2)
    owned = [
        {n for n in r["owned"] if "word_embeddings_layernorm" in n or "ln_f" in n} for r in results
    ]
    assert any("word_embeddings_layernorm" in n for n in owned[0]), owned
    assert any("ln_f" in n for n in owned[1]), owned


def test_pp2_tied_embeddings_kept_on_both_ends():
    saved = {}
    results = _spawn("parity", 2, keep = saved, arch = "llama", tie = True)
    _check_parity(results, saved, 2)
    assert all(r["tied"] for r in results)


@pytest.mark.parametrize("n_mb", [2, 4])
def test_pp2_microbatches_equal_gradient_accumulation(n_mb):
    for res in _spawn("ga", 2, arch = "qwen3", n_mb = n_mb):
        assert res["loss"] == pytest.approx(res["ga_loss"], rel = 1e-5)
        assert res["grads"]
        for name, grad in res["grads"].items():
            torch.testing.assert_close(grad, res["ga_grads"][name], rtol = 1e-4, atol = 1e-6, msg = name)


def test_pp2_grad_scaler_found_inf_is_max_over_stages():
    results = _spawn("scaler", 2)
    assert all(r["unchanged"] for r in results), [r["unchanged"] for r in results]
    assert all(r["scale"] == 512.0 for r in results)
    assert results[0]["loss"] == pytest.approx(results[1]["loss"])


def test_pp2_refusals():
    out = _spawn("refusals", 2)[0]
    assert "mixture-of-experts" in (out["moe"] or ""), out["moe"]
    assert "mixture-of-experts" in (out["mla"] or "") or "latent attention" in (
        out["mla"] or ""
    ), out["mla"]
    assert "tied input / output embeddings" in (out["tied_trainable"] or ""), out["tied_trainable"]
    assert "at least one layer per stage" in (out["too_many_stages"] or ""), out["too_many_stages"]
    assert "1-D pp mesh" in (out["mesh_2d"] or ""), out["mesh_2d"]
    assert "PR #2071" in (out["old_zoo_1f1b"] or ""), out["old_zoo_1f1b"]
    assert out["new_zoo_1f1b"] is None, out["new_zoo_1f1b"]
    assert (
        "packed sequences" in out["step_errors"]
        and "multiple of n_microbatches" in out["step_errors"]
    ), out["step_errors"]


def test_split_layers_balances_lm_head():
    # Qwen3-0.6B: 28 layers of ~15.7M weights, lm_head 155.6M on the last stage.
    bounds = ppn.split_layers([15.7] * 28, 2, last_extra = 155.6)
    assert bounds == [(0, 19), (19, 28)]
    assert ppn.split_layers([1] * 8, 4) == [(0, 2), (2, 4), (4, 6), (6, 8)]
    with pytest.raises(ValueError, match = "at least one layer per stage"):
        ppn.split_layers([1] * 2, 3)
    # Every stage keeps >= 1 layer even when the head outweighs everything.
    assert ppn.split_layers([1] * 3, 3, last_extra = 100) == [(0, 1), (1, 2), (2, 3)]


def test_derive_split_and_parts_are_structural_or_from_the_plan():
    model = build(LLAMA, layers = 8)
    parts = ppn._find_parts(model)
    assert parts.source == "transformers" and parts.layers_name == "layers"
    assert [n for n, _ in parts.norms] == ["norm"]
    bounds = ppn.derive_pp_split(model, 2)
    # tiny Llama: lm_head (32000 x 16) outweighs the 8 tiny layers, so the last stage keeps one layer.
    assert bounds == [(0, 7), (7, 8)], bounds
    assert ppn.derive_pp_split(model, 2, ppn.PPConfig(lm_head_cost = 0.0)) == [(0, 4), (4, 8)]
    assert ppn.derive_pp_split(model, 2, ppn.PPConfig(layers_per_stage = [5, 3])) == [(0, 5), (5, 8)]
    with pytest.raises(ValueError, match = "layers_per_stage"):
        ppn.derive_pp_split(model, 2, ppn.PPConfig(layers_per_stage = [4, 3]))
    assert ppn._layer_call_format(parts.decoder, parts.layers_name) == "tensor"
    # Without base_model_pp_plan the same parts come from the structure.
    config = ppn._text_config(model)
    saved = type(config).base_model_pp_plan
    try:
        type(config).base_model_pp_plan = None
        assert ppn._find_parts(model).source == "structural"
    finally:
        type(config).base_model_pp_plan = saved


def test_layer_call_format_reads_tuple_loops():
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList()

        def forward(self, hidden_states):
            for idx, decoder_layer in enumerate(self.layers):
                layer_outputs = decoder_layer(hidden_states)
                hidden_states = layer_outputs[0]
            return hidden_states

    assert ppn._layer_call_format(Decoder(), "layers") == "tuple"


def test_pp2_entry_points_are_refused():
    results = _spawn("guards", 2)
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    for res in results:
        assert (
            res.pop("unsloth_format") == "tuple"
        )  # LlamaModel_fast_forward reads layer_outputs[0]
        for key, value in res.items():
            if key.startswith("is_pp:"):
                assert value is True, key
            elif key.startswith("fsdp2:"):
                assert value and "does not compose with Unsloth pipeline parallelism" in value, (
                    key,
                    value,
                )
            elif key.startswith("tp:"):
                assert value and "does not compose with Unsloth pipeline parallelism" in value, (
                    key,
                    value,
                )
            elif "for_inference" in key:
                assert value and "not supported on a pipeline parallel model" in value, (key, value)
            elif key.startswith("generate:"):
                assert (
                    value and "generation is not supported on a pipeline parallel model" in value
                ), (key, value)
            elif key.startswith(("trainer:", "model_init:")):
                assert (
                    value
                    and "pipeline parallel model (apply_pipeline_parallel) cannot be trained"
                    in value
                ), (
                    key,
                    value,
                )


def test_old_torch_pipelining_is_refused_before_the_model_is_cut(monkeypatch):
    import torch.distributed.pipelining as pipelining

    class OldGPipe:  # torch 2.6: no scale_grads, step without return_outputs
        def __init__(
            self,
            stage,
            n_microbatches,
            loss_fn = None,
        ):
            pass

        def step(
            self,
            *args,
            target = None,
            losses = None,
            **kwargs,
        ):
            pass

    if _torch_pipelining_error() is None:
        monkeypatch.setattr(pipelining, "ScheduleGPipe", OldGPipe)
    model = build(LLAMA)
    before = [type(layer).__name__ for layer in model.base_model.model.model.layers]
    with pytest.raises(RuntimeError, match = "needs torch >= 2.10"):
        ppn.apply_pipeline_parallel(model, None, n_microbatches = 2)
    assert [type(layer).__name__ for layer in model.base_model.model.model.layers] == before
    assert not ppn.is_pipeline_parallel(model)
