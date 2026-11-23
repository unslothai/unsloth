# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 optimizer resume on a gloo CPU world of 2 when a trainable parameter never got a
gradient before the save: it stays stateless and the next step is bitwise equal to the
uninterrupted run; a checkpoint not listing a trainable parameter is still refused."""

import datetime
import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import copy  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

REPO_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _model(seed):
    import transformers

    config = transformers.AutoConfig.from_pretrained(REPO_ID)
    config.pad_token_id = None
    config.tie_word_embeddings = False
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    if model.lm_head.weight is model.model.embed_tokens.weight:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator = g) * 0.01)
    return model


def _batch(step, rank):
    g = torch.Generator().manual_seed(100 * step + rank)
    ids = torch.randint(3, 1000, (2, 16), generator = g)
    return {"input_ids": ids, "labels": ids.clone(), "use_cache": False}


def _step(
    model,
    opt,
    step,
    rank,
    skip = None,
):
    loss = model(**_batch(step, rank)).loss
    loss.backward()
    if skip is not None:
        skip.grad = None  # never reached by the loss: no gradient, no optimizer state
    opt.step()
    opt.zero_grad(set_to_none = True)
    return loss.detach().clone()


def _full_model(model):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    sd = get_model_state_dict(model, options = StateDictOptions(full_state_dict = True))
    return {k: v.detach().clone() for k, v in sd.items()}


def _full_optim(model, opt):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict
    return get_optimizer_state_dict(model, opt, options = StateDictOptions(full_state_dict = True))


def _diff_models(a, b):
    out = sorted(set(a) ^ set(b))
    return out + [k for k in a if k in b and not torch.equal(a[k], b[k])]


def _diff_optims(a, b):
    out = sorted(set(a["state"]) ^ set(b["state"]))
    for k, v in a["state"].items():
        w = b["state"].get(k, {})
        for name in sorted(set(v) | set(w)):
            t, u = v.get(name), w.get(name)
            if isinstance(t, torch.Tensor) or isinstance(u, torch.Tensor):
                if not (
                    isinstance(t, torch.Tensor)
                    and isinstance(u, torch.Tensor)
                    and torch.equal(t, u)
                ):
                    out.append(f"{k}.{name}")
            elif t != u:
                out.append(f"{k}.{name}")

    def _groups(o):
        # The full-state load hands betas back as a list (torch's state-dict flattening).
        return [
            {k: list(v) if isinstance(v, tuple) else v for k, v in g.items()}
            for g in o["param_groups"]
        ]

    if _groups(a) != _groups(b):
        out.append("param_groups")
    return out


_SKIP = "model.layers.0.post_attention_layernorm.weight"


def _new(native, seed):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    model = native.shard_model(_model(seed), native.FSDP2Config(param_dtype = torch.float32))
    return (
        model,
        torch.optim.AdamW(model.parameters(), lr = 1e-2, foreach = False),
        (
            StateDictOptions,
            set_model_state_dict,
        ),
    )


def _skip(model):
    return model.model.layers[0].post_attention_layernorm.weight


def _has_state(opt, param):
    return bool(opt.state.get(param))


def _case_resume(native, dist, rank, tmp):
    model, opt, (Options, set_model) = _new(native, 1)
    for step in (1, 2):
        _step(model, opt, step, rank, skip = _skip(model))
    path = os.path.join(tmp, "optimizer.pt")
    native.native_save_optimizer(model, opt, path, should_save = rank == 0)
    saved = torch.load(path, map_location = "cpu", weights_only = True)  # rank 0's full state
    weights = _full_model(model)
    res = {
        "saved_has_state": _has_state(opt, _skip(model)),
        "saved_in_file": _SKIP in saved["state"],
        "saved_in_groups": any(_SKIP in g["params"] for g in saved["param_groups"]),
    }
    want = {
        "loss": _step(model, opt, 3, rank).item(),
        "after": _full_model(model),
        "optim": _full_optim(model, opt),
    }
    for arm, src in (("file", path), ("state", copy.deepcopy(saved))):
        m, o, _ = _new(native, 2)
        set_model(m, weights, options = Options(full_state_dict = True))
        try:
            native.native_load_optimizer(m, o, src)
        except BaseException as exc:
            res[f"{arm}_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            continue
        res[f"{arm}_has_state"] = _has_state(o, _skip(m))
        res[f"{arm}_n_state"] = sum(1 for p in m.parameters() if _has_state(o, p))
        loss = _step(m, o, 3, rank).item()
        res[f"{arm}_loss"] = [want["loss"], loss]
        res[f"{arm}_after_diff"] = _diff_models(want["after"], _full_model(m))
        res[f"{arm}_optim_diff"] = [
            str(d)[:200] for d in _diff_optims(want["optim"], _full_optim(m, o))
        ]
    res["n_trainable"] = sum(1 for p in model.parameters() if p.requires_grad)
    return res


def _case_refuse(native, dist, rank, tmp):
    """A checkpoint missing the parameter entirely, or from another model, still raises."""
    model, opt, _ = _new(native, 1)
    for step in (1, 2):
        _step(model, opt, step, rank, skip = _skip(model))
    saved = _full_optim(model, opt)  # full on every rank (no cpu_offload)
    res = {}
    dropped = copy.deepcopy(saved)
    for g in dropped["param_groups"]:
        g["params"] = [f for f in g["params"] if f != _SKIP]
    other = copy.deepcopy(saved)
    other["state"] = {f"other.{k}": v for k, v in other["state"].items()}
    for g in other["param_groups"]:
        g["params"] = [f"other.{f}" for f in g["params"]]
    for arm, src in (("dropped", dropped), ("other", other)):
        m, o, _ = _new(native, 2)
        try:
            native.native_load_optimizer(m, o, src)
            res[f"{arm}_error"] = ""
        except BaseException as exc:
            res[f"{arm}_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return res


def _worker(rank, world, port, out, case, tmp):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        native = _load_native()
        res = (_case_resume if case == "resume" else _case_refuse)(native, dist, rank, tmp)
        out.put((rank, res))
    except BaseException as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(
    case,
    tmp,
    world = 2,
    timeout = 600,
):
    import queue

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target = _worker, args = (r, world, port, out, case, tmp)) for r in range(world)
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, r = out.get(timeout = timeout)
            results[rank] = r
    except queue.Empty:
        pass
    finally:
        for p in procs:
            p.join(timeout = 60)
            if p.is_alive():
                p.kill()
    assert sorted(results) == list(range(world)), f"{case}: ranks did not report: {results}"
    for rank, r in results.items():
        print("OPTIM_UNSTEPPED", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_resume_with_a_parameter_never_stepped(tmp_path):
    pytest.importorskip("transformers")
    for rank, r in _run("resume", str(tmp_path)).items():
        # The setup: the saved optimizer lists the parameter but holds no state for it.
        assert r["saved_has_state"] is False and r["saved_in_file"] is False, r
        assert r["saved_in_groups"] is True, r
        for arm in ("file", "state"):
            assert f"{arm}_error" not in r, r
            assert r[f"{arm}_has_state"] is False, r
            assert r[f"{arm}_n_state"] == r["n_trainable"] - 1, r
            assert r[f"{arm}_loss"][0] == r[f"{arm}_loss"][1], r
            assert r[f"{arm}_after_diff"] == [], r
            assert r[f"{arm}_optim_diff"] == [], r


def test_resume_refuses_another_models_optimizer(tmp_path):
    pytest.importorskip("transformers")
    for rank, r in _run("refuse", str(tmp_path)).items():
        assert r["dropped_error"], r
        assert r["other_error"], r
