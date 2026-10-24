# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 and the Trainer's constructor arguments, on a gloo CPU world of 2. The real
`_unsloth_trainer_init` (taken from unsloth/models/_utils.py) wraps transformers' Trainer.__init__:

* an optimizer passed positionally (Trainer's own order) is refused like a keyword one;
* `model_init` (and with it hyperparameter_search) is refused instead of training unsharded;
* `save_on_each_node=True` across nodes is refused (only global rank 0 holds the gathered state);
* a resume of a non-default active PEFT adapter reads `<ckpt>/<adapter>/`, never the root adapter.
Plus the launcher env: an explicit native backend turns accelerate's FSDP launch env off."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import ast  # noqa: E402
import builtins  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import (  # noqa: E402
    REPO_ID,
    _free_port,
    _full,
    _stub_unsloth_save,
)


def _install_wrapped_trainer_init():
    """Trainer.__init__ wrapped by the `_unsloth_trainer_init` source in _utils.py, with the
    unrelated helpers it calls stubbed (they need the whole unsloth package)."""
    from transformers import Trainer

    if getattr(Trainer, "_test_wrapped", False):
        return
    path = os.path.join(REPO, "unsloth", "models", "_utils.py")
    tree = ast.parse(open(path, encoding = "utf-8").read())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_trainer_init"
    )
    fsdp2 = types.ModuleType("unsloth.models.fsdp2")
    fsdp2.install_fsdp2_policy_provenance = lambda: None
    sys.modules["unsloth.models.fsdp2"] = fsdp2
    noop = lambda *a, **k: None  # noqa: E731
    ns = {
        "__name__": "unsloth.models._utils",
        "__package__": "unsloth.models",
        "os": os,
        "_original_trainer_init": Trainer.__init__,
        "apply_accepts_loss_kwargs_fix": noop,
        "_replan_auto_offload_safely": noop,
        "_block_swap_for_trainer_backend": noop,
        "_enable_ddp_batched_grad_copy": noop,
    }
    # Any other module-level helper the wrapper calls is stubbed too, so a new one (#14404 added
    # _size_auto_checkpoint_skip_for_trainer) cannot fail every rank with a NameError here.
    bound = {
        name.id
        for name in ast.walk(fn)
        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
    } | {
        (alias.asname or alias.name).split(".")[0]
        for node in ast.walk(fn)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in ns and name not in bound and not hasattr(builtins, name):
                ns[name] = noop
    exec(compile(ast.Module(body = [fn], type_ignores = []), path, "exec"), ns)
    Trainer.__init__ = ns["_unsloth_trainer_init"]
    Trainer._test_wrapped = True


def _lora(
    seed,
    same_config_adapter = False,
    active = "default",
):
    import transformers
    from peft import LoraConfig, get_peft_model

    config = transformers.AutoConfig.from_pretrained(REPO_ID)
    config.pad_token_id = None
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = None
    cfg = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj"])
    model = get_peft_model(model, cfg)
    if same_config_adapter:
        model.add_adapter(
            "other", LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj"])
        )
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.normal_(0, 0.02, generator = g)
    if active != "default":
        model.set_adapter(active)
    return model


def _targs(out, **kw):
    from transformers import TrainingArguments
    return TrainingArguments(output_dir = out, report_to = [], use_cpu = True, **kw)


def _worker(rank, world, port, out, case, tmp, env):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    os.environ.update(env)
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        native = _load_native()
        _stub_unsloth_save()
        _install_wrapped_trainer_init()
        from transformers import Trainer

        res = {}
        if case == "positional_optimizer":
            model = _lora(1)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
            targs = _targs(os.path.join(tmp, "out"))
            try:
                # Trainer's positional order: model, args, data_collator, train_dataset, eval_dataset,
                # processing_class, model_init, compute_loss_func, compute_metrics, callbacks, optimizers.
                trainer = Trainer(
                    model, targs, None, None, None, None, None, None, None, None, (optimizer, None)
                )
                params = {id(p) for p in trainer.model.parameters()}
                res["stale"] = sum(
                    id(p) not in params for g in optimizer.param_groups for p in g["params"]
                )
            except RuntimeError as exc:
                res["error"] = str(exc)
        elif case == "model_init":
            targs = _targs(os.path.join(tmp, "out"))
            try:
                trainer = Trainer(args = targs, model_init = lambda: _lora(1))
                res["sharded"] = native.is_native_fsdp2(trainer.model)
            except RuntimeError as exc:
                res["error"] = str(exc)
        elif case == "save_on_each_node":
            # Two nodes of one process each: both ranks are local rank 0, so both `should_save`.
            node_dir = os.path.join(tmp, f"node{rank}")
            targs = _targs(node_dir, save_on_each_node = True)
            try:
                trainer = Trainer(model = _lora(1), args = targs)
                trainer.create_optimizer()
                trainer._save_optimizer_and_scheduler(node_dir)
                state = torch.load(os.path.join(node_dir, "optimizer.pt"), weights_only = True)
                res["optimizer_keys"] = sorted(state)
            except RuntimeError as exc:
                res["error"] = str(exc)
        elif case == "named_adapter":
            cfg = native.FSDP2Config(param_dtype = torch.float32)
            ckpt = os.path.join(tmp, "ckpt")
            src = native.shard_model(_lora(1, same_config_adapter = True, active = "other"), cfg)
            Trainer(model = src, args = _targs(os.path.join(tmp, "out"))).save_model(ckpt)
            dist.barrier()
            want = _full(src)
            dst = native.shard_model(_lora(2, same_config_adapter = True, active = "other"), cfg)
            try:
                native.native_load_model_checkpoint(dst, ckpt)
                got = _full(dst)
                other = [n for n in want if ".other." in n]
                res["n_other"] = len(other)
                res["other_diff"] = max((got[n] - want[n]).abs().max().item() for n in other)
                res["files"] = sorted(
                    os.path.relpath(os.path.join(d, f), ckpt)
                    for d, _, fs in os.walk(ckpt)
                    for f in fs
                    if f.startswith("adapter_model")
                )
            except Exception as exc:
                res["load_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        dist.destroy_process_group()


def _run(
    case,
    tmp_path,
    env = None,
):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        _worker,
        args = (2, _free_port(), out, case, str(tmp_path), dict(env or {})),
        nprocs = 2,
        join = True,
        start_method = "spawn",
    )
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("ARGS_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_positional_optimizer_is_refused(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("positional_optimizer", tmp_path).items():
        assert "optimizer built before the Trainer" in r.get("error", ""), (rank, r)


def test_model_init_is_refused(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("model_init", tmp_path).items():
        assert "model_init" in r.get("error", ""), (rank, r)


def test_save_on_each_node_across_nodes_is_refused(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    env = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1"}
    for rank, r in _run("save_on_each_node", tmp_path, env).items():
        assert "save_on_each_node" in r.get("error", ""), (rank, r)


def test_resume_reads_named_adapter_from_its_subdirectory(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("named_adapter", tmp_path).items():
        assert "load_error" not in r, (rank, r)
        assert r["n_other"] > 0 and r["other_diff"] == 0.0, (rank, r)


# ---- launcher env (no process group) ----------------------------------------------------------


def _load_launch():
    path = os.path.join(REPO, "unsloth", "fsdp2_launch.py")
    spec = importlib.util.spec_from_file_location("_unsloth_fsdp2_launch_args", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_env(monkeypatch):
    saved = dict(os.environ)
    for key in ("ACCELERATE_USE_FSDP", "FSDP_VERSION", "UNSLOTH_FSDP2_BACKEND", "RANK"):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setenv("UNSLOTH_FSDP2", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_explicit_native_backend_turns_accelerate_fsdp_env_off(clean_env, monkeypatch, capsys):
    launch = _load_launch()
    # What `accelerate launch` exports for an FSDP config (accelerate/utils/launch.py).
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    monkeypatch.setenv("FSDP_VERSION", "2")
    monkeypatch.setenv("UNSLOTH_FSDP2_BACKEND", "native")
    monkeypatch.setenv("RANK", "0")
    assert launch.unsloth_fsdp2_native_requested()
    assert launch.apply_unsloth_fsdp2_launch_env() is False
    assert os.environ.get("ACCELERATE_USE_FSDP", "false").lower() != "true"
    assert "FSDP_VERSION" not in os.environ
    # Still native once the accelerate variables are gone, and the second call is a no-op.
    assert launch.unsloth_fsdp2_native_requested()
    assert launch.apply_unsloth_fsdp2_launch_env() is False
    assert capsys.readouterr().out.count("UNSLOTH_FSDP2_BACKEND") == 1


def test_accelerate_fsdp_launch_without_backend_is_left_alone(clean_env, monkeypatch):
    launch = _load_launch()
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    monkeypatch.setenv("FSDP_VERSION", "2")
    assert not launch.unsloth_fsdp2_native_requested()
    launch.apply_unsloth_fsdp2_launch_env()
    assert os.environ["ACCELERATE_USE_FSDP"] == "true" and os.environ["FSDP_VERSION"] == "2"
