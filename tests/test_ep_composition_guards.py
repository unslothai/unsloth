# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Expert parallel composition and launch guards (gloo CPU, 2 ranks)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "_shared"))
REPO = os.path.dirname(_HERE)
MOE = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
STACKS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
_ENV = (
    "UNSLOTH_PARALLEL",
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_CP_SIZE",
    "UNSLOTH_EP_SIZE",
    "UNSLOTH_FSDP2",
)


def _package_modules(*names):
    """unsloth.<name> from this tree without running unsloth/__init__.py."""
    for name, path in (("unsloth", ("unsloth",)), ("unsloth.models", ("unsloth", "models"))):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [os.path.join(REPO, *path)]
            sys.modules[name] = pkg
    out = [importlib.import_module(f"unsloth.{n}") for n in names]
    for m in out:
        assert os.path.realpath(m.__file__).startswith(os.path.realpath(REPO)), m.__file__
    return out


_AXES = {
    "tp": ("UNSLOTH_TP_SIZE", "2", "tp=2"),
    "pp": ("UNSLOTH_PP_SIZE", "2", "pp=2"),
    "fsdp": ("UNSLOTH_FSDP2", "1", "fsdp=1"),
}


@pytest.mark.parametrize("form", ["legacy", "inherited_ep", "canonical_ep"])
@pytest.mark.parametrize("axis", sorted(_AXES))
def test_launch_refuses_ep_with_another_axis(axis, form):
    (pe,) = _package_modules("parallel_env")
    var, value, key = _AXES[axis]
    environ = {"WORLD_SIZE": "4"}
    if form == "legacy":  # no UNSLOTH_PARALLEL: both legacy variables
        environ.update({"UNSLOTH_EP_SIZE": "2", var: value})
    elif form == "inherited_ep":  # the other axis in the flag, UNSLOTH_EP_SIZE exported by a parent
        environ.update({"UNSLOTH_PARALLEL": key, "UNSLOTH_EP_SIZE": "2"})
    else:  # ep in the flag, the other axis exported
        environ.update({"UNSLOTH_PARALLEL": "ep=2", var: value})
    before = dict(environ)
    with pytest.raises(ValueError, match = "expert parallel \\(ep\\) cannot be combined"):
        pe.apply_parallel_env(environ)
    assert environ == before  # nothing written


def test_launch_composes_ep_with_cp():
    # Expert parallel x context parallel (the registry's ep x cp cells decide the backend and model).
    (pe,) = _package_modules("parallel_env")
    for environ in (
        {"WORLD_SIZE": "4", "UNSLOTH_EP_SIZE": "2", "UNSLOTH_CP_SIZE": "2"},
        {"WORLD_SIZE": "4", "UNSLOTH_PARALLEL": "cp=2", "UNSLOTH_EP_SIZE": "2"},
        {"WORLD_SIZE": "4", "UNSLOTH_PARALLEL": "ep=2,cp=2"},
    ):
        environ = dict(environ)
        pe.apply_parallel_env(environ)
        assert environ["UNSLOTH_EP_SIZE"] == "2" and environ["UNSLOTH_CP_SIZE"] == "2", environ
    # ep in the flag with an exported cp the flag does not give: the usual flag / variable conflict.
    with pytest.raises(ValueError, match = "UNSLOTH_CP_SIZE=2 asks for something else"):
        pe.apply_parallel_env(
            {"WORLD_SIZE": "4", "UNSLOTH_PARALLEL": "ep=2", "UNSLOTH_CP_SIZE": "2"}
        )


def test_launch_keeps_ep_alone_and_size_one_axes():
    (pe,) = _package_modules("parallel_env")
    for environ in (
        {"WORLD_SIZE": "2", "UNSLOTH_EP_SIZE": "2"},
        {"WORLD_SIZE": "2", "UNSLOTH_EP_SIZE": "2", "UNSLOTH_TP_SIZE": "1", "UNSLOTH_FSDP2": "0"},
        {"WORLD_SIZE": "2", "UNSLOTH_EP_SIZE": "1", "UNSLOTH_TP_SIZE": "2"},
        {"WORLD_SIZE": "2", "UNSLOTH_PARALLEL": "ep=2"},
    ):
        pe.apply_parallel_env(dict(environ))


@pytest.mark.parametrize("axis", ["tp", "pp"])
def test_parallelize_refuses_an_inherited_ep(monkeypatch, axis):
    (pe,) = _package_modules("parallel_env")
    for name in _ENV:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("UNSLOTH_EP_SIZE", "2")
    monkeypatch.setattr(pe, "_SET_BY_PARALLELIZE", [None, {}])
    monkeypatch.setattr(pe, "_reinstall_after_import", lambda: None)
    before = dict(os.environ)
    try:
        with pytest.raises(ValueError, match = "expert parallel \\(ep\\) cannot be combined"):
            pe.parallelize(**{axis: 2})
        assert dict(os.environ) == before  # nothing written
        assert pe._SET_BY_PARALLELIZE == [None, {}]
    finally:  # an accepted call must not leak what it wrote into the later (spawned) tests
        for name in set(os.environ) - set(before):
            del os.environ[name]


def test_auto_refuses_ep():
    pe, auto = _package_modules("parallel_env", "parallel_auto")
    with pytest.raises(ValueError, match = "UNSLOTH_EP_SIZE=2 asks for another axis"):
        auto.check_auto_env({"UNSLOTH_PARALLEL": "auto", "UNSLOTH_EP_SIZE": "2"})


def test_installer_predicates_include_ep(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_EP_SIZE", "2")
    from unsloth_models_loader import load_models_module

    assert load_models_module("tp_env").parallel_env_requested()
    src = open(os.path.join(REPO, "unsloth", "__init__.py"), encoding = "utf-8").read()
    assert '"UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_CP_SIZE", "UNSLOTH_EP_SIZE"' in src


def test_ep_env_mesh_refuses_cuda_without_nccl(monkeypatch):
    from unsloth_models_loader import load_models_module

    epn = load_models_module("ep_native")
    import torch.distributed as dist

    monkeypatch.setenv("UNSLOTH_EP_SIZE", "2")
    monkeypatch.setattr(epn, "_ENV_MESH", [])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: False)

    def no_call(*a, **k):
        raise AssertionError("reached CUDA / process group setup")

    monkeypatch.setattr(torch.cuda, "set_device", no_call)
    monkeypatch.setattr(dist, "init_process_group", no_call)
    with pytest.raises(RuntimeError, match = "is_nccl_available\\(\\) is False"):
        epn.ep_env_mesh()
    assert epn._ENV_MESH == []


def test_trainer_refuses_accelerate_fsdp_on_ep_model():
    from unsloth_models_loader import load_models_module

    epn = load_models_module("ep_native")
    model = torch.nn.Linear(2, 2)
    model.__dict__[epn._EP] = object()
    with pytest.raises(NotImplementedError, match = "accelerate's FSDP"):
        epn.ep_refuse_trainer_fsdp(types.SimpleNamespace(model = model, is_fsdp_enabled = True))
    epn.ep_refuse_trainer_fsdp(types.SimpleNamespace(model = model, is_fsdp_enabled = False))
    with pytest.raises(NotImplementedError, match = "DeepSpeed"):
        epn.ep_refuse_trainer_fsdp(
            types.SimpleNamespace(model = model, is_fsdp_enabled = False, is_deepspeed_enabled = True)
        )
    epn.ep_refuse_trainer_fsdp(
        types.SimpleNamespace(model = torch.nn.Linear(2, 2), is_deepspeed_enabled = True)
    )
    epn.ep_refuse_trainer_fsdp(
        types.SimpleNamespace(model = torch.nn.Linear(2, 2), is_fsdp_enabled = True)
    )


@pytest.mark.parametrize(
    "flag, what",
    [("load_in_8bit", "8-bit weights"), ("float8_training", "fp8 training")],
)
def test_ep_env_load_refuses_unsupported_precision_before_loading(monkeypatch, flag, what):
    from unsloth_models_loader import load_models_module

    for name in _ENV:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_EP_SIZE", "2")
    tp_env = load_models_module("tp_env")
    assert tp_env.env_layout() is None  # the expert parallel branch, not the tp / pp one
    calls = []

    def from_pretrained(
        model_name,
        load_in_4bit = True,
        load_in_8bit = False,
        float8_training = None,
        **kwargs,
    ):
        calls.append(model_name)
        return torch.nn.Linear(2, 2)

    with pytest.raises(NotImplementedError, match = f"expert parallel\\) does not support {what}"):
        tp_env.tensor_parallel_env_load(from_pretrained)("x", **{flag: True})
    assert calls == []


def _snapshot(model):
    return (
        [(n, p.data_ptr(), tuple(p.shape), type(p).__name__) for n, p in model.named_parameters()],
        [(n, type(m).__name__, m.__dict__.get("forward")) for n, m in model.named_modules()],
    )


def _moe(lora = True):
    from test_tp_native_oracle import build
    if not lora:
        return build(MOE, lora = False).train()
    return build(MOE, target_parameters = STACKS).train()


def _refused(fn):
    try:
        fn()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _case_entry_points(rank, world, tmp):
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth_models_loader import load_models_module

    epn = load_models_module("ep_native")
    tpn = load_models_module("tp_native")
    ppn = load_models_module("pp_native")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    out = {}

    tp_mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]
    pp_mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    for key, apply, applied in (
        ("tp", lambda m: tpn.apply_tensor_parallel(m, tp_mesh), tpn.is_tensor_parallel),
        ("pp", lambda m: ppn.apply_pipeline_parallel(m, pp_mesh), ppn.is_pipeline_parallel),
    ):
        model = _moe()
        epn.apply_expert_parallel(model, mesh)
        assert epn.is_expert_parallel(model)
        snap = _snapshot(model)
        out[key] = _refused(lambda: apply(model))
        out[f"{key}_same"] = _snapshot(model) == snap and not applied(model)

    from test_pp_moe_aux_gloo import build as pp_build

    stage = pp_build(lora = True)
    ppn.apply_pipeline_parallel(stage, pp_mesh)
    snap = _snapshot(stage)
    out["ep_on_pp"] = _refused(lambda: epn.apply_expert_parallel(stage, mesh))
    out["ep_on_pp_same"] = _snapshot(stage) == snap and not epn.is_expert_parallel(stage)

    model = _moe()
    model.vllm_engine = object()  # as FastLanguageModel(fast_inference = True) leaves it
    snap = _snapshot(model)
    out["ep_on_vllm"] = _refused(lambda: epn.apply_expert_parallel(model, mesh))
    out["ep_on_vllm_same"] = _snapshot(model) == snap and not epn.is_expert_parallel(model)
    return out


def test_manual_entry_points_refuse_before_any_change():
    results = _spawn("_case_entry_points")
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["tp"] and "NotImplementedError" in r["tp"] and "expert parallel" in r["tp"], r
        assert r["pp"] and "NotImplementedError" in r["pp"] and "expert parallel" in r["pp"], r
        assert r["ep_on_pp"] and "pipeline parallel stage" in r["ep_on_pp"], r
        assert r["ep_on_vllm"] and "vLLM" in r["ep_on_vllm"], r
        assert all(r[k] for k in r if k.endswith("_same")), r


def _case_hf_distributed_kwargs(rank, world, tmp):
    from unsloth_models_loader import load_models_module

    epn = load_models_module("ep_native")
    tpn = load_models_module("tp_native")
    tp_env = load_models_module("tp_env")
    calls = []

    def from_pretrained(
        model_name,
        tp_plan = None,
        **kwargs,
    ):
        calls.append(tp_plan)
        return torch.nn.Linear(2, 2)

    out = {
        "tp_env": _refused(
            lambda: tp_env.tensor_parallel_env_load(from_pretrained)("x", tp_plan = "auto")
        )
    }
    out["tp_env_calls"] = len(calls)

    # A refusal raised inside the sharded load is not swallowed into a whole load with the same kwarg.
    def raising(*a, **k):
        calls.append("raising")
        if len(calls) == 1:
            raise tpn.HFDistributedKwargsError(
                "Unsloth: expert parallel sharded loading cannot be combined"
            )
        return "whole"

    calls.clear()
    out["ep_env_load"] = _refused(lambda: epn.ep_env_load(raising, (), {}, True))
    out["ep_env_load_calls"] = len(calls)
    return out


def test_ep_load_refuses_hf_distributed_kwargs():
    results = _spawn("_case_hf_distributed_kwargs", env = _env())
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["tp_env"] and "HFDistributedKwargsError" in r["tp_env"], r
        assert "transformers' own tp_plan" in r["tp_env"] and r["tp_env_calls"] == 0, r
        assert r["ep_env_load"] and "HFDistributedKwargsError" in r["ep_env_load"], r


def _case_plain_trainer(rank, world, tmp):
    from test_tp_native_gloo import _import_checkout_unsloth, _Rows, _trainer_args

    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import transformers
    from peft import LoraConfig, get_peft_model

    from unsloth.models import ep_native

    calls = []
    setup = ep_native.ep_trainer_setup

    def counted(trainer):
        calls.append(1)
        return setup(trainer)

    ep_native.ep_trainer_setup = counted
    torch.manual_seed(1)
    model = get_peft_model(
        _moe(lora = False),
        LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj"], target_parameters = STACKS),
    )
    data = _Rows(list(torch.randint(1, 100, (4, 9), generator = torch.Generator().manual_seed(1))))
    args = _trainer_args(os.path.join(tmp, "out"), 1, gradient_accumulation_steps = 1)
    trainer = transformers.Trainer(model = model, args = args, train_dataset = data)
    out = {
        "calls": len(calls),
        "is_ep": ep_native.is_expert_parallel(trainer.model),
        "backward": getattr(trainer.accelerator.backward, "__name__", ""),
    }
    out["model_init"] = _refused(
        lambda: transformers.Trainer(
            model_init = lambda: _moe(lora = False), args = args, train_dataset = data
        )
    )
    return out


def test_plain_trainer_under_ep_env_runs_the_ep_setup():
    results = _spawn("_case_plain_trainer", env = _env())
    if results[0] is not None and "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["calls"] == 1 and r["is_ep"] and r["backward"] == "ep_backward", r
        assert r["model_init"] and "model_init" in r["model_init"], r


def _env():
    from tests._rendezvous_port import free_port
    return {"UNSLOTH_EP_SIZE": "2", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(free_port())}


def _worker(rank, world, tmp, case, env):
    os.environ.update(
        RANK = str(rank), WORLD_SIZE = str(world), LOCAL_RANK = str(rank), LOCAL_WORLD_SIZE = str(world)
    )
    os.environ.update(env or {})
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    if "MASTER_PORT" not in (env or {}):  # else Unsloth creates the group from the environment
        dist.init_process_group(
            "gloo",
            store = dist.FileStore(os.path.join(tmp, "rendezvous"), world),
            rank = rank,
            world_size = world,
            timeout = datetime.timedelta(seconds = 300),
        )
    try:
        res = globals()[case](rank, world, tmp)
    except Exception as exc:
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-6000:]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    if dist.is_initialized():
        dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
    env = None,
):
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, world, tmp, case, env)) for r in range(world)]
        for p in procs:
            p.start()
        deadline = time.monotonic() + 600
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            time.sleep(0.2)
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(30)
        out = []
        for r in range(world):
            path = os.path.join(tmp, f"result_{r}.pt")
            out.append(torch.load(path, weights_only = False) if os.path.exists(path) else None)
        return out
