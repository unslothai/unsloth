# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth/optimizers/flat_adamw8bit.py: when the fused 8-bit AdamW step is (not) installed. CPU only:
the plan and install logic, with stand-in parameters where the fused path would need CUDA."""

import importlib.util
import os
import pathlib
import tempfile
import types

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flat = _load("_unsloth_flat_adamw8bit_eligibility", "unsloth/optimizers/flat_adamw8bit.py")
fsdp2_bnb = _load("_unsloth_fsdp2_bnb_optim_eligibility", "unsloth/models/fsdp2_bnb_optim.py")


def _bnb_opt(n = 4096):
    p = torch.nn.Parameter(torch.zeros(n))
    return bnb.optim.AdamW8bit([p], lr = 1e-3)


class _FakeCudaParam:
    """Passes _group_plan's per-parameter checks without a CUDA device."""

    def __init__(self, n):
        self.data = torch.empty(0)
        self.device = torch.device("cuda")
        self.dtype = torch.float32
        self._n = n

    def numel(self):
        return self._n

    def is_contiguous(self):
        return True


def _fake_opt(**args):
    base = dict(optim_bits = 8, min_8bit_size = 4096, max_unorm = 0.0)
    base.update(args)
    params = [_FakeCudaParam(4096), _FakeCudaParam(8192)]
    opt = types.SimpleNamespace(
        args = types.SimpleNamespace(**base),
        is_paged = False,
        mng = types.SimpleNamespace(index2config = {}, pid2config = {}),
    )
    group = {"params": params}
    opt.param_groups = [group]
    return opt, group


# bnb 0.50 dropped percentile_clipping and block_wise from args: absent means the defaults.
_REJECTED = [
    ({"percentile_clipping": 5}, "percentile_clipping"),
    ({"percentile_clipping": 99}, "percentile_clipping"),
    ({"max_unorm": 1.0}, "max_unorm"),
    ({"block_wise": False}, "block_wise"),
    ({"optim_bits": 32}, "optim_bits"),
    ({"skip_zeros": True}, "skip_zeros"),
]


@pytest.mark.parametrize("args, reason", _REJECTED, ids = [r for _, r in _REJECTED])
def test_group_plan_rejects_per_tensor_options(args, reason):
    opt, group = _fake_opt(**args)
    assert flat._group_plan(opt, 0, group) is None
    assert reason in flat._unsupported_args(opt)


@pytest.mark.parametrize(
    "args",
    [{}, {"percentile_clipping": 100, "block_wise": True, "skip_zeros": False}],
    ids = ["bnb050_args", "bnb045_defaults"],
)
def test_group_plan_accepts_stock_defaults(args):
    opt, group = _fake_opt(**args)
    plan = flat._group_plan(opt, 0, group)
    assert plan is not None and plan["segments"] == [(0, 2)]
    assert flat._unsupported_args(opt) is None


def test_group_plan_rejects_paged():
    opt, group = _fake_opt()
    opt.is_paged = True
    assert flat._group_plan(opt, 0, group) is None


def test_install_refuses_percentile_clipping():
    opt = _bnb_opt()
    opt.args.percentile_clipping = 5  # bnb < 0.50 AdamW8bit(percentile_clipping = 5)
    assert flat.install_flat_adamw8bit(opt) is None
    assert not getattr(type(opt), "_unsloth_flat_adamw8bit", False)


def test_install_on_plain_bnb_adamw8bit():
    opt = _bnb_opt()
    assert flat.install_flat_adamw8bit(opt) is opt
    assert type(opt)._unsloth_flat_adamw8bit
    assert type(opt)._unsloth_flat_base is bnb.optim.AdamW8bit


def _trainer(**attrs):
    calls = []

    def create_scheduler(num_training_steps, optimizer = None):
        calls.append(optimizer)

    trainer = types.SimpleNamespace(create_scheduler = create_scheduler, **attrs)
    return trainer, create_scheduler, calls


def _state(**kw):
    return types.SimpleNamespace(
        accelerator = types.SimpleNamespace(state = types.SimpleNamespace(**kw))
    )


_DISTRIBUTED = [
    ("is_fsdp_enabled", {"is_fsdp_enabled": True}),
    ("is_fsdp_xla_enabled", {"is_fsdp_xla_enabled": True}),
    ("is_deepspeed_enabled", {"is_deepspeed_enabled": True}),
    ("args_fsdp", {"args": types.SimpleNamespace(fsdp = ["full_shard"], deepspeed = None)}),
    ("args_deepspeed", {"args": types.SimpleNamespace(fsdp = [], deepspeed = "ds.json")}),
    ("fsdp_plugin", vars(_state(fsdp_plugin = object()))),
    ("deepspeed_plugin", vars(_state(deepspeed_plugin = object()))),
    ("distributed_type_fsdp", vars(_state(distributed_type = "DistributedType.FSDP"))),
    ("distributed_type_deepspeed", vars(_state(distributed_type = "DistributedType.DEEPSPEED"))),
]


@pytest.mark.parametrize("attrs", [a for _, a in _DISTRIBUTED], ids = [i for i, _ in _DISTRIBUTED])
def test_hook_not_installed_under_fsdp_or_deepspeed(attrs):
    trainer, original, _ = _trainer(**attrs)
    assert flat.install_flat_adamw8bit_hook(trainer) is False
    assert trainer.create_scheduler is original


@pytest.mark.parametrize("env", ["ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED"])
def test_hook_not_installed_under_accelerate_env(monkeypatch, env):
    monkeypatch.setenv(env, "true")
    trainer, original, _ = _trainer()
    assert flat.install_flat_adamw8bit_hook(trainer) is False
    assert trainer.create_scheduler is original


def test_hook_installs_for_plain_trainer_and_rechecks_at_call(monkeypatch):
    for env in ("ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED"):
        monkeypatch.delenv(env, raising = False)
    trainer, _, calls = _trainer(
        args = types.SimpleNamespace(fsdp = [], deepspeed = None),
        **vars(_state(distributed_type = "DistributedType.MULTI_GPU")),
    )
    assert flat.install_flat_adamw8bit_hook(trainer) is True
    opt = _bnb_opt()
    trainer.create_scheduler(10, optimizer = opt)
    assert calls == [opt] and type(opt)._unsloth_flat_adamw8bit

    # FSDP switched on after Trainer.__init__ (before create_scheduler): the class stays stock.
    trainer.is_fsdp_enabled = True
    opt = _bnb_opt()
    trainer.create_scheduler(10, optimizer = opt)
    assert type(opt) is bnb.optim.AdamW8bit


def test_localize_restores_flat_class_for_fsdp2():
    # accelerate FSDP2 prepare -> _localize_or_refuse_bnb -> localize_bnb_optimizer: the flat class
    # overrides step, so the plain-class check refused it and the Trainer raised ValueError.
    opt = _bnb_opt()
    assert flat.install_flat_adamw8bit(opt) is opt
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    assert fsdp2_bnb.localize_bnb_optimizer(opt) is True
    assert fsdp2_bnb.is_local_shard_bnb_optimizer(opt)
    assert not getattr(type(opt), "_unsloth_flat_adamw8bit", False)
    assert "_unsloth_flat_plans" not in vars(opt)
    # The scheduler's instance step points at the localized class step, not the flat one.
    assert opt.step.__wrapped__ is type(opt).step
    del sched


@pytest.fixture
def _single_rank_gloo():
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    if dist.is_initialized():
        yield
        return
    with tempfile.TemporaryDirectory() as d:
        dist.init_process_group(
            "gloo", init_method = f"file://{os.path.join(d, 'store')}", rank = 0, world_size = 1
        )
        try:
            yield
        finally:
            dist.destroy_process_group()


def test_install_refuses_dtensor_params(_single_rank_gloo):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    mesh = init_device_mesh("cpu", (1,))
    p = torch.nn.Parameter(distribute_tensor(torch.zeros(4096), mesh, [Shard(0)]))
    opt = bnb.optim.AdamW8bit([p], lr = 1e-3)
    assert flat.install_flat_adamw8bit(opt) is None
    assert type(opt) is bnb.optim.AdamW8bit
    assert flat._has_dtensor_params(opt)
