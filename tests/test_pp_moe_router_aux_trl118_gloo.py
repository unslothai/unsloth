# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL >= 1.18 (trl#8103) passes output_router_logits per forward instead of writing the config; the pp
Trainer step skips compute_loss, so it must honour aux_loss_enabled. gloo pp2 vs one process, tiny Qwen3-MoE."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pp_moe_aux_gloo import REPO, _load, batch, build  # noqa: E402

ppn = _load("pp_native")
pp_env = _load("pp_env")
assert os.path.realpath(pp_env.__file__).startswith(os.path.realpath(REPO))  # this checkout's tree

CONFIG_COEF = (
    1.0  # large enough that dropping the term moves the router gradients far past the tolerance
)


def _gpu_compute_loss(self, *a, **k):
    raise AssertionError("the pipeline step never calls compute_loss")


def _zoo_cpu_compute_loss(self, *a, **k):
    raise AssertionError((self.router_aux_loss_coef, self.current_gradient_accumulation_steps))


_zoo_cpu_compute_loss._unsloth_cpu = True  # unsloth_zoo.cpu.trainer._patch_sft_compute_loss


def _trainer(stage, aux_loss_enabled, coef, zoo_cpu):
    """aux_loss_enabled None = a plain transformers Trainer."""
    body = {"compute_loss": _zoo_cpu_compute_loss if zoo_cpu else _gpu_compute_loss}
    body["_prepare_inputs"] = lambda self, x: x
    trainer = type("SFTTrainer", (), body)()
    trainer.args = types.SimpleNamespace(
        device = torch.device("cpu"),
        gradient_accumulation_steps = 1,
        average_tokens_across_devices = False,
        world_size = 2,
        loss_type = "nll",
    )
    trainer.accelerator = types.SimpleNamespace(scaler = None, unwrap_model = lambda m: m)
    trainer.optimizer = None
    trainer.current_gradient_accumulation_steps = 1
    if aux_loss_enabled is not None:
        trainer.aux_loss_enabled = aux_loss_enabled
        trainer.router_aux_loss_coef = coef
    trainer.__dict__["_unsloth_pipeline_parallel"] = stage
    return trainer


def _one_process(config_flag, kwarg, coef, ids, labels, attention_mask):
    model = build(aux = config_flag, coef = CONFIG_COEF)
    model.router_aux_loss_coef = (
        coef  # what Unsloth's SFTTrainer init hands the CausalLM (models/rl.py)
    )
    n_items = int((labels[:, 1:] != -100).sum())
    extra = {} if kwarg is None else {"output_router_logits": kwarg}
    out = model(
        input_ids = ids,
        labels = labels,
        attention_mask = attention_mask,
        num_items_in_batch = n_items,
        use_cache = False,
        **extra,
    )
    out.loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    aux = float(out.aux_loss) if getattr(out, "aux_loss", None) is not None else 0.0
    return float(out.loss), aux, grads


def _case(rank, world, tmp, config_flag, aux_loss_enabled, coef, zoo_cpu, n_mb):
    from torch.distributed.device_mesh import init_device_mesh

    ids, labels, attention_mask = batch()
    kwarg = True if aux_loss_enabled else None
    out = {"ref": _one_process(config_flag, kwarg, coef, ids, labels, attention_mask)[:2]}
    ref_grads = _one_process(config_flag, kwarg, coef, ids, labels, attention_mask)[2]
    model = build(aux = config_flag, coef = CONFIG_COEF)
    model.router_aux_loss_coef = coef
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    try:
        stage = ppn.apply_pipeline_parallel(model, mesh, n_microbatches = n_mb, schedule = "gpipe")
    except ValueError as exc:
        if "must be greater than or equal to the number of stages" not in str(exc):
            raise
        out["skip"] = str(exc)  # torch < 2.11 GPipe: one micro-batch cannot run over 2 stages
        return out
    trainer = _trainer(stage, aux_loss_enabled, coef, zoo_cpu)
    inputs = {"input_ids": ids, "labels": labels, "attention_mask": attention_mask}
    n_items = int((labels[:, 1:] != -100).sum())
    try:
        loss = pp_env._pp_training_step(trainer, model, inputs, num_items_in_batch = n_items)
    except NotImplementedError as exc:
        out["refused"] = str(exc)
        return out
    out["refused"] = None
    out["loss"] = float(loss)
    out["aux"] = float(stage.last_aux_loss)
    out["ctx_cleared"] = getattr(stage.ctx, "output_router_logits", None) is None
    diffs = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        want = ref_grads[name]
        group = "router" if "mlp.gate." in name else "other"
        d, s = diffs.get(group, (0.0, 0.0))
        diffs[group] = (max(d, float((p.grad - want).abs().max())), max(s, float(want.abs().max())))
    out["diffs"] = diffs
    return out


def _worker(rank, world, port, tmp, kwargs):
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

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 180)
    )
    try:
        torch.save(_case(rank, world, tmp, **kwargs), os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(world = 2, **kwargs):
    try:
        ppn._check_torch_pipelining()
    except RuntimeError as exc:
        pytest.skip(str(exc))
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
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


# (config.output_router_logits after the trainer init, trainer.aux_loss_enabled, trainer coef, zoo CPU compute_loss)
CELLS = {
    # TRL >= 1.18: the config keeps the checkpoint's False; only aux_loss_enabled turns the loss on.
    "trl118_on": dict(config_flag = False, aux_loss_enabled = True, coef = 0.5, zoo_cpu = False),
    "trl118_on_zoo_cpu": dict(config_flag = False, aux_loss_enabled = True, coef = 0.5, zoo_cpu = True),
    # TRL 1.15 - 1.17 wrote the config too: both on, added once.
    "trl117_on": dict(config_flag = True, aux_loss_enabled = True, coef = 0.5, zoo_cpu = False),
    "trl_off": dict(config_flag = False, aux_loss_enabled = False, coef = 0.0, zoo_cpu = False),
    "plain_trainer_config_on": dict(
        config_flag = True, aux_loss_enabled = None, coef = CONFIG_COEF, zoo_cpu = False
    ),
}


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_pp2_trainer_step_keeps_the_trl_router_aux_loss(cell):
    results = _spawn(n_mb = 1, **CELLS[cell])
    if "skip" in results[0]:
        pytest.skip(results[0]["skip"])
    ref_loss, ref_aux = results[0]["ref"]
    on = CELLS[cell]["config_flag"] or bool(CELLS[cell]["aux_loss_enabled"])
    assert (ref_aux > 0) == on, (cell, ref_aux)
    for res in results:
        assert res["refused"] is None, res["refused"]
        assert res["ctx_cleared"], cell
        assert res["loss"] == pytest.approx(ref_loss, rel = 1e-5, abs = 1e-6), (
            cell,
            res["loss"],
            ref_loss,
        )
        assert res["aux"] == pytest.approx(ref_aux if on else 0.0, rel = 1e-5, abs = 1e-6), (
            cell,
            res["aux"],
        )
        for group, (diff, scale) in res["diffs"].items():
            assert diff <= 1e-6 + 1e-4 * scale, (cell, group, diff, scale)


def test_pp2_trainer_step_refuses_split_aux_on_trl118():
    # Summing per-micro-batch aux losses is a different objective: refused.
    results = _spawn(n_mb = 2, **CELLS["trl118_on"])
    for res in results:
        assert "micro-batch" in (res["refused"] or ""), res["refused"]
