# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallel evaluation on a gloo CPU world vs one process, fp32.

- The HF Trainer with eval_strategy = "steps" (every step), Trainer.evaluate and Trainer.predict on a tp model
  (and dp x tp after shard_model): the gather covers the dp replicas only, so eval_loss / predictions equal
  one process (the tp ranks of a replica hold identical results; a world gather would count each tp times)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tempfile  # noqa: E402

from test_tp_native_gloo import _free_port, _import_checkout_unsloth, _Rows, _trainer_args  # noqa: E402
from test_tp_native_oracle import LLAMA, _batch, build  # noqa: E402


def _case_eval(rank, world, tmp, dp, tp):
    """3 Trainer steps evaluating every step, then evaluate() and predict() on 7 rows (a dp remainder)."""
    import transformers

    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth.models import fsdp2_native, tp_native
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)

    class SequentialTrainer(transformers.Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    model = build(LLAMA)
    data = _Rows(list(_batch(model, bsz = 12 * dp, seq = 9)))
    held_out = _Rows(list(_batch(model, seed = 2, bsz = 7, seq = 9)))
    if tp > 1:
        mesh = init_device_mesh("cpu", (dp, tp), mesh_dim_names = ("dp", "tp"))
        tp_native.apply_tensor_parallel(model, mesh["tp"])
        if dp > 1:
            fsdp2_native.shard_model(
                model, fsdp2_native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"])
            )
    args = _trainer_args(
        tmp,
        2 * dp if tp == 1 else 2,
        eval_strategy = "steps",
        eval_steps = 1,
        per_device_eval_batch_size = 1,
        dataloader_drop_last = False,
    )
    trainer = SequentialTrainer(
        model = model, args = args, train_dataset = data, eval_dataset = held_out
    )
    trainer.train()
    history = trainer.state.log_history
    final = trainer.evaluate()
    predicted = trainer.predict(held_out)
    return {
        "losses": [h["loss"] for h in history if "loss" in h],
        "eval_losses": [h["eval_loss"] for h in history if "eval_loss" in h],
        "final_eval": final["eval_loss"],
        "predictions": torch.as_tensor(predicted.predictions),
        "label_ids": torch.as_tensor(predicted.label_ids),
        "predict_loss": predicted.metrics["test_loss"],
    }


CASES = {"eval": _case_eval}


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


def _spawn(case, world = 2, **kwargs):
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


@pytest.mark.parametrize("dp, tp", [(1, 2), (2, 2)])
def test_trainer_evaluates_a_tensor_parallel_model_like_one_process(dp, tp):
    ref = _spawn("eval", 1, dp = dp, tp = 1)[0]
    if "skip" in ref:
        pytest.skip(f"import unsloth failed on this host: {ref['skip']}")
    results = _spawn("eval", dp * tp, dp = dp, tp = tp)
    assert len(ref["losses"]) == 3 and len(ref["eval_losses"]) == 4  # 3 during training + evaluate()
    assert ref["predictions"].shape[0] == 7
    for rank, res in enumerate(results):
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-5), (rank, res["losses"], ref["losses"])
        assert res["eval_losses"] == pytest.approx(ref["eval_losses"], rel = 1e-5), (
            rank,
            res["eval_losses"],
            ref["eval_losses"],
        )
        assert res["final_eval"] == pytest.approx(ref["final_eval"], rel = 1e-5)
        assert res["predict_loss"] == pytest.approx(ref["predict_loss"], rel = 1e-5)
        assert res["predictions"].shape == ref["predictions"].shape, (rank, res["predictions"].shape)
        torch.testing.assert_close(res["predictions"], ref["predictions"], rtol = 1e-4, atol = 1e-5)
        assert torch.equal(res["label_ids"], ref["label_ids"])
