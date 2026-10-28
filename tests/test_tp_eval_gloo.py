# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallel evaluation and cached generate on a gloo CPU world vs one process, fp32.

- The HF Trainer with eval_strategy = "steps" (every step), Trainer.evaluate and Trainer.predict on a tp model
  (and dp x tp after shard_model): the gather covers the dp replicas only, so eval_loss / predictions equal
  one process (the tp ranks of a replica hold identical results; a world gather would count each tp times).
- Cached greedy generate after a forward wrapper is installed over the tensor parallel forward guard (what the
  GRPO hidden states wrapper in models/rl.py does to the model the GRPO trainer generates with): the gathered
  replica decodes with a KV cache instead of being refused mid-run."""

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
    trainer = SequentialTrainer(model = model, args = args, train_dataset = data, eval_dataset = held_out)
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


def _case_generate(rank, world, tmp, tp):
    """Greedy cached generate after the GRPO hidden states forward wrapper wraps the tp forward guard."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth.models import rl, tp_native, tp_replica

    model = build(LLAMA)
    prompt = _batch(model, seed = 3, bsz = 2, seq = 5)
    if tp > 1:
        mesh = init_device_mesh("cpu", (tp,), mesh_dim_names = ("tp",))
        tp_native.apply_tensor_parallel(model, mesh["tp"])
    wrapped = rl._install_grpo_hidden_states_forward_wrapper(model)
    model.eval()
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids = prompt,
                attention_mask = torch.ones_like(prompt),
                max_new_tokens = 6,
                do_sample = False,
                use_cache = True,
                pad_token_id = 0,
            )
        error = None
    except Exception as e:
        out, error = None, f"{type(e).__name__}: {e}"
    # Training forward after generate runs on the shards again (replica released).
    active = tp_replica.replica_active(model) if tp > 1 else False
    return {"out": out, "error": error, "wrapped": wrapped, "replica_after": active}


CASES = {"eval": _case_eval, "generate": _case_generate}


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
    world = 2,
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


def _torch_below_2_7():
    from packaging.version import Version
    return Version(torch.__version__.split("+")[0]) < Version("2.7")


@pytest.mark.parametrize("dp, tp", [(1, 2), (2, 2)])
def test_trainer_evaluates_a_tensor_parallel_model_like_one_process(dp, tp):
    if dp > 1 and _torch_below_2_7():
        pytest.skip(
            "torch < 2.7: tensor parallel x FSDP2 is refused (no fully_shard(ignored_params))"
        )
    ref = _spawn("eval", 1, dp = dp, tp = 1)[0]
    if "skip" in ref:
        pytest.skip(f"import unsloth failed on this host: {ref['skip']}")
    results = _spawn("eval", dp * tp, dp = dp, tp = tp)
    assert (
        len(ref["losses"]) == 3 and len(ref["eval_losses"]) == 4
    )  # 3 during training + evaluate()
    assert ref["predictions"].shape[0] == 7
    for rank, res in enumerate(results):
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-5), (
            rank,
            res["losses"],
            ref["losses"],
        )
        assert res["eval_losses"] == pytest.approx(ref["eval_losses"], rel = 1e-5), (
            rank,
            res["eval_losses"],
            ref["eval_losses"],
        )
        assert res["final_eval"] == pytest.approx(ref["final_eval"], rel = 1e-5)
        assert res["predict_loss"] == pytest.approx(ref["predict_loss"], rel = 1e-5)
        assert res["predictions"].shape == ref["predictions"].shape, (
            rank,
            res["predictions"].shape,
        )
        torch.testing.assert_close(res["predictions"], ref["predictions"], rtol = 1e-4, atol = 1e-5)
        assert torch.equal(res["label_ids"], ref["label_ids"])


def test_cached_generate_through_a_wrapped_tp_forward_matches_one_process():
    ref = _spawn("generate", 1, tp = 1)[0]
    if "skip" in ref:
        pytest.skip(f"import unsloth failed on this host: {ref['skip']}")
    assert ref["error"] is None, ref["error"]
    results = _spawn("generate", 2, tp = 2)
    for rank, res in enumerate(results):
        assert res[
            "wrapped"
        ], "the GRPO hidden states wrapper was not installed: the test proves nothing"
        assert res["error"] is None, (rank, res["error"])
        assert torch.equal(res["out"], ref["out"]), (rank, res["out"], ref["out"])
        assert res["replica_after"] is False
