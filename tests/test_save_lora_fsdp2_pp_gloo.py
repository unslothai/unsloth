# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`model.save_lora(dir)` (models/_utils.py save_lora_adapter, the GRPO notebook save) on two gloo CPU ranks.

Every rank makes the call, as a notebook script under torchrun does, after a few training steps:
- native FSDP2 (fully_shard): the LoRA shards are gathered and rank 0 writes the 1-GPU adapter. Before, every
  rank handed its DTensor shards to save_pretrained and wrote them into the same directory.
- pipeline parallel (UNSLOTH_PP_SIZE=2, two stages): every stage's adapter is gathered and rank 0 writes it.
  Before, each stage wrote its own half of the adapter into the same directory.
The written adapter is loaded into a one-process model and must equal the trained LoRA weights exactly.
"""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

if (
    sys.platform == "win32"
    or not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available()
):
    pytest.skip("needs torch.distributed with gloo", allow_module_level = True)
pytest.importorskip("peft")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_gloo import (  # noqa: E402
    _Rows,
    _free_port,
    _import_checkout_unsloth,
    _save_checkpoint,
    _trainer_args,
)

pytestmark = pytest.mark.xdist_group("test_save_lora_fsdp2_pp_gloo")

_ENV_KEYS = (
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_PP_MICROBATCHES",
    "UNSLOTH_PP_SCHEDULE",
    "UNSLOTH_FSDP2",
    "ACCELERATE_USE_FSDP",
)


def _lora_config():
    from peft import LoraConfig
    return LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)


def _reload(ckpt, directory):
    """The saved adapter on a one-process model: {parameter name: tensor} of its LoRA weights."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype = torch.float32, attn_implementation = "eager"
    )
    # gloo / CPU: say so, rather than let PEFT read the device off a probe the CUDA spoof patches.
    model = PeftModel.from_pretrained(base, directory, torch_device = "cpu")
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def _gathered_trainable(model):
    """Every rank's trainable LoRA tensors, merged by name (collective); whole tensors on the CPU."""
    import torch.distributed as dist

    local = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        value = param.full_tensor() if hasattr(param, "full_tensor") else param
        local[name] = value.detach().to("cpu", copy = True)
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, local)
    merged = {}
    for part in parts:
        merged.update(part)
    return merged


def _case_fsdp2(rank, world, tmp, ckpt):
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import torch.distributed as dist
    from peft import get_peft_model
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from transformers import AutoModelForCausalLM

    from unsloth.models._utils import save_lora_adapter

    dist.init_process_group("gloo", rank = rank, world_size = world)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype = torch.float32, attn_implementation = "eager"
    )
    torch.manual_seed(1)
    model = get_peft_model(model, _lora_config())
    mesh = init_device_mesh("cpu", (world,))
    for layer in model.get_base_model().model.layers:
        fully_shard(layer, mesh = mesh)
    fully_shard(model, mesh = mesh)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 0.5)
    g = torch.Generator().manual_seed(rank)
    for _ in range(2):
        ids = torch.randint(0, 1000, (2, 9), generator = g)
        model(input_ids = ids, labels = ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    trained = _gathered_trainable(model)
    directory = os.path.join(tmp, "adapter")
    returned = save_lora_adapter(model, directory)
    res = {"returned": returned, "trained": trained if rank == 0 else None}
    dist.barrier()
    if rank == 0:
        res["reloaded"] = _reload(ckpt, directory)
    return res


def _case_pp(rank, world, tmp, ckpt):
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import torch.distributed as dist
    import transformers
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    from unsloth.models import pp_native, tp_env
    from unsloth.models._utils import save_lora_adapter

    def load(model_name, **kwargs):
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype = torch.float32, attn_implementation = "eager"
        )

    def peft(model):
        torch.manual_seed(1)
        return get_peft_model(model, _lora_config())

    model = tp_env.tensor_parallel_env_peft(peft)(tp_env.tensor_parallel_env_load(load)(ckpt))
    ids = torch.randint(0, 1000, (8, 9), generator = torch.Generator().manual_seed(1))
    trainer = transformers.Trainer(
        model = model,
        args = _trainer_args(os.path.join(tmp, "out"), 2, max_steps = 2),
        train_dataset = _Rows(list(ids)),
    )
    trainer.train()
    stage = pp_native.pipeline_stage_of(trainer.model)
    trained = _gathered_trainable(trainer.model)
    directory = os.path.join(tmp, "adapter")
    returned = save_lora_adapter(trainer.model, directory)
    res = {
        "stage": None if stage is None else stage.stage_index,
        "returned": returned,
        "trained": trained if rank == 0 else None,
    }
    dist.barrier()
    if rank == 0:
        res["reloaded"] = _reload(ckpt, directory)
    return res


CASES = {"fsdp2": _case_fsdp2, "pp": _case_pp}


def _worker(rank, world, port, case, tmp, env, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)
    torch.set_num_threads(2)
    import torch.distributed as dist

    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(
    case,
    world,
    env = None,
):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _save_checkpoint("llama", os.path.join(tmp, "ckpt"), {"num_hidden_layers": 4})
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, env or {}, {"ckpt": ckpt}))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        codes = [p.exitcode for p in procs]
        for p in procs:
            if p.is_alive():
                p.kill()
        assert codes == [0] * world, f"rank exit codes {codes}"
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    return results


def _same(reloaded, trained):
    assert trained and set(reloaded) == set(trained), sorted(set(reloaded) ^ set(trained))[:6]
    for name, value in trained.items():
        assert torch.equal(reloaded[name], value), name


def _pipelining_available():
    try:
        import inspect

        from torch.distributed.pipelining import ScheduleGPipe
    except ImportError:
        return False
    return "return_outputs" in inspect.signature(ScheduleGPipe.step).parameters


def test_save_lora_fsdp2_writes_the_gathered_adapter_from_rank_0():
    fsdp = pytest.importorskip("torch.distributed.fsdp")
    if not hasattr(fsdp, "fully_shard"):
        pytest.skip("torch has no FSDP2 fully_shard")
    results = _spawn("fsdp2", 2)
    _same(results[0]["reloaded"], results[0]["trained"])
    assert [r["returned"] for r in results] == [None, None], [r["returned"] for r in results]


@pytest.mark.skipif(
    not _pipelining_available(),
    reason = "torch.distributed.pipelining with step(return_outputs) (torch >= 2.10)",
)
def test_save_lora_pipeline_parallel_writes_every_stage_from_rank_0():
    results = _spawn("pp", 2, {"UNSLOTH_PP_SIZE": "2"})
    assert sorted(r["stage"] for r in results) == [0, 1]
    _same(results[0]["reloaded"], results[0]["trained"])
