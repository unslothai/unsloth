# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 resume of a DoRA adapter (LoraConfig(use_dora = True)), gloo world of 2. PEFT saves
DoRA's magnitude `lora_magnitude_vector.<adapter>.weight` as `lora_magnitude_vector` (peft
tuners/lora/model.py LoraModel._get_adapter_state_dict, called from get_peft_model_state_dict in
peft/utils/save_and_load.py). The native resume now asks PEFT's own save path for the key of every
parameter, and its fallback rule (no PEFT answer) mirrors the DoRA rename too: both restore every
adapter tensor exactly."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import datetime  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import (  # noqa: E402
    REPO_ID,
    _free_port,
    _full,
    _stub_unsloth_save,
    _trainer,
)


def _dora(seed):
    import transformers
    from peft import LoraConfig, get_peft_model

    config = transformers.AutoConfig.from_pretrained(REPO_ID)
    config.pad_token_id = None
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = None
    cfg = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj"], use_dora = True)
    model = get_peft_model(model, cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.normal_(0, 0.02, generator = g)
    return model


def _worker(rank, world, port, out, tmp, fallback):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
    )
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 60)
    )
    try:
        native = _load_native()
        _stub_unsloth_save()
        if fallback:
            native._peft_save_keys = lambda root, adapter: {}  # no PEFT answer: the rule alone
        cfg = native.FSDP2Config(param_dtype = torch.float32)
        ckpt = os.path.join(tmp, "ckpt")
        src = native.shard_model(_dora(1), cfg)
        _trainer(src, tmp).save_model(ckpt)
        dist.barrier()
        want = _full(src, trainable_only = True)
        dst = native.shard_model(_dora(2), cfg)
        res = {"magnitudes": sum("lora_magnitude_vector" in n for n in want)}
        keys = native._peft_save_keys(native._native_root(dst), "default")
        res["peft_magnitude_keys"] = sum(
            v.endswith(".lora_magnitude_vector") for k, v in keys.items() if "magnitude" in k
        )
        try:
            native.native_load_model_checkpoint(dst, ckpt)
            got = _full(dst, trainable_only = True)
            res["max_diff"] = max((got[n] - want[n]).abs().max().item() for n in want)
        except Exception as exc:
            res["load_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(
    tmp_path,
    fallback,
    timeout = 180,
):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker,
        args = (2, _free_port(), out, str(tmp_path), fallback),
        nprocs = 2,
        join = False,
        start_method = "spawn",
    )
    deadline = time.time() + timeout
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail("dora: the ranks did not finish (hung collective)")
    except mp.ProcessExitedException as exc:
        pytest.fail(f"dora: a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("ROUND14_RESULT dora", fallback, rank, r)
        assert "crash" not in r, r["crash"]
    return results


@pytest.mark.parametrize("fallback", [False, True], ids = ["peft_keys", "rule_only"])
def test_dora_adapter_resumes_exactly(tmp_path, fallback):
    pytest.importorskip("peft")
    for rank, r in _run(tmp_path, fallback).items():
        assert r["magnitudes"] > 0, (rank, r)
        # PEFT answers on its own (DoRA's key without ".weight"), unless switched off for the rule.
        assert r["peft_magnitude_keys"] == (0 if fallback else r["magnitudes"]), (rank, r)
        assert "load_error" not in r and r["max_diff"] == 0.0, (rank, r)
