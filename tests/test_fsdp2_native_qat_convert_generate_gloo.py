# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 (gloo CPU world of 2): `generate` after torchao's QATConfig(step = "convert").

The QAT notebooks train a LoRA model with FakeQuantizedLinear layers, then call
`quantize_(model, QATConfig(step = "convert"))`, which swaps each FakeQuantizedLinear for a new
nn.Linear holding the same (sharded) Parameter, then `model.generate(...)`. FSDP2's FSDPParam still
points at the dropped modules, so the gather wrote the full weights there and the live nn.Linear
kept its DTensor shard: "aten.mm.default: got mixed torch.Tensor and DTensor". The unsharded
context now rebinds FSDP2 to the swapped-in modules; greedy tokens must equal one process."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402

REPO_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
PROMPT = [[11, 523, 9, 77, 1024, 5]]
NEW_TOKENS = 8


def _qat_lora_model():
    import transformers
    from peft import LoraConfig, get_peft_model
    from torchao.quantization import quantize_
    from torchao.quantization.qat import IntxFakeQuantizeConfig, QATConfig

    config = transformers.AutoConfig.from_pretrained(REPO_ID)
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    model.generation_config.pad_token_id = config.eos_token_id
    cfg = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.normal_(0, 0.05, generator = g)
    # Unsloth's _prepare_model_for_qat skips the LoRA adapters; so does this.
    weight_config = IntxFakeQuantizeConfig(torch.int4, group_size = 8)
    quantize_(
        model,
        QATConfig(weight_config = weight_config, step = "prepare"),
        filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear) and "lora_" not in fqn,
    )
    return model


def _convert(model):
    from torchao.quantization import quantize_
    from torchao.quantization.qat import FakeQuantizedLinear, QATConfig

    assert any(isinstance(m, FakeQuantizedLinear) for m in model.modules())
    quantize_(model, QATConfig(step = "convert"))
    assert not any(isinstance(m, FakeQuantizedLinear) for m in model.modules())


def _greedy(model):
    ids = torch.tensor(PROMPT)
    out = model.generate(
        input_ids = ids,
        attention_mask = torch.ones_like(ids),
        max_new_tokens = NEW_TOKENS,
        do_sample = False,
    )
    return out[0, ids.shape[1] :].tolist()


def _reference_tokens():
    model = _qat_lora_model()
    model(input_ids = torch.tensor(PROMPT), labels = torch.tensor(PROMPT)).loss.backward()
    _convert(model)
    model.eval()
    return _greedy(model)


def _case(native):
    from torch.distributed.tensor import DTensor

    model = native.shard_model(_qat_lora_model(), native.FSDP2Config(param_dtype = torch.float32))
    model(input_ids = torch.tensor(PROMPT), labels = torch.tensor(PROMPT)).loss.backward()
    _convert(model)
    model.eval()
    res = {}
    try:
        res["tokens"] = _greedy(model)
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    # Resharded on exit: every swapped-in Linear holds its DTensor shard again.
    res["weights_sharded_after"] = all(
        isinstance(m.weight, DTensor)
        for name, m in model.named_modules()
        if type(m) is torch.nn.Linear and "lora_" not in name
    )
    return res


def _worker(rank, world, port, out):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 60)
    )
    try:
        out.put((rank, _case(_load_native())))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(timeout = 240):
    import torch.multiprocessing as mp

    out = mp.get_context("spawn").Queue()
    procs = mp.start_processes(
        _worker,
        args = (2, _free_port(), out),
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
                pytest.fail(f"the ranks did not finish within {timeout} s (hung collective)")
    except mp.ProcessExitedException as exc:
        pytest.fail(f"a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("QAT_CONVERT_GENERATE", rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_generate_after_qat_convert_matches_one_process():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("torchao.quantization.qat")
    want = _reference_tokens()
    assert len(want) == NEW_TOKENS
    for rank, r in _run().items():
        assert "error" not in r, (rank, r)
        assert r["tokens"] == want, (rank, r, want)
        assert r["weights_sharded_after"], (rank, r)
