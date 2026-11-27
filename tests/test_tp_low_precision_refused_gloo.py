# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""apply_tensor_parallel on a model whose Linear layers train in fp8 (torchao Float8Linear from
float8_training, or fp8 LoRA frozen bases from enable_fp8_lora), gloo world of 2 on the CPU. Tensor parallel
runs every projection through its own local-shard GEMM, so the fp8 path would never run (the model would
silently train in 16 bit) and per-shard scales would differ from one process: it is refused before any weight
is sharded, and the fp8 layers are left as they were."""

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

pytest.importorskip("torchao.float8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_gloo import _free_port, _import_checkout_unsloth  # noqa: E402


def _model(kind):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size = 256,
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 16,
            tie_word_embeddings = False,
        )
    ).to(torch.bfloat16)
    if kind in ("float8_training", "float8_subclass"):
        import dataclasses

        from torchao.float8 import Float8LinearConfig
        from torchao.float8.float8_linear import Float8Linear

        from unsloth.models.float8_training import convert_model_to_float8_training

        config = dataclasses.replace(Float8LinearConfig.from_recipe_name("rowwise"), emulate = True)
        assert convert_model_to_float8_training(model, config, size_filter = False)
        if kind == "float8_subclass":
            # A renamed subclass still runs torchao's fp8 forward.
            renamed = type("MyFp8Linear", (Float8Linear,), {})
            for module in model.modules():
                if type(module) is Float8Linear:
                    module.__class__ = renamed
        return model
    from peft import LoraConfig, get_peft_model

    from unsloth.models.float8_training import enable_fp8_lora

    model = get_peft_model(
        model,
        LoraConfig(
            r = 8, lora_alpha = 16, target_modules = ["q_proj", "v_proj", "gate_proj", "down_proj"]
        ),
    )
    assert enable_fp8_lora(model, "fp8_lora_tensorwise", allow_cpu = True)
    return model


def _worker(rank, world, port, tmp, kind):
    import datetime

    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FP8_LORA_FUSED_OPS = "0",
    )
    torch.set_num_threads(2)
    error = _import_checkout_unsloth()
    if error is not None:
        torch.save({"skip": error}, os.path.join(tmp, f"result_{rank}.pt"))
        return
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from unsloth.models import tp_native as tpn

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 300)
    )
    try:
        model = _model(kind)
        before = {n: (tuple(p.shape), p.detach().clone()) for n, p in model.named_parameters()}
        refused = None
        try:
            tpn.apply_tensor_parallel(
                model, init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]
            )
        except ValueError as e:
            refused = str(e)
        after = dict(model.named_parameters())
        res = {
            "refused": refused,
            "dtensors": sum(tpn._is_dtensor(p) for p in model.parameters()),
            "changed": [
                n
                for n, (shape, value) in before.items()
                if n not in after
                or tuple(after[n].shape) != shape
                or tpn._is_dtensor(after[n])
                or not torch.equal(after[n].detach(), value)
            ],
            "float8_linears": sum(
                any(c.__name__ == "Float8Linear" for c in type(m).__mro__) for m in model.modules()
            ),
            "fp8_lora_bases": sum("_unsloth_fp8_lora" in m.__dict__ for m in model.modules()),
            "is_tp": tpn.is_tensor_parallel(model),
        }
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["float8_training", "float8_subclass", "fp8_lora"])
def test_tp2_low_precision_model_is_refused_before_sharding(kind):
    with tempfile.TemporaryDirectory() as tmp:
        ctx = multiprocessing.get_context("spawn")
        port = _free_port()
        procs = [ctx.Process(target = _worker, args = (r, 2, port, tmp, kind)) for r in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        assert [p.exitcode for p in procs] == [0, 0]
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False) for r in range(2)
        ]
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    for res in results:
        assert res["refused"] is not None and "fp8" in res["refused"], res
        assert res["dtensors"] == 0 and res["changed"] == [] and not res["is_tp"], res
        assert res["float8_linears"] + res["fp8_lora_bases"] > 0, res
