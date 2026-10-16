# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Trainer.save_model of a LoRA model under FSDP2 + context parallelism, two gloo ranks on CPU.

Unsloth installs a ParallelismConfig for the (dp_shard, cp) mesh. transformers < 5 dispatches
save_model on that config before FSDP, skipping the full state dict gather, so PEFT is handed DTensor
shards and fails. The patched save_model must write whole adapter tensors equal to the unsharded ones,
and leave the installed config in place afterwards.
"""

from __future__ import annotations

import os
import socket
import types

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
pytest.importorskip("torch.distributed.fsdp")
pytest.importorskip("peft")
pytest.importorskip("accelerate")

if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("gloo backend unavailable", allow_module_level = True)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _accelerator_less_import():
    # Spawned ranks skip conftest, whose CUDA spoof lets `import unsloth` run on CPU-only runners.
    if torch.cuda.is_available():
        return
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("_unsloth_tests_conftest", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def _worker(rank, port, out_dir):
    _accelerator_less_import()
    os.environ.update(
        RANK = str(rank),
        LOCAL_RANK = str(rank),
        WORLD_SIZE = "2",
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
    )
    dist.init_process_group("gloo", rank = rank, world_size = 2)
    try:
        import json

        import transformers
        import trl
        from peft import LoraConfig, get_peft_model
        from torch.distributed.fsdp import fully_shard
        from transformers import LlamaConfig, LlamaForCausalLM

        from accelerate import Accelerator
        from accelerate.utils import DistributedType
        from unsloth import context_parallel as cp

        cp.DEVICE_TYPE_TORCH = "cpu"
        manager = cp.ContextParallelManager(2, "ulysses", fsdp2 = True)
        state = types.SimpleNamespace(
            device_mesh = manager.device_mesh,
            parallelism_config = None,
            fsdp_plugin = types.SimpleNamespace(state_dict_type = "FULL_STATE_DICT"),
        )
        cp._install_fsdp2_parallelism_config(state, manager)
        installed = state.parallelism_config

        torch.manual_seed(0)
        cfg = LlamaConfig(
            vocab_size = 32,
            hidden_size = 16,
            intermediate_size = 32,
            num_attention_heads = 2,
            num_key_value_heads = 2,
            num_hidden_layers = 1,
        )
        lora = LoraConfig(r = 2, target_modules = ["q_proj", "v_proj"], init_lora_weights = False)
        model = get_peft_model(LlamaForCausalLM(cfg), lora)
        reference = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
        mesh = manager.device_mesh["dp_shard_cp"]
        for layer in model.base_model.model.model.layers:
            fully_shard(layer, mesh = mesh)
        fully_shard(model, mesh = mesh)

        class SFTTrainer(transformers.Trainer):
            pass

        trl.SFTTrainer = SFTTrainer
        cp.patch_sft_trainer()

        class Acc:
            parallelism_config = property(lambda self: self.state.parallelism_config)
            distributed_type = DistributedType.FSDP
            is_fsdp2 = True
            should_save_model = Accelerator.should_save_model
            get_state_dict = Accelerator.get_state_dict

            def unwrap_model(self, model, **kwargs):
                return model

        trainer = object.__new__(SFTTrainer)
        trainer.model = trainer.model_wrapped = model
        trainer.is_fsdp_enabled = True
        trainer.is_deepspeed_enabled = False
        trainer.processing_class = trainer.data_collator = None
        trainer.args = types.SimpleNamespace(
            output_dir = out_dir, should_save = rank == 0, push_to_hub = False, save_safetensors = True
        )
        trainer.accelerator = Acc()
        trainer.accelerator.state = state
        trainer._context_parallel_manager = manager

        result = {}
        try:
            trainer.save_model(out_dir)
            result["error"] = None
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
        result["config_kept"] = state.parallelism_config is installed
        dist.barrier()
        if rank == 0 and result["error"] is None:
            from safetensors.torch import load_file

            saved = load_file(os.path.join(out_dir, "adapter_model.safetensors"))
            equal = 0
            for name, ref in reference.items():
                got = saved.get(name.replace(".default", ""))
                equal += int(got is not None and torch.equal(got, ref))
            result["lora_tensors"] = len(reference)
            result["saved_whole_and_equal"] = equal
        gathered = [None] * 2
        dist.all_gather_object(gathered, result)
        if rank == 0:
            with open(os.path.join(out_dir, "result.json"), "w") as f:
                json.dump(gathered, f)
    finally:
        dist.destroy_process_group()


def test_save_model_gathers_fsdp2_lora_under_context_parallel(tmp_path):
    import json

    import torch.multiprocessing as mp

    mp.spawn(_worker, args = (_free_port(), str(tmp_path)), nprocs = 2, join = True)
    with open(tmp_path / "result.json") as f:
        results = json.load(f)
    for r in results:
        assert r["error"] is None, r["error"]
        assert r["config_kept"]
    assert results[0]["lora_tensors"] == 4
    assert results[0]["saved_whole_and_equal"] == 4
