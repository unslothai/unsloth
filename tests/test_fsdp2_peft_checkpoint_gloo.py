# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP2 + PEFT FULL_STATE_DICT checkpoints (what Trainer writes for resuming), two gloo ranks on CPU.

accelerate < 1.15 builds the adapter-only full state dict with `get_peft_model_state_dict`, i.e.
each rank's DTensor shard, and rank 0 writes only its own; `install_fsdp2_peft_full_state_dict`
backports the 1.15 gather. The saved adapter must hold whole tensors equal to the unsharded ones,
and loading it back must restore every rank's shard.
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


def _build():
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 96,
        tie_word_embeddings = False,
    )
    torch.manual_seed(0)
    lora = LoraConfig(
        r = 4, lora_alpha = 8, target_modules = ["q_proj", "down_proj"], init_lora_weights = False
    )
    return get_peft_model(LlamaForCausalLM(cfg), lora)


def _accelerator_less_import():
    # Spawned ranks skip conftest, whose CUDA spoof lets `import unsloth` run on CPU-only runners.
    if torch.cuda.is_available():
        return
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("_unsloth_tests_conftest", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def _worker(rank, port, out_dir, backport):
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

        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor

        from accelerate import PartialState
        from accelerate.utils import FullyShardedDataParallelPlugin
        from accelerate.utils import fsdp_utils
        from unsloth.models import fsdp2

        PartialState(cpu = True)  # accelerate's logger refuses to run without it
        if backport:
            assert fsdp2.install_fsdp2_peft_full_state_dict()
        else:
            # `import unsloth` installed it already; undo that to reproduce the unpatched accelerate.
            for name in ("_get_model_state_dict", "_set_model_state_dict"):
                fn = getattr(fsdp_utils, name)
                setattr(fsdp_utils, name, getattr(fn, "__wrapped__", fn))
        result = {"accelerate": __import__("accelerate").__version__}
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names = ("dp_shard",))
        plugin = FullyShardedDataParallelPlugin(
            fsdp_version = 2,
            auto_wrap_policy = "TRANSFORMER_BASED_WRAP",
            transformer_cls_names_to_wrap = ["LlamaDecoderLayer"],
            state_dict_type = "FULL_STATE_DICT",
        )
        accelerator = types.SimpleNamespace(
            state = types.SimpleNamespace(fsdp_plugin = plugin),
            torch_device_mesh = mesh,
            parallelism_config = types.SimpleNamespace(fsdp_dim_names = ("dp_shard",)),
            mixed_precision = "no",
            device = torch.device("cpu"),
            num_processes = 2,
            process_index = rank,
            is_main_process = rank == 0,
            is_fsdp2 = True,
            wait_for_everyone = dist.barrier,
        )
        model = _build()
        reference = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
        model = fsdp_utils.fsdp2_prepare_model(accelerator, model)
        fsdp_utils.save_fsdp_model(plugin, accelerator, model, out_dir, adapter_only = True)
        dist.barrier()

        if rank == 0:
            saved = torch.load(os.path.join(out_dir, "pytorch_model_fsdp.bin"), weights_only = False)
            # peft's keys drop the adapter name, get_model_state_dict's keep it.
            saved = {k.replace(".default", ""): v for k, v in saved.items() if "lora_" in k}
            whole = 0
            for name, ref in reference.items():
                got = saved.get(name.replace(".default", ""))
                if isinstance(got, DTensor):
                    got = got.to_local()
                if (
                    got is not None
                    and tuple(got.shape) == tuple(ref.shape)
                    and torch.equal(got.cpu(), ref)
                ):
                    whole += 1
            result["lora_tensors"] = len(reference)
            result["saved_whole_and_equal"] = whole

        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_" in name:
                    p.zero_()
        fsdp_utils.load_fsdp_model(plugin, accelerator, model, out_dir, adapter_only = True)
        restored = 0
        for name, p in model.named_parameters():
            if "lora_" not in name:
                continue
            full = p.full_tensor() if isinstance(p, DTensor) else p
            restored += int(torch.equal(full.detach().cpu(), reference[name]))
        result["restored"] = restored
        if rank == 0:
            with open(os.path.join(out_dir, "result.json"), "w") as f:
                json.dump(result, f)
    finally:
        dist.destroy_process_group()


def _run(tmp_path, backport):
    import json

    import torch.multiprocessing as mp

    mp.spawn(_worker, args = (_free_port(), str(tmp_path), backport), nprocs = 2, join = True)
    with open(tmp_path / "result.json") as f:
        return json.load(f)


def test_full_state_dict_adapter_checkpoints_hold_every_rank_shard(tmp_path):
    r = _run(tmp_path, backport = True)
    assert r["lora_tensors"] == 8
    assert (
        r["saved_whole_and_equal"] == r["lora_tensors"]
    ), "the checkpoint holds only rank 0's shards"
    assert r["restored"] == r["lora_tensors"]
