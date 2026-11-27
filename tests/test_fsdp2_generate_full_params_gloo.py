# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""accelerate FSDP2 + Unsloth generate(): the fast decode reads weights directly (inference RMSNorm,
fast linear), outside FSDP's per-module hooks, so under accelerate's FSDP2 the final norm reached
`fast_rms_layernorm_inference` as a DTensor shard ("aten.mul_.Tensor got mixed torch.Tensor and
DTensor") and GRPO without vLLM crashed at its first rollout.

`fsdp2_full_params_for_generate` gathers every unit of the prepared root for one generate() and keeps
them gathered across block forwards (whose post-forward would reshard them before the next decode
step reads them), then reshards. Two gloo ranks on CPU, a real fully_shard tree."""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
    pytest.skip("gloo backend unavailable", allow_module_level = True)

_FSDP2 = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "fsdp2.py"

_WORKER = textwrap.dedent(
    r"""
    import datetime, importlib.util, json, sys
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    rank, out, path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    dist.init_process_group("gloo", rank = rank, world_size = 2, timeout = datetime.timedelta(seconds = 60))
    spec = importlib.util.spec_from_file_location("_unsloth_fsdp2_under_test", path)
    f2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f2)
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:
        from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.device_mesh import init_device_mesh

    H = 8
    class Inner(torch.nn.Module):  # the CausalLM generate() runs on (PeftModel's base model)
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.layers = torch.nn.ModuleList(torch.nn.Linear(H, H) for _ in range(2))
            self.norm = torch.nn.Module()
            self.norm.weight = torch.nn.Parameter(torch.rand(H) + 0.5)
    class Root(torch.nn.Module):  # accelerate prepares (and fully_shards) the wrapper
        def __init__(self):
            super().__init__()
            self.base = Inner()
        def forward(self, x):
            for layer in self.base.layers:
                x = layer(x)
            return x * self.base.norm.weight

    mesh = init_device_mesh("cpu", (2,))
    root = Root()
    inner = root.base
    reference = {n: p.detach().clone() for n, p in inner.named_parameters()}
    for layer in inner.layers:
        fully_shard(layer, mesh = mesh)  # reshard_after_forward = True
    fully_shard(root, mesh = mesh)
    if hasattr(f2, "_FSDP2_ROOTS"):
        f2._FSDP2_ROOTS.add(root)  # what after_fsdp2_wrap records

    def decode(x):
        # Unsloth's prefill calls the blocks as modules (FSDP hooks fire); the decode steps and the final
        # norm read the weights directly.
        for layer in inner.layers:
            x = layer(x)
        for layer in inner.layers:
            x = torch.nn.functional.linear(x, layer.weight, layer.bias)
        return x * inner.norm.weight

    x = torch.ones(2, H)
    result = {"rank": rank}
    with torch.no_grad():
        gather = getattr(f2, "fsdp2_full_params_for_generate", None)
        if gather is None:
            result["with"] = "missing fsdp2_full_params_for_generate"
        else:
            with gather(inner) as gathered:
                result["gathered"] = gathered
                y = decode(x)
                result["inside_dtensor"] = any(isinstance(p, DTensor) for p in inner.parameters())
            expected = x
            for _ in range(2):  # the module pass, then the direct-read pass
                for i in range(2):
                    expected = torch.nn.functional.linear(
                        expected, reference[f"layers.{i}.weight"], reference[f"layers.{i}.bias"]
                    )
            result["with"] = "ok"
            result["max_err"] = (y - expected * reference["norm.weight"]).abs().max().item()
            result["after_dtensor"] = all(isinstance(p, DTensor) for p in inner.parameters())
            with gather(torch.nn.Linear(2, 2)) as other:
                result["unrelated"] = other
    # GRPO generates before any training forward: the root must still be the root afterwards.
    if gather is not None:
        try:
            root(x).sum().backward()
            result["root_forward"] = "ok"
        except Exception as e:
            result["root_forward"] = f"{type(e).__name__}: {str(e)[:160]}"
    # A fresh tree without the gather: the direct reads see shards (the reported crash).
    fresh = Root()
    for layer in fresh.base.layers:
        fully_shard(layer, mesh = mesh)
    fully_shard(fresh, mesh = mesh)
    inner = fresh.base
    with torch.no_grad():
        try:
            decode(x)
            result["without"] = "ok"
        except Exception as e:
            result["without"] = f"{type(e).__name__}: {str(e)[:120]}"
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(result, fh)
    dist.destroy_process_group()
    """
)


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def test_generate_gathers_every_fsdp2_unit_and_reshards_after(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    env = dict(
        os.environ, MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(_free_port()), CUDA_VISIBLE_DEVICES = ""
    )
    out = tmp_path / "r"
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(rank), str(out), str(_FSDP2)],
            env = env,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
        )
        for rank in range(2)
    ]
    logs = [p.communicate(timeout = 300)[0] for p in procs]
    for rank in range(2):
        path = Path(f"{out}.{rank}")
        assert path.exists(), f"rank {rank} wrote nothing:\n{logs[rank][-4000:]}"
        r = __import__("json").loads(path.read_text())
        # Without the gather the direct reads see shards: the reported crash.
        assert r["without"] != "ok", r
        assert r["with"] == "ok", r
        assert r["gathered"] is True and r["inside_dtensor"] is False, r
        assert r["max_err"] < 1e-5, r
        # Resharded on exit; a module outside any prepared root is left alone.
        assert r["after_dtensor"] is True and r["unrelated"] is False, r
        assert r["root_forward"] == "ok", r
