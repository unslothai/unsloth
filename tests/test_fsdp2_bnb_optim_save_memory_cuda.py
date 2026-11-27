# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GPU memory of the native FSDP2 optimizer.pt export with a local-shard bitsandbytes optimizer:
the peak stays about one parameter, tensors equal torch's export, and a resume restores the 8-bit
state bitwise. `python tests/test_fsdp2_bnb_optim_save_memory_cuda.py --out result.json [--arm torch]`
runs a worker."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
N_PARAMS = 16
NUMEL = 1 << 20

if __name__ != "__main__":
    if not torch.cuda.is_available() or not torch.distributed.is_nccl_available():
        pytest.skip("needs a CUDA GPU with NCCL", allow_module_level = True)
    pytest.importorskip("bitsandbytes")


def _worker(out, arm, rdzv):
    sys.path.insert(0, str(_HERE))
    from test_fsdp2_native_units import _load_native

    native = _load_native()
    import bitsandbytes as bnb
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method = f"file://{rdzv}", rank = 0, world_size = 1)
    mesh = init_device_mesh("cuda", (1,))

    def build(seed):
        model = torch.nn.Module()
        g = torch.Generator(device = "cuda").manual_seed(seed)
        for i in range(N_PARAMS):
            full = torch.randn(NUMEL // 256, 256, device = "cuda", generator = g)
            model.register_parameter(
                f"w{i}", torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
            )
        setattr(model, native._NATIVE_FLAG, True)
        if arm == "torch":
            opt = torch.optim.AdamW(model.parameters(), lr = 1e-3, fused = True)
        else:
            opt = bnb.optim.AdamW8bit(model.parameters(), lr = 1e-3)
            from unsloth.models.fsdp2_bnb_optim import localize_bnb_optimizer
            assert localize_bnb_optimizer(opt)
        return model, opt

    model, opt = build(0)
    for step in range(2):
        g = torch.Generator(device = "cuda").manual_seed(10 + step)
        for p in model.parameters():
            p.grad = distribute_tensor(
                torch.randn(p.shape, device = "cuda", generator = g), mesh, [Shard(0)]
            )
        opt.step()
    opt.zero_grad(set_to_none = True)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    state = native.native_save_optimizer(model, opt)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    retained = torch.cuda.memory_allocated() - before
    on_cpu = all(
        v.device.type == "cpu"
        for s in state["state"].values()
        for v in s.values()
        if torch.is_tensor(v)
    )
    ref = get_optimizer_state_dict(
        model, opt, options = StateDictOptions(full_state_dict = True, cpu_offload = True)
    )
    equal = state["state"].keys() == ref["state"].keys() and all(
        state["state"][k].keys() == ref["state"][k].keys()
        and all(
            torch.equal(v, ref["state"][k][key])
            if torch.is_tensor(v)
            else v == ref["state"][k][key]
            for key, v in state["state"][k].items()
        )
        for k in state["state"]
    )
    resumed = None
    if arm != "torch":
        model2, opt2 = build(1)
        native.native_load_optimizer(model2, opt2, state)
        raw = {n: opt.state[p] for n, p in model.named_parameters()}
        raw2 = {n: opt2.state[p] for n, p in model2.named_parameters()}
        resumed = all(
            all(
                torch.equal(raw[n][key], raw2[n][key])
                for key in ("state1", "state2", "absmax1", "absmax2")
            )
            for n in raw
        )
    result = {
        "arm": arm,
        "elements": N_PARAMS * NUMEL,
        "one_param_fp32_moments_bytes": 2 * 4 * NUMEL,
        "train_allocated_bytes": before,
        "save_extra_peak_bytes": peak,
        "save_extra_retained_bytes": retained,
        "state_on_cpu": on_cpu,
        "equal_to_torch_export": bool(equal),
        "resume_bitwise": resumed,
    }
    Path(out).write_text(json.dumps(result))
    dist.destroy_process_group()


def _run(tmp_path, arm = "bnb"):
    out = tmp_path / f"{arm}.json"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--out",
            str(out),
            "--arm",
            arm,
            "--rdzv",
            str(tmp_path / "rdzv"),
        ],
        check = True,
        env = env,
        timeout = 600,
    )
    return json.loads(out.read_text())


def test_local_shard_bnb_save_keeps_one_parameter_on_the_gpu(tmp_path):
    r = _run(tmp_path)
    assert r["state_on_cpu"] and r["equal_to_torch_export"] and r["resume_bitwise"], r
    # Unbounded export peaked at 144 MiB here; one parameter with scratch is about 32 MiB.
    assert r["save_extra_peak_bytes"] <= 5 * r["one_param_fp32_moments_bytes"], r
    assert r["save_extra_retained_bytes"] <= 1 << 20, r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required = True)
    parser.add_argument("--arm", default = "bnb", choices = ["bnb", "torch"])
    parser.add_argument("--rdzv", default = None)
    args = parser.parse_args()
    _worker(args.out, args.arm, args.rdzv or (args.out + ".rdzv"))
