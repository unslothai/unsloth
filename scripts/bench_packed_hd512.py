# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Compare packed HD512 Triton with efficient varlen on identical Q/K/V and documents.

Run in a CUDA Unsloth environment with this checkout installed:
    python scripts/bench_packed_hd512.py --lengths 1024 2048 512 4096 --heads 32 --kv-heads 4
Reports warm forward/backward device time and incremental peak allocated memory
for one attention call.
"""

import argparse
import json
import os
import statistics
import types

import torch

from unsloth.utils import hf_packed_attention as hpa


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--lengths", type = int, nargs = "+", default = [1024, 2048, 512, 4096])
    parser.add_argument("--heads", type = int, default = 32)
    parser.add_argument("--kv-heads", type = int, default = 4)
    parser.add_argument("--scale", type = float, default = 1.0)
    parser.add_argument("--warmup", type = int, default = 5)
    parser.add_argument("--repeats", type = int, default = 20)
    args = parser.parse_args()
    if (
        not args.lengths
        or min(args.lengths) <= 0
        or args.heads <= 0
        or args.kv_heads <= 0
        or args.heads % args.kv_heads
        or args.warmup < 1
        or args.repeats < 1
    ):
        parser.error(
            "positive lengths, head counts, warmup and repeats are required; KV heads must divide heads"
        )
    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("requires an NVIDIA CUDA GPU")
    lengths = torch.tensor(args.lengths, dtype = torch.int32)
    T = sum(args.lengths)
    torch.manual_seed(0)
    q = torch.randn(1, args.heads, T, 512, device = "cuda", dtype = torch.bfloat16).requires_grad_()
    k = torch.randn(1, args.kv_heads, T, 512, device = "cuda", dtype = torch.bfloat16).requires_grad_()
    v = torch.randn_like(k).requires_grad_()
    grad = torch.randn(1, T, args.heads, 512, device = "cuda", dtype = q.dtype)
    doc = torch.repeat_interleave(torch.arange(len(args.lengths)), lengths.long()).cuda()
    # Reuse the same block-causal mask across routes and iterations.
    mask = ((doc[:, None] == doc[None, :]).tril())[None, None]
    module = types.SimpleNamespace(is_causal = True, training = True)

    def unexpected_sdpa(*args, **kwargs):
        raise RuntimeError("benchmark reached wrapped SDPA instead of the requested packed route")

    hpa._ORIG_SDPA[0] = unexpected_sdpa

    def run():
        for tensor in (q, k, v):
            tensor.grad = None
        out, _ = hpa._sdpa_packed_varlen(
            module, q, k, v, mask, scaling = args.scale, packed_seq_lengths = lengths
        )
        out.backward(grad)

    results = {}
    # Alternate routes to reduce order bias.
    for arm in ("efficient", "large_head", "large_head", "efficient"):
        os.environ["UNSLOTH_HF_PACKED_HD512"] = "1" if arm == "large_head" else "0"
        before = hpa.HF_PACKED_ATTENTION_STATS[arm]
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize()
        if hpa.HF_PACKED_ATTENTION_STATS[arm] - before != args.warmup:
            raise RuntimeError(
                f"{arm} did not engage on every warmup; check dtype, policy, and self-check"
            )
        for tensor in (q, k, v):
            tensor.grad = None
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        samples = []
        before = hpa.HF_PACKED_ATTENTION_STATS[arm]
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            start.record()
            run()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        if hpa.HF_PACKED_ATTENTION_STATS[arm] - before != args.repeats:
            raise RuntimeError(f"{arm} fell back during measurement; discard these timings")
        peak = (torch.cuda.max_memory_allocated() - allocated) / 2**20
        results.setdefault(arm, []).append(
            {"median_ms": statistics.median(samples), "extra_peak_MiB": peak}
        )
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "lengths": args.lengths,
                "heads": args.heads,
                "kv_heads": args.kv_heads,
                "scale": args.scale,
                "dtype": str(q.dtype),
                "results": results,
            },
            indent = 2,
        )
    )


if __name__ == "__main__":
    main()
