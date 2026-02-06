#!/usr/bin/env python3
"""
Compare Llama attention implementations (FlashAttention, PyTorch SDPA, Torch reference).
Provides numerical diff (forward/backward) against the reference, dtype stability checks,
and forward/backward performance measurements under varying input sizes.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import os

os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

# 等价于：PYTHONPATH="/workspace/unsloth:${PYTHONPATH:-}"
os.environ["PYTHONPATH"] = "/workspace/unsloth" + (
    (":" + os.environ["PYTHONPATH"]) if "PYTHONPATH" in os.environ else ""
)

os.environ["HIP_VISIBLE_DEVICES"] = "2"

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTENTION = True
except Exception:  # pragma: no cover - flash_attn is optional
    flash_attn_func = None
    HAS_FLASH_ATTENTION = False


DTYPE_MAP: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class AttentionImplementation:
    name: str
    fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool],
        torch.Tensor,
    ]
    requires_cuda: bool = False
    requires_flash: bool = False
    supported_dtypes: Optional[Sequence[torch.dtype]] = None

    def is_supported(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[bool, str]:
        if self.requires_cuda and device.type != "cuda":
            return False, "CUDA only"
        if self.requires_flash and not HAS_FLASH_ATTENTION:
            return False, "flash_attn not available"
        if self.supported_dtypes is not None and dtype not in self.supported_dtypes:
            pretty = ", ".join(
                sorted({k for k, v in DTYPE_MAP.items() if v in self.supported_dtypes})
            )
            return False, f"dtype must be one of ({pretty})"
        return True, ""


def torch_reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = True,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        causal_mask = torch.ones(
            q.size(-2), k.size(-2), device = q.device, dtype = torch.bool
        ).triu(1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
    if attn_mask is not None:
        scores = scores + attn_mask
    probs = torch.softmax(scores, dim = -1)
    return torch.matmul(probs, v)


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = True,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask = attn_mask,
        is_causal = causal,
    )


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = True,
) -> torch.Tensor:
    if attn_mask is not None:
        raise ValueError("FlashAttention path does not support explicit attn_mask")
    if flash_attn_func is None:
        raise RuntimeError("flash_attn package not available")
    q_flash = q.transpose(1, 2).contiguous()  # (bsz, seq, heads, dim)
    k_flash = k.transpose(1, 2).contiguous()
    v_flash = v.transpose(1, 2).contiguous()
    out = flash_attn_func(q_flash, k_flash, v_flash, causal = causal)
    return out.transpose(1, 2).contiguous()


REFERENCE_IMPL = AttentionImplementation(
    name = "torch_ref",
    fn = torch_reference_attention,
)
IMPLEMENTATIONS: List[AttentionImplementation] = [
    AttentionImplementation(
        name = "flash_attn",
        fn = flash_attention,
        requires_cuda = True,
        requires_flash = True,
        supported_dtypes = (torch.float16, torch.bfloat16),
    ),
    AttentionImplementation(
        name = "sdpa",
        fn = sdpa_attention,
    ),
]


class StepTimer:
    def __init__(self, device: torch.device):
        self.device = device
        if device.type == "cuda":
            self._fwd_start = torch.cuda.Event(enable_timing = True)
            self._fwd_end = torch.cuda.Event(enable_timing = True)
            self._bwd_end = torch.cuda.Event(enable_timing = True)
        else:
            self._fwd_start = 0.0
            self._fwd_end = 0.0
            self._bwd_end = 0.0

    def start_forward(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self._fwd_start.record()
        else:
            self._fwd_start = time.perf_counter()

    def end_forward(self):
        if self.device.type == "cuda":
            self._fwd_end.record()
        else:
            self._fwd_end = time.perf_counter()

    def end_backward(self):
        if self.device.type == "cuda":
            self._bwd_end.record()
            torch.cuda.synchronize(self.device)
            fwd_ms = self._fwd_start.elapsed_time(self._fwd_end)
            bwd_ms = self._fwd_end.elapsed_time(self._bwd_end)
        else:
            self._bwd_end = time.perf_counter()
            fwd_ms = (self._fwd_end - self._fwd_start) * 1000.0
            bwd_ms = (self._bwd_end - self._fwd_end) * 1000.0
        return fwd_ms, bwd_ms


def clone_inputs(
    base: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    return tuple(t.clone().detach().requires_grad_(True) for t in base)


def run_forward_backward(
    impl: AttentionImplementation,
    base_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    attn_mask: Optional[torch.Tensor],
    causal: bool,
    timer: StepTimer,
) -> Tuple[
    torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], float, float, bool
]:
    q, k, v = clone_inputs(base_inputs)
    timer.start_forward()
    out = impl.fn(q, k, v, attn_mask, causal)
    timer.end_forward()
    loss = out.sum()
    loss.backward()
    fwd_ms, bwd_ms = timer.end_backward()
    grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())
    is_finite = torch.isfinite(out).all().item() and all(
        torch.isfinite(g).all().item() for g in grads
    )
    return out.detach().clone(), grads, fwd_ms, bwd_ms, is_finite


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def summarize_impl(
    name: str,
    status: str,
    fwd_diff: float,
    bwd_diff: float,
    fwd_ms: float,
    bwd_ms: float,
    stable: bool,
):
    diff_str = f"fwd_diff={fwd_diff:.3e} bwd_diff={bwd_diff:.3e}"
    time_str = f"fwd={fwd_ms:.2f}ms bwd={bwd_ms:.2f}ms"
    print(f"  - {name:<10} | {status:<24} | {diff_str} | {time_str} | stable={stable}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Benchmark Unsloth attention kernels")
    parser.add_argument("--batch-sizes", nargs = "+", type = int, default = [1, 4])
    parser.add_argument("--seq-lens", nargs = "+", type = int, default = [128, 2048])
    parser.add_argument("--num-heads", type = int, default = 32)
    parser.add_argument("--head-dim", type = int, default = 128)
    parser.add_argument(
        "--dtypes", nargs = "+", choices = DTYPE_MAP.keys(), default = ["fp16", "bf16"]
    )
    parser.add_argument(
        "--num-iters", type = int, default = 20, help = "Timed iterations per configuration"
    )
    parser.add_argument(
        "--warmup", type = int, default = 5, help = "Warmup iterations (not timed)"
    )
    parser.add_argument(
        "--device", type = str, default = ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument(
        "--no-causal", action = "store_true", help = "Disable causal masking"
    )
    parser.add_argument(
        "--allow-tf32", action = "store_true", help = "Enable TF32 on CUDA matmul"
    )
    return parser.parse_args()


def prepare_inputs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(shape, dtype = dtype, device = device)
    k = torch.randn(shape, dtype = dtype, device = device)
    v = torch.randn(shape, dtype = dtype, device = device)
    return q, k, v


def benchmark_configuration(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    args: argparse.Namespace,
):
    print(
        f"\nConfig: batch={batch_size} seq={seq_len} heads={num_heads} dim={head_dim} dtype={dtype}"
    )
    base_inputs = prepare_inputs(
        batch_size, num_heads, seq_len, head_dim, dtype, device
    )
    attn_mask = None

    timer = StepTimer(device)
    ref_out, ref_grads, _, _, ref_stable = run_forward_backward(
        REFERENCE_IMPL,
        base_inputs,
        attn_mask,
        causal = not args.no_causal,
        timer = timer,
    )
    # Warmup runs (reference)
    for _ in range(args.warmup):
        _, _, _, _, warmup_stable = run_forward_backward(
            REFERENCE_IMPL,
            base_inputs,
            attn_mask,
            not args.no_causal,
            timer,
        )
        ref_stable = ref_stable and warmup_stable

    ref_total_fwd = 0.0
    ref_total_bwd = 0.0
    for _ in range(args.num_iters):
        _, _, fwd_ms, bwd_ms, iter_stable = run_forward_backward(
            REFERENCE_IMPL,
            base_inputs,
            attn_mask,
            not args.no_causal,
            timer,
        )
        ref_total_fwd += fwd_ms
        ref_total_bwd += bwd_ms
        ref_stable = ref_stable and iter_stable

    summarize_impl(
        REFERENCE_IMPL.name,
        status = "timed",
        fwd_diff = 0.0,
        bwd_diff = 0.0,
        fwd_ms = ref_total_fwd / args.num_iters,
        bwd_ms = ref_total_bwd / args.num_iters,
        stable = ref_stable,
    )

    timed_impls = IMPLEMENTATIONS

    for impl in timed_impls:
        supported, reason = impl.is_supported(device, dtype)
        if not supported:
            summarize_impl(
                impl.name,
                status = f"skipped ({reason})",
                fwd_diff = float("nan"),
                bwd_diff = float("nan"),
                fwd_ms = float("nan"),
                bwd_ms = float("nan"),
                stable = False,
            )
            continue

        # Warmup
        for _ in range(args.warmup):
            run_forward_backward(
                impl, base_inputs, attn_mask, not args.no_causal, timer
            )

        total_fwd = 0.0
        total_bwd = 0.0
        diff_fwd = 0.0
        diff_bwd = 0.0
        stable = True

        for iter_idx in range(args.num_iters):
            out, grads, fwd_ms, bwd_ms, is_finite = run_forward_backward(
                impl,
                base_inputs,
                attn_mask,
                not args.no_causal,
                timer,
            )
            total_fwd += fwd_ms
            total_bwd += bwd_ms
            if iter_idx == 0:
                diff_fwd = tensor_diff(out, ref_out)
                grad_diffs = [tensor_diff(g, rg) for g, rg in zip(grads, ref_grads)]
                diff_bwd = max(grad_diffs)
            stable = stable and is_finite

        summarize_impl(
            impl.name,
            status = "timed",
            fwd_diff = diff_fwd,
            bwd_diff = diff_bwd,
            fwd_ms = total_fwd / args.num_iters,
            bwd_ms = total_bwd / args.num_iters,
            stable = stable,
        )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    for dtype_str in args.dtypes:
        dtype = DTYPE_MAP[dtype_str]
        for batch in args.batch_sizes:
            for seq in args.seq_lens:
                benchmark_configuration(
                    batch,
                    seq,
                    args.num_heads,
                    args.head_dim,
                    dtype,
                    device,
                    args,
                )


if __name__ == "__main__":
    main()
