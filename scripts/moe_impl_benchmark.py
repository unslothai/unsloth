#!/usr/bin/env python3
"""
Benchmark Qwen3-MoE SparseMoeBlock vs 纯 PyTorch 参考实现（Transformers 4.56.2 旧版接口）。

- 目标模块：transformers.models.qwen3_moe.modeling_qwen3_moe:Qwen3MoeSparseMoeBlock
- 参考模块：脚本内的 ReferenceQwen3MoeSparseMoeBlock，逻辑与 HF 4.56.2 一致
- 对比：前向输出、梯度、router logits 的 max_abs_diff；同时计时 fwd/bwd
"""

import argparse
import importlib
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

# 等价于：PYTHONPATH="/workspace/unsloth:${PYTHONPATH:-}"
os.environ["PYTHONPATH"] = "/workspace/unsloth" + (
    (":" + os.environ["PYTHONPATH"]) if "PYTHONPATH" in os.environ else ""
)
os.environ["HIP_VISIBLE_DEVICES"] = "2"

try:
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.activations import ACT2FN
except Exception as exc:
    raise SystemExit("需要 transformers>=4.56.2，请先安装。") from exc

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


class StepTimer:
    def __init__(self, device: torch.device):
        self.device = device
        if device.type == "cuda":
            self._fwd_start = torch.cuda.Event(enable_timing=True)
            self._fwd_end = torch.cuda.Event(enable_timing=True)
            self._bwd_end = torch.cuda.Event(enable_timing=True)
        else:
            self._fwd_start = self._fwd_end = self._bwd_end = 0.0

    def start_forward(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self._fwd_start.record()
        else:
            import time
            self._fwd_start = time.perf_counter()

    def end_forward(self):
        if self.device.type == "cuda":
            self._fwd_end.record()
        else:
            import time
            self._fwd_end = time.perf_counter()

    def end_backward(self):
        if self.device.type == "cuda":
            self._bwd_end.record()
            torch.cuda.synchronize(self.device)
            fwd_ms = self._fwd_start.elapsed_time(self._fwd_end)
            bwd_ms = self._fwd_end.elapsed_time(self._bwd_end)
        else:
            import time
            self._bwd_end = time.perf_counter()
            fwd_ms = (self._fwd_end - self._fwd_start) * 1e3
            bwd_ms = (self._bwd_end - self._fwd_end) * 1e3
        return fwd_ms, bwd_ms


class ReferenceQwen3MoeSparseMoeBlock(nn.Module):
    """参考实现，等价于 4.56.2 的 Qwen3MoeSparseMoeBlock."""
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [self._make_mlp(config) for _ in range(self.num_experts)]
        )

    @staticmethod
    def _make_mlp(config):
        hidden = config.hidden_size
        inter = config.moe_intermediate_size
        mlp = nn.Module()
        mlp.gate_proj = nn.Linear(hidden, inter, bias=False)
        mlp.up_proj = nn.Linear(hidden, inter, bias=False)
        mlp.down_proj = nn.Linear(inter, hidden, bias=False)
        mlp.act_fn = ACT2FN[config.hidden_act]
        def forward(x):
            return mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))
        mlp.forward = forward
        return mlp

    def copy_from(self, target: nn.Module):
        with torch.no_grad():
            self.gate.weight.copy_(target.gate.weight)
            for s, t in zip(self.experts, target.experts):
                s.gate_proj.weight.copy_(t.gate_proj.weight)
                s.up_proj.weight.copy_(t.up_proj.weight)
                s.down_proj.weight.copy_(t.down_proj.weight)

    def forward(self, hidden_states: torch.Tensor):
        bsz, seqlen, hidden = hidden_states.shape
        flat = hidden_states.view(-1, hidden)
        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)
        final = torch.zeros_like(flat)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = flat[None, top_x].reshape(-1, hidden)
            current_hidden = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final.index_add_(0, top_x, current_hidden.to(flat.dtype))
        return final.view(bsz, seqlen, hidden), router_logits


def import_from_path(path: str):
    if ":" not in path:
        raise ValueError(f"Expected 'module:ClassName', got '{path}'")
    module_name, class_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def instantiate_target(class_path, config, device, dtype, train):
    cls = import_from_path(class_path)
    try:
        inst = cls(config)
    except TypeError:
        inst = cls(config=config)
    inst = inst.to(device=device, dtype=dtype)
    inst.train(train)
    return inst


def tensor_max_diff(a, b):
    return (a - b).abs().max().item()


def run_once(module, inputs, timer=None):
    module.zero_grad(set_to_none=True)
    x = inputs.clone().detach().requires_grad_(True)
    if timer:
        timer.start_forward()
    out = module(x)
    if timer:
        timer.end_forward()
    if isinstance(out, tuple):
        hidden, router_logits = out
    else:
        hidden, router_logits = out, None
    loss = hidden.sum()
    loss.backward()
    if timer:
        fwd_ms, bwd_ms = timer.end_backward()
    else:
        fwd_ms = bwd_ms = 0.0
    grad = x.grad.detach().clone()
    logits = router_logits.detach().clone() if router_logits is not None else None
    stable = torch.isfinite(hidden).all().item() and torch.isfinite(grad).all().item()
    return hidden.detach().clone(), grad, logits, fwd_ms, bwd_ms, stable


def average_timing(module, base_input, n_iters, timer):
    total_fwd = total_bwd = 0.0
    stable = True
    for _ in range(n_iters):
        out, _, _, fwd_ms, bwd_ms, ok = run_once(module, base_input, timer)
        total_fwd += fwd_ms
        total_bwd += bwd_ms
        stable = stable and ok and torch.isfinite(out).all().item()
    return total_fwd / max(1, n_iters), total_bwd / max(1, n_iters), stable


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Qwen3-MoE SparseMoeBlock (TF 4.56.2)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512])
    p.add_argument("--dtypes", choices=DTYPE_MAP.keys(), nargs="+", default=["fp16", "bf16"])
    p.add_argument("--device", type=str, default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--num-iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--moe-intermediate-size", type=int, default=8192)
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--norm-topk-prob", action="store_true")
    p.add_argument("--hidden-act", type=str, default="silu")
    p.add_argument(
        "--target-class",
        type=str,
        default="transformers.models.qwen3_moe.modeling_qwen3_moe:Qwen3MoeSparseMoeBlock",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False  # 可按需改 True

    config = Qwen3MoeConfig(
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        num_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        norm_topk_prob=args.norm_topk_prob,
        hidden_act=args.hidden_act,
    )

    for dtype_name in args.dtypes:
        dtype = DTYPE_MAP[dtype_name]
        print(f"\n=== dtype={dtype_name} ({dtype}) ===")
        try:
            target = instantiate_target(args.target_class, config, device, dtype, train=False)
        except Exception as exc:
            raise SystemExit(f"实例化目标模块失败: {exc}") from exc

        ref = ReferenceQwen3MoeSparseMoeBlock(config).to(device=device, dtype=dtype)
        ref.eval()
        ref.copy_from(target)

        for batch in args.batch_sizes:
            for seq in args.seq_lens:
                print(f"\nConfig: batch={batch}, seq={seq}, hidden={config.hidden_size}, experts={config.num_experts}")
                inputs = torch.randn(batch, seq, config.hidden_size, dtype=dtype, device=device)

                tgt_out, tgt_grad, tgt_logits, _, _, tgt_stable = run_once(target, inputs)
                ref_out, ref_grad, ref_logits, _, _, ref_stable = run_once(ref, inputs)

                fwd_diff = tensor_max_diff(tgt_out, ref_out)
                bwd_diff = tensor_max_diff(tgt_grad, ref_grad)
                router_diff = tensor_max_diff(tgt_logits, ref_logits) if tgt_logits is not None else 0.0
                print(
                    f"Parity\t fwd_diff={fwd_diff:.3e} | bwd_diff={bwd_diff:.3e} | router_diff={router_diff:.3e} | stable={tgt_stable and ref_stable}"
                )

                timer_tgt = StepTimer(device)
                timer_ref = StepTimer(device)
                for _ in range(args.warmup):
                    run_once(target, inputs, timer_tgt)
                    run_once(ref, inputs, timer_ref)

                tgt_fwd, tgt_bwd, tgt_stable2 = average_timing(target, inputs, args.num_iters, timer_tgt)
                ref_fwd, ref_bwd, ref_stable2 = average_timing(ref, inputs, args.num_iters, timer_ref)
                print(
                    "Perf\t target: fwd={:.2f}ms bwd={:.2f}ms | ref: fwd={:.2f}ms bwd={:.2f}ms | stable={}".format(
                        tgt_fwd, tgt_bwd, ref_fwd, ref_bwd, tgt_stable2 and ref_stable2
                    )
                )


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        raise SystemExit("Python >= 3.9 is required")
    main()
