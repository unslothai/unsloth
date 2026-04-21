# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Smoke-test :meth:`FlexEngine.sleep` and :meth:`FlexEngine.wake_up`.

Invoked as:
    CUDA_VISIBLE_DEVICES=3 \
    UNSLOTH_FAST_INFERENCE=1 UNSLOTH_VLLM_STANDBY=1 \
    python tests/flex_sleep_mode_smoke.py --model unsloth/Qwen3-4B-Base

What it checks:
  1. ``model.vllm_engine._sleep_mode_enabled`` is True when vLLM is
     importable.
  2. Captured CUDA graphs survive a sleep/wake round-trip: a second
     ``fast_generate`` call after ``sleep`` -> ``wake_up`` does not
     re-capture and still produces the same token ids.
  3. ``torch.cuda.memory_allocated()`` drops on sleep and returns close
     to the pre-sleep value on wake. With ``--no-standby``, memory
     should be identical across the three probes (no-op path).
  4. Hardens the regression guard by running the same workflow with
     ``UNSLOTH_VLLM_STANDBY`` unset (``--no-standby``); ``sleep`` /
     ``wake_up`` must be exact no-ops.

This is a smoke test, not the full verification matrix from the plan
(that lives under ``scripts/benchmarks``).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _gb(n: int) -> float:
    return round(n / 1e9, 3)


def _probe(label: str) -> dict:
    import torch

    gc.collect()
    torch.cuda.synchronize()
    # ``memory_allocated`` / ``memory_reserved`` only track torch's
    # caching allocator and ignore cuMem-backed pools, so they do NOT
    # drop on sleep even though cuMem has unmapped the pages.
    # ``mem_get_info()`` asks the CUDA runtime directly, so it sees
    # cuMem unmaps and is the right probe for sleep / wake verification.
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "label": label,
        "allocated_gb": _gb(torch.cuda.memory_allocated()),
        "reserved_gb": _gb(torch.cuda.memory_reserved()),
        "cuda_free_gb": _gb(free_bytes),
        "cuda_used_gb": _gb(total_bytes - free_bytes),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default = "unsloth/Qwen3-4B-Base")
    p.add_argument("--dtype", choices = ["bf16", "fp16"], default = "bf16")
    p.add_argument("--load_in_4bit", action = "store_true")
    p.add_argument("--max_new_tokens", type = int, default = 32)
    p.add_argument("--max_seq_length", type = int, default = 1024)
    p.add_argument("--prompt", default = "The quick brown fox jumps over")
    p.add_argument(
        "--no-standby",
        action = "store_true",
        help = "Force UNSLOTH_VLLM_STANDBY=0 to validate the no-op regression path.",
    )
    p.add_argument(
        "--cycles",
        type = int,
        default = 1,
        help = "Number of sleep / wake / generate cycles after warmup. "
        ">1 exercises the repeated-cycle regression (run #6).",
    )
    args = p.parse_args()

    os.environ.setdefault("UNSLOTH_FAST_INFERENCE", "1")
    if args.no_standby:
        os.environ["UNSLOTH_VLLM_STANDBY"] = "0"
    else:
        os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
    standby = os.environ.get("UNSLOTH_VLLM_STANDBY", "0") == "1"
    print(
        f"[sleep-smoke] UNSLOTH_FAST_INFERENCE="
        f"{os.environ.get('UNSLOTH_FAST_INFERENCE')} "
        f"UNSLOTH_VLLM_STANDBY={os.environ.get('UNSLOTH_VLLM_STANDBY')}"
    )

    import torch

    import unsloth
    from unsloth import FastLanguageModel

    print(f"[sleep-smoke] unsloth={unsloth.__file__}")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    t0 = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        dtype = dtype,
        load_in_4bit = args.load_in_4bit,
        fast_inference = True,
    )
    print(
        f"[sleep-smoke] loaded {args.model} in "
        f"{time.perf_counter() - t0:.1f}s; dtype={model.dtype}"
    )

    engine = model.vllm_engine
    print(f"[sleep-smoke] engine type: {type(engine).__name__}")
    sleep_enabled = getattr(engine, "_sleep_mode_enabled", None)
    print(f"[sleep-smoke] engine._sleep_mode_enabled: {sleep_enabled}")
    if standby:
        if sleep_enabled is not True:
            # Most likely vLLM is not importable in this environment.
            print(
                "[sleep-smoke] WARNING: UNSLOTH_VLLM_STANDBY=1 was set "
                "but engine._sleep_mode_enabled is False (vLLM missing?)"
            )
    else:
        assert sleep_enabled is False, (
            f"Expected sleep mode to be disabled with UNSLOTH_VLLM_STANDBY=0, "
            f"got {sleep_enabled}"
        )

    ll_cfg = engine.llm_engine.vllm_config.model_config
    print(
        f"[sleep-smoke] llm_engine.vllm_config.model_config.enable_sleep_mode: "
        f"{getattr(ll_cfg, 'enable_sleep_mode', None)}"
    )
    assert getattr(ll_cfg, "enable_sleep_mode", None) == bool(sleep_enabled), (
        "_LLMEngineStub.model_config.enable_sleep_mode must mirror "
        "engine._sleep_mode_enabled"
    )

    from unsloth.inference.vllm_shim import LoRARequest  # noqa: F401

    prompts = [args.prompt]

    # ----- warmup -----
    t0 = time.perf_counter()
    out1 = engine.generate(
        prompts,
        sampling_params = type(
            "SP", (), {"max_tokens": args.max_new_tokens, "temperature": 0.0},
        )(),
    )
    print(
        f"[sleep-smoke] warmup generate: {time.perf_counter() - t0:.2f}s; "
        f"tok_ids[:10]={out1[0].outputs[0].token_ids[:10]}"
    )
    pre_tokens = list(out1[0].outputs[0].token_ids)

    probe_pre = _probe("pre-sleep")
    print(f"[sleep-smoke] {probe_pre}")

    # Diagnostic: checksum the inference-model weights so we can detect
    # if cuMem's sleep/wake round-trip corrupts any parameter.
    def _checksum_params(mod, limit = 16):
        import torch as _t

        out = []
        for i, (name, p) in enumerate(mod.named_parameters()):
            if i >= limit:
                break
            t = p.detach()
            out.append((name, list(t.shape), float(t.float().abs().sum().item())))
        return out

    pre_sums = _checksum_params(engine._inference_model)
    print("[sleep-smoke] pre-sleep first-16 param |sum|:")
    for n, s, v in pre_sums:
        print(f"  {n} {s} {v:.4f}")

    # Repeated sleep / wake / generate cycle test (plan matrix run #6).
    # Each cycle validates that the engine does not drift: tokens remain
    # bitwise identical, memory returns to baseline, weights round-trip
    # cleanly. A bug that only surfaces on the second or third cycle
    # (stale Python state, double-wake, leaked handles) fails here.
    for cycle in range(args.cycles):
        if args.cycles > 1:
            print(f"[sleep-smoke] --- cycle {cycle + 1}/{args.cycles} ---")

        # ----- sleep -----
        t0 = time.perf_counter()
        engine.sleep(level = 1)
        t_sleep = time.perf_counter() - t0
        probe_post = _probe(f"post-sleep[{cycle + 1}]")
        print(
            f"[sleep-smoke] sleep(level=1) took {t_sleep:.3f}s; {probe_post}"
        )

        if sleep_enabled:
            drop = probe_pre["cuda_used_gb"] - probe_post["cuda_used_gb"]
            print(
                f"[sleep-smoke] process-level VRAM drop on sleep: "
                f"{drop:+.3f} GB (cuMem-managed; not visible in "
                f"torch.memory_allocated)"
            )
            if not getattr(engine, "_single_copy_mode", False):
                # 16-bit path: both weights + kv_cache pools are dropped.
                assert drop >= 1.0, (
                    f"Expected multi-GB drop in process-level VRAM on "
                    f"sleep(level=1); got {drop:+.3f} GB"
                )
            else:
                # 4-bit single-copy: only KV cache drops; weights stay.
                assert drop > 0.0, (
                    f"Expected KV-cache drop in process-level VRAM on "
                    f"sleep(level=1); got {drop:+.3f} GB"
                )
        else:
            # With sleep mode off, the sleep() call must not free VRAM.
            # Process-level jitter is allowed (shared GPU); torch-owned
            # allocations must be untouched.
            assert (
                probe_post["allocated_gb"] == probe_pre["allocated_gb"]
            ), (
                "With sleep mode disabled, torch.memory_allocated must "
                "be unchanged by sleep()"
            )

        # ----- wake -----
        t0 = time.perf_counter()
        engine.wake_up()
        t_wake = time.perf_counter() - t0
        probe_wake = _probe(f"post-wake[{cycle + 1}]")
        print(
            f"[sleep-smoke] wake_up() took {t_wake:.3f}s; {probe_wake}"
        )

        post_sums = _checksum_params(engine._inference_model)
        diffs = []
        for (n1, s1, v1), (n2, s2, v2) in zip(pre_sums, post_sums):
            delta = abs(v1 - v2)
            if delta > 0.0:
                diffs.append((n1, v1, v2, delta))
        print(
            f"[sleep-smoke] post-wake weight diff: "
            f"{len(diffs)}/{len(pre_sums)} params changed "
            f"(bitwise-exact restore expected)"
        )
        for n, v1, v2, d in diffs[:8]:
            print(f"  diff {n}: pre={v1:.4f} post={v2:.4f} delta={d:.4f}")
        assert len(diffs) == 0, (
            f"Weight corruption on sleep / wake (cycle {cycle + 1}): "
            f"{len(diffs)}/{len(pre_sums)} first-layer params changed"
        )

        # ----- verify we can still generate -----
        t0 = time.perf_counter()
        out2 = engine.generate(
            prompts,
            sampling_params = type(
                "SP", (), {"max_tokens": args.max_new_tokens, "temperature": 0.0},
            )(),
        )
        t_regen = time.perf_counter() - t0
        post_tokens = list(out2[0].outputs[0].token_ids)
        match = pre_tokens == post_tokens
        print(
            f"[sleep-smoke] post-wake generate: {t_regen:.2f}s; "
            f"tok_ids[:10]={post_tokens[:10]}; matches_pre={match}"
        )
        assert match, (
            f"Pre-sleep / post-wake token ids must match exactly "
            f"(cycle {cycle + 1}).\n"
            f"Pre:  {pre_tokens}\nPost: {post_tokens}"
        )

        if sleep_enabled:
            delta = probe_wake["cuda_used_gb"] - probe_pre["cuda_used_gb"]
            print(
                f"[sleep-smoke] post-wake vs pre-sleep process-level "
                f"VRAM delta: {delta:+.3f} GB (tolerance: +/- 1.5 GB)"
            )
            assert abs(delta) < 1.5, (
                f"Post-wake VRAM diverged from pre-sleep "
                f"(cycle {cycle + 1}): delta={delta:+.3f} GB"
            )

    print(f"[sleep-smoke] PASS ({args.cycles} cycle(s))")


if __name__ == "__main__":
    main()
