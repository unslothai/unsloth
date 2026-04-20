"""Main-thread synchronous driver for `ContinuousBatchProcessor`.

`ContinuousBatchingManager.start()` spawns a background thread that owns the
decode loop. That thread conflicts with two things we want to enable here:

1. `torch.compile(mode="reduce-overhead")` which uses `cudagraph_trees` and
   requires main-thread TLS.
2. Raw `torch.cuda.CUDAGraph` capture / replay on the decode forward.

The manager's dead `warmup()` path suggests CB was supposed to grow CUDA
graph support upstream but `init_continuous_batching` raises
`NotImplementedError` on `use_cuda_graph=True`. This driver side-steps the
whole manager thread, so the cudagraph integration point is now available.

Key fixed-shape invariant: with `slice_inputs=False`, the full pre-allocated
tensor buffers (input_ids, position_ids, cu_seq_lens_*, attention_mask,
read_index / write_index) are returned as views of the same storage every
step, so their shapes are constant across iterations. That is the
precondition for graph replay / cudagraph_trees to be safe.

Greedy sampling only (`do_sample=False`). `torch.multinomial` is not
graph-friendly.

Usage:
    driver = SyncCBDriver(model, gen_config, CBSyncConfig(compile_mode="reduce-overhead"))
    # Reuse across many rollouts (cache / compiled forward stay warm):
    for batch in batches:
        driver.add_requests(batch)
        out = driver.drive_until_empty()
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.continuous_batching import (
    PagedAttentionCache,
    RequestStatus,
)
from transformers.generation.continuous_batching.continuous_api import (
    ContinuousBatchProcessor,
    ContinuousBatchingManager,
)
from transformers.generation.continuous_batching.scheduler import FIFOScheduler


@dataclass
class CBSyncConfig:
    """Tunables for the sync driver."""

    max_new_tokens: int = 512
    # torch.compile mode for the model forward. None = eager.
    # "reduce-overhead" triggers cudagraph_trees, which captures a CUDA
    # graph per unique input shape and replays it afterwards.
    compile_mode: Optional[str] = None
    do_sample: bool = False  # greedy only (graph-safe)
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    # `slice_inputs=False` would give fixed shapes every step (graph-friendly)
    # but forces every decode step to do `max_batch_tokens` tokens of work,
    # which at max_batch_tokens=8192 is ~256x more than the real decode batch.
    # Prefer `slice_inputs=True` (natural shapes) and let torch.compile bucket
    # per shape. Steady-state decode has one shape so most steps replay the
    # same graph anyway.
    slice_inputs: bool = True
    # Paged cache upper bounds.
    max_batch_tokens: int = 8192
    num_blocks: int = 8192
    # `torch._dynamo.config.cache_size_limit`: raise when varying shapes.
    dynamo_cache_size_limit: int = 256
    on_step: Optional[callable] = field(default = None)


class SyncCBDriver:
    """Main-thread driver that owns the PagedAttentionCache,
    ContinuousBatchProcessor, and optionally a `torch.compile`-compiled
    forward. Reusable across multiple `drive_until_empty` calls -- the cache
    and compiled forward stay warm between rounds.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        generation_config: GenerationConfig,
        cfg: CBSyncConfig,
    ):
        self.model = model.eval()
        self.cfg = cfg
        gc = GenerationConfig.from_dict(generation_config.to_dict())
        gc.do_sample = cfg.do_sample
        if cfg.max_new_tokens:
            gc.max_new_tokens = cfg.max_new_tokens
        if cfg.eos_token_id is not None:
            gc.eos_token_id = cfg.eos_token_id
        if cfg.pad_token_id is not None:
            gc.pad_token_id = cfg.pad_token_id
        gc.max_batch_tokens = cfg.max_batch_tokens
        gc.num_blocks = cfg.num_blocks
        self.generation_config = gc

        self.manager = ContinuousBatchingManager(
            model = self.model,
            generation_config = gc,
            manual_eviction = False,
            streaming = False,
            slice_inputs = cfg.slice_inputs,
        )

        self.cache = PagedAttentionCache(
            self.model.config,
            gc,
            self.model.device,
            self.model.dtype,
            tp_size = getattr(self.model, "_tp_size", None),
        )
        self.batch_processor = ContinuousBatchProcessor(
            self.cache,
            self.model.config,
            gc,
            self.manager.input_queue,
            self.manager.output_queue,
            self.manager.stop_event,
            self.model.device,
            self.model.dtype,
            FIFOScheduler(self.cache),
            streaming = False,
            manual_eviction = False,
            slice_inputs = cfg.slice_inputs,
        )
        self.manager.batch_processor = self.batch_processor

        # torch.compile on the model forward. With slice_inputs=True the
        # forward sees varying shapes (prefill bursts + decode steady state);
        # `dynamic=True` lets Inductor bucket per shape without re-tracing
        # every call, and `mode="reduce-overhead"` wraps each bucket in a
        # CUDA graph replay path.
        if cfg.compile_mode:
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = cfg.dynamo_cache_size_limit
            # GRPO's `requires_grad_` issue doesn't apply here (eval mode).
            try:
                torch._dynamo.config.allow_unspec_int_on_nn_module = True
            except AttributeError:
                pass
            print(f"[cb_sync] torch.compile(model, mode='{cfg.compile_mode}', "
                  f"dynamic=True)")
            self.model.forward = torch.compile(
                self.model.forward,
                mode = cfg.compile_mode,
                dynamic = True,
                fullgraph = False,
            )
        self._step_count = 0

    def add_requests(self, prompt_ids_list: list[list[int]]) -> list[str]:
        return [self.manager.add_request(ids) for ids in prompt_ids_list]

    def drive_until_empty(self) -> dict[str, list[int]]:
        """Run the decode loop until every request finishes. Returns a dict
        {request_id: generated_token_ids}.

        Reusable across calls on the same driver -- the paged cache and the
        compiled forward stay warm.
        """
        results: dict[str, list[int]] = {}
        while True:
            if (self.manager.input_queue.empty()
                    and not self.batch_processor.has_pending_requests()):
                break
            if not self.batch_processor.prepare_next_batch():
                break
            # With `reduce-overhead`, Inductor captures a CUDA graph on the
            # first call with a given shape signature and replays it after.
            # Our shapes are constant (slice_inputs=False), so the second call
            # is already replaying. No per-step torch.cuda.synchronize() --
            # the replay itself fences appropriately.
            self.manager._generation_step(self.batch_processor)
            self.batch_processor.update_batch()
            self._step_count += 1
            if self.cfg.on_step is not None:
                self.cfg.on_step(self._step_count, 0)
            # Drain output_queue as requests finish.
            while True:
                try:
                    out = self.manager.output_queue.get_nowait()
                except queue.Empty:
                    break
                if out.status == RequestStatus.FINISHED:
                    results[out.request_id] = out.generated_tokens
        while True:
            try:
                out = self.manager.output_queue.get_nowait()
            except queue.Empty:
                break
            if out.status == RequestStatus.FINISHED:
                results[out.request_id] = out.generated_tokens
        return results

    def close(self):
        self.cache = None
        self.batch_processor = None
        self.manager.batch_processor = None


# Simple microbench harness so the file is runnable standalone.
if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys
    from pathlib import Path

    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE))

    import flash_attn_fa4_shim  # noqa: E402

    flash_attn_fa4_shim.apply()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = "unsloth/Qwen3-4B-Base")
    parser.add_argument("--n_prompts", type = int, default = 32)
    parser.add_argument("--n_rounds", type = int, default = 2)
    parser.add_argument("--max_new_tokens", type = int, default = 512)
    parser.add_argument("--attn_impl", default = "paged_attention")
    parser.add_argument("--compile_mode", default = None,
                        choices = [None, "default", "reduce-overhead",
                                   "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--max_batch_tokens", type = int, default = 8192)
    parser.add_argument("--num_blocks", type = int, default = 8192)
    parser.add_argument("--lora_adapter", default = None)
    parser.add_argument("--stats_path", required = True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype = torch.bfloat16,
        attn_implementation = args.attn_impl,
    ).to("cuda")
    model.eval()

    if args.lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, str(Path(args.lora_adapter).resolve()), is_trainable = False
        )
        model.eval()

    from unsloth_grpo_common import (
        SYSTEM_PROMPT,
        apply_chat_template_to_tokenizer,
    )
    from datasets import load_dataset

    apply_chat_template_to_tokenizer(tok)
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split = "train")
    ds = ds.shuffle(seed = 3407).select(range(args.n_prompts))
    messages = [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": x["prompt"]}]
        for x in ds
    ]
    prompt_ids = [tok.apply_chat_template(m, add_generation_prompt = True, tokenize = True)
                  for m in messages]

    gc_cfg = GenerationConfig(
        max_new_tokens = args.max_new_tokens,
        do_sample = False,
        pad_token_id = tok.pad_token_id,
        bos_token_id = tok.bos_token_id,
        eos_token_id = tok.eos_token_id,
        use_cache = True,
    )

    cfg = CBSyncConfig(
        max_new_tokens = args.max_new_tokens,
        compile_mode = args.compile_mode,
        max_batch_tokens = args.max_batch_tokens,
        num_blocks = args.num_blocks,
        eos_token_id = tok.eos_token_id,
        pad_token_id = tok.pad_token_id or tok.eos_token_id,
    )

    torch.cuda.reset_peak_memory_stats()

    # One driver, multiple rounds -- cache + compiled forward stay warm.
    driver = SyncCBDriver(model, gc_cfg, cfg)

    # Warmup (first 16 prompts). With compile, this amortizes the capture.
    print("[cb_sync] warmup...")
    driver.add_requests(prompt_ids[:16])
    _ = driver.drive_until_empty()
    torch.cuda.synchronize()

    wall_times = []
    total_decoded = 0
    for r in range(args.n_rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        driver.add_requests(prompt_ids)
        results = driver.drive_until_empty()
        torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)
        total_decoded = sum(len(v) for v in results.values())
        print(f"[cb_sync] round {r}: {wall_times[-1]:.2f}s, {total_decoded} tokens, "
              f"{total_decoded / wall_times[-1]:.1f} tok/s")

    med = sorted(wall_times)[len(wall_times) // 2]
    out = {
        "backend": "cb_sync_driver",
        "compile_mode": args.compile_mode,
        "attn_impl": args.attn_impl,
        "lora_adapter": args.lora_adapter,
        "n_prompts": args.n_prompts,
        "n_decoded_tokens": total_decoded,
        "wall_times_s": wall_times,
        "median_wall_s": med,
        "decode_tps": total_decoded / med if med else 0,
        "max_new_tokens": args.max_new_tokens,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)) or ".", exist_ok = True)
    with open(args.stats_path, "w") as f:
        json.dump(out, f, indent = 2)
    print(json.dumps(out, indent = 2))
    driver.close()
    os._exit(0)
