"""Main-thread synchronous driver for `ContinuousBatchProcessor`.

`ContinuousBatchingManager.start()` spawns a background thread that owns the
decode loop. That thread conflicts with two things we want to enable here:

1. `torch.compile(mode="reduce-overhead")` which uses `cudagraph_trees` and
   requires main-thread TLS.
2. Raw `torch.cuda.CUDAGraph` capture/replay on the decode forward, which is
   the hot path. Captures *can* live in a child thread in principle, but
   integrating with Inductor and debugging goes much smoother on the main
   thread.

The manager's dead `warmup()` path suggests CB was supposed to grow CUDA
graph support upstream, but `init_continuous_batching` currently raises
`NotImplementedError` on `use_cuda_graph=True`. This driver side-steps that
entirely by not going through `manager.start()` at all.

Key fixed-shape invariant: with `slice_inputs=False`, the full pre-allocated
tensor buffers (input_ids, position_ids, cu_seq_lens_*, attention_mask,
read_index / write_index) are returned as *views of the same storage* every
step, so their shapes are constant across iterations. That is the precondition
for CUDA graph replay to be safe.

Greedy sampling only (`do_sample=False`). `torch.multinomial` is not
CUDA-graph-friendly; a downstream stochastic sanity check runs in a separate,
non-graphed path.

Usage:
    from cb_sync_driver import cb_sync_generate, CBSyncConfig
    cfg = CBSyncConfig(max_new_tokens=512, use_cuda_graph=True)
    outputs = cb_sync_generate(model, generation_config, prompt_ids_list, cfg)
"""

from __future__ import annotations

import queue
import threading
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
    use_cuda_graph: bool = True
    # Number of eager warmup steps before capturing a CUDA graph.
    warmup_steps: int = 2
    # Generation config knobs (forwarded to the manager's GenerationConfig).
    do_sample: bool = False  # greedy only (CUDA-graph safe)
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    # Paged cache upper bounds; keep well above the default 256 / 4096.
    max_batch_tokens: int = 8192
    num_blocks: int = 8192
    # Progress callback (step_index, tokens_produced_total) -> None.
    on_step: Optional[callable] = field(default=None)


class SyncCBDriver:
    """Main-thread driver that owns the PagedAttentionCache,
    ContinuousBatchProcessor, and (optionally) a captured CUDA graph.

    Unlike `ContinuousBatchingManager.start()`, there is no background
    thread; `drive_until_empty()` blocks until every pending request is
    finished.
    """

    def __init__(self, model: torch.nn.Module, generation_config: GenerationConfig,
                 cfg: CBSyncConfig):
        self.model = model.eval()
        self.cfg = cfg
        # Force-greedy + upper-bound overrides on a copy.
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
        # Paged cache reads these at init.
        self.generation_config = gc

        # We reuse the Manager's methods but never call `.start()`. Its
        # constructor builds: logit processor, do_sample flag, etc.
        self.manager = ContinuousBatchingManager(
            model=self.model,
            generation_config=gc,
            manual_eviction=False,
            streaming=False,
            slice_inputs=False,  # fixed-shape views -> CUDA-graph safe
        )
        # The manager's `use_cuda_graph` is checked inside `warmup()`, but its
        # `__init__` refuses to set it. Set it directly now that we bypass
        # `init_continuous_batching`.
        self.manager.use_cuda_graph = cfg.use_cuda_graph

        # Stand up the cache + processor ourselves so `_inner_generation_loop`
        # has everything it needs.
        self.cache = PagedAttentionCache(
            self.model.config,
            gc,
            self.model.device,
            self.model.dtype,
            tp_size=getattr(self.model, "_tp_size", None),
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
            streaming=False,
            manual_eviction=False,
            slice_inputs=False,
        )
        self.manager.batch_processor = self.batch_processor
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._step_count = 0

    def add_requests(self, prompt_ids_list: list[list[int]]) -> list[str]:
        return [self.manager.add_request(ids) for ids in prompt_ids_list]

    def _graphed_step(self):
        """Capture or replay the decode CUDA graph."""
        if self._graph is None:
            # Eager warmup to populate allocator + workspaces.
            for _ in range(self.cfg.warmup_steps):
                self.manager._generation_step(self.batch_processor)
            torch.cuda.synchronize()
            stream = torch.cuda.Stream(device=self.model.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.manager._generation_step(self.batch_processor)
            torch.cuda.current_stream().wait_stream(stream)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=stream):
                self.manager._generation_step(self.batch_processor)
        else:
            self._graph.replay()

    def drive_until_empty(self) -> dict[str, list[int]]:
        """Run the decode loop until every request finishes. Returns a dict
        {request_id: generated_token_ids}."""
        results: dict[str, list[int]] = {}
        # prepare_next_batch drains self.input_queue into the scheduler; we
        # have to call it at least once before has_pending_requests() can
        # return True. Loop until both the input_queue is empty AND the
        # scheduler has nothing queued/active.
        while True:
            input_empty = self.manager.input_queue.empty()
            nothing_scheduled = not self.batch_processor.has_pending_requests()
            if input_empty and nothing_scheduled:
                break
            # 1. CPU: schedule the next batch (prepare_next_batch reads the
            #    input_queue, packs shapes).
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if not self.batch_processor.prepare_next_batch():
                # prepare_next_batch returns False if both the input queue
                # drained empty AND the scheduler has no active requests. If
                # we reach here with items still in input_queue, something is
                # wrong -- bail to avoid an infinite loop.
                break
            # 2. GPU: forward (graphed on decode steps, eager on prefill).
            if self.cfg.use_cuda_graph and self._is_pure_decode():
                self._graphed_step()
            else:
                self.manager._generation_step(self.batch_processor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # 3. CPU: append new tokens, detect EOS, update scheduler.
            self.batch_processor.update_batch()
            self._step_count += 1
            if self.cfg.on_step is not None:
                self.cfg.on_step(self._step_count, self._produced())
            # 4. Drain output_queue into results dict.
            while True:
                try:
                    out = self.manager.output_queue.get_nowait()
                except queue.Empty:
                    break
                if out.status == RequestStatus.FINISHED:
                    results[out.request_id] = out.generated_tokens
        # Final drain after loop exits.
        while True:
            try:
                out = self.manager.output_queue.get_nowait()
            except queue.Empty:
                break
            if out.status == RequestStatus.FINISHED:
                results[out.request_id] = out.generated_tokens
        return results

    def _is_pure_decode(self) -> bool:
        """A decode-only batch has every request contributing exactly one
        query token (q_len == b_size). Prefill batches have q_len >> b_size.
        Shape consistency between decodes is what makes the graph replayable.
        """
        try:
            return (self.batch_processor.total_query_length
                    == self.batch_processor.total_batch_size)
        except Exception:
            return False

    def _produced(self) -> int:
        return sum(len(r.generated_tokens) for r
                   in getattr(self.batch_processor.scheduler, "active_requests", {}).values())

    def close(self):
        # Caches hold GPU memory; free them explicitly.
        self._graph = None
        self.cache = None
        self.batch_processor = None
        self.manager.batch_processor = None


def cb_sync_generate(model: torch.nn.Module, generation_config: GenerationConfig,
                     prompt_ids_list: list[list[int]],
                     cfg: CBSyncConfig) -> dict[str, list[int]]:
    """One-shot entrypoint: build a driver, submit, drain, close.

    Matches the semantics of `model.generate_batch(...)` but on the main
    thread with optional CUDA graph capture.
    """
    driver = SyncCBDriver(model, generation_config, cfg)
    driver.add_requests(prompt_ids_list)
    try:
        return driver.drive_until_empty()
    finally:
        driver.close()


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
    parser.add_argument("--model_name", default="unsloth/Qwen3-4B-Base")
    parser.add_argument("--n_prompts", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--attn_impl", default="paged_attention")
    parser.add_argument("--use_cuda_graph", action="store_true")
    parser.add_argument("--max_batch_tokens", type=int, default=8192)
    parser.add_argument("--num_blocks", type=int, default=8192)
    parser.add_argument("--stats_path", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).to("cuda")
    model.eval()

    from unsloth_grpo_common import (
        SYSTEM_PROMPT, apply_chat_template_to_tokenizer,
    )
    from datasets import load_dataset
    apply_chat_template_to_tokenizer(tok)
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    ds = ds.shuffle(seed=3407).select(range(args.n_prompts))
    messages = [[{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": x["prompt"]}] for x in ds]
    prompt_ids = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True)
                  for m in messages]

    gc = GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=False,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id, use_cache=True,
    )

    cfg = CBSyncConfig(
        max_new_tokens=args.max_new_tokens,
        use_cuda_graph=args.use_cuda_graph,
        max_batch_tokens=args.max_batch_tokens,
        num_blocks=args.num_blocks,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )

    torch.cuda.reset_peak_memory_stats()
    # Warmup (first 16 prompts).
    _ = cb_sync_generate(model, gc, prompt_ids[:16], cfg)
    torch.cuda.synchronize()

    wall_times = []
    total_decoded = 0
    for _ in range(2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = cb_sync_generate(model, gc, prompt_ids, cfg)
        torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)
        total_decoded = sum(len(v) for v in results.values())

    med = sorted(wall_times)[len(wall_times) // 2]
    out = {
        "backend": "cb_sync_driver",
        "use_cuda_graph": args.use_cuda_graph,
        "attn_impl": args.attn_impl,
        "n_prompts": args.n_prompts,
        "n_decoded_tokens": total_decoded,
        "wall_times_s": wall_times,
        "median_wall_s": med,
        "decode_tps": total_decoded / med if med else 0,
        "max_new_tokens": args.max_new_tokens,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)) or ".", exist_ok=True)
    with open(args.stats_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
