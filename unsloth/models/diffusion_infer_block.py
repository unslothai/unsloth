# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Block diffusion decoding (SDAR, LLaDA2) with an exact prefix KV cache.

Attention is block-causal, so a committed block's keys and values never change: they are computed once, and each
denoising step forwards only the active block, and a finished block is committed in the same forward as the next
block's first step, so the forward count matches the recompute sampler. Token selection runs on the device; the
host syncs once per step (has the block finished early) and once per block (EOS)."""

import math
import time

import torch
import torch.nn.functional as F

__all__ = ["cached_block_sample"]

# The sampler kwargs cached_block_sample takes; other family kwargs are dropped as the recompute sampler does.
_CACHED_BLOCK_KWARGS = (
    "max_new_tokens",
    "temperature",
    "mask_id",
    "eos_ids",
    "confidence_threshold",
    "top_k",
    "top_p",
    "on_frame",
    "check_stop",
)


class _Frames:
    """Throttled canvas snapshots and cancellation, host side and outside any compiled region."""

    def __init__(
        self,
        on_frame,
        check_stop,
        interval = 0.1,
    ):
        self.on_frame, self.check_stop, self.interval = on_frame, check_stop, interval
        self.last = 0.0

    def step(
        self,
        x,
        prompt_length,
        end,
        block,
        step,
        total,
        mask_id,
        force = False,
    ):
        if self.check_stop is not None:
            self.check_stop()
        if self.on_frame is None:
            return
        now = time.monotonic()
        if not force and now - self.last < self.interval:
            return
        self.last = now
        self.on_frame(
            {
                "block": block,
                "step": step,
                "total": total,
                "tokens": x[0, prompt_length:end].tolist(),
                "mask_id": mask_id,
            }
        )


class _PrefixCache:
    """Committed keys / values per layer. A denoising step reads them without writing, so nothing is cropped."""

    def __init__(self):
        self.keys, self.values, self.commit = {}, {}, False

    def update(
        self,
        key,
        value,
        layer_idx,
        cache_kwargs = None,
    ):
        if layer_idx in self.keys:
            key = torch.cat([self.keys[layer_idx], key], dim = -2)
            value = torch.cat([self.values[layer_idx], value], dim = -2)
        if self.commit:
            self.keys[layer_idx], self.values[layer_idx] = key, value
        return key, value

    def get_seq_length(self, layer_idx = 0):
        return self.keys[layer_idx].shape[-2] if layer_idx in self.keys else 0

    def crop(self, tokens_to_remove):
        # Negative: drop that many trailing tokens (what DynamicCache.crop does with a negative value).
        for layer_idx in self.keys:
            self.keys[layer_idx] = self.keys[layer_idx][..., :tokens_to_remove, :]
            self.values[layer_idx] = self.values[layer_idx][..., :tokens_to_remove, :]


def _llada2_runner(base, ids, mask, position_ids, cache):
    """LLaDA2's model forward only takes a square [B, 1, L, L] mask, so a cached step drives its layers directly
    with the same mask preparation and logits dtype as its forward."""
    inner = base.model
    module = __import__(type(inner).__module__, fromlist = ["_"])
    hidden = inner.word_embeddings(ids)
    past = cache.get_seq_length() if cache is not None else 0
    mask = module._prepare_4d_causal_attention_mask_for_sdpa(mask, ids.shape, hidden, past)
    position_embeddings = inner.rotary_emb(hidden, position_ids)
    for layer in inner.layers:
        hidden = layer(
            hidden,
            attention_mask = mask,
            position_ids = position_ids,
            past_key_value = cache,
            use_cache = cache is not None,
            position_embeddings = position_embeddings,
        )[0]
    return base.lm_head(inner.norm(hidden)).float()


def _sdar_runner(base, ids, mask, position_ids, cache, store_kv):
    return base(
        ids,
        attention_mask = mask,
        position_ids = position_ids,
        past_key_values = cache,
        use_cache = True,
        store_kv = store_kv,
    ).logits.float()


def _select(logits, mask_index, temperature, top_k, top_p, count, threshold):
    """The block sampler's per-step choice, on the device: sample / argmax, then accept every masked token above
    `threshold`, or else the `count` most confident (never more than the masks left)."""
    if temperature > 0:
        logits = logits / temperature
        if top_k and top_k > 0:
            kth = torch.topk(logits, top_k).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, -math.inf)
        if top_p is not None and top_p < 1.0:
            sorted_logits, order = torch.sort(logits, descending = True)
            drop = F.softmax(sorted_logits, dim = -1).cumsum(-1) > top_p
            drop[..., 1:] = drop[..., :-1].clone()
            drop[..., 0] = False
            logits = logits.masked_fill(drop.scatter(-1, order, drop), -math.inf)
        probs = F.softmax(logits, dim = -1)
        x0 = torch.multinomial(probs.flatten(0, 1), 1).view(probs.shape[:-1])
    else:
        probs = F.softmax(logits, dim = -1)
        x0 = probs.argmax(dim = -1)
    x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    confidence = torch.where(mask_index, x0_p, -math.inf)
    high = confidence > threshold
    use_high = high.sum(-1, keepdim = True) >= count
    k = torch.minimum(count, mask_index.sum(-1, keepdim = True))
    order = torch.topk(confidence, confidence.shape[-1]).indices
    rank = torch.empty_like(order).scatter_(
        -1, order, torch.arange(order.shape[-1], device = order.device).expand_as(order)
    )
    transfer = torch.where(use_high, high, rank < k)
    return x0, transfer


@torch.no_grad()
def cached_block_sample(
    base,
    input_ids,
    *,
    family,
    max_new_tokens,
    steps,
    block_length,
    temperature,
    mask_id,
    eos_ids,
    confidence_threshold,
    top_k = 0,
    top_p = 1.0,
    on_frame = None,
    check_stop = None,
):
    from transformers import DynamicCache

    batch, prompt_length = input_ids.shape
    num_blocks = (prompt_length + max_new_tokens + block_length - 1) // block_length
    total_length = num_blocks * block_length
    device = input_ids.device
    block_causal = torch.tril(torch.ones(num_blocks, num_blocks, device = device, dtype = torch.bool))
    allowed = block_causal.repeat_interleave(block_length, 0).repeat_interleave(block_length, 1)[
        None, None
    ]
    if family == "llada2":
        dtype = next(base.parameters()).dtype
        attention = torch.zeros(allowed.shape, dtype = dtype, device = device).masked_fill(
            ~allowed, torch.finfo(dtype).min
        )

        def run(ids, start, end, cache, store):
            cache.commit = store
            return _llada2_runner(
                base, ids, attention[..., start:end, :end], position_ids[:, start:end], cache
            )
    else:
        attention = allowed

        def run(ids, start, end, cache, store):
            return _sdar_runner(
                base, ids, attention[..., start:end, :end], position_ids[:, start:end], cache, store
            )

    position_ids = torch.arange(total_length, device = device)[None]
    x = torch.full((batch, total_length), mask_id, dtype = torch.long, device = device)
    x[:, :prompt_length] = input_ids
    schedule = torch.full((steps,), block_length // steps, dtype = torch.long)
    schedule[: block_length % steps] += 1
    counts = schedule.to(device)[:, None, None].expand(steps, batch, 1)
    frames = _Frames(on_frame, check_stop)
    eos = torch.tensor(eos_ids, device = device) if eos_ids else None

    cache = _PrefixCache() if family == "llada2" else DynamicCache()

    def commit_then_step(commit_start, start, end):
        """Commit x[:, commit_start:start] and run the first denoising step of x[:, start:end] in one forward,
        then drop the active block's keys / values again: the active rows attend exactly as in a step forward."""
        logits = run(x[:, commit_start:end], commit_start, end, cache, True)
        cache.crop(-(end - start))
        return logits[:, start - commit_start :]

    first = prompt_length // block_length
    pending = None
    if first > 0 and first < num_blocks:
        # A prompt block only attends to earlier prompt blocks, so it is committed once up front.
        pending = commit_then_step(0, first * block_length, (first + 1) * block_length)
    for block in range(first, num_blocks):
        start, end = block * block_length, (block + 1) * block_length
        window = x[:, start:end]
        for step in range(steps + 1):
            mask_index = window == mask_id
            if not bool(mask_index.any()):
                break
            frames.step(x, prompt_length, end, block, step, steps, mask_id)
            logits = pending if pending is not None else run(window, start, end, cache, False)
            pending = None
            x0, transfer = _select(
                logits,
                mask_index,
                temperature,
                top_k,
                top_p,
                counts[min(step, steps - 1)],
                confidence_threshold,
            )
            window.copy_(torch.where(transfer, x0, window))
        pending = None
        frames.step(x, prompt_length, end, block, steps, steps, mask_id, force = True)
        if eos is not None and bool(torch.isin(x[:, prompt_length:end], eos).any()):
            return x[:, prompt_length:end]
        if block + 1 < num_blocks:
            # The next block is all masks: its first step rides along with this block's commit.
            pending = commit_then_step(start, end, end + block_length)
    return x[:, prompt_length:]
