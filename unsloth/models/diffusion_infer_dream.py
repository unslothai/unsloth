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
"""Dream's greedy diffusion_generate (generation_utils.py `_sample`, alg entropy / maskgit_plus / topk_margin,
alg_temp 0), with logits only for the generated region and the transfer schedule worked out up front.

The reference projects every position to the vocabulary each step and reads the masked count back to the host.
The masked count drops by exactly the scheduled amount unless the model proposes the mask token itself, so the
schedule is worked out before the loop; a run where that happened returns None and the caller falls back."""

import math

import torch
import torch.nn.functional as F

from .diffusion_infer_block import _Frames

__all__ = ["dream_fast_supported", "dream_fast_sample"]

_ALGS = ("entropy", "maskgit_plus", "topk_margin")


def dream_fast_supported(
    input_ids, attention_mask, mask_id, temperature, top_p, top_k, alg, alg_temp
):
    # An empty prompt is left to the reference: its position 0 reads its own logits, not a shifted row.
    return (
        input_ids.shape[1] > 0
        and not temperature
        and top_p is None
        and top_k is None
        and alg in _ALGS
        and not alg_temp
        and (attention_mask is None or bool(attention_mask.all()))
        and not bool((input_ids == mask_id).any())
    )


@torch.no_grad()
def dream_fast_sample(
    base,
    input_ids,
    *,
    max_new_tokens,
    steps,
    mask_id,
    alg = "entropy",
    eps = 1e-3,
    on_frame = None,
    check_stop = None,
    progress = None,
):
    batch, prompt_length = input_ids.shape
    x = F.pad(input_ids, (0, max_new_tokens), value = mask_id)
    # The reference's own float32 timesteps, read back once.
    timesteps = torch.linspace(1, eps, steps + 1, device = x.device).cpu()
    masked = torch.tensor(max_new_tokens * batch) / batch
    counts = []
    for i in range(steps):
        t, s = timesteps[i], timesteps[i + 1]
        count = int(masked * (1 - s / t)) if i < steps - 1 else int(masked)
        counts.append(count)
        masked = masked - count
    frames = _Frames(on_frame, check_stop, progress = progress)
    generated = x[:, prompt_length:]
    remasked = torch.zeros((), dtype = torch.bool, device = x.device)
    for i, count in enumerate(counts):
        frames.step(x, prompt_length, x.shape[1], 0, i, steps, mask_id)
        mask_index = generated == mask_id
        hidden = base.model(input_ids = x, attention_mask = "full")[0]
        # Position j reads the logits of j - 1 (Dream shifts them right by one).
        logits = base.lm_head(hidden[:, prompt_length - 1 : -1])
        probs = torch.softmax(logits, dim = -1)
        confidence, x0 = probs.max(dim = -1)
        if alg == "topk_margin":
            top2 = torch.topk(probs, 2, dim = -1).values
            confidence = top2[..., 0] - top2[..., 1]
        elif alg == "entropy":
            confidence = torch.sum(probs * torch.log(probs + 1e-10), dim = -1)
        if count > 0:
            # Full canvas width (prompt at -inf) so ties break exactly as in the reference's topk.
            full = F.pad(
                torch.where(mask_index, confidence, -math.inf), (prompt_length, 0), value = -math.inf
            )
            index = torch.topk(full, count).indices - prompt_length
            chosen = torch.where(mask_index, x0, mask_id).gather(1, index)
            remasked |= (chosen == mask_id).any()
            generated.scatter_(1, index, chosen)
    if bool(remasked):
        return None
    frames.step(x, prompt_length, x.shape[1], 0, steps, steps, mask_id, force = True)
    return x
