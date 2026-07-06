# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""PrefixGrouper layout builder + completion-logprob extraction for the Unsloth GRPO
packed path (all archs that route through the varlen attention dispatch).

Given the de-padded, LEFT-PACKED input_ids the packed GRPO path already works with, this
module:

  1. Detects consecutive ``num_generations`` rows that share a prompt prefix (byte-
     identical prompt precondition; falls back / returns None otherwise).
  2. Builds ONE flat shared-prefix stream across all groups
     ``[ prefix_g0, suf_g0_0 .. suf_g0_{G-1}, prefix_g1, ... ]`` with position_ids that
     continue each prefix positionally, plus a ``PrefixSegInfo`` segment table for the
     FlexAttention shared-prefix kernel.
  3. Extracts completion logprobs via the index map (completion pos ``j==0`` predicted
     from the shared prefix's last token; ``j>=1`` from the preceding suffix token) and
     scatters them back into ``[total_rows, W]`` EXACTLY where the full-row packed path
     puts them (dest = ``orig_row*L + orig_col``), so grpo_compute_loss / completion_mask
     / TIS / metrics are byte-untouched.

The flat stream is built by GATHERING original (row, col) coordinates out of input_ids,
so the grad path's autograd flows to the same embedding rows as today (the shared prefix
now contributes grad once = the sum of the G repeats, which is mathematically identical).

``chunked_hidden_states_selective_log_softmax`` (from unsloth_zoo, passed in) is reused
verbatim over the gathered predicting-position hidden states, so fp32 accumulation,
logit_scale/softcapping/temperature are all preserved.

Env:
  UNSLOTH_GRPO_PREFIX_GROUPER=1            engage (default ON; set 0 to disable). Auto-off under vLLM.
  UNSLOTH_GRPO_PREFIX_GROUPER_TOKR=1.3     tok_r auto-gate threshold (env-overridable)
  UNSLOTH_GRPO_PREFIX_GROUPER_VERIFY=1     first-step self-verify (default ON)
  UNSLOTH_GRPO_PREFIX_GROUPER_TOL=0.7      self-verify PASS band (nats)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .prefix_grouper_kernel import build_seg_info_multigroup, PrefixSegInfo


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------
def env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() not in ("0", "false", "no", "off")


# One-time env reads; the helpers stay callable since unsloth_zoo imports and calls them.
_ENABLED = env_on("UNSLOTH_GRPO_SEQ_PACKING", "1") and env_on("UNSLOTH_GRPO_PREFIX_GROUPER", "1")
_VERIFY_ON = env_on("UNSLOTH_GRPO_PREFIX_GROUPER_VERIFY", "1")
_TOKR_THRESHOLD = float(os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER_TOKR", "1.3"))
_TOL_OK = float(os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER_TOL", "0.7"))


def prefix_grouper_enabled() -> bool:
    """PrefixGrouper requires seq-packing on (it reuses its de-pad + scatter machinery)."""
    return _ENABLED


def verify_on() -> bool:
    return _VERIFY_ON


def tokr_threshold() -> float:
    return _TOKR_THRESHOLD


def tol_ok() -> float:
    return _TOL_OK


# diff >= TOL_KILL = broken mask/isolation -> structure permanently unsafe; between
# tol_ok and TOL_KILL -> fall back for this shape but keep trying others.
TOL_KILL = 1.5


@dataclass
class GroupLayout:
    """Everything the GRPO forward needs to run + extract the shared-prefix path."""

    flat_ids: torch.Tensor  # [1, T]  (T == seg.T)
    position_ids: torch.Tensor  # [1, T]
    prefix_seg_info: PrefixSegInfo
    # per completion target token, aligned 1:1:
    tgt_rows: torch.Tensor  # [N] original row index
    tgt_cols: torch.Tensor  # [N] original padded column in that row
    tgt_pred: torch.Tensor  # [N] flat predicting index (into the T stream)
    tgt_flat: torch.Tensor  # [N] flat index of the target token itself (into T)
    total_rows: int
    L: int  # original padded seq length (input_ids.shape[1])
    W: int  # logits_to_keep + max_left_pad (scatter width)
    tok_r: float
    signature: Tuple

    def extract_logps(
        self,
        hidden,
        lm_head,
        chunked_fn,
        chunks,
        logit_scale_multiply,
        logit_scale_divide,
        logit_softcapping,
        temperature,
    ) -> torch.Tensor:
        """hidden: [1, T, Hdim] (pre-lm_head hidden states, UNSLOTH_RETURN_HIDDEN_STATES=1).
        Returns [total_rows, W] float32, byte-compatible with the packed path result."""
        # In a sharded model hidden may live on the lm-head device; move the small index
        # maps to hidden.device before indexing.
        device = hidden.device
        pred_h = hidden[0, self.tgt_pred.to(device), :].unsqueeze(0)  # [1, N, Hdim]
        tgt_ids = self.flat_ids[0, self.tgt_flat].to(device).unsqueeze(0)  # [1, N]
        sel = chunked_fn(
            pred_h,
            lm_head,
            tgt_ids,
            chunks,
            logit_scale_multiply,
            logit_scale_divide,
            logit_softcapping,
            temperature,
        )[0]  # [N] logprobs
        dest = self.tgt_rows.to(device) * self.L + self.tgt_cols.to(device)
        result = (
            torch.zeros(self.total_rows * self.L, dtype = torch.float32, device = device)
            .index_put((dest,), sel.to(torch.float32))
            .view(self.total_rows, self.L)[:, -self.W :]
        )
        return result


def _build_groups(ids_cpu, real_cols_cpu, cstart_cpu, num_generations, total_rows):
    """CPU-side grouping. Returns group dicts or None. Mirrors the packed _pk_* partition.

    A row's REAL tokens are the columns where input != pad. Its completion region (what
    the packed path scatters, then completion_mask masks) is the real columns with
    original col >= cstart_r, where cstart_r = (L - logits_to_keep) - left_pad_r. The
    prompt is the real columns < cstart_r. Within a GRPO group all G rows share the same
    prompt => same left_pad => same cstart => the prompt real columns are BYTE-IDENTICAL
    across the group (the shared prefix). We require that byte-identity (falls back
    otherwise). No prompt-tail special-casing: every suffix token is scattered exactly
    like the packed path; completion_mask masks the leading prompt-tail positions.
    """
    G = num_generations
    if G is None or G < 2 or total_rows % G != 0:
        return None
    groups = []
    for g0 in range(0, total_rows, G):
        rows = list(range(g0, g0 + G))
        prompt_cols_per_row = []  # real cols < cstart
        prompt_toks_per_row = []
        comp_cols_per_row = []  # real cols >= cstart  (the completion region packed scatters)
        for r in rows:
            cs = cstart_cpu[r]
            rc = real_cols_cpu[r]
            p_cols = [c for c in rc if c < cs]
            c_cols = [c for c in rc if c >= cs]
            prompt_cols_per_row.append(p_cols)
            prompt_toks_per_row.append([ids_cpu[r][c] for c in p_cols])
            comp_cols_per_row.append(c_cols)
        if any(len(p) == 0 for p in prompt_toks_per_row):
            return None
        # require BYTE-IDENTICAL prompts across the group (shared-prefix precondition).
        P = len(prompt_toks_per_row[0])
        if any(len(prompt_toks_per_row[k]) != P for k in range(1, G)):
            return None
        p0 = prompt_toks_per_row[0]
        if any(prompt_toks_per_row[k] != p0 for k in range(1, G)):
            return None
        if P == 0:
            return None
        R_list = [len(c) for c in comp_cols_per_row]
        if sum(R_list) == 0:
            return None
        groups.append(
            dict(
                rows = rows,
                P = P,
                prefix_cols = prompt_cols_per_row[0],  # shared prompt real columns (row0)
                prefix_row = rows[0],
                R_list = R_list,
                suf_cols = comp_cols_per_row,  # per-row completion-region real columns
            )
        )
    return groups


def _tok_r(groups) -> float:
    tok_full = 0
    tok_sp = 0
    for gm in groups:
        P = gm["P"]
        Rs = gm["R_list"]
        tok_full += sum(P + r for r in Rs)  # G*P + sumR
        tok_sp += P + sum(Rs)  # P + sumR
    return (tok_full / tok_sp) if tok_sp else 1.0


def build_group_layout(
    input_ids,
    logits_to_keep,
    pad_id,
    num_generations,
    left_pad_tokens_per_prompt,
    *,
    apply_tokr_gate = True,
    max_segment_cap = None,
):
    """Build the shared-prefix GroupLayout, or return None to fall back to the packed path.

    input_ids : [B, L]. GRPO's layout is left-padded in the prompt and right-padded in
        the completion. Real tokens of a row are a contiguous run not necessarily
        starting at column 0.
    logits_to_keep : int
    left_pad_tokens_per_prompt : [B] long tensor (per-row left-pad count in the prompt).
    """
    device = input_ids.device
    total_rows, L = input_ids.shape
    keep = input_ids != pad_id
    # completion start column per row (matches create_completion_attention_mask / _pk_cstart).
    cstart = ((L - logits_to_keep) - left_pad_tokens_per_prompt).to(torch.long)
    cstart_cpu = cstart.tolist()
    ids_cpu = input_ids.tolist()
    # per-row real (non-pad) columns. GRPO rows are one contiguous real run, so derive
    # [first, first+n) on GPU; the O(B*L) scan is only a non-contiguous fallback.
    n_real = keep.sum(dim = 1)
    first = torch.argmax(keep.to(torch.int8), dim = 1)
    ar = torch.arange(L, device = device)
    contiguous = bool(
        (keep == ((ar >= first.unsqueeze(1)) & (ar < (first + n_real).unsqueeze(1)))).all()
    )
    if contiguous:
        real_cols_cpu = [list(range(f, f + n)) for f, n in zip(first.tolist(), n_real.tolist())]
    else:
        keep_cpu = keep.tolist()
        real_cols_cpu = [[c for c in range(L) if keep_cpu[r][c]] for r in range(total_rows)]

    groups = _build_groups(ids_cpu, real_cols_cpu, cstart_cpu, num_generations, total_rows)
    if groups is None:
        return None

    # sliding-window guard: a group's PG span is P + max(R); fall back if it exceeds the window.
    if max_segment_cap is not None:
        for gm in groups:
            if gm["P"] + max(gm["R_list"]) > max_segment_cap:
                return None

    tok_r = _tok_r(groups)
    if apply_tokr_gate and tok_r < tokr_threshold():
        return None  # low reuse -> not worth it; use the full-row packed path

    # Build flat stream by gathering original (row, col) coordinates.
    group_specs = [(gm["P"], gm["R_list"]) for gm in groups]
    seg, group_meta = build_seg_info_multigroup(group_specs, device)

    flat_src_rows: List[int] = []
    flat_src_cols: List[int] = []
    pos_list: List[int] = []
    tgt_rows: List[int] = []
    tgt_cols: List[int] = []
    tgt_pred: List[int] = []
    tgt_flat: List[int] = []

    for gm, meta in zip(groups, group_meta):
        rows = gm["rows"]
        P = gm["P"]
        r0 = gm["prefix_row"]
        prefix_cols = gm["prefix_cols"]  # ORIGINAL real prompt columns (len P) of row0
        plast = meta["prefix_last_index"]  # base + P - 1
        # gather the shared prefix once, from row0.
        flat_src_rows.extend([r0] * P)
        flat_src_cols.extend(prefix_cols)
        pos_list.extend(range(P))
        # suffixes: every suffix token is a completion-region target (scattered like the
        # packed path; completion_mask hides prompt-tail positions).
        for i, r in enumerate(rows):
            cols = gm["suf_cols"][i]
            r_i = len(cols)
            s, e = meta["suffix_slices"][i]  # flat offsets [s, e)
            flat_src_rows.extend([r] * r_i)
            flat_src_cols.extend(cols)
            pos_list.extend(range(P, P + r_i))
            for j in range(r_i):
                # pos 0 is predicted from the prefix's last token; j>=1 from the previous suffix token.
                pred = plast if j == 0 else (s + j - 1)
                tgt_rows.append(r)
                tgt_cols.append(cols[j])  # ORIGINAL padded column in row r
                tgt_pred.append(pred)
                tgt_flat.append(s + j)  # flat index of the target token itself

    T = len(flat_src_rows)
    assert T == seg.T, f"flat stream len {T} != seg.T {seg.T}"
    fr = torch.tensor(flat_src_rows, device = device, dtype = torch.long)
    fc = torch.tensor(flat_src_cols, device = device, dtype = torch.long)
    flat_ids = input_ids[fr, fc].unsqueeze(0)  # [1, T] (grad-safe gather)
    position_ids = torch.tensor(pos_list, device = device, dtype = torch.long).unsqueeze(0)

    max_left_pad = int(left_pad_tokens_per_prompt.max().item()) if total_rows else 0
    W = logits_to_keep + max_left_pad

    # self-verify cache key: the mask/index-map/scatter logic is structural, so key on
    # (num_groups, group_sizes), not exact lengths -- GRPO lengths change every step and
    # keying on T would re-verify forever ("verify once, then trust", like the packed path).
    grp_sizes = tuple(sorted(len(gm["R_list"]) for gm in groups))
    sig = (len(groups), grp_sizes)

    return GroupLayout(
        flat_ids = flat_ids,
        position_ids = position_ids,
        prefix_seg_info = seg,
        tgt_rows = torch.tensor(tgt_rows, device = device, dtype = torch.long),
        tgt_cols = torch.tensor(tgt_cols, device = device, dtype = torch.long),
        tgt_pred = torch.tensor(tgt_pred, device = device, dtype = torch.long),
        tgt_flat = torch.tensor(tgt_flat, device = device, dtype = torch.long),
        total_rows = total_rows,
        L = L,
        W = W,
        tok_r = tok_r,
        signature = sig,
    )


__all__ = [
    "GroupLayout",
    "build_group_layout",
    "prefix_grouper_enabled",
    "verify_on",
    "tokr_threshold",
    "tol_ok",
    "TOL_KILL",
    "env_on",
]
