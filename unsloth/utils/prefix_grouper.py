# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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

"""GRPO PrefixGrouper: each group of ``num_generations`` rows with a byte-identical prompt becomes
one flat stream ``[prefix_g, suffix_g_0 .. suffix_g_{G-1}, ...]`` gathered from the original
(row, col) coordinates, so the shared prefix gets grad once (= sum of the G repeats). Completion
logprobs are scattered back to ``orig_row * L + orig_col``, byte-compatible with the packed path.

Env: UNSLOTH_GRPO_PREFIX_GROUPER (default 1, off under vLLM), _TOKR (1.3), _VERIFY (1),
_TOL (0.7 nats), _COMPILE_CHECK (1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .prefix_grouper_kernel import (
    build_seg_info_multigroup,
    compile_check_failed,
    PrefixSegInfo,
    PREFIX_GROUPER_MIN_TORCH,
    TORCH_SUPPORTS_PREFIX_GROUPER,
)


def env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() not in ("0", "false", "no", "off")


# One-time env reads; the helpers stay callable, since unsloth_zoo imports and calls them.
_ENV_ENABLED = env_on("UNSLOTH_GRPO_SEQ_PACKING", "1") and env_on(
    "UNSLOTH_GRPO_PREFIX_GROUPER", "1"
)
_ENABLED = _ENV_ENABLED and TORCH_SUPPORTS_PREFIX_GROUPER
_OFF_LOGGED = False
_VERIFY_ON = env_on("UNSLOTH_GRPO_PREFIX_GROUPER_VERIFY", "1")
_TOKR_THRESHOLD = float(os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER_TOKR", "1.3"))
_TOL_OK = float(os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER_TOL", "0.7"))


def prefix_grouper_enabled() -> bool:
    """Needs seq-packing, torch >= 2.7 and a flex kernel that passed its compile check."""
    global _OFF_LOGGED
    if _ENV_ENABLED and not _ENABLED and not _OFF_LOGGED:
        _OFF_LOGGED = True
        print(
            f"Unsloth: GRPO PrefixGrouper is off on torch {torch.__version__} (needs torch >= "
            f"{PREFIX_GROUPER_MIN_TORCH[0]}.{PREFIX_GROUPER_MIN_TORCH[1]}: its flex attention kernel "
            "does not compile there); using the packed path.",
            flush = True,
        )
    return _ENABLED and compile_check_failed() is None


def verify_on() -> bool:
    return _VERIFY_ON


def tokr_threshold() -> float:
    return _TOKR_THRESHOLD


def tol_ok() -> float:
    return _TOL_OK


# TOL_KILL or above means broken mask or isolation: unsafe forever. Between tol_ok and TOL_KILL, fall back for this shape only.
TOL_KILL = 1.5


@dataclass
class GroupLayout:
    """Everything the GRPO forward needs to run + extract the shared-prefix path."""

    flat_ids: torch.Tensor  # [1, T]  (T == seg.T)
    position_ids: torch.Tensor  # [1, T]
    prefix_seg_info: PrefixSegInfo
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
        # In a sharded model hidden may live on the lm-head device.
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
    """CPU-side grouping mirroring the packed _pk_* partition, or None.

    Prompt = real columns < cstart_r = (L - logits_to_keep) - left_pad_r; every row of a group
    must have a byte-identical prompt. Suffix tokens scatter exactly like the packed path.
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


def _build_group_layout_reference(
    input_ids,
    logits_to_keep,
    pad_id,
    num_generations,
    left_pad_tokens_per_prompt,
    *,
    apply_tokr_gate = True,
    max_segment_cap = None,
):
    """Token-by-token shared-prefix GroupLayout (build_group_layout's fallback), or None.

    input_ids [B, L]: prompt left-padded, completion right-padded.
    """
    device = input_ids.device
    total_rows, L = input_ids.shape
    keep = input_ids != pad_id
    # Completion start column per row, matching create_completion_attention_mask / _pk_cstart.
    cstart = ((L - logits_to_keep) - left_pad_tokens_per_prompt).to(device = device, dtype = torch.long)
    # GRPO rows are one contiguous real run; the O(B*L) scan is only a fallback.
    n_real = keep.sum(dim = 1)
    first = torch.argmax(keep.to(torch.int8), dim = 1)
    ar = torch.arange(L, device = device)
    contiguous = (keep == ((ar >= first.unsqueeze(1)) & (ar < (first + n_real).unsqueeze(1)))).all()
    # One device-to-host read for everything below.
    host = torch.cat(
        [
            input_ids.reshape(-1).to(torch.long),
            cstart.reshape(-1),
            first.to(torch.long),
            n_real.to(torch.long),
            contiguous.to(torch.long).reshape(1),
        ]
    ).tolist()
    n_ids = total_rows * L
    ids_cpu = [host[r * L : (r + 1) * L] for r in range(total_rows)]
    cstart_cpu = host[n_ids : n_ids + total_rows]
    first_cpu = host[n_ids + total_rows : n_ids + 2 * total_rows]
    n_real_cpu = host[n_ids + 2 * total_rows : n_ids + 3 * total_rows]
    if host[-1]:
        real_cols_cpu = [list(range(f, f + n)) for f, n in zip(first_cpu, n_real_cpu)]
    else:
        real_cols_cpu = [
            [c for c in range(L) if ids_cpu[r][c] != pad_id] for r in range(total_rows)
        ]

    groups = _build_groups(ids_cpu, real_cols_cpu, cstart_cpu, num_generations, total_rows)
    if groups is None:
        return None

    # Sliding-window guard: a group's PG span is P + max(R), so fall back if it exceeds the window.
    if max_segment_cap is not None:
        for gm in groups:
            if gm["P"] + max(gm["R_list"]) > max_segment_cap:
                return None

    tok_r = _tok_r(groups)
    if apply_tokr_gate and tok_r < tokr_threshold():
        return None  # low reuse -> not worth it; use the full-row packed path

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
        flat_src_rows.extend([r0] * P)
        flat_src_cols.extend(prefix_cols)
        pos_list.extend(range(P))
        for i, r in enumerate(rows):
            cols = gm["suf_cols"][i]
            r_i = len(cols)
            s, e = meta["suffix_slices"][i]  # flat offsets [s, e)
            flat_src_rows.extend([r] * r_i)
            flat_src_cols.extend(cols)
            pos_list.extend(range(P, P + r_i))
            for j in range(r_i):
                # Position 0 is predicted from the prefix's last token; j >= 1 from the previous suffix token.
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

    # cstart = (L - logits_to_keep) - left_pad, so max left pad comes from the host copy.
    max_left_pad = ((L - logits_to_keep) - min(cstart_cpu)) if total_rows else 0
    W = logits_to_keep + max_left_pad

    # The logic is structural, so key on (num_groups, group_sizes), not exact lengths.
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


_INDEX_DTYPES = (torch.int64, torch.int32)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


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

    Same result as _build_group_layout_reference, with one device->host and one
    host->device copy instead of per-token loops.
    """
    args = (input_ids, logits_to_keep, pad_id, num_generations, left_pad_tokens_per_prompt)
    kwargs = dict(apply_tokr_gate = apply_tokr_gate, max_segment_cap = max_segment_cap)
    lp = left_pad_tokens_per_prompt
    if not (
        isinstance(input_ids, torch.Tensor)
        and input_ids.dim() == 2
        and input_ids.shape[0] > 0
        and input_ids.dtype in _INDEX_DTYPES
        and isinstance(lp, torch.Tensor)
        and lp.dim() == 1
        and lp.shape[0] == input_ids.shape[0]
        and lp.dtype in _INDEX_DTYPES
        and lp.device == input_ids.device
        and _is_int(logits_to_keep)
        and _is_int(pad_id)
        and (num_generations is None or _is_int(num_generations))
        and torch.iinfo(input_ids.dtype).min <= pad_id <= torch.iinfo(input_ids.dtype).max
    ):
        return _build_group_layout_reference(*args, **kwargs)
    device = input_ids.device
    total_rows, L = input_ids.shape
    G = num_generations
    # Completion start column per row, matching create_completion_attention_mask / _pk_cstart.
    cstart = ((L - logits_to_keep) - lp).to(torch.long)
    host = torch.cat(
        [input_ids.to(torch.long), cstart.unsqueeze(1), lp.to(torch.long).unsqueeze(1)], dim = 1
    ).cpu()
    ids = host[:, :L]
    cs = host[:, L]
    keep = ids != pad_id
    n_real = keep.sum(dim = 1)
    first = torch.argmax(keep.to(torch.int8), dim = 1)
    ar = torch.arange(L, device = "cpu")
    end = first + n_real
    if not bool((keep == ((ar >= first.unsqueeze(1)) & (ar < end.unsqueeze(1)))).all()):
        return _build_group_layout_reference(*args, **kwargs)
    if G is None or G < 2 or total_rows % G != 0:
        return None
    # Prompt real columns [first, pe), completion [pe, end).
    pe = torch.minimum(torch.maximum(cs, first), end)
    P_row = pe - first
    R_row = end - pe
    num_groups = total_rows // G
    P_g = P_row.view(num_groups, G)
    if bool((P_g == 0).any()) or bool((P_g != P_g[:, :1]).any()):
        return None
    R_g = R_row.view(num_groups, G)
    if bool((R_g.sum(dim = 1) == 0).any()):
        return None
    first_g = first.view(num_groups, G)
    P_list = P_g[:, 0].tolist()
    for gi, P in enumerate(P_list):
        cols = first_g[gi].unsqueeze(1) + torch.arange(P, device = "cpu")
        prompts = ids[gi * G : (gi + 1) * G].gather(1, cols)
        if not bool((prompts == prompts[:1]).all()):
            return None
    R_lists = R_g.tolist()
    groups = [dict(P = P, R_list = R_list) for P, R_list in zip(P_list, R_lists)]

    # Sliding-window guard: a group's PG span is P + max(R), so fall back if it exceeds the window.
    if max_segment_cap is not None:
        for gm in groups:
            if gm["P"] + max(gm["R_list"]) > max_segment_cap:
                return None

    tok_r = _tok_r(groups)
    if apply_tokr_gate and tok_r < tokr_threshold():
        return None  # low reuse -> not worth it; use the full-row packed path

    from .prefix_grouper_kernel import _labels_to_device, _seg_labels_host

    group_specs = [(gm["P"], gm["R_list"]) for gm in groups]
    labels, T, T_pad, _, sig_parts = _seg_labels_host(group_specs)

    # Flat order per group: shared prefix (from its first row), then one suffix per row.
    rows_g = torch.arange(total_rows, device = "cpu").view(num_groups, G)
    P_t = P_g[:, :1]
    seg_len = torch.cat([P_t, R_g], dim = 1).reshape(-1)
    seg_row = torch.cat([rows_g[:, :1], rows_g], dim = 1).reshape(-1)
    seg_col = torch.cat([first_g[:, :1], pe.view(num_groups, G)], dim = 1).reshape(-1)
    seg_pos = torch.cat([torch.zeros_like(P_t), P_t.expand(num_groups, G)], dim = 1).reshape(-1)
    seg_is_suffix = torch.ones((num_groups, G + 1), dtype = torch.bool, device = "cpu")
    seg_is_suffix[:, 0] = False
    seg_is_suffix = seg_is_suffix.reshape(-1)
    seg_start = torch.cumsum(seg_len, dim = 0) - seg_len
    group_base = seg_start.view(num_groups, G + 1)[:, :1]
    seg_plast = (group_base + P_t - 1).expand(num_groups, G + 1).reshape(-1)

    flat = torch.arange(T, device = "cpu")
    j = flat - torch.repeat_interleave(seg_start, seg_len, output_size = T)
    flat_src_rows = torch.repeat_interleave(seg_row, seg_len, output_size = T)
    flat_src_cols = torch.repeat_interleave(seg_col, seg_len, output_size = T) + j
    pos = torch.repeat_interleave(seg_pos, seg_len, output_size = T) + j
    is_tgt = torch.repeat_interleave(seg_is_suffix, seg_len, output_size = T)
    tgt_flat = flat[is_tgt]
    tgt_j = j[is_tgt]
    # Position 0 is predicted from the prefix's last token; j >= 1 from the previous suffix token.
    tgt_pred = torch.where(
        tgt_j == 0,
        torch.repeat_interleave(seg_plast, seg_len, output_size = T)[is_tgt],
        tgt_flat - 1,
    )
    N = int(tgt_flat.numel())
    table = torch.cat(
        [
            flat_src_rows,
            flat_src_cols,
            pos,
            flat_src_rows[is_tgt],
            flat_src_cols[is_tgt],
            tgt_pred,
            tgt_flat,
            labels.reshape(-1),
        ]
    ).to(device)
    pieces = torch.split(table, [T, T, T, N, N, N, N, 3 * T_pad])
    fr, fc, position_ids, tgt_rows, tgt_cols, tgt_pred_d, tgt_flat_d, labels_d = pieces
    group_of_kv, is_prefix, suffix_of_kv = _labels_to_device(labels_d.view(3, T_pad), T_pad, device)
    seg = PrefixSegInfo(
        group_of_kv = group_of_kv,
        is_prefix = is_prefix,
        suffix_of_kv = suffix_of_kv,
        signature = ("multi", T_pad, tuple(sig_parts)),
        T = T,
        T_pad = T_pad,
    )
    flat_ids = input_ids[fr, fc].unsqueeze(0)  # [1, T] (grad-safe gather)

    max_left_pad = int(host[:, L + 1].max().item())
    W = logits_to_keep + max_left_pad

    # The logic is structural, so key on (num_groups, group_sizes), not exact lengths.
    sig = (num_groups, tuple(sorted(len(gm["R_list"]) for gm in groups)))

    return GroupLayout(
        flat_ids = flat_ids,
        position_ids = position_ids.clone().unsqueeze(0),
        prefix_seg_info = seg,
        tgt_rows = tgt_rows.clone(),
        tgt_cols = tgt_cols.clone(),
        tgt_pred = tgt_pred_d.clone(),
        tgt_flat = tgt_flat_d.clone(),
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
