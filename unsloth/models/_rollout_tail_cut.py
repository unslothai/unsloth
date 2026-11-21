# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Opt-in GRPO rollout tail cut for colocated vLLM generation.

A colocated `llm.generate` call decodes until its longest row ends. Once every row still running belongs to a
group that already has `k_min` finished siblings, the remaining rows are stopped and returned as partial
completions (sufficiency-gated termination, TailRoll, arXiv 2611.22206). Cut rows are handled exactly like
length-truncated rows: they keep their reward in the group statistics and `mask_truncated_completions = True`
removes them from the loss. The cut only engages when that masking is on, so the objective never changes silently.

Off unless UNSLOTH_ROLLOUT_TAIL_CUT=<k_min> is set:
  UNSLOTH_ROLLOUT_TAIL_CUT_MAX_FRAC   most rows cut per generate call, as a fraction of its rows (default 0.125)
  UNSLOTH_ROLLOUT_TAIL_CUT_MIN_TOKENS a row is cut only after this many generated tokens (default 128)
"""

import contextlib
import os
from collections import Counter

__all__ = [
    "tail_cut_settings",
    "rollout_tail_cut_scope",
    "cut_row_flags",
    "TailCutEngine",
]

_WARNED = set()


def _warn_once(key, message):
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"Unsloth: rollout tail cut is off: {message}")


def tail_cut_settings():
    """(k_min, max_frac, min_tokens) from the environment, or None when the cut is off."""
    raw = os.environ.get("UNSLOTH_ROLLOUT_TAIL_CUT", "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return None
    try:
        k_min = int(raw)
        max_frac = float(os.environ.get("UNSLOTH_ROLLOUT_TAIL_CUT_MAX_FRAC", "0.125"))
        min_tokens = int(os.environ.get("UNSLOTH_ROLLOUT_TAIL_CUT_MIN_TOKENS", "128"))
    except ValueError:
        _warn_once("parse", "UNSLOTH_ROLLOUT_TAIL_CUT* must be numbers.")
        return None
    if k_min < 1 or not (0.0 < max_frac < 1.0) or min_tokens < 0:
        _warn_once("range", "needs k_min >= 1, 0 < MAX_FRAC < 1 and MIN_TOKENS >= 0.")
        return None
    return k_min, max_frac, min_tokens


def cut_row_flags(trainer, completion_ids):
    """Per row: True when the row is a tail-cut partial completion of the latest rollout."""
    rows = getattr(trainer, "_unsloth_tail_cut_rows", None)
    if not rows:
        return None
    remaining = Counter(rows)
    flags = []
    for ids in completion_ids:
        key = tuple(int(t) for t in ids)
        if remaining.get(key, 0) > 0:
            remaining[key] -= 1
            flags.append(True)
        else:
            flags.append(False)
    return flags


def _innermost(fn, name = "_generate_and_score_completions"):
    """Follow Unsloth's method wrappers (closures over the original method, or __wrapped__) to the source body."""
    for _ in range(8):
        inner = getattr(fn, "__wrapped__", None)
        if inner is None:
            for cell in getattr(fn, "__closure__", None) or ():
                try:
                    value = cell.cell_contents
                except ValueError:
                    continue
                if callable(value) and getattr(value, "__name__", "") == name and value is not fn:
                    inner = value
                    break
        if inner is None:
            return fn
        fn = inner
    return fn


def _masking_site_patched(trainer_class):
    """The trainer's truncation mask goes through _unsloth_grpo_is_truncated (TRL 0.24+ source shape)."""
    cached = trainer_class.__dict__.get("_unsloth_tail_cut_supported", None)
    if cached is not None:
        return cached
    import inspect

    supported = False
    fn = getattr(trainer_class, "_generate_and_score_completions", None)
    if fn is not None:
        try:
            source = inspect.getsource(_innermost(fn))
            supported = "_unsloth_grpo_is_truncated(self, completion_ids_list" in source
        except (OSError, TypeError):
            supported = False
    try:
        setattr(trainer_class, "_unsloth_tail_cut_supported", supported)
    except Exception:
        pass
    return supported


class TailCutEngine:
    """Wraps one vLLM LLMEngine's add_request / step for the duration of a rollout.

    Requests are asked for cumulative outputs, so the latest output of a running row is its partial completion
    (vLLM builds finished outputs the same way for FINAL_ONLY). Rows are grouped by arrival order, `group_size`
    consecutive requests per group, and a group is only trusted when its rows share one prompt."""

    def __init__(
        self,
        engine,
        group_size,
        k_min,
        max_frac,
        min_tokens,
        vllm_types = None,
    ):
        if vllm_types is None:
            from vllm.sampling_params import RequestOutputKind, SamplingParams
        else:
            RequestOutputKind, SamplingParams = vllm_types

        self.engine = engine
        self.group_size = int(group_size)
        self.k_min = int(k_min)
        self.max_frac = float(max_frac)
        self.min_tokens = int(min_tokens)
        self._cumulative = RequestOutputKind.CUMULATIVE
        self._sampling_params_type = SamplingParams
        self._params_cache = {}
        self.cut_rows = []  # completion token tuples of every cut row
        self.num_rows = 0
        self.num_cut = 0
        self._reset_batch()

    def _reset_batch(self):
        self.order = {}  # request id -> row index in this generate call
        self.running = set()
        self.latest = {}
        self.finished_per_group = Counter()
        self.group_prompt = {}
        self.checked = set()
        self.batch_ok = True
        self.batch_rows = 0
        self.batch_cut = 0

    # -- wrapped engine methods ----------------------------------------------------------------------------
    def add_request(self, original, *args, **kwargs):
        if not self.running:
            self._reset_batch()
        args = list(args)
        params = args[2] if len(args) > 2 else kwargs.get("params", None)
        if isinstance(params, self._sampling_params_type) and getattr(params, "n", 1) == 1:
            clone = self._params_cache.get(id(params))
            if clone is None or clone[0] is not params:
                new = params.clone()
                new.output_kind = self._cumulative
                clone = (params, new)
                self._params_cache[id(params)] = clone
            if len(args) > 2:
                args[2] = clone[1]
            else:
                kwargs["params"] = clone[1]
        else:
            # n > 1 or pooling: one request holds a whole group or is not generation; never cut this call.
            self.batch_ok = False
        request_id = args[0] if args else kwargs.get("request_id")
        out = original(*args, **kwargs)
        rid = str(request_id)
        self.order[rid] = self.batch_rows
        self.running.add(rid)
        self.batch_rows += 1
        self.num_rows += 1
        return out

    def step(self, original):
        outputs = original()
        returned = []
        for output in outputs:
            rid = str(getattr(output, "request_id", ""))
            if rid not in self.order:
                returned.append(output)
                continue
            if rid not in self.running:
                continue  # already returned as a cut row
            self._check_prompt(rid, output)
            if output.finished:
                self.running.discard(rid)
                self.latest.pop(rid, None)
                self.finished_per_group[self.order[rid] // self.group_size] += 1
                returned.append(output)
            else:
                self.latest[rid] = output
        if self.running:
            returned.extend(self._maybe_cut())
        return returned

    # -- decisions --------------------------------------------------------------------------------------
    def _check_prompt(self, rid, output):
        if rid in self.checked:
            return
        self.checked.add(rid)
        prompt = getattr(output, "prompt_token_ids", None)
        if prompt is None:
            return
        group = self.order[rid] // self.group_size
        key = tuple(prompt)
        seen = self.group_prompt.setdefault(group, key)
        if seen != key:
            self.batch_ok = False

    def _maybe_cut(self):
        if not self.batch_ok or self.batch_rows % self.group_size != 0:
            return []
        budget = int(self.max_frac * self.batch_rows) - self.batch_cut
        running = self.running
        if len(running) > budget:
            return []
        for rid in running:
            output = self.latest.get(rid, None)
            if output is None or len(output.outputs) != 1:
                return []
            if len(output.outputs[0].token_ids) < self.min_tokens:
                return []
            if self.finished_per_group[self.order[rid] // self.group_size] < self.k_min:
                return []
        cut_ids = sorted(running, key = lambda r: self.order[r])
        self.engine.abort_request(cut_ids)
        cut_outputs = []
        for rid in cut_ids:
            output = self.latest.pop(rid)
            completion = output.outputs[0]
            completion.token_ids = list(completion.token_ids)
            if completion.logprobs is not None:
                completion.logprobs = completion.logprobs[: len(completion.token_ids)]
            completion.finish_reason = "abort"
            completion.stop_reason = None
            output.finished = True
            self.cut_rows.append(tuple(int(t) for t in completion.token_ids))
            cut_outputs.append(output)
        self.running.clear()
        self.batch_cut += len(cut_ids)
        self.num_cut += len(cut_ids)
        return cut_outputs


def _engine_of(trainer):
    llm = getattr(getattr(trainer, "vllm_generation", None), "llm", None)
    if llm is None:
        llm = getattr(trainer, "llm", None)
    return getattr(llm, "llm_engine", None)


def _unsupported_reason(trainer):
    if not getattr(trainer, "use_vllm", False):
        return "needs use_vllm / fast_inference."
    if getattr(trainer, "vllm_mode", "colocate") != "colocate":
        return "needs vllm_mode='colocate'."
    if not getattr(trainer, "mask_truncated_completions", False):
        return "needs mask_truncated_completions=True so cut rows are masked out of the loss."
    if getattr(trainer, "tools", None) or getattr(trainer, "rollout_func", None) is not None:
        return "tool calling and rollout_func are not supported."
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and getattr(accelerator, "num_processes", 1) != 1:
        return "a GRPO group can span processes; single process only."
    if int(getattr(getattr(trainer, "args", None), "vllm_tensor_parallel_size", 1) or 1) != 1:
        return "vllm_tensor_parallel_size must be 1."
    if os.environ.get("UNSLOTH_GRPO_ROUTING_REPLAY", "0") == "1":
        return "routing replay needs the full routed experts of every row."
    if not _masking_site_patched(type(trainer)):
        return "this TRL version's truncation mask is not routed through Unsloth."
    if _engine_of(trainer) is None:
        return "no vLLM LLMEngine found."
    return None


@contextlib.contextmanager
def rollout_tail_cut_scope(trainer):
    """Install the tail cut on the colocated engine for one `_generate` call (training mode only)."""
    settings = tail_cut_settings()
    trainer._unsloth_tail_cut_rows = None
    model = getattr(trainer, "model", None)
    if settings is None or (model is not None and not getattr(model, "training", True)):
        yield
        return
    reason = _unsupported_reason(trainer)
    if reason is not None:
        _warn_once(reason, reason)
        yield
        return
    k_min, max_frac, min_tokens = settings
    group_size = int(getattr(trainer, "num_generations", 0) or 0)
    if group_size < 2 or k_min >= group_size:
        _warn_once("k_min", f"k_min = {k_min} must be below num_generations = {group_size}.")
        yield
        return
    engine = _engine_of(trainer)
    state = TailCutEngine(engine, group_size, k_min, max_frac, min_tokens)
    # The same list grows as rows are cut, so the truncation check inside `_generate` (clipped_ratio) sees them too.
    trainer._unsloth_tail_cut_rows = state.cut_rows
    original_add = engine.add_request
    original_step = engine.step
    had_add = "add_request" in engine.__dict__
    had_step = "step" in engine.__dict__
    engine.add_request = lambda *a, **k: state.add_request(original_add, *a, **k)
    engine.step = lambda: state.step(original_step)
    try:
        yield
    finally:
        if had_add:
            engine.add_request = original_add
        else:
            del engine.add_request
        if had_step:
            engine.step = original_step
        else:
            del engine.step
        metrics = getattr(trainer, "_metrics", None)
        if state.num_rows and isinstance(metrics, dict) and "train" in metrics:
            try:
                metrics["train"]["completions/tail_cut_ratio"].append(
                    state.num_cut / state.num_rows
                )
            except Exception:
                pass
