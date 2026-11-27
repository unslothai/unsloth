# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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

"""Opt-in partial rollouts for colocated vLLM GRPO (UNSLOTH_GRPO_PARTIAL_ROLLOUT=<spare fraction>).

A colocated rollout waits for its longest row. With partial rollouts the engine holds E spare groups besides the B
groups of the step (E = ceil(fraction * B)). Generation stops as soon as all but E groups are complete; those train,
and every unfinished row is aborted and carried, with its tokens and its sampler logprobs, into the next step. There
it is re-prefilled from prompt + generated tokens (the KV cache is not kept, so it works across a standby sleep) and
continues under the new weights for its remaining token budget.

Group integrity: a group is held until every member has finished, so a trained group always has all its rewards and
its advantages are computed exactly as for a fresh rollout. Carried tokens keep the logprobs of the policy that
sampled them; the vLLM importance-sampling correction (required) weights them against the current policy, as it does
for AsyncGRPO. Each carried token records its policy version (logged as partial_rollout/token_staleness_mean). A group
may be carried at most UNSLOTH_GRPO_PARTIAL_ROLLOUT_MAX_AGE steps (default 2); older groups must finish first.

The first step trains B - E groups (the spare pool starts filling); every later step trains B. At the end of training
the E carried groups are not trained. The carry lives in memory: a checkpoint resume starts with an empty pool.
"""

import functools
import inspect
import math
import os

__all__ = [
    "partial_rollout_settings",
    "PartialRolloutEngine",
    "PartialRolloutState",
    "wrap_grpo_partial_rollout",
]

_WARNED = set()
_OFF = ("", "0", "false", "no", "off")


def _warn_once(key, message):
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"Unsloth: GRPO partial rollout is off: {message}")


def partial_rollout_settings():
    """(spare_fraction, max_age) from the environment, or None when partial rollouts are off."""
    raw = os.environ.get("UNSLOTH_GRPO_PARTIAL_ROLLOUT", "").strip().lower()
    if raw in _OFF:
        return None
    try:
        fraction = 0.25 if raw in ("1", "true", "yes", "on") else float(raw)
        max_age = int(os.environ.get("UNSLOTH_GRPO_PARTIAL_ROLLOUT_MAX_AGE", "2"))
    except ValueError:
        _warn_once("parse", "UNSLOTH_GRPO_PARTIAL_ROLLOUT* must be numbers.")
        return None
    if not (0.0 < fraction < 1.0) or max_age < 1:
        _warn_once("range", "needs 0 < UNSLOTH_GRPO_PARTIAL_ROLLOUT < 1 and MAX_AGE >= 1.")
        return None
    return fraction, max_age


class _Row:
    __slots__ = ("input", "prompt_ids", "tokens", "logprobs", "versions", "done")

    def __init__(self, example):
        self.input = example
        self.prompt_ids = None
        self.tokens = []
        self.logprobs = []
        self.versions = []
        self.done = False


class _Group:
    __slots__ = ("rows", "age", "serial")

    def __init__(self, rows, serial):
        self.rows = rows
        self.age = 0
        self.serial = serial

    @property
    def complete(self):
        return all(row.done for row in self.rows)


class PartialRolloutState:
    """The carried groups of one trainer plus counters."""

    def __init__(self, run = None):
        self.carry = []
        self.serial = 0
        self.steps = 0
        self.aborted_rows = 0
        self.run = run  # id of the TrainerState this carry belongs to


class PartialRolloutEngine:
    """Wraps one vLLM LLMEngine's add_request / step for one generate call of the partial rollout.

    rows are added in order; row i belongs to group row_group[i] and resumes after resume_len[i] generated tokens.
    The call stops once `target` groups are complete (complete_before of them had nothing left to generate) and no
    group in `must_finish` is still running; every running row is then aborted and returned as a finished partial
    output built by vLLM itself (the same final output an abort makes for a streaming client)."""

    def __init__(
        self,
        engine,
        row_group,
        resume_len,
        target,
        complete_before = 0,
        must_finish = (),
        finish_reason = None,
    ):
        self.engine = engine
        self.row_group = list(row_group)
        self.resume_len = list(resume_len)
        self.target = int(target)
        self.complete = int(complete_before)
        self.must_finish = set(must_finish)
        self.need = {}
        for group in self.row_group:
            self.need[group] = self.need.get(group, 0) + 1
        self.order = {}
        self.running = set()
        self.aborted = set()
        self.num_added = 0
        self.can_cut = True
        self.cut_error = None
        self._finish_reason = finish_reason

    def _params_supported(self, params):
        if getattr(params, "n", 1) != 1:
            return "n > 1 requests"
        if getattr(params, "stop", None):
            return "stop strings"
        if float(getattr(params, "presence_penalty", 0.0) or 0.0) != 0.0:
            return "presence_penalty"
        if float(getattr(params, "frequency_penalty", 0.0) or 0.0) != 0.0:
            return "frequency_penalty"
        # Each of these depends on the output history or on the request's own RNG stream, which a resubmitted
        # request (old output moved into the prompt) does not keep.
        for name in (
            "structured_outputs",
            "guided_decoding",
            "logits_processors",
            "bad_words",
            "repetition_detection",
        ):
            if getattr(params, name, None):
                return name
        for name in (
            "seed",
            "truncate_prompt_tokens",
            "thinking_token_budget",
            "trace_decode_token_ids",
            "prompt_logprobs",
            "prompt_logprob_token_ids",
        ):
            if getattr(params, name, None) is not None:
                return name
        if getattr(params, "watermarking", False):
            return "watermarking"
        config = getattr(getattr(self.engine, "vllm_config", None), "watermark_config", None)
        if config is not None and float(getattr(params, "temperature", 1.0) or 0.0) > 0.0:
            return "watermarking"
        return None

    def add_request(self, original, *args, **kwargs):
        args = list(args)
        params = args[2] if len(args) > 2 else kwargs.get("params", None)
        prompt = args[1] if len(args) > 1 else kwargs.get("prompt", None)
        # Older vLLM hands add_request a prepared EngineCoreRequest that carries its own sampling params.
        prepared = getattr(prompt, "sampling_params", None)
        row = self.num_added
        if row >= len(self.row_group):
            self.can_cut = False
            self.cut_error = "more requests than rows"
        elif self.can_cut:
            reason = self._params_supported(params)
            if reason is None and prepared is not None:
                reason = self._params_supported(prepared)
            if reason is not None:
                self.can_cut = False
                self.cut_error = reason
        resumed = row < len(self.resume_len) and self.resume_len[row] > 0
        if resumed:
            if self.cut_error is not None:
                raise RuntimeError(
                    f"Unsloth: GRPO partial rollout cannot resume a carried row ({self.cut_error})."
                )
            done = self.resume_len[row]
            new = (prepared if prepared is not None else params).clone()
            requested = getattr(params, "max_tokens", None)
            if requested is not None:
                new.max_tokens = int(requested) - done
            # requested None: vLLM derives the budget from the (now longer) prompt, a prepared request already has.
            if getattr(new, "min_tokens", 0):
                new.min_tokens = max(0, int(new.min_tokens) - done)
            if prepared is not None:
                prompt.sampling_params = new
            if len(args) > 2:
                args[2] = new
            else:
                kwargs["params"] = new
        request_id = args[0] if args else kwargs.get("request_id")
        out = original(*args, **kwargs)
        rid = str(request_id)
        self.order[rid] = row
        self.running.add(rid)
        self.num_added += 1
        return out

    def step(self, original):
        outputs = original()
        for output in outputs:
            rid = str(getattr(output, "request_id", ""))
            if rid not in self.running or not getattr(output, "finished", False):
                continue
            self.running.discard(rid)
            group = self.row_group[self.order[rid]]
            self.need[group] -= 1
            if self.need[group] == 0:
                self.complete += 1
        if self.running and self.can_cut and self._should_stop():
            outputs = list(outputs) + self._cut()
        return outputs

    def _should_stop(self):
        if self.num_added != len(self.row_group) or self.complete < self.target:
            return False
        return not any(self.need.get(group, 0) > 0 for group in self.must_finish)

    def _abort_output(self, processor, rid):
        mapping = getattr(processor, "external_req_ids", None)
        internal = list(mapping.get(rid, ())) if mapping is not None else [rid]
        if len(internal) != 1:
            raise RuntimeError(f"request {rid} maps to {len(internal)} engine requests")
        state = processor.request_states[internal[0]]
        output = state.make_request_output(
            new_token_ids = [],
            pooling_output = None,
            finish_reason = self._finish_reason,
            stop_reason = None,
        )
        if output is None or not getattr(output, "finished", False):
            raise RuntimeError("vLLM built no final output for an aborted request")
        return output

    def _cut(self):
        if self._finish_reason is None:
            try:
                self._finish_reason = _abort_finish_reason()
            except Exception as error:
                self.can_cut, self.cut_error = False, repr(error)
                return []
        processor = getattr(self.engine, "output_processor", None)
        ids = sorted(self.running, key = lambda r: self.order[r])
        try:
            if processor is None:
                raise RuntimeError("the engine has no output_processor")
            cut = [self._abort_output(processor, rid) for rid in ids]
        except Exception as error:
            # vLLM internals moved: never cut, so every row runs to its end and nothing is lost.
            self.can_cut, self.cut_error = False, repr(error)
            _warn_once(
                "abort",
                f"cannot build partial outputs on this vLLM ({error}); rows run to the end.",
            )
            return []
        self.engine.abort_request(ids)
        self.running.clear()
        self.aborted.update(ids)
        return cut

    def aborted_rows(self):
        return {self.order[rid] for rid in self.aborted}


def _abort_finish_reason():
    from vllm.v1.engine import FinishReason
    return FinishReason.ABORT


def _engine_of(trainer):
    llm = getattr(getattr(trainer, "vllm_generation", None), "llm", None)
    if llm is None:
        llm = getattr(trainer, "llm", None)
    return getattr(llm, "llm_engine", None)


def _single_turn_parameters(trainer_class):
    fn = getattr(trainer_class, "_generate_single_turn", None)
    if fn is None or not hasattr(trainer_class, "_tokenize_prompts"):
        return None
    try:
        names = list(inspect.signature(fn).parameters)[1:]
    except (TypeError, ValueError):
        return None
    if names[:3] != ["prompt_ids", "images", "multimodal_fields"]:
        return None
    return names


def _unsupported_reason(trainer):
    if not getattr(trainer, "use_vllm", False):
        return "needs use_vllm / fast_inference."
    if getattr(trainer, "vllm_mode", "colocate") != "colocate":
        return "needs vllm_mode='colocate'."
    if _single_turn_parameters(type(trainer)) is None:
        return "this TRL version has no _tokenize_prompts / _generate_single_turn(prompt_ids, images, ...)."
    if getattr(trainer, "tools", None) or getattr(trainer, "rollout_func", None) is not None:
        return "tool calling and rollout_func are not supported."
    if getattr(trainer, "environment_factories", None) is not None or getattr(
        trainer, "environments", None
    ):
        return "environments are not supported."
    if getattr(trainer, "_is_vlm", False):
        return "vision models are not supported."
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and getattr(accelerator, "num_processes", 1) != 1:
        return "a GRPO group can span processes; single process only."
    args = getattr(trainer, "args", None)
    if int(getattr(args, "vllm_tensor_parallel_size", 1) or 1) != 1:
        return "vllm_tensor_parallel_size must be 1."
    if not getattr(
        trainer,
        "vllm_importance_sampling_correction",
        getattr(args, "vllm_importance_sampling_correction", False),
    ):
        return "needs vllm_importance_sampling_correction=True to weight carried tokens."
    if (
        getattr(getattr(trainer, "vllm_generation", None), "return_sampling_mask", False)
        or getattr(args, "vllm_return_sampling_mask", False)
        or os.environ.get("UNSLOTH_GRPO_TRUNCATED_SUPPORT", "0") == "1"
    ):
        return "sampling-support masks are not carried."
    if os.environ.get("UNSLOTH_GRPO_ROUTING_REPLAY", "0") == "1":
        return "routing replay needs the routed experts of every token in one rollout."
    tail_cut = os.environ.get("UNSLOTH_ROLLOUT_TAIL_CUT", "").strip().lower()
    if tail_cut not in _OFF:
        return "UNSLOTH_ROLLOUT_TAIL_CUT is on; use one of the two."
    if int(getattr(trainer, "num_generations", 0) or 0) < 2:
        return "needs num_generations >= 2."
    if _engine_of(trainer) is None:
        return "no vLLM LLMEngine found."
    return None


def _configure_rollout_logprobs(trainer):
    """What the generated _generate_and_score_completions prologue does first: pick the vLLM logprob convention
    (pre top_k / top_p / min_p) for this trainer. The partial rollout generates before that prologue runs, so a
    carried row's first tokens would otherwise be logged under the other convention."""
    try:
        from unsloth_zoo.vllm_utils import set_vllm_untruncated_logprobs
    except Exception:
        return
    llm = getattr(trainer, "llm", None) or getattr(
        getattr(trainer, "vllm_generation", None), "llm", None
    )
    try:
        trainer._unsloth_vllm_untruncated_logprobs = bool(
            set_vllm_untruncated_logprobs(
                bool(getattr(trainer, "use_vllm", False))
                and bool(getattr(trainer, "vllm_importance_sampling_correction", False))
                and llm is not None,
                llm,
            )
        )
    except Exception:
        trainer._unsloth_vllm_untruncated_logprobs = False


def _kept_set_reason(trainer):
    """Kept-set scoring (zoo's automatic top_k / top_p / min_p support) would score a carried token over the new
    policy's kept set, which need not contain it."""
    try:
        from unsloth_zoo.rl_replacements import grpo_truncated_support_kwargs
        support = grpo_truncated_support_kwargs(trainer) or {}
    except Exception:
        return None
    if (
        int(support.get("support_top_k", 0) or 0) > 0
        or float(support.get("support_top_p", 1.0)) < 1.0
        or float(support.get("support_min_p", 0.0) or 0.0) > 0.0
    ):
        return (
            "kept-set (truncated support) scoring is on; carried tokens cannot be scored over it."
        )
    return None


def _groups_of(inputs, group_size):
    if group_size < 2 or len(inputs) % group_size:
        return None
    groups = []
    for start in range(0, len(inputs), group_size):
        rows = inputs[start : start + group_size]
        prompt = rows[0].get("prompt", None)
        if prompt is None or any(row.get("prompt", None) != prompt for row in rows[1:]):
            return None
        if any(("image" in row) or ("images" in row) for row in rows):
            return None
        groups.append(rows)
    return groups


def _unit(group_size, steps_per_generation):
    """Smallest number of groups whose rows split evenly over steps_per_generation micro-batches."""
    steps = max(int(steps_per_generation or 1), 1)
    return steps // math.gcd(group_size, steps)


def _spare_groups(fraction, num_groups, unit):
    spare = unit * math.ceil(fraction * num_groups / unit)
    if spare >= num_groups:
        spare = num_groups - unit
    return spare


def _call_single_turn(trainer, prompt_ids, images, multimodal_fields):
    names = _single_turn_parameters(type(trainer))
    kwargs = {}
    if "num_generations" in names:
        kwargs["num_generations"] = trainer.num_generations
    return trainer._generate_single_turn(prompt_ids, images, multimodal_fields, **kwargs)


def _generate_rows(trainer, state, groups, target, max_age):
    """Stage 1: one real generate call over every unfinished row. Returns the rows that were aborted."""
    pending = [(group, row) for group in groups for row in group.rows if not row.done]
    prompts = [row.input["prompt"] for _, row in pending]
    try:  # the same cleanup the generated scorer applies before it tokenizes
        from .rl_replacements import _unsloth_grpo_clean_video_prompts
        prompts = _unsloth_grpo_clean_video_prompts(prompts, trainer)
    except ImportError:
        pass
    prompt_ids, images, multimodal_fields = trainer._tokenize_prompts(prompts)
    if images is not None:
        raise RuntimeError("Unsloth: GRPO partial rollout got images from a text-only batch.")
    request_ids = []
    for (_, row), ids in zip(pending, prompt_ids):
        ids = list(ids)
        if row.prompt_ids is None:
            row.prompt_ids = ids
        elif row.prompt_ids != ids:
            raise RuntimeError(
                "Unsloth: GRPO partial rollout: a carried prompt re-tokenized differently."
            )
        request_ids.append(ids + list(row.tokens))
    index = {id(group): i for i, group in enumerate(groups)}
    complete_before = sum(1 for group in groups if group.complete)
    must_finish = {
        index[id(group)] for group in groups if group.age >= max_age and not group.complete
    }
    engine = _engine_of(trainer)
    wrapper = PartialRolloutEngine(
        engine,
        [index[id(group)] for group, _ in pending],
        [len(row.tokens) for _, row in pending],
        target,
        complete_before = complete_before,
        must_finish = must_finish,
    )
    had_add = "add_request" in engine.__dict__
    had_step = "step" in engine.__dict__
    original_add, original_step = engine.add_request, engine.step
    engine.add_request = lambda *a, **k: wrapper.add_request(original_add, *a, **k)
    engine.step = lambda: wrapper.step(original_step)
    try:
        result = _call_single_turn(trainer, request_ids, None, multimodal_fields)
    finally:
        if had_add:
            engine.add_request = original_add
        else:
            del engine.add_request
        if had_step:
            engine.step = original_step
        else:
            del engine.step
    completion_ids, logprobs = result[0], result[1]
    tail = tuple(
        result[2:]
    )  # TRL 1.0-1.16 returns 2 values, later ones a sampling mask / extra fields
    if any(x not in (None, {}) for x in tail):
        raise RuntimeError(
            "Unsloth: GRPO partial rollout cannot carry extra per-token generation fields."
        )
    if logprobs is None:
        raise RuntimeError("Unsloth: GRPO partial rollout needs vLLM sampler logprobs.")
    if len(completion_ids) != len(pending) or len(logprobs) != len(pending):
        raise RuntimeError(
            "Unsloth: GRPO partial rollout got a different number of completions than requests."
        )
    aborted = wrapper.aborted_rows()
    version = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
    for i, (_, row) in enumerate(pending):
        tokens, lps = list(completion_ids[i]), list(logprobs[i])
        if len(tokens) != len(lps):
            raise RuntimeError(
                "Unsloth: GRPO partial rollout got logprobs that do not line up with tokens."
            )
        row.tokens.extend(tokens)
        row.logprobs.extend(lps)
        row.versions.extend([version] * len(tokens))
        row.done = i not in aborted
    return len(aborted), tail, (None if wrapper.can_cut else wrapper.cut_error)


def _replay(rows, tail):
    """Stage 2 stand-in for _generate_single_turn: the trained rows, full completions, in order."""

    def replay(prompt_ids, images, multimodal_fields, *args, **kwargs):
        if len(prompt_ids) != len(rows) or any(
            list(a) != b.prompt_ids for a, b in zip(prompt_ids, rows)
        ):
            raise RuntimeError(
                "Unsloth: GRPO partial rollout replay saw different prompts than it generated."
            )
        return (
            [list(r.tokens) for r in rows],
            [list(r.logprobs) for r in rows],
            *({} if isinstance(x, dict) else None for x in tail),
        )

    return replay


def _working_copy(state):
    """Generation extends carried rows in place; work on copies so a failed scoring call can be retried."""
    work = PartialRolloutState(state.run)
    work.serial, work.steps, work.aborted_rows = state.serial, state.steps, state.aborted_rows
    for group in state.carry:
        rows = []
        for row in group.rows:
            new = _Row(row.input)
            new.prompt_ids, new.done = row.prompt_ids, row.done
            new.tokens, new.logprobs, new.versions = (
                list(row.tokens),
                list(row.logprobs),
                list(row.versions),
            )
            rows.append(new)
        copy = _Group(rows, group.serial)
        copy.age = group.age
        work.carry.append(copy)
    return work


def _log(trainer, name, value):
    metrics = getattr(trainer, "_metrics", None)
    if isinstance(metrics, dict) and "train" in metrics:
        try:
            metrics["train"][name].append(value)
        except Exception:
            pass


def partial_rollout_generate_and_score(trainer, original, inputs, *args, **kwargs):
    settings = partial_rollout_settings()
    model = getattr(trainer, "model", None)
    state = getattr(trainer, "_unsloth_partial_rollout", None)
    run = id(getattr(trainer, "state", None))
    if state is not None and state.run != run:
        # A new train() call or a checkpoint resume replaced TrainerState: never feed the old run's rows into it.
        state = None
        for name in ("_unsloth_partial_rollout", "_unsloth_partial_rollout_last"):
            trainer.__dict__.pop(name, None)
    if settings is None or (model is not None and not getattr(model, "training", True)):
        return original(trainer, inputs, *args, **kwargs)
    if state is not None and getattr(trainer, "_unsloth_partial_rollout_refused", None):
        return original(trainer, inputs, *args, **kwargs)
    if state is None:
        reason = getattr(trainer, "_unsloth_partial_rollout_refused", None) or _unsupported_reason(
            trainer
        )
        if reason is None:
            group_size = int(trainer.num_generations)
            steps = int(getattr(trainer.args, "steps_per_generation", 1) or 1)
            unit = _unit(group_size, steps)
            if len(inputs) % group_size or (len(inputs) // group_size) <= unit:
                reason = "needs more than one micro-batch unit of groups per generation batch."
        if reason is not None:
            trainer._unsloth_partial_rollout_refused = reason
            _warn_once(reason, reason)
            return original(trainer, inputs, *args, **kwargs)
        _configure_rollout_logprobs(trainer)
        reason = _kept_set_reason(trainer)
        if reason is not None:
            trainer._unsloth_partial_rollout_refused = reason
            _warn_once(reason, reason)
            return original(trainer, inputs, *args, **kwargs)
        state = PartialRolloutState(run)
    fraction, max_age = settings
    group_size = int(trainer.num_generations)
    fresh = _groups_of(inputs, group_size)
    if fresh is None:
        if not state.carry:
            _warn_once("groups", "the batch is not whole groups of one prompt (text only).")
            return original(trainer, inputs, *args, **kwargs)
        raise RuntimeError(
            "Unsloth: GRPO partial rollout got a batch that is not whole groups of one prompt."
        )
    unit = _unit(group_size, int(getattr(trainer.args, "steps_per_generation", 1) or 1))
    spare = _spare_groups(fraction, len(fresh), unit)
    state = _working_copy(state)
    groups = list(state.carry)
    for rows in fresh:
        groups.append(_Group([_Row(example) for example in rows], state.serial))
        state.serial += 1
    target = len(groups) - spare
    aged = sum(1 for group in groups if group.age >= max_age)
    target = max(target, unit * math.ceil(aged / unit))
    _configure_rollout_logprobs(trainer)
    num_aborted, tail, cut_error = _generate_rows(trainer, state, groups, target, max_age)
    complete = sorted((g for g in groups if g.complete), key = lambda g: (-g.age, g.serial))
    take = min(len(complete), target)
    if cut_error is not None:
        # This engine or these sampling params cannot be cut: every row ran to its end, so train all of them
        # (what a normal rollout does) and stop carrying.
        take = len(complete)
    take -= take % unit
    if take == 0:
        raise RuntimeError(
            "Unsloth: GRPO partial rollout finished generation without a trainable group."
        )
    train = complete[:take]
    chosen = {id(g) for g in train}
    state.carry = [g for g in groups if id(g) not in chosen]
    for group in state.carry:
        group.age += 1
    state.steps += 1
    state.aborted_rows += num_aborted
    rows = [row for group in train for row in group.rows]
    version = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
    tokens = sum(len(r.versions) for r in rows)
    stale = sum(sum(1 for v in r.versions if v != version) for r in rows)
    lag = sum(sum(version - v for v in r.versions) for r in rows)
    record = dict(
        trained_rows = len(rows),
        carried_groups = len(state.carry),
        aborted_rows = num_aborted,
        stale_token_frac = stale / max(tokens, 1),
        token_staleness_mean = lag / max(tokens, 1),
        versions = [list(r.versions) for r in rows],
    )
    had = "_generate_single_turn" in trainer.__dict__
    previous = trainer.__dict__.get("_generate_single_turn", None)
    trainer._generate_single_turn = _replay(rows, tail)
    try:
        scored = original(trainer, [row.input for row in rows], *args, **kwargs)
    finally:
        if had:
            trainer._generate_single_turn = previous
        else:
            del trainer._generate_single_turn
    # Published only after scoring succeeded: a retry of the same batch starts from the previous carry.
    trainer._unsloth_partial_rollout = state
    trainer._unsloth_partial_rollout_last = record
    if cut_error is not None:
        trainer._unsloth_partial_rollout_refused = cut_error
        _warn_once(cut_error, f"cannot stop this rollout early ({cut_error}).")
    _log(trainer, "partial_rollout/trained_groups", float(len(train)))
    _log(trainer, "partial_rollout/carried_groups", float(len(state.carry)))
    _log(trainer, "partial_rollout/aborted_rows", float(num_aborted))
    _log(trainer, "partial_rollout/stale_token_frac", record["stale_token_frac"])
    _log(trainer, "partial_rollout/token_staleness_mean", record["token_staleness_mean"])
    return scored


def wrap_grpo_partial_rollout(trainer_class):
    """Route trainer_class._generate_and_score_completions through the partial rollout (a no-op while it is off)."""
    original = getattr(trainer_class, "_generate_and_score_completions", None)
    if original is None or getattr(original, "_unsloth_partial_rollout_wrapped", False):
        return

    @functools.wraps(original)
    def _generate_and_score_completions(self, inputs, *args, **kwargs):
        return partial_rollout_generate_and_score(self, original, inputs, *args, **kwargs)

    _generate_and_score_completions._unsloth_partial_rollout_wrapped = True
    trainer_class._generate_and_score_completions = _generate_and_score_completions
