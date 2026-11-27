# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""On-policy distillation helpers (Thinking Machines, "On-Policy Distillation"; tinker-cookbook
``distillation/train_on_policy.py``).

- ``teacher_model = "base"`` for GKDTrainer / DistillationTrainer: a LoRA student's own base weights, scored with the
  adapter disabled, are the teacher. No second copy of the weights is loaded.
- GRPO ``unsloth_distillation_coef``: adds ``A_t = -coef * (log q(x_t) - log p_teacher(x_t))`` (no discount) to the
  environment advantages, where ``q`` is the sampler that drew the tokens. The teacher is the adapter-off base
  (``unsloth_distillation_teacher = "base"``) or a model passed as ``GRPOTrainer(..., teacher_model = ...)``.
- AsyncGRPO ``unsloth_sampler_kl_coef``: Tinker's StableReinforce KL to the sampler, ``coef * sum_t k(q, p)``."""

import contextlib
import functools
import os
import sys

import torch

__all__ = [
    "ADAPTER_OFF_TEACHER",
    "AdapterOffTeacher",
    "is_adapter_off_teacher",
    "resolve_teacher_model",
    "on_policy_distillation_advantages",
    "score_completion_logps",
    "forwards_are_collective",
    "grpo_setup_distillation",
    "grpo_check_sampler_kl",
    "grpo_add_distillation_advantages",
    "wrap_grpo_on_policy_distillation",
    "sampler_kl_per_token",
]

ADAPTER_OFF_TEACHER = "base"


def is_adapter_off_teacher(teacher_model):
    """True for the ``teacher_model = "base"`` request (a local folder named base is ``"./base"``)."""
    return isinstance(teacher_model, str) and teacher_model.strip().lower() == ADAPTER_OFF_TEACHER


def _unwrap(model):
    seen = 0
    while seen < 8 and model is not None:
        inner = getattr(model, "module", None)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) and inner is not None:
            model = inner
        elif getattr(model, "_orig_mod", None) is not None:
            model = model._orig_mod
        else:
            break
        seen += 1
    return model


def _peft_student(model, who):
    student = _unwrap(model)
    if (
        student is None
        or not callable(getattr(student, "disable_adapter", None))
        or not hasattr(student, "peft_config")
    ):
        raise ValueError(
            f"Unsloth: {who} uses the base model with the adapter disabled as the teacher, which needs a LoRA "
            "student. Call FastLanguageModel.get_peft_model first, or pass a separate teacher model."
        )
    return student


@contextlib.contextmanager
def _adapter_off(student):
    # Nested disable_adapter() would re-enable the adapter on the inner exit: only disable once.
    if getattr(student, "_unsloth_adapter_off_depth", 0) > 0:
        yield
        return
    student._unsloth_adapter_off_depth = 1
    try:
        with student.disable_adapter():
            yield
    finally:
        student._unsloth_adapter_off_depth = 0


def _base_module(module):
    original = getattr(module, "original_module", None)
    return original if isinstance(original, torch.nn.Module) else module


class AdapterOffTeacher(torch.nn.Module):
    """A LoRA student with its adapter disabled, standing in for a frozen teacher.

    It owns no parameters or submodules, so accelerate's ``prepare_model``, ``.eval()`` and ``.to()`` leave the student
    alone; attribute reads (``config``, ``base_model``, ``get_output_embeddings``, ...) go to the student's causal LM,
    the object TRL reaches through a PEFT model. Every call runs under ``torch.no_grad()`` with the adapter off."""

    _unsloth_adapter_off_teacher = True

    def __init__(self, student):
        super().__init__()
        peft = _peft_student(student, 'teacher_model = "base"')
        inner = getattr(getattr(peft, "base_model", None), "model", None) or peft
        object.__setattr__(self, "_unsloth_student", peft)
        object.__setattr__(self, "_unsloth_adapter_off_student_model", inner)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__.get("_unsloth_adapter_off_student_model", None)
            if inner is None:
                raise
            return getattr(inner, name)

    def __call__(self, *args, **kwargs):
        with torch.no_grad(), _adapter_off(self._unsloth_student):
            return super().__call__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self._unsloth_student(*args, **kwargs)

    def get_output_embeddings(self):
        # A head trained through modules_to_save is a ModulesToSaveWrapper whose weight is the adapter's copy outside
        # this context; callers that project hidden states themselves (DistillationTrainer) need the base head.
        return _base_module(self._unsloth_adapter_off_student_model.get_output_embeddings())

    def get_input_embeddings(self):
        return _base_module(self._unsloth_adapter_off_student_model.get_input_embeddings())

    def generate(self, *args, **kwargs):
        # GKD seq_kd samples targets from the teacher: the base model's, so with the adapter off as well.
        with torch.no_grad(), _adapter_off(self._unsloth_student):
            return self._unsloth_student.generate(*args, **kwargs)

    def __repr__(self):
        return f"AdapterOffTeacher({type(self._unsloth_adapter_off_student_model).__name__}, adapter disabled)"


def _native_parallel_modes(model):
    """Unsloth's environment-driven native sharding of ``model`` (never reflected in ``args.fsdp``)."""
    checks = (
        ("pipeline parallel", "pp_native", "is_pipeline_parallel"),
        ("FSDP2", "fsdp2_native", "is_native_fsdp2"),
        ("tensor parallel", "tp_native", "is_tensor_parallel"),
        ("expert parallel", "ep_native", "is_expert_parallel"),
    )
    modes = []
    for name, module, check in checks:
        # A mode whose module was never imported has not sharded anything.
        fn = getattr(sys.modules.get(f"unsloth.models.{module}", None), check, None)
        try:
            if callable(fn) and fn(model):
                modes.append(name)
        except Exception:
            pass
    return modes


def resolve_teacher_model(
    teacher_model,
    model,
    args = None,
    who = "GKDTrainer",
    sharded_forward_ok = False,
):
    """``"base"`` becomes the adapter-off view of ``model`` (also inside a DistillationTrainer teacher mapping);
    anything else is returned unchanged.

    ``sharded_forward_ok``: the caller only runs whole forwards of the teacher, the same number on every rank (GRPO's
    scoring), so Unsloth's native FSDP2 / tensor / expert parallel student can serve as the teacher. TRL's GKD and
    DistillationTrainer also call it on their own (generation, projecting hidden states through the head), which
    sharded weights do not support."""
    if isinstance(teacher_model, dict):
        if not any(is_adapter_off_teacher(t) for t in teacher_model.values()):
            return teacher_model
        return {
            k: resolve_teacher_model(t, model, args, who, sharded_forward_ok)
            for k, t in teacher_model.items()
        }
    if not is_adapter_off_teacher(teacher_model):
        return teacher_model
    if args is not None and (getattr(args, "deepspeed", None) or getattr(args, "fsdp", None)):
        raise ValueError(
            f'Unsloth: {who}(teacher_model = "base") scores the student\'s own weights with the adapter off, which '
            "DeepSpeed and FSDP shard. Use DDP or one GPU, or pass a separate teacher model."
        )
    if model is None or isinstance(model, str):
        raise ValueError(
            f'Unsloth: {who}(teacher_model = "base") needs the loaded LoRA student as model.'
        )
    modes = _native_parallel_modes(model)
    # A pipeline stage holds only its own layers: a plain forward of it is not the model.
    refused = [m for m in modes if not sharded_forward_ok or m == "pipeline parallel"]
    if refused:
        raise ValueError(
            f'Unsloth: {who}(teacher_model = "base") scores the student\'s own weights with the adapter off, but '
            f"they are sharded by {' and '.join(refused)}. Use DDP or one GPU, or pass a separate teacher model."
        )
    return AdapterOffTeacher(model)


def on_policy_distillation_advantages(
    student_logps,
    teacher_logps,
    mask,
    coef = 1.0,
):
    """Tinker incorporate_kl_penalty with kl_discount_factor = 0: ``-coef * mask * (log q - log p_teacher)``.

    float32, zero off the mask, and zero where either log-prob is not finite (an unscored vLLM token)."""
    student = student_logps.detach().to(torch.float32)
    teacher = teacher_logps.detach().to(device = student.device, dtype = torch.float32)
    diff = student - teacher
    keep = mask.to(device = student.device).bool() & torch.isfinite(diff)
    return torch.where(keep, -float(coef) * diff, 0.0)


def _rows_per_chunk(
    seq_len,
    vocab_size,
    budget = 1 << 28,
):
    return max(1, int(budget // max(1, seq_len * vocab_size)))


def _columns_per_chunk(
    rows,
    vocab_size,
    budget = 1 << 25,
):
    # Completion columns scored at once: the float32 [rows, columns, V] gather and log_softmax stay near
    # budget * 4 bytes each instead of growing with the completion length.
    return max(1, int(budget // max(1, rows * vocab_size)))


def _distributed_world():
    try:
        import torch.distributed as dist
    except Exception:
        return None
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return None
    return dist


def _max_over_ranks(value, device):
    """The largest ``value`` over all ranks (one tiny all_reduce), or ``value`` when not distributed."""
    dist = _distributed_world()
    if dist is None:
        return value
    backend = str(dist.get_backend()).lower()
    if backend == "gloo":
        device = torch.device("cpu")
    elif device.type == "cpu" and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    t = torch.tensor([int(value)], dtype = torch.int64, device = device)
    dist.all_reduce(t, op = dist.ReduceOp.MAX)
    return int(t.item())


def forwards_are_collective(*models):
    """True when a forward of any of ``models`` runs collectives (FSDP / ZeRO-3 gathers, tensor / expert parallel
    all-reduces and all-to-alls), so every rank must run the same number of forwards."""
    if _distributed_world() is None:
        return False
    try:
        from unsloth_zoo.rl_replacements import grpo_forwards_are_collective
    except Exception:
        # An older unsloth_zoo: keep every rank in step whenever training is distributed.
        return True
    for model in models:
        model = (
            model.__dict__.get("_unsloth_student", model)
            if isinstance(model, torch.nn.Module)
            else model
        )
        if not isinstance(model, torch.nn.Module):
            continue
        try:
            if grpo_forwards_are_collective(model):
                return True
        except Exception:
            return True
    return False


def _autocast(device, dtype):
    if dtype is not None and device.type in ("cuda", "xpu", "cpu"):
        return torch.autocast(device_type = device.type, dtype = dtype)
    return contextlib.nullcontext()


def _no_op_forward(model, device, autocast):
    # A padding chunk on a rank with fewer rows than another: one token, so its collectives match the other ranks'.
    ids = torch.zeros((1, 1), dtype = torch.long, device = device)
    with autocast:
        model(input_ids = ids, attention_mask = torch.ones_like(ids), use_cache = False)


@torch.no_grad()
def score_completion_logps(
    model,
    prompt_ids,
    prompt_mask,
    completion_ids,
    completion_mask,
    temperature = 1.0,
    autocast_dtype = None,
    rows_per_chunk = None,
    sync_chunks = False,
    columns_per_chunk = None,
):
    """Per-token log-probs of ``completion_ids`` under ``model`` in the completion layout ``[B, C]``.

    Each left-padded row is rolled to start at column 0 first: Unsloth's training forward drops the 2D attention mask
    (#11885), and with right padding the causal mask alone already keeps pads out of every real token's context.

    ``sync_chunks``: every rank runs the largest chunk count of any rank, padding with one-token forwards, for models
    whose forwards are collectives (``forwards_are_collective``). The chunk count follows the rank's own length."""
    input_ids = torch.cat([prompt_ids, completion_ids], dim = 1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim = 1).to(torch.long)
    B, L = input_ids.shape
    P, C = prompt_ids.shape[1], completion_ids.shape[1]
    out = torch.zeros(B, C, dtype = torch.float32, device = completion_ids.device)
    empty = B == 0 or C == 0
    if empty and not sync_chunks:
        return out
    # A teacher on another device than the rollout: run it where its embeddings live.
    device = input_ids.device
    get_embed = getattr(model, "get_input_embeddings", None)
    try:
        device = get_embed().weight.device if callable(get_embed) else device
    except Exception:
        pass
    if empty:
        for _ in range(_max_over_ranks(0, device)):
            _no_op_forward(model, device, _autocast(device, autocast_dtype))
        return out
    left_pad = (attention_mask.cumsum(dim = 1) == 0).sum(dim = 1)
    roll = (torch.arange(L, device = input_ids.device).unsqueeze(0) + left_pad.unsqueeze(1)) % L
    ids = input_ids.gather(1, roll)
    att = attention_mask.gather(1, roll)
    # Completion token j sits at P + j - left_pad after the roll, so the logit predicting it is one column earlier.
    pos = (P - 1 - left_pad).unsqueeze(1) + torch.arange(C, device = input_ids.device).unsqueeze(0)
    pos = pos.clamp(min = 0)
    ids, att, pos = ids.to(device), att.to(device), pos.to(device)
    head = None
    get_head = getattr(model, "get_output_embeddings", None)
    if callable(get_head):
        head = get_head()
    vocab = getattr(getattr(head, "weight", None), "shape", (None,))[0]
    # Only the columns from each chunk's earliest predicting position on reach the head (logits_to_keep), so the
    # teacher's logits memory and head FLOPs scale with the completion, not the prompt. The chunk size stays sized by
    # the full length: the decoder still runs over every position, and more rows per chunk would raise its peak.
    if rows_per_chunk is None:
        rows_per_chunk = _rows_per_chunk(L, vocab or 1 << 17)
    pass_keep = True
    autocast = _autocast(ids.device, autocast_dtype)
    n_chunks = -(-B // rows_per_chunk)
    total_chunks = _max_over_ranks(n_chunks, device) if sync_chunks else n_chunks
    # Real logits, with the model's own scaling and softcapping: no hidden-state shortcut while scoring.
    prior_hidden_states = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
    try:
        for chunk in range(total_chunks):
            if chunk >= n_chunks:
                _no_op_forward(model, device, autocast)
                continue
            start = chunk * rows_per_chunk
            rows = slice(start, start + rows_per_chunk)
            keep = L - int(pos[rows].min())
            kwargs = {"logits_to_keep": keep} if pass_keep and keep < L else {}
            with autocast:
                try:
                    outputs = model(
                        input_ids = ids[rows], attention_mask = att[rows], use_cache = False, **kwargs
                    )
                except TypeError as error:
                    # A forward without logits_to_keep (custom or pre-4.50 model): score the full width.
                    if not kwargs or "logits_to_keep" not in str(error):
                        raise
                    pass_keep = False
                    outputs = model(input_ids = ids[rows], attention_mask = att[rows], use_cache = False)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            if vocab is not None and logits.shape[-1] != vocab:
                raise RuntimeError(
                    f"Unsloth: distillation scoring expected logits over {vocab} tokens, but the forward "
                    f"returned a last dimension of {logits.shape[-1]}."
                )
            # A forward that accepts but ignores logits_to_keep returns all L columns: index from the real width.
            width = logits.shape[1]
            if width != L and width != keep:
                raise RuntimeError(
                    f"Unsloth: distillation scoring asked for the last {keep} of {L} logit columns, but the forward "
                    f"returned {width}."
                )
            del outputs
            index = (pos[rows] - (L - width)).to(logits.device)
            targets = completion_ids[rows].to(logits.device)
            n_rows, V = index.shape[0], logits.shape[-1]
            step = columns_per_chunk or _columns_per_chunk(n_rows, V)
            # Completion columns a slice at a time: log_softmax is per position, so each slice's values are the
            # full-width ones, while the float32 copies stay [rows, step, V] instead of [rows, C, V].
            for c0 in range(0, C, step):
                cols = slice(c0, c0 + step)
                picked = logits.gather(1, index[:, cols].unsqueeze(-1).expand(-1, -1, V)).to(
                    torch.float32
                )
                if temperature is not None and temperature != 1.0:
                    picked = picked / temperature
                logps = torch.log_softmax(picked, dim = -1)
                del picked
                out[rows, cols] = (
                    logps.gather(-1, targets[:, cols].unsqueeze(-1)).squeeze(-1).to(out.device)
                )
                del logps
            del logits
    finally:
        if prior_hidden_states is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior_hidden_states
    return out.masked_fill(~completion_mask.to(out.device).bool(), 0.0)


def _autocast_dtype(trainer):
    if getattr(trainer, "_autocast_enabled", None) is not None:
        return trainer._autocast_dtype if trainer._autocast_enabled else None
    args = getattr(trainer, "args", None)
    if getattr(args, "bf16", False):
        return torch.bfloat16
    if getattr(args, "fp16", False):
        return torch.float16
    return None


def _vocab_size(model):
    config = getattr(model, "config", None)
    if config is None:
        # A PEFT model forwards its config; read the causal LM it wraps when the wrapper does not.
        config = getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None)
    try:
        return config.get_text_config().vocab_size
    except Exception:
        return getattr(config, "vocab_size", None)


def grpo_check_sampler_kl(trainer):
    """The sync GRPO sampler KL lives in unsloth_zoo's grpo_compute_loss: refuse an older zoo, which would drop it."""
    args = getattr(trainer, "args", None)
    coef = float(getattr(args, "unsloth_sampler_kl_coef", 0.0) or 0.0)
    estimator = getattr(args, "unsloth_sampler_kl_estimator", None) or "k1"
    if not (0.0 <= coef < float("inf")):
        raise ValueError(
            f"Unsloth: unsloth_sampler_kl_coef must be a finite float >= 0, not {coef!r}."
        )
    if estimator not in ("k1", "k3"):
        raise ValueError(
            f"Unsloth: unsloth_sampler_kl_estimator must be 'k1' or 'k3', not {estimator!r}."
        )
    if coef != 0.0:
        _require_zoo("GRPO_SAMPLER_KL", "unsloth_sampler_kl_coef")
        _refuse_subclass_objective(trainer, "unsloth_sampler_kl_coef")
    return coef


def _refuse_subclass_objective(trainer, option):
    # A GRPO subclass with its own _compute_loss (TRL's GMPOTrainer) never reaches the fused loss these join.
    try:
        from unsloth.models._grpo_subclass_loss import grpo_subclass_compute_loss
    except Exception:
        return
    if grpo_subclass_compute_loss(trainer) is not None:
        raise ValueError(
            f"Unsloth: {option} is applied in Unsloth's GRPO loss, but {type(trainer).__name__} overrides "
            "_compute_loss with its own objective, which would ignore it."
        )


def _refuse_unreconstructable_sampler(trainer, rescoring = False):
    # Rescoring the sampled tokens over the full vocabulary at args.temperature is not the sampler under generation
    # filters, penalties or a temperature override. vLLM's sampling logprobs avoid the rescoring, so setup skips vLLM;
    # the first rollout that comes without them (an older TRL with the IS correction off) checks with rescoring = True.
    args = getattr(trainer, "args", None)
    if not rescoring and getattr(trainer, "use_vllm", getattr(args, "use_vllm", False)):
        return
    defaults = {
        "top_k": (None, 0, -1),
        "top_p": (None, 1, 1.0),
        "min_p": (None, 0, 0.0),
        "repetition_penalty": (None, 1, 1.0),
    }
    temperature = getattr(args, "temperature", 1.0)
    used = set()
    for source in (
        args,
        getattr(args, "vllm_sampling_params", None),
        getattr(args, "generation_kwargs", None),
    ):
        if source is None:
            continue
        read = (
            source.get
            if isinstance(source, dict)
            else (lambda k, d = None, src = source: getattr(src, k, d))
        )
        used |= {k for k, ok in defaults.items() if read(k, None) not in ok}
        if source is not args and read("temperature", None) not in (None, temperature):
            used.add("a temperature override")
    if used:
        raise ValueError(
            f"Unsloth: unsloth_distillation_coef rescores the sampled tokens over the full vocabulary at "
            f"args.temperature when the rollout has no sampling logprobs, which does not reproduce the sampler under "
            f"{', '.join(sorted(used))}. Use vLLM with its sampling logprobs (vllm_importance_sampling_correction = "
            "True) or leave those at their defaults."
        )


def _refuse_untruncated_filtered_logps(trainer):
    # Colocated vLLM with the IS correction returns pre-truncation logprobs (zoo #2794); under top_k / top_p / min_p
    # those are not the sampler's, so they cannot stand in for q.
    try:
        from unsloth_zoo.rl_replacements import (
            grpo_rollout_logprobs_untruncated,
            grpo_sampling_support,
        )
    except Exception:
        return
    if grpo_rollout_logprobs_untruncated(trainer) and grpo_sampling_support(trainer) is not None:
        raise ValueError(
            "Unsloth: unsloth_distillation_coef needs the sampler's own logprobs, but this vLLM engine returns them "
            "from before top_k / top_p / min_p. Set UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS=0 (vLLM's processed logprobs "
            "over the kept tokens) or sample without those filters."
        )


def _require_zoo(flag, option):
    try:
        import unsloth_zoo.rl_replacements as _zoo_rl
    except Exception:
        _zoo_rl = None
    if not getattr(_zoo_rl, flag, False):
        raise RuntimeError(
            f"Unsloth: {option} needs a newer unsloth_zoo. Please upgrade: pip install -U unsloth_zoo"
        )


_VISION_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_attention_mask",
    "image_sizes",
    "pixel_values_videos",
)


def grpo_setup_distillation(trainer, teacher_model = None):
    """Validate the GRPO distillation config once, after TRL's __init__; returns the coefficient (0 = off)."""
    args = getattr(trainer, "args", None)
    coef = float(getattr(args, "unsloth_distillation_coef", 0.0) or 0.0)
    if not (0.0 <= coef < float("inf")):
        raise ValueError(
            f"Unsloth: unsloth_distillation_coef must be a finite float >= 0, not {coef!r}."
        )
    trainer._unsloth_distillation_coef = coef
    trainer._unsloth_distillation_teacher = None
    if coef == 0.0:
        if teacher_model is not None:
            raise ValueError(
                "Unsloth: GRPOTrainer got a teacher_model but unsloth_distillation_coef is 0."
            )
        return coef
    # Per-token advantages must be realigned to the zoo's left-packed loss layout.
    _require_zoo("GRPO_TOKEN_ADVANTAGES", "unsloth_distillation_coef")
    _refuse_subclass_objective(trainer, "unsloth_distillation_coef")
    _refuse_unreconstructable_sampler(trainer)
    if teacher_model is None:
        teacher_model = getattr(args, "unsloth_distillation_teacher", None) or ADAPTER_OFF_TEACHER
    if isinstance(teacher_model, str):
        if not is_adapter_off_teacher(teacher_model):
            raise ValueError(
                "Unsloth: unsloth_distillation_teacher takes 'base' (the adapter-off base model). Pass any other "
                "teacher as a loaded model: GRPOTrainer(..., teacher_model = teacher)."
            )
        # Through resolve_teacher_model, which refuses DeepSpeed / FSDP sharding as for GKD. Native FSDP2 / tensor /
        # expert parallel are fine here: scoring runs whole forwards, the same number on every rank.
        teacher = resolve_teacher_model(
            ADAPTER_OFF_TEACHER, trainer.model, args, "GRPOTrainer", sharded_forward_ok = True
        )
    else:
        teacher = teacher_model
        if isinstance(teacher, torch.nn.Module):
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
        s_vocab, t_vocab = _vocab_size(_unwrap(trainer.model)), _vocab_size(teacher)
        if s_vocab is not None and t_vocab is not None and s_vocab != t_vocab:
            raise ValueError(
                f"Unsloth: the student has vocab_size {s_vocab} but the teacher has {t_vocab}. On-policy distillation "
                "compares per-token log-probs, so the teacher needs the student's tokenizer."
            )
    trainer._unsloth_distillation_teacher = teacher
    return coef


def grpo_add_distillation_advantages(trainer, output):
    """Score the rollout under the teacher and store the per-token distillation advantages in ``output``."""
    coef = getattr(trainer, "_unsloth_distillation_coef", 0.0)
    teacher = getattr(trainer, "_unsloth_distillation_teacher", None)
    if not coef or teacher is None or not isinstance(output, dict):
        return output
    if any(output.get(k, None) is not None for k in _VISION_KEYS):
        raise NotImplementedError("Unsloth: GRPO on-policy distillation supports text models only.")
    prompt_ids, prompt_mask = output["prompt_ids"], output["prompt_mask"]
    completion_ids, completion_mask = output["completion_ids"], output["completion_mask"]
    dtype = _autocast_dtype(trainer)
    student = _unwrap(trainer.accelerator.unwrap_model(trainer.model))
    # Each rank's chunk count follows its own rollout length: under collective forwards all ranks run the largest.
    sync = forwards_are_collective(student, teacher)
    student_logps = output.get("sampling_per_token_logps", None)
    if student_logps is not None:
        _refuse_untruncated_filtered_logps(trainer)
    rescore = student_logps is None or student_logps.shape != completion_ids.shape
    if rescore:
        # No sampling logprobs: the policy sampled the tokens itself, at the trainer's temperature.
        _refuse_unreconstructable_sampler(trainer, rescoring = True)
    # Every rank joins this all_reduce, rescoring or not.
    any_rescore = _max_over_ranks(int(rescore), completion_ids.device) if sync else rescore
    if any_rescore:
        # A rank that has its sampling logprobs while another rescores runs padding forwards only (zero rows).
        rows = slice(None) if rescore else slice(0, 0)
        scored = score_completion_logps(
            student,
            prompt_ids[rows],
            prompt_mask[rows],
            completion_ids[rows],
            completion_mask[rows],
            temperature = getattr(trainer.args, "temperature", 1.0),
            autocast_dtype = dtype,
            sync_chunks = sync,
        )
        student_logps = scored if rescore else student_logps
    # The teacher is scored untempered: the advantage is then the reverse KL of the sampler against the teacher.
    teacher_logps = score_completion_logps(
        teacher,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        temperature = 1.0,
        autocast_dtype = dtype,
        sync_chunks = sync,
    )
    mask = completion_mask
    tool_mask = output.get("tool_mask", None)
    if tool_mask is not None and tool_mask.shape == completion_mask.shape:
        mask = completion_mask * tool_mask
    advantages = on_policy_distillation_advantages(student_logps, teacher_logps, mask, coef)
    output["unsloth_distillation_advantages"] = advantages
    try:
        _log_teacher_kl(trainer, advantages, mask, coef)
    except Exception:
        pass
    return output


def _log_teacher_kl(trainer, advantages, mask, coef):
    # Tinker's teacher_kl: the mean per-token log q - log p_teacher over the sampled tokens.
    mask = mask.to(torch.float32)
    stats = torch.stack([(-advantages / coef * mask).sum(), mask.sum()])
    stats = (
        trainer.accelerator.reduce(stats, reduction = "sum")
        if trainer.accelerator.num_processes > 1
        else stats
    )
    mode = "train" if trainer.model.training else "eval"
    metrics = getattr(trainer, "_metrics", None)
    if isinstance(metrics, dict) and mode in metrics:
        metrics[mode]["distillation/teacher_kl"].append((stats[0] / stats[1].clamp(min = 1.0)).item())


def wrap_grpo_on_policy_distillation(trainer_cls):
    original = getattr(trainer_cls, "_generate_and_score_completions", None)
    if original is None or getattr(original, "_unsloth_distillation_wrapped", False):
        return

    @functools.wraps(original)
    def _generate_and_score_completions(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        if getattr(self, "_unsloth_distillation_coef", 0.0):
            output = grpo_add_distillation_advantages(self, output)
        return output

    _generate_and_score_completions._unsloth_distillation_wrapped = True
    trainer_cls._generate_and_score_completions = _generate_and_score_completions


def sampler_kl_per_token(
    log_probs,
    sampler_log_probs,
    estimator = "k1",
):
    """Tinker stable_reinforce KL(q || p) per sampled token: k1 = log q - log p, k3 = r - 1 - log r, r = p / q."""
    log_ratio = torch.clamp(log_probs - sampler_log_probs, min = -20.0, max = 20.0)
    if estimator == "k3":
        return torch.expm1(log_ratio) - log_ratio
    return -log_ratio
