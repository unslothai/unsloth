# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Choose between padding-free and padded batches for SFT when the user left `padding_free` unset.

Padding-free removes pad tokens, but its varlen batches skip the per-block compile, so each step
launches more kernels from Python. Padded batches pay for their pad tokens on the GPU but run the
compiled blocks. Which one is faster depends on whether a micro batch is bound by the host or by the
GPU: measured on A100, L4, T4 and RTX PRO 6000 Blackwell with Qwen3 0.6B / 4B LoRA (16 bit and
4 bit), padded + block compile was 5-17% faster only when a padded micro batch costs the GPU well
under the ~0.25-0.4 s the host spends launching it, and up to 57% slower once the GPU is the
bottleneck.

Rule (all estimates, every unknown keeps padding-free, the old default):
    padded GPU time  = (linear FLOPs per token * padded tokens + attention FLOPs per pair * pairs)
                       / (datasheet 16 bit peak * _EFFICIENCY)
    pick padded iff  padded GPU time < the measured threshold for the GPU's compute capability
                     (_MAX_PADDED_GPU_S_BY_CAPABILITY; any other capability keeps padding-free)
                 and the run has enough micro batches to repay the extra block compile
                     (about 25 s in 16 bit, 100 s in 4 bit) at ~_SAVED_S_PER_MICROBATCH each.

Only Linux + NVIDIA CUDA on a measured compute capability is ever switched: ROCm / HIP, Windows,
XPU, MPS, CPU and unmeasured capabilities (H100 / H200 sm90, B200 / B300 sm100 / sm103 have
datasheet peaks but no measurement yet) keep padding-free, unless `UNSLOTH_PADDING_CHOICE=padded`
is set explicitly.

`UNSLOTH_PADDING_CHOICE=padding_free` restores the previous behaviour (always padding-free),
`=padded` forces padded batches, `=auto` (default) applies the rule. An explicit
`padding_free=True/False` on the config always wins.
"""

import logging
import math
import os
import random
import sys

__all__ = [
    "padding_choice_mode",
    "estimate_microbatch_tokens",
    "choose_padding",
    "decide_auto_padding_free",
]

logger = logging.getLogger(__name__)

_EFFICIENCY = (
    0.5  # fraction of the datasheet peak a LoRA step reaches when GPU bound (A100 measured 0.5)
)
_MAX_PADDED_GPU_S = 0.10  # padded micro batch GPU estimate below which the host dominates
# Per compute capability, only where measured (padded + block compile vs padding-free, Qwen3 0.6B /
# 4B LoRA r16, 16 and 4 bit, 3 interleaved reps). A capability missing here keeps padding-free.
#   sm75 T4, sm80 A100, sm89 L4 (torch 2.14.1): 0.10 s.
#   sm120 RTX PRO 6000 Blackwell (Colab G4, torch 2.14.1 and 2.10.0): padded lost 24-29% at a
#   ~92 ms estimate on torch 2.14.1 and won (+5% on 2.14.1, +15-64% on 2.10.0) at <= 45 ms.
_MAX_PADDED_GPU_S_BY_CAPABILITY = {
    (7, 5): 0.10,
    (8, 0): 0.10,
    (8, 9): 0.10,
    (12, 0): 0.06,
}
_SAVED_S_PER_MICROBATCH = 0.04  # host time block compile saved per micro batch where it won
_EXTRA_COMPILE_S_16BIT = 25.0
_EXTRA_COMPILE_S_4BIT = 100.0
_REPAY_FACTOR = 2.0
_SAMPLE_ROWS = 512
_SIM_BATCHES = 256


def padding_choice_mode():
    mode = os.environ.get("UNSLOTH_PADDING_CHOICE", "auto").strip().lower()
    if mode in ("padding_free", "padding-free", "packed", "0", "off", "false", "no"):
        return "padding_free"
    if mode in ("padded", "pad", "block_compile"):
        return "padded"
    return "auto"


def estimate_microbatch_tokens(
    lengths,
    batch_size,
    max_length = None,
    seed = 0,
):
    """Expected (real tokens, padded tokens, attention pairs when padded) per micro batch.

    Batches are drawn uniformly from `lengths` (the trainer shuffles); padded = batch * longest.
    """
    lengths = [int(x) for x in lengths if x is not None and int(x) > 0]
    if not lengths or batch_size < 1:
        return None
    if max_length:
        lengths = [min(x, int(max_length)) for x in lengths]
    rng = random.Random(seed)
    real = padded = pairs = 0
    for _ in range(_SIM_BATCHES):
        batch = [lengths[rng.randrange(len(lengths))] for _ in range(batch_size)]
        longest = max(batch)
        real += sum(batch)
        padded += batch_size * longest
        pairs += batch_size * longest * (longest + 1) / 2
    n = _SIM_BATCHES
    return real / n, padded / n, pairs / n


def choose_padding(
    linear_flops_per_token,
    attention_flops_per_pair,
    padded_tokens,
    padded_pairs,
    peak_tflops,
    total_microbatches = None,
    load_in_4bit = False,
    capability = None,
):
    """Pure rule. Returns (use_padding_free: bool, reason: str, padded_gpu_s: float | None).

    `capability` is the CUDA (major, minor); one that was never measured keeps padding-free.
    None applies the default threshold (callers that know the device always pass it)."""
    if not peak_tflops or peak_tflops <= 0 or not linear_flops_per_token or not padded_tokens:
        return True, "no estimate", None
    max_gpu_s = _MAX_PADDED_GPU_S
    if capability is not None:
        capability = tuple(int(x) for x in capability)
        max_gpu_s = _MAX_PADDED_GPU_S_BY_CAPABILITY.get(capability)
        if max_gpu_s is None:
            return True, f"sm{capability[0]}{capability[1]} not measured", None
    flops = linear_flops_per_token * padded_tokens + (attention_flops_per_pair or 0) * (
        padded_pairs or 0
    )
    gpu_s = flops / (peak_tflops * 1e12 * _EFFICIENCY)
    if gpu_s >= max_gpu_s:
        return True, f"GPU bound (padded micro batch ~{gpu_s * 1e3:.0f} ms of GPU work)", gpu_s
    extra = _EXTRA_COMPILE_S_4BIT if load_in_4bit else _EXTRA_COMPILE_S_16BIT
    needed = math.ceil(_REPAY_FACTOR * extra / _SAVED_S_PER_MICROBATCH)
    if total_microbatches is None or total_microbatches < needed:
        return (
            True,
            f"run too short to repay the block compile ({total_microbatches} < {needed} micro batches)",
            gpu_s,
        )
    return False, f"host bound (padded micro batch ~{gpu_s * 1e3:.0f} ms of GPU work)", gpu_s


# ---------------------------------------------------------------------------------------------
# Feature gathering. Every failure returns None, which keeps padding-free.


def _sample_indices(
    n,
    k = _SAMPLE_ROWS,
    seed = 0,
):
    if n <= k:
        return list(range(n))
    return sorted(random.Random(seed).sample(range(n), k))


def _tokenizer_of(processing_class):
    tok = getattr(processing_class, "tokenizer", None) or processing_class
    return tok if callable(tok) else None


def _row_texts(rows, columns, config, formatting_func, tok):
    """One string per row as SFTTrainer would build it, or None when the format is not known."""
    if formatting_func is not None:
        out = []
        for row in rows:
            text = formatting_func(row)
            if isinstance(text, (list, tuple)):
                text = text[0] if text else ""
            if not isinstance(text, str):
                return None
            out.append(text)
        return out
    field = getattr(config, "dataset_text_field", None) or "text"
    if field in columns:
        texts = [row[field] for row in rows]
        return texts if all(isinstance(t, str) for t in texts) else None
    chat = getattr(tok, "apply_chat_template", None)
    if "messages" in columns or "conversations" in columns:
        key = "messages" if "messages" in columns else "conversations"
        if chat is None or getattr(tok, "chat_template", None) is None:
            return None
        return [chat(row[key], tokenize = False) for row in rows]
    if "prompt" in columns and "completion" in columns:
        out = []
        for row in rows:
            p, c = row["prompt"], row["completion"]
            if isinstance(p, str) and isinstance(c, str):
                out.append(p + c)
            elif isinstance(p, list) and isinstance(c, list) and chat is not None:
                out.append(chat(p + c, tokenize = False))
            else:
                return None
        return out
    return None


def _sample_lengths(dataset, config, processing_class, formatting_func):
    if dataset is None or isinstance(dataset, dict):
        return None
    try:
        n = len(dataset)
    except Exception:
        return None  # IterableDataset: no cheap sample
    if n == 0:
        return None
    columns = list(getattr(dataset, "column_names", None) or [])
    idx = _sample_indices(n)
    if "input_ids" in columns and formatting_func is None:
        return [len(x) for x in dataset.select(idx)["input_ids"]]
    tok = _tokenizer_of(processing_class)
    if tok is None:
        return None
    rows = dataset.select(idx)
    rows = [rows[i] for i in range(len(rows))]
    texts = _row_texts(rows, columns, config, formatting_func, tok)
    if not texts:
        return None
    return [len(x) for x in tok(texts, add_special_tokens = False)["input_ids"]]


def _block_compile_installed(model):
    try:
        from ..models._block_compile import _MARK, _OFF, _enabled
    except Exception:
        return False
    if not _enabled():
        return False
    for module in model.modules():
        if (
            getattr(type(module).forward, _MARK, False)
            and module.__dict__.get("_unsloth_block_mode", None) != _OFF
        ):
            return True
    return False


def _is_moe(config):
    config = getattr(config, "get_text_config", lambda: config)()
    for key in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"):
        if (getattr(config, key, None) or 0) > 1:
            return True
    return False


def _is_4bit(model):
    if getattr(model, "is_loaded_in_4bit", False):
        return True
    return any(type(m).__name__ == "Linear4bit" for m in model.modules())


def _total_microbatches(config, n_rows):
    bs = int(getattr(config, "per_device_train_batch_size", 1) or 1)
    ga = int(getattr(config, "gradient_accumulation_steps", 1) or 1)
    max_steps = getattr(config, "max_steps", -1)
    if max_steps is not None and max_steps > 0:
        return int(max_steps) * ga
    epochs = float(getattr(config, "num_train_epochs", 1) or 1)
    if not n_rows:
        return None
    return int(epochs * math.ceil(n_rows / bs))


def _effective_max_length(model, config):
    """The row cap training will use. This runs before the generated SFTTrainer init, which
    replaces SFTConfig's default `max_length` with the model's `max_seq_length` and keeps only a
    caller-set value below it (unsloth/models/rl.py), so mirror that precedence here."""
    given = getattr(config, "max_length", None)
    default = None
    try:
        import dataclasses
        for field in dataclasses.fields(type(config)):
            if field.name == "max_length":
                default = field.default
    except Exception:
        pass
    explicit = given if (given or 0) > 0 and given != default else None
    model_length = getattr(model, "max_seq_length", None)
    if model_length:
        return min(int(model_length), int(explicit)) if explicit else int(model_length)
    return given


def decide_auto_padding_free(
    model,
    config,
    train_dataset,
    processing_class,
    formatting_func = None,
):
    """True = padding-free (old default), False = padded batches. Never raises."""
    mode = padding_choice_mode()
    if mode == "padding_free":
        return True, "UNSLOTH_PADDING_CHOICE=padding_free"
    if mode == "padded":
        return False, "UNSLOTH_PADDING_CHOICE=padded"
    try:
        return _decide(model, config, train_dataset, processing_class, formatting_func)
    except Exception as exc:  # the old default on any surprise
        logger.info(f"Unsloth: padding choice skipped ({exc}).")
        return True, f"padding choice skipped ({type(exc).__name__})"


def _decide(model, config, train_dataset, processing_class, formatting_func):
    import torch

    if model is None or isinstance(model, str) or not hasattr(model, "modules"):
        return True, "model not loaded"
    # Measured only on Linux + NVIDIA CUDA. ROCm / HIP (torch.cuda.is_available() is True there),
    # Windows (slower host, triton-windows compile), XPU, MPS and CPU keep padding-free unchanged.
    if getattr(torch.version, "hip", None):
        return True, "ROCm / HIP device (not measured)"
    if sys.platform == "win32":
        return True, "Windows (not measured)"
    if not torch.cuda.is_available() or not getattr(torch.version, "cuda", None):
        return True, "not an NVIDIA CUDA device"
    if int(os.environ.get("WORLD_SIZE", "1") or 1) > 1:
        return True, "distributed run"
    model_config = getattr(model, "config", None)
    if model_config is None or _is_moe(model_config):
        return True, "MoE or unknown config"
    if not _block_compile_installed(model):
        return True, "per-block compile not installed"
    bs = int(getattr(config, "per_device_train_batch_size", 1) or 1)
    lengths = _sample_lengths(train_dataset, config, processing_class, formatting_func)
    if not lengths:
        return True, "dataset lengths unknown"
    est = estimate_microbatch_tokens(lengths, bs, _effective_max_length(model, config))
    if est is None:
        return True, "dataset lengths unknown"
    _, padded_tokens, padded_pairs = est
    from .mfu import count_attention_flops_per_pair, count_linear_flops_per_token, get_peak_tflops

    linear, _ = count_linear_flops_per_token(model, gradient_checkpointing = True)
    attention = count_attention_flops_per_pair(model_config, gradient_checkpointing = True)
    dtype = torch.bfloat16 if getattr(config, "bf16", False) else torch.float16
    device = next((p.device for p in model.parameters() if p.device.type == "cuda"), None)
    peak = get_peak_tflops(device, dtype)
    capability = torch.cuda.get_device_capability(device)
    try:
        n_rows = len(train_dataset)
    except Exception:
        n_rows = None
    use_pf, reason, _ = choose_padding(
        linear,
        attention,
        padded_tokens,
        padded_pairs,
        peak,
        total_microbatches = _total_microbatches(config, n_rows),
        load_in_4bit = _is_4bit(model),
        capability = capability,
    )
    return use_pf, reason
