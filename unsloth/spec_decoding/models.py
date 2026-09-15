# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thin model/tokenizer loading and forward helpers. Plain `transformers` by default so
it runs on a laptop; optional Unsloth fast path for a CUDA box."""

from __future__ import annotations

import warnings

import torch

from .sampling import SamplingParams, sampling_distribution


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, requested: str | None = None) -> torch.dtype:
    """Default to full precision off-CUDA, as the conservative choice for a tool whose
    entire output is a comparison of two distributions.

    The cost of *not* doing so is small: :func:`sequence_logits` casts to fp32 before any
    softmax, so half precision only perturbs the forward pass itself. Measured on
    gpt2 <= distilgpt2, fp16 moves alpha by 0.001 (0.71113 -> 0.71216) against a
    per-sequence spread of 0.084. Pass ``dtype="float16"`` when memory or speed matters --
    a 1.5B target plus a 0.5B draft will not fit in 16GB at fp32.
    """
    if requested:
        return getattr(torch, requested)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model_and_tokenizer(
    name: str,
    *,
    device: str | None = None,
    dtype: str | None = None,
    use_unsloth: bool = False,
    max_seq_length: int = 4096,
):
    """Return ``(model, tokenizer, device)`` with the model in eval mode and frozen."""
    device = pick_device(device)
    torch_dtype = pick_dtype(device, dtype)

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel  # noqa: PLC0415
        except Exception as e:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "use_unsloth=True but `unsloth` could not be imported "
                f"(it needs a CUDA/ROCm/XPU GPU): {e}"
            ) from e
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = name,
            max_seq_length = max_seq_length,
            dtype = torch_dtype,
            load_in_4bit = False,
        )
        FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        from transformers import __version__ as tf_version  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(name)
        # transformers >= 5 renamed torch_dtype -> dtype. The auto classes take
        # **kwargs, so the name cannot be probed from the signature -- check the version.
        kw = "dtype" if int(tf_version.split(".")[0]) >= 5 else "torch_dtype"
        model = AutoModelForCausalLM.from_pretrained(name, **{kw: torch_dtype})
        model.to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer, device


def check_vocab_compatible(target_tok, draft_tok) -> None:
    """Warn (don't fail) if the two tokenizers look incompatible for speculative decoding."""
    tv = getattr(target_tok, "vocab_size", None)
    dv = getattr(draft_tok, "vocab_size", None)
    if tv is not None and dv is not None and tv != dv:
        warnings.warn(
            f"target vocab_size={tv} != draft vocab_size={dv}; logits will be truncated "
            "to the common prefix. Exact only if the difference is padding tokens.",
            stacklevel = 2,
        )
    t_name = getattr(target_tok, "name_or_path", "")
    d_name = getattr(draft_tok, "name_or_path", "")
    try:
        probe = "The quick brown fox jumps over the lazy dog. 12345 — café."
        if target_tok(probe)["input_ids"] != draft_tok(probe)["input_ids"]:
            warnings.warn(
                f"target ({t_name}) and draft ({d_name}) tokenize a probe string "
                "differently. Standard speculative decoding assumes a shared tokenizer; "
                "acceptance numbers will be misleading.",
                stacklevel = 2,
            )
    except Exception:  # pragma: no cover - tokenizer quirks
        pass


@torch.no_grad()
def sequence_logits(model, input_ids: torch.Tensor, device: str) -> torch.Tensor:
    """Full ``[seq_len, vocab]`` float32 logits for a single 1-D sequence."""
    ids = input_ids.to(device).long().unsqueeze(0)
    out = model(ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits[0].float().cpu()


@torch.no_grad()
def generate_continuation(
    model,
    input_ids: torch.Tensor,
    n_tokens: int,
    sampling: SamplingParams,
    device: str,
    *,
    seed: int = 0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Extend ``input_ids`` with ``n_tokens`` sampled from ``model`` under ``sampling``.

    Uses ``model.generate`` (KV-cached) when available, otherwise a plain forward loop so
    stand-in models work too. Returns the full 1-D sequence (prompt + continuation).
    """
    ids = input_ids.long().flatten()
    if n_tokens <= 0:
        return ids

    if hasattr(model, "generate"):
        torch.manual_seed(seed)
        eos = (
            eos_token_id
            if eos_token_id is not None
            else getattr(model.config, "eos_token_id", None)
        )
        pad = getattr(model.config, "pad_token_id", None)
        if pad is None:
            pad = eos if isinstance(eos, int) else None
        kwargs = dict(
            max_new_tokens = n_tokens,
            do_sample = not sampling.is_greedy,
            pad_token_id = pad,
            eos_token_id = eos,
        )
        if not sampling.is_greedy:
            kwargs.update(
                temperature = sampling.temperature, top_k = sampling.top_k or 0, top_p = sampling.top_p
            )
        batch = ids.unsqueeze(0).to(device)
        # pass an explicit all-ones mask: the sequence is unpadded, and without it
        # transformers warns whenever pad_token_id == eos_token_id (Qwen, Llama, ...)
        out = model.generate(batch, attention_mask = torch.ones_like(batch), **kwargs)
        return out[0].cpu().long()

    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_tokens):
        probs = sampling_distribution(sequence_logits(model, ids, device)[-1], sampling)
        tok = (
            int(probs.argmax())
            if sampling.is_greedy
            else int(torch.multinomial(probs, 1, generator = gen))
        )
        ids = torch.cat([ids, torch.tensor([tok], dtype = torch.long)])
        if eos_token_id is not None and tok == eos_token_id:
            break
    return ids
