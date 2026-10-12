# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Text generation for text-diffusion LMs: one sampler per DiffusionProfile name, each the family's published
decoding recipe. Families shipping a sampler in their modeling code (Dream, LLaDA2, Nemotron-Labs-Diffusion,
DiffusionGemma) are called as is; LLaDA / LLaDA-MoE and SDAR samplers live in separate scripts upstream and are
re-implemented here."""

import contextlib
import math

import torch
import torch.nn.functional as F

from .diffusion_profiles import resolve_diffusion_profile, unwrap_diffusion_model

__all__ = ["DiffusionGenerationCancelled", "generate_text", "diffusion_profile_name"]


class DiffusionGenerationCancelled(RuntimeError):
    """Raised out of generate_text when stop_callback() returned True mid-generation."""


_SAMPLERS = {}


def _register(*names):
    def wrap(function):
        for name in names:
            _SAMPLERS[name] = function
        return function

    return wrap


def diffusion_profile_name(model):
    """The DiffusionProfile name a loaded (possibly PEFT / DDP wrapped) model was routed with, or None."""
    base = unwrap_diffusion_model(model)
    for candidate in (model, base):
        name = getattr(candidate, "_unsloth_diffusion_profile", None)
        if name is not None:
            return getattr(name, "name", name)
    profile = resolve_diffusion_profile(getattr(base, "config", None))
    return profile.name if profile is not None else None


def _text_tokenizer(tokenizer):
    return getattr(tokenizer, "tokenizer", tokenizer)


_UNIFORM_NOISE = frozenset({"diffusion_gemma"})


def _profile_mask_id(base, tokenizer, name):
    from .diffusion_profiles import diffusion_profiles

    profile = next((p for p in diffusion_profiles() if p.name == name), None)
    candidates = (
        getattr(base, "_unsloth_mask_token_id", None),
        getattr(base, "_unsloth_diffusion_mask_token_id", None),
        getattr(getattr(base, "config", None), "mask_token_id", None),
        getattr(profile, "default_mask_token_id", None),
        (getattr(profile, "defaults", None) or {}).get("mask_token_id"),
        getattr(_text_tokenizer(tokenizer), "mask_token_id", None),
    )
    for value in candidates:
        if value is not None:
            return int(value)
    if name in _UNIFORM_NOISE:
        return None  # uniform-noise families (DiffusionGemma) have no [MASK] token to strip
    raise ValueError(f"Unsloth: no [MASK] token id for the {name} sampler.")


def _eos_ids(base, tokenizer):
    ids = []
    for source in (
        getattr(base, "generation_config", None),
        getattr(base, "config", None),
        _text_tokenizer(tokenizer),
    ):
        value = getattr(source, "eos_token_id", None)
        for item in value if isinstance(value, (list, tuple)) else (value,):
            if item is not None and int(item) not in ids:
                ids.append(int(item))
    return ids


_TURN_PROBE = "Unsloth turn probe"


def _turn_end_ids(tokenizer):
    """The special token the chat template writes right after an assistant reply. Dream's eos_token_id is only
    <|endoftext|>, but its template closes a turn with <|im_end|>, so a chat reply would run on past it."""
    text_tokenizer = _text_tokenizer(tokenizer)
    try:
        rendered = text_tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": _TURN_PROBE}],
            tokenize = False,
        )
        ids = text_tokenizer(rendered.split(_TURN_PROBE, 1)[1], add_special_tokens = False)[
            "input_ids"
        ]
    except Exception:
        return []
    special = set(getattr(text_tokenizer, "all_special_ids", None) or ())
    decoder = getattr(text_tokenizer, "added_tokens_decoder", None) or {}
    special.update(i for i, token in decoder.items() if getattr(token, "special", False))
    return [int(ids[0])] if ids and ids[0] in special else []


def _device(base):
    try:
        return next(base.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _encode(tokenizer, prompt, messages, device):
    if messages is not None:
        text = None
        for source in (tokenizer, _text_tokenizer(tokenizer)):
            try:
                text = source.apply_chat_template(
                    messages, add_generation_prompt = True, tokenize = False
                )
                break
            except Exception:
                continue
        if text is None:
            raise ValueError(
                "Unsloth: this tokenizer has no usable chat template; pass prompt= instead."
            )
        add_special_tokens = False
    elif prompt is not None:
        text, add_special_tokens = prompt, True
    else:
        raise ValueError("Unsloth: generate_text needs prompt= or messages=.")
    encoded = _text_tokenizer(tokenizer)(
        text, return_tensors = "pt", add_special_tokens = add_special_tokens
    )
    return encoded["input_ids"].to(device), encoded.get("attention_mask")


def _round_up(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def _contains_any(tokens, ids):
    return bool(ids) and bool(torch.isin(tokens, torch.tensor(ids, device = tokens.device)).any())


@_register("llada", "llada_moe")
@torch.no_grad()
def _llada_sample(
    base,
    tokenizer,
    input_ids,
    attention_mask,
    *,
    max_new_tokens,
    steps,
    block_length,
    temperature,
    mask_id,
    eos_ids,
    remasking = "low_confidence",
    eos_early_stop = True,
    **unused,
):
    """Semi-autoregressive low-confidence remasking (LLaDA generate.py, also LLaDA-MoE's model card):
    blocks of `block_length` masks, a linear schedule unmasking the most confident predictions each step."""
    block_length = int(block_length or 32)
    gen_length = _round_up(max_new_tokens, block_length)
    num_blocks = gen_length // block_length
    steps = _round_up(int(steps or gen_length), num_blocks)
    steps_per_block = steps // num_blocks
    batch, prompt_length = input_ids.shape
    x = torch.full(
        (batch, prompt_length + gen_length), mask_id, dtype = torch.long, device = input_ids.device
    )
    x[:, :prompt_length] = input_ids
    for block in range(num_blocks):
        start = prompt_length + block * block_length
        end = start + block_length
        remaining = (x[:, start:end] == mask_id).sum(dim = 1, keepdim = True)
        schedule = remaining // steps_per_block + (
            torch.arange(steps_per_block, device = x.device)[None, :] < remaining % steps_per_block
        )
        for step in range(steps_per_block):
            mask_index = x == mask_id
            logits = base(x).logits
            if temperature > 0:
                # Gumbel-max in float64: low precision noise hurts MDM generation quality (arXiv:2409.02908).
                noisy = logits.to(torch.float64)
                noisy = noisy.exp() / (-torch.log(torch.rand_like(noisy))) ** temperature
            else:
                noisy = logits
            x0 = noisy.argmax(dim = -1)
            if remasking == "low_confidence":
                probs = F.softmax(logits, dim = -1)
                x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device = x0.device)
            else:
                raise ValueError(f"Unsloth: unknown LLaDA remasking {remasking!r}.")
            x0_p[:, end:] = -math.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -math.inf)
            transfer = torch.zeros_like(x0, dtype = torch.bool)
            for row in range(batch):
                _, index = torch.topk(confidence[row], k = int(schedule[row, step]))
                transfer[row, index] = True
            x[transfer] = x0[transfer]
        if eos_early_stop and _contains_any(x[:, prompt_length:end], eos_ids):
            # Later blocks never change earlier ones, so stopping here only drops text after the EOS.
            return x[:, prompt_length:end]
    return x[:, prompt_length:]


def _block_sample(
    base,
    tokenizer,
    input_ids,
    attention_mask,
    *,
    max_new_tokens,
    steps,
    block_length,
    temperature,
    mask_id,
    eos_ids,
    mask_format,
    confidence_threshold,
    top_k = 0,
    top_p = 1.0,
    **unused,
):
    """Block diffusion decoding shared by SDAR (JetAstra/SDAR generate.py, MIT, low_confidence_dynamic) and
    LLaDA2 (its modeling generate, Apache-2.0): blocks aligned to absolute positions, block-causal attention,
    every token above `confidence_threshold` accepted at once, else the scheduled most confident ones. The
    visible prefix is recomputed each step (as LLaDA2 does) instead of SDAR's custom KV cache. temperature 0
    is greedy."""
    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + max_new_tokens + block_length - 1) // block_length
    total_length = num_blocks * block_length
    device = input_ids.device
    block_causal = torch.tril(torch.ones(num_blocks, num_blocks, device = device, dtype = torch.bool))
    allowed = block_causal.repeat_interleave(block_length, 0).repeat_interleave(block_length, 1)[
        None
    ]
    if mask_format == "additive":
        dtype = next(base.parameters()).dtype
        attention = torch.zeros(allowed.shape, dtype = dtype, device = device)
        attention = attention.masked_fill(~allowed, torch.finfo(dtype).min)[:, None]
    else:
        attention = allowed
    position_ids = torch.arange(total_length, device = device)[None]
    x = torch.full((input_ids.shape[0], total_length), mask_id, dtype = torch.long, device = device)
    x[:, :prompt_length] = input_ids
    schedule = torch.full((steps,), block_length // steps, dtype = torch.long)
    schedule[: block_length % steps] += 1
    for block in range(prompt_length // block_length, num_blocks):
        start, end = block * block_length, (block + 1) * block_length
        for step in range(steps + 1):
            mask_index = x[:, start:end] == mask_id
            if not bool(mask_index.any()):
                break
            logits = (
                base(
                    x[:, :end],
                    attention_mask = attention[..., :end, :end],
                    position_ids = position_ids[:, :end],
                )
                .logits[:, start:end]
                .float()
            )
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
            count = int(schedule[min(step, steps - 1)])
            transfer = torch.zeros_like(mask_index)
            for row in range(x.shape[0]):
                high = confidence[row] > confidence_threshold
                if int(high.sum()) >= count:
                    transfer[row] = high
                else:
                    # Never more than the masks left: the official top-k could overwrite prompt tokens sharing
                    # the first decoded block.
                    k = min(count, int(mask_index[row].sum()))
                    transfer[row, torch.topk(confidence[row], k).indices] = True
            window = x[:, start:end]
            window[transfer] = x0[transfer]
        if _contains_any(x[:, prompt_length:end], eos_ids):
            return x[:, prompt_length:end]
    return x[:, prompt_length:]


@_register("sdar")
@torch.no_grad()
def _sdar_sample(base, tokenizer, input_ids, attention_mask, *, steps, block_length, **kwargs):
    """SDAR model card / generate.py defaults: block 4, steps = block length, threshold 0.85."""
    block_length = int(block_length or getattr(base.config, "block_size", None) or 4)
    kwargs.setdefault("confidence_threshold", 0.85)
    return _block_sample(
        base,
        tokenizer,
        input_ids,
        attention_mask,
        steps = int(steps or block_length),
        block_length = block_length,
        mask_format = "bool",
        **kwargs,
    )


@_register("llada2")
@torch.no_grad()
def _llada2_sample(
    base,
    tokenizer,
    input_ids,
    attention_mask,
    *,
    steps,
    block_length,
    threshold = 0.95,
    **kwargs,
):
    """LLaDA2 model card defaults: block 32, steps 32, threshold 0.95. Its own generate builds a bfloat16 mask
    (fails on fp16 / fp32 weights) and samples even at temperature 0, so the shared block sampler runs instead."""
    kwargs.setdefault("confidence_threshold", threshold)
    return _block_sample(
        base,
        tokenizer,
        input_ids,
        attention_mask,
        steps = int(steps or 32),
        block_length = int(block_length or 32),
        mask_format = "additive",
        **kwargs,
    )


@_register("dream")
@torch.no_grad()
def _dream_sample(
    base,
    tokenizer,
    input_ids,
    attention_mask,
    *,
    max_new_tokens,
    steps,
    block_length,
    temperature,
    mask_id,
    eos_ids,
    top_p = None,
    alg = "entropy",
    alg_temp = 0.0,
    **unused,
):
    """Dream's own diffusion_generate (model card: steps = max_new_tokens, alg "entropy", alg_temp 0)."""
    if top_p is None and temperature > 0:
        top_p = 0.95
    out = base.diffusion_generate(
        input_ids,
        attention_mask = None if attention_mask is None else attention_mask.to(input_ids.device),
        max_new_tokens = max_new_tokens,
        steps = int(steps or max_new_tokens),
        temperature = temperature,
        top_p = top_p,
        alg = alg,
        alg_temp = alg_temp,
        output_history = False,
        return_dict_in_generate = True,
    )
    return out.sequences[:, input_ids.shape[1] :]


@_register("nemotron_labs_diffusion")
@torch.no_grad()
def _nemotron_sample(
    base,
    tokenizer,
    input_ids,
    attention_mask,
    *,
    max_new_tokens,
    steps,
    block_length,
    temperature,
    mask_id,
    eos_ids,
    threshold = None,
    **unused,
):
    """Nemotron-Labs-Diffusion's own generate: block diffusion with a causal context between blocks."""
    block_length = int(block_length or getattr(base.config, "block_size", None) or 32)
    output_ids, _nfe = base.generate(
        input_ids,
        max_new_tokens = _round_up(max_new_tokens, block_length),
        block_length = block_length,
        threshold = threshold,
        temperature = temperature,
        eos_token_id = eos_ids[0] if eos_ids else None,
    )
    return output_ids[:, input_ids.shape[1] :]


@_register("diffusion_gemma")
@torch.no_grad()
def _diffusion_gemma_sample(
    base,
    tokenizer,
    input_ids,
    attention_mask,
    *,
    max_new_tokens,
    steps,
    block_length,
    temperature,
    mask_id,
    eos_ids,
    **unused,
):
    """transformers' DiffusionGemma generate (entropy-bound sampler, its own temperature schedule)."""
    kwargs = {"max_new_tokens": max_new_tokens}
    if steps is not None:
        kwargs["max_denoising_steps"] = int(steps)
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(input_ids.device)
    out = base.generate(input_ids, **kwargs)
    return getattr(out, "sequences", out)[:, input_ids.shape[1] :]


@contextlib.contextmanager
def _cancellation(base, stop_callback):
    if stop_callback is None:
        yield
        return

    def check(
        module,
        args,
        kwargs = None,
    ):
        if stop_callback():
            raise DiffusionGenerationCancelled("Unsloth: diffusion generation cancelled.")

    # Every embedding table: DiffusionGemma's generate calls encoder and decoder directly, bypassing forward.
    targets = [m for m in base.modules() if isinstance(m, torch.nn.Embedding)] or [base]
    handles = [m.register_forward_pre_hook(check, with_kwargs = True) for m in targets]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def _restore_attention_modes(base):
    """Nemotron-Labs-Diffusion's generate flips each layer's `self_attn.diffusion_lm` for its causal prefill
    and flips it back at the end; a cancelled generate would leave the model causal. Put them back."""
    saved = [
        (module, module.diffusion_lm)
        for module in base.modules()
        if isinstance(getattr(module, "diffusion_lm", None), bool)
    ]
    try:
        yield
    finally:
        for module, value in saved:
            module.diffusion_lm = value


@contextlib.contextmanager
def _eval_mode(module):
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


def generate_text(
    model,
    tokenizer,
    prompt = None,
    messages = None,
    max_new_tokens = 256,
    steps = None,
    block_length = None,
    temperature = 0.0,
    stop_callback = None,
    seed = None,
    **kwargs,
):
    """Decode one completion with the model family's diffusion sampler and return it as text (prompt
    removed, cut at the first EOS, special tokens skipped). stop_callback() is checked before every forward;
    returning True raises DiffusionGenerationCancelled. `steps` / `block_length` default to each family's
    published settings; extra kwargs go to the family sampler (e.g. remasking, threshold, top_p, alg)."""
    name = diffusion_profile_name(model)
    sampler = _SAMPLERS.get(name)
    if sampler is None:
        raise ValueError(
            f"Unsloth: no diffusion sampler for {name!r}; supported: {sorted(_SAMPLERS)}."
        )
    base = unwrap_diffusion_model(model)
    device = _device(base)
    input_ids, attention_mask = _encode(tokenizer, prompt, messages, device)
    eos_ids = _eos_ids(base, tokenizer)
    if messages is not None:
        eos_ids += [i for i in _turn_end_ids(tokenizer) if i not in eos_ids]
    mask_id = _profile_mask_id(base, tokenizer, name)
    # manual_seed reseeds every CUDA device, so snapshot every device a sharded model uses.
    cuda_devices = sorted({p.device.index for p in base.parameters() if p.device.type == "cuda"})
    rng = (
        torch.random.fork_rng(devices = cuda_devices)
        if seed is not None
        else contextlib.nullcontext()
    )
    with rng, _eval_mode(base), _restore_attention_modes(base), _cancellation(base, stop_callback):
        if seed is not None:
            torch.manual_seed(int(seed))
        generated = sampler(
            base,
            tokenizer,
            input_ids,
            attention_mask,
            max_new_tokens = int(max_new_tokens),
            steps = steps,
            block_length = block_length,
            temperature = float(temperature or 0.0),
            mask_id = mask_id,
            eos_ids = eos_ids,
            **kwargs,
        )
    tokens = generated[0][: int(max_new_tokens)].tolist()
    for index, token in enumerate(tokens):
        if token in eos_ids:
            tokens = tokens[:index]
            break
    if mask_id is not None:
        tokens = [token for token in tokens if token != mask_id]
    return _text_tokenizer(tokenizer).decode(tokens, skip_special_tokens = True).strip()
