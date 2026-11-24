# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AUTOMATIC1111 prompt weighting for CLIP SD/SDXL pipelines; single-chunk weight 1 matches diffusers."""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from loggers import get_logger

logger = get_logger(__name__)

_ATTENTION_RE = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)
_BREAK_RE = re.compile(r"\s*\bBREAK\b\s*", re.S)

ROUND_MULTIPLIER = 1.1
SQUARE_MULTIPLIER = 1 / 1.1
BREAK_WEIGHT = -1.0
CHUNK_TOKENS = 75
COMMA_BACKTRACK = 20
# webui and ComfyUI chunk without a limit; 32 chunks (2400 tokens) bounds what one request can encode.
MAX_CHUNKS = 32
# Text read per prompt (32 chars/token over the cap) bounds parsing however large the body is.
MAX_PROMPT_CHARS = MAX_CHUNKS * CHUNK_TOKENS * 32


def parse_prompt_attention(text: str) -> list[list[Any]]:
    """``[[fragment, weight], ...]`` with A1111 semantics; a BREAK is ``["BREAK", -1.0]``."""
    res: list[list[Any]] = []
    round_open: list[int] = []
    square_open: list[int] = []

    def multiply(start: int, factor: float) -> None:
        for pos in range(start, len(res)):
            res[pos][1] *= factor

    for match in _ATTENTION_RE.finditer(text):
        token = match.group(0)
        weight = match.group(1)
        if token.startswith("\\"):
            res.append([token[1:], 1.0])
        elif token == "(":
            round_open.append(len(res))
        elif token == "[":
            square_open.append(len(res))
        elif weight is not None and round_open:
            try:
                value = float(weight)
            except ValueError:
                # A1111 raises on a malformed number such as "(a:1.2.3)"; keep the text as typed instead.
                start = round_open.pop()
                res.insert(start, ["(", 1.0])
                square_open[:] = [pos + 1 if pos >= start else pos for pos in square_open]
                res.append([token, 1.0])
            else:
                multiply(round_open.pop(), value)
        elif token == ")" and round_open:
            multiply(round_open.pop(), ROUND_MULTIPLIER)
        elif token == "]" and square_open:
            multiply(square_open.pop(), SQUARE_MULTIPLIER)
        else:
            for i, part in enumerate(_BREAK_RE.split(token)):
                if i > 0:
                    res.append(["BREAK", BREAK_WEIGHT])
                res.append([part, 1.0])
    for pos in round_open:
        multiply(pos, ROUND_MULTIPLIER)
    for pos in square_open:
        multiply(pos, SQUARE_MULTIPLIER)
    if not res:
        res = [["", 1.0]]
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res


def has_prompt_syntax(text: Optional[str]) -> bool:
    """True when parsing changes ``text``: any bracket, escape or BREAK. Plain prompts keep the pipeline's own path."""
    if not text:
        return False
    text = text[:MAX_PROMPT_CHARS]
    parsed = parse_prompt_attention(text)
    return not (len(parsed) == 1 and parsed[0][0] == text and parsed[0][1] == 1.0)


def exceeds_one_chunk(pipe: Any, text: Optional[str]) -> bool:
    """True when ``text`` holds more than 75 CLIP tokens in any of the pipeline's tokenizers, i.e. when
    ``encode_prompt`` would truncate it (77 = ``<bos>`` + 75 + ``<eos>``). Counted after the pipeline's own
    ``maybe_convert_prompt``, as ``encode_prompt`` tokenizes."""
    if not text:
        return False
    text = text[:MAX_PROMPT_CHARS]
    convert = getattr(pipe, "maybe_convert_prompt", None)
    for tokenizer, _ in _encoders(pipe):
        converted = convert(text, tokenizer) if convert is not None else text
        warned = getattr(tokenizer, "deprecation_warnings", None)
        if isinstance(warned, dict):
            warned["sequence-length-is-longer-than-the-specified-maximum"] = True
        ids = tokenizer(converted, truncation = False, add_special_tokens = False, verbose = False)[
            "input_ids"
        ]
        if len(ids) > CHUNK_TOKENS:
            return True
    return False


def _special_ids(tokenizer: Any) -> tuple[int, int, int, Optional[int]]:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    comma = tokenizer.get_vocab().get(",</w>")
    return int(bos), int(eos), int(pad), comma


def tokenize_chunks(
    text: str,
    tokenizer: Any,
    *,
    weighted: bool = True,
    convert: Any = None,
) -> list[tuple[list[int], list[float]]]:
    """A1111 ``tokenize_line``: ``[(77 token ids, 77 weights), ...]``; ``weighted=False`` still chunks.
    Stops at ``MAX_CHUNKS``: nothing past the cap is tokenized or kept."""
    truncated = len(text) > MAX_PROMPT_CHARS
    text = text[:MAX_PROMPT_CHARS]
    parsed = parse_prompt_attention(text) if weighted else [[text, 1.0]]
    bos, eos, pad, comma = _special_ids(tokenizer)
    warned = getattr(tokenizer, "deprecation_warnings", None)
    if isinstance(warned, dict):
        warned["sequence-length-is-longer-than-the-specified-maximum"] = True
    chunks: list[tuple[list[int], list[float]]] = []
    tokens: list[int] = []
    weights: list[float] = []
    last_comma = -1

    class _Full(Exception):
        pass

    def close_chunk() -> None:
        nonlocal tokens, weights, last_comma
        if len(chunks) >= MAX_CHUNKS:
            raise _Full
        fill = CHUNK_TOKENS - len(tokens)
        ids = [bos] + tokens + [eos] * fill + [eos]
        mults = [1.0] + weights + [1.0] * fill + [1.0]
        # webui process_tokens: everything after the first <eos> is the pad id (OpenCLIP's differs from <eos>).
        first_eos = ids.index(eos, 1)
        ids = ids[: first_eos + 1] + [pad] * (len(ids) - first_eos - 1)
        chunks.append((ids, mults))
        tokens, weights, last_comma = [], [], -1

    try:
        for fragment, weight in parsed:
            if fragment == "BREAK" and weight == BREAK_WEIGHT:
                close_chunk()
                continue
            if convert is not None and fragment:
                fragment = convert(fragment, tokenizer)
            ids = tokenizer(fragment, truncation = False, add_special_tokens = False, verbose = False)[
                "input_ids"
            ]
            for token in ids:
                if token == comma:
                    last_comma = len(tokens)
                elif (
                    COMMA_BACKTRACK
                    and len(tokens) == CHUNK_TOKENS
                    and last_comma != -1
                    and len(tokens) - last_comma <= COMMA_BACKTRACK
                ):
                    cut = last_comma + 1
                    carry_tokens, carry_weights = tokens[cut:], weights[cut:]
                    tokens, weights = tokens[:cut], weights[:cut]
                    close_chunk()
                    tokens, weights = carry_tokens, carry_weights
                if len(tokens) == CHUNK_TOKENS:
                    close_chunk()
                if len(chunks) >= MAX_CHUNKS:
                    raise _Full
                tokens.append(int(token))
                weights.append(float(weight))
        if tokens or not chunks:
            close_chunk()
    except _Full:
        truncated = True
    if truncated:
        logger.warning(
            "Prompt spans more than %d chunks of %d tokens; encoding the first %d and dropping the rest.",
            MAX_CHUNKS,
            CHUNK_TOKENS,
            MAX_CHUNKS,
        )
    return chunks


def _empty_chunk(tokenizer: Any) -> tuple[list[int], list[float]]:
    bos, eos, pad, _ = _special_ids(tokenizer)
    return [bos, eos] + [pad] * CHUNK_TOKENS, [1.0] * (CHUNK_TOKENS + 2)


def apply_emphasis(z: Any, weights: Any) -> Any:
    """webui ``EmphasisOriginal.after_transformers``: scale per token, then restore the chunk's mean."""
    original_mean = z.mean()
    z = z * weights.reshape(weights.shape + (1,)).expand(z.shape)
    new_mean = z.mean()
    return z * (original_mean / new_mean)


def _encoders(pipe: Any) -> list[tuple[Any, Any]]:
    pairs = []
    for tok_attr, enc_attr in (("tokenizer", "text_encoder"), ("tokenizer_2", "text_encoder_2")):
        tokenizer = getattr(pipe, tok_attr, None)
        encoder = getattr(pipe, enc_attr, None)
        if tokenizer is not None and encoder is not None:
            pairs.append((tokenizer, encoder))
    return pairs


def is_sdxl_like(pipe: Any) -> bool:
    """Two CLIP encoders whose penultimate layers concatenate (SDXL); else the SD 1.x single-encoder layout."""
    return getattr(pipe, "text_encoder_2", None) is not None


def _encode_chunk(
    pipe: Any, encoder: Any, ids: list[int], clip_skip: Optional[int], device: Any, sdxl: bool
) -> tuple[Any, Any]:
    """One 77-token chunk through ``encoder``, picking the layer exactly as the pipeline's ``encode_prompt`` does."""
    import torch

    input_ids = torch.tensor([ids], dtype = torch.long).to(device)
    if sdxl:
        out = encoder(input_ids, output_hidden_states = True)
        pooled = out[0] if out[0].ndim == 2 else None
        if clip_skip is None:
            z = out.hidden_states[-2]
        else:
            z = out.hidden_states[-(clip_skip + 2)]
        return z, pooled
    mask = None
    if getattr(getattr(encoder, "config", None), "use_attention_mask", False):
        mask = torch.ones_like(input_ids)
    if clip_skip is None:
        z = encoder(input_ids, attention_mask = mask)[0]
    else:
        out = encoder(input_ids, attention_mask = mask, output_hidden_states = True)
        # transformers 5 flattened CLIPTextModel (no ``.text_model``); 4.x nests it.
        final_norm = getattr(encoder, "text_model", encoder).final_layer_norm
        z = final_norm(out[-1][-(clip_skip + 1)])
    return z, None


def _encode_texts(
    pipe: Any,
    texts: Sequence[str],
    *,
    weighted: bool,
    clip_skip: Optional[int],
    device: Any,
    min_chunks: int = 1,
) -> tuple[Any, Any, int]:
    """``(embeds [B, 77k, D], pooled [B, P] or None, k)`` for ``texts``, every row padded to the longest with
    empty-prompt chunks (webui ``process_texts`` / ``empty_chunk``)."""
    import torch

    sdxl = is_sdxl_like(pipe)
    convert = getattr(pipe, "maybe_convert_prompt", None)
    pairs = _encoders(pipe)
    per_encoder = [
        [tokenize_chunks(t, tok, weighted = weighted, convert = convert) for t in texts]
        for tok, _ in pairs
    ]
    count = max([min_chunks] + [len(chunks) for rows in per_encoder for chunks in rows])
    rows_out = []
    pooled_rows = []
    for row in range(len(texts)):
        enc_parts = []
        pooled = None
        for (tokenizer, encoder), rows in zip(pairs, per_encoder):
            chunks = list(rows[row]) + [_empty_chunk(tokenizer)] * (count - len(rows[row]))
            zs = []
            for index, (ids, mults) in enumerate(chunks):
                z, chunk_pooled = _encode_chunk(pipe, encoder, ids, clip_skip, device, sdxl)
                if index == 0 and pooled is None and chunk_pooled is not None:
                    pooled = chunk_pooled
                weights = torch.tensor(mults, dtype = torch.float32, device = z.device)
                if bool((weights != 1.0).any()):
                    z = apply_emphasis(z.float(), weights.unsqueeze(0)).to(z.dtype)
                zs.append(z)
            enc_parts.append(torch.cat(zs, dim = 1))
        rows_out.append(torch.cat(enc_parts, dim = -1) if sdxl else enc_parts[0])
        pooled_rows.append(pooled)
    embeds = torch.cat(rows_out, dim = 0)
    pooled_out = torch.cat(pooled_rows, dim = 0) if all(p is not None for p in pooled_rows) else None
    return embeds, pooled_out, count


def _target_dtype(pipe: Any) -> Any:
    encoder_2 = getattr(pipe, "text_encoder_2", None)
    if encoder_2 is not None:
        return encoder_2.dtype
    encoder = getattr(pipe, "text_encoder", None)
    if encoder is not None:
        return encoder.dtype
    return pipe.unet.dtype


def encode_weighted_prompts(
    pipe: Any,
    prompts: Sequence[str],
    negative_prompts: Optional[Sequence[str]],
    *,
    guidance_engaged: bool,
    weighted: bool = True,
    clip_skip: Optional[int] = None,
) -> dict[str, Any]:
    """Pipeline kwargs (``prompt_embeds``, ``pooled_prompt_embeds`` and their negatives) for ``prompts``. Both share one chunk count (CFG batches them)."""
    import torch

    device = getattr(pipe, "_execution_device", None) or pipe.device
    dtype = _target_dtype(pipe)
    embeds, pooled, count = _encode_texts(
        pipe, list(prompts), weighted = weighted, clip_skip = clip_skip, device = device
    )
    kwargs: dict[str, Any] = {}
    neg_embeds = neg_pooled = None
    if guidance_engaged:
        zero_out = negative_prompts is None and bool(
            getattr(getattr(pipe, "config", None), "force_zeros_for_empty_prompt", False)
        )
        if not zero_out:
            texts = list(negative_prompts) if negative_prompts is not None else [""] * len(prompts)
            neg_embeds, neg_pooled, neg_count = _encode_texts(
                pipe,
                texts,
                weighted = weighted,
                clip_skip = clip_skip,
                device = device,
                min_chunks = count,
            )
            if neg_count > count:
                embeds, pooled, count = _encode_texts(
                    pipe,
                    list(prompts),
                    weighted = weighted,
                    clip_skip = clip_skip,
                    device = device,
                    min_chunks = neg_count,
                )
    embeds = embeds.to(dtype = dtype, device = device)
    kwargs["prompt_embeds"] = embeds
    if pooled is not None:
        kwargs["pooled_prompt_embeds"] = pooled.to(device = device)
    if guidance_engaged:
        if neg_embeds is None:
            neg_embeds = torch.zeros_like(embeds)
            neg_pooled = torch.zeros_like(pooled) if pooled is not None else None
        kwargs["negative_prompt_embeds"] = neg_embeds.to(dtype = dtype, device = device)
        if neg_pooled is not None:
            kwargs["negative_pooled_prompt_embeds"] = neg_pooled.to(device = device)
    return kwargs
