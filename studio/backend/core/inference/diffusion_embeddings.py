# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Textual-inversion embeddings (EasyNegative and friends) for the CLIP-conditioned image families.

Files live in ``<studio_home>/embeddings/diffusion`` (beside ``loras/diffusion``). As in AUTOMATIC1111 the
filename stem is the trigger: a prompt or negative prompt naming it (any case, as a whole word, or ComfyUI's
``embedding:NAME``) uses the embedding for that generation. safetensors only (``.pt`` is pickled).

Formats, as diffusers' ``load_textual_inversion`` reads them (docs: "Textual inversion", SDXL section):
  * SD 1.x: one tensor ``[n, 768]`` (A1111 ``emb_params`` or a single diffusers ``<token>`` key);
  * SDXL: ``clip_l [n, 768]`` for ``text_encoder`` / ``tokenizer`` and ``clip_g [n, 1280]`` for
    ``text_encoder_2`` / ``tokenizer_2``.

Applying one never reloads the pipeline and never mutates a loaded component: each tokenizer is swapped for
a copy carrying the added tokens (``tok``, ``tok_1``, ... as diffusers names multi-vector embeddings, so the
pipeline's own ``maybe_convert_prompt`` expands them), and each text encoder's input embedding is wrapped by a
module that serves the new ids and calls the original for every other id. ``restore`` puts the original
objects back, so a generation that names no embedding runs the unchanged modules. The denoiser is untouched.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from utils.paths.path_utils import is_appledouble_metadata
from utils.paths.storage_roots import account_path

KIND_SD1 = "sd1"
KIND_SD2 = "sd2"
KIND_SDXL = "sdxl"
_EXTS = (".safetensors",)
_MAX_BYTES = 64 * 1024 * 1024
_MAX_VECTORS = 75
_PROMPT_KEYS = ("prompt", "prompt_2", "negative_prompt", "negative_prompt_2")
# Internal token prefix: lower case so a lowercasing tokenizer normaliser cannot split it.
_TOKEN_PREFIX = "<unsloth-ti-"
# (tokenizer attr, text encoder attr, SDXL tensor key)
_ENCODER_SLOTS = (
    ("tokenizer", "text_encoder", "clip_l"),
    ("tokenizer_2", "text_encoder_2", "clip_g"),
)


def embeddings_dir() -> Path:
    d = account_path("embeddings/diffusion")
    d.mkdir(parents = True, exist_ok = True)
    return d


@dataclass(frozen = True)
class EmbeddingEntry:
    id: str  # filename stem: the trigger word
    path: str
    kind: Optional[str]  # KIND_* or None when the tensors match no known encoder
    vectors: int
    size_bytes: int


def _shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Tensor shapes from the safetensors header, without reading the data."""
    from safetensors import safe_open

    out: dict[str, tuple[int, ...]] = {}
    with safe_open(str(path), framework = "pt") as f:
        for key in f.keys():
            out[key] = tuple(f.get_slice(key).get_shape())
    return out


def _vectors(shape: tuple[int, ...]) -> int:
    return shape[0] if len(shape) == 2 else 1 if len(shape) == 1 else 0


def classify(shapes: dict[str, tuple[int, ...]]) -> tuple[Optional[str], int]:
    """``(kind, vectors)`` of an embedding file's tensors; kind None when it is not one we can apply."""
    if "clip_l" in shapes and "clip_g" in shapes:
        l, g = shapes["clip_l"], shapes["clip_g"]
        if l[-1:] == (768,) and g[-1:] == (1280,) and _vectors(l) == _vectors(g) > 0:
            return KIND_SDXL, _vectors(l)
        return None, 0
    if len(shapes) != 1:
        return None, 0
    (shape,) = shapes.values()
    n = _vectors(shape)
    if n <= 0:
        return None, 0
    if shape[-1] == 768:
        return KIND_SD1, n
    if shape[-1] == 1024:
        return KIND_SD2, n
    return None, 0


def _account_allows():
    from .diffusion_lora import _account_allows as allows
    return allows()


def list_embeddings(kind: Optional[str] = None) -> list[EmbeddingEntry]:
    """Embedding files in ``embeddings_dir()``, optionally only those of ``kind``. Never raises on one bad file."""
    root = embeddings_dir()
    allows = _account_allows()
    try:
        children = sorted(root.iterdir())
    except OSError:
        return []
    entries: list[EmbeddingEntry] = []
    seen: set[str] = set()
    for p in children:
        if p.suffix.lower() not in _EXTS or is_appledouble_metadata(p):
            continue
        try:
            if not p.is_file() or not allows(p):
                continue
            size = p.stat().st_size
            if size > _MAX_BYTES:
                continue
            found_kind, vectors = classify(_shapes(p))
        except Exception:  # noqa: BLE001 - one unreadable file never fails the listing
            continue
        if vectors > _MAX_VECTORS:
            continue
        # Case-insensitive triggers: the first of two stems differing only in case wins.
        if p.stem.lower() in seen:
            continue
        seen.add(p.stem.lower())
        if kind is not None and found_kind != kind:
            continue
        entries.append(
            EmbeddingEntry(
                id = p.stem,
                path = str(p),
                kind = found_kind,
                vectors = vectors,
                size_bytes = size,
            )
        )
    return entries


def _trigger_pattern(names: Iterable[str]) -> Optional[re.Pattern]:
    # Longest first so "bad-hands-5" wins over "bad"; one pass, so a replacement is never re-matched.
    ordered = sorted({n for n in names if n}, key = lambda n: (-len(n), n))
    if not ordered:
        return None
    alternation = "|".join(re.escape(n) for n in ordered)
    return re.compile(
        rf"(?<![\w-])(?:embedding:)?({alternation})(?![\w-])",
        re.IGNORECASE,
    )


def _texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def find_triggers(texts: Iterable[str], entries: Iterable[EmbeddingEntry]) -> list[EmbeddingEntry]:
    """The entries whose trigger appears in any of ``texts``, in listing order."""
    by_lower = {e.id.lower(): e for e in entries}
    pattern = _trigger_pattern(e.id for e in by_lower.values())
    if pattern is None:
        return []
    hit: set[str] = set()
    for text in texts:
        for m in pattern.finditer(text or ""):
            hit.add(m.group(1).lower())
    return [e for key, e in by_lower.items() if key in hit]


def _internal_token(index: int, stem: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "emb"
    return f"{_TOKEN_PREFIX}{index}-{slug}>"


def _read_tensors(path: str) -> dict[str, Any]:
    """The file's tensors, read through a handle pinned to the account-readable file."""
    from safetensors.torch import load as load_bytes

    from .diffusion_lora import _open_pinned

    with _open_pinned(Path(path)) as f:
        data = f.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise ValueError(f"Embedding '{Path(path).name}' is too large.")
    return load_bytes(data)


def _fingerprint(path: str) -> str:
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    return hashlib.sha1(f"{path}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16]


def _vectors_for_slot(tensors: dict[str, Any], kind: str, sdxl_key: str) -> Any:
    import torch

    if kind == KIND_SDXL:
        t = tensors[sdxl_key]
    else:
        (t,) = tensors.values()
    t = t.detach()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2:
        raise ValueError("embedding tensor must be [vectors, width]")
    return t.to(torch.float32)


def _fits(tensors: dict[str, Any], kind: str, widths: list[int]) -> bool:
    """Whether ``tensors`` is an embedding for these text encoders (widths in slot order)."""

    def shape(t: Any) -> tuple[int, ...]:
        return tuple(getattr(t, "shape", ()))

    if kind == KIND_SDXL:
        if "clip_l" not in tensors or "clip_g" not in tensors or len(widths) < 2:
            return False
        l, g = shape(tensors["clip_l"]), shape(tensors["clip_g"])
        return (
            l[-1:] == (widths[0],)
            and g[-1:] == (widths[1],)
            and 0 < _vectors(l) == _vectors(g) <= _MAX_VECTORS
        )
    if len(tensors) != 1:
        return False
    (t,) = tensors.values()
    return shape(t)[-1:] == (widths[0],) and 0 < _vectors(shape(t)) <= _MAX_VECTORS


def _make_extended_embedding(base: Any, extra: Any) -> Any:
    """``base`` (an nn.Embedding, left untouched) plus ``extra`` rows for ids >= ``base.num_embeddings``."""
    import torch

    class _ExtendedTokenEmbedding(torch.nn.Module):
        def __init__(self, base_module, extra_rows):
            super().__init__()
            self.base = base_module
            self.register_buffer("extra", extra_rows, persistent = False)
            self.base_count = int(base_module.num_embeddings)

        @property
        def weight(self):  # read by code that inspects the embedding width / dtype
            return self.base.weight

        @property
        def num_embeddings(self):
            return self.base_count + int(self.extra.shape[0])

        @property
        def embedding_dim(self):
            return self.base.embedding_dim

        def forward(self, input_ids):
            is_extra = input_ids >= self.base_count
            out = self.base(input_ids.masked_fill(is_extra, 0))
            rows = self.extra.to(device = out.device, dtype = out.dtype)
            picked = rows[(input_ids - self.base_count).clamp(min = 0, max = rows.shape[0] - 1)]
            return torch.where(is_extra.unsqueeze(-1), picked, out)

    return _ExtendedTokenEmbedding(base, extra)


@dataclass
class AppliedEmbeddings:
    """What ``apply`` installed on one pipe; ``restore`` undoes it exactly."""

    pipe: Any
    tokenizers: list[tuple[str, Any]]
    encoders: list[tuple[Any, Any]]
    tokens: dict[str, str]
    used: list[str]
    skipped: list[str]
    pattern: Optional[re.Pattern]
    marker_owners: list[Any]

    def rewrite(self, call_kwargs: dict) -> dict:
        """``call_kwargs`` with every trigger replaced by its internal token (a new dict; the input is kept)."""
        if self.pattern is None or not self.tokens:
            return call_kwargs
        out = dict(call_kwargs)

        def sub(text: str) -> str:
            return self.pattern.sub(lambda m: self.tokens.get(m.group(1).lower(), m.group(0)), text)

        for key in _PROMPT_KEYS:
            value = out.get(key)
            if isinstance(value, str):
                out[key] = sub(value)
            elif isinstance(value, list):
                out[key] = [sub(v) if isinstance(v, str) else v for v in value]
        return out

    def restore(self) -> None:
        for encoder, original in reversed(self.encoders):
            try:
                encoder.set_input_embeddings(original)
            except Exception:  # noqa: BLE001 - restoring the rest still matters
                pass
        for attr, original in reversed(self.tokenizers):
            try:
                setattr(self.pipe, attr, original)
            except Exception:  # noqa: BLE001
                pass
        for owner in self.marker_owners:
            try:
                if "_unsloth_embeddings" in owner.__dict__:
                    del owner.__dict__["_unsloth_embeddings"]
            except Exception:  # noqa: BLE001
                pass
        self.encoders.clear()
        self.tokenizers.clear()


def _slots_for(pipe: Any, kind: str) -> list[tuple[str, Any, Any, str]]:
    slots = []
    for tok_attr, enc_attr, key in _ENCODER_SLOTS:
        tokenizer = getattr(pipe, tok_attr, None)
        encoder = getattr(pipe, enc_attr, None)
        if tokenizer is None or encoder is None:
            continue
        slots.append((tok_attr, tokenizer, encoder, key))
        if kind != KIND_SDXL:
            break  # SD 1.x / 2.x: one encoder
    return slots


def apply(
    pipe: Any,
    kind: Optional[str],
    call_kwargs: dict,
    *,
    entries: Optional[list[EmbeddingEntry]] = None,
    marker_owners: Iterable[Any] = (),
    logger: Any = None,
) -> Optional[AppliedEmbeddings]:
    """Install the embeddings the prompts in ``call_kwargs`` name; None (nothing touched) when they name none.

    ``kind`` is the family's ``textual_inversion`` format. A named file of another format is skipped (as A1111
    does) and reported in ``skipped``."""
    if kind is None:
        return None
    texts = [t for key in _PROMPT_KEYS for t in _texts(call_kwargs.get(key))]
    if not any(texts):
        return None
    if entries is None:
        entries = list_embeddings()
    named = find_triggers(texts, entries)
    if not named:
        return None
    slots = _slots_for(pipe, kind)
    want = 2 if kind == KIND_SDXL else 1
    if len(slots) < want:
        raise ValueError("The loaded pipeline has no CLIP text encoder to apply embeddings to.")
    widths = [int(enc.get_input_embeddings().embedding_dim) for _, _, enc, _ in slots]
    usable: list[EmbeddingEntry] = []
    loaded: list[tuple[int, EmbeddingEntry, dict]] = []
    skipped: list[str] = []
    for e in named:
        # Fit is checked against the loaded encoders, not the listing's kind: a file for another model is skipped.
        tensors = _read_tensors(e.path)
        if _fits(tensors, kind, widths):
            loaded.append((len(usable), e, tensors))
            usable.append(e)
        else:
            skipped.append(e.id)
    applied = AppliedEmbeddings(
        pipe = pipe,
        tokenizers = [],
        encoders = [],
        tokens = {},
        used = [],
        skipped = skipped,
        pattern = None,
        marker_owners = [],
    )
    if not usable:
        if logger is not None and skipped:
            logger.info("diffusion.embeddings: skipped (other model type): %s", ", ".join(skipped))
        return applied
    try:
        import torch
        for tok_attr, tokenizer, encoder, key in slots:
            original_embedding = encoder.get_input_embeddings()
            base_count = int(original_embedding.num_embeddings)
            width = int(original_embedding.embedding_dim)
            tok_copy = copy.deepcopy(tokenizer)
            extra_rows: dict[int, Any] = {}
            for i, entry, tensors in loaded:
                vectors = _vectors_for_slot(tensors, kind, key)
                if vectors.shape[-1] != width:
                    raise ValueError(
                        f"Embedding '{entry.id}' is {vectors.shape[-1]} wide; this text encoder takes {width}."
                    )
                token = _internal_token(i, entry.id)
                names = [token] + [f"{token}_{j}" for j in range(1, vectors.shape[0])]
                tok_copy.add_tokens(names)
                for name, row in zip(names, vectors):
                    token_id = tok_copy.convert_tokens_to_ids(name)
                    if not isinstance(token_id, int) or token_id < base_count:
                        raise ValueError(f"Could not add a token for embedding '{entry.id}'.")
                    extra_rows[token_id] = row
                applied.tokens[entry.id.lower()] = token
            top = max(extra_rows) - base_count + 1
            table = torch.zeros(top, width, dtype = torch.float32)
            for token_id, row in extra_rows.items():
                table[token_id - base_count] = row
            weight = getattr(original_embedding, "weight", None)
            if weight is not None and getattr(weight, "device", None) is not None:
                if weight.device.type != "meta":
                    table = table.to(device = weight.device, dtype = weight.dtype)
            applied.encoders.append((encoder, original_embedding))
            encoder.set_input_embeddings(_make_extended_embedding(original_embedding, table))
            applied.tokenizers.append((tok_attr, tokenizer))
            setattr(pipe, tok_attr, tok_copy)
    except BaseException:
        applied.restore()
        raise
    applied.used = [e.id for e in usable]
    applied.pattern = _trigger_pattern(e.id for e in usable)
    # Prompt caches key on this: the same text means a different encoding while it is set.
    marker = tuple((e.id, _fingerprint(e.path)) for e in usable)
    for owner in (pipe, *marker_owners):
        if owner is None or owner in applied.marker_owners:
            continue
        try:
            owner.__dict__["_unsloth_embeddings"] = marker
            applied.marker_owners.append(owner)
        except Exception:  # noqa: BLE001
            pass
    if logger is not None:
        logger.info(
            "diffusion.embeddings: applied %s%s",
            ", ".join(applied.used),
            f" (skipped {', '.join(skipped)})" if skipped else "",
        )
    return applied


def embedding_state(pipe: Any) -> tuple:
    """The embeddings applied on ``pipe`` right now, () when none (prompt-cache key)."""
    try:
        return tuple(pipe.__dict__.get("_unsloth_embeddings", ()) or ())
    except Exception:  # noqa: BLE001
        return ()
