# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LoRA and LyCORIS LoKR files on Anima, Krea 2 and MOSS-Image, plus Ideogram 4's key layouts. LoKR runs as
``KronLinear`` slots sharing PEFT's adapter names: :func:`set_adapter_weights` / :func:`unload_adapters` replace
diffusers' adapter calls."""

from __future__ import annotations

import os
import re
import weakref
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

KRON_ATTR = "_unsloth_kron_adapters"
# Adapter names whose file held LoKR modules only: diffusers has no PEFT adapter of that name to set.
KRON_ONLY_ATTR = "_unsloth_kron_only"
# ``(weakref to parameter, requires_grad)`` of every LoRA-loadable component, taken before the first adapter of a set.
# PEFT freezes the base weights it injects into and unloading leaves them frozen, while torch.matmul picks its kernel
# from requires_grad even under no_grad (a non-contiguous 3-D input folds into mm, else runs bmm: other bits on CUDA).
# Weak, so a quantize after a load-time bake still frees the dense weights it replaced.
REQUIRES_GRAD_ATTR = "_unsloth_lora_requires_grad"
_PREFIXES = ("diffusion_model.", "model.diffusion_model.", "transformer.")
# Kohya / SimpleTuner keys: the reference layer path with its dots written as underscores.
_KOHYA_PREFIXES = ("lora_unet_", "lycoris_")
_LOKR_SUFFIXES = (
    "lokr_w1",
    "lokr_w2",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_t2",
)
NORMALIZED_FAMILIES = ("anima", "krea-2", "moss-image")
QWEN21_FAMILY = "qwen-image-2.1"
# ai-toolkit keeps Qwen-Image-2.1's SwiGLU input as one ``img_mlp.gate_up`` ([gate; up] rows); diffusers has
# ``gate_layer`` and ``proj``, so PEFT would drop both halves with only a log line.
_QWEN21_GATE_UP_KEY = re.compile(r"\.img_mlp\.gate_up\.(?:lora_(?:A|B|down|up)\.|alpha$)")
_QWEN21_GATE_UP = re.compile(r"^(.*\.img_mlp\.)gate_up\.(lora_[AB])\.(weight|bias)$")
IDEOGRAM4_FAMILY = "ideogram-4"
# Ideogram 4's checkpoint module names, which diffusers' converter reads under ``diffusion_model.`` only. The fused
# ``attention.qkv`` / ``attention.o`` mark the layout: diffusers' own names share ``feed_forward`` / ``adaln_modulation``.
_IDEOGRAM4_NATIVE = re.compile(
    r"(layers\.\d+\.(?:attention\.(?:qkv|o)|feed_forward\.w[123]|adaln_modulation))\.(.+)"
)
_IDEOGRAM4_FUSED = re.compile(r"layers\.\d+\.attention\.(?:qkv|o)\..+")
# musubi-tuner's kohya spelling: the same path with underscores, under ``lora_unet_``.
_IDEOGRAM4_KOHYA = re.compile(
    r"lora_unet_layers_(\d+)_(attention_qkv|attention_o|feed_forward_w[123]|adaln_modulation)\.(.+)"
)
_IDEOGRAM4_KOHYA_PATHS = {
    "attention_qkv": "attention.qkv",
    "attention_o": "attention.o",
    "feed_forward_w1": "feed_forward.w1",
    "feed_forward_w2": "feed_forward.w2",
    "feed_forward_w3": "feed_forward.w3",
    "adaln_modulation": "adaln_modulation",
}


def read_lora_file(path: str) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


# "0": a rewritten file ignores its diffusers lora_adapter_metadata (every module at scale 1, as before).
REWRITE_METADATA_ENV = "UNSLOTH_DIFFUSION_LORA_REWRITE_METADATA"
_METADATA_KEY = "lora_adapter_metadata"  # diffusers' LORA_ADAPTER_METADATA_KEY


def read_lora_metadata(path: str) -> Optional[dict]:
    """The transformer's ``LoraConfig`` kwargs diffusers reads from the header on a path load, or None."""
    import json

    from safetensors import safe_open

    try:
        with safe_open(str(path), framework = "pt") as handle:
            raw = (handle.metadata() or {}).get(_METADATA_KEY)
        metadata = json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001 - unreadable header: no metadata, as diffusers treats it
        return None
    if not isinstance(metadata, dict):
        return None
    # Every normalised family loads under the "transformer" prefix, so diffusers keeps only those keys.
    config = {
        k[len("transformer.") :]: v for k, v in metadata.items() if k.startswith("transformer.")
    }
    return config if "lora_alpha" in config else None


# safetensors' own cap on the JSON header it will parse.
_MAX_HEADER_BYTES = 100_000_000


def _is_unset(value: Any) -> bool:
    """A newer PEFT's value for a config field left off: dropping the field then changes nothing."""
    return value is None or value is False or (isinstance(value, (dict, list, str)) and not value)


def _lora_config_fields() -> Optional[frozenset]:
    import dataclasses
    try:
        from peft import LoraConfig
        return frozenset(field.name for field in dataclasses.fields(LoraConfig))
    except Exception:  # noqa: BLE001 - no PEFT (or not a dataclass): leave the header to diffusers
        return None


def _read_header(path: str) -> Optional[dict]:
    """The safetensors JSON header, or None when it is unreadable or its declared length is implausible."""
    import json
    import struct

    try:
        size = os.path.getsize(str(path))
        with open(str(path), "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                return None
            (length,) = struct.unpack("<Q", prefix)
            if length > _MAX_HEADER_BYTES or 8 + length > size:
                return None
            header = json.loads(handle.read(length))
    except Exception:  # noqa: BLE001 - unreadable header: the plain load reports it itself
        return None
    return header if isinstance(header, dict) else None


def compatible_lora_header(path: str) -> Optional[dict]:
    """Header minus unset metadata fields this PEFT's ``LoraConfig`` lacks (else diffusers' ``LoraConfig(**metadata)``
    fails on newer-PEFT files, e.g. Qwen-Image-2.1-Lightning); None when nothing to drop. Other keys stay untouched."""
    import json

    fields = _lora_config_fields()
    if fields is None:
        return None
    header = _read_header(path)
    if header is None:
        return None
    try:
        metadata = json.loads(header["__metadata__"][_METADATA_KEY])
    except Exception:  # noqa: BLE001 - no metadata: nothing for LoraConfig to reject
        return None
    if not isinstance(metadata, dict):
        return None
    dropped = [
        key
        for key, value in metadata.items()
        if "." in key and key.split(".", 1)[1] not in fields and _is_unset(value)
    ]
    if not dropped:
        return None
    logger.info(
        "diffusion.lora: %s ignores unset LoRA options this PEFT lacks: %s",
        os.path.basename(str(path)),
        ", ".join(sorted(dropped)),
    )
    header["__metadata__"][_METADATA_KEY] = json.dumps(
        {k: v for k, v in metadata.items() if k not in dropped}, indent = 2, sort_keys = True
    )
    return header


def _write_with_header(path: str, out: str, header: dict) -> None:
    """``path``'s tensors under ``header``: data offsets count from the header's end, so only the header changes."""
    import json
    import shutil
    import struct

    blob = json.dumps(header, separators = (",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    with open(str(path), "rb") as src, open(out, "wb") as dst:
        (length,) = struct.unpack("<Q", src.read(8))
        src.seek(8 + length)
        dst.write(struct.pack("<Q", len(blob)))
        dst.write(blob)
        shutil.copyfileobj(src, dst, 16 << 20)


def _load_lora_weights(pipe: Any, source: Any, name: str) -> None:
    """``pipe.load_lora_weights``, through a copy whose adapter metadata this PEFT can instantiate when needed."""
    header = compatible_lora_header(source) if isinstance(source, (str, os.PathLike)) else None
    if header is None:
        pipe.load_lora_weights(source, adapter_name = name)
        return
    import tempfile

    with tempfile.TemporaryDirectory(prefix = "unsloth_lora_") as tmp:
        copy = os.path.join(tmp, os.path.basename(str(source)))
        _write_with_header(str(source), copy, header)
        pipe.load_lora_weights(copy, adapter_name = name)


def _pattern_value(patterns: Any, stem: str, default: Any) -> Any:
    """PEFT's ``get_pattern_key`` rule: the first key that matches the end of ``stem`` on a dot boundary."""
    for key, value in (patterns or {}).items():
        if re.match(rf"(.*\.)?({key})$", stem):
            return value
    return default


def _metadata_scale(metadata: dict, stem: str, rank: int) -> float:
    """PEFT's ``scaling`` for module ``stem`` under this config."""
    import math

    alpha = float(_pattern_value(metadata.get("alpha_pattern"), stem, metadata["lora_alpha"]))
    return alpha / (math.sqrt(rank) if metadata.get("use_rslora") else rank)


def _strip(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _lokr_stem(key: str) -> Optional[str]:
    for suffix in _LOKR_SUFFIXES:
        if key.endswith("." + suffix):
            return key[: -len(suffix) - 1]
    return None


def split_lokr(state: dict) -> tuple[dict, dict]:
    """``(lokr, rest)``: LoKR tensors by module stem (each with its ``.alpha``), and every other tensor."""
    lokr: dict[str, dict] = {}
    for key, value in state.items():
        stem = _lokr_stem(key)
        if stem is not None:
            lokr.setdefault(stem, {})[key[len(stem) + 1 :]] = value
    rest = {}
    for key, value in state.items():
        if _lokr_stem(key) is not None:
            continue
        stem, _, suffix = key.rpartition(".")
        if suffix in ("alpha", "dora_scale") and stem in lokr:
            lokr[stem][suffix] = value
            continue
        rest[key] = value
    return lokr, rest


def normalize_lora(
    state: dict,
    family: Optional[str],
    metadata: Optional[dict] = None,
) -> dict:
    """Plain LoRA tensors as ``lora_A`` / ``lora_B``, ``.alpha`` (else ``metadata``'s scale) folded into ``lora_B``."""
    import torch

    renamed: dict = {}
    alphas: dict = {}
    for key, value in state.items():
        if key.endswith(".alpha"):
            alphas[key[: -len(".alpha")]] = value
            continue
        key = key.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")
        renamed[key] = value
    scales: dict = {}
    for stem, alpha in alphas.items():
        a = renamed.get(stem + ".lora_A.weight")
        if a is not None:
            scales[stem] = float(alpha) / int(a.shape[0])
    # DoRA too: its norm reads W + scaling * B @ A, so scaling folded into B at scaling 1 is the same layer.
    if metadata is not None:
        for key, a in renamed.items():
            if key.endswith(".lora_A.weight") and key[: -len(".lora_A.weight")] not in scales:
                stem = key[: -len(".lora_A.weight")]
                scales[stem] = _metadata_scale(metadata, _strip(stem), int(a.shape[0]))
    for stem, scale in scales.items():
        if renamed.get(stem + ".lora_B.weight") is None:
            continue  # an alpha with no low-rank pair (some converters write it for every module)
        if scale != 1.0:
            # PEFT scales the whole lora_B output, its bias included
            for key in (stem + ".lora_B.weight", stem + ".lora_B.bias"):
                t = renamed.get(key)
                if t is not None:
                    renamed[key] = (t.to(torch.float32) * scale).to(t.dtype)
    if family == "anima":
        renamed = _anima_embedding(renamed)
    elif family == "moss-image":
        renamed = _moss_fused_qkv(renamed)
    return renamed


def _anima_embedding(state: dict) -> dict:
    """An ``llm_adapter.embed`` pair is a Linear delta ``B @ A`` on the ``[vocab, dim]`` embedding
    table; PEFT's embedding LoRA computes ``(B' @ A').T``, so ``A' = B.T`` and ``B' = A.T``."""
    out = dict(state)
    for key in [k for k in state if ".llm_adapter.embed.lora_A.weight" in k]:
        stem = key[: -len(".lora_A.weight")]
        a, b = out.pop(stem + ".lora_A.weight"), out.pop(stem + ".lora_B.weight", None)
        if b is None:
            continue
        out[stem + ".lora_embedding_A"] = b.T.contiguous()
        out[stem + ".lora_embedding_B"] = a.T.contiguous()
    return out


_MOSS_QKV = re.compile(r"^(.*\.proj\.)qkv\.(lora_[AB])\.weight$")
# SimpleTuner trains Krea-2 with ``attn.to_q`` / ``to_k`` / ``to_v`` fused into one ``attn.to_qkv`` (rows q, k, v);
# the in-tree model keeps the three apart, with k and v narrower than q under grouped-query attention.
_KREA2_QKV = ("to_q", "to_k", "to_v")
_KREA2_FUSED_QKV = re.compile(r"^(.*\.attn\.)to_qkv\.(.+)$")
_KREA2_FUSED_STEM = re.compile(r"^(.*\.attn\.)to_qkv$")
_MAGNITUDE_SUFFIXES = ("dora_scale", "lora_magnitude_vector", "lora_magnitude_vector.weight")
# The tensors a module's delta is the product of: one that is all zero leaves the layer untouched.
_FACTOR_SUFFIXES = (
    "lora_A.weight",
    "lora_B.weight",
    "lora_down.weight",
    "lora_up.weight",
) + _LOKR_SUFFIXES


def _moss_fused_qkv(state: dict) -> dict:
    """The fused ``proj.qkv`` as the in-tree ``proj.q`` / ``k`` / ``v``: ``A`` is shared, ``B``'s rows split."""
    out: dict = {}
    for key, value in state.items():
        match = _MOSS_QKV.match(key)
        if match is None:
            out[key] = value
            continue
        stem, part = match.groups()
        if part == "lora_A":
            for name in ("q", "k", "v"):
                out[f"{stem}{name}.lora_A.weight"] = value
        else:
            rows = int(value.shape[0]) // 3
            for i, name in enumerate(("q", "k", "v")):
                out[f"{stem}{name}.lora_B.weight"] = value[i * rows : (i + 1) * rows].contiguous()
    return out


def _krea2_module(root: Any, key: str) -> Any:
    """The layer a Krea-2 path names: the reference layout (``blocks.0.attn.wq``) or the diffusers one
    (``transformer_blocks.0.attn.to_q``, what SimpleTuner and LyCORIS on the diffusers model write)."""
    from .diffusion_single_file_converters import krea2_checkpoint_to_diffusers

    try:
        name = next(iter(krea2_checkpoint_to_diffusers({key + ".weight": None})))[: -len(".weight")]
    except ValueError:
        name = key
    return root.get_submodule(name)


def _krea2_is_reference_path(key: str) -> bool:
    from .diffusion_single_file_converters import krea2_checkpoint_to_diffusers
    try:
        krea2_checkpoint_to_diffusers({key + ".weight": None})
    except ValueError:
        return False
    return True


def _krea2_qkv_widths(root: Any, attn: str) -> tuple[int, ...]:
    """Rows of ``attn``'s q, k and v projections in this model, the order they sit in a fused ``to_qkv``."""
    return tuple(
        int(_base_layer(root.get_submodule(attn + name)).weight.shape[0]) for name in _KREA2_QKV
    )


def _dora_magnitude_is_initial(root: Any, stem: str, magnitude: Any) -> bool:
    """True when a DoRA magnitude is still what DoRA starts from: the base weight's norm per output row (PEFT) or per
    input column (LyCORIS ``wd_on_out=False``). With a zero ``B @ A`` that layer is then the base layer."""
    if root is None:
        return False
    try:
        weight = _base_layer(_krea2_module(root, _strip(stem))).weight
    except (AttributeError, ValueError):
        return False
    w = weight.detach().float().flatten(1)
    m = magnitude.detach().float().flatten()
    for norm in (w.norm(dim = 1), w.norm(dim = 0)):
        if m.numel() == norm.numel() and bool(
            ((m - norm.to(m.device)).abs() <= 1e-2 * norm.abs().to(m.device) + 1e-6).all()
        ):
            return True
    return False


def _is_untrained(
    state: dict,
    stem: str,
    root: Any = None,
) -> bool:
    """True when ``stem`` leaves its layer untouched: one of its factors is all zero (so ``B @ A`` is zero), every bias
    it carries is zero, and any DoRA magnitude is still the base weight's norm (a trained one rescales the layer even
    with a zero ``B @ A``)."""
    own = {k[len(stem) + 1 :]: v for k, v in state.items() if k.startswith(stem + ".")}
    factors = [v for k, v in own.items() if k in _FACTOR_SUFFIXES]
    if not factors or all(bool(t.any()) for t in factors):
        return False
    if any(k.endswith(".bias") and bool(v.any()) for k, v in own.items()):
        return False
    return all(
        _dora_magnitude_is_initial(root, stem, v)
        for k, v in own.items()
        if k in _MAGNITUDE_SUFFIXES
    )


def krea2_drop_unused_projections(state: dict, root: Any = None) -> dict:
    """A trainer that fuses q / k / v still saves the separate ``to_q`` / ``to_k`` / ``to_v`` it never ran, untrained
    (a zero factor, zero bias, initial DoRA magnitude): drop them so the fused delta is the only one on those layers.
    A trained one beside a fused one has no single meaning, so it is refused. ``root`` (the transformer) prices a
    DoRA magnitude; without it a sibling carrying one counts as trained."""
    fused = {m.group(1) for m in map(_KREA2_FUSED_QKV.match, state) if m is not None}
    if not fused:
        return state
    drop: set = set()
    for attn in fused:
        for name in _KREA2_QKV:
            stem = attn + name
            keys = [k for k in state if k.startswith(stem + ".")]
            if not keys:
                continue
            if not _is_untrained(state, stem, root):
                raise ValueError(
                    f"{stem} is trained beside the fused {attn}to_qkv; the file holds two deltas for one layer"
                )
            drop.update(keys)
    return {k: v for k, v in state.items() if k not in drop}


def krea2_fused_qkv(root: Any, state: dict) -> dict:
    """A fused ``attn.to_qkv`` as the model's ``to_q`` / ``to_k`` / ``to_v``: ``lora_A`` (and ``alpha``) is shared,
    ``lora_B``, its bias and a DoRA magnitude split by rows. Row ``i`` of the fused delta ``B @ A`` is row ``i`` of
    ``B`` times ``A``, and a DoRA magnitude scales whole rows, so each piece is exactly that projection's delta."""
    import torch

    out: dict = {}
    for key, value in state.items():
        match = _KREA2_FUSED_QKV.match(key)
        if match is None:
            out[key] = value
            continue
        attn, suffix = match.groups()
        if suffix in ("lora_A.weight", "alpha"):
            for name in _KREA2_QKV:
                out[f"{attn}{name}.{suffix}"] = value
            continue
        if suffix == "dora_scale":
            # diffusers' Krea-2 loader filters every ``dora_scale`` key, so a split one would load as plain LoRA.
            raise ValueError(
                f"{key}: a fused q/k/v DoRA saved as dora_scale is not supported (the magnitudes would be dropped)"
            )
        if suffix not in ("lora_B.weight", "lora_B.bias") + _MAGNITUDE_SUFFIXES:
            raise ValueError(f"{key}: a fused q/k/v tensor this loader cannot split")
        widths = _krea2_qkv_widths(root, _strip(attn))
        if value.dim() == 0 or int(value.shape[0]) != sum(widths):
            raise ValueError(
                f"{key} has {int(value.shape[0]) if value.dim() else 0} rows, but q / k / v here are {widths}"
            )
        for name, piece in zip(_KREA2_QKV, torch.split(value, list(widths))):
            out[f"{attn}{name}.{suffix}"] = piece.contiguous()
    return out


def _qwen21_split_gate_up(state: dict) -> dict:
    """The fused ``img_mlp.gate_up`` as ``gate_layer`` / ``proj``: ``A`` is shared, ``B``'s rows split gate first."""
    out: dict = {}
    for key, value in state.items():
        match = _QWEN21_GATE_UP.match(key)
        if match is None:
            out[key] = value
            continue
        stem, part, kind = match.groups()
        if part == "lora_A":
            for name in ("gate_layer", "proj"):
                out[f"{stem}{name}.lora_A.{kind}"] = value
            continue
        rows = int(value.shape[0])
        if rows % 2:
            raise ValueError(
                f"{key}: a fused gate_up LoRA with an odd row count ({rows}) cannot be split into gate_layer and proj"
            )
        half = rows // 2
        out[f"{stem}gate_layer.lora_B.{kind}"] = value[:half].contiguous()
        out[f"{stem}proj.lora_B.{kind}"] = value[half:].contiguous()
    return out


def _load_qwen21(pipe: Any, path: str, name: str) -> None:
    """A file with no fused ``gate_up`` loads by path as before; one with it loads rewritten, refused if nothing lands."""
    raw = read_lora_file(path)
    if not any(_QWEN21_GATE_UP_KEY.search(k) for k in raw):
        _load_lora_weights(pipe, path, name)
        return
    metadata = None
    if (os.environ.get(REWRITE_METADATA_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    ):
        metadata = read_lora_metadata(path)
    normalized = normalize_lora(raw, QWEN21_FAMILY, metadata)
    fused = sum(1 for k in normalized if k.endswith(".img_mlp.gate_up.lora_A.weight"))
    _load_lora_weights(pipe, _qwen21_split_gate_up(normalized), name)
    if not _adapter_loaded(pipe, name):
        raise ValueError(f"{name}: none of its LoRA keys name a layer of this model")
    logger.info(
        "diffusion.lora: %s splits %d fused img_mlp.gate_up module(s) into gate_layer + proj",
        name,
        fused,
    )


def _kron_delta(x: Any, w1: Any, w2: Any, rows: Optional[tuple[int, int]]) -> Any:
    """``x @ kron(w1, w2)[rows].T`` without the dense product: row ``i * b1 + k`` and column ``j * b2 + l`` of the
    Kronecker product is ``w1[i, j] * w2[k, l]``, so two small matmuls on a reshaped input give it."""
    a1, a2 = w1.shape
    b1, b2 = w2.shape
    lead = x.shape[:-1]
    t = x.reshape(*lead, a2, b2) @ w2.T  # (..., a2, b1)
    y = (w1 @ t).reshape(*lead, a1 * b1)  # (..., a1, b1) -> rows i * b1 + k
    return y if rows is None else y[..., rows[0] : rows[1]]


def _module_class():
    from torch import nn
    class KronLinear(nn.Module):
        """A Linear plus LoKR deltas, a module rather than a hook (fused int8 paths and compiled forwards skip hooks).
        Each 0-d ``scale`` buffer is filled in place, so compiled code and CUDA graphs need no retrace."""

        def __init__(self, base_layer):
            super().__init__()
            self.base_layer = base_layer
            self.slots: list = []

        @property
        def weight(self):
            return self.base_layer.weight

        @property
        def bias(self):
            return self.base_layer.bias

        def get_base_layer(self):
            base = self.base_layer
            get = getattr(base, "get_base_layer", None)
            return get() if callable(get) else base

        def forward(self, x, *args, **kwargs):
            out = self.base_layer(x, *args, **kwargs)
            for prefix, rows in self.slots:
                w1 = getattr(self, prefix + "_w1").to(x.dtype)
                w2 = getattr(self, prefix + "_w2").to(x.dtype)
                out = out + _kron_delta(x, w1, w2, rows) * getattr(self, prefix + "_scale").to(
                    out.dtype
                )
            return out

    return KronLinear


_KRON_CLASS = None


def kron_linear_class():
    global _KRON_CLASS
    if _KRON_CLASS is None:
        _KRON_CLASS = _module_class()
    return _KRON_CLASS


def _parent_of(root: Any, module: Any) -> tuple[Any, str]:
    for name, child in root.named_modules():
        for attr, sub in child.named_children():
            if sub is module:
                return child, attr
    raise ValueError("the LoKR target is not a submodule of its component")


class KronAdapter:
    def __init__(
        self,
        root: Any,
        module: Any,
        w1: Any,
        w2: Any,
        alpha_scale: float,
        rows: Optional[tuple[int, int]] = None,
    ):
        import torch

        cls = kron_linear_class()
        base = _base_layer(module)
        device = _device_of(base)
        dtype = _dtype_of(base)
        self.alpha_scale = float(alpha_scale)
        # Allocated before the layer is swapped, so a failure (OOM) leaves the model as it was.
        buffers = (
            w1.to(device = device, dtype = dtype),
            w2.to(device = device, dtype = dtype),
            torch.tensor(self.alpha_scale, dtype = torch.float32, device = device),
        )
        if not isinstance(module, cls):
            parent, attr = _parent_of(root, module)
            wrapper = cls(module)
            setattr(parent, attr, wrapper)
            module = wrapper
        self.wrapper = module
        self.root = root
        self.prefix = f"kron{len(module.slots)}"
        for suffix, tensor in zip(("_w1", "_w2", "_scale"), buffers):
            module.register_buffer(self.prefix + suffix, tensor, persistent = False)
        module.slots.append((self.prefix, rows))

    def set_weight(self, weight: float) -> None:
        import torch
        with torch.no_grad():
            getattr(self.wrapper, self.prefix + "_scale").fill_(float(weight) * self.alpha_scale)

    def remove(self) -> None:
        wrapper = self.wrapper
        wrapper.slots = [slot for slot in wrapper.slots if slot[0] != self.prefix]
        for suffix in ("_w1", "_w2", "_scale"):
            wrapper._buffers.pop(self.prefix + suffix, None)
        if not wrapper.slots:
            parent, attr = _parent_of(self.root, wrapper)
            setattr(parent, attr, wrapper.base_layer)


# A Kronecker factor is never larger than the layer it builds (16384 x 16384), so a low-rank pair whose product is
# bigger is malformed: refuse it before multiplying, as a ~400 KB pair can otherwise ask for tens of GB.
_MAX_KRON_FACTOR_ELEMENTS = 1 << 28


def _low_rank_product(a: Any, b: Any, stem: str) -> Any:
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"LoKR factors {stem}_a / {stem}_b do not multiply")
    if int(a.shape[0]) * int(b.shape[1]) > _MAX_KRON_FACTOR_ELEMENTS:
        raise ValueError(
            f"LoKR factor {stem} is {int(a.shape[0])} x {int(b.shape[1])}, larger than any layer"
        )
    return a.float() @ b.float()


def _kron_factors(parts: dict) -> tuple[Any, Any, float]:
    """``(w1, w2, alpha_scale)`` like the reference runtime: a factor stored as a low-rank pair is its product, and
    ``alpha / rank`` applies only when one is (rank = that pair's inner size)."""
    import torch

    dim = None
    w1 = parts.get("lokr_w1")
    if w1 is None:
        w1 = _low_rank_product(parts["lokr_w1_a"], parts["lokr_w1_b"], "lokr_w1")
        dim = int(parts["lokr_w1_b"].shape[0])
    w2 = parts.get("lokr_w2")
    if w2 is None:
        if "lokr_t2" in parts:
            raise ValueError("a LoKR convolution (lokr_t2) cannot apply to a Linear layer")
        w2 = _low_rank_product(parts["lokr_w2_a"], parts["lokr_w2_b"], "lokr_w2")
        dim = int(parts["lokr_w2_b"].shape[0])
    if w1.dim() != 2 or w2.dim() != 2:
        raise ValueError("only Linear LoKR modules are supported")
    if "dora_scale" in parts:
        raise ValueError("LoKR modules trained with DoRA (dora_scale) are not supported")
    alpha = parts.get("alpha")
    scale = float(alpha) / dim if alpha is not None and dim else 1.0
    return w1.to(torch.float32), w2.to(torch.float32), scale


def _device_of(module: Any) -> Any:
    weight = getattr(module, "weight", None)
    return weight.device if weight is not None and weight.device.type != "meta" else None


def _dtype_of(module: Any) -> Any:
    """The compute dtype of a layer: a quantized weight reports its storage dtype, so fall back to its bias or bf16."""
    import torch

    for name in ("weight", "bias"):
        tensor = getattr(module, name, None)
        if tensor is not None and tensor.is_floating_point() and tensor.element_size() >= 2:
            return tensor.dtype
    return torch.bfloat16


def _base_layer(module: Any) -> Any:
    get_base = getattr(module, "get_base_layer", None)
    return get_base() if callable(get_base) else module


def _lokr_targets(
    pipe: Any, stem: str, family: Optional[str]
) -> list[tuple[Any, Any, Optional[tuple[int, int]]]]:
    key = _strip(stem)
    if family == "krea-2":
        root = pipe.transformer
        match = _KREA2_FUSED_STEM.match(key)
        if match is None:
            return [(root, _krea2_module(root, key), None)]
        modules = [root.get_submodule(match.group(1) + n) for n in _KREA2_QKV]
        widths = _krea2_qkv_widths(root, match.group(1))
        starts = [sum(widths[:i]) for i in range(len(widths))]
        return [(root, m, (s, s + w)) for m, s, w in zip(modules, starts, widths)]
    if family == "moss-image":
        root = pipe.transformer
        match = re.match(r"^(.*\.proj\.)qkv$", key)
        if match is None:
            return [(root, root.get_submodule(key), None)]
        modules = [root.get_submodule(match.group(1) + n) for n in ("q", "k", "v")]
        width = int(_base_layer(modules[0]).weight.shape[0])
        return [(root, m, (i * width, (i + 1) * width)) for i, m in enumerate(modules)]
    if family == "anima":
        from diffusers.loaders.lora_conversion_utils import (
            _convert_non_diffusers_anima_lora_to_diffusers,
        )

        name = next(
            iter(
                _convert_non_diffusers_anima_lora_to_diffusers(
                    {"diffusion_model." + key + ".x": None}
                )
            )
        )
        component, _, path = name[: -len(".x")].partition(".")
        root = getattr(pipe, component)
        return [(root, root.get_submodule(path), None)]
    raise ValueError(f"LoKR adapters are not supported for {family or 'this model'}")


def _attach_lokr(pipe: Any, name: str, lokr: dict, family: Optional[str]) -> int:
    from torch import nn

    adapters: list[KronAdapter] = []
    try:
        for stem, parts in lokr.items():
            w1, w2, scale = _kron_factors(parts)
            targets = _lokr_targets(pipe, stem, family)
            spans = [rows for _, _, rows in targets if rows is not None]
            if spans and int(w1.shape[0] * w2.shape[0]) != max(end for _, end in spans):
                # The per-target check below only sees each slice: a fused adapter of another height would pass it.
                raise ValueError(
                    f"LoKR module {_strip(stem)} has {int(w1.shape[0] * w2.shape[0])} rows, but the fused "
                    f"projection it splits into here has {max(end for _, end in spans)}"
                )
            for root, module, rows in targets:
                base = _base_layer(module)
                if isinstance(base, nn.Embedding):  # 2-D like a Linear, but its input is token ids
                    raise ValueError(
                        f"LoKR module {_strip(stem)} targets an embedding table; only Linear LoKR modules are supported"
                    )
                out_f, in_f = (
                    (int(d) for d in base.weight.shape) if base.weight.dim() == 2 else (0, 0)
                )
                full = int(w1.shape[0] * w2.shape[0])
                width = rows[1] - rows[0] if rows else full
                if in_f != int(w1.shape[1] * w2.shape[1]) or out_f != width:
                    raise ValueError(
                        f"LoKR module {_strip(stem)} is {full} x {int(w1.shape[1] * w2.shape[1])}, "
                        f"but the layer it targets is {out_f} x {in_f}"
                    )
                adapters.append(KronAdapter(root, module, w1, w2, scale, rows))
    except Exception:
        for adapter in adapters:
            adapter.remove()
        raise
    registry = pipe.__dict__.setdefault(KRON_ATTR, {})
    registry[name] = adapters
    return len(adapters)


def _anima_unreachable(pipe: Any, paths: list, names: dict) -> set:
    """Splits whose Anima rename names no registered submodule: ``_lokr_targets`` would raise for each.
    One batched rename; an output collision returns nothing ruled out (every split is then tried)."""
    import torch

    try:
        from diffusers.loaders.lora_conversion_utils import (
            _convert_non_diffusers_anima_lora_to_diffusers,
        )
    except ImportError:
        return set()

    keys = ["diffusion_model." + _strip(path) + ".x" for path in paths]
    renamed = list(_convert_non_diffusers_anima_lora_to_diffusers(dict.fromkeys(keys)))
    if len(renamed) != len(keys):
        return set()
    out = set()
    for path, name in zip(paths, renamed):
        component, _, sub = name[: -len(".x")].partition(".")
        if component not in names:
            root = getattr(pipe, component, None)
            names[component] = (
                {n for n, _ in root.named_modules(remove_duplicate = False)}
                if isinstance(root, torch.nn.Module)
                else None
            )
        known = names[component]
        if known is not None and sub not in known:
            out.add(path)
    return out


def _resolve_kohya_stem(
    pipe: Any,
    stem: str,
    family: Optional[str],
    names: Optional[dict] = None,
) -> str:
    """The one dotted path, of every way to re-split ``stem``'s underscores, that names a layer of this model."""
    tokens = stem.split("_")
    if len(tokens) > 16:
        raise ValueError(f"LoRA key {stem} names no layer of this model")
    paths = [
        tokens[0]
        + "".join(("." if mask >> i & 1 else "_") + token for i, token in enumerate(tokens[1:]))
        for mask in range(1 << (len(tokens) - 1))
    ]
    skip = (
        _anima_unreachable(pipe, paths, {} if names is None else names) if family == "anima" else ()
    )
    found = []
    for path in paths:
        if path in skip:
            continue
        try:
            _lokr_targets(pipe, path, family)
        except Exception:  # noqa: BLE001 -- this split names no layer
            continue
        found.append(path)
    if len(found) != 1:
        raise ValueError(f"LoRA key {stem} names no single layer of this model")
    return found[0]


def _dotted_kohya_keys(pipe: Any, state: dict, family: Optional[str]) -> dict:
    out: dict = {}
    paths: dict = {}
    names: dict = {}
    for key, value in state.items():
        prefix = next((p for p in _KOHYA_PREFIXES if key.startswith(p)), None)
        if prefix is None:
            out[key] = value
            continue
        stem, _, suffix = key[len(prefix) :].partition(".")
        if stem not in paths:
            paths[stem] = _resolve_kohya_stem(pipe, stem, family, names)
        path = paths[stem]
        # diffusers converts only reference-layout names under ``diffusion_model.``; its own layout goes as is.
        lead = (
            "transformer."
            if family == "krea-2" and not _krea2_is_reference_path(path)
            else "diffusion_model."
        )
        out[f"{lead}{path}.{suffix}"] = value
    return out


def _adapter_loaded(pipe: Any, name: str) -> bool:
    return any(name in names for names in (pipe.get_list_adapters() or {}).values())


def _ideogram4_renames(keys: list) -> dict:
    """``key -> diffusion_model. key`` for an Ideogram 4 file naming the checkpoint modules another way: musubi-tuner's
    ``lora_unet_layers_0_attention_qkv.lora_down.weight``, or a ``transformer.`` prefix on the fused names (its
    convert_lora.py with ``--diffusers_prefix transformer``). Empty when nothing needs renaming."""
    fused = any(
        k.startswith("transformer.") and _IDEOGRAM4_FUSED.fullmatch(k[len("transformer.") :])
        for k in keys
    )
    out: dict = {}
    for key in keys:
        kohya = _IDEOGRAM4_KOHYA.fullmatch(key)
        if kohya is not None:
            index, module, suffix = kohya.groups()
            out[key] = f"diffusion_model.layers.{index}.{_IDEOGRAM4_KOHYA_PATHS[module]}.{suffix}"
        elif fused and key.startswith("transformer."):
            native = _IDEOGRAM4_NATIVE.fullmatch(key[len("transformer.") :])
            if native is not None:
                out[key] = "diffusion_model." + native.group(0)
    return out


def ideogram4_native_lora(state: dict, metadata: Optional[dict] = None) -> Optional[dict]:
    """The ``diffusion_model.`` layout diffusers' Ideogram 4 converter splits into ``to_q`` / ``to_k`` / ``to_v``, with
    ``.alpha`` (else ``metadata``'s scale) folded into ``lora_B``; None when no key needs renaming."""
    renames = _ideogram4_renames(list(state))
    if not renames:
        return None
    return normalize_lora(
        {renames.get(k, k): v for k, v in state.items()}, IDEOGRAM4_FAMILY, metadata
    )


def _lora_keys(path: str) -> list:
    from safetensors import safe_open
    with safe_open(str(path), framework = "pt") as handle:
        return list(handle.keys())


def _load_ideogram4(pipe: Any, path: str, name: str) -> None:
    """diffusers converts only the ``diffusion_model.`` spelling of the checkpoint names: it raises on the
    ``transformer.`` one and loads none of the kohya one, so both are renamed first."""
    source: Any = path
    if _ideogram4_renames(_lora_keys(path)):
        metadata = None
        if (os.environ.get(REWRITE_METADATA_ENV) or "").strip().lower() not in (
            "0",
            "off",
            "false",
            "no",
        ):
            metadata = read_lora_metadata(path)
        source = ideogram4_native_lora(read_lora_file(path), metadata)
    _load_lora_weights(pipe, source, name)
    # diffusers drops keys outside the components it loads into without an error.
    if not _adapter_loaded(pipe, name):
        raise ValueError(f"{name}: none of its LoRA keys name a layer of this model")


def _lora_components(pipe: Any) -> list:
    import torch

    names = getattr(pipe, "_lora_loadable_modules", None) or ("transformer", "unet")
    roots = [getattr(pipe, name, None) for name in names]
    return list({id(m): m for m in roots if isinstance(m, torch.nn.Module)}.values())


def _remember_requires_grad(pipe: Any) -> None:
    if REQUIRES_GRAD_ATTR in pipe.__dict__:
        return  # an adapter of this set is already loaded: the flags were taken before the first one
    pipe.__dict__[REQUIRES_GRAD_ATTR] = [
        (weakref.ref(param), param.requires_grad)
        for root in _lora_components(pipe)
        for param in root.parameters()
    ]


def _restore_requires_grad(pipe: Any) -> None:
    for ref, flag in pipe.__dict__.pop(REQUIRES_GRAD_ATTR, ()):
        param = ref()
        if param is not None and param.requires_grad != flag and param.is_leaf:
            param.requires_grad_(flag)


def load_adapter(pipe: Any, path: str, name: str, family: Optional[str]) -> None:
    """One file under ``name``: plain LoRA via ``pipe.load_lora_weights``, LoKR modules as ``KronLinear`` slots."""
    from .diffusion_lora_addmm import install as _install_lora_addmm

    _install_lora_addmm()  # unfolded adapters add their delta in the B GEMM's epilogue
    _remember_requires_grad(pipe)
    if family == IDEOGRAM4_FAMILY:
        _load_ideogram4(pipe, path, name)
        return
    if family == QWEN21_FAMILY:
        _load_qwen21(pipe, path, name)
        return
    if family not in NORMALIZED_FAMILIES:
        _load_lora_weights(pipe, path, name)
        return
    raw = read_lora_file(path)
    dotted = _dotted_kohya_keys(pipe, raw, family)
    if family == "krea-2":
        dotted = krea2_drop_unused_projections(dotted, pipe.transformer)
    lokr, rest = split_lokr(dotted)

    def normalized(metadata: Optional[dict] = None) -> dict:
        out = normalize_lora(rest, family, metadata)
        return krea2_fused_qkv(pipe.transformer, out) if family == "krea-2" else out

    plain = normalized()
    if not any(".lora_" in k for k in plain):
        plain = {}
    if plain:
        # A file needing no rewrite loads by path: diffusers reads lora_alpha / rank from its metadata, not a dict.
        unchanged = not lokr and plain.keys() == raw.keys() and all(plain[k] is raw[k] for k in raw)
        if not unchanged and (os.environ.get(REWRITE_METADATA_ENV) or "").strip().lower() not in (
            "0",
            "off",
            "false",
            "no",
        ):
            metadata = read_lora_metadata(path)
            if metadata is not None:
                plain = normalized(metadata)
        # PEFT injects only into bare Linears: unwrap LoKR while it runs, then rewrap whatever sits there now.
        wrappers = _unwrap_kron(pipe)
        try:
            _load_lora_weights(pipe, path if unchanged else plain, name)
        finally:
            _rewrap_kron(wrappers)
            if wrappers:
                _refresh_group_offload(_kron_roots(pipe))
        # diffusers drops keys outside the components it loads into without an error.
        if not _adapter_loaded(pipe, name):
            raise ValueError(f"{name}: none of its LoRA keys name a layer of this model")
    if lokr:
        attached = _attach_lokr(pipe, name, lokr, family)
        _refresh_group_offload(_kron_roots(pipe))
        if not plain:
            pipe.__dict__.setdefault(KRON_ONLY_ATTR, set()).add(name)
        logger.info("diffusion.lora: %s applies %d LoKR module(s)", name, attached)
    if not plain and not lokr:
        raise ValueError(f"{name} holds no LoRA or LoKR weights this loader recognises")


def _kron_roots(pipe: Any) -> list:
    roots = {
        id(adapter.root): adapter.root
        for adapters in pipe.__dict__.get(KRON_ATTR, {}).values()
        for adapter in adapters
    }
    return list(roots.values())


def _refresh_group_offload(roots: list) -> None:
    """Re-apply group offloading on ``roots`` (as diffusers does after PEFT injection): groups list their tensors when
    applied, so LoKR buffers added or dropped since would be missed."""
    if not roots:
        return
    from diffusers.hooks.group_offloading import _maybe_remove_and_reapply_group_offloading

    for root in roots:
        _maybe_remove_and_reapply_group_offloading(root)


def _unwrap_kron(pipe: Any) -> list:
    out = []
    seen: set = set()
    for adapters in pipe.__dict__.get(KRON_ATTR, {}).values():
        for adapter in adapters:
            wrapper = adapter.wrapper
            if id(wrapper) in seen:
                continue
            seen.add(id(wrapper))
            parent, attr = _parent_of(adapter.root, wrapper)
            setattr(parent, attr, wrapper.base_layer)
            out.append((parent, attr, wrapper))
    return out


def _rewrap_kron(wrappers: list) -> None:
    for parent, attr, wrapper in wrappers:
        wrapper.base_layer = getattr(parent, attr)
        setattr(parent, attr, wrapper)


def kron_names(pipe: Any) -> tuple[str, ...]:
    return tuple(pipe.__dict__.get(KRON_ATTR, {}))


def set_adapter_weights(pipe: Any, names: list, weights: list) -> None:
    """``pipe.set_adapters`` for both kinds; a LoKR-only name has no PEFT adapter to set."""
    set_kron_weights(pipe, dict(zip(names, weights)))
    kron_only = pipe.__dict__.get(KRON_ONLY_ATTR, set())
    pairs = [(n, w) for n, w in zip(names, weights) if n not in kron_only]
    if pairs or not kron_only:
        pipe.set_adapters([n for n, _w in pairs], adapter_weights = [w for _n, w in pairs])


def kron_only_selection(pipe: Any) -> bool:
    names = [entry[0] for entry in getattr(pipe, "_unsloth_loras", ()) or ()]
    kron_only = pipe.__dict__.get(KRON_ONLY_ATTR, set())
    return bool(names) and all(name in kron_only for name in names)


def set_kron_weights(pipe: Any, weights: dict) -> None:
    """Every LoKR adapter's weight (name -> weight; an omitted name is 0, as a pinned PEFT scale is)."""
    for adapter_name, adapters in pipe.__dict__.get(KRON_ATTR, {}).items():
        for adapter in adapters:
            adapter.set_weight(float(weights.get(adapter_name, 0.0)))


def unload_adapters(pipe: Any) -> None:
    roots = _kron_roots(pipe)
    for adapters in pipe.__dict__.pop(KRON_ATTR, {}).values():
        for adapter in adapters:
            adapter.remove()
    pipe.__dict__.pop(KRON_ONLY_ATTR, None)
    try:
        pipe.unload_lora_weights()
    finally:
        # Without this an attach + detach does not reproduce the render before it (HiDream-I1's to_out).
        _restore_requires_grad(pipe)
    _refresh_group_offload(roots)
