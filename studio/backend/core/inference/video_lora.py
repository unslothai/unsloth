# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load-time LoRA bake for the video backend: adapters merge into the dense bf16 denoiser before any
quantisation, compile or offload, so GGUF / single-file / hosted-quantized loads are refused."""

from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

# Shared with core/inference/diffusion.py so both backends describe a baked set the same way.
LORAS_ATTR = "_unsloth_loras"
LORAS_BAKED_ATTR = "_unsloth_loras_baked"
BAKED_SPECS_ATTR = "_unsloth_baked_lora_specs"

# Key prefixes a LoRA file may carry before the denoiser's module path (diffusers save_lora_weights, PEFT, kohya).
_PREFIXES = ("transformer.", "base_model.model.", "diffusion_model.", "model.diffusion_model.")
_A_SUFFIXES = (".lora_A.weight", ".lora_A.default.weight", ".lora_down.weight")
_B_SUFFIXES = (".lora_B.weight", ".lora_B.default.weight", ".lora_up.weight")
_DENOISER_COMPONENTS = frozenset({"transformer", "transformer_2", "transformer_ref"})
# Wan's original module names (the lightx2v / kohya adapters) -> diffusers WanTransformer3DModel's.
_WAN_NATIVE_MODULE = re.compile(
    r"^(blocks\.\d+\.)(self_attn|cross_attn|ffn)\.(q|k|v|o|k_img|v_img|0|2)$"
)
# musubi-tuner's default Wan format: lora_unet_blocks_N_self_attn_q.
_WAN_MUSUBI_MODULE = re.compile(
    r"^lora_unet_blocks_(\d+)_(self_attn|cross_attn|ffn)_(q|k|v|o|k_img|v_img|0|2)$"
)
_WAN_NATIVE_LEAF = {
    ("self_attn", "q"): "attn1.to_q",
    ("self_attn", "k"): "attn1.to_k",
    ("self_attn", "v"): "attn1.to_v",
    ("self_attn", "o"): "attn1.to_out.0",
    ("cross_attn", "q"): "attn2.to_q",
    ("cross_attn", "k"): "attn2.to_k",
    ("cross_attn", "v"): "attn2.to_v",
    ("cross_attn", "o"): "attn2.to_out.0",
    ("cross_attn", "k_img"): "attn2.add_k_proj",
    ("cross_attn", "v_img"): "attn2.add_v_proj",
    ("ffn", "0"): "ffn.net.0.proj",
    ("ffn", "2"): "ffn.net.2",
}


# MiniMax-H3 LoRAs in the community layout name the original modules: fused attn.qkv_proj, gated mlp.fc1, out_proj,
# fc2 and adaln_proj.linear. They map onto the diffusers denoiser by the row map its weight loader already uses.
_H3_FAMILY = "minimax-h3"
_H3_COMMUNITY_MODULE = re.compile(
    r"^(?:blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2)|adaln_proj\.linear)"
    r"|token_refiner\.blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2)))$"
)


_H3_COMMUNITY_RENAMES = (
    (".attn.out_proj", ".attn.to_out.0"),
    (".mlp.fc2", ".ff.net.2"),
)


def h3_community_key_map(module: str, rows: int) -> list:
    """``[(diffusers module, B row slices or None)]`` for one community-layout H3 module with ``rows`` output rows:
    fused qkv splits by thirds into to_q / to_k / to_v, gated fc1 [gate; value] becomes ff.net.0.proj [value; gate]."""
    if module.startswith("token_refiner.blocks."):
        name = "token_refiner.refiner_blocks." + module[len("token_refiner.blocks.") :]
    else:
        name = "transformer_blocks." + module[len("blocks.") :]
    for old, new in _H3_COMMUNITY_RENAMES:
        if name.endswith(old):
            name = name[: -len(old)] + new
    if name.endswith(".attn.qkv_proj"):
        if rows % 3:
            raise ValueError(f"{module}: fused qkv with {rows} rows is not three equal projections")
        third = rows // 3
        stem = name[: -len("qkv_proj")]
        return [
            (f"{stem}{part}", [(i * third, third)])
            for i, part in enumerate(("to_q", "to_k", "to_v"))
        ]
    if name.endswith(".mlp.fc1"):
        if rows % 2:
            raise ValueError(f"{module}: gated fc1 with an odd row count {rows}")
        half = rows // 2
        return [(name[: -len(".mlp.fc1")] + ".ff.net.0.proj", [(half, half), (0, half)])]
    return [(name, None)]


def _h3_community_targets(module: str, b: Any) -> Optional[list]:
    """``[(diffusers module, B)]`` for a community-layout H3 module, else None; A is shared by every split."""
    if not _H3_COMMUNITY_MODULE.match(module):
        return None
    import torch

    return [
        (key, b if rows is None else torch.cat([b[s : s + n] for (s, n) in rows], dim = 0))
        for key, rows in h3_community_key_map(module, int(b.shape[0]))
    ]


def _diffusers_module_name(module: str) -> str:
    """Rename a Wan original (dotted or musubi-flattened) block projection to diffusers; else unchanged."""
    hit = _WAN_NATIVE_MODULE.match(module)
    if hit is None:
        musubi = _WAN_MUSUBI_MODULE.match(module)
        if musubi is not None:
            block, group, proj = musubi.groups()
            leaf = _WAN_NATIVE_LEAF.get((group, proj))
            return f"blocks.{block}.{leaf}" if leaf else module
    leaf = _WAN_NATIVE_LEAF.get((hit.group(2), hit.group(3))) if hit else None
    return hit.group(1) + leaf if leaf else module


def normalize_specs(loras: Any) -> list[tuple[str, float]]:
    """``[(id, weight), ...]`` from LoraSpec models, dicts or pairs; anything unrecognised is dropped."""
    out: list[tuple[str, float]] = []
    for entry in loras or ():
        if isinstance(entry, dict):
            lora_id, weight = entry.get("id"), entry.get("weight", 1.0)
        elif hasattr(entry, "id"):
            lora_id, weight = getattr(entry, "id"), getattr(entry, "weight", 1.0)
        else:
            try:
                lora_id, weight = entry
            except (TypeError, ValueError):
                continue
        try:
            weight = float(1.0 if weight is None else weight)
        except (TypeError, ValueError):
            continue
        if lora_id:
            out.append((str(lora_id), weight))
    return out


def active_specs(loras: Any) -> list[tuple[str, float]]:
    """The specs that change the weights (weight != 0); empty means a plain load."""
    return [(i, w) for (i, w) in normalize_specs(loras) if w != 0]


def has_active(loras: Any) -> bool:
    return bool(active_specs(loras))


def is_denoiser_component(name: str) -> bool:
    return name in _DENOISER_COMPONENTS


def _precision_unset(value: Optional[str]) -> bool:
    return value is None or str(value).strip().lower() in ("", "auto")


def _precision_dense(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() in ("none", "off")


def load_refusal(
    fam: Any,
    kind: Optional[str],
    transformer_quant: Optional[str],
    loras: Any = None,
) -> Optional[str]:
    """Why this video load cannot bake LoRAs, or None when it can. Network-free. With ``loras``, a two-expert
    family also refuses any pick that is not a catalog high / low-noise pair."""
    if fam is None:
        return None
    label = getattr(fam, "label", None) or getattr(fam, "name", None) or "this model"
    if kind != "pipeline":
        what = "GGUF" if kind == "gguf" else "single-file"
        return (
            f"LoRAs are baked into the bfloat16 denoiser of a full diffusers pipeline, and a {what} {label} "
            "checkpoint has none. Load the base repo (no checkpoint file) with the LoRA selected."
        )
    if getattr(fam, "is_moe", False):
        single = next((i for (i, _w) in active_specs(loras) if not _is_pair(i)), None)
        if single is not None:
            return _pair_mismatch(single, True)
    precision = str(transformer_quant or "").strip().lower()
    if (
        getattr(fam, "modular_workflow", None)
        and side_branch_enabled(fam)
        and not (_precision_unset(transformer_quant) or _precision_dense(transformer_quant))
        and precision != "int8"
    ):
        # The side branch wraps nn.Linear layers only; the hosted fp8 denoiser becomes stored-fp8 native layers on
        # cards without an fp8 GEMM (Ampere), which it cannot adapt.
        return (
            f"{label} LoRAs run beside the released bfloat16 or the hosted int8 denoiser, not the hosted "
            f"'{transformer_quant}' one. Load with transformer_quant unset, 'int8' or 'none' to use the LoRA."
        )
    if (
        getattr(fam, "modular_workflow", None)
        and not side_branch_enabled(fam)
        and not (_precision_unset(transformer_quant) or _precision_dense(transformer_quant))
    ):
        return (
            f"{label} LoRAs merge into the released bfloat16 denoiser, so they cannot ride on the hosted "
            f"'{transformer_quant}' denoiser. Leave transformer_quant unset (or 'none') to load with the LoRA."
        )
    return None


def dense_transformer_quant(
    fam: Any, transformer_quant: Optional[str], loras: Any
) -> Optional[str]:
    """MiniMax-H3: an unset precision takes the hosted quantized denoiser. A baked LoRA pins the released one; the
    side branch leaves the codes untouched, so it keeps the hosted pick like a plain load."""
    if (
        has_active(loras)
        and not side_branch_enabled(fam)
        and getattr(fam, "modular_workflow", None)
        and _precision_unset(transformer_quant)
    ):
        return "none"
    return transformer_quant


# Layers the hosted pruned-modulation (curve form) MiniMax-H3 denoiser rebuilds with other shapes: every block's
# adaln_proj.linear and norm_out.linear take the 8-wide curve basis instead of the 2688-wide time embedding, and the
# time embedder / time_proj are replaced (video_minimax_h3_adaln.apply_h3_adaln_curve). Community names included.
_H3_CURVE_REPLACED = re.compile(
    r"(?:^|\.)(?:adaln_proj|norm_out|time_embedder|time_proj|time_embed)(?:\.|$)"
)


def h3_curve_replaced_target(files: Iterable[str]) -> Optional[str]:
    """The first module an H3 LoRA in ``files`` adapts that the hosted curve-form denoiser replaces, else None.
    Reads tensor names from the safetensors header only; an unreadable file is left for the load to report."""
    known = _A_SUFFIXES + _B_SUFFIXES
    for path in filter(None, files):
        try:
            from safetensors import safe_open
            with safe_open(str(path), framework = "pt", device = "cpu") as f:
                keys = list(f.keys())
        except Exception:  # noqa: BLE001
            continue
        for key in keys:
            hit = next((s for s in known if key.endswith(s)), None)
            if hit is None:
                continue
            name = key[: -len(hit)]
            for prefix in _PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
            if _H3_CURVE_REPLACED.search(name):
                return name
    return None


def side_branch_transformer_quant(
    fam: Any, transformer_quant: Optional[str], resolved: Iterable[tuple]
) -> Optional[str]:
    """The side branch keeps an unset precision on the hosted int8 denoiser, whose pruned modulation and time embedder
    do not have the released shapes. A LoRA adapting those layers pins the released bf16 denoiser when the precision
    is unset, and is refused under an explicit 'int8'; attention / MLP LoRAs keep the hosted pick."""
    if not (side_branch_enabled(fam) and getattr(fam, "modular_workflow", None)):
        return transformer_quant
    if not (
        _precision_unset(transformer_quant) or str(transformer_quant).strip().lower() == "int8"
    ):
        return transformer_quant
    hit = h3_curve_replaced_target(
        p for e in resolved for p in (e[1], e[3] if len(e) > 3 else None)
    )
    if hit is None:
        return transformer_quant
    if _precision_unset(transformer_quant):
        return "none"
    label = getattr(fam, "label", None) or getattr(fam, "name", None) or "this model"
    raise ValueError(
        f"This LoRA adapts {hit}, which the hosted int8 {label} denoiser replaces with a pruned modulation of "
        "another shape. Load with transformer_quant unset or 'none' to use the LoRA on the released denoiser."
    )


def planned_side_branch_quant(
    fam: Any,
    transformer_quant: Optional[str],
    loras: Any,
    *,
    hf_token: Optional[str],
    local_files_only: bool = True,
    cancel_event: Optional[threading.Event] = None,
) -> Optional[str]:
    """``side_branch_transformer_quant`` for a download plan, made before the denoiser shards are planned. The
    adapters are resolved (cache only unless ``local_files_only`` is False); one that cannot be resolved yet has
    unknown targets, so an unset precision plans the released denoiser, which every target can ride on."""
    if not (side_branch_enabled(fam) and getattr(fam, "modular_workflow", None)):
        return transformer_quant
    if not (
        _precision_unset(transformer_quant) or str(transformer_quant).strip().lower() == "int8"
    ):
        return transformer_quant
    try:
        resolved = resolve(
            active_specs(loras),
            family = getattr(fam, "name", None),
            hf_token = hf_token,
            cancel_event = cancel_event,
            local_files_only = local_files_only,
            expert_pair = bool(getattr(fam, "is_moe", False)),
        )
    except Exception:  # noqa: BLE001 - not cached (or unresolvable): the load resolves and reports it
        return "none" if _precision_unset(transformer_quant) else transformer_quant
    return side_branch_transformer_quant(fam, transformer_quant, resolved)


def check_families(specs: Iterable[tuple[str, float]], family: Optional[str]) -> None:
    """Network-free family gate for catalog ids, so a mismatched pick is a 400 before anything is evicted.
    Ids the catalog does not know (public HF repo ids) are resolved, and gated, by the load itself."""
    from core.inference import diffusion_lora

    req = (family or "").strip().lower()
    specs = [(i, w) for (i, w) in specs if w != 0]
    if not specs:
        return
    catalog = diffusion_lora._catalog_by_id()
    for lora_id, _weight in specs:
        entry = catalog.get(lora_id)
        if entry is None:
            if "/" not in lora_id:
                raise diffusion_lora.LoraSelectionError(
                    f"unknown LoRA '{lora_id}': not a local adapter, catalog entry, or HF repo id"
                )
            continue
        fams = {f.lower() for f in entry.families}
        if fams and req and req not in fams:
            raise diffusion_lora.LoraSelectionError(
                f"LoRA '{lora_id}' is for {', '.join(entry.families)}, not the loaded "
                f"'{family}' model family; pick a LoRA built for this family."
            )
        if entry.fmt != "safetensors":
            raise diffusion_lora.LoraSelectionError(
                f"LoRA '{lora_id}' is a GGUF adapter; video loads take .safetensors adapters only."
            )


def resolve(
    specs: Iterable[tuple[str, float]],
    *,
    family: Optional[str],
    hf_token: Optional[str],
    cancel_event: Optional[threading.Event] = None,
    local_files_only: bool = False,
    expert_pair: bool = False,
) -> tuple[tuple[str, str, float, Optional[str]], ...]:
    """``(name, path, weight, path_2)`` per active spec, resolved and family-gated like the Images load; a cache-only
    load resolves Hub adapters from the cache alone. ``expert_pair`` (a two-expert model): every spec must be a
    catalog pair, whose ``path_2`` adapts ``transformer_2``; elsewhere ``path_2`` is None."""
    from core.inference import diffusion_lora

    resolved = diffusion_lora.resolve_specs(
        [(i, w) for (i, w) in specs if w != 0],
        family = family,
        hf_token = hf_token,
        cancel_event = cancel_event,
        local_files_only = local_files_only,
    )
    bad = [r.id for r in resolved if r.fmt != "safetensors"]
    if bad:
        raise diffusion_lora.LoraSelectionError(
            f"GGUF LoRA adapters cannot be baked into a video model ({', '.join(bad)}); use a .safetensors adapter."
        )
    out: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for r in resolved:
        if bool(r.path_2) != expert_pair:
            raise diffusion_lora.LoraSelectionError(_pair_mismatch(r.id, expert_pair))
        name, n = r.alias, 1
        while name in seen:
            n += 1
            name = f"{r.alias}_{n}"
        seen.add(name)
        for path in filter(None, (r.path, r.path_2)):
            _fetch_hub_adapter_config(path, hf_token, local_files_only)
            unsupported = _unbakeable_feature(path)
            if unsupported:
                raise diffusion_lora.LoraSelectionError(
                    f"'{r.id}' is a {unsupported} adapter; video loads bake plain LoRA deltas into the weights only, so "
                    "it would merge wrong. Use a plain LoRA adapter."
                )
        out.append((name, r.path, r.weight, r.path_2))
    return tuple(out)


def _pair_mismatch(lora_id: str, expert_pair: bool) -> str:
    if expert_pair:
        return (
            f"'{lora_id}' is one adapter, but this two-expert model swaps between a high-noise and a low-noise "
            "denoiser, so it needs a pair: pick a two-expert LoRA pair from the catalog (for example "
            "lightx2v/Wan2.2-Distill-Loras)."
        )
    return f"'{lora_id}' is a two-expert LoRA pair; it loads only on a two-expert model such as Wan2.2-T2V-A14B."


def _is_pair(lora_id: str) -> bool:
    """A curated two-expert pair (only the in-memory catalog holds pairs; anything else resolves to one file)."""
    from core.inference import diffusion_lora
    return any(e.id == lora_id and e.weight_name_2 for e in diffusion_lora._CURATED)


def recommended_scheduler(pipe: Any) -> dict:
    """Scheduler overrides the baked step-distilled adapter wants (lists kept); {} when none."""
    from core.inference import diffusion_lora

    rows = diffusion_lora.recommended_scheduler(
        i for (i, _w) in getattr(pipe, BAKED_SPECS_ATTR, ()) or ()
    )
    return {k: list(v) if isinstance(v, tuple) else v for k, v in rows}


def recommended_params(pipe: Any) -> Optional[tuple[int, float]]:
    """``(steps, guidance)`` a baked step-distilled adapter asks for (its catalog preset), else None."""
    from core.inference import diffusion_lora
    return diffusion_lora.recommended_params(
        i for (i, _w) in getattr(pipe, BAKED_SPECS_ATTR, ()) or ()
    )


def _unbakeable_feature(path: str) -> Optional[str]:
    """What a plain ``scale * B @ A`` weight bake cannot reproduce: DoRA's learned per-weight magnitude, a trained
    lora_B bias (PEFT lora_bias), transposed fan_in_fan_out factors, or full-weight tensors beside the factors. None
    for a plain adapter."""
    meta = _file_metadata(path)
    if meta.get("use_dora"):
        return "DoRA"
    if meta.get("lora_bias"):
        return "LoRA-bias"
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework = "pt", device = "cpu") as f:
            keys = list(f.keys())
    except Exception:  # noqa: BLE001 - unreadable here fails later with its own error
        return None
    if any("lora_magnitude_vector" in k or k.endswith(".dora_scale") for k in keys):
        return "DoRA"
    if any(k.endswith((".lora_B.bias", ".lora_B.default.bias", ".lora_up.bias")) for k in keys):
        return "LoRA-bias"
    if meta.get("fan_in_fan_out"):
        return "fan_in_fan_out"
    # Anything but A / B / alpha (modules_to_save copies, .diff / .diff_b full deltas) would be skipped by the bake.
    known = _A_SUFFIXES + _B_SUFFIXES + (".alpha",)
    if any(not k.endswith(known) for k in keys):
        return "full-weight (modules_to_save or diff)"
    return None


def _hub_cache_location(path: str) -> Optional[tuple[str, str, str, str]]:
    """``(cache_dir, repo_id, revision, file in repo)`` for a file inside a Hugging Face cache snapshot, else None."""
    parts = Path(str(path)).parts
    for i, part in enumerate(parts):
        if part.startswith("models--") and len(parts) > i + 3 and parts[i + 1] == "snapshots":
            owner_name = part[len("models--") :].split("--", 1)
            if len(owner_name) != 2:
                return None
            return (
                str(Path(*parts[:i])),
                "/".join(owner_name),
                parts[i + 2],
                "/".join(parts[i + 3 :]),
            )
    return None


def _fetch_hub_adapter_config(path: str, hf_token: Optional[str], local_files_only: bool) -> None:
    """A Hub PEFT adapter may keep lora_alpha / rsLoRA only in adapter_config.json beside the weights; the resolver
    downloads the weights alone, so fetch the config too (same revision, cache-only when asked). Absent: no-op."""
    if Path(str(path)).with_name("adapter_config.json").is_file():
        return
    where = _hub_cache_location(path)
    if where is None:
        return
    cache_dir, repo_id, revision, filename = where
    folder = filename.rsplit("/", 1)[0] + "/" if "/" in filename else ""
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id,
            folder + "adapter_config.json",
            revision = revision,
            token = hf_token,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
        )
    except Exception:  # noqa: BLE001 - no config in the repo (or offline): the file's own metadata decides
        pass


def _file_metadata(path: str) -> dict:
    """The adapter config a file carries (diffusers ``lora_adapter_metadata``, ``transformer.`` stripped), else a
    PEFT ``adapter_config.json`` beside it, else ``{}``."""
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework = "pt", device = "cpu") as f:
            raw = (f.metadata() or {}).get("lora_adapter_metadata")
        if raw:
            meta = json.loads(raw)
            return {
                k.split(".", 1)[1] if k.startswith("transformer.") else k: v
                for k, v in meta.items()
            }
    except Exception:  # noqa: BLE001 - unreadable metadata reads as alpha = r
        pass
    sidecar = Path(str(path)).with_name("adapter_config.json")
    if sidecar.is_file():
        try:
            return dict(json.loads(sidecar.read_text(encoding = "utf-8")))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _studio_sidecar_alpha(path: str) -> Optional[tuple[int, float]]:
    """``(rank, alpha)`` from the Studio trainer's ``<stem>.json``: LTX-2 adapters are saved without adapter
    metadata, so this is the only record of an alpha that differs from the rank."""
    try:
        data = json.loads(Path(str(path)).with_suffix(".json").read_text(encoding = "utf-8"))
        rank, alpha = int(data["lora_rank"]), float(data["lora_alpha"])
    except Exception:  # noqa: BLE001 - absent or foreign sidecar
        return None
    return (rank, alpha) if rank > 0 and alpha > 0 else None


def _pattern_value(patterns: Any, module: str, default: Any) -> Any:
    import re
    for key, value in (patterns or {}).items():
        if module == key or module.endswith("." + key) or re.fullmatch(key, module):
            return value
    return default


def read_deltas(files: Iterable[tuple[str, float]], *, family: Optional[str] = None) -> dict:
    """``{module path: [(A, B, scale * weight), ...]}`` for ``[(path, weight), ...]`` diffusers / PEFT LoRA files.
    Scale is ``lora_alpha / r`` from the file's adapter config (``/ sqrt(r)`` under rsLoRA), else from ``.alpha``
    keys, else from the Studio trainer sidecar, else 1 (alpha = r, as diffusers reads such a file). ``family``
    ``minimax-h3`` also maps the community layout (fused qkv, gated fc1) onto the diffusers module names."""
    from safetensors.torch import load_file

    out: dict = {}
    for path, weight in files:
        if not float(weight):
            continue
        state = load_file(str(path))
        meta = _file_metadata(str(path))
        sidecar = None if meta else _studio_sidecar_alpha(str(path))
        parts: dict = {}
        for key, tensor in state.items():
            name = key
            for prefix in _PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
            for slot, suffixes in (("a", _A_SUFFIXES), ("b", _B_SUFFIXES), ("alpha", (".alpha",))):
                hit = next((s for s in suffixes if name.endswith(s)), None)
                if hit is not None:
                    parts.setdefault(_diffusers_module_name(name[: -len(hit)]), {})[slot] = tensor
                    break
        if not parts:
            raise ValueError(f"{Path(str(path)).name} holds no LoRA weights this loader recognises")
        for module, slots in parts.items():
            if "a" not in slots or "b" not in slots:
                raise ValueError(
                    f"{Path(str(path)).name}: {module} has only one of its LoRA factors"
                )
            a, b = slots["a"].float(), slots["b"].float()
            r = int(a.shape[0])
            rank = int(_pattern_value(meta.get("rank_pattern"), module, meta.get("r", r)) or r)
            if "alpha" in slots:
                alpha = float(slots["alpha"].float().reshape(-1)[0])
            elif sidecar is not None and sidecar[0] == r:
                alpha = sidecar[1]
            else:
                alpha = float(
                    _pattern_value(meta.get("alpha_pattern"), module, meta.get("lora_alpha", rank))
                    or rank
                )
            scale = alpha / (math.sqrt(rank) if meta.get("use_rslora") else rank)
            split = _h3_community_targets(module, b) if family == _H3_FAMILY else None
            for target, part in split if split is not None else ((module, b),):
                out.setdefault(target, []).append((a, part, scale * float(weight)))
    return out


def _plain_float(tensor: Any) -> bool:
    import torch
    return (
        type(tensor) in (torch.Tensor, torch.nn.Parameter)
        and tensor.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and not tensor.is_meta
    )


def bake_into_module_(
    module: Any,
    files: Iterable[tuple[str, float]],
    *,
    device: Any = None,
    label: str = "denoiser",
    family: Optional[str] = None,
) -> int:
    """Add every LoRA delta in ``files`` to ``module``'s dense Linear weights in place (parameter identity kept);
    returns the number of layers merged. Every adapted layer must exist and hold a plain float weight, else nothing
    is merged and ValueError says why: a partial bake would be a silently different model."""
    import torch

    deltas = read_deltas(files, family = family)
    params = dict(module.named_parameters())
    missing = [m for m in deltas if f"{m}.weight" not in params]
    if missing:
        modules = dict(module.named_modules())
        absent = [m for m in missing if m not in modules]
        names = absent or missing
        shown = ", ".join(sorted(names)[:4]) + (" ..." if len(names) > 4 else "")
        if absent:
            raise ValueError(
                f"This LoRA adapts layers the {label} does not have ({shown}); it was trained for a different model."
            )
        # the layers exist but keep quantized codes (e.g. a ModelOpt FP8 runtime) rather than a float weight
        raise ValueError(
            f"The {label} layers {shown} hold quantized weights, not a dense float weight; LoRAs bake only into "
            "the bfloat16 denoiser."
        )
    for m, terms in deltas.items():
        weight = params[f"{m}.weight"]
        if not _plain_float(weight.data):
            raise ValueError(
                f"The {label} layer {m} is not a dense float weight here ({type(weight.data).__name__}, "
                f"{weight.dtype}); LoRAs bake only into the bfloat16 denoiser."
            )
        want = tuple(weight.shape)
        for a, b, _scale in terms:
            got = (int(b.shape[0]), *tuple(a.shape[1:]))
            if a.ndim != 2 or b.ndim != 2 or got != want or b.shape[1] != a.shape[0]:
                raise ValueError(f"LoRA for {m} is {got}, the {label} layer is {want}")
    with torch.no_grad():
        for m, terms in deltas.items():
            param = params[f"{m}.weight"]
            dev = torch.device(device) if device is not None else param.device
            w = param.data.to(dev, torch.float32)
            for a, b, scale in terms:
                w.addmm_(b.to(dev, torch.float32), a.to(dev, torch.float32), alpha = float(scale))
            param.data.copy_(w.to(param.dtype))
            del w
    if deltas:
        # Read by every cache writer keyed on the base's shards: a merged build must never be served as the base.
        setattr(module, LORAS_BAKED_ATTR, True)
    return len(deltas)


SIDE_BRANCH_ENV = "UNSLOTH_VIDEO_LORA_SIDE_BRANCH"
SIDE_BRANCH_ADAPTER = "unsloth_video_lora"
SIDE_BRANCH_ATTR = "_unsloth_lora_side_branch"


def side_branch_enabled(fam: Any = None) -> bool:
    """MiniMax-H3 keeps LoRAs as an unmerged low-rank branch rather than a bf16 bake, whose nearest rounding drops most
    of a LoRA's per-weight update; ``UNSLOTH_VIDEO_LORA_SIDE_BRANCH=0`` restores the bake."""
    import os

    if fam is not None and getattr(fam, "name", None) != _H3_FAMILY:
        return False
    return (os.environ.get(SIDE_BRANCH_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def attach_side_branch_(
    module: Any,
    files: Iterable[tuple[str, float]],
    *,
    label: str = "denoiser",
    family: Optional[str] = None,
    dtype: Any = None,
) -> int:
    """Attach every LoRA in ``files`` to ``module`` as one unmerged PEFT LoRA layer per adapted Linear: the forward is
    ``base(x) + (x @ A.T) @ B.T`` with ``A`` the adapters' factors stacked along the rank and ``B`` carrying each
    file's ``scale * weight``, so the base weights (bf16 or hosted int8 codes) are never rewritten and the update is
    never rounded into them. Validates every layer first; returns the number of adapted layers."""
    import torch
    from peft import LoraConfig, inject_adapter_in_model

    modules = dict(module.named_modules())
    deltas = {}
    for m, terms in read_deltas(files, family = family).items():
        layer = modules.get(m)
        # a small-M int8 layer sits inside a row-padding wrapper (diffusion_quant_pad.PadToMinM): adapt its Linear
        while (
            layer is not None
            and not isinstance(layer, torch.nn.Linear)
            and isinstance(getattr(layer, "inner", None), torch.nn.Module)
        ):
            m, layer = m + ".inner", layer.inner
        deltas[m] = terms
        if layer is None:
            raise ValueError(
                f"This LoRA adapts a layer the {label} does not have ({m}); it was trained for a different model."
            )
        if not isinstance(layer, torch.nn.Linear):
            raise ValueError(f"The {label} layer {m} is a {type(layer).__name__}, not a Linear.")
        want = (int(layer.out_features), int(layer.in_features))
        for a, b, _scale in terms:
            got = (int(b.shape[0]), int(a.shape[1]))
            if got != want or b.shape[1] != a.shape[0]:
                raise ValueError(f"LoRA for {m} is {got}, the {label} layer is {want}")
    if not deltas:
        return 0
    factors = {}
    for m, terms in deltas.items():
        a = torch.cat([t[0].float() for t in terms], dim = 0)
        b = torch.cat([t[1].float() * float(t[2]) for t in terms], dim = 1)
        factors[m] = (a, b)
    ranks = {m: int(a.shape[0]) for m, (a, _b) in factors.items()}
    first = next(iter(ranks.values()))
    config = LoraConfig(
        r = first,
        lora_alpha = first,
        target_modules = sorted(factors),
        rank_pattern = {m: r for m, r in ranks.items()},
        alpha_pattern = {m: r for m, r in ranks.items()},
        lora_dropout = 0.0,
        bias = "none",
    )
    # Plain lora.Linear for every target: PEFT would hand a torchao (hosted int8) weight to TorchaoLoraLinear, whose
    # merge requantizes the codes; the side branch never merges, so the base stays any Linear's own forward.
    from peft.tuners.lora.layer import Linear as LoraLinear

    config._register_custom_module({torch.nn.Linear: LoraLinear})
    inject_adapter_in_model(config, module, adapter_name = SIDE_BRANCH_ADAPTER)
    wrapped = dict(module.named_modules())
    with torch.no_grad():
        for m, (a, b) in factors.items():
            layer = wrapped[m]
            base = layer.get_base_layer()
            target = dtype or (
                base.weight.dtype if base.weight.dtype.is_floating_point else torch.bfloat16
            )
            la, lb = layer.lora_A[SIDE_BRANCH_ADAPTER], layer.lora_B[SIDE_BRANCH_ADAPTER]
            if float(layer.scaling[SIDE_BRANCH_ADAPTER]) != 1.0:
                raise ValueError(
                    f"{m}: side-branch scaling is {layer.scaling[SIDE_BRANCH_ADAPTER]}, expected 1"
                )
            la.weight = torch.nn.Parameter(a.to(target).contiguous(), requires_grad = False)
            lb.weight = torch.nn.Parameter(b.to(target).contiguous(), requires_grad = False)
    module.requires_grad_(False)
    setattr(module, SIDE_BRANCH_ATTR, len(factors))
    # Read by every cache writer keyed on the base's shards: an adapted denoiser must never be served as the base.
    setattr(module, LORAS_BAKED_ATTR, True)
    return len(factors)


def mark_baked(pipe: Any, baked: tuple, specs: Any) -> None:
    setattr(pipe, LORAS_ATTR, tuple(baked))
    setattr(pipe, LORAS_BAKED_ATTR, bool(baked))
    setattr(pipe, BAKED_SPECS_ATTR, tuple(active_specs(specs)) if baked else ())


def baked_lora_names(pipe: Any) -> list[str]:
    if pipe is None or not getattr(pipe, LORAS_BAKED_ATTR, False):
        return []
    names = []
    for entry in getattr(pipe, LORAS_ATTR, ()) or ():
        try:
            name = entry[0]
        except Exception:  # noqa: BLE001
            continue
        if name:
            names.append(str(name))
    return names


def baked_lora_specs(pipe: Any) -> list[dict]:
    if pipe is None or not getattr(pipe, LORAS_BAKED_ATTR, False):
        return []
    return [{"id": i, "weight": w} for (i, w) in getattr(pipe, BAKED_SPECS_ATTR, ()) or ()]


# ``lora_fuse`` status of a video load (the images status field, one value here): every video LoRA merges into the
# dense bf16 denoiser at load, so a load with adapters is always fused.
LORA_FUSE_BAKED = "baked"


def lora_fuse_status(pipe: Any) -> Optional[str]:
    """``baked`` when the load merged adapters into the denoiser, else None. Never raises."""
    try:
        return LORA_FUSE_BAKED if baked_lora_names(pipe) else None
    except Exception:  # noqa: BLE001 -- status must never fail
        return None
