# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Convert a published ``.pt`` pre-quant checkpoint (denoiser or text encoder) to ``.safetensors``.

    python scripts/convert_prequant_to_safetensors.py SRC.pt [SRC2.pt ...] --out-dir DIR \
        [--repo owner/name --revision SHA]

Writes ``DIR/<owner>/<name>/<same stem>.safetensors`` (``DIR/<stem>.safetensors`` without ``--repo``), the
layout ``UNSLOTH_DIFFUSION_PREQUANT_MIRROR`` reads. Nothing is uploaded.

Denoisers use the container Qwen-Image-2.1's published artifacts already use (``prequant_safetensors``:
torchao's flattened tensors plus ``unsloth_format`` / ``unsloth_metadata``), so a released Studio that
resolves ``<Model>-<SCHEME>.safetensors`` reads it too. Two header keys are added: ``unsloth_quant_layout``
(how to rebuild every weight from its plain tensors without torchao, and for int8 that the weights came
from torchao v1, so torchao <= 0.17 rebuilds the v1 class the ``.pt`` held) and ``unsloth_source`` (the
file, its sha256 and the repo revision it was converted from). ``unsloth_metadata`` is copied unchanged.

Text encoders are plain fp8 / bf16 tensors and are written in the plain layout
``load_plain_prequant_safetensors`` reads, with no torchao involved.

Every weight is checked after writing: the file is read back through Studio's loader and each tensor
compared bit for bit with the ``.pt`` as the same install reads it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"


def sha256_of(path: str, chunk: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _record_v1_facts(facts: list):
    """Wrap the legacy decoder so each v1 weight's facts are recorded before it is rebuilt (torchao >= 0.18).

    Patches a module global for the duration of one read, so conversions must not run concurrently in one
    process (this is a single-threaded CLI)."""
    from core.inference import prequant_legacy_int8 as legacy

    original = legacy._rebuild_weight

    def recording(name, w, standins, api):
        facts.append(_v1_facts_of(w, standins))
        return original(name, w, standins, api)

    legacy._rebuild_weight = recording
    return lambda: setattr(legacy, "_rebuild_weight", original)


def _v1_facts_of(w, standins = None) -> dict:
    """The INT8_V1_FACTS view of one v1 weight (a real torchao object or the decoder's stand-in)."""
    from core.inference.prequant_legacy_int8 import _ACT_QUANT, _LAYOUT

    state = getattr(w, "__dict__", {})
    aqt = state.get("original_weight_tensor")
    impl = getattr(aqt, "tensor_impl", None)
    act = state.get("input_quant_func")
    scale = getattr(impl, "scale", None)
    zero_point = getattr(impl, "zero_point", None)
    domain = getattr(aqt, "zero_point_domain", None)
    layout = getattr(impl, "_layout", None)
    if standins is not None:
        # torchao >= 0.18 read the pickle into stand-ins registered under the v1 names: report those names.
        if act is standins.get(_ACT_QUANT):
            act = _ACT_QUANT
        if isinstance(layout, standins.get(_LAYOUT, ())):
            layout = _LAYOUT
    return {
        "act_quant": (act if isinstance(act, str) else getattr(act, "__name__", str(act))).rsplit(
            ".", 1
        )[-1],
        "quant_kwargs": dict(state.get("quant_kwargs") or {}),
        "zero_point": None if zero_point is None else "tensor",
        "zero_point_domain": getattr(domain, "name", str(domain)),
        "quant_min": getattr(aqt, "quant_min", "?"),
        "quant_max": getattr(aqt, "quant_max", "?"),
        "scale_shape": "out" if scale is not None and scale.dim() == 1 else "other",
        "layout": layout.rsplit(".", 1)[-1] if isinstance(layout, str) else type(layout).__name__,
    }


def _to_int8_tensor_dict(state_dict: dict, facts: list) -> dict:
    """torchao <= 0.17 hands back v1 objects: record their facts and rebuild them as Int8Tensor to flatten."""
    from core.inference.prequant_legacy_int8 import (
        LEGACY_INT8_CLASS_NAMES,
        _ACT_QUANT,
        _int8_tensor_api,
        _rebuild_weight,
        _resolve,
    )

    classes = {n: _resolve(n) for n in (*LEGACY_INT8_CLASS_NAMES, _ACT_QUANT)}
    laqt = classes[LEGACY_INT8_CLASS_NAMES[0]]
    if laqt is None:
        return state_dict
    api = _int8_tensor_api()
    out = {}
    for key, value in state_dict.items():
        if isinstance(value, laqt):
            facts.append(_v1_facts_of(value))
            out[key] = _rebuild_weight(key, value, classes, api)
        else:
            out[key] = value
    return out


def _canonical(value):
    """``(class name, {attr: tensor or value})`` of a weight, recursing into torchao subclasses."""
    import torch

    if type(value) is torch.Tensor:
        return ("Tensor", value)
    flatten = getattr(value, "__tensor_flatten__", None)
    if flatten is None:
        return (type(value).__name__, value)
    names, ctx = flatten()
    return (type(value).__name__, {n: _canonical(getattr(value, n)) for n in names}, repr(ctx))


def _same(a, b) -> bool:
    import torch

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            return False
        # Raw bytes, not torch.equal: -0.0 == 0.0 and NaN != NaN would otherwise decide.
        return a.numel() == 0 or torch.equal(
            a.detach().contiguous().cpu().view(torch.uint8),
            b.detach().contiguous().cpu().view(torch.uint8),
        )
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict = True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def compare_state_dicts(reference: dict, candidate: dict) -> list:
    """Keys whose class, attributes or tensor bytes differ (empty = bit-identical)."""
    bad = sorted(set(reference) ^ set(candidate))
    for key in reference:
        if key in candidate and not _same(_canonical(reference[key]), _canonical(candidate[key])):
            bad.append(key)
    return bad


def _untie(flat: dict) -> dict:
    """safetensors refuses tensors that share storage (tied embeddings): write each one as its own copy.

    Studio's text-encoder loader re-ties after ``load_state_dict`` (``tie_weights``), exactly as it does for the
    published safetensors encoders, so the copy costs disk only."""
    seen: set = set()
    out = {}
    for key, value in flat.items():
        value = value.contiguous()
        ptr = value.untyped_storage().data_ptr() if value.numel() else None
        if ptr is not None and ptr in seen:
            value = value.clone()
        elif ptr is not None:
            seen.add(ptr)
        out[key] = value
    return out


def write_plain_safetensors(
    path: str, *, fmt: str, state_dict: dict, metadata: dict, extra: dict
) -> None:
    """The torchao-free plain layout ``load_plain_prequant_safetensors`` reads (text encoders)."""
    import torch
    from safetensors.torch import save_file

    from core.inference.prequant_safetensors import (
        UNSLOTH_FORMAT_KEY,
        UNSLOTH_METADATA_KEY,
        UNSLOTH_ROOT_KEYS_KEY,
        UNSLOTH_ROOT_PREFIX,
    )

    names, roots, flat = [], [], {}
    for key, value in state_dict.items():
        if type(value) is not torch.Tensor:
            raise ValueError(f"{key!r} is a {type(value).__name__}, not a plain tensor")
        if "." in key:
            names.append(key)
            flat[key] = value.contiguous()
        else:
            roots.append(key)
            flat[f"{UNSLOTH_ROOT_PREFIX}{key}"] = value.contiguous()
    header = {name: json.dumps({"_type": "Tensor"}) for name in names}
    header["tensor_names"] = json.dumps(names)
    header[UNSLOTH_FORMAT_KEY] = fmt
    header[UNSLOTH_METADATA_KEY] = json.dumps(metadata, default = str)
    if roots:
        header[UNSLOTH_ROOT_KEYS_KEY] = json.dumps(roots)
    header.update(extra)
    save_file(_untie(flat), path, metadata = header)


def convert(
    src: str, out_dir: str, *, repo: str | None, revision: str | None, sha: str | None
) -> dict:
    import torch

    from core.inference import diffusion_prequant as dp
    from core.inference import prequant_safetensors as ps
    from core.inference.diffusion_te_prequant import TE_PREQUANT_FORMAT
    from core.inference.prequant_native import (
        INT8_SOURCE_V1,
        INT8_V1_FACTS,
        QUANT_LAYOUT_KEY,
        SOURCE_KEY,
        quant_layout_header,
    )

    t0 = time.time()
    name = os.path.basename(src)
    stem = os.path.splitext(name)[0] if name.endswith((".pt", ".pth")) else name
    dest_dir = os.path.join(out_dir, *repo.split("/")) if repo else out_dir
    os.makedirs(dest_dir, exist_ok = True)
    dst = os.path.join(dest_dir, stem + ".safetensors")
    source = {"file": name, "sha256": sha or sha256_of(src), "bytes": os.path.getsize(src)}
    if repo:
        source["repo"] = repo
    if revision:
        source["revision"] = revision
    try:
        import torchao
        source["converter_torchao"] = torchao.__version__
    except Exception:  # noqa: BLE001
        pass
    source["converter_torch"] = torch.__version__

    # Read as Studio does (allowlisted weights_only); on torchao >= 0.18 v1 int8 loads as Int8Tensor.
    facts: list = []
    restore = _record_v1_facts(facts)
    try:
        ckpt = dp._load_prequant_checkpoint(src, map_location = "cpu", mmap = True)
    finally:
        restore()
    fmt = ckpt.get("format")
    is_te = fmt == TE_PREQUANT_FORMAT
    metadata = dict(ckpt.get("metadata") or {})
    state_dict = ckpt["state_dict"]
    extra = {SOURCE_KEY: json.dumps(source)}
    if is_te:
        write_plain_safetensors(dst, fmt = fmt, state_dict = state_dict, metadata = metadata, extra = extra)
        back = ps.load_plain_prequant_safetensors(dst)
    else:
        if fmt not in dp.PREQUANT_FORMATS:
            raise ValueError(f"{src}: unrecognised format {fmt!r}")
        flat_sd = _to_int8_tensor_dict(state_dict, facts)
        int8_source = None
        if facts:
            int8 = sum(type(v).__name__ == "Int8Tensor" for v in flat_sd.values())
            if len(facts) != int8:
                raise ValueError(
                    f"{src}: {int8} int8 weights but {len(facts)} came from torchao v1"
                )
            distinct = {json.dumps(f, sort_keys = True, default = str) for f in facts}
            if distinct != {json.dumps(INT8_V1_FACTS, sort_keys = True)}:
                raise ValueError(
                    f"{src}: v1 int8 weights do not match the supported v1 layout: {sorted(distinct)[:3]}"
                )
            int8_source = INT8_SOURCE_V1
        layout = quant_layout_header(flat_sd, int8_source = int8_source)
        from safetensors.torch import save_file

        helpers = ps._torchao_helpers()
        if helpers is None:
            raise RuntimeError("torchao >= 0.16 is required to write the flattened layout")
        undotted = ps.unsupported_state_dict_keys(flat_sd)
        if undotted:
            raise ValueError(
                f"{src}: root-level quantized weights cannot be flattened: {undotted[:8]}"
            )
        roots = ps._root_level_keys(flat_sd)
        quantizable = {k: v for k, v in flat_sd.items() if k not in set(roots)}
        flat, torchao_metadata = helpers[0](quantizable)
        header = dict(torchao_metadata or {})
        header[ps.UNSLOTH_FORMAT_KEY] = str(fmt)
        header[ps.UNSLOTH_METADATA_KEY] = json.dumps(metadata, default = str)
        header[QUANT_LAYOUT_KEY] = json.dumps(layout)
        header.update(extra)
        if roots:
            flat = dict(flat)
            for key in roots:
                flat[f"{ps.UNSLOTH_ROOT_PREFIX}{key}"] = flat_sd[key].contiguous()
            header[ps.UNSLOTH_ROOT_KEYS_KEY] = json.dumps([str(k) for k in roots])
        save_file(_untie(dict(flat)), dst, metadata = header)
        back = ps.load_prequant_safetensors(dst, mmap = True)
    # Bit-identity against the .pt as this install reads it.
    bad = compare_state_dicts(state_dict, back["state_dict"])
    if bad:
        raise AssertionError(f"{dst}: {len(bad)} tensors differ from {src}, e.g. {bad[:3]}")
    if back["format"] != fmt or back["metadata"] != metadata:
        raise AssertionError(f"{dst}: format or metadata changed in the round trip")
    rec = {
        "src": src,
        "dst": dst,
        "kind": "text_encoder" if is_te else "transformer",
        "format": fmt,
        "scheme": metadata.get("scheme"),
        "tensors": len(state_dict),
        "quantized": sum(1 for v in state_dict.values() if type(v) is not torch.Tensor),
        "int8_source": None if is_te else layout.get("int8_source"),
        "reader": back.get("reader"),
        "src_bytes": source["bytes"],
        "src_sha256": source["sha256"],
        "dst_bytes": os.path.getsize(dst),
        "dst_sha256": sha256_of(dst),
        "seconds": round(time.time() - t0, 1),
        "verified_bit_identical": True,
    }
    return rec


def main(argv = None) -> int:
    p = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    p.add_argument("src", nargs = "+")
    p.add_argument("--out-dir", required = True)
    p.add_argument(
        "--repo", default = None, help = "owner/name the .pt came from (mirror layout + provenance)"
    )
    p.add_argument("--revision", default = None)
    p.add_argument("--sha256", default = None, help = "known sha256 of a single SRC (skips hashing it)")
    p.add_argument("--report", default = None, help = "append one JSON line per file here")
    args = p.parse_args(argv)
    sys.path.insert(0, str(BACKEND))
    for src in args.src:
        rec = convert(
            src,
            args.out_dir,
            repo = args.repo,
            revision = args.revision,
            sha = args.sha256 if len(args.src) == 1 else None,
        )
        line = json.dumps(rec)
        print(line, flush = True)
        if args.report:
            with open(args.report, "a") as fh:
                fh.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
