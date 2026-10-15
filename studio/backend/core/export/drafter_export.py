# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Carry a drafter trained with the fine-tune (core/training/drafter.py) into merged and GGUF exports.

The MTP head needs no copying: the export reload re-attaches `mtp_head.safetensors` and the merged / GGUF
savers stitch it. An EAGLE-3 drafter is a separate model, copied next to the export. Each export gets a
`drafter.json` saying how to serve the pair; Studio's llama-server launcher reads it beside a GGUF.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

DRAFTER_MANIFEST = "drafter.json"
EAGLE3_DIRNAME = "eagle3"


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding = "utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def training_manifest(checkpoint_dir: Optional[str]) -> Optional[dict]:
    """The manifest the training run wrote, when its drafter is still complete on disk."""
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None
    manifest = _read_json(os.path.join(checkpoint_dir, DRAFTER_MANIFEST))
    if not manifest or manifest.get("strategy") not in ("mtp", "eagle3"):
        return None
    if manifest["strategy"] == "eagle3" and not os.path.isfile(
        os.path.join(checkpoint_dir, EAGLE3_DIRNAME, "config.json")
    ):
        return None
    if manifest["strategy"] == "mtp" and not (
        os.path.isfile(os.path.join(checkpoint_dir, "mtp_head.safetensors"))
        or not os.path.isfile(os.path.join(checkpoint_dir, "adapter_config.json"))
    ):
        return None
    return manifest


def _engines(manifest: dict) -> dict:
    # Manifests from before per-engine support was recorded meant both.
    engines = manifest.get("engines")
    if not isinstance(engines, dict):
        return {"vllm": True, "llama_cpp": True}
    return {"vllm": bool(engines.get("vllm")), "llama_cpp": bool(engines.get("llama_cpp"))}


def clear_stale(directory: str, ggufs: Optional[list[str]] = None) -> None:
    """Remove a previous export's drafter from a reused folder, so nothing pairs it with this model."""
    info = _read_json(os.path.join(directory, DRAFTER_MANIFEST))
    if not info or info.get("strategy") not in ("mtp", "eagle3"):
        return
    listed = info.get("ggufs") or []
    if ggufs is not None and listed and not set(listed) & {os.path.basename(g) for g in ggufs}:
        return  # describes other GGUFs in this folder
    for name in info.get("files") or []:
        path = os.path.join(directory, os.path.basename(name))
        if os.path.basename(name) == EAGLE3_DIRNAME and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors = True)
        elif name.lower().endswith(".gguf") and os.path.isfile(path):
            os.remove(path)
    os.remove(os.path.join(directory, DRAFTER_MANIFEST))


def _vllm_command(target: str, spec: dict, hub: bool) -> str:
    """`vllm serve` for the exported pair. Hub repos: the repo id, after a download when the EAGLE-3
    drafter sits in a subfolder (vLLM loads a speculative model from a repo id or a local path only)."""
    prefix = ""
    if hub and spec["method"] == "eagle3":
        local = target.rstrip("/").split("/")[-1] or "model"
        prefix = f"hf download {shlex.quote(target)} --local-dir {shlex.quote(local)} && "
        target, spec = local, dict(spec, model = f"{local}/{EAGLE3_DIRNAME}")
    return f"{prefix}vllm serve {shlex.quote(target)} --speculative-config " + shlex.quote(
        json.dumps(spec, separators = (",", ":"))
    )


def _steps(manifest: dict) -> int:
    try:
        return max(1, int(manifest.get("draft_steps") or 3))
    except (TypeError, ValueError):
        return 3


def _write(directory: str, info: dict) -> None:
    path = os.path.join(directory, DRAFTER_MANIFEST)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding = "utf-8") as f:
        json.dump(info, f, indent = 1)
    os.replace(tmp, path)


def _set_verifier(eagle_dir: str, verifier: str) -> None:
    config_path = os.path.join(eagle_dir, "config.json")
    config = _read_json(config_path)
    if config is None:
        raise RuntimeError(f"{config_path} is not a readable EAGLE-3 config")
    spec = config.setdefault("speculators_config", {})
    spec.setdefault("verifier", {})["name_or_path"] = verifier
    with open(config_path, "w", encoding = "utf-8") as f:
        json.dump(config, f, indent = 2)


def package_merged(
    checkpoint_dir: Optional[str],
    final_dir: str,
    *,
    verifier: Optional[str] = None,
    allow_mtp: bool = True,
) -> Optional[dict]:
    """Write `drafter.json` (and copy the EAGLE-3 drafter) into a merged export. None when the run
    trained no drafter. `verifier` is what the drafter's config names as its target: the Hub repo on a
    push, else the base model id (vLLM refuses a local verifier path inside a speculators config, so
    local serving passes the target explicitly). `allow_mtp` is False for exports that cannot carry
    a stitched head (compressed-tensors, torchao): only an EAGLE-3 drafter ships with those."""
    manifest = training_manifest(checkpoint_dir)
    if manifest is None or (manifest["strategy"] == "mtp" and not allow_mtp):
        clear_stale(final_dir)
        return None
    strategy, steps = manifest["strategy"], _steps(manifest)
    engines = _engines(manifest)
    hub = verifier is not None
    verifier = verifier or manifest.get("base_model") or ""
    target = verifier if hub else os.path.abspath(final_dir)
    if strategy == "mtp":
        spec = {"method": "mtp", "num_speculative_tokens": steps}
        files: list[str] = []
    else:
        dest = os.path.join(final_dir, EAGLE3_DIRNAME)
        shutil.rmtree(dest, ignore_errors = True)
        shutil.copytree(os.path.join(checkpoint_dir, EAGLE3_DIRNAME), dest)
        _set_verifier(dest, verifier)
        spec = {
            "method": "eagle3",
            "model": os.path.join(target, EAGLE3_DIRNAME),
            "num_speculative_tokens": steps,
        }
        files = [EAGLE3_DIRNAME]
    info = {
        "strategy": strategy,
        "files": files,
        "engines": engines,
        "vllm_command": _vllm_command(target, spec, hub) if engines["vllm"] else None,
        "llama_cpp_args": None,
        "base_model": manifest.get("base_model"),
        "verifier": verifier if strategy == "eagle3" else None,
        "draft_steps": steps,
        "metrics": manifest.get("metrics") or {},
    }
    _write(final_dir, info)
    return info


def eagle3_gguf_name(ggufs: list[str]) -> str:
    stem = os.path.basename(ggufs[0]).split(".")[0] if ggufs else ""
    # `eagle3-` prefix: Studio's model pickers treat it as a drafter companion, not a model.
    return f"eagle3-{stem or 'model'}-F16.gguf"


def package_gguf(
    checkpoint_dir: Optional[str], merged_dir: Optional[str], gguf_dir: str, ggufs: list[str]
) -> tuple[Optional[dict], list[str]]:
    """(drafter info, extra files written into gguf_dir). The MTP head is already inside each GGUF; an
    EAGLE-3 drafter is converted beside them against the merged target (the converter reads its config
    and tokenizer). Raises when the drafter cannot be converted: a pair missing its drafter is incomplete."""
    manifest = training_manifest(checkpoint_dir)
    engines = _engines(manifest) if manifest else {}
    if manifest is None or not engines["llama_cpp"]:
        clear_stale(gguf_dir, ggufs)
        return None, []
    strategy, steps = manifest["strategy"], _steps(manifest)
    extra: list[str] = []
    names = [os.path.basename(g) for g in ggufs]
    if strategy == "mtp":
        args = ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(steps)]
    else:
        if not merged_dir or not os.path.isfile(os.path.join(merged_dir, "config.json")):
            raise RuntimeError(
                "the merged target checkpoint needed to convert the EAGLE-3 drafter is missing"
            )
        from unsloth_zoo.speculative import convert_eagle3_to_gguf

        out = os.path.join(gguf_dir, eagle3_gguf_name(ggufs))
        convert_eagle3_to_gguf(
            os.path.join(checkpoint_dir, EAGLE3_DIRNAME), merged_dir, out, outtype = "f16"
        )
        if not os.path.isfile(out):
            raise RuntimeError("EAGLE-3 drafter conversion wrote no GGUF")
        extra.append(out)
        args = [
            "--model-draft",
            os.path.basename(out),
            "--spec-type",
            "draft-eagle3",
            "--spec-draft-n-max",
            str(steps),
        ]
    info = {
        "strategy": strategy,
        "files": [os.path.basename(f) for f in extra],
        "ggufs": names,
        "engines": engines,
        "mtp_trained": strategy == "mtp",
        "vllm_command": None,
        "llama_cpp_args": args,
        "base_model": manifest.get("base_model"),
        "draft_steps": steps,
        "metrics": manifest.get("metrics") or {},
    }
    _write(gguf_dir, info)
    extra.append(os.path.join(gguf_dir, DRAFTER_MANIFEST))
    return info, extra


def package_lora(
    checkpoint_dir: Optional[str],
    adapter_dir: str,
    head_saved: Optional[bool] = None,
) -> Optional[dict]:
    """Carry the training manifest with a LoRA export that kept the trained MTP head beside it, so a
    later merge / GGUF of that adapter still marks the head as trained (Studio's MTP launch policy).
    `head_saved` overrides the sidecar check (a Hub push saves the adapter elsewhere)."""
    manifest = training_manifest(checkpoint_dir)
    if head_saved is None:
        head_saved = os.path.isfile(os.path.join(adapter_dir, "mtp_head.safetensors"))
    if manifest is None or manifest["strategy"] != "mtp" or not head_saved:
        clear_stale(adapter_dir)
        return None
    keep = (
        "version",
        "strategy",
        "base_model",
        "draft_steps",
        "metrics",
        "engines",
        "family",
        "llama_cpp_status",
        "vllm",
        "llama_cpp",
        "created_at",
    )
    info = {k: manifest[k] for k in keep if k in manifest}
    info["files"] = ["mtp_head.safetensors"]
    _write(adapter_dir, info)
    return info


def export_drafter_details(output_path: Optional[str]) -> Optional[dict[str, Any]]:
    """The `drafter` block an export response carries (the frontend contract)."""
    if not output_path or not os.path.isdir(output_path):
        return None
    info = _read_json(os.path.join(output_path, DRAFTER_MANIFEST))
    if not info or info.get("strategy") not in ("mtp", "eagle3"):
        return None
    return {
        "strategy": info["strategy"],
        "files": list(info.get("files") or []),
        "vllm_command": info.get("vllm_command"),
        "llama_cpp_args": info.get("llama_cpp_args"),
    }


def eagle3_sidecar_path(gguf_path: Optional[str]) -> Optional[str]:
    """The EAGLE-3 drafter GGUF exported next to this GGUF, when its manifest pairs them."""
    info = gguf_drafter_sidecar(gguf_path)
    if not info or info.get("strategy") != "eagle3":
        return None
    directory = os.path.dirname(os.path.abspath(gguf_path))
    for name in info.get("files") or []:
        path = os.path.join(directory, os.path.basename(name))
        if name.lower().endswith(".gguf") and os.path.isfile(path):
            return path
    return None


def gguf_drafter_sidecar(gguf_path: Optional[str]) -> Optional[dict]:
    """The export manifest next to a GGUF, when it names that GGUF (llama-server launch policy)."""
    if not gguf_path:
        return None
    directory = os.path.dirname(os.path.abspath(gguf_path))
    info = _read_json(os.path.join(directory, DRAFTER_MANIFEST))
    if not info or info.get("strategy") not in ("mtp", "eagle3"):
        return None
    ggufs = info.get("ggufs") or []
    if ggufs and os.path.basename(gguf_path) not in ggufs:
        return None
    return info
