# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastFlowLM Q4NX output for the AMD Ryzen AI NPU (XDNA 2).

Kept free of torch/unsloth imports: the backend process converts an existing GGUF directly,
without loading a model into the export worker.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from loggers import get_logger

logger = get_logger(__name__)

# GGUF types FastFlowLM's Q4NX packs directly; any other quant is dequantized and rounded again.
SOURCE_QUANTS = ("q4_0", "q4_1", "q4_k_m")
# What FLM loads next to model.q4nx; it hard-exits without tokenizer_config.json, and reads the
# chat template from it or from chat_template.jinja, where transformers 5 saves it.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _installer():
    studio_dir = Path(__file__).resolve().parents[3]
    if str(studio_dir) not in sys.path:
        sys.path.insert(0, str(studio_dir))
    import install_q4nx_converter

    return install_q4nx_converter


def source_gguf(ggufs: List[str], quant_methods: List[str]) -> Optional[str]:
    """The exported GGUF to convert: the first selected quant Q4NX packs directly."""
    for quant in quant_methods:
        if quant not in SOURCE_QUANTS:
            continue
        for path in ggufs:
            if os.path.basename(path).lower().endswith(f".{quant}.gguf"):
                return path
    return None


def convert_gguf_to_q4nx(gguf_path: str, out_dir: Path) -> None:
    from utils.paths.storage_roots import studio_root

    script = _installer().install(studio_root() / "q4nx_converter")
    out_dir.mkdir(parents = True, exist_ok = True)
    logger.info(f"Converting {os.path.basename(gguf_path)} to Q4NX for the AMD NPU in {out_dir}")
    # Runs in this env: the converter only needs torch, gguf, einops and safetensors.
    subprocess.run(
        [sys.executable, str(script), "-i", gguf_path, "-o", str(out_dir)],
        cwd = str(script.parent),
        check = True,
    )
    if not (out_dir / "model.q4nx").is_file():
        raise RuntimeError(f"The Q4NX converter wrote no model.q4nx to {out_dir}")


def _fetch_base_file(base_model: str, name: str, token) -> Optional[Path]:
    local = Path(base_model)
    if local.is_dir():
        candidate = local / name
        return candidate if candidate.is_file() else None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return Path(hf_hub_download(base_model, name, token = token))
    except EntryNotFoundError:
        return None


def _gguf_chat_template(gguf_path: Path) -> Optional[str]:
    import gguf
    field = gguf.GGUFReader(str(gguf_path)).fields.get("tokenizer.chat_template")
    return field.contents() if field is not None else None


def convert_existing_gguf(
    gguf_path: Path,
    base_model: str,
    save_directory: Path,
    token = None,
) -> Path:
    """Convert a GGUF already on disk; config and tokenizer files come from ``base_model``.

    ``base_model`` is the original (non-GGUF) Hub repo or a local model folder. Returns the
    folder FastFlowLM loads.
    """
    out_dir = Path(save_directory) / f"{gguf_path.stem}-q4nx"
    found = {
        name: _fetch_base_file(base_model, name, token)
        for name in ("config.json", *TOKENIZER_FILES)
    }
    missing = [n for n in ("config.json", "tokenizer_config.json") if found[n] is None]
    if missing:
        raise RuntimeError(f"{base_model} has no {' or '.join(missing)}, which FastFlowLM needs.")
    convert_gguf_to_q4nx(str(gguf_path), out_dir)
    for name, path in found.items():
        # The HF tokenizer.json replaces the one the converter rebuilds from the GGUF.
        if path is not None:
            shutil.copyfile(path, out_dir / name)
    if found["chat_template.jinja"] is None:
        config = json.loads((out_dir / "tokenizer_config.json").read_text(encoding = "utf-8"))
        template = config.get("chat_template") or _gguf_chat_template(gguf_path)
        if template and not config.get("chat_template"):
            (out_dir / "chat_template.jinja").write_text(template, encoding = "utf-8")
    return out_dir
