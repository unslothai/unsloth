# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastFlowLM Q4NX output for the AMD Ryzen AI NPU (XDNA 2).

Kept free of torch/unsloth imports: the backend process converts an existing GGUF directly,
without loading a model into the export worker.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

from loggers import get_logger

logger = get_logger(__name__)

# GGUF types FastFlowLM's Q4NX packs directly; any other quant is dequantized and rounded again.
SOURCE_QUANTS = ("q4_0", "q4_1", "q4_k_m")
# Loaded beside model.q4nx; FLM hard-exits without tokenizer_config.json.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
# Read for token ids only. No config.json is written: FLM's own carries the flm_version its
# catalog checks, and one without it makes FLM delete the folder's weights and re-pull stock.
CONFIG_FILES = ("config.json", "generation_config.json")
CONVERTER_MODULES = ("torch", "gguf", "einops", "safetensors", "numpy", "mpmath")


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


def require_converter_deps() -> None:
    """Raise before any download when this env cannot run the converter (e.g. --no-torch)."""
    missing = [m for m in CONVERTER_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(
            f"The Q4NX converter needs {', '.join(missing)}, which this Unsloth Studio install "
            "lacks (a GGUF-only install has no torch)."
        )


def _gguf_architecture(gguf_path: str) -> str:
    import gguf
    field = gguf.GGUFReader(gguf_path).fields.get("general.architecture")
    return field.contents() if field is not None else ""


# convert.py's __main__ overwrites sys.argv at the older pin, and importing it first is circular
# (q4nx.models.phi4 imports convert), so drive the package's own entry point.
_RUN_CONVERTER = (
    "import sys; from q4nx import create_converter; "
    "create_converter(sys.argv[1], '').convert(q4nx_path = sys.argv[2], weights_type = 'language')"
)


@contextlib.contextmanager
def staged_output(out_dir: Path):
    """Yield a fresh folder to build into; its files replace ``out_dir``'s only on success,
    so a failed re-conversion keeps the previous export."""

    # Only the save directory is account-checked; a planted symlink here would redirect writes.
    def refuse_symlink():
        if out_dir.is_symlink():
            raise RuntimeError(f"Refusing to write Q4NX output through the symlink {out_dir}")

    refuse_symlink()
    out_dir.parent.mkdir(parents = True, exist_ok = True)
    staging = Path(tempfile.mkdtemp(prefix = f".{out_dir.name}-", dir = out_dir.parent))
    try:
        yield staging
        out_dir.mkdir(exist_ok = True)
        # Again at publish: the conversion is long enough for the folder to be swapped.
        refuse_symlink()
        # The last model's companions must not survive beside the new weights.
        for name in ("model.q4nx", "config.json", *TOKENIZER_FILES):
            (out_dir / name).unlink(missing_ok = True)
        for path in staging.iterdir():
            os.replace(path, out_dir / path.name)
    finally:
        shutil.rmtree(staging, ignore_errors = True)


def convert_gguf_to_q4nx(gguf_path: str, out_dir: Path) -> None:
    from utils.paths.storage_roots import studio_root

    require_converter_deps()
    installer = _installer()
    name = installer.converter_for_architecture(_gguf_architecture(gguf_path))
    script = installer.install(studio_root() / "q4nx_converter", name = name)
    out_dir.mkdir(parents = True, exist_ok = True)
    logger.info(
        f"Converting {os.path.basename(gguf_path)} to Q4NX ({name}) for the AMD NPU in {out_dir}"
    )
    from utils.child_stdio import utf8_child_env
    from utils.process_lifetime import child_popen_kwargs, spawn_on_lifetime_thread
    from utils.subprocess_compat import windows_hidden_subprocess_kwargs

    # Dies with its parent: a cancelled export's worker must not leave a converter holding a model.
    process = spawn_on_lifetime_thread(
        lambda: subprocess.Popen(
            [sys.executable, "-c", _RUN_CONVERTER, gguf_path, str(out_dir)],
            cwd = str(script.parent),
            env = utf8_child_env(),
            stderr = subprocess.PIPE,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            **windows_hidden_subprocess_kwargs(),
            **child_popen_kwargs(),
        )
    )
    _, stderr = process.communicate()
    if process.returncode != 0:
        tail = (stderr or "").strip().splitlines()
        raise RuntimeError(
            f"The Q4NX converter exited with code {process.returncode}"
            + (f": {tail[-1]}" if tail else "")
        )
    if not (out_dir / "model.q4nx").is_file():
        raise RuntimeError(f"The Q4NX converter wrote no model.q4nx to {out_dir}")


def _fetch_base_file(base_model: str, name: str, token) -> Optional[Path]:
    local = Path(base_model).expanduser()
    if local.is_dir():
        from core.training.account_jobs import account_path

        candidate = local / name
        if not candidate.is_file():
            return None
        # Only the folder was account-checked; a file in it may link to another account's.
        account_path(candidate)
        return candidate
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


def _token_ids(value) -> List[int]:
    values = value if isinstance(value, list) else [value]
    return [v for v in values if isinstance(v, int) and not isinstance(v, bool)]


def _vocab_id(tokenizer_json: dict, token) -> Optional[int]:
    if isinstance(token, dict):
        token = token.get("content")
    if not isinstance(token, str):
        return None
    for added in tokenizer_json.get("added_tokens") or []:
        if added.get("content") == token:
            return added.get("id")
    vocab = (tokenizer_json.get("model") or {}).get("vocab")
    return vocab.get(token) if isinstance(vocab, dict) else None


def write_flm_tokenizer_config(out_dir: Path, *configs: Optional[dict]) -> None:
    """Add the ids FastFlowLM exits without (an ``eos_token_id`` array, ``bos_token_id`` with a
    ``bos_token``); its uploads list the configs' stop ids too (Llama 3.2: 128001, 128008, 128009).
    """
    path = out_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(path.read_text(encoding = "utf-8"))
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer_json = (
        json.loads(tokenizer_path.read_text(encoding = "utf-8")) if tokenizer_path.is_file() else {}
    )
    eos = _token_ids(_vocab_id(tokenizer_json, tokenizer_config.get("eos_token")))
    eos += _token_ids(tokenizer_config.get("eos_token_id"))
    for config in configs:
        eos += _token_ids((config or {}).get("eos_token_id"))
    if not eos:
        raise RuntimeError("FastFlowLM needs the end-of-sequence token id, and none was found.")
    tokenizer_config["eos_token_id"] = list(dict.fromkeys(eos))
    if tokenizer_config.get("bos_token") is not None and not _token_ids(
        tokenizer_config.get("bos_token_id")
    ):
        bos = _token_ids(_vocab_id(tokenizer_json, tokenizer_config["bos_token"]))
        for config in configs:
            bos += _token_ids((config or {}).get("bos_token_id"))
        if not bos:
            raise RuntimeError("FastFlowLM needs the bos_token id, and none was found.")
        tokenizer_config["bos_token_id"] = bos[0]
    path.write_text(
        json.dumps(tokenizer_config, indent = 2, ensure_ascii = False) + "\n", encoding = "utf-8"
    )


# One conversion at a time: each holds a whole model in RAM, and retries must not interleave.
_CONVERT_LOCK = threading.Lock()


def _read_json(path: Optional[Path]) -> Optional[dict]:
    return json.loads(path.read_text(encoding = "utf-8")) if path is not None else None


def convert_existing_gguf(
    gguf_path: Path,
    base_model: str,
    save_directory: Path,
    token = None,
) -> Path:
    """Convert a GGUF already on disk; tokenizer files come from ``base_model``.

    ``base_model`` is the original (non-GGUF) Hub repo or a local model folder. Returns the
    folder to copy over a FastFlowLM catalog model of the same family and size.
    """
    out_dir = Path(save_directory) / f"{gguf_path.stem}-q4nx"
    found = {
        name: _fetch_base_file(base_model, name, token)
        for name in (*CONFIG_FILES, *TOKENIZER_FILES)
    }
    if found["tokenizer_config.json"] is None:
        raise RuntimeError(f"{base_model} has no tokenizer_config.json, which FastFlowLM needs.")
    with _CONVERT_LOCK, staged_output(out_dir) as staging:
        convert_gguf_to_q4nx(str(gguf_path), staging)
        for name in TOKENIZER_FILES:
            # The HF tokenizer.json replaces the one the converter rebuilds from the GGUF.
            if found[name] is not None:
                shutil.copyfile(found[name], staging / name)
        if found["chat_template.jinja"] is None:
            config = json.loads((staging / "tokenizer_config.json").read_text(encoding = "utf-8"))
            template = config.get("chat_template") or _gguf_chat_template(gguf_path)
            if template and not config.get("chat_template"):
                (staging / "chat_template.jinja").write_text(template, encoding = "utf-8")
        write_flm_tokenizer_config(staging, *(_read_json(found[name]) for name in CONFIG_FILES))
    return out_dir
