# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 Gemma 4 text encoder, in a spawned worker on the transformers 5.10 sidecar.

The server process runs Studio's default transformers, which has no ``Gemma4UnifiedForConditionalGeneration``. This
entry point runs in a fresh spawned interpreter: it prepends ``.venv_t5_510`` to ``sys.path`` BEFORE anything imports
transformers (the same activation the chat / training workers use), loads the encoder and tokenizer from the LTX-2.5
pack, and answers encode requests with the packed hidden states diffusers' ``LTX2Pipeline._get_gemma_prompt_embeds``
computes (every hidden state stacked on a last axis and flattened, left padding to ``max_sequence_length``).

Protocol (multiprocessing queues): the worker sends ``{"type": "ready", ...}`` once the model is loaded, then answers
``{"type": "encode", "id", "prompts", "max_sequence_length"}`` with ``{"type": "embeds", "id", "embeds", "mask"}``
(host tensors; bf16 ``[B, L, hidden * (layers + 1)]`` and int64 ``[B, L]``); ``{"type": "shutdown"}`` ends it. Any
failure is ``{"type": "error", "error": "..."}``. Must import nothing heavy at module level.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any

_BACKEND_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _version_tuple(text: str) -> tuple[int, ...]:
    out = []
    for part in str(text).split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def _activate_sidecar(venv_dir: str) -> None:
    # A transformers already imported here (a parent __main__ re-imported by spawn) shadows the sidecar; the version
    # check in _load then names the problem.
    if not venv_dir or not os.path.isdir(venv_dir):
        raise RuntimeError(f"the transformers 5.10 sidecar is missing at '{venv_dir}'")
    if venv_dir not in sys.path:
        sys.path.insert(0, venv_dir)
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = venv_dir + (os.pathsep + pythonpath if pythonpath else "")


def _single_file_folder(config: dict) -> str:
    """A scratch folder that reads like a transformers checkpoint for the single-file release's encoder: its embedded
    config and tokenizer written out, the weights linked in (renamed at load by ``key_mapping``)."""
    import json
    import tempfile

    from safetensors import safe_open

    repo, name = config["repo"], config["single_file"]
    if os.path.isdir(os.path.expanduser(repo)):
        path = os.path.join(os.path.expanduser(repo), name)
        if not os.path.isfile(path):
            raise RuntimeError(f"'{name}' is missing from '{repo}'")
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo,
            name,
            token = config.get("hf_token") or None,
            cache_dir = config.get("cache_dir") or None,
            local_files_only = bool(config.get("local_files_only")),
        )
    folder = tempfile.mkdtemp(prefix = "unsloth_ltx25_te_")
    with safe_open(path, framework = "pt", device = "cpu") as handle:
        meta = handle.metadata() or {}
        if "gemma_config" not in meta:
            raise RuntimeError(f"'{name}' has no embedded encoder config")
        with open(os.path.join(folder, "config.json"), "w", encoding = "utf-8") as fh:
            json.dump(json.loads(meta["gemma_config"]), fh)
        keys = set(handle.keys())
        for tensor_name, file_name in (
            ("tokenizer_json", "tokenizer.json"),
            ("hf_asset__tokenizer_config.json", "tokenizer_config.json"),
        ):
            if tensor_name not in keys:
                raise RuntimeError(f"'{name}' has no embedded {file_name}")
            with open(os.path.join(folder, file_name), "wb") as fh:
                fh.write(handle.get_tensor(tensor_name).numpy().tobytes())
    os.symlink(os.path.realpath(path), os.path.join(folder, "model.safetensors"))
    return folder


def _load(config: dict) -> tuple[Any, Any]:
    import torch
    import transformers

    minimum = config.get("min_transformers") or "5.10.1"
    if _version_tuple(transformers.__version__) < _version_tuple(minimum):
        raise RuntimeError(
            f"the sidecar provides transformers {transformers.__version__}, but LTX-2.5's Gemma 4 encoder needs "
            f">= {minimum}"
        )
    model_cls = getattr(transformers, "Gemma4UnifiedForConditionalGeneration", None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} in the sidecar has no Gemma4UnifiedForConditionalGeneration"
        )
    dtype = getattr(torch, str(config.get("dtype") or "bfloat16"))
    common = {
        "token": config.get("hf_token") or None,
        "cache_dir": config.get("cache_dir") or None,
        "local_files_only": bool(config.get("local_files_only")),
    }
    if config.get("single_file"):
        import shutil

        folder = _single_file_folder(config)
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(folder)
            model = model_cls.from_pretrained(
                folder, torch_dtype = dtype, key_mapping = config.get("key_mapping") or None
            )
        finally:
            shutil.rmtree(folder, ignore_errors = True)
    else:
        repo = config["repo"]
        tokenizer = transformers.AutoTokenizer.from_pretrained(repo, subfolder = "tokenizer", **common)
        model = model_cls.from_pretrained(repo, subfolder = "text_encoder", torch_dtype = dtype, **common)
    model.eval()
    model.requires_grad_(False)
    if config.get("resident"):
        model.to(config["device"])
    return tokenizer, model


def _encode(tokenizer: Any, model: Any, config: dict, prompts: list, max_len: int) -> tuple[Any, Any]:
    import torch

    device = config["device"]
    resident = bool(config.get("resident"))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        [str(p).strip() for p in prompts],
        padding = "max_length",
        max_length = int(max_len),
        truncation = True,
        add_special_tokens = True,
        return_tensors = "pt",
    )
    if not resident:
        model.to(device)
    try:
        with torch.inference_mode():
            ids = inputs.input_ids.to(device)
            mask = inputs.attention_mask.to(device)
            out = model(input_ids = ids, attention_mask = mask, output_hidden_states = True)
            stacked = torch.stack(out.hidden_states, dim = -1)
            embeds = stacked.flatten(2, 3).to(dtype = model.dtype)
            embeds_host = embeds.to("cpu")
            mask_host = mask.to("cpu")
            del out, stacked, embeds
    finally:
        if not resident:
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return embeds_host, mask_host


def run_text_encoder_worker(*, cmd_queue: Any, resp_queue: Any, config: dict) -> None:
    if _BACKEND_PATH not in sys.path:
        sys.path.insert(0, _BACKEND_PATH)
    try:
        _activate_sidecar(config.get("venv_dir") or "")
        tokenizer, model = _load(config)
        import transformers

        resp_queue.put(
            {
                "type": "ready",
                "transformers": transformers.__version__,
                "model_type": getattr(model.config, "model_type", None),
                "hidden_size": getattr(
                    getattr(model.config, "text_config", model.config), "hidden_size", None
                ),
                "resident": bool(config.get("resident")),
            }
        )
    except Exception as exc:  # noqa: BLE001 -- every failure goes back to the server as a message
        resp_queue.put(
            {
                "type": "error",
                "error": f"LTX-2.5 text encoder failed to load: {exc}",
                "stack": traceback.format_exc(limit = 20),
            }
        )
        return
    while True:
        msg = cmd_queue.get()
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "shutdown" or kind is None:
            return
        if kind != "encode":
            continue
        try:
            embeds, mask = _encode(
                tokenizer, model, config, msg["prompts"], msg.get("max_sequence_length") or 1024
            )
            resp_queue.put({"type": "embeds", "id": msg.get("id"), "embeds": embeds, "mask": mask})
        except Exception as exc:  # noqa: BLE001 -- report, keep serving
            resp_queue.put(
                {
                    "type": "error",
                    "id": msg.get("id"),
                    "error": f"LTX-2.5 text encoder failed: {exc}",
                    "stack": traceback.format_exc(limit = 20),
                }
            )
