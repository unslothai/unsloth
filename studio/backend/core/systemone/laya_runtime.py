# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-process Laya runtime behind ``POST /v1/systemone``: one checkpoint resident at a time."""

from __future__ import annotations

import gc
import logging
import threading
import time
from pathlib import Path
from typing import Any

from .catalog import LOCAL_NAME, Checkpoint

logger = logging.getLogger(__name__)

LOAD_WAIT_S = 20.0
RUN_WAIT_S = 30.0
# Requests past this answer 529 at once: each waiter holds a worker from the threadpool every sync
# route in Studio shares, so an unbounded queue behind a first download would stall the whole backend.
MAX_PENDING = 8
FAILURE_BACKOFF_S = 60.0
_REQUIRED_DIRS = ("encoder", "tokenizer")

_state_lock = threading.Lock()
_run_lock = threading.Lock()
_admission = threading.BoundedSemaphore(MAX_PENDING)
_agent = None
_loaded: Checkpoint | None = None
_device_name: str | None = None
_loader: threading.Thread | None = None
_loading: Checkpoint | None = None
_failure: tuple[Checkpoint, str, float] | None = None


class Unavailable(Exception):
    def __init__(
        self,
        status: int,
        error_type: str,
        message: str,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = message
        self.retry_after = retry_after


def _device() -> str:
    from utils.systemone_settings import get_device as preferred_device

    # CPU unless asked, for the reason core.rag.embeddings._device gives: a CUDA context opened in the
    # backend process is never returned while it lives.
    if preferred_device() != "gpu":
        return "cpu"
    from utils.hardware.hardware import DeviceType, get_device

    device = get_device()
    if device == DeviceType.MLX:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    candidate = {DeviceType.CUDA: "cuda", DeviceType.XPU: "xpu"}.get(device, "cpu")
    if candidate == "cpu":
        return candidate
    from utils.torch_device_probe import device_can_allocate

    return candidate if device_can_allocate(candidate) else "cpu"


_WEIGHT_FILES = ("rl_agent_config.json", "model.safetensors")


def _wanted(path: str, subfolder: str | None) -> bool:
    prefix = f"{subfolder}/" if subfolder else ""
    if not path.startswith(prefix):
        return False
    rest = path[len(prefix) :]
    return rest in _WEIGHT_FILES or rest.startswith(tuple(f"{name}/" for name in _REQUIRED_DIRS))


def _checkpoint_dir(checkpoint: Checkpoint, *, local_only: bool = False) -> Path:
    if checkpoint.name == LOCAL_NAME and not checkpoint.is_local:
        raise FileNotFoundError(
            f"UNSLOTH_SYSTEMONE_MODEL={checkpoint.source} is neither a known model nor a directory"
        )
    if checkpoint.is_local:
        root = Path(checkpoint.source).expanduser()
    else:
        from huggingface_hub import snapshot_download

        from utils.hf_cache_settings import active_hf_hub_cache
        from utils.utils import hf_env_offline

        prefix = f"{checkpoint.subfolder}/" if checkpoint.subfolder else ""
        root = Path(
            snapshot_download(
                checkpoint.source,
                cache_dir = active_hf_hub_cache(),
                local_files_only = local_only or hf_env_offline(),
                allow_patterns = [prefix + name for name in _WEIGHT_FILES]
                + [f"{prefix}{name}/*" for name in _REQUIRED_DIRS],
            )
        )
    folder = root / checkpoint.subfolder if checkpoint.subfolder else root
    # Without these, laya falls back to downloading the base encoder from the Hub on every load.
    missing = [name for name in _REQUIRED_DIRS if not (folder / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Laya checkpoint at {folder} is missing {', '.join(n + '/' for n in missing)}"
        )
    return root


def is_cached(checkpoint: Checkpoint) -> bool:
    try:
        root = _checkpoint_dir(checkpoint, local_only = True)
    except Exception:
        return False
    folder = root / checkpoint.subfolder if checkpoint.subfolder else root
    return all((folder / name).is_file() for name in _WEIGHT_FILES)


def download_plan(checkpoint: Checkpoint) -> dict[str, Any]:
    cached = is_cached(checkpoint)
    plan = {"repo": None, "files": [], "size_bytes": 0, "cached": cached, "error": None}
    if checkpoint.name == LOCAL_NAME:
        if not cached:
            plan["error"] = f"No complete Laya checkpoint at {checkpoint.source}"
        return plan
    plan["repo"] = checkpoint.source
    plan["size_bytes"] = checkpoint.download_bytes
    if cached:
        return plan
    try:
        from huggingface_hub import HfApi
        entries = HfApi().list_repo_tree(checkpoint.source, recursive = True)
        files = [
            (entry.path, getattr(entry, "size", 0) or 0)
            for entry in entries
            if hasattr(entry, "size") and _wanted(entry.path, checkpoint.subfolder)
        ]
    except Exception as exc:
        plan["error"] = f"Could not list {checkpoint.source}: {type(exc).__name__}"
        return plan
    plan["files"] = sorted(path for path, _ in files)
    plan["size_bytes"] = sum(size for _, size in files) or checkpoint.download_bytes
    return plan


def _load_checkpoint(checkpoint: Checkpoint):
    root = _checkpoint_dir(checkpoint)
    import laya

    device = _device()
    return laya.load(str(root), subfolder = checkpoint.subfolder, device = device), device


def _load(checkpoint: Checkpoint) -> None:
    global _agent, _loaded, _device_name, _loading, _failure
    started = time.monotonic()
    try:
        with _run_lock:
            _agent = _loaded = _device_name = None
            gc.collect()
        agent, device = _load_checkpoint(checkpoint)
    except Exception as exc:
        message = (
            "System One needs the laya package; run `unsloth studio update`"
            if isinstance(exc, ImportError)
            else f"Could not load {checkpoint.name}: {type(exc).__name__}: {exc}"
        )
        logger.warning("System One load failed: %s", message)
        with _state_lock:
            _failure = (checkpoint, message, time.monotonic() + FAILURE_BACKOFF_S)
            _loading = None
        return
    with _state_lock:
        _agent, _loaded, _device_name = agent, checkpoint, str(getattr(agent, "device", device))
        _failure = _loading = None
    logger.info(
        "System One loaded %s on %s in %.1fs",
        checkpoint.name,
        _device_name,
        time.monotonic() - started,
    )


def _ensure_loading(checkpoint: Checkpoint) -> threading.Thread | None:
    global _loader, _loading
    with _state_lock:
        if _loaded == checkpoint and _agent is not None:
            return None
        if _failure and _failure[0] == checkpoint and time.monotonic() < _failure[2]:
            raise Unavailable(
                503,
                "model_unavailable",
                _failure[1],
                retry_after = max(1.0, _failure[2] - time.monotonic()),
            )
        if _loader is not None and _loader.is_alive():
            if _loading != checkpoint:
                raise Unavailable(
                    503, "model_loading", f"{_loading.name} is loading", retry_after = 5
                )
            return _loader
        _loading = checkpoint
        _loader = threading.Thread(
            target = _load, args = (checkpoint,), name = "systemone-load", daemon = True
        )
        _loader.start()
        return _loader


def _agent_for(checkpoint: Checkpoint, wait: float):
    loader = _ensure_loading(checkpoint)
    if loader is not None:
        loader.join(wait)
        if loader.is_alive() or _ensure_loading(checkpoint) is not None:
            raise Unavailable(
                503, "model_loading", f"{checkpoint.name} is still loading", retry_after = 5
            )
    return _agent


def _to_laya(question: dict[str, Any]) -> dict[str, Any]:
    out = {"type": question["type"], "instructions": question.get("instructions") or ""}
    if question.get("criteria") is not None:
        out["criteria"] = question["criteria"]
    return out


def _state_truncated(agent, state, questions: dict[str, dict[str, Any]]) -> bool:
    from laya.common import build_sequence, serialize_state

    tok = agent.tok
    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    state_len = len(
        tok(serialize_state(state).replace(tok.mask_token, " "), add_special_tokens = False)[
            "input_ids"
        ]
    )
    for question in questions.values():
        head, _ = build_sequence(tok, "", agent._to_internal(question), max_len, head_max_len)
        if len(head) + state_len > max_len:
            return True
    return False


def _wire_answer(answer: dict[str, Any]) -> dict[str, Any]:
    kind = answer["type"]
    if kind == "noul":
        return {"type": "noul", "noul": float(answer["noul"])}
    if kind == "choice":
        return {
            "type": "choice",
            "choice": answer["choice"],
            "confidence": float(answer["confidence"]),
            "probabilities": {k: float(v) for k, v in answer["probabilities"].items()},
        }
    return {
        "type": "score",
        "score": float(answer["score"]),
        "confidence": float(answer["confidence"]),
        "legend": answer["legend"],
        "probabilities": {k: float(v) for k, v in answer["probabilities"].items()},
    }


def decide(checkpoint: Checkpoint, state, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not _admission.acquire(blocking = False):
        raise Unavailable(529, "overloaded", "System One is busy; retry shortly", retry_after = 1)
    try:
        return _decide(checkpoint, state, questions)
    finally:
        _admission.release()


def _decide(checkpoint: Checkpoint, state, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    agent = _agent_for(checkpoint, LOAD_WAIT_S)
    if not _run_lock.acquire(timeout = RUN_WAIT_S):
        raise Unavailable(529, "overloaded", "System One is busy; retry shortly", retry_after = 1)
    try:
        if agent is None or _agent is not agent or _loaded != checkpoint:
            raise Unavailable(
                503, "model_loading", f"{checkpoint.name} is reloading", retry_after = 5
            )
        laya_questions = {name: _to_laya(q) for name, q in questions.items()}
        try:
            result = agent.predict(state, laya_questions)
        except ValueError as exc:
            raise Unavailable(400, "invalid_request_error", str(exc)) from None
        truncated = _state_truncated(agent, state, laya_questions)
    finally:
        _run_lock.release()
    return {
        "model": checkpoint.name,
        "answers": {name: _wire_answer(result["answers"][name]) for name in questions},
        "usage": {"input_tokens": int(result["usage"]["input_tokens"]), "output_tokens": 0},
        "truncated": truncated,
    }


def status() -> dict[str, Any]:
    with _state_lock:
        return {
            "loaded_model": _loaded.name if _loaded else None,
            "device": _device_name,
            "loading_model": _loading.name if _loading else None,
            "error": _failure[1] if _failure else None,
        }


def unload() -> bool:
    global _agent, _loaded, _device_name, _failure
    with _state_lock:
        if _loader is not None and _loader.is_alive():
            raise Unavailable(409, "model_loading", "Wait for the load to finish before unloading")
    with _run_lock:
        was_loaded = _agent is not None
        _agent = _loaded = _device_name = None
        _failure = None
    gc.collect()
    return was_loaded
