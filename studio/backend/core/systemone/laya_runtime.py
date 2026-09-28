# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-process Laya runtime behind ``POST /v1/systemone``: one checkpoint resident at a time."""

from __future__ import annotations

import gc
import importlib.util
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .catalog import LOCAL_NAME, Checkpoint

logger = logging.getLogger(__name__)

LOAD_WAIT_S = 20.0
RUN_WAIT_S = 30.0
# Bounded: each waiter holds a shared threadpool worker, so an unbounded queue stalls every sync route.
MAX_PENDING = 8
FAILURE_BACKOFF_S = 60.0
_REQUIRED_DIRS = ("encoder", "tokenizer")
# laya 0.3.5 ships inside Studio (see vendor/README.md), so the Decision API never installs anything.
_VENDORED_LAYA = Path(__file__).resolve().parent.parent.parent / "vendor" / "laya"

_state_lock = threading.Lock()
_run_lock = threading.Lock()
_admission = threading.BoundedSemaphore(MAX_PENDING)
_agent = None
_loaded: Checkpoint | None = None
_device_name: str | None = None
_loader: threading.Thread | None = None
_loading: Checkpoint | None = None
_failure: tuple[Checkpoint, str, float] | None = None
_import_lock = threading.Lock()


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

    # CPU unless asked: a CUDA context opened in the backend process is never returned (see core.rag.embeddings._device).
    if preferred_device() != "gpu":
        return "cpu"
    from utils.hardware.hardware import DeviceType, get_device

    device = get_device()
    if device == DeviceType.MLX:
        if _mlx_available():
            return "mlx"
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    candidate = {DeviceType.CUDA: "cuda", DeviceType.XPU: "xpu"}.get(device, "cpu")
    if candidate == "cpu":
        return candidate
    from utils.torch_device_probe import device_can_allocate

    return candidate if device_can_allocate(candidate) else "cpu"


def _mlx_available() -> bool:
    try:
        import unsloth_zoo.mlx.decision  # noqa: F401
    except ImportError:
        return False
    return True


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


def _laya():
    """The vendored laya package, registered as top-level ``laya`` (its modules import each other relatively).

    Loaded by file path, not from ``sys.path``: a laya installed in the venv (Studio pinned one before
    vendoring it) must not replace this copy, since this module drives laya internals.
    """
    if (module := sys.modules.get("laya")) is not None:
        return module
    with _import_lock:
        if (module := sys.modules.get("laya")) is not None:
            return module
        init = _VENDORED_LAYA / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "laya", init, submodule_search_locations = [str(_VENDORED_LAYA)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["laya"] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for name in [n for n in sys.modules if n == "laya" or n.startswith("laya.")]:
                del sys.modules[name]
            raise
        return module


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


def _release_memory() -> None:
    gc.collect()
    # MLX keeps freed buffers in its allocator cache until told otherwise; mlx below 0.24.1 names it mx.metal.clear_cache.
    if (mx := sys.modules.get("mlx.core")) is not None:
        clear = getattr(mx, "clear_cache", None) or getattr(
            getattr(mx, "metal", None), "clear_cache", None
        )
        if clear is not None:
            clear()
    # Torch's allocator too; only already-initialised backends, so a CPU-only Studio never opens a device context.
    if (torch := sys.modules.get("torch")) is not None:
        try:
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_initialized():
                torch.xpu.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            logger.debug("Could not clear the Decision API device cache", exc_info = True)


def _evict() -> None:
    global _agent, _loaded, _device_name
    with _run_lock:
        _agent = _loaded = _device_name = None
    _release_memory()


class _MLXAgent:
    """The part of ``laya.Agent`` this module drives, over the unsloth-zoo MLX decision model."""

    device = "mlx"

    def __init__(self, folder: Path):
        import json

        from transformers import AutoTokenizer
        from unsloth_zoo.mlx.decision import load_decision_model

        laya = _laya()
        clamp_temperature = laya.common.clamp_temperature
        self.folder = folder
        self.cfg = json.loads((folder / "rl_agent_config.json").read_text(encoding = "utf-8"))
        self.tok = AutoTokenizer.from_pretrained(str(folder / "tokenizer"))
        self.temperature = [
            clamp_temperature(t) for t in self.cfg.get("temperature", [1.0, 1.0, 1.0])
        ]
        self.temperature_by_options = {
            k: clamp_temperature(v) for k, v in self.cfg.get("temperature_by_options", {}).items()
        }
        self._to_internal = laya.agent.Agent._to_internal
        self.model = load_decision_model(folder)


def _load_checkpoint(checkpoint: Checkpoint):
    root = _checkpoint_dir(checkpoint)
    laya = _laya()

    # Evict only once the new checkpoint is on disk, so a long or failed download leaves the resident model serving.
    _evict()
    device = _device()
    if device == "mlx":
        folder = root / checkpoint.subfolder if checkpoint.subfolder else root
        return _MLXAgent(folder), device
    return laya.load(str(root), subfolder = checkpoint.subfolder, device = device), device


def _hub_download_active(checkpoint: Checkpoint) -> bool:
    # A Hub download job writing the same repo would share blobs with our snapshot_download.
    if checkpoint.is_local:
        return False
    try:
        from hub.utils.download_registry import get_models_registry
        if not get_models_registry().active_job_refs(checkpoint.source):
            return False
    except Exception:
        return False
    return not is_cached(checkpoint)


def loading_repo_ids() -> tuple[str, ...]:
    with _state_lock:
        if _loading is not None and not _loading.is_local:
            return (_loading.source,)
    return ()


def _load(checkpoint: Checkpoint) -> None:
    global _agent, _loaded, _device_name, _loading, _failure
    started = time.monotonic()
    try:
        agent, device = _load_checkpoint(checkpoint)
    except Exception as exc:
        message = f"Could not load {checkpoint.name}: {type(exc).__name__}: {exc}"
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
        if _loading is not None:
            if _loading != checkpoint:
                raise Unavailable(
                    503, "model_loading", f"{_loading.name} is loading", retry_after = 5
                )
            if _loader is not None and _loader.is_alive():
                return _loader
            raise Unavailable(503, "model_loading", f"{checkpoint.name} is loading", retry_after = 5)
        # Claimed before the registry probe, so a Hub download admitted from now on sees this load.
        _loading = checkpoint
    # Outside _state_lock: the registry calls loading_repo_ids() while holding its own lock.
    if _hub_download_active(checkpoint):
        with _state_lock:
            _loading = None
        raise Unavailable(503, "model_loading", f"{checkpoint.name} is downloading", retry_after = 5)
    with _state_lock:
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


# Long states tokenized in growing prefixes; margin keeps kept tokens clear of the cut (prefix tokenization differs).
_PREFIX_MARGIN = 64
_HEAD_CACHE_SIZE = 1024


def _state_ids(tok, state, room: int) -> tuple[list[int], bool]:
    serialize_state = _laya().common.serialize_state

    text = serialize_state(state).replace(tok.mask_token, " ")
    chars = max(4096, room * 16)
    while chars < len(text):
        ids = tok(text[:chars], add_special_tokens = False)["input_ids"]
        if len(ids) > room + _PREFIX_MARGIN:
            return ids[:room], True
        chars *= 4
    ids = tok(text, add_special_tokens = False)["input_ids"]
    return ids[:room], len(ids) > room


def _head(agent, question: dict[str, Any], max_len: int, head_max_len: int):
    import json

    common = _laya().common
    build_sequence = common.build_sequence
    render_options = common.render_options

    cache = agent.__dict__.setdefault("_unsloth_heads", {})
    key = json.dumps(question, ensure_ascii = False)
    if key not in cache:
        internal = agent._to_internal(question)
        ids, markers = build_sequence(agent.tok, "", internal, max_len, head_max_len)
        if len(markers) != len(render_options(internal)):
            raise ValueError("question options exceed head_max_len=%d" % head_max_len)
        if len(cache) >= _HEAD_CACHE_SIZE:
            cache.clear()
        cache[key] = (ids, markers, internal)
    return cache[key]


def _predict(agent, state, questions: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    import numpy as np

    common = _laya().common
    QTYPES = common.QTYPES
    confidence_from_probs = common.confidence_from_probs

    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    names = list(questions)
    heads = []
    for name in names:
        try:
            heads.append(_head(agent, questions[name], max_len, head_max_len))
        except ValueError:
            raise ValueError(
                f"Question {name!r} options exceed the Laya context window ({max_len} tokens). "
                "Use fewer or shorter criteria."
            ) from None
    room = max(0, max_len - min(len(ids) for ids, _, _ in heads))
    state_ids, state_cut = _state_ids(agent.tok, state, room)
    items, truncated = [], False
    for ids, markers, internal in heads:
        keep = max(0, max_len - len(ids))
        truncated = truncated or state_cut or len(state_ids) > keep
        items.append(
            {
                "ids": (ids[:-1] + state_ids[:keep] + ids[-1:])[:max_len],
                "markers": markers,
                "qtype": QTYPES[internal["t"]],
            }
        )
    logits, usage = _forward(agent, items)
    answers = {}
    for row, (name, (_, markers, internal)) in enumerate(zip(names, heads)):
        k = len(markers)
        p = _probabilities(agent, logits[row], k, QTYPES[internal["t"]])
        confidence = round(confidence_from_probs(p, k), 4)
        if internal["t"] == "choice":
            keys = list(internal["crit"])
            answers[name] = {
                "type": "choice",
                "choice": keys[int(p.argmax())],
                "probabilities": {key: round(float(v), 4) for key, v in zip(keys, p)},
                "confidence": confidence,
            }
        elif internal["t"] == "score":
            answers[name] = {
                "type": "score",
                "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(internal["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": confidence,
            }
        else:
            answers[name] = {
                "type": "noul",
                "noul": round(float(p[1]), 4),
                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
            }
    return {"answers": answers, "usage": {"input_tokens": usage, "output_tokens": 0}}, truncated


def _forward(agent, items: list[dict[str, Any]]):
    global _agent, _loaded, _device_name
    import torch

    collate_items = _laya().common.collate_items

    batch = collate_items([items], agent.tok.pad_token_id)
    if agent.device == "mlx":
        try:
            return agent.model.logits(batch), int(batch["attention_mask"].sum())
        except RuntimeError as exc:
            # MLX reports exhausted memory as "[malloc] Unable to allocate ..." or "Insufficient Memory".
            reason = str(exc).lower()
            if "memory" not in reason and "allocate" not in reason:
                raise
        # Past the handler, so the traceback no longer keeps the MLX arrays alive while the CPU copy loads.
        laya = _laya()

        logger.warning("Laya ran out of GPU memory; moving it to CPU")
        agent.model = None
        _release_memory()
        try:
            cpu = laya.load(str(agent.folder), device = "cpu")
        except Exception:
            # Callers hold _run_lock, so drop the half-moved agent here; the next request loads it again.
            _agent = _loaded = _device_name = None
            raise
        agent.model, agent.device, agent.dtype = cpu.model, cpu.device, cpu.dtype
        _device_name = "cpu"
    try:
        logits = _run_model(agent, batch)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        # Same fallback as laya's Agent.predict: a GPU that runs out of memory moves the model to CPU.
        reason = str(exc).lower()
        if agent.device.type == "cpu" or ("memory" not in reason and "cuda" not in reason):
            raise
        logger.warning("Laya ran out of GPU memory; moving it to CPU")
        agent.device, agent.dtype = torch.device("cpu"), torch.float32
        agent.model.to(agent.device)
        _device_name = "cpu"
        _release_memory()
        logits = _run_model(agent, batch)
    return logits.float().cpu().numpy(), int(batch["attention_mask"].sum())


def _run_model(agent, batch):
    import torch

    device = agent.device
    with (
        torch.inference_mode(),
        torch.autocast(device_type = device.type, dtype = agent.dtype, enabled = device.type == "cuda"),
    ):
        logits, _ = agent.model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        )
    return logits


def _probabilities(agent, row, k: int, qtype: int):
    import numpy as np

    temp_bucket = _laya().common.temp_bucket

    scale = agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype])
    z = row[:k] / scale
    p = np.exp(z - z.max())
    return p / p.sum()


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
            result, truncated = _predict(agent, state, laya_questions)
        except ValueError as exc:
            raise Unavailable(422, "invalid_request_error", str(exc)) from None
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
        failure = _failure if _failure and time.monotonic() < _failure[2] else None
        return {
            "loaded_model": _loaded.name if _loaded else None,
            "device": _device_name,
            "loading_model": _loading.name if _loading else None,
            # Kept for the settings API: laya is vendored, so there is never an install in flight.
            "installing": False,
            "error": failure[1] if failure else None,
            "error_model": failure[0].name if failure else None,
        }


def unload() -> bool:
    global _agent, _loaded, _device_name, _failure
    with _state_lock:
        # _loading: a load claimed but not yet started would otherwise land after this unload.
        if _loading is not None or (_loader is not None and _loader.is_alive()):
            raise Unavailable(409, "model_loading", "Wait for the load to finish before unloading")
    with _run_lock:
        was_loaded = _agent is not None
        _agent = _loaded = _device_name = None
        _failure = None
    _release_memory()
    return was_loaded
