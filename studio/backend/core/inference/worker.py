# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference subprocess entry point.

Each inference session runs in a persistent subprocess (mp.get_context("spawn")).
This gives us a clean Python interpreter with no stale module state —
solving the transformers version-switching problem completely.

The subprocess stays alive while a model is loaded, accepting commands
(generate, load, unload) via mp.Queue. It exits on shutdown or unload.

Pattern follows core/training/worker.py.
"""

from __future__ import annotations

import base64
import structlog
from loggers import get_logger
import os
import queue as _queue
import sys
import threading
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any

logger = get_logger(__name__)
from utils.hardware import apply_gpu_ids


def _activate_transformers_version(model_name: str) -> None:
    """Activate the correct transformers version BEFORE any ML imports.

    Uses get_transformers_tier() to decide between .venv_t5_550/ (5.5.0),
    .venv_t5_530/ (5.3.0), or the default 4.57.x.
    """
    # Ensure backend is on path for utils imports
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.transformers_version import (
        get_transformers_tier,
        _resolve_base_model,
        _ensure_venv_t5_530_exists,
        _ensure_venv_t5_550_exists,
        _VENV_T5_530_DIR,
        _VENV_T5_550_DIR,
    )

    resolved = _resolve_base_model(model_name)
    tier = get_transformers_tier(resolved)

    if tier == "550":
        if not _ensure_venv_t5_550_exists():
            raise RuntimeError(
                f"Cannot activate transformers 5.5.0: .venv_t5_550 missing at {_VENV_T5_550_DIR}"
            )
        if _VENV_T5_550_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_550_DIR)
        logger.info("Activated transformers 5.5.0 from %s", _VENV_T5_550_DIR)
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_550_DIR + (os.pathsep + _pp if _pp else "")
    elif tier == "530":
        if not _ensure_venv_t5_530_exists():
            raise RuntimeError(
                f"Cannot activate transformers 5.3.0: .venv_t5_530 missing at {_VENV_T5_530_DIR}"
            )
        if _VENV_T5_530_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_530_DIR)
        logger.info("Activated transformers 5.3.0 from %s", _VENV_T5_530_DIR)
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_530_DIR + (os.pathsep + _pp if _pp else "")
    else:
        logger.info("Using default transformers (4.57.x) for %s", model_name)


def _decode_image(image_base64: str):
    """Decode base64 string to PIL.Image."""
    from PIL import Image

    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


def _resize_image(img, max_size: int = 800):
    """Resize image while maintaining aspect ratio."""
    if img is None:
        return None
    if img.size[0] > max_size or img.size[1] > max_size:
        from PIL import Image

        ratio = min(max_size / img.size[0], max_size / img.size[1])
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def _send_response(resp_queue: Any, response: dict) -> None:
    """Send a response to the parent process."""
    try:
        resp_queue.put(response)
    except (OSError, ValueError) as exc:
        logger.error("Failed to send response: %s", exc)


def _build_model_config(config: dict):
    """Build a ModelConfig from the config dict."""
    from utils.models import ModelConfig

    model_name = config["model_name"]
    hf_token = config.get("hf_token")
    hf_token = hf_token if hf_token and hf_token.strip() else None
    gguf_variant = config.get("gguf_variant")

    mc = ModelConfig.from_identifier(
        model_id = model_name,
        hf_token = hf_token,
        gguf_variant = gguf_variant,
    )
    if not mc:
        raise ValueError(f"Invalid model identifier: {model_name}")
    return mc


def _get_hf_download_state(
    model_names: list[str] | None = None,
) -> tuple[int, bool] | None:
    """Return (total_bytes, has_incomplete) for the HF Hub cache, or None on error.

    When *model_names* is provided, only those models' ``blobs/``
    directories are checked instead of scanning every cached model --
    much faster on systems with many models. Accepts multiple names so
    that LoRA loads can watch both the adapter repo and the base model
    repo simultaneously.

    *has_incomplete* is True when any ``*.incomplete`` files exist in the
    watched blobs directories, indicating that ``huggingface_hub`` is
    actively downloading.

    Returns None if the state cannot be determined (import error,
    permission error, etc.) so callers can skip stall logic.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache = Path(HF_HUB_CACHE)
        if not cache.exists():
            return (0, False)

        total = 0
        has_incomplete = False
        blobs_dirs: list[Path] = []

        if model_names:
            from utils.paths import resolve_cached_repo_id_case

            for name in model_names:
                if not name:
                    continue
                # Skip local filesystem paths -- HF model IDs use forward
                # slashes (org/model) but never start with / . ~ or contain
                # backslashes. This distinguishes them from absolute paths,
                # relative paths, and Windows paths.
                if name.startswith(("/", ".", "~")) or "\\" in name:
                    continue
                name = resolve_cached_repo_id_case(name)
                # HF cache dir format: models--org--name (slashes -> --)
                cache_dir_name = "models--" + name.replace("/", "--")
                blobs_dir = cache / cache_dir_name / "blobs"
                if blobs_dir.exists():
                    blobs_dirs.append(blobs_dir)
        else:
            blobs_dirs = list(cache.glob("models--*/blobs"))

        for bdir in blobs_dirs:
            for f in bdir.iterdir():
                try:
                    if f.is_file():
                        total += f.stat().st_size
                        if f.name.endswith(".incomplete"):
                            has_incomplete = True
                except OSError:
                    pass

        return (total, has_incomplete)
    except Exception as e:
        logger.debug("Failed to determine HF download state: %s", e)
        return None


def _start_heartbeat(
    resp_queue: Any,
    interval: float = 30.0,
    stall_timeout: float = 180.0,
    xet_disabled: bool = False,
    model_names: list[str] | None = None,
) -> threading.Event:
    """Start a daemon thread that sends periodic status heartbeats.

    Monitors the HF Hub cache directory for download activity. A stall
    is only reported when ``*.incomplete`` files are present (indicating
    ``huggingface_hub`` is actively downloading) **and** the total cache
    size has not changed for *stall_timeout* seconds.

    Once the download finishes (no more ``.incomplete`` files), the stall
    timer resets, so post-download initialization (quantization, GPU
    weight loading) is never misclassified as a stalled download.

    Returns a stop event -- set it to terminate the heartbeat thread.
    """
    stop = threading.Event()
    transport = "https" if xet_disabled else "xet"

    def _beat():
        state = _get_hf_download_state(model_names)
        last_size = state[0] if state is not None else 0
        last_change = time.monotonic()

        while not stop.wait(interval):
            state = _get_hf_download_state(model_names)
            now = time.monotonic()

            # Skip stall logic if we cannot measure the cache
            if state is None:
                _send_response(
                    resp_queue,
                    {
                        "type": "status",
                        "message": f"Loading model ({transport} transport)...",
                        "ts": time.time(),
                    },
                )
                continue

            current_size, has_incomplete = state

            if current_size != last_size:
                last_size = current_size
                last_change = now

            # Only fire stall when .incomplete files are present,
            # confirming a download is actively in progress.
            # Once downloads finish (no .incomplete), reset the timer
            # so model init time is not counted as a stall.
            if not has_incomplete:
                last_change = now
            elif now - last_change >= stall_timeout:
                _send_response(
                    resp_queue,
                    {
                        "type": "stall",
                        "message": (
                            f"Download appears stalled ({transport} transport) "
                            f"-- no progress for {int(now - last_change)}s"
                        ),
                        "ts": time.time(),
                    },
                )
                # Only fire once -- the orchestrator will kill us
                return

            _send_response(
                resp_queue,
                {
                    "type": "status",
                    "message": f"Loading model ({transport} transport)...",
                    "ts": time.time(),
                },
            )

    t = threading.Thread(target = _beat, daemon = True)
    t.start()
    return stop


def _handle_load(backend, config: dict, resp_queue: Any) -> None:
    """Handle a load command: load a model into the backend."""
    try:
        mc = _build_model_config(config)

        hf_token = config.get("hf_token")
        hf_token = hf_token if hf_token and hf_token.strip() else None

        # Auto-detect quantization for LoRA adapters
        load_in_4bit = config.get("load_in_4bit", True)
        if mc.is_lora and mc.path:
            import json
            from pathlib import Path

            adapter_cfg_path = Path(mc.path) / "adapter_config.json"
            if adapter_cfg_path.exists():
                try:
                    with open(adapter_cfg_path) as f:
                        adapter_cfg = json.load(f)
                    training_method = adapter_cfg.get("unsloth_training_method")
                    if training_method == "lora" and load_in_4bit:
                        logger.info(
                            "adapter_config.json says lora — setting load_in_4bit=False"
                        )
                        load_in_4bit = False
                    elif training_method == "qlora" and not load_in_4bit:
                        logger.info(
                            "adapter_config.json says qlora — setting load_in_4bit=True"
                        )
                        load_in_4bit = True
                    elif not training_method:
                        if (
                            mc.base_model
                            and "-bnb-4bit" not in mc.base_model.lower()
                            and load_in_4bit
                        ):
                            logger.info(
                                "No training method, base model has no -bnb-4bit — setting load_in_4bit=False"
                            )
                            load_in_4bit = False
                except Exception as e:
                    logger.warning("Could not read adapter_config.json: %s", e)

        # Auto-enable trust_remote_code for Nemotron models only.
        # NemotronH has config parsing bugs requiring trust_remote_code=True.
        # Other transformers 5.x models are native and do NOT need it.
        trust_remote_code = config.get("trust_remote_code", False)
        if not trust_remote_code:
            model_name = config["model_name"]
            _mn_lower = model_name.lower()
            if (
                "nemotron" in _mn_lower
                and (_mn_lower.startswith("unsloth/") or _mn_lower.startswith("nvidia/"))
            ):
                trust_remote_code = True
                logger.info(
                    "Auto-enabled trust_remote_code for Nemotron model: %s",
                    model_name,
                )

        # Send heartbeats every 30s so the orchestrator knows we're still alive
        # (download / weight loading can take a long time on slow connections)
        xet_disabled = os.environ.get("HF_HUB_DISABLE_XET") == "1"

        # Watch both the model repo and base model repo (for LoRA loads
        # where the base model download is the actual bottleneck)
        watch_repos = [mc.identifier]
        base = getattr(mc, "base_model", None)
        if base and str(base) != mc.identifier:
            watch_repos.append(str(base))

        heartbeat_stop = _start_heartbeat(
            resp_queue,
            interval = 30.0,
            xet_disabled = xet_disabled,
            model_names = watch_repos,
        )
        try:
            success = backend.load_model(
                config = mc,
                max_seq_length = config.get("max_seq_length", 2048),
                load_in_4bit = load_in_4bit,
                hf_token = hf_token,
                trust_remote_code = trust_remote_code,
                gpu_ids = config.get("resolved_gpu_ids"),
            )
        finally:
            heartbeat_stop.set()

        if success:
            # Build model_info for the parent to mirror
            model_info = {
                "identifier": mc.identifier,
                "display_name": mc.display_name,
                "is_vision": mc.is_vision,
                "is_lora": mc.is_lora,
                "is_gguf": False,
                "is_audio": getattr(mc, "is_audio", False),
                "audio_type": getattr(mc, "audio_type", None),
                "has_audio_input": getattr(mc, "has_audio_input", False),
            }
            _send_response(
                resp_queue,
                {
                    "type": "loaded",
                    "success": True,
                    "model_info": model_info,
                    "ts": time.time(),
                },
            )
        else:
            _send_response(
                resp_queue,
                {
                    "type": "loaded",
                    "success": False,
                    "error": "Failed to load model",
                    "ts": time.time(),
                },
            )

    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": "loaded",
                "success": False,
                "error": str(exc),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            },
        )


def _handle_generate(
    backend,
    cmd: dict,
    resp_queue: Any,
    cancel_event,
) -> None:
    """Handle a generate command: stream tokens back via resp_queue.

    cancel_event is an mp.Event shared with the parent process.
    The parent can set it at any time (e.g. user stops generation,
    or user loads a new model while generating) and generation
    stops within 1-2 tokens.
    """
    request_id = cmd.get("request_id", "")

    try:
        # Decode image if provided
        image = None
        image_b64 = cmd.get("image_base64")
        if image_b64:
            image = _decode_image(image_b64)
            image = _resize_image(image)

        # Build generation kwargs
        gen_kwargs = {
            "messages": cmd["messages"],
            "system_prompt": cmd.get("system_prompt", ""),
            "image": image,
            "temperature": cmd.get("temperature", 0.7),
            "top_p": cmd.get("top_p", 0.9),
            "top_k": cmd.get("top_k", 40),
            "min_p": cmd.get("min_p", 0.0),
            "max_new_tokens": cmd.get("max_new_tokens", 256),
            "repetition_penalty": cmd.get("repetition_penalty", 1.0),
            "cancel_event": cancel_event,
        }

        # Choose generation path
        use_adapter = cmd.get("use_adapter")
        if use_adapter is not None:
            generator = backend.generate_with_adapter_control(
                use_adapter = use_adapter,
                **gen_kwargs,
            )
        else:
            generator = backend.generate_chat_response(**gen_kwargs)

        logger.info("Starting text generation for request_id=%s", request_id)

        for cumulative_text in generator:
            # cancel_event is an mp.Event — checked instantly, no queue polling
            if cancel_event.is_set():
                logger.info("Generation cancelled for request %s", request_id)
                break

            _send_response(
                resp_queue,
                {
                    "type": "token",
                    "request_id": request_id,
                    "text": cumulative_text,
                    "ts": time.time(),
                },
            )

        _send_response(
            resp_queue,
            {
                "type": "gen_done",
                "request_id": request_id,
                "ts": time.time(),
            },
        )
        logger.info("Finished text generation for request_id=%s", request_id)

    except Exception as exc:
        logger.error("Generation error: %s", exc, exc_info = True)
        _send_response(
            resp_queue,
            {
                "type": "gen_error",
                "request_id": request_id,
                "error": str(exc),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            },
        )


def _handle_generate_audio(
    backend,
    cmd: dict,
    resp_queue: Any,
) -> None:
    """Handle TTS audio generation — returns WAV bytes + sample_rate."""
    request_id = cmd.get("request_id", "")
    try:
        logger.info("Starting audio generation for request_id=%s", request_id)
        wav_bytes, sample_rate = backend.generate_audio_response(
            text = cmd["text"],
            temperature = cmd.get("temperature", 0.6),
            top_p = cmd.get("top_p", 0.95),
            top_k = cmd.get("top_k", 50),
            min_p = cmd.get("min_p", 0.0),
            max_new_tokens = cmd.get("max_new_tokens", 2048),
            repetition_penalty = cmd.get("repetition_penalty", 1.0),
            use_adapter = cmd.get("use_adapter"),
        )

        # Send WAV bytes as base64 (bytes can't go through mp.Queue directly)
        _send_response(
            resp_queue,
            {
                "type": "audio_done",
                "request_id": request_id,
                "wav_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "sample_rate": sample_rate,
                "ts": time.time(),
            },
        )
        logger.info("Finished audio generation for request_id=%s", request_id)

    except Exception as exc:
        logger.error("Audio generation error: %s", exc, exc_info = True)
        _send_response(
            resp_queue,
            {
                "type": "audio_error",
                "request_id": request_id,
                "error": str(exc),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            },
        )


def _handle_generate_audio_input(
    backend,
    cmd: dict,
    resp_queue: Any,
    cancel_event,
) -> None:
    """Handle audio input generation (ASR/Whisper) — streams text tokens back."""
    request_id = cmd.get("request_id", "")

    try:
        import numpy as np

        # Decode audio array from list (numpy arrays can't go through mp.Queue)
        audio_array = np.array(cmd["audio_data"], dtype = np.float32)

        audio_type = cmd.get("audio_type")

        if audio_type == "whisper":
            generator = backend.generate_whisper_response(
                audio_array = audio_array,
                cancel_event = cancel_event,
            )
        else:
            generator = backend.generate_audio_input_response(
                messages = cmd.get("messages", []),
                system_prompt = cmd.get("system_prompt", ""),
                audio_array = audio_array,
                temperature = cmd.get("temperature", 0.7),
                top_p = cmd.get("top_p", 0.9),
                top_k = cmd.get("top_k", 40),
                min_p = cmd.get("min_p", 0.0),
                max_new_tokens = cmd.get("max_new_tokens", 512),
                repetition_penalty = cmd.get("repetition_penalty", 1.0),
                cancel_event = cancel_event,
            )

        logger.info("Starting audio input generation for request_id=%s", request_id)

        for text_chunk in generator:
            if cancel_event.is_set():
                logger.info(
                    "Audio input generation cancelled for request %s", request_id
                )
                break

            _send_response(
                resp_queue,
                {
                    "type": "token",
                    "request_id": request_id,
                    "text": text_chunk,
                    "ts": time.time(),
                },
            )

        _send_response(
            resp_queue,
            {
                "type": "gen_done",
                "request_id": request_id,
                "ts": time.time(),
            },
        )
        logger.info("Finished audio input generation for request_id=%s", request_id)

    except Exception as exc:
        logger.error("Audio input generation error: %s", exc, exc_info = True)
        _send_response(
            resp_queue,
            {
                "type": "gen_error",
                "request_id": request_id,
                "error": str(exc),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            },
        )


def _handle_unload(backend, cmd: dict, resp_queue: Any) -> None:
    """Handle an unload command."""
    model_name = cmd.get("model_name", "")
    try:
        if model_name and model_name in backend.models:
            backend.unload_model(model_name)
        elif backend.active_model_name:
            backend.unload_model(backend.active_model_name)

        _send_response(
            resp_queue,
            {
                "type": "unloaded",
                "model_name": model_name,
                "ts": time.time(),
            },
        )
    except Exception as exc:
        logger.error("Unload error: %s", exc)
        _send_response(
            resp_queue,
            {
                "type": "unloaded",
                "model_name": model_name,
                "error": str(exc),
                "ts": time.time(),
            },
        )


def run_inference_process(
    *,
    cmd_queue: Any,
    resp_queue: Any,
    cancel_event,
    config: dict,
) -> None:
    """Subprocess entrypoint. Persistent — runs command loop until shutdown.

    Args:
        cmd_queue: mp.Queue for receiving commands from parent.
        resp_queue: mp.Queue for sending responses to parent.
        cancel_event: mp.Event shared with parent — set by parent to cancel generation.
        config: Initial configuration dict with model info.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = (
        "ignore"  # Suppress warnings at C-level before imports
    )

    if config.get("disable_xet"):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        logger.info("Xet transport disabled (HF_HUB_DISABLE_XET=1)")

    import warnings
    from loggers.config import LogConfig

    if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
        warnings.filterwarnings("ignore")

    LogConfig.setup_logging(
        service_name = "unsloth-studio-inference-worker",
        env = os.getenv("ENVIRONMENT_TYPE", "production"),
    )

    apply_gpu_ids(config.get("resolved_gpu_ids"))

    model_name = config["model_name"]

    # ── 1. Activate correct transformers version BEFORE any ML imports ──
    try:
        _activate_transformers_version(model_name)
    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": "error",
                "error": f"Failed to activate transformers version: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            },
        )
        return

    # ── 1b. On Windows, check Triton availability (must be before import torch) ──
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401

            logger.info("Triton available — torch.compile enabled")
        except ImportError:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            logger.warning(
                "Triton not found on Windows — torch.compile disabled. "
                'Install for better performance: pip install "triton-windows<3.7"'
            )

    # ── 2. Import ML libraries (fresh in this clean process) ──
    try:
        _send_response(
            resp_queue,
            {
                "type": "status",
                "message": "Importing Unsloth...",
                "ts": time.time(),
            },
        )

        backend_path = str(Path(__file__).resolve().parent.parent.parent)
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from core.inference.inference import InferenceBackend

        import transformers

        logger.info("Subprocess loaded transformers %s", transformers.__version__)

    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": "error",
                "error": f"Failed to import ML libraries: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            },
        )
        return

    # ── 3. Create inference backend and load initial model ──
    try:
        backend = InferenceBackend()

        _send_response(
            resp_queue,
            {
                "type": "status",
                "message": "Loading model...",
                "ts": time.time(),
            },
        )

        _handle_load(backend, config, resp_queue)

    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": "error",
                "error": f"Failed to initialize inference backend: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            },
        )
        return

    # ── 4. Command loop — process commands until shutdown ──
    # cancel_event is an mp.Event shared with parent — parent can set it
    # at any time to cancel generation instantly (no queue polling needed).
    logger.info("Inference subprocess ready, entering command loop")

    while True:
        try:
            cmd = cmd_queue.get(timeout = 1.0)
        except _queue.Empty:
            continue
        except (EOFError, OSError):
            logger.info("Command queue closed, shutting down")
            return

        if cmd is None:
            continue

        cmd_type = cmd.get("type", "")
        logger.info("Received command: %s", cmd_type)

        try:
            if cmd_type == "generate":
                cancel_event.clear()
                _handle_generate(backend, cmd, resp_queue, cancel_event)

            elif cmd_type == "load":
                # Load a new model (reusing this subprocess)
                # First unload current model
                if backend.active_model_name:
                    backend.unload_model(backend.active_model_name)
                _handle_load(backend, cmd, resp_queue)

            elif cmd_type == "generate_audio":
                cancel_event.clear()
                _handle_generate_audio(backend, cmd, resp_queue)

            elif cmd_type == "generate_audio_input":
                cancel_event.clear()
                _handle_generate_audio_input(backend, cmd, resp_queue, cancel_event)

            elif cmd_type == "unload":
                _handle_unload(backend, cmd, resp_queue)

            elif cmd_type == "cancel":
                # Redundant with mp.Event but handle gracefully
                cancel_event.set()
                logger.info("Cancel command received")

            elif cmd_type == "reset":
                cancel_event.set()
                backend.reset_generation_state()
                _send_response(
                    resp_queue,
                    {
                        "type": "reset_ack",
                        "ts": time.time(),
                    },
                )

            elif cmd_type == "status":
                # Return current status
                _send_response(
                    resp_queue,
                    {
                        "type": "status_response",
                        "active_model": backend.active_model_name,
                        "models": {
                            name: {
                                "is_vision": info.get("is_vision", False),
                                "is_lora": info.get("is_lora", False),
                            }
                            for name, info in backend.models.items()
                        },
                        "loading": list(backend.loading_models),
                        "ts": time.time(),
                    },
                )

            elif cmd_type == "shutdown":
                logger.info("Shutdown command received, exiting")
                # Unload all models
                for model_name in list(backend.models.keys()):
                    try:
                        backend.unload_model(model_name)
                    except Exception:
                        pass
                _send_response(
                    resp_queue,
                    {
                        "type": "shutdown_ack",
                        "ts": time.time(),
                    },
                )
                return

            else:
                logger.warning("Unknown command type: %s", cmd_type)
                _send_response(
                    resp_queue,
                    {
                        "type": "error",
                        "error": f"Unknown command type: {cmd_type}",
                        "ts": time.time(),
                    },
                )

        except Exception as exc:
            logger.error(
                "Error handling command '%s': %s", cmd_type, exc, exc_info = True
            )
            _send_response(
                resp_queue,
                {
                    "type": "error",
                    "error": f"Command '{cmd_type}' failed: {exc}",
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                },
            )
