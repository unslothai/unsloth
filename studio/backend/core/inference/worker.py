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
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any

logger = get_logger(__name__)


def _activate_transformers_version(model_name: str) -> None:
    """Activate the correct transformers version BEFORE any ML imports.

    If the model needs transformers 5.x, prepend the pre-installed .venv_t5/
    directory to sys.path. Otherwise do nothing (default 4.57.x in .venv/).
    """
    # Ensure backend is on path for utils imports
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.transformers_version import (
        needs_transformers_5,
        _resolve_base_model,
        _ensure_venv_t5_exists,
        _VENV_T5_DIR,
    )

    resolved = _resolve_base_model(model_name)
    if needs_transformers_5(resolved):
        if not _ensure_venv_t5_exists():
            raise RuntimeError(
                f"Cannot activate transformers 5.x: .venv_t5 missing at {_VENV_T5_DIR}"
            )
        if _VENV_T5_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_DIR)
        logger.info("Activated transformers 5.x from %s", _VENV_T5_DIR)
        # Propagate to child subprocesses (e.g. GGUF converter)
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_DIR + (os.pathsep + _pp if _pp else "")
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

        success = backend.load_model(
            config = mc,
            max_seq_length = config.get("max_seq_length", 2048),
            load_in_4bit = load_in_4bit,
            hf_token = hf_token,
            trust_remote_code = config.get("trust_remote_code", False),
        )

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

    import warnings
    from loggers.config import LogConfig

    if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
        warnings.filterwarnings("ignore")

    LogConfig.setup_logging(
        service_name = "unsloth-studio-inference-worker",
        env = os.getenv("ENVIRONMENT_TYPE", "production"),
    )

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
