# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference subprocess entry point.

Each session runs in a persistent spawn subprocess, giving a clean interpreter
with no stale module state (solves transformers version-switching). It stays
alive while a model is loaded, taking commands (generate, load, unload) via
mp.Queue, and exits on shutdown or unload. Pattern follows core/training/worker.py.
"""

from __future__ import annotations

import base64
import json
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
from utils.hardware import apply_gpu_ids

_SHARE_OBJECT_MAX_BYTES = 1 << 20
_SHARE_OBJECT_ERROR_SIZE = -1

# studio/backend root, prepended to sys.path so the spawned subprocess can
# import the utils/core packages.
_BACKEND_PATH = str(Path(__file__).resolve().parent.parent.parent)


def _ensure_backend_on_path() -> None:
    if _BACKEND_PATH not in sys.path:
        sys.path.insert(0, _BACKEND_PATH)


def _activate_transformers_version(model_name: str, hf_token: str | None = None) -> None:
    """Activate the correct transformers version BEFORE any ML imports."""
    _ensure_backend_on_path()

    from utils.transformers_version import activate_transformers_for_subprocess

    activate_transformers_for_subprocess(model_name, hf_token)


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
    """Send a response to the parent process; stamps ``ts`` if absent."""
    response.setdefault("ts", time.time())
    try:
        resp_queue.put(response)
    except (OSError, ValueError) as exc:
        logger.error("Failed to send response: %s", exc)


def _encode_share_object(obj: Any) -> bytes:
    data = json.dumps(obj, separators = (",", ":"), ensure_ascii = False).encode("utf-8")
    if len(data) > _SHARE_OBJECT_MAX_BYTES:
        raise ValueError("Distributed object share payload is too large")
    return data


def _decode_share_object(data: Any) -> Any:
    return json.loads(bytes(data.tolist()).decode("utf-8"))


def _clean_token(value: str | None) -> str | None:
    """Normalize an HF token: blank or whitespace-only becomes None."""
    return value if value and value.strip() else None


def _build_model_config(config: dict):
    """Build a ModelConfig from the config dict."""
    from utils.models import ModelConfig

    model_name = config["model_name"]
    mc = ModelConfig.from_identifier(
        model_id = model_name,
        hf_token = _clean_token(config.get("hf_token")),
        gguf_variant = config.get("gguf_variant"),
    )
    if not mc:
        raise ValueError(f"Invalid model identifier: {model_name}")
    return mc


_NEMOTRON_TRUST_SUBSTRINGS = ("nemotron_h", "nemotron-h", "nemotron-3-nano")


def _needs_nemotron_trust(model_name: str, hf_token: str | None = None) -> bool:
    """Whether *model_name* is a NemotronH/Nano model that needs trust_remote_code.

    NemotronH/Nano have config-parsing bugs that require it. Must NOT match
    Llama-Nemotron (standard Llama arch), so also require the unsloth/ or nvidia/
    namespace, and a genuine first-party Hub repo (not a local path or a spoof
    name starting with "unsloth/"). The repo check is authenticated so private
    first-party repos still resolve, and runs only after the cheap checks pass.
    """
    mn = model_name.lower()
    if not (
        any(sub in mn for sub in _NEMOTRON_TRUST_SUBSTRINGS)
        and (mn.startswith("unsloth/") or mn.startswith("nvidia/"))
    ):
        return False

    from utils.security.trusted_org import is_trusted_org_repo

    return is_trusted_org_repo(model_name, hf_token = hf_token)


def _resolve_lora_4bit(mc, load_in_4bit: bool) -> bool:
    """Reconcile load_in_4bit with a LoRA adapter's recorded training method.

    lora -> base is full precision (4bit off); qlora -> base is quantized (4bit
    on); unknown method -> force off only when the base is not a -bnb-4bit repo.
    A missing or unreadable adapter_config.json leaves the value unchanged.
    """
    if not (mc.is_lora and mc.path):
        return load_in_4bit

    adapter_cfg_path = Path(mc.path) / "adapter_config.json"
    if not adapter_cfg_path.exists():
        return load_in_4bit

    import json

    try:
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        training_method = adapter_cfg.get("unsloth_training_method")
        if training_method == "lora" and load_in_4bit:
            logger.info("adapter_config.json says lora — setting load_in_4bit=False")
            return False
        if training_method == "qlora" and not load_in_4bit:
            logger.info("adapter_config.json says qlora — setting load_in_4bit=True")
            return True
        if (
            not training_method
            and mc.base_model
            and "-bnb-4bit" not in mc.base_model.lower()
            and load_in_4bit
        ):
            logger.info(
                "No training method, base model has no -bnb-4bit — setting load_in_4bit=False"
            )
            return False
    except Exception as e:
        logger.warning("Could not read adapter_config.json: %s", e)
    return load_in_4bit


def _ensure_ssm_kernels(targets: list, resp_queue: Any) -> bool:
    """Install the SSM kernels the given model(s) lazy-import in from_pretrained; no-op for
    non-SSM models, idempotent. Returns True on success; on a fatal mamba-ssm failure sends a
    'loaded' failure response and returns False. Call BEFORE importing transformers, which
    snapshots its optional-backend gates at import (a later install may not be picked up).
    """
    try:
        from utils.ssm_runtime import ensure_ssm_runtime
    except Exception as exc:
        logger.debug("ssm_runtime unavailable (%s); skipping SSM kernel pre-install", exc)
        return True

    _ssm_status = lambda m: _send_response(resp_queue, {"type": "status", "message": m})
    try:
        for ssm_target in dict.fromkeys(t for t in targets if t):
            ensure_ssm_runtime(ssm_target, status_cb = _ssm_status)
        return True
    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": "loaded",
                "success": False,
                "message": (
                    f"This model needs SSM kernel libraries (causal-conv1d / "
                    f"mamba-ssm) that could not be installed: {exc}"
                ),
                "error_kind": "ssm_runtime_install_failed",
            },
        )
        return False


def _run_security_gates(
    targets: list,
    *,
    trust_remote_code: bool,
    hf_token: str | None,
    approved_fingerprint: str | None,
    resp_queue: Any,
    compute_subdirs: bool = True,
    subject: str | None = None,
) -> bool:
    """Malware + (when trust_remote_code) remote-code consent gates over *targets*
    (model + base). Sends the matching 'loaded' failure and returns False if blocked; True
    when every target is clear.

    ``compute_subdirs=False`` keeps the gate transformers-free (``security_load_subdirs``
    imports ``model_config`` -> ``transformers``, which would snapshot optional-backend
    availability before the SSM kernels are installed): used for the pre-import preflight,
    where ``_handle_load`` re-runs the authoritative gate with full subdir scoping.
    """
    targets = list(dict.fromkeys(t for t in targets if t))

    # A poisoned pickle deserializes during from_pretrained even with trust_remote_code
    # False, so check HF's security scan every load (for a LoRA, the base deserializes).
    from utils.security import evaluate_file_security

    if compute_subdirs:
        from utils.security import security_load_subdirs

    for target in targets:
        _subdirs = security_load_subdirs(target, hf_token) if compute_subdirs else ()
        _fs = evaluate_file_security(target, hf_token = hf_token, load_subdirs = _subdirs)
        if _fs.blocked:
            _send_response(
                resp_queue,
                {
                    "type": "loaded",
                    "success": False,
                    "message": _fs.reason,
                    "error_kind": "malware_blocked",
                    "security": _fs.response_payload(),
                },
            )
            return False

    # Scan auto_map code before it runs; block CRITICAL/HIGH unless pinned-approved. Adapter
    # and base are scanned as one unit, pinned by a single fingerprint.
    if trust_remote_code:
        from utils.security import evaluate_remote_code_consent_for_targets
        _rc = evaluate_remote_code_consent_for_targets(
            targets,
            hf_token = hf_token,
            trust_remote_code = True,
            approved_fingerprint = approved_fingerprint,
            subject = subject,
        )
        if _rc.blocked:
            _send_response(
                resp_queue,
                {
                    "type": "loaded",
                    "success": False,
                    "message": (
                        f"Model '{_rc.model_name}' ships custom code flagged as "
                        f"{_rc.max_severity} by the security scan. Review "
                        f"and approve it to proceed."
                    ),
                    "error_kind": "remote_code_blocked",
                    "remote_code": _rc.response_payload(),
                },
            )
            return False

    return True


def _handle_load(backend, config: dict, resp_queue: Any) -> None:
    """Handle a load command: load a model into the backend."""
    try:
        mc = _build_model_config(config)

        hf_token = _clean_token(config.get("hf_token"))
        load_in_4bit = _resolve_lora_4bit(mc, config.get("load_in_4bit", True))

        # Latest-transformers sidecar models load 16-bit: bnb 4-bit feeds quantized
        # expert weights into unvalidated paths (e.g. grouped-MoE torch._grouped_mm).
        if load_in_4bit:
            from utils.transformers_version import latest_tier_active_for
            if latest_tier_active_for(config["model_name"], hf_token):
                load_in_4bit = False
                logger.info(
                    "Latest-transformers sidecar active for %s - forcing a 16-bit "
                    "load (4-bit is disabled for brand-new architectures)",
                    config["model_name"],
                )

        trust_remote_code = config.get("trust_remote_code", False)
        if not trust_remote_code and _needs_nemotron_trust(config["model_name"], hf_token = hf_token):
            trust_remote_code = True
            logger.info(
                "Auto-enabled trust_remote_code for Nemotron model: %s", config["model_name"]
            )

        # Authoritative gates over the model + the LoRA base resolved via mc. Must run before
        # the SSM install so a blocked model never triggers a native kernel build.
        targets = [config["model_name"]]
        if mc.is_lora and getattr(mc, "base_model", None):
            targets.append(str(mc.base_model))
        if not _run_security_gates(
            targets,
            trust_remote_code = trust_remote_code,
            hf_token = hf_token,
            approved_fingerprint = config.get("approved_remote_code_fingerprint"),
            resp_queue = resp_queue,
            subject = config.get("subject"),
        ):
            return

        # Install SSM/Mamba kernels: a no-op for the initial load (pre-installed before import)
        # but still needed for a LoRA's base (resolved only now via mc) and in-process loads.
        # Skip on MLX (no macOS wheel). Probe the base, not the adapter id / local path.
        if getattr(backend, "device", None) != "mlx":
            from utils.ssm_runtime import ssm_probe_identifier

            _ssm_base = (
                str(mc.base_model) if (mc.is_lora and getattr(mc, "base_model", None)) else None
            )
            ssm_targets = [ssm_probe_identifier(config["model_name"], _ssm_base)]
            if not _ensure_ssm_kernels(ssm_targets, resp_queue):
                return

        # Heartbeat keeps the orchestrator's inactivity deadline alive during slow
        # loads; a no-progress Xet download is reported as a stall so the parent
        # can respawn over HTTP. Watch model + base repos (base is the LoRA
        # download bottleneck).
        from utils.hf_xet_fallback import start_watchdog

        watch_repos = [mc.identifier]
        base = getattr(mc, "base_model", None)
        if base and str(base) != mc.identifier:
            watch_repos.append(str(base))

        heartbeat_stop = start_watchdog(
            repo_ids = watch_repos,
            on_stall = lambda msg: _send_response(resp_queue, {"type": "stall", "message": msg}),
            on_heartbeat = lambda msg: _send_response(resp_queue, {"type": "status", "message": msg}),
            xet_disabled = os.environ.get("HF_HUB_DISABLE_XET") == "1",
        )
        try:
            load_kwargs = {
                "config": mc,
                "max_seq_length": config.get("max_seq_length", 2048),
                "load_in_4bit": load_in_4bit,
                "hf_token": hf_token,
                "trust_remote_code": trust_remote_code,
                "gpu_ids": config.get("resolved_gpu_ids"),
            }
            if getattr(backend, "device", None) == "mlx":
                load_kwargs["parallel_mode"] = config.get("mlx_parallel_mode")
                load_kwargs["distributed_group"] = config.get("_mlx_distributed_group")
            success = backend.load_model(**load_kwargs)
        finally:
            heartbeat_stop.set()

        if success:
            model_info = {
                "identifier": mc.identifier,
                "display_name": mc.display_name,
                "is_vision": mc.is_vision,
                "is_lora": mc.is_lora,
                "is_gguf": False,
                # MLX backend sets device="mlx"; lets the UI tag MLX models.
                "is_mlx": getattr(backend, "device", None) == "mlx",
                "is_audio": getattr(mc, "is_audio", False),
                "audio_type": getattr(mc, "audio_type", None),
                "has_audio_input": getattr(mc, "has_audio_input", False),
            }
            _bm = getattr(backend, "models", {}) or {}
            _entry = (
                _bm.get(mc.identifier) or _bm.get(getattr(backend, "active_model_name", None)) or {}
            )
            try:
                _context_length = _entry.get("context_length")
                if _context_length is not None:
                    model_info["context_length"] = int(_context_length)
            except Exception as _ctx_exc:
                logger.warning("context_length forward failed: %s", _ctx_exc)
            # Forward chat_template_info so the parent can classify capabilities.
            try:
                _tpl_info = _entry.get("chat_template_info")
                if isinstance(_tpl_info, dict):
                    model_info["chat_template_info"] = {
                        "has_template": bool(_tpl_info.get("has_template", False)),
                        "template": _tpl_info.get("template"),
                        "format_type": _tpl_info.get("format_type", "generic"),
                        "template_name": _tpl_info.get("template_name"),
                        "special_tokens": _tpl_info.get("special_tokens", {}) or {},
                    }
            except Exception as _tpl_exc:
                logger.warning("chat_template_info forward failed: %s", _tpl_exc)
            _send_response(
                resp_queue,
                {
                    "type": "loaded",
                    "success": True,
                    "model_info": model_info,
                },
            )
        else:
            _send_response(
                resp_queue,
                {
                    "type": "loaded",
                    "success": False,
                    "error": "Failed to load model",
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
            },
        )


def _drain_skip_generate(cmd: dict, resp_queue: Any, drain_event) -> bool:
    """Skip a generate queued behind a cancelled one during an unload.

    The parent sets ``drain_event`` for the whole unload. Because the parent's
    per-token ``cancel_event`` is cleared at the start of every generate, a cancel
    set while this generate was still queued would otherwise be lost when it is
    dequeued. If the drain is in effect, emit an immediate (empty) ``gen_done`` so
    the parent's stream/mailbox drains fast and the switch stays fast, and report
    the generate was skipped so the caller does not clear the cancel or run it.
    """
    if drain_event is None or not drain_event.is_set():
        return False
    request_id = cmd.get("request_id", "")
    logger.info("Skipping generate for request %s: unload draining", request_id)
    _send_response(
        resp_queue,
        {
            "type": "gen_done",
            "request_id": request_id,
            "cancelled": True,
            "stats": None,
        },
    )
    return True


def _handle_generate(backend, cmd: dict, resp_queue: Any, cancel_event) -> None:
    """Handle a generate command: stream tokens back via resp_queue.

    cancel_event is an mp.Event the parent can set anytime (user stop, or new
    model load mid-generate); generation stops within 1-2 tokens.
    """
    request_id = cmd.get("request_id", "")

    try:
        image = None
        image_b64 = cmd.get("image_base64")
        if image_b64:
            image = _decode_image(image_b64)
            image = _resize_image(image)

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
            "presence_penalty": cmd.get("presence_penalty", 0.0),
            "cancel_event": cancel_event,
        }

        # Forward only present optional keys so the backend signature can evolve.
        for opt_key in (
            "tools",
            "enable_thinking",
            "reasoning_effort",
            "preserve_thinking",
        ):
            if opt_key in cmd:
                gen_kwargs[opt_key] = cmd[opt_key]

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
            # cancel_event is an mp.Event — checked instantly, no queue polling.
            if cancel_event.is_set():
                logger.info("Generation cancelled for request %s", request_id)
                break

            _send_response(
                resp_queue,
                {
                    "type": "token",
                    "request_id": request_id,
                    "text": cumulative_text,
                },
            )

        _send_response(
            resp_queue,
            {
                "type": "gen_done",
                "request_id": request_id,
                # usage/timings from the MLX backend (None elsewhere).
                "stats": getattr(backend, "last_generation_stats", None),
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
            },
        )


def _handle_share_object(backend, cmd: dict, resp_queue: Any) -> None:
    """Share a small Python object across MLX distributed ranks."""
    request_id = cmd.get("request_id", "")
    group = getattr(backend, "_distributed_group", None)
    rank = int(getattr(backend, "_distributed_rank", 0) or 0)
    world_size = int(getattr(backend, "_distributed_world_size", 1) or 1)
    obj = cmd.get("object")

    try:
        if group is None or world_size <= 1:
            shared = obj
        else:
            import mlx.core as mx
            if rank == 0:
                if obj is None:
                    mx.eval(mx.distributed.all_sum(mx.array(0), group = group))
                    shared = None
                else:
                    try:
                        data = mx.array(_encode_share_object(obj), dtype = mx.uint8)
                    except Exception:
                        mx.eval(
                            mx.distributed.all_sum(
                                mx.array(_SHARE_OBJECT_ERROR_SIZE),
                                group = group,
                            )
                        )
                        raise
                    mx.eval(mx.distributed.all_sum(mx.array(data.size), group = group))
                    mx.eval(mx.distributed.all_sum(data, group = group))
                    shared = obj
            else:
                size = int(mx.distributed.all_sum(mx.array(0), group = group).item())
                if size == _SHARE_OBJECT_ERROR_SIZE:
                    raise RuntimeError("Failed to share distributed object")
                if size == 0:
                    shared = None
                else:
                    data = mx.zeros(size, dtype = mx.uint8)
                    data = mx.distributed.all_sum(data, group = group)
                    shared = _decode_share_object(data)
        _send_response(
            resp_queue,
            {
                "type": "shared",
                "request_id": request_id,
                "object": shared,
            },
        )
    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": "share_error",
                "request_id": request_id,
                "error": str(exc),
                "stack": traceback.format_exc(limit = 20),
            },
        )


def _handle_generate_audio(backend, cmd: dict, resp_queue: Any) -> None:
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

        # Send WAV bytes as base64 (bytes can't go through mp.Queue directly).
        _send_response(
            resp_queue,
            {
                "type": "audio_done",
                "request_id": request_id,
                "wav_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "sample_rate": sample_rate,
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
            },
        )


def _handle_generate_audio_input(backend, cmd: dict, resp_queue: Any, cancel_event) -> None:
    """Handle audio input generation (ASR/Whisper) — streams text tokens back."""
    request_id = cmd.get("request_id", "")

    try:
        import numpy as np

        # numpy arrays can't go through mp.Queue, so decode from list.
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
                logger.info("Audio input generation cancelled for request %s", request_id)
                break

            _send_response(
                resp_queue,
                {
                    "type": "token",
                    "request_id": request_id,
                    "text": text_chunk,
                },
            )

        _send_response(
            resp_queue,
            {
                "type": "gen_done",
                "request_id": request_id,
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
            },
        )


def run_inference_process(
    *,
    cmd_queue: Any,
    resp_queue: Any,
    cancel_event,
    config: dict,
    drain_event = None,
) -> None:
    """Subprocess entrypoint. Persistent — runs the command loop until shutdown.

    Args:
        cmd_queue: mp.Queue for receiving commands from parent.
        resp_queue: mp.Queue for sending responses to parent.
        cancel_event: mp.Event the parent sets to cancel generation.
        config: Initial configuration dict with model info.
        drain_event: mp.Event the parent sets for the duration of an unload. Unlike
            cancel_event (cleared at the start of every generate), it is never cleared
            here, so a generate still queued behind a cancelled one is skipped rather
            than run — the cancel survives the queue handoff.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = "ignore"  # Suppress warnings at C-level before imports

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

    # ── 0. MLX fast-path — skip torch/transformers ──
    _ensure_backend_on_path()

    from utils.hardware import hardware as _hw

    _hw.detect_hardware()
    if _hw.DEVICE == _hw.DeviceType.MLX:
        # Non-fatal: fall through with the installed version, but log the cause
        # instead of swallowing it (issue #6103).
        try:
            _activate_transformers_version(model_name, config.get("hf_token") or None)
        except Exception as exc:
            logger.warning(
                "Failed to activate transformers version for '%s' (MLX inference); "
                "inference may fail if this model requires a specific version. Error: %s",
                model_name,
                exc,
            )
        try:
            from core.inference.mlx_inference import MLXInferenceBackend, _init_mlx_distributed

            backend = MLXInferenceBackend()
            if config.get("mlx_distributed"):
                group, rank, size = _init_mlx_distributed()
                config["_mlx_distributed_group"] = group
                if size <= 1:
                    # A singleton group (MLX built without distributed support,
                    # or an invalid launch env/hostfile) would leave nonzero ranks
                    # looping forever on share_distributed_object. Fail the load
                    # instead of silently continuing without sharding.
                    raise RuntimeError(
                        "MLX distributed launch requested but initialized a singleton "
                        "group (size 1). Ensure the installed MLX has distributed "
                        "support and the launch environment/hostfile is valid, or run "
                        "without distributed."
                    )
                logger.info(
                    "MLX distributed initialized in worker: rank=%s size=%s mode=%s",
                    rank,
                    size,
                    config.get("mlx_parallel_mode"),
                )
            _send_response(
                resp_queue,
                {"type": "status", "message": "Loading model..."},
            )
            _handle_load(backend, config, resp_queue)
        except Exception as exc:
            _send_response(
                resp_queue,
                {
                    "type": "error",
                    "error": f"MLX inference init failed: {exc}",
                    "stack": traceback.format_exc(limit = 20),
                },
            )
            return

        # Enter the same command loop as the GPU path.
        logger.info("MLX inference subprocess ready, entering command loop")
        while True:
            try:
                cmd = cmd_queue.get(timeout = 1.0)
            except _queue.Empty:
                continue
            except (EOFError, OSError):
                return
            if cmd is None:
                continue
            cmd_type = cmd.get("type", "")
            try:
                if cmd_type == "generate":
                    if _drain_skip_generate(cmd, resp_queue, drain_event):
                        continue
                    cancel_event.clear()
                    # Re-check the drain after clearing: the parent sets drain_event
                    # then cancel_event for an unload, so if that pair landed between
                    # the check above and this clear, the clear just erased the unload's
                    # cancel. Skip here so the outgoing model is not run to completion,
                    # which would stall the switch until the dispatcher idle-timeout.
                    if _drain_skip_generate(cmd, resp_queue, drain_event):
                        continue
                    _handle_generate(backend, cmd, resp_queue, cancel_event)
                elif cmd_type == "share_object":
                    _handle_share_object(backend, cmd, resp_queue)
                elif cmd_type == "load":
                    if backend.active_model_name:
                        backend.unload_model(backend.active_model_name)
                    _handle_load(backend, cmd, resp_queue)
                elif cmd_type == "unload":
                    _handle_unload(backend, cmd, resp_queue)
                elif cmd_type == "cancel":
                    cancel_event.set()
                elif cmd_type == "reset":
                    cancel_event.set()
                    backend.reset_generation_state()
                    _send_response(resp_queue, {"type": "reset_ack"})
                elif cmd_type == "status":
                    _send_response(
                        resp_queue,
                        {
                            "type": "status_response",
                            "active_model": backend.active_model_name,
                            "models": {
                                k: {kk: vv for kk, vv in v.items() if kk != "model"}
                                for k, v in backend.models.items()
                            },
                            "loading": list(backend.loading_models),
                        },
                    )
                elif cmd_type == "shutdown":
                    return
            except Exception as exc:
                logger.error("MLX command error (%s): %s", cmd_type, exc)
                _send_response(
                    resp_queue,
                    {
                        "type": "gen_error" if cmd_type == "generate" else "error",
                        "request_id": cmd.get("request_id"),
                        "error": str(exc),
                        "stack": traceback.format_exc(limit = 20),
                    },
                )
        return

    # ── Windows: check Triton availability ──
    # Placed ahead of the torchao stub below (which imports torch on win32 to detect ROCm),
    # matching the training and export workers' gate-then-stub ordering.
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

    # ── Stub torchao on Windows ROCm before ANY transformers import ──
    # Must precede every path that pulls transformers, not just the ML imports in section 2:
    # a local LoRA adapter with no recorded base reaches transformers here via
    # _resolve_base_model -> utils.models. See core/_torchao_stub.py; no-op off Windows ROCm.
    from core._torchao_stub import install_torchao_windows_rocm_stub

    install_torchao_windows_rocm_stub()

    # ── Resolve the effective base once, before activation/gates/install ──
    # No ML import on the common path; a local adapter with no recorded base pulls
    # transformers via utils.models, which is why the stub above precedes this.
    # A remote LoRA's base is in its Hub adapter_config.json (else surfaced only by ModelConfig
    # after import). _lora_base is set only for a genuine adapter, never a full fine-tune's base.
    import json as _json

    _ensure_backend_on_path()
    from utils.transformers_version import _remote_lora_base, _resolve_base_model

    _hf_token = _clean_token(config.get("hf_token"))
    _lora_base = None
    _local_adapter_cfg = Path(model_name) / "adapter_config.json"
    if _local_adapter_cfg.is_file():
        try:
            _lora_base = (
                _json.loads(_local_adapter_cfg.read_text()).get("base_model_name_or_path") or None
            )
        except Exception:
            _lora_base = None
    if not _lora_base:
        _lora_base = _remote_lora_base(model_name, hf_token = _hf_token)
    # Base for tier activation + the SSM-kernel heuristic: the LoRA base if any, else a full
    # fine-tune's recorded base from config.json (its name reveals the SSM/sidecar arch).
    _base = _lora_base or _resolve_base_model(model_name)

    # ── 1. Activate transformers version (on the resolved base) BEFORE any ML imports ──
    try:
        _activate_transformers_version(_base, _hf_token)
    except Exception as exc:
        _send_response(
            resp_queue,
            {
                "type": "error",
                "error": f"Failed to activate transformers version: {exc}",
                "stack": traceback.format_exc(limit = 20),
            },
        )
        return

    # ── 1b. Security gates, then SSM/Mamba kernels, BEFORE importing transformers ──
    # transformers snapshots its optional-backend gates at import, so a hybrid model's kernels
    # must be installed before the import below ("mamba-ssm is required" otherwise). The gates
    # are metadata-only, so run them first and refuse a blocked model before any native build.
    # Gate only the model + a genuine LoRA base (matching _handle_load), never a full fine-tune's
    # unloaded base; _handle_load re-runs the authoritative gates with the mc base.
    _gate_targets = [model_name]
    if _lora_base:
        _gate_targets.append(_lora_base)
    _trust_remote_code = config.get("trust_remote_code", False) or _needs_nemotron_trust(
        model_name, hf_token = _hf_token
    )
    if not _run_security_gates(
        _gate_targets,
        trust_remote_code = _trust_remote_code,
        hf_token = _hf_token,
        approved_fingerprint = config.get("approved_remote_code_fingerprint"),
        resp_queue = resp_queue,
        compute_subdirs = False,  # stay transformers-free until the SSM kernels are installed
        subject = config.get("subject"),
    ):
        return
    # Probe the resolved base for SSM kernels, not the adapter id / local checkpoint path
    # (arbitrary names must not match the SSM substrings).
    from utils.ssm_runtime import ssm_probe_identifier

    _ssm_targets = [ssm_probe_identifier(model_name, _base)]
    if not _ensure_ssm_kernels(_ssm_targets, resp_queue):
        return

    # ── 2. Import ML libraries (fresh in this clean process) ──
    try:
        _send_response(
            resp_queue,
            {
                "type": "status",
                "message": "Importing Unsloth...",
            },
        )

        _ensure_backend_on_path()

        # Recover from any namespace-package shadow before importing Unsloth.
        from core.import_guards import ensure_real_packages

        ensure_real_packages("unsloth_zoo", "unsloth")

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
            },
        )
        return

    # ── 4. Command loop — process commands until shutdown ──
    # cancel_event is an mp.Event the parent can set anytime to cancel
    # generation instantly (no queue polling needed).
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
                if _drain_skip_generate(cmd, resp_queue, drain_event):
                    continue
                cancel_event.clear()
                # Re-check the drain after clearing: the parent sets drain_event then
                # cancel_event for an unload, so if that pair landed between the check
                # above and this clear, the clear just erased the unload's cancel. Skip
                # here so the outgoing model is not run to completion, which would stall
                # the switch until the dispatcher idle-timeout tears the subprocess down.
                if _drain_skip_generate(cmd, resp_queue, drain_event):
                    continue
                _handle_generate(backend, cmd, resp_queue, cancel_event)

            elif cmd_type == "share_object":
                _handle_share_object(backend, cmd, resp_queue)

            elif cmd_type == "load":
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
                # Redundant with mp.Event but handle gracefully.
                cancel_event.set()
                logger.info("Cancel command received")

            elif cmd_type == "reset":
                cancel_event.set()
                backend.reset_generation_state()
                _send_response(
                    resp_queue,
                    {
                        "type": "reset_ack",
                    },
                )

            elif cmd_type == "status":
                _send_response(
                    resp_queue,
                    {
                        "type": "status_response",
                        "active_model": backend.active_model_name,
                        "models": {
                            name: {
                                "is_vision": info.get("is_vision", False),
                                "is_lora": info.get("is_lora", False),
                                "context_length": info.get("context_length"),
                            }
                            for name, info in backend.models.items()
                        },
                        "loading": list(backend.loading_models),
                    },
                )

            elif cmd_type == "shutdown":
                logger.info("Shutdown command received, exiting")
                for name in list(backend.models.keys()):
                    try:
                        backend.unload_model(name)
                    except Exception:
                        pass
                _send_response(
                    resp_queue,
                    {
                        "type": "shutdown_ack",
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
                    },
                )

        except Exception as exc:
            logger.error("Error handling command '%s': %s", cmd_type, exc, exc_info = True)
            _send_response(
                resp_queue,
                {
                    "type": "error",
                    "error": f"Command '{cmd_type}' failed: {exc}",
                    "stack": traceback.format_exc(limit = 20),
                },
            )
