# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Torch/Transformers child process for one Clef worker."""

from __future__ import annotations

import hashlib
import importlib.util
import queue as _queue
import sys
from io import BytesIO
from pathlib import Path
from typing import Any


MAX_CONTEXT_TOKENS = 16_384
# Overflow guard for the reference encoder's max_length argument.
_UNBOUNDED_ENCODE_LENGTH = (1 << 31) - 1
_SOURCE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
_VENDORED_CLEF = Path(__file__).resolve().parents[4] / "unsloth" / "_vendor" / "clef"


class ClefWorkerError(RuntimeError):
    """A child-side error whose public message is safe to return to the parent."""


class ClefWorkerCancelled(ClefWorkerError):
    """The parent cancelled a load or decision before it completed."""


def _send(response_queue, response: dict[str, Any]) -> None:
    try:
        response_queue.put(response)
    except (OSError, ValueError):
        return


def _cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ClefWorkerCancelled("Clef model operation was cancelled.")


class _UTF8Path(type(Path())):
    def read_text(
        self,
        encoding = None,
        errors = None,
        **kwargs,
    ):
        return super().read_text(encoding = encoding or "utf-8", errors = errors, **kwargs)


def _reference_module():
    """Load the audited, byte-pinned Apache source without trust_remote_code."""
    source = _VENDORED_CLEF / "joint_schema_model.py"
    if not source.is_file():
        spec = importlib.util.find_spec("unsloth")
        if spec is not None and spec.origin:
            source = Path(spec.origin).parent / "_vendor" / "clef" / "joint_schema_model.py"
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise ClefWorkerError("The audited Clef runtime source is missing.") from exc
    if digest != _SOURCE_SHA256:
        raise ClefWorkerError("The audited Clef runtime source does not match its approved hash.")
    name = "unsloth_clef_joint_schema"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ClefWorkerError("Could not load the audited Clef runtime source.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    module.Path = _UTF8Path
    return module


def _actual_device(torch, requested_device: str) -> tuple[str, Any, bool]:
    backends = (
        ("cuda", getattr(torch, "cuda", None)),
        ("xpu", getattr(torch, "xpu", None)),
        ("mps", getattr(getattr(torch, "backends", None), "mps", None)),
    )
    available = next(((name, api) for name, api in backends if api and api.is_available()), None)
    if requested_device != "gpu":
        return "cpu", torch.float32, available is not None
    if available is None:
        raise ClefWorkerError(
            "No usable PyTorch GPU is available. Select CPU or install the matching accelerator runtime."
        )
    name, api = available
    bf16 = name != "mps" and api.is_bf16_supported()
    device = "mps" if name == "mps" else f"{name}:0"
    return device, torch.bfloat16 if bf16 else torch.float16, True


def _load(snapshot_path: str, requested_device: str, cancel_event):
    _cancelled(cancel_event)
    path = Path(snapshot_path)
    if not path.is_dir():
        raise ClefWorkerError("The Clef checkpoint is not available in the local model cache.")

    # Torch stays in the child so its CUDA context dies with the worker.
    import torch

    clef = _reference_module()
    device, dtype, gpu_available = _actual_device(torch, requested_device)
    _cancelled(cancel_event)
    try:
        model, processor = clef.load_release_model(
            path,
            device = device,
            dtype = dtype,
            local_files_only = True,
            use_safetensors = True,
            trust_remote_code = False,
        )
    except Exception as exc:
        raise ClefWorkerError(
            f"Could not load the local Clef checkpoint: {type(exc).__name__}: {exc}"
        ) from exc
    _cancelled(cancel_event)
    return model, processor, device.split(":")[0], gpu_available


def _decode_images(images: list[bytes]) -> list[Any]:
    if not isinstance(images, list):
        raise ValueError("images must be a list of image bytes")
    decoded = []
    for index, blob in enumerate(images):
        if not isinstance(blob, bytes):
            raise ValueError(f"image {index + 1} is not encoded image data")
        try:
            from PIL import Image, ImageOps
            with Image.open(BytesIO(blob)) as image:
                image.load()
                upright = ImageOps.exif_transpose(image) or image
                decoded.append(upright.convert("RGB").copy())
        except Exception as exc:
            raise ValueError(f"image {index + 1} could not be decoded") from exc
    return decoded


def encode_record_untruncated(clef, tokenizer, record: dict[str, Any], processor):
    """Measure the reference record before its normal truncation path."""
    encoded = clef.encode_record(
        tokenizer,
        record,
        max_length = _UNBOUNDED_ENCODE_LENGTH,
        processor = processor,
    )
    length = len(encoded.input_ids)
    if length > MAX_CONTEXT_TOKENS:
        raise ValueError(
            f"State, images, and questions require {length} tokens; Clef accepts at most "
            f"{MAX_CONTEXT_TOKENS}. Shorten the request."
        )
    return encoded


def _decide(
    model,
    processor,
    clef,
    model_name: str,
    state: Any,
    questions: dict[str, dict[str, Any]],
    images: list[bytes],
    cancel_event,
) -> dict[str, Any]:
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    _cancelled(cancel_event)
    record: dict[str, Any] = {"model": model_name, "state": state, "questions": questions}
    if images:
        record["images"] = _decode_images(images)
    encoded = encode_record_untruncated(clef, processor.tokenizer, record, processor)
    _cancelled(cancel_event)

    import torch

    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(clef.collate_records([encoded], processor.tokenizer.pad_token_id, device))[0]
    _cancelled(cancel_event)
    answers = {
        question.question_id: clef.systemone_answer(
            questions[question.question_id],
            dict(zip(question.option_ids, question_logits.float().softmax(-1).tolist())),
        )
        for question, question_logits in zip(encoded.questions, logits)
    }
    return {
        "model": model_name,
        "answers": answers,
        "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
    }


def _error_response(phase: str, exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, ClefWorkerCancelled):
        return {"type": "error", "phase": phase, "kind": "cancelled", "message": str(exc)}
    if phase == "decide" and isinstance(exc, ValueError):
        return {
            "type": "error",
            "phase": phase,
            "kind": "invalid_request_error",
            "message": str(exc) or "Invalid Clef request.",
        }
    return {
        "type": "error",
        "phase": phase,
        "kind": "worker_error",
        "message": str(exc) or type(exc).__name__,
    }


def run_clef_worker(*, cmd_queue, resp_queue, cancel_event) -> None:
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model = processor = clef = None
    while True:
        try:
            command = cmd_queue.get(timeout = 1.0)
        except _queue.Empty:
            continue
        except (EOFError, OSError):
            return
        if not isinstance(command, dict):
            continue
        phase = str(command.get("type") or "")
        try:
            if phase == "load":
                if model is not None:
                    raise ClefWorkerError("The Clef worker already has a loaded checkpoint.")
                requested_device = str(command.get("requested_device") or "cpu")
                if requested_device != "gpu":
                    # CPU mode must not attach CUDA from inherited visibility.
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                model, processor, device, gpu_available = _load(
                    str(command["snapshot_path"]),
                    requested_device,
                    cancel_event,
                )
                clef = _reference_module()
                _send(
                    resp_queue,
                    {
                        "type": "loaded",
                        "device": device,
                        "gpu_available": gpu_available,
                    },
                )
            elif phase == "decide":
                if model is None or processor is None or clef is None:
                    raise ClefWorkerError("The Clef worker has no loaded checkpoint.")
                _send(
                    resp_queue,
                    {
                        "type": "result",
                        "result": _decide(
                            model,
                            processor,
                            clef,
                            str(command["model"]),
                            command["state"],
                            command["questions"],
                            command.get("images") or [],
                            cancel_event,
                        ),
                    },
                )
            elif phase == "shutdown":
                _send(resp_queue, {"type": "shutdown_ack"})
                return
            else:
                raise ClefWorkerError(f"Unknown Clef worker command '{phase}'.")
        except BaseException as exc:  # every command must release a waiting parent
            _send(resp_queue, _error_response(phase, exc))
            if phase == "load":
                # Exit to release a possibly partial device context.
                return
