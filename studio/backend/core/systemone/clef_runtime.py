# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cloudflare Clef decision models for the Decision API, served from a spawn child.

Clef is a Qwen3.5 / Qwen3.8 backbone (9B or 27B) with a joint schema head, so it only runs on a
GPU, through Unsloth's loader and its fast linear attention kernels. Unsloth patches transformers
process-wide and a CUDA context is never returned while its process lives, so the model gets its
own process: unloading ends it and the memory goes with it (as core/inference/stt_transformers_worker.py).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

_CTX = mp.get_context("spawn")
_BACKEND_PATH = str(Path(__file__).resolve().parent.parent.parent)
LOAD_TIMEOUT_S = 1800.0
DECIDE_TIMEOUT_S = 300.0
CLOSE_TIMEOUT_S = 10.0
# Clef's own default; the model reads 64K, but a longer prompt only costs prefill memory.
MAX_LENGTH = 16384
# bf16 weights plus activations must fit, else the backbone loads in 4-bit.
_BF16_HEADROOM = 1.2


class ClefWorkerError(RuntimeError):
    pass


def _weight_bytes(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.glob("model*.safetensors"))


class _Events:
    # The training worker's installers report progress to a queue; nobody watches this one.
    def put(self, event) -> None:
        pass


def _fast_paths(folder: str) -> None:
    # The same prebuilt causal-conv1d a Qwen3.5 training run gets; the gated delta kernels ship with
    # unsloth_zoo. A failed install leaves transformers' torch path, which is slower but right.
    try:
        from core.training.worker import _ensure_causal_conv1d_fast_path, _install_fast_path_hooks
        _ensure_causal_conv1d_fast_path(_Events(), folder, required = True)
        _install_fast_path_hooks(_Events(), folder, install_causal_conv1d = True)
    except Exception as exc:
        print(f"Clef: causal-conv1d fast path unavailable ({exc}); using the torch path.")


def _load(folder: str):
    import torch

    from core.import_guards import ensure_real_packages

    _fast_paths(folder)
    ensure_real_packages("unsloth_zoo", "unsloth")
    from unsloth import FastDecisionModel

    if not torch.cuda.is_available():
        raise ClefWorkerError("Clef needs an NVIDIA GPU; this machine has none torch can use.")
    free, _ = torch.cuda.mem_get_info()
    load_in_4bit = free < _weight_bytes(Path(folder)) * _BF16_HEADROOM
    model, processor = FastDecisionModel.from_pretrained(
        folder,
        max_seq_length = MAX_LENGTH,
        load_in_4bit = load_in_4bit,
        use_gradient_checkpointing = False,
    )
    FastDecisionModel.for_inference(model)
    return model, getattr(processor, "tokenizer", processor), load_in_4bit


def _decide(model, tokenizer, state, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import torch

    from unsloth.models.clef import encode_record, systemone_answer
    from unsloth.models.decision import QUESTION_TYPES, _clef_amp_dtype, _served_temperatures

    encoded = encode_record(
        tokenizer, {"state": state, "questions": questions}, max_length = MAX_LENGTH
    )
    device = next(model.parameters()).device
    ids = torch.tensor([encoded.input_ids], device = device)
    amp_dtype = _clef_amp_dtype(model, device)
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype = amp_dtype, enabled = amp_dtype is not None),
    ):
        logits, _ = model(input_ids = ids, attention_mask = torch.ones_like(ids), records = [encoded])
    rows = [
        row[: len(question.option_ids)]
        for question, row in zip(encoded.questions, logits.float().cpu())
    ]
    # The trainer's own rule: per type, or per (type, option count) bucket, after any folded scale.
    scales = _served_temperatures(
        model.decision_config,
        rows,
        [{"qtype": QUESTION_TYPES.index(q["type"])} for q in questions.values()],
    )
    answers = {}
    for question, row, scale in zip(encoded.questions, rows, scales):
        probabilities = (row / scale).softmax(-1).tolist()
        answers[question.question_id] = systemone_answer(
            questions[question.question_id], dict(zip(question.option_ids, probabilities))
        )
    return {
        "answers": answers,
        "input_tokens": len(encoded.input_ids),
        "truncated": _truncated(tokenizer, state, questions, encoded),
    }


def _truncated(tokenizer, state, questions, encoded) -> bool:
    # A cut state fills the budget exactly, but so does one that fits exactly: one more token of
    # room tells them apart, and is only spent on prompts at the limit.
    if len(encoded.input_ids) < MAX_LENGTH:
        return False
    from unsloth.models.clef import encode_record

    record = {"state": state, "questions": questions}
    return len(encode_record(tokenizer, record, max_length = MAX_LENGTH + 1).input_ids) > MAX_LENGTH


def run_clef_worker(conn, folder: str) -> None:
    """Child entrypoint, imported by name in the spawn child."""
    if _BACKEND_PATH not in sys.path:
        sys.path.insert(0, _BACKEND_PATH)
    try:
        model, tokenizer, load_in_4bit = _load(folder)
    except BaseException as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ready", {"load_in_4bit": load_in_4bit}))
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message[0] == "close":
            return
        _, state, questions = message
        try:
            conn.send(("ok", _decide(model, tokenizer, state, questions)))
        except ValueError as exc:
            conn.send(("invalid", str(exc)))
        except BaseException as exc:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class ClefAgent:
    """Parent-side handle; the Decision API keeps it where it keeps a laya agent."""

    device = "cuda"

    def __init__(
        self,
        folder: Path,
        cancelled: Callable[[], bool] | None = None,
    ):
        self._lock = threading.Lock()
        self._broken: str | None = None
        self._conn, child = _CTX.Pipe()
        env = os.environ.get("UNSLOTH_IS_PRESENT")
        os.environ["UNSLOTH_IS_PRESENT"] = "1"
        try:
            self._process = _CTX.Process(
                target = run_clef_worker,
                args = (child, str(folder)),
                name = "systemone-clef",
                daemon = True,
            )
            self._process.start()
        finally:
            if env is None:
                os.environ.pop("UNSLOTH_IS_PRESENT", None)
        child.close()
        try:
            # Polled in slices so a training run starting mid-load stops the worker before both need the GPU.
            deadline = time.monotonic() + LOAD_TIMEOUT_S
            while time.monotonic() < deadline and not self._conn.poll(1.0):
                if cancelled is not None and cancelled():
                    raise ClefWorkerError("A training run took the GPU while Clef loaded.")
            if not self._conn.poll(0):
                raise ClefWorkerError(
                    f"The Clef worker did not answer within {LOAD_TIMEOUT_S:.0f}s"
                )
            kind, payload = self._receive(0)
        except BaseException:
            self.close()
            raise
        if kind != "ready":
            self.close()
            raise ClefWorkerError(payload)
        self.load_in_4bit = payload["load_in_4bit"]

    def _receive(self, timeout: float):
        try:
            if not self._conn.poll(timeout):
                raise ClefWorkerError(f"The Clef worker did not answer within {timeout:.0f}s")
            return self._conn.recv()
        except (EOFError, OSError):
            raise ClefWorkerError(
                f"The Clef worker exited (code {self._process.exitcode})"
            ) from None

    def decide(self, state, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            # After a timeout the late answer is still in the pipe, so the worker is never asked again.
            if self._broken is not None:
                raise ClefWorkerError(self._broken)
            try:
                self._conn.send(("decide", state, questions))
                kind, payload = self._receive(DECIDE_TIMEOUT_S)
            except (OSError, ValueError, ClefWorkerError) as exc:
                self._broken = (
                    str(exc)
                    if isinstance(exc, ClefWorkerError)
                    else (f"The Clef worker exited (code {self._process.exitcode})")
                )
                raise ClefWorkerError(self._broken) from None
        if kind == "invalid":
            raise ValueError(payload)
        if kind != "ok":
            raise ClefWorkerError(payload)
        return payload

    def close(self) -> None:
        try:
            self._conn.send(("close",))
        except (OSError, ValueError):
            pass
        self._process.join(CLOSE_TIMEOUT_S)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(CLOSE_TIMEOUT_S)
        self._conn.close()
