# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LLM decision models (FastDecisionModel on Qwen, Llama, Gemma...) for the Decision API.

The LLM runs through Unsloth's loader, which patches transformers process-wide, and a CUDA context
is never returned while its process lives, so the model gets its own spawn child: unloading ends it
and the GPU memory goes with it (as core/inference/stt_transformers_worker.py).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import threading
from pathlib import Path
from typing import Any

_CTX = mp.get_context("spawn")
_BACKEND_PATH = str(Path(__file__).resolve().parent.parent.parent)
CONFIG_FILE = "decision_config.json"
HEAD_FILE = "decision_head.safetensors"
LOAD_TIMEOUT_S = 1800.0
DECIDE_TIMEOUT_S = 300.0
CLOSE_TIMEOUT_S = 10.0


class LLMWorkerError(RuntimeError):
    pass


def is_llm_decision_folder(folder: Path) -> bool:
    return (folder / CONFIG_FILE).is_file() and (folder / HEAD_FILE).is_file()


class _Events:
    # The training worker's installers report progress to a queue; nobody watches this one.
    def put(self, event) -> None:
        pass


def _base_model(folder: str) -> str:
    try:
        adapter = json.loads((Path(folder) / "adapter_config.json").read_text(encoding = "utf-8"))
        return adapter.get("base_model_name_or_path") or folder
    except (OSError, ValueError):
        return folder


def _fast_paths(folder: str) -> None:
    # The kernels a training run of the same LLM gets (causal-conv1d for Qwen3.5). A failed
    # install leaves transformers' torch path, which is slower but right.
    try:
        from core.training.worker import _ensure_causal_conv1d_fast_path, _install_fast_path_hooks
        from utils.ssm_runtime import resolved_model_wants_causal_conv1d

        base = _base_model(folder)
        wants = resolved_model_wants_causal_conv1d(base, base, os.environ.get("HF_TOKEN"))
        _ensure_causal_conv1d_fast_path(_Events(), base, required = wants)
        _install_fast_path_hooks(_Events(), base, install_causal_conv1d = wants)
    except Exception as exc:
        print(f"Decision API: fast kernels unavailable ({exc}); using the torch path.")


def _load(folder: str):
    import torch

    from core.import_guards import ensure_real_packages

    if not torch.cuda.is_available():
        raise LLMWorkerError("LLM decision models need an NVIDIA or AMD GPU; none is available.")
    _fast_paths(folder)
    ensure_real_packages("unsloth_zoo", "unsloth")
    from unsloth import FastDecisionModel

    config = json.loads((Path(folder) / CONFIG_FILE).read_text(encoding = "utf-8"))
    model, tokenizer = FastDecisionModel.from_pretrained(
        folder,
        load_in_4bit = bool(config.get("load_in_4bit", True)),
        use_gradient_checkpointing = False,
    )
    FastDecisionModel.for_inference(model)
    return model, tokenizer


def _answer(internal: dict, keys: list, p: list, confidence_from_probs) -> dict[str, Any]:
    import numpy as np

    # Laya's answer shapes and rounding, so a client sees the same response for either model.
    k = len(keys)
    confidence = round(confidence_from_probs(np.array(p), k), 4)
    probabilities = {key: round(float(v), 4) for key, v in zip(keys, p)}
    if internal["t"] == "choice":
        return {
            "type": "choice",
            "choice": keys[int(np.argmax(p))],
            "probabilities": probabilities,
            "confidence": confidence,
        }
    if internal["t"] == "score":
        return {
            "type": "score",
            "score": round(float(sum(i * v for i, v in enumerate(p))), 4),
            "legend": {str(i): c for i, c in enumerate(internal["crit"])},
            "probabilities": probabilities,
            "confidence": confidence,
        }
    true = float(p[1])
    return {"type": "noul", "noul": round(true, 4), "confidence": round(max(true, 1.0 - true), 4)}


def _decide(model, tokenizer, state, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from unsloth.models.decision import _laya, _probabilities

    confidence_from_probs = _laya().common.confidence_from_probs
    rows = _probabilities(model, tokenizer, state, questions)
    max_len = int(model.decision_config["max_len"])
    return {
        "answers": {
            name: _answer(internal, keys, p, confidence_from_probs)
            for name, (internal, keys, p, _) in zip(questions, rows)
        },
        "input_tokens": sum(len(ids) for *_, ids in rows),
        # The builder only cuts the state, and a cut state fills the context exactly.
        "truncated": any(len(ids) >= max_len for *_, ids in rows),
    }


def run_llm_worker(conn, folder: str) -> None:
    """Child entrypoint, imported by name in the spawn child."""
    if _BACKEND_PATH not in sys.path:
        sys.path.insert(0, _BACKEND_PATH)
    try:
        model, tokenizer = _load(folder)
    except BaseException as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ready", None))
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


class LLMDecisionAgent:
    """Parent-side handle; the Decision API keeps it where it keeps a laya agent."""

    device = "cuda"

    def __init__(self, folder: Path):
        self._lock = threading.Lock()
        self._conn, child = _CTX.Pipe()
        env = os.environ.get("UNSLOTH_IS_PRESENT")
        os.environ["UNSLOTH_IS_PRESENT"] = "1"
        try:
            self._process = _CTX.Process(
                target = run_llm_worker,
                args = (child, str(folder)),
                name = "systemone-llm",
                daemon = True,
            )
            self._process.start()
        finally:
            if env is None:
                os.environ.pop("UNSLOTH_IS_PRESENT", None)
        child.close()
        kind, payload = self._receive(LOAD_TIMEOUT_S)
        if kind != "ready":
            self.close()
            raise LLMWorkerError(payload)

    def _receive(self, timeout: float):
        try:
            if not self._conn.poll(timeout):
                raise LLMWorkerError(
                    f"The decision model worker did not answer within {timeout:.0f}s"
                )
            return self._conn.recv()
        except (EOFError, OSError):
            raise LLMWorkerError(
                f"The decision model worker exited (code {self._process.exitcode})"
            ) from None

    def decide(self, state, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            self._conn.send(("decide", state, questions))
            kind, payload = self._receive(DECIDE_TIMEOUT_S)
        if kind == "invalid":
            raise ValueError(payload)
        if kind != "ok":
            raise LLMWorkerError(payload)
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
