# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat on an OpenVINO IR (Intel Arc / Core Ultra) through a private OpenAI-compatible sidecar.

Shaped like ``npu_backend``: the routes load, unload and proxy chat through a ``ManagedUpstream``.
The sidecar (``openvino_sidecar.py``) runs in a Python with ``openvino_genai``: the one in
``UNSLOTH_OPENVINO_PYTHON``, else Studio's own when it has the package.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from core.inference.npu_backend import ManagedUpstream
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    is_process_shutting_down,
    spawn_on_lifetime_thread,
    terminate_pid,
)
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

MODEL_PREFIX = "openvino:"
PROVIDER_TYPE = "custom"
PYTHON_ENV = "UNSLOTH_OPENVINO_PYTHON"
# A text-only export has openvino_model.xml; an image-text-to-text one splits the language model out.
_IR_MARKERS = ("openvino_model.xml", "openvino_language_model.xml")
_SIDECAR = Path(__file__).with_name("openvino_sidecar.py")
# ponytail: fixed 10 min ready timeout; a 35B IR compiles in ~30 s warm, minutes cold.
_READY_TIMEOUT_S = 600.0


class OpenVinoError(RuntimeError):
    pass


class OpenVinoLoadCancelled(OpenVinoError):
    pass


@dataclass(frozen = True)
class OpenVinoModel:
    id: str
    model_path: str
    directory: str
    vision: bool
    reasoning: bool = True
    tools: bool = False
    max_context_length: Optional[int] = None


@dataclass(frozen = True)
class OpenVinoResident:
    model: OpenVinoModel
    context_length: Optional[int]
    requested_context_length: Optional[int]
    base_url: str


def _is_ir_dir(path: Path) -> bool:
    return path.is_dir() and any((path / m).is_file() for m in _IR_MARKERS)


def _hub_cache() -> Path:
    try:
        from utils.hf_cache_settings import get_hf_cache_paths
        return Path(get_hf_cache_paths().hub_cache)
    except Exception:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)


def resolve_openvino_dir(model_path: Optional[str]) -> Optional[Path]:
    """The OpenVINO IR directory *model_path* names, or None when it is not one.

    Accepts ``openvino:<path>``, a local directory, or a repo id cached in the HF hub cache (the
    ``refs/main`` snapshot first, then any snapshot holding an IR).
    """
    if not isinstance(model_path, str) or not model_path.strip():
        return None
    raw = model_path.strip()
    if raw.startswith(MODEL_PREFIX):
        raw = raw[len(MODEL_PREFIX) :].strip()
    local = Path(raw).expanduser()
    if _is_ir_dir(local):
        return local.resolve()
    if "/" not in raw or raw.count("/") != 1 or local.is_absolute():
        return None
    repo = _hub_cache() / ("models--" + raw.replace("/", "--"))
    snapshots = repo / "snapshots"
    if not snapshots.is_dir():
        return None
    ref = repo / "refs" / "main"
    if ref.is_file():
        main = snapshots / ref.read_text().strip()
        if _is_ir_dir(main):
            return main
    return next((s for s in sorted(snapshots.iterdir()) if _is_ir_dir(s)), None)


def is_openvino_model_path(model_path: Optional[str]) -> bool:
    return resolve_openvino_dir(model_path) is not None


def sidecar_python() -> str:
    override = os.environ.get(PYTHON_ENV, "").strip()
    if override:
        if not Path(override).is_file():
            raise OpenVinoError(f"{PYTHON_ENV}={override} does not exist.")
        return override
    if importlib.util.find_spec("openvino_genai") is None:
        raise OpenVinoError(
            "OpenVINO GenAI is not installed. Run `pip install openvino-genai` in Studio's "
            f"environment, or point {PYTHON_ENV} at a Python that has it."
        )
    return sys.executable


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OpenVinoBackend:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._resident: Optional[OpenVinoResident] = None
        self._loading: Optional[str] = None
        self._cancel = threading.Event()
        self._tail: deque[str] = deque(maxlen = 40)
        self._drain_thread: Optional[threading.Thread] = None

    @property
    def is_loaded(self) -> bool:
        return self._resident is not None

    @property
    def loaded_model(self) -> Optional[OpenVinoModel]:
        resident = self._resident
        return resident.model if resident is not None else None

    def resident(self) -> Optional[OpenVinoResident]:
        return self._resident

    @property
    def loading_model(self) -> Optional[str]:
        return self._loading

    def _drain(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                self._tail.append(line)
                logger.debug("[openvino] %s", line)

    def _kill_locked(self) -> None:
        proc, self._process = self._process, None
        if proc is not None and proc.poll() is None:
            terminate_pid(proc.pid, timeout = 5.0, owner_verified = True)

    def load(
        self,
        model_path: str,
        requested_context_length: Optional[int] = None,
    ) -> OpenVinoResident:
        directory = resolve_openvino_dir(model_path)
        if directory is None:
            raise OpenVinoError(f"No OpenVINO IR found for {model_path!r}.")
        python = sidecar_python()
        with self._lock:
            self._cancel.clear()
            self._loading = model_path
            try:
                self._kill_locked()
                self._resident = None
                port = _find_free_port()
                model_id = model_path.removeprefix(MODEL_PREFIX)
                cmd = [
                    python,
                    str(_SIDECAR),
                    "--model",
                    str(directory),
                    "--model-id",
                    model_id,
                    "--port",
                    str(port),
                ]
                logger.info("Starting OpenVINO sidecar: %s", " ".join(cmd))
                self._tail.clear()
                proc = spawn_on_lifetime_thread(
                    lambda: subprocess.Popen(
                        cmd,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.STDOUT,
                        stdin = subprocess.DEVNULL,
                        text = True,
                        encoding = "utf-8",
                        errors = "replace",
                        **windows_hidden_subprocess_kwargs(),
                        **child_popen_kwargs(),
                    )
                )
                adopt_pid(proc.pid)
                self._process = proc
                if is_process_shutting_down():
                    self._kill_locked()
                    raise OpenVinoError("Unsloth is shutting down; not starting OpenVINO.")
                self._drain_thread = threading.Thread(target = self._drain, args = (proc,), daemon = True)
                self._drain_thread.start()
                base_url = f"http://127.0.0.1:{port}"
                self._wait_ready(proc, base_url)
                model = OpenVinoModel(
                    id = model_id,
                    model_path = model_path,
                    directory = str(directory),
                    vision = (directory / "openvino_vision_embeddings_model.xml").is_file(),
                )
                self._resident = OpenVinoResident(
                    model = model,
                    context_length = requested_context_length,
                    requested_context_length = requested_context_length,
                    base_url = base_url,
                )
                return self._resident
            except BaseException:
                self._kill_locked()
                raise
            finally:
                self._loading = None

    def _wait_ready(self, proc: subprocess.Popen, base_url: str) -> None:
        deadline = time.monotonic() + _READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                raise OpenVinoLoadCancelled("Loading the OpenVINO model was cancelled.")
            if proc.poll() is not None:
                self._drain_thread.join(timeout = 2.0)  # the last lines explain the exit
                tail = "\n".join(self._tail)
                raise OpenVinoError(f"The OpenVINO sidecar exited. Last output:\n{tail}")
            try:
                if httpx.get(f"{base_url}/health", timeout = 2.0).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise OpenVinoError("The OpenVINO sidecar did not become ready in time.")

    def cancel_load(self, model_path: Optional[str] = None) -> bool:
        loading = self._loading
        if loading is None or (model_path is not None and model_path != loading):
            return False
        self._cancel.set()
        return True

    def unload(self) -> Optional[str]:
        with self._lock:
            resident, self._resident = self._resident, None
            self._kill_locked()
            return resident.model.model_path if resident is not None else None

    def upstream(self) -> ManagedUpstream:
        resident = self._resident
        if resident is None:
            raise OpenVinoError("No OpenVINO model is loaded.")
        model = resident.model
        return ManagedUpstream(
            provider_type = PROVIDER_TYPE,
            base_url = f"{resident.base_url}/v1",
            api_key = "",
            model = model.id,
            public_model = model.model_path,
            supports_vision = False,  # the sidecar flattens messages to text
            supports_tools = model.tools,
            supports_reasoning = model.reasoning,
            context_length = resident.context_length,
        )


_backend: Optional[OpenVinoBackend] = None
_backend_lock = threading.Lock()


def get_openvino_backend() -> OpenVinoBackend:
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = OpenVinoBackend()
        return _backend


def peek_openvino_backend() -> Optional[OpenVinoBackend]:
    return _backend
