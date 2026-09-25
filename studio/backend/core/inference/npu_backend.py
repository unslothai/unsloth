# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Manage FastFlowLM NPU models through a private Lemonade server.

Installs and validates the runtime, manages its catalog, and loads models by
``lemonade:<id>``. Health checks confirm residency and context length.
GGUF models are unsupported. This module does not import the GPU stack.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from core.inference.lemonade_server import LemonadeServer, LemonadeUnavailable
from utils.hardware.npu import detect_amd_npu
from utils.native_path_leases import child_env_without_native_path_secret
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    forget_pid,
    is_process_shutting_down,
    spawn_on_lifetime_thread,
    terminate_pid,
)
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

MODEL_PREFIX = "lemonade:"
PROVIDER_TYPE = "lemonade"
DEFAULT_CONTEXT_LENGTH = 8192
# Names the lemond install whose NPU passed `flm validate`, so a restart needs no new Enable.
_VALIDATED_MARKER = "npu_validated.json"

# Model modes Unsloth's chat cannot serve: embeddings and transcription have no chat endpoint,
# and a single-turn ("flash") model answers only the first message of a conversation.
_EXCLUDED_LABELS = frozenset({"embeddings", "transcription", "single-turn"})

_VALIDATE_PROBLEMS = (
    ("amd_device_found", "No AMD XDNA 2 NPU was found."),
    ("npu_driver_ok", "The AMD NPU driver is missing or too old."),
    ("all_fw_ok", "The NPU firmware is too old; update linux-firmware or the AMD NPU driver."),
    ("kernel_ok", "This Linux kernel has no amdxdna support; use kernel 7.0+ or amdxdna-dkms."),
    (
        "memlock_ok",
        "The locked-memory limit is too low. Set memlock to unlimited in "
        "/etc/security/limits.conf, then log out and back in.",
    ),
)
DRIVER_HELP_URL = {
    "linux": "https://lemonade-server.ai/flm_npu_linux.html",
    "win32": "https://lemonade-server.ai/driver_install.html",
}


class NpuError(RuntimeError):
    """An NPU operation failed; the message is safe to show the user."""


class NpuLoadCancelled(NpuError):
    """A load was stopped by cancel_load."""


@dataclass(frozen = True)
class NpuModel:
    id: str
    checkpoint: str
    size_gb: Optional[float]
    downloaded: bool
    labels: tuple[str, ...]
    max_context_length: Optional[int]

    @property
    def vision(self) -> bool:
        return "vision" in self.labels

    @property
    def reasoning(self) -> bool:
        return "reasoning" in self.labels

    @property
    def tools(self) -> bool:
        return "tool-calling" in self.labels

    @property
    def model_path(self) -> str:
        return MODEL_PREFIX + self.id

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_path": self.model_path,
            "checkpoint": self.checkpoint,
            "size_gb": self.size_gb,
            "downloaded": self.downloaded,
            "labels": list(self.labels),
            "supports_vision": self.vision,
            "supports_reasoning": self.reasoning,
            "supports_tools": self.tools,
            "max_context_length": self.max_context_length,
        }


@dataclass(frozen = True)
class ManagedUpstream:
    """Server-owned routing and credentials for the loaded NPU model."""

    provider_type: str
    base_url: str
    api_key: str
    model: str
    public_model: str
    supports_vision: bool
    supports_tools: bool
    supports_reasoning: bool
    context_length: Optional[int]


@dataclass
class _Loaded:
    model: NpuModel
    context_length: Optional[int]
    requested_context_length: Optional[int]


@dataclass(frozen = True)
class NpuResident:
    """One consistent read of the loaded model, taken without the lock long operations hold."""

    model: NpuModel
    context_length: Optional[int]
    requested_context_length: Optional[int]
    base_url: str
    api_key: str


def is_npu_model_path(model_path: Optional[str]) -> bool:
    return isinstance(model_path, str) and model_path.startswith(MODEL_PREFIX)


def model_id_from_path(model_path: str) -> str:
    model_id = model_path[len(MODEL_PREFIX) :].strip()
    if not model_id or "/" in model_id or "\\" in model_id:
        raise NpuError(f"Invalid NPU model id: {model_path!r}")
    return model_id


def _installer_module():
    studio_dir = Path(__file__).resolve().parents[3]
    if str(studio_dir) not in sys.path:
        sys.path.insert(0, str(studio_dir))
    import install_lemonade_prebuilt

    return install_lemonade_prebuilt


def _npu_root() -> Path:
    from utils.paths.storage_roots import studio_root
    return studio_root() / "lemonade"


def _error_message(response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500] or f"HTTP {response.status_code}"
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
        if body.get("message"):
            return str(body["message"])
    return json.dumps(body)[:500]


def _failed(response) -> bool:
    """Whether lemond refused, including its 200 answers that carry an error body."""
    if response.status_code != 200:
        return True
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and ("error" in body or body.get("status") == "error")


def _kill_tree(process: subprocess.Popen) -> None:
    # The whole tree: a child's children hold the pipes communicate() waits on.
    terminate_pid(process.pid, timeout = 5.0, owner_verified = True)


def _positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


class LemonadeNpuBackend:
    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = Path(root) if root is not None else None
        self._lock = threading.RLock()
        self._server: Optional[LemonadeServer] = None
        self._loaded: Optional[_Loaded] = None
        self._loading: Optional[str] = None
        self._load_cancelled = threading.Event()
        # Makes cancel_load and a load's commit exclusive, so a cancel either wins or finds nothing.
        self._commit_lock = threading.Lock()
        self._closing = threading.Event()
        self._validate_process: Optional[subprocess.Popen] = None
        self._hardware: Optional[dict[str, Any]] = None
        self._validation: Optional[dict[str, Any]] = None
        self._state = "idle"
        self._error: Optional[str] = None

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else _npu_root()

    def _installed_lemond(self) -> Optional[Path]:
        try:
            return _installer_module().installed_lemond(self.root)
        except Exception as exc:  # noqa: BLE001 -- a broken pin file reads as "not installed"
            logger.warning("Could not read the Lemonade install: %s", exc)
            return None

    def _flm_binary(self) -> Optional[Path]:
        name = "flm.exe" if sys.platform == "win32" else "flm"
        base = self.root / "cache" / "bin" / "flm" / "npu"
        candidates = sorted(base.rglob(name)) if base.is_dir() else []
        return candidates[0] if candidates else None

    def hardware(self) -> dict[str, Any]:
        if self._hardware is None:
            self._hardware = detect_amd_npu()
        return self._hardware

    @property
    def is_loaded(self) -> bool:
        return self.resident() is not None

    @property
    def loaded_model(self) -> Optional[NpuModel]:
        resident = self.resident()
        return resident.model if resident is not None else None

    @property
    def loaded_context_length(self) -> Optional[int]:
        resident = self.resident()
        return resident.context_length if resident is not None else None

    def resident(self) -> Optional[NpuResident]:
        """The loaded model, or None: one snapshot, since unload clears it concurrently."""
        loaded, server = self._loaded, self._server
        if loaded is None or server is None:
            return None
        port, api_key = server.port, server.api_key
        if port is None or not server.is_alive():
            return None
        return NpuResident(
            model = loaded.model,
            context_length = loaded.context_length,
            requested_context_length = loaded.requested_context_length,
            base_url = f"http://127.0.0.1:{port}",
            api_key = api_key or "",
        )

    @property
    def loading_model(self) -> Optional[str]:
        return self._loading

    def _validated_install(self) -> Optional[str]:
        try:
            marker = json.loads((self.root / _VALIDATED_MARKER).read_text(encoding = "utf-8"))
        except (OSError, ValueError):
            return None
        return marker.get("lemond") if isinstance(marker, dict) else None

    def status(self) -> dict[str, Any]:
        hardware = self.hardware()
        binary = self._installed_lemond()
        installed = binary is not None
        ready = self._state == "ready" or (
            self._state == "idle" and installed and self._validated_install() == str(binary)
        )
        running = self._server is not None and self._server.is_alive()
        resident = self.resident()
        platform_key = "linux" if sys.platform.startswith("linux") else sys.platform
        return {
            "supported": bool(hardware.get("supported")),
            "hardware": hardware,
            "runtime_installed": installed,
            "runtime_running": running,
            "state": self._state,
            "ready": ready,
            "error": self._error,
            "validation": self._validation,
            "help_url": DRIVER_HELP_URL.get(platform_key),
            "loaded_model": resident.model.model_path if resident else None,
            "context_length": resident.context_length if resident else None,
            "loading_model": self._loading,
        }

    def _server_for(self, binary: Path) -> LemonadeServer:
        return LemonadeServer(
            binary,
            cache_dir = self.root / "cache",
            config_dir = self.root / "config",
            flm_model_dir = self.root / "flm",
        )

    def _require_supported(self) -> None:
        hardware = self.hardware()
        if not hardware.get("present"):
            raise NpuError("No AMD Ryzen AI NPU was found on this machine.")
        if not hardware.get("supported"):
            raise NpuError(
                "FastFlowLM needs an XDNA 2 NPU (Ryzen AI 300/400 or Ryzen AI Max); "
                "this NPU is XDNA 1."
            )

    def _ensure_running(self) -> LemonadeServer:
        """Start the installed runtime if needed and return its server."""
        with self._lock:
            if self._server is not None and self._server.is_alive():
                return self._server
            binary = self._installed_lemond()
            if binary is None:
                raise NpuError("The NPU runtime is not installed. Enable it first.")
            if self._server is not None:
                self._server.close()
            self._loaded = None
            self._server = self._server_for(binary)
            try:
                self._server.start()
            except LemonadeUnavailable as exc:
                self._server = None
                raise NpuError(str(exc)) from exc
            return self._server

    def enable(self) -> dict[str, Any]:
        """Install Lemonade and FastFlowLM, start them and validate the NPU. Idempotent."""
        self._require_supported()
        with self._lock:
            self._error = None
            (self.root / _VALIDATED_MARKER).unlink(missing_ok = True)
            try:
                self._state = "installing"
                binary = _installer_module().install(self.root, cancel = self._closing)
                self._state = "starting"
                server = self._ensure_running()
                self._state = "installing_flm"
                response = server.request(
                    "POST",
                    "/v1/install",
                    json_body = {"recipe": "flm", "backend": "npu", "stream": False},
                    timeout = 900.0,
                )
                if _failed(response):
                    raise NpuError(f"Installing FastFlowLM failed: {_error_message(response)}")
                self._state = "validating"
                self._validation = self._validate()
                if not self._validation.get("ready"):
                    raise NpuError(
                        " ".join(self._validation.get("problems") or ["NPU validation failed."])
                    )
                marker = self.root / _VALIDATED_MARKER
                tmp = marker.with_suffix(".json.tmp")
                tmp.write_text(json.dumps({"lemond": str(binary)}) + "\n", encoding = "utf-8")
                tmp.replace(marker)
                self._state = "ready"
            except NpuError as exc:
                self._state = "failed"
                self._error = str(exc)
                raise
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
                self._state = "failed"
                self._error = f"Enabling the NPU runtime failed: {exc}"
                raise NpuError(self._error) from exc
        return self.status()

    def _validate(self) -> dict[str, Any]:
        binary = self._flm_binary()
        if binary is None:
            return {"ready": False, "problems": ["FastFlowLM was not found after installing it."]}
        env = child_env_without_native_path_secret()
        env["FLM_MODEL_PATH"] = str(self.root / "flm")
        env["FLM_DISABLE_UPDATE_CHECK"] = "1"
        if self._closing.is_set():
            return {"ready": False, "problems": ["Unsloth is shutting down."]}
        try:
            # On the lifetime thread: shutdown() kills it, and it dies with a crashed Studio.
            process = spawn_on_lifetime_thread(
                lambda: subprocess.Popen(
                    [str(binary), "validate", "--json"],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE,
                    stdin = subprocess.DEVNULL,
                    text = True,
                    encoding = "utf-8",
                    errors = "replace",
                    env = env,
                    **windows_hidden_subprocess_kwargs(),
                    **child_popen_kwargs(),
                )
            )
        except OSError as exc:
            return {"ready": False, "problems": [f"flm validate could not run: {exc}"]}
        adopt_pid(process.pid)
        self._validate_process = process
        # After adopting: a shutdown sweep that already ran would miss it.
        if self._closing.is_set() or is_process_shutting_down():
            _kill_tree(process)
        try:
            stdout, _stderr = process.communicate(timeout = 120)
        except subprocess.TimeoutExpired as exc:
            _kill_tree(process)
            process.communicate()
            return {"ready": False, "problems": [f"flm validate could not run: {exc}"]}
        finally:
            self._validate_process = None
            if process.poll() is not None:
                forget_pid(process.pid)
        try:
            report = json.loads(stdout)
        except ValueError:
            return {
                "ready": False,
                "problems": [f"flm validate returned no report: {stdout[-300:].strip()}"],
            }
        problems = [message for key, message in _VALIDATE_PROBLEMS if report.get(key) is False]
        ready = bool(report.get("ready"))
        if not ready and not problems:
            problems = ["NPU validation failed."]
        return {"ready": ready, "problems": problems, "report": report}

    def shutdown(self) -> None:
        self._closing.set()
        self.cancel_load()
        server = self._server
        if server is not None:
            server.stop()
        process = self._validate_process
        if process is not None:
            _kill_tree(process)
        with self._lock:
            self._loaded = None
            if self._server is not None:
                self._server.close()
                self._server = None
            if self._state not in ("failed",):
                self._state = "idle"
            self._closing.clear()

    def catalog(self) -> list[NpuModel]:
        server = self._ensure_running()
        response = server.request("GET", "/v1/models?show_all=true", timeout = 60.0)
        if _failed(response):
            raise NpuError(f"Listing NPU models failed: {_error_message(response)}")
        models: list[NpuModel] = []
        for row in response.json().get("data") or []:
            if not isinstance(row, dict) or row.get("recipe") != "flm":
                continue
            labels = tuple(str(label) for label in row.get("labels") or ())
            if "chat" not in labels or _EXCLUDED_LABELS.intersection(labels):
                continue
            size = row.get("size")
            models.append(
                NpuModel(
                    id = str(row.get("id")),
                    checkpoint = str(row.get("checkpoint") or ""),
                    size_gb = float(size) if isinstance(size, (int, float)) else None,
                    downloaded = bool(row.get("downloaded")),
                    labels = labels,
                    max_context_length = _positive_int(row.get("max_context_window")),
                )
            )
        return sorted(models, key = lambda model: (not model.downloaded, model.id))

    def _model(self, model_id: str) -> NpuModel:
        for model in self.catalog():
            if model.id == model_id:
                return model
        raise NpuError(f"{model_id} is not in the FastFlowLM catalog.")

    def loadable_model(self, model_id: str) -> NpuModel:
        """The catalog entry ``load`` would accept, checked before a caller unloads anything."""
        try:
            model = self._model(model_id)
        except LemonadeUnavailable as exc:
            raise NpuError(str(exc)) from exc
        if not model.downloaded:
            raise NpuError(f"{model.id} is not downloaded yet.")
        return model

    def download(self, model_id: str) -> Iterator[dict[str, Any]]:
        """Pull a model, yielding lemond's progress events, then a final ``complete`` one.

        Raises unless lemond sent its ``complete`` event: a stream cut short leaves a partial model.
        """
        model = self._model(model_id)
        server = self._ensure_running()
        event = "progress"
        completed = False
        with server.stream(
            "POST", "/v1/pull", json_body = {"model_name": model.id, "stream": True}
        ) as response:
            if response.status_code != 200:
                response.read()
                raise NpuError(f"Downloading {model.id} failed: {_error_message(response)}")
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event = line[len("event:") :].strip() or "progress"
                    continue
                # A plain JSON body instead of a stream: lemond's HTTP 200 error answer.
                raw = line[len("data:") :] if line.startswith("data:") else line
                if not line.startswith("data:") and not line.startswith("{"):
                    continue
                try:
                    data = json.loads(raw.strip())
                except ValueError:
                    continue
                if event == "error" or (isinstance(data, dict) and data.get("error")):
                    message = data.get("error") if isinstance(data, dict) else data
                    if isinstance(message, dict):
                        message = message.get("message")
                    raise NpuError(f"Downloading {model.id} failed: {message}")
                if isinstance(data, dict) and data.get("status") == "error":
                    raise NpuError(f"Downloading {model.id} failed: {data.get('message') or data}")
                completed = completed or event == "complete"
                if isinstance(data, dict):
                    yield {"event": event, **data}
                event = "progress"
        if not completed:
            raise NpuError(f"Downloading {model.id} ended before it completed.")
        yield {"event": "complete", "model": model.id, "percent": 100}

    def delete(self, model_id: str) -> None:
        with self._lock:
            resident = self.resident()
            if resident is not None and resident.model.id == model_id:
                raise NpuError("Unload the model before deleting it.")
            server = self._ensure_running()
            response = server.request("POST", "/v1/delete", json_body = {"model_name": model_id})
            if _failed(response):
                raise NpuError(f"Deleting {model_id} failed: {_error_message(response)}")

    def load(
        self,
        model_id: str,
        context_length: Optional[int] = None,
    ) -> NpuModel:
        """Load a downloaded model onto the NPU and return it once lemond reports it resident."""
        with self._lock:
            self._loading = model_id
            self._load_cancelled.clear()
            replaced = False
            loaded = False
            try:
                model = self._model(model_id)
                if not model.downloaded:
                    raise NpuError(f"{model.id} is not downloaded yet.")
                limit = model.max_context_length
                ctx = context_length if context_length and context_length > 0 else None
                ctx = ctx or min(DEFAULT_CONTEXT_LENGTH, limit or DEFAULT_CONTEXT_LENGTH)
                if limit:
                    ctx = min(ctx, limit)
                server = self._ensure_running()
                replaced = True
                self._loaded = None
                self._raise_if_load_cancelled(model_id)
                response = server.request(
                    "POST",
                    "/v1/load",
                    json_body = {"model_name": model.id, "ctx_size": ctx, "save_options": False},
                    timeout = 900.0,
                )
                self._raise_if_load_cancelled(model_id)
                if _failed(response):
                    raise NpuError(f"Loading {model.id} failed: {_error_message(response)}")
                resident_ctx = self._resident_context(server, model.id)
                if resident_ctx is None:
                    raise NpuError(f"Lemonade did not report {model.id} as loaded on the NPU.")
                with self._commit_lock:
                    self._raise_if_load_cancelled(model_id)
                    self._loaded = _Loaded(
                        model = model,
                        context_length = resident_ctx,
                        requested_context_length = (
                            context_length if context_length and context_length > 0 else None
                        ),
                    )
                    self._loading = None
                    loaded = True
                self._state = "ready"
                self._error = None
                return model
            except LemonadeUnavailable as exc:
                self._raise_if_load_cancelled(model_id)
                raise NpuError(f"Loading {model_id} failed: {exc}") from exc
            finally:
                self._loading = None
                if replaced and not loaded and self._server is not None:
                    # lemond may still hold the previous model, which Studio no longer records.
                    self._server.stop()

    def _raise_if_load_cancelled(self, model_id: str) -> None:
        if self._load_cancelled.is_set():
            raise NpuLoadCancelled(f"Loading {model_id} was cancelled.")

    def cancel_load(self, model_id: Optional[str] = None) -> bool:
        """Stop an in-flight load by stopping lemond. Does not take the lock the load holds."""
        with self._commit_lock:
            loading = self._loading
            if loading is None or (model_id is not None and loading != model_id):
                return False
            self._load_cancelled.set()
        server = self._server
        if server is not None:
            server.stop()
        return True

    @staticmethod
    def _resident_context(server: LemonadeServer, model_id: str) -> Optional[int]:
        response = server.request("GET", "/v1/health", timeout = 30.0)
        if response.status_code != 200:
            return None
        for row in response.json().get("all_models_loaded") or []:
            if (
                isinstance(row, dict)
                and row.get("model_name") == model_id
                and row.get("device") == "npu"
                and row.get("loaded", True)
            ):
                options = row.get("recipe_options") or {}
                return _positive_int(options.get("ctx_size")) or DEFAULT_CONTEXT_LENGTH
        return None

    def unload(self) -> Optional[str]:
        """Unload the resident NPU model. Returns its model path, or None if nothing was loaded."""
        with self._lock:
            loaded = self._loaded
            self._loaded = None
            if loaded is None:
                return None
            server = self._server
            if server is not None and server.is_alive():
                try:
                    response = server.request(
                        "POST",
                        "/v1/unload",
                        json_body = {"model_name": loaded.model.id},
                        timeout = 60.0,
                    )
                    failure = _error_message(response) if _failed(response) else None
                except LemonadeUnavailable as exc:
                    failure = str(exc)
                if failure is not None:
                    logger.warning(
                        "Unloading %s from Lemonade failed: %s", loaded.model.id, failure
                    )
                    server.stop()
            return loaded.model.model_path

    def upstream(self) -> ManagedUpstream:
        # Lock-free: a chat must not wait behind an enable() or delete() holding the lock.
        resident = self.resident()
        if resident is None:
            raise NpuError("No NPU model is loaded.")
        model = resident.model
        return ManagedUpstream(
            provider_type = PROVIDER_TYPE,
            base_url = f"{resident.base_url}/v1",
            api_key = resident.api_key,
            model = model.id,
            public_model = model.model_path,
            supports_vision = model.vision,
            supports_tools = model.tools,
            supports_reasoning = model.reasoning,
            context_length = resident.context_length,
        )


_backend: Optional[LemonadeNpuBackend] = None
_backend_lock = threading.Lock()


def get_npu_backend() -> LemonadeNpuBackend:
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = LemonadeNpuBackend()
        return _backend


def peek_npu_backend() -> Optional[LemonadeNpuBackend]:
    """The backend if anything has touched it, without creating one (for shutdown)."""
    return _backend
