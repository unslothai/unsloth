# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs FastFlowLM NPU models via a private Lemonade server; loads lemonade:<id>; GGUF unsupported;
does not import the GPU stack."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Optional
from urllib.parse import quote

import structlog

from core.inference import npu_local_models as local_q4nx
from core.inference import npu_memory
from core.inference.flm_files import (
    FlmDownloadCancelled,
    FlmDownloadError,
    FlmFile,
    FlmModelFiles,
    download_files,
    folder_checkpoints,
    model_folder_checkpoints,
    read_model_files,
    read_model_repo,
)
from core.inference.lemonade_server import LemonadeServer, LemonadeUnavailable, OutputWindow
from core.inference.npu_context import DEFAULT_CONTEXT_LENGTH, plan_context
from core.inference.npu_flm_args import clamp_prefill_chunk, merge_flm_args
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
_events = structlog.get_logger(__name__)

MODEL_PREFIX = "lemonade:"
PROVIDER_TYPE = "lemonade"
# FastFlowLM publishes its NPU builds as FastFlowLM/<name>-NPU2; Unsloth Studio shows <name>-NPU2.
_REPO_OWNER = "fastflowlm/"
_DISPLAY_NAME_SUFFIX = re.compile(r"-npu\d*$", re.IGNORECASE)
# Names the lemond install whose NPU passed `flm validate`, so a restart needs no new Enable.
_VALIDATED_MARKER = "npu_validated.json"

# Unservable modes: no chat endpoint for embeddings/transcription; single-turn answers once
_EXCLUDED_LABELS = frozenset({"embeddings", "transcription", "single-turn"})
# Engine limits below checkpoint metadata, from the FastFlowLM v1.0.12 model cards (docs/docs/models)
# and qwen3_core_backend.hpp; the 27B pair fails above 8192 (ROCm/FastFlowLM#1001). Recheck on pin bumps.
_FLM_CONTEXT_LIMITS = {
    "qwen3.8-mtp:27b": 8192,
    "qwen3.8-dflash2:27b": 8192,
    **dict.fromkeys(
        (
            "qwen3:0.6b",
            "qwen3:1.7b",
            "qwen3:4b",
            "qwen3:8b",
            "qwen3-core:0.6b",
            "qwen3-core:1.7b",
            "qwen3-core:4b",
            "qwen3-core:8b",
            "lfm2:1.2b",
            "lfm2:2.6b",
            "lfm2-trans:2.6b",
            "lfm2.5-it:1.2b",
            "lfm2.5-tk:1.2b",
            "lfm3:1.7b",
        ),
        32768,
    ),
}
_LEMOND_ERROR = re.compile(r"\[(?:Error|Fatal)\]\s*\([^)]*\)\s*(.+)")
# FastFlowLM failures a user can act on, by what flm or lemond prints. First match wins.
_FLM_ADVICE = (
    # FastFlowLM#904 and flm's buffer.hpp. Ahead of the hardware context entry on purpose.
    (
        re.compile(
            r"hw context instruction buffers"
            r"|Failed to allocate (?:flm_rt::ext::bo|bytes of size)",
            re.IGNORECASE,
        ),
        "The NPU ran out of memory. Windows gives the NPU about half of system RAM, so close "
        "other apps, lower the context length or pick a smaller model.",
    ),
    (
        re.compile(r"CREATE_HWCTX", re.IGNORECASE),
        "The NPU could not open a hardware context. Close other apps using the NPU, and update "
        "the AMD NPU driver if it keeps happening.",
    ),
    # FastFlowLM#685: weights from an older flm pass its file check, then crash the load.
    (
        re.compile(r"Local model \S+ version: \S+ < "),
        "These model files are from an older FastFlowLM. Delete the model and download it again.",
    ),
    # FastFlowLM#973: STATUS_SYSTEM_INTEGRITY_POLICY_VIOLATION, also as lemond prints an exit code.
    (
        re.compile(r"0xC0E90002|-1058471934|\b3236495362\b", re.IGNORECASE),
        "Windows Smart App Control or an App Control policy blocked FastFlowLM (flm.exe).",
    ),
)

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


class NpuBusy(NpuError):
    """A delete and a pull of the same model would write the same folder."""


class NpuLoadCancelled(NpuError):
    """A load was stopped by cancel_load."""


class NpuDownloadCancelled(NpuError):
    """A download was stopped on request; its partial files stay for a resume."""


@dataclass(frozen = True)
class NpuModel:
    id: str
    checkpoint: str
    size_gb: Optional[float]
    downloaded: bool
    labels: tuple[str, ...]
    max_context_length: Optional[int]
    resume_percent: Optional[int] = None
    hf_repo: Optional[str] = None
    hf_revision: Optional[str] = None
    memory_needed_gb: Optional[float] = None
    fits_memory: Optional[bool] = None
    # Set on a Q4NX folder converted in Unsloth Studio: {"folder", "base", "problem"}.
    local: Optional[dict[str, Any]] = None

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

    @property
    def sampling_id(self) -> str:
        """The id sampling recommendations are looked up by: a conversion's base model's."""
        base = (self.local or {}).get("base")
        return f"{base.replace(':', '-')}-FLM" if base else self.id

    @property
    def display_name(self) -> Optional[str]:
        """The Hugging Face repo name, as Unsloth Studio shows the model."""
        repo = (self.hf_repo or "").strip()
        return repo.rsplit("/", 1)[-1] or None

    def answers_to(self, name: str) -> bool:
        """Whether name is this model's Hugging Face repo or repo name, in any case."""
        repo = (self.hf_repo or "").strip().lower()
        wanted = name.strip().lower()
        return bool(repo) and wanted in (repo, repo.rsplit("/", 1)[-1])

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
            "resume_percent": self.resume_percent,
            "hf_repo": self.hf_repo,
            "hf_revision": self.hf_revision,
            "memory_needed_gb": self.memory_needed_gb,
            "fits_memory": self.fits_memory,
            "local": self.local,
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
    sampling_model: Optional[str] = None


@dataclass
class _Loaded:
    model: NpuModel
    context_length: Optional[int]
    requested_context_length: Optional[int]
    flm_args: Optional[str] = None
    fitted_context_length: Optional[int] = None


@dataclass(frozen = True)
class NpuResident:
    """One consistent read of the loaded model, taken without the lock long operations hold."""

    model: NpuModel
    context_length: Optional[int]
    requested_context_length: Optional[int]
    base_url: str
    api_key: str
    flm_args: Optional[str] = None
    fitted_context_length: Optional[int] = None

    @property
    def max_context_length(self) -> Optional[int]:
        if self.fitted_context_length is not None:
            return self.fitted_context_length
        return self.model.max_context_length


def is_npu_model_path(model_path: Optional[str]) -> bool:
    return isinstance(model_path, str) and model_path.startswith(MODEL_PREFIX)


def looks_like_npu_display_name(name: Optional[str]) -> bool:
    """A FastFlowLM repo (`FastFlowLM/<name>`) or a bare name ending in `-NPU<n>`, as shown in the app."""
    if not isinstance(name, str):
        return False
    text = name.strip()
    if text.lower().startswith(_REPO_OWNER):
        rest = text[len(_REPO_OWNER) :]
        return bool(rest) and "/" not in rest and ":" not in rest
    return "/" not in text and ":" not in text and bool(_DISPLAY_NAME_SUFFIX.search(text))


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


def _pinned_flm_version() -> Optional[str]:
    try:
        return str(_installer_module().load_pins()["fastflowlm"]["version"]).lstrip("v")
    except Exception:  # noqa: BLE001 -- an unreadable pin skips the check, never blocks Enable
        return None


def _npu_root() -> Path:
    from utils.paths.storage_roots import studio_root
    return studio_root() / "lemonade"


def npu_model_dir() -> Path:
    """FastFlowLM's FLM_MODEL_PATH. Fixed under the Unsloth Studio home; not movable yet."""
    return _npu_root() / "flm"


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


def _already_said(message: str, reason: str) -> bool:
    # lemond wraps one failure in prefixes ("Backend load failed: FLM pull failed ...").
    # A one-word tail ("1" of "exit code: 1") would match digits in a model name.
    while " " in reason:
        if reason in message or message in reason:
            return True
        reason = reason.partition(": ")[2]
    return False


def _explain(server: Optional[LemonadeServer], window: Optional[OutputWindow], message: str) -> str:
    """``message`` plus the first error lemond printed in ``window`` that it does not already
    say, and what to do about a FastFlowLM failure it names. Logs that output at warning, since
    lemond's lines are otherwise debug-only, and notes the outcome in the Lemonade log. A window
    shared with another request gets no reason pinned on it."""
    lines = server.output_since(window.start) if server is not None and window is not None else []
    shared = window is not None and window.shared
    explained = message
    for line in [] if shared else lines:
        match = _LEMOND_ERROR.search(line)
        reason = match.group(1).strip() if match else ""
        if reason and not reason.endswith(":") and not _already_said(message, reason):
            explained = f"{message} ({reason[:300]})"
            break
    seen = "\n".join([message] if shared else [message, *lines])
    advice = next((text for pattern, text in _FLM_ADVICE if pattern.search(seen)), "")
    log_name = server.log_path.name if server is not None and server.log_path else None
    if server is not None:
        server.note(f"Request failed: {explained}")
    if lines:
        _events.warning(
            "Lemonade output before the NPU error",
            error = message,
            lemonade_log = log_name,
            lemonade_output = lines,
            overlapped_another_request = shared,
        )
    if explained == message and log_name:
        explained = f"{message} (Lemonade's output is on the Logs page.)"
    return f"{explained} {advice}".rstrip()


def _kill_tree(process: subprocess.Popen) -> None:
    # The whole tree: a child's children hold the pipes communicate() waits on.
    terminate_pid(process.pid, timeout = 5.0, owner_verified = True)


def _positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _max_context_length(checkpoint: str, reported: Any) -> Optional[int]:
    reported = _positive_int(reported)
    limit = _FLM_CONTEXT_LIMITS.get(checkpoint)
    if limit is None:
        return reported
    return min(reported, limit) if reported else limit


class LemonadeNpuBackend:
    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = Path(root) if root is not None else None
        self._lock = threading.RLock()
        self._server: Optional[LemonadeServer] = None
        self._loaded: Optional[_Loaded] = None
        self._loading: Optional[str] = None
        self._downloads = 0
        self._downloads_lock = threading.Lock()
        self._load_cancelled = threading.Event()
        # Makes cancel_load and a load's commit exclusive, so a cancel either wins or finds nothing.
        self._commit_lock = threading.Lock()
        self._closing = threading.Event()
        self._validate_process: Optional[subprocess.Popen] = None
        self._hardware: Optional[dict[str, Any]] = None
        self._validation: Optional[dict[str, Any]] = None
        self._state = "idle"
        self._error: Optional[str] = None
        self._flm_version_cache: Optional[tuple[tuple[str, int, int], Optional[str]]] = None
        self._last_versions: dict[str, Optional[str]] = {"lemonade": None, "fastflowlm": None}

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else _npu_root()

    @property
    def _flm_model_dir(self) -> Path:
        """FastFlowLM's FLM_MODEL_PATH, where its models and Studio's downloads of them live."""
        return self.root / "flm"

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
            flm_args = loaded.flm_args,
            fitted_context_length = loaded.fitted_context_length,
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

    def _upgrade_pending(self, binary: Optional[Path]) -> bool:
        """Whether the NPU was enabled, but a Studio update has since moved the runtime's pins."""
        validated = self._validated_install()
        return binary is None and isinstance(validated, str) and bool(validated)

    def _flm_version(self, binary: Optional[Path] = None) -> Optional[str]:
        binary = binary or self._flm_binary()
        if binary is None:
            return None
        env = child_env_without_native_path_secret()
        env["FLM_MODEL_PATH"] = str(self._flm_model_dir)
        env["FLM_DISABLE_UPDATE_CHECK"] = "1"
        try:
            completed = subprocess.run(
                [str(binary), "version", "--json"],
                capture_output = True,
                stdin = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                env = env,
                **windows_hidden_subprocess_kwargs(),
            )
            version = json.loads(completed.stdout).get("version")
        except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
            return None
        return str(version).lstrip("v") if version else None

    def _installed_flm_version(self) -> Optional[str]:
        binary = self._flm_binary()
        if binary is None:
            return None
        try:
            stat = binary.stat()
        except OSError:
            return None
        key = (str(binary), stat.st_mtime_ns, stat.st_size)
        cached = self._flm_version_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        # Failures are cached too, so a mute binary is not rerun per status read.
        version = self._flm_version(binary)
        self._flm_version_cache = (key, version)
        return version

    def _installed_lemonade_version(self, lemond: Optional[Path]) -> Optional[str]:
        if lemond is None:
            return None
        try:
            return _installer_module().installed_version(lemond)
        except Exception as exc:  # noqa: BLE001 -- a missing version never hides the status
            logger.warning("Could not read the Lemonade version: %s", exc)
            return None

    def _versions(
        self, binary: Optional[Path], validated: Optional[str]
    ) -> dict[str, Optional[str]]:
        # Non-blocking: Enable replaces flm under this lock (a running flm blocks that on Windows).
        if not self._lock.acquire(blocking = False):
            return dict(self._last_versions)
        try:
            lemond = binary or (
                Path(validated) if isinstance(validated, str) and validated else None
            )
            self._last_versions = {
                "lemonade": self._installed_lemonade_version(lemond),
                "fastflowlm": self._installed_flm_version(),
            }
            return dict(self._last_versions)
        finally:
            self._lock.release()

    def status(self) -> dict[str, Any]:
        hardware = self.hardware()
        binary = self._installed_lemond()
        validated = self._validated_install()
        # An enabled NPU stays enabled across a pin change: its first use upgrades the runtime.
        upgrade_pending = self._state == "idle" and self._upgrade_pending(binary)
        installed = binary is not None or upgrade_pending
        ready = (
            self._state == "ready"
            or upgrade_pending
            or (self._state == "idle" and installed and validated == str(binary))
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
            "versions": self._versions(binary, validated),
            "memory_window_gb": npu_memory.npu_memory_window_gb(npu_memory.system_ram_gb()),
        }

    def _server_for(self, binary: Path) -> LemonadeServer:
        return LemonadeServer(
            binary,
            cache_dir = self.root / "cache",
            config_dir = self.root / "config",
            flm_model_dir = self._flm_model_dir,
            log_dir = self.root.parent / "logs" / "lemonade",
            flm_config_path = self._write_flm_list(),
        )

    @property
    def _local_dir(self) -> Path:
        return self.root / "local_q4nx"

    def _write_flm_list(self) -> Optional[Path]:
        """FastFlowLM's own catalog plus the registered converted model, or None to use it as is."""
        overlay = self._local_dir / "model_list.json"
        binary = self._flm_binary()
        listing_path = local_q4nx.find_flm_list(binary) if binary is not None else None
        try:
            registered = json.loads((self._local_dir / "registered.json").read_text("utf-8"))
            listing = json.loads(listing_path.read_text("utf-8")) if listing_path else None
        except (OSError, ValueError):
            registered = listing = None
        if not isinstance(registered, dict) or not isinstance(listing, dict):
            overlay.unlink(missing_ok = True)
            return None
        # Rebuilt from the installed catalog every start, so a FastFlowLM update is never shadowed.
        listing.setdefault("models", {})[registered["family"]] = {
            local_q4nx.LOCAL_SIZE: registered["entry"]
        }
        tmp = overlay.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(listing, indent = 2) + "\n", encoding = "utf-8")
        tmp.replace(overlay)
        return overlay

    def _require_supported(self) -> None:
        hardware = self.hardware()
        if not hardware.get("present"):
            raise NpuError("No AMD Ryzen AI NPU was found on this machine.")
        if hardware.get("family") == "XDNA2" and hardware.get("enabled") is False:
            raise NpuError(
                "The AMD NPU is turned off by UNSLOTH_STUDIO_ENABLE_NPU. Unset it or set it "
                "to 1, then restart Unsloth Studio to use the NPU."
            )
        if not hardware.get("supported"):
            raise NpuError(
                "FastFlowLM needs an XDNA 2 NPU (Ryzen AI 300/400 or Ryzen AI Max); "
                "this NPU is XDNA 1."
            )

    def _ensure_running(self) -> LemonadeServer:
        """Start the installed runtime if needed and return its server."""
        # A runtime enabled before the flag was turned off must not start again.
        self._require_supported()
        with self._lock:
            if self._server is not None and self._server.is_alive():
                return self._server
            binary = self._installed_lemond()
            if self._upgrade_pending(binary):
                logger.info("Upgrading the NPU runtime to the Lemonade and FastFlowLM pins")
                self.enable()
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
                # lemond keeps the old FastFlowLM and answers success when an update download fails.
                expected, actual = _pinned_flm_version(), self._flm_version()
                if expected and actual and actual != expected:
                    raise NpuError(
                        f"Updating FastFlowLM to v{expected} failed; v{actual} is still installed. "
                        "Check the internet connection, then try again."
                    )
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
        env["FLM_MODEL_PATH"] = str(self._flm_model_dir)
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

    def _flm_rows(self) -> list[dict[str, Any]]:
        """Every FastFlowLM model lemond knows, including the ones the catalog cannot chat with."""
        server = self._ensure_running()
        response = server.request("GET", "/v1/models?show_all=true", timeout = 60.0)
        if _failed(response):
            raise NpuError(f"Listing NPU models failed: {_error_message(response)}")
        return [
            row
            for row in response.json().get("data") or []
            if isinstance(row, dict) and row.get("recipe") == "flm"
        ]

    def catalog(self) -> list[NpuModel]:
        rows = self._flm_rows()
        models: list[NpuModel] = []
        flm_binary = self._flm_binary()
        window_gb = npu_memory.npu_memory_window_gb(npu_memory.system_ram_gb())
        for row in rows:
            # A registered conversion is listed from the acting account's exports below.
            if local_q4nx.is_local_checkpoint(str(row.get("checkpoint") or "")):
                continue
            labels = tuple(str(label) for label in row.get("labels") or ())
            if "chat" not in labels or _EXCLUDED_LABELS.intersection(labels):
                continue
            size = row.get("size")
            size_gb = float(size) if isinstance(size, (int, float)) else None
            checkpoint = str(row.get("checkpoint") or "")
            downloaded = bool(row.get("downloaded"))
            files = None if downloaded else self._model_files(checkpoint, flm_binary)
            hf_repo, hf_revision = read_model_repo(flm_binary, checkpoint)
            models.append(
                NpuModel(
                    id = str(row.get("id")),
                    checkpoint = checkpoint,
                    size_gb = size_gb,
                    downloaded = downloaded,
                    labels = labels,
                    max_context_length = _max_context_length(
                        checkpoint, row.get("max_context_window")
                    ),
                    resume_percent = files.resume_percent() if files else None,
                    hf_repo = hf_repo,
                    hf_revision = hf_revision,
                    memory_needed_gb = npu_memory.npu_memory_needed_gb(size_gb),
                    fits_memory = npu_memory.fits_npu_memory(size_gb, window_gb),
                )
            )
        models.extend(self._local_rows(models, window_gb))
        return sorted(models, key = lambda model: (not model.downloaded, model.id))

    def _local_rows(self, catalog: list[NpuModel], window_gb: Optional[float]) -> list[NpuModel]:
        by_checkpoint = {model.checkpoint: model for model in catalog}
        rows = []
        for local in self._account_local_models():
            base = by_checkpoint.get(local.base or "")
            labels = tuple(
                label
                for label in (base.labels if base else ("chat",))
                if local.vision or label != "vision"
            )
            size_gb = base.size_gb if base else round(local.size_bytes / 1e9, 2)
            rows.append(
                NpuModel(
                    id = local.id,
                    checkpoint = local.checkpoint,
                    size_gb = size_gb,
                    downloaded = True,
                    labels = labels,
                    max_context_length = base.max_context_length if base else None,
                    memory_needed_gb = npu_memory.npu_memory_needed_gb(size_gb),
                    fits_memory = npu_memory.fits_npu_memory(size_gb, window_gb),
                    local = {
                        "folder": local.display,
                        "base": local.base,
                        "problem": local.problem,
                    },
                )
            )
        return rows

    def _hidden_local(self) -> list[str]:
        try:
            hidden = json.loads((self._local_dir / "hidden.json").read_text("utf-8"))
        except (OSError, ValueError):
            return []
        return [str(stamp) for stamp in hidden] if isinstance(hidden, list) else []

    def _account_local_models(self) -> list[local_q4nx.LocalQ4nx]:
        """The acting account's conversions, less the ones deleted from the list since converting."""
        hidden = set(self._hidden_local())
        return [local for local in local_q4nx.account_local_models() if local.stamp not in hidden]

    def _local_model(self, model_id: str) -> local_q4nx.LocalQ4nx:
        for local in self._account_local_models():
            if local.id == model_id:
                return local
        raise NpuError(f"{model_id} is not a converted model in this account's exports.")

    def _register_local(self, local: local_q4nx.LocalQ4nx) -> bool:
        """Link the folder in as a FastFlowLM model; True when lemond must restart to list it."""
        if local.problem:
            raise NpuError(local.problem)
        binary = self._flm_binary()
        if binary is None:
            raise NpuError("The NPU runtime is not installed. Enable it first.")
        listing_path = local_q4nx.find_flm_list(binary)
        try:
            listing = json.loads(listing_path.read_text("utf-8")) if listing_path else None
        except (OSError, ValueError):
            listing = None
        if not isinstance(listing, dict):
            raise NpuError("FastFlowLM's model list could not be read; enable the NPU again.")
        base = local_q4nx.base_entry(listing, local.base or "")
        if base is None:
            raise NpuError(
                f"This FastFlowLM version does not ship {local.base}, which this model needs."
            )
        staged = local_q4nx.staged_folder(self._flm_model_dir, listing, local)
        stamp = local.stamp
        config = self._base_config(binary, listing_path.parent, listing, local.base or "", base)
        try:
            local_q4nx.link_files(local, staged)
            # An export republishing the folder meanwhile would pair new weights with the old base.
            if local.stamp != stamp or local_q4nx.base_of(local.folder) != local.base:
                raise NpuError(f"{local.display} changed while loading; load it again.")
            config = local_q4nx.local_config(config, local)
            (staged / "config.json").write_text(
                json.dumps(config, indent = 2) + "\n", encoding = "utf-8"
            )
        except OSError as exc:
            raise NpuError(f"Preparing {local.display} for the NPU failed: {exc}") from exc
        version = config.get("flm_version")
        registered = {
            "family": local.family,
            "folder": str(local.folder),
            "entry": local_q4nx.local_entry(base, local, str(version) if version else None),
        }
        path = self._local_dir / "registered.json"
        try:
            previous = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            previous = None
        if previous == registered and (self._local_dir / "model_list.json").is_file():
            return False
        self._local_dir.mkdir(parents = True, exist_ok = True)
        path.write_text(json.dumps(registered, indent = 2) + "\n", encoding = "utf-8")
        return True

    def _base_config(
        self, binary: Path, manifests: Path, listing: dict[str, Any], tag: str, base: dict[str, Any]
    ) -> dict[str, Any]:
        """The base model's config.json: from its download, else fetched alone and kept."""
        models = self._flm_model_dir / str(listing.get("model_path") or "models")
        cached = self._local_dir / "configs" / base["name"]
        for folder in (models / base["name"], cached):
            try:
                config = json.loads((folder / "config.json").read_text("utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(config, dict) and config.get("flm_version"):
                return config
        # read_model_files reads the manifests beside the path it is given, which may be share/flm.
        files = read_model_files(manifests / binary.name, self._flm_model_dir, tag)
        wanted = [f for f in files.files if f.name == "config.json"] if files else []
        if not wanted:
            raise NpuError(f"FastFlowLM's file list has no config.json for {tag}.")
        try:
            for _ in download_files(FlmModelFiles(folder = cached, files = tuple(wanted))):
                pass
            return json.loads((cached / "config.json").read_text("utf-8"))
        except (FlmDownloadError, OSError, ValueError) as exc:
            raise NpuError(f"Fetching {tag}'s config.json failed: {exc}") from exc

    def _local_files(self, local: local_q4nx.LocalQ4nx) -> Optional[FlmModelFiles]:
        """The staged folder Auto context sizes from; FastFlowLM's manifests do not list it."""
        binary = self._flm_binary()
        listing_path = local_q4nx.find_flm_list(binary) if binary is not None else None
        try:
            listing = json.loads(listing_path.read_text("utf-8")) if listing_path else None
            if not isinstance(listing, dict):
                return None
            staged = local_q4nx.staged_folder(self._flm_model_dir, listing, local)
            files = tuple(
                FlmFile(name = path.name, url = "", size = path.stat().st_size, digest = "", lfs = False)
                for path in staged.iterdir()
                if path.is_file()
            )
        except (OSError, ValueError):
            return None
        return FlmModelFiles(folder = staged, files = files) if files else None

    def _drop_staged(self, family: str) -> None:
        """Drop the linked copy FastFlowLM reads; the export folder itself is left alone."""
        if not family or "/" in family or "\\" in family or not local_q4nx.is_local_id(family):
            return
        path = self._local_dir / "registered.json"
        try:
            registered = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            registered = None
        if isinstance(registered, dict) and registered.get("family") == family:
            path.unlink(missing_ok = True)
        for staged in self._flm_model_dir.glob(f"*/{family}"):
            if staged.is_dir() and not staged.is_symlink():
                shutil.rmtree(staged, ignore_errors = True)

    def _unregister_local(self, local: local_q4nx.LocalQ4nx) -> None:
        self._drop_staged(local.family)
        if local.stamp:
            # Keyed by weights stamp: a re-conversion lists it again; stale stamps are dropped.
            hidden = [
                stamp
                for stamp in self._hidden_local()
                if stamp != local.stamp and local_q4nx.stamp_is_current(stamp)
            ]
            self._local_dir.mkdir(parents = True, exist_ok = True)
            (self._local_dir / "hidden.json").write_text(
                json.dumps([*hidden, local.stamp], indent = 2) + "\n", encoding = "utf-8"
            )

    def can_run(self) -> bool:
        """A supported NPU with the runtime installed: the catalog is readable once it starts."""
        return bool(self.hardware().get("supported")) and self._installed_lemond() is not None

    def model_path_for_name(self, name: str, *, start_runtime: bool) -> Optional[str]:
        """`lemonade:<id>` for the model whose Hugging Face repo or repo name is name, or None.

        The loaded model answers without I/O; the others are read from the catalog when the
        runtime is running, or ``start_runtime`` lets it start.
        """
        if not looks_like_npu_display_name(name):
            return None
        loaded = self.loaded_model
        if loaded is not None and loaded.answers_to(name):
            return loaded.model_path
        server = self._server
        if not start_runtime and (server is None or not server.is_alive()):
            return None
        try:
            matches = [model for model in self.catalog() if model.answers_to(name)]
        except (NpuError, LemonadeUnavailable):
            return None
        if len(matches) > 1:
            paths = ", ".join(sorted(model.model_path for model in matches))
            raise NpuError(
                f"{name.strip()} names more than one NPU model: {paths}. Use one of those."
            )
        return matches[0].model_path if matches else None

    def _model_files(self, checkpoint: str, flm_binary: Optional[Path]) -> Optional[FlmModelFiles]:
        return read_model_files(flm_binary, self._flm_model_dir, checkpoint)

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
        if model.local and model.local.get("problem"):
            raise NpuError(str(model.local["problem"]))
        if not model.downloaded:
            raise NpuError(f"{model.id} is not downloaded yet.")
        return model

    def download(
        self,
        model_id: str,
        cancel: Optional[threading.Event] = None,
    ) -> Iterator[dict[str, Any]]:
        """Raises unless lemond sends its complete event, since a cut-short stream leaves a partial
        model. Setting ``cancel`` raises NpuDownloadCancelled at the next chunk."""
        with self._downloads_lock:
            self._downloads += 1
        try:
            yield from self._download(model_id, cancel)
        finally:
            with self._downloads_lock:
                self._downloads -= 1

    def _download(
        self, model_id: str, cancel: Optional[threading.Event]
    ) -> Iterator[dict[str, Any]]:
        cancel = cancel or threading.Event()
        if local_q4nx.is_local_id(model_id):
            raise NpuError(f"{model_id} is a converted model already on this device.")
        model = self._model(model_id)
        server = self._ensure_running()
        files = (
            None if model.downloaded else self._model_files(model.checkpoint, self._flm_binary())
        )
        if files is not None:
            try:
                yield from download_files(files, cancel = cancel)
            except FlmDownloadCancelled as exc:
                raise NpuDownloadCancelled(f"Downloading {model.id} was cancelled.") from exc
            except FlmDownloadError as exc:
                raise NpuError(str(exc)) from exc
            yield {"event": "finishing", "percent": 99}
        elif not model.downloaded:
            logger.warning(
                "FastFlowLM's file list is unreadable; lemond downloads %s, without resume",
                model.id,
            )
            # FastFlowLM's next pull trusts a truncated file an interrupted one left; clear it first.
            server.request("POST", "/v1/delete", json_body = {"model_name": model.id})
        # Loading a converted model restarts lemond to list it; register on whichever one runs now.
        server = self._ensure_running()
        event = "progress"
        completed = False
        with server.stream(
            "POST",
            "/v1/pull",
            # Not --force, which would hash every file FastFlowLM already has once more.
            json_body = {"model_name": model.id, "stream": True, "do_not_upgrade": True},
        ) as response:
            if response.status_code != 200:
                response.read()
                raise NpuError(f"Downloading {model.id} failed: {_error_message(response)}")
            for line in response.iter_lines():
                if cancel.is_set():
                    raise NpuDownloadCancelled(f"Downloading {model.id} was cancelled.")
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
                # After Studio's own download lemond only registers the files: relay nothing, its
                # per-file percent would move the bar backwards and its complete precedes Studio's.
                if isinstance(data, dict) and files is None:
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
            if local_q4nx.is_local_id(model_id):
                try:
                    self._unregister_local(self._local_model(model_id))
                except NpuError:
                    # Its export folder is gone: the staged copy is all that is left to delete.
                    self._drop_staged(local_q4nx.family_of_id(model_id))
                return
            server = self._ensure_running()
            response = server.request("POST", "/v1/delete", json_body = {"model_name": model_id})
            if _failed(response):
                raise NpuError(f"Deleting {model_id} failed: {_error_message(response)}")

    def model_folders(self) -> list[tuple[Path, Optional[str]]]:
        """Each entry under FLM_MODEL_PATH, with the checkpoint FastFlowLM's manifest gives a model
        folder. Reads the disk only, so it never starts lemond."""
        base = self._flm_model_dir
        subfolder, checkpoints = model_folder_checkpoints(self._flm_binary())
        models = base / subfolder if subfolder else None
        rows: list[tuple[Path, Optional[str]]] = []
        try:
            children = sorted(base.iterdir()) if base.is_dir() else []
            for child in children:
                if models is not None and child == models and child.is_dir():
                    rows.extend(
                        (entry, checkpoints.get(entry.name) or _local_checkpoint(entry.name))
                        for entry in sorted(child.iterdir())
                        # Studio's per-model download locks (flm_files), not models.
                        if not entry.name.endswith(".unsloth-download.lock")
                    )
                else:
                    rows.append((child, None))
        except OSError as exc:
            logger.debug("Could not list %s: %s", base, exc)
        return rows

    def delete_checkpoint(
        self,
        checkpoint: str,
        reserve: Callable[[list[str]], ContextManager[Any]] = lambda model_ids: nullcontext(),
    ) -> str:
        """Settings > Storage lists folders by checkpoint, chat models or not; lemond deletes by id.
        reserve(model_ids) keeps pulls of those ids out of the folder until the delete ends."""
        if local_q4nx.is_local_checkpoint(checkpoint):
            model_id = local_q4nx.id_of_checkpoint(checkpoint)
            resident = self.resident()
            if resident is not None and resident.model.id == model_id:
                raise NpuError(f"Unload {model_id} before deleting this folder.")
            with reserve([model_id]):
                self.delete(model_id)
            return model_id
        # A twin left registered would read as downloaded once its shared folder is gone.
        checkpoints = folder_checkpoints(self._flm_binary(), checkpoint)
        try:
            rows = [row for row in self._flm_rows() if row.get("checkpoint") in checkpoints]
            rows.sort(key = lambda row: row.get("checkpoint") != checkpoint)
            model_ids = [str(row.get("id")) for row in rows]
        except LemonadeUnavailable as exc:
            raise NpuError(str(exc)) from exc
        if not model_ids:
            raise NpuError(f"Lemonade has no FastFlowLM model for {checkpoint}.")
        resident = self.resident()
        if resident is not None and resident.model.id in model_ids:
            raise NpuError(f"Unload {resident.model.id} before deleting this folder.")
        with reserve(model_ids):
            for model_id in model_ids:
                self.delete(model_id)
        return model_ids[0]

    def delete_downloaded(self) -> tuple[int, list[str]]:
        """Delete every downloaded model through lemond, so its registry matches the disk."""
        # A failed pull keeps its files to resume, but lemond calls that model not downloaded.
        on_disk = {checkpoint for _, checkpoint in self.model_folders() if checkpoint is not None}
        try:
            rows = [
                row
                for row in self._flm_rows()
                if row.get("downloaded") or row.get("checkpoint") in on_disk
            ]
        except (NpuError, LemonadeUnavailable) as exc:
            return 0, [str(exc)]
        deleted = 0
        errors: list[str] = []
        resident = self.resident()
        if resident is not None:
            # A twin's delete would take the resident's shared folder with it.
            held = folder_checkpoints(self._flm_binary(), resident.model.checkpoint)
            if any(row.get("checkpoint") in held for row in rows):
                errors.append(f"Unload {resident.model.id} before deleting its folder.")
            rows = [row for row in rows if row.get("checkpoint") not in held]
        model_ids = [str(row.get("id")) for row in rows]
        held_id = resident.model.id if resident is not None else None
        for checkpoint in sorted(on_disk):
            local_id = local_q4nx.id_of_checkpoint(checkpoint)
            if local_q4nx.is_local_checkpoint(checkpoint) and local_id not in (*model_ids, held_id):
                model_ids.append(local_id)
        for model_id in model_ids:
            try:
                self.delete(model_id)
                deleted += 1
            except (NpuError, LemonadeUnavailable) as exc:
                errors.append(str(exc))
        return deleted, errors

    def load(
        self,
        model_id: str,
        context_length: Optional[int] = None,
        flm_args: Optional[str] = None,
    ) -> NpuModel:
        """Load a downloaded model onto the NPU and return it once lemond reports it resident."""
        with self._lock:
            # Cleared before the load is published, so a cancel that sees it is never wiped.
            with self._commit_lock:
                self._load_cancelled.clear()
                self._loading = model_id
            window = None
            replaced = False
            loaded = False
            try:
                local = None
                if local_q4nx.is_local_id(model_id):
                    local = self._local_model(model_id)
                    self._require_supported()
                    if self._downloads:
                        # Listing a conversion restarts lemond, which would kill a running pull.
                        raise NpuError("Wait for the NPU download to finish, then load this model.")
                    if self._register_local(local) and self._server is not None:
                        # lemond reads FastFlowLM's catalog at start; a restart lists the new entry.
                        self._server.stop()
                model = self._model(model_id)
                if model.local and model.local.get("problem"):
                    raise NpuError(str(model.local["problem"]))
                if not model.downloaded:
                    raise NpuError(f"{model.id} is not downloaded yet.")
                files = (
                    self._local_files(local)
                    if local is not None
                    else self._model_files(model.checkpoint, self._flm_binary())
                )
                # Unload first so the free-RAM sample excludes the old weights and caches.
                if self._loaded is not None:
                    replaced = True
                    self.unload()
                    self._raise_if_load_cancelled(model_id)
                server = self._ensure_running()
                plan = plan_context(model, files, context_length)
                ctx = plan.context_length
                replaced = True
                self._raise_if_load_cancelled(model_id)
                json_body = {"model_name": model.id, "ctx_size": ctx, "save_options": False}
                if flm_args:
                    # lemond gets the chunk held to the context this load resolved (only the load
                    # knows it under Auto); the resident keeps the one asked for, as settings saved it.
                    json_body["flm_args"] = clamp_prefill_chunk(
                        merge_flm_args(self._saved_flm_args(server, model.id), flm_args), ctx
                    )
                server.note(f"Loading {model.id} with a {ctx}-token context")
                window = server.open_window()
                with server.backend_output(window):
                    response = server.request(
                        "POST",
                        "/v1/load",
                        json_body = json_body,
                        timeout = 900.0,
                    )
                    self._raise_if_load_cancelled(model_id)
                    if _failed(response):
                        reason = _explain(server, window, _error_message(response))
                        raise NpuError(f"Loading {model.id} failed: {reason}")
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
                        flm_args = flm_args or None,
                        fitted_context_length = plan.fits,
                    )
                    self._loading = None
                    loaded = True
                self._state = "ready"
                self._error = None
                return model
            except LemonadeUnavailable as exc:
                self._raise_if_load_cancelled(model_id)
                reason = self.explain_failure(window, str(exc))
                raise NpuError(f"Loading {model_id} failed: {reason}") from exc
            finally:
                self._loading = None
                if replaced and not loaded and self._server is not None:
                    # lemond may have loaded the replacement before failing its health check.
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
    def _saved_flm_args(server: LemonadeServer, model_id: str) -> str:
        """The model's ``flm_args`` from its saved, architecture and server-wide layers."""
        response = server.request(
            "GET", f"/v1/models/{quote(model_id, safe = '')}/options", timeout = 30.0
        )
        if response.status_code != 200:
            return ""
        try:
            effective = response.json().get("effective") or {}
        except ValueError:
            return ""
        value = effective.get("flm_args") if isinstance(effective, dict) else None
        return value if isinstance(value, str) else ""

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

    def output_mark(self, upstream: ManagedUpstream) -> Optional[OutputWindow]:
        server = self._server
        # Each daemon start gets a fresh key. A restart after routing must not claim this chat.
        if server is None or server.api_key != upstream.api_key:
            return None
        return server.open_window()

    def explain_failure(self, window: Optional[OutputWindow], message: str) -> str:
        """An NPU chat error ``message`` with the reason lemond printed in ``window``."""
        return _explain(window.server, window, message) if window is not None else message

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
            sampling_model = model.sampling_id,
        )


def _local_checkpoint(folder_name: str) -> Optional[str]:
    # A staged conversion is in no FastFlowLM manifest, so Storage could not delete it otherwise.
    if local_q4nx.is_local_checkpoint(folder_name):
        return f"{folder_name}:{local_q4nx.LOCAL_SIZE}"
    return None


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
