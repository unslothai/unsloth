# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One ``audiocpp_server`` child process serving one audio.cpp model.

Shared by the dictation sidecar (speech to text) and the audio worker (speech and
music), which each own an instance. The server is bound to 127.0.0.1 on an
ephemeral port and configured with exactly one model under a per-launch random id,
so the readiness probe can tell our child from any other process that won the bind
race: it must answer ``/health`` and list that id in ``/v1/models``.

Binary discovery mirrors whisper.cpp's: ``AUDIOCPP_SERVER_PATH`` (the binary),
``UNSLOTH_AUDIO_CPP_PATH`` (an install dir), the managed ``<UNSLOTH_HOME>/audio.cpp``
tree ``install_audio_cpp_prebuilt.py`` writes, then PATH.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from core.inference.audio_cpp_models import AudioCppModel
from loggers import get_logger
from utils.prebuilt.child_env import isolate_home, scrub_env
from utils.prebuilt.runtime_libs import dedupe_existing_dirs
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    forget_pid,
    is_process_shutting_down,
)

logger = get_logger(__name__)

BINARY_NAME = "audiocpp_server.exe" if sys.platform == "win32" else "audiocpp_server"
INSTALL_RECORD = "UNSLOTH_AUDIO_CPP_PREBUILT_INFO.json"

# Model load happens before the server answers, and a multi-GB music model can take a while from a cold disk.
_SERVER_START_TIMEOUT_SECONDS = 600.0
_PROBE_TIMEOUT_SECONDS = 2.0


class AudioCppUnavailableError(RuntimeError):
    """audiocpp_server is missing, failed to start, or stopped answering."""


class AudioCppStartCancelledError(RuntimeError):
    """A start was cancelled (training pre-empted it, or Unsloth is shutting down)."""


class AudioCppRequestCancelledError(RuntimeError):
    """The caller's cancel event fired while a request was in flight."""


class AudioCppRequestError(RuntimeError):
    """audiocpp_server answered a request with a non-2xx status."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"audio.cpp returned HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def managed_audio_cpp_dir() -> Path:
    """``<UNSLOTH_HOME>/audio.cpp``, else ``<STUDIO_HOME>/audio.cpp`` in custom mode, else ``~/.unsloth/audio.cpp``."""
    legacy = Path.home() / ".unsloth" / "audio.cpp"
    try:
        from utils.paths.storage_roots import studio_root, unsloth_home

        master = unsloth_home()
        if master is not None:
            return master / "audio.cpp"
        resolved = studio_root()
        legacy_studio = Path.home() / ".unsloth" / "studio"
        try:
            is_legacy = resolved.resolve() == legacy_studio.resolve()
        except (OSError, ValueError):
            is_legacy = resolved == legacy_studio
        return legacy if is_legacy else (resolved / "audio.cpp")
    except (ImportError, OSError, ValueError):
        override = (
            os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or ""
        ).strip()
        if override:
            return Path(override).expanduser() / "audio.cpp"
        return legacy


def _is_runnable(p: Path) -> bool:
    try:
        return p.is_file() and (sys.platform == "win32" or os.access(p, os.X_OK))
    except OSError:
        # An unreadable install dir reads as engine-unavailable, never a 500.
        return False


def _layout_candidates(d: Path) -> list[Path]:
    recorded: list[Path] = []
    try:
        with open(d / INSTALL_RECORD, "r", encoding = "utf-8") as f:
            relpath = json.load(f).get("server_relpath")
        if isinstance(relpath, str) and relpath and ".." not in Path(relpath).parts:
            recorded.append(d / relpath)
    except (OSError, ValueError, AttributeError):
        pass
    return [
        *recorded,
        d / BINARY_NAME,
        d / "bin" / BINARY_NAME,
        d / "build" / "bin" / BINARY_NAME,
        d / "build" / "bin" / "Release" / BINARY_NAME,
    ]


def find_audio_cpp_server_binary() -> Optional[str]:
    env_path = os.environ.get("AUDIOCPP_SERVER_PATH")
    if env_path and _is_runnable(Path(env_path)):
        return str(Path(env_path))
    custom_dir = os.environ.get("UNSLOTH_AUDIO_CPP_PATH")
    if custom_dir:
        for p in _layout_candidates(Path(custom_dir)):
            if _is_runnable(p):
                return str(p)
    for p in _layout_candidates(managed_audio_cpp_dir()):
        if _is_runnable(p):
            return str(p)
    return shutil.which(BINARY_NAME)


def read_install_record(binary: str) -> dict:
    """The prebuilt install record beside or above ``binary``, or ``{}`` for a custom build."""
    path = Path(binary).resolve()
    for parent in list(path.parents)[:4]:
        record = parent / INSTALL_RECORD
        try:
            with open(record, "r", encoding = "utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            continue
    return {}


def is_available() -> bool:
    return find_audio_cpp_server_binary() is not None


def ensure_binary() -> str:
    binary = find_audio_cpp_server_binary()
    if binary is None:
        raise AudioCppUnavailableError(
            "The audio runtime is not installed. Run `unsloth studio update` to install it."
        )
    return binary


_ESPEAK_DATA_NAMES = ("espeak-ng-data.bin", "espeak-ng-data.gguf", "espeak-ng-data")


def binary_has_espeak(binary: Optional[str]) -> bool:
    """Whether this build can phonemize with eSpeak-ng: a static build ships its data beside the server.

    Upstream bundles are built without it, so Kokoro, Piper, KittenTTS and Inflect cannot run on them.
    """
    if not binary:
        return False
    if read_install_record(binary).get("espeak") is True:
        return True
    bin_dir = Path(binary).parent
    return any((bin_dir / name).exists() for name in _ESPEAK_DATA_NAMES)


def model_runtime_problem(model: AudioCppModel, binary: Optional[str] = None) -> Optional[str]:
    """Why ``model`` cannot run on the installed runtime, or None when it can."""
    if model.unsupported:
        return model.unsupported
    binary = binary if binary is not None else find_audio_cpp_server_binary()
    if binary is None:
        return "The audio runtime is not installed. Run `unsloth studio update` to install it."
    if model.needs_espeak and not binary_has_espeak(binary):
        return (
            f"{model.display_name} needs an audio runtime built with eSpeak-ng, and the installed one "
            "has none. Run `unsloth studio update` to install the Unsloth audio runtime."
        )
    from core.inference.audio_cpp_files import served_path_problem

    try:
        return served_path_problem(model)
    except Exception:  # noqa: BLE001 - an unreadable cache setting is materialize's to report
        return None


def select_backend(binary: str, force_cpu: bool) -> str:
    """The ``--backend`` for this launch: CPU when asked, else what the install was built for."""
    if force_cpu:
        return "cpu"
    override = (os.environ.get("UNSLOTH_AUDIO_CPP_BACKEND") or "").strip().lower()
    if override:
        return override
    recorded = str(read_install_record(binary).get("backend") or "").strip().lower()
    if recorded in ("cpu", "cuda", "vulkan", "metal", "hip"):
        return recorded
    if sys.platform == "darwin":
        return "metal"
    # A custom build with no record: CUDA when an NVIDIA driver is present, since that is audio.cpp's optimized path.
    if shutil.which("nvidia-smi"):
        return "cuda"
    return "cpu"


def child_env(binary: str) -> dict[str, str]:
    """Secrets scrubbed, home repointed at a scratch dir, co-located libs first on the loader path."""
    binary = str(Path(binary).resolve())
    env = scrub_env(os.environ)
    # Outside the managed install dir (the installer refuses to replace a directory it does not own) and
    # per user: eSpeak extracts its phoneme data here, so a shared or predictable path would let another
    # local account plant data the server reads.
    isolate_home(env, str(_child_home_dir()))
    bin_dir = str(Path(binary).parent)
    runtime_dirs: list[str] = []
    if read_install_record(binary).get("backend") == "cuda" or not read_install_record(binary):
        # A CUDA bundle installed without its cudart archive relies on the CUDA runtime torch ships,
        # after the bundle's own directory so co-located DLLs still win.
        try:
            from utils.prebuilt.runtime_libs import python_runtime_dirs
            runtime_dirs = python_runtime_dirs()
        except Exception:  # noqa: BLE001 - no torch runtime to offer
            runtime_dirs = []
    if sys.platform == "win32":
        var = "PATH"
    elif sys.platform == "darwin":
        var, runtime_dirs = "DYLD_LIBRARY_PATH", []
    else:
        var = "LD_LIBRARY_PATH"
    existing = [p for p in env.get(var, "").split(os.pathsep) if p]
    env[var] = os.pathsep.join(dedupe_existing_dirs([bin_dir, *runtime_dirs, *existing]))
    return env


# eSpeak-ng keeps its data dir in a fixed buffer (N_PATH_HOME: 160 bytes on POSIX, 230 on Windows), and
# audio.cpp extracts it ~65 bytes below the child home (.cache/audio.cpp/espeak-data/<id>/espeak-ng-data).
_MAX_CHILD_HOME_LEN = 150 if sys.platform == "win32" else 85


def _child_home_dir() -> Path:
    """A private scratch home for the child: ``<studio_root>/cache/audiocpp-home``, else (unusable or
    too long for eSpeak-ng) a per-user temp dir."""
    try:
        from utils.paths.storage_roots import studio_root
        home = studio_root() / "cache" / "audiocpp-home"
        if len(str(home)) <= _MAX_CHILD_HOME_LEN:
            home.mkdir(parents = True, exist_ok = True)
            return home
    except Exception:  # noqa: BLE001 - an unusable studio root falls back to the temp dir
        pass
    user = str(os.getuid()) if hasattr(os, "getuid") else (os.environ.get("USERNAME") or "user")
    home = Path(tempfile.gettempdir()) / f"unsloth-audiocpp-home-{user}"
    home.mkdir(mode = 0o700, parents = True, exist_ok = True)
    if hasattr(os, "getuid"):
        info = home.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise AudioCppUnavailableError(
                f"{home} is not a private directory of this user; remove it and retry."
            )
    return home


def _cpu_threads() -> Optional[int]:
    """``UNSLOTH_CPU_THREADS`` as a positive int, the thread budget the other native engines honour."""
    try:
        value = int(os.environ.get("UNSLOTH_CPU_THREADS") or 0)
    except ValueError:
        return None
    return value if value > 0 else None


def _reserve_free_port() -> tuple[socket.socket, int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    return s, s.getsockname()[1]


def _abort_connection(connection: http.client.HTTPConnection) -> None:
    try:
        if connection.sock is not None:
            connection.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        connection.close()
    except Exception:  # noqa: BLE001 - best effort
        pass


class AudioCppServer:
    """One running ``audiocpp_server`` bound to one model. Not thread-safe; owners serialise."""

    def __init__(
        self,
        process: subprocess.Popen,
        port: int,
        model: AudioCppModel,
        model_id: str,
        backend: str,
        config_dir: Path,
    ) -> None:
        self.process = process
        self.port = port
        self.model = model
        self.model_id = model_id
        self.backend = backend
        self._config_dir = config_dir

    @classmethod
    def start(
        cls,
        model: AudioCppModel,
        model_path: str,
        *,
        force_cpu: bool = False,
        cancel_event: Optional[threading.Event] = None,
        on_process: Optional[Any] = None,
    ) -> "AudioCppServer":
        """Launch the server for ``model`` loaded from ``model_path`` and wait until it serves.

        ``on_process`` is called with the Popen as soon as it exists, so an owner can
        terminate a start that training pre-empts.
        """
        binary = ensure_binary()
        problem = model_runtime_problem(model, binary)
        if problem:
            raise AudioCppUnavailableError(problem)
        backend = select_backend(binary, force_cpu)
        model_id = f"studio-{uuid.uuid4().hex[:12]}"
        entry: dict[str, Any] = {
            "id": model_id,
            "family": model.family,
            "path": model_path,
            "task": model.server_task,
            "mode": "offline",
        }
        entry.update(model.model_options)
        config_dir = Path(tempfile.mkdtemp(prefix = "unsloth-audiocpp-"))
        reservation, port = _reserve_free_port()
        config = {
            "host": "127.0.0.1",
            "port": port,
            "backend": backend,
            "lazy_load": False,
            "max_loaded_models": 1,
            # Music generation can take minutes; the owner bounds its own wait and cancels.
            "busy_timeout_ms": 0,
            "models": [entry],
        }
        config_path = config_dir / "server.json"
        config_path.write_text(json.dumps(config), encoding = "utf-8")
        log_path = config_dir / "server.log"
        command = [binary, "--config", str(config_path), "--no-ui"]
        threads = _cpu_threads()
        if threads:
            command += ["--threads", str(threads)]
        logger.info(
            "Starting audiocpp_server (%s, %s) for %s on 127.0.0.1:%s",
            model.family,
            backend,
            model.id,
            port,
        )
        if is_process_shutting_down():
            reservation.close()
            shutil.rmtree(config_dir, ignore_errors = True)
            raise AudioCppStartCancelledError(
                "Unsloth is shutting down; not starting audiocpp_server."
            )
        # Released as late as possible: the server binds the port moments after this close.
        reservation.close()
        try:
            with open(log_path, "wb") as log:
                process = subprocess.Popen(
                    command,
                    stdout = log,
                    stderr = subprocess.STDOUT,
                    stdin = subprocess.DEVNULL,
                    env = child_env(binary),
                    cwd = str(config_dir),
                    **child_popen_kwargs(),
                )
        except OSError as exc:
            # A corrupt, quarantined or wrong-architecture binary: report it like any other runtime failure.
            shutil.rmtree(config_dir, ignore_errors = True)
            raise AudioCppUnavailableError(
                f"The audio runtime could not be started: {exc}"
            ) from exc
        adopt_pid(process.pid)
        if on_process is not None:
            on_process(process)
        server = cls(process, port, model, model_id, backend, config_dir)
        try:
            if is_process_shutting_down():
                raise AudioCppStartCancelledError(
                    "Unsloth is shutting down; not starting audiocpp_server."
                )
            server._wait_until_ready(cancel_event)
        except BaseException:
            server.stop()
            raise
        return server

    def log_tail(self, limit: int = 2000) -> str:
        try:
            data = (self._config_dir / "server.log").read_bytes()
        except OSError:
            return ""
        return data[-limit:].decode("utf-8", "replace").strip()

    def alive(self) -> bool:
        return self.process.poll() is None

    def _wait_until_ready(self, cancel_event: Optional[threading.Event]) -> None:
        deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise AudioCppStartCancelledError("Audio model loading was cancelled.")
            if not self.alive():
                tail = self.log_tail()
                logger.warning("audiocpp_server exited during startup: %s", tail)
                raise AudioCppUnavailableError(
                    "The audio runtime exited before becoming ready; the model file may be "
                    "incomplete or unsupported by this build."
                    + (f" Last output: {tail[-400:]}" if tail else "")
                )
            if self._probe():
                return
            time.sleep(0.25)
        raise AudioCppUnavailableError("The audio runtime did not start in time.")

    def _get_json(self, path: str) -> Optional[dict]:
        # http.client, not urllib: urlopen routes loopback through an ambient HTTP_PROXY.
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout = _PROBE_TIMEOUT_SECONDS
        )
        try:
            connection.request("GET", path)
            with connection.getresponse() as response:
                return json.loads(response.read(65536).decode("utf-8"))
        except Exception:
            return None
        finally:
            connection.close()

    def _probe(self) -> bool:
        """Ready only when our child is alive, reports ok, and lists this launch's model id."""
        health = self._get_json("/health")
        if not isinstance(health, dict) or health.get("status") != "ok":
            return False
        models = self._get_json("/v1/models")
        if not self.alive() or not isinstance(models, dict):
            return False
        ids = {str(m.get("id")) for m in models.get("data") or [] if isinstance(m, dict)}
        return self.model_id in ids

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes = b"",
        content_type: str = "application/json",
        timeout: float = 3600.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple[int, str, bytes]:
        """One HTTP round trip. A set ``cancel_event`` closes the socket and raises."""
        if cancel_event is not None and cancel_event.is_set():
            raise AudioCppRequestCancelledError("Request cancelled.")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout = timeout)
        done = threading.Event()
        outcome: dict = {}

        def round_trip() -> None:
            try:
                connection.request(method, path, body = body, headers = {"Content-Type": content_type})
                with connection.getresponse() as response:
                    payload = response.read()
                    outcome["result"] = (
                        response.status,
                        response.getheader("Content-Type") or "",
                        payload,
                    )
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                outcome["error"] = exc
            finally:
                done.set()

        try:
            if cancel_event is None:
                round_trip()
            else:
                # Windows does not wake a blocked recv when another thread shuts the socket down, so
                # the round trip runs on its own thread and a cancel returns without waiting for it.
                threading.Thread(target = round_trip, daemon = True).start()
                while not done.wait(0.1):
                    if cancel_event.is_set():
                        _abort_connection(connection)
                        raise AudioCppRequestCancelledError("Request cancelled.")
            if "error" in outcome:
                raise outcome["error"]
            return outcome["result"]
        except AudioCppRequestCancelledError:
            raise
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise AudioCppRequestCancelledError("Request cancelled.") from exc
            if not self.alive():
                raise AudioCppUnavailableError(
                    "The audio runtime stopped while serving the request."
                    + (f" Last output: {self.log_tail()[-400:]}" if self.log_tail() else "")
                ) from exc
            raise AudioCppUnavailableError(f"The audio runtime did not answer: {exc}") from exc
        finally:
            done.set()
            connection.close()

    def post_json(self, path: str, payload: dict, **kwargs) -> tuple[str, bytes]:
        status, ctype, data = self.request(
            "POST", path, body = json.dumps(payload).encode("utf-8"), **kwargs
        )
        if not 200 <= status < 300:
            raise AudioCppRequestError(status, _error_detail(data))
        return ctype, data

    def post_multipart(
        self,
        path: str,
        fields: dict[str, str],
        file_name: str,
        file_bytes: bytes,
        file_type: str,
        **kwargs,
    ) -> tuple[str, bytes]:
        boundary = uuid.uuid4().hex
        parts: list[bytes] = []
        for name, value in fields.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode(
                    "utf-8"
                )
            )
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
            f"Content-Type: {file_type}\r\n\r\n".encode("utf-8")
            + file_bytes
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        status, ctype, data = self.request(
            "POST",
            path,
            body = b"".join(parts),
            content_type = f"multipart/form-data; boundary={boundary}",
            **kwargs,
        )
        if not 200 <= status < 300:
            raise AudioCppRequestError(status, _error_detail(data))
        return ctype, data

    def stop(self) -> None:
        process = self.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout = 10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout = 10)
        forget_pid(process.pid)
        shutil.rmtree(self._config_dir, ignore_errors = True)


def _error_detail(data: bytes) -> str:
    text = data[:2000].decode("utf-8", "replace").strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        return text or "no detail"
    if isinstance(parsed, dict):
        err = parsed.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
    return text
