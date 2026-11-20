# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD RyzenAI NPU models (onnxruntime-genai) for Windows, without Lemonade or any unsigned EXE.

Install keeps only the DLLs of AMD's ryzenai-server.zip; ryzenai_server.py serves a model from
Studio's own Python, so Smart App Control has nothing new to refuse.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx

from utils.child_stdio import utf8_child_env
from utils.code_integrity import code_integrity_block_reason, code_integrity_user_message
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

ID_PREFIX = "ryzenai-"
_STUDIO_DIR = Path(__file__).resolve().parents[3]
PINS_PATH = _STUDIO_DIR / "ryzenai_prebuilt_pins.json"
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_MARKER = "UNSLOTH_RYZENAI_INFO.json"
_COMPLETE = ".unsloth-complete"


class RyzenAiError(RuntimeError):
    """The message is safe to show the user."""


class RyzenAiDownloadCancelled(RyzenAiError):
    pass


def load_pins(path: Path = PINS_PATH) -> dict[str, Any]:
    pins = json.loads(Path(path).read_text(encoding = "utf-8"))
    if pins.get("schema_version") != 1:
        raise RyzenAiError(f"{path} has an unsupported schema_version.")
    return pins


def supported_platform() -> bool:
    # AMD publishes the RyzenAI NPU runtime for Windows x64 only; Linux keeps FastFlowLM.
    return sys.platform == "win32"


def is_ryzenai_id(model_id: str) -> bool:
    return model_id.startswith(ID_PREFIX)


def model_id_for(repo: str) -> str:
    return ID_PREFIX + repo.rsplit("/", 1)[-1]


@dataclass(frozen = True)
class RyzenAiModel:
    id: str
    repo: str
    size_gb: Optional[float]
    labels: tuple[str, ...]
    downloaded: bool


class RyzenAiRuntime:
    def __init__(
        self,
        root: Path,
        pins_path: Path = PINS_PATH,
    ) -> None:
        self.root = Path(root)
        self.pins_path = pins_path

    @property
    def pins(self) -> dict[str, Any]:
        return load_pins(self.pins_path)

    @property
    def context_length(self) -> int:
        return int(self.pins.get("context_length") or 4096)

    def runtime_dir(self) -> Path:
        return self.root / "runtime" / self.pins["runtime"]["version"]

    def models_dir(self) -> Path:
        return self.root / "models"

    def installed(self) -> Optional[Path]:
        """The DLL folder of a complete install matching the pin, else None."""
        target = self.runtime_dir()
        try:
            marker = json.loads((target / _MARKER).read_text(encoding = "utf-8"))
        except (OSError, ValueError):
            return None
        if (
            not isinstance(marker, dict)
            or marker.get("sha256") != self.pins["runtime"]["asset"]["sha256"]
        ):
            return None
        from core.inference.ryzenai_oga import RUNTIME_DLLS

        return target if all((target / name).is_file() for name in RUNTIME_DLLS) else None

    def install(self, cancel: Optional[threading.Event] = None) -> Path:
        """Download the pinned ryzenai-server.zip and keep its DLLs; never its EXE."""
        existing = self.installed()
        if existing is not None:
            return existing
        if str(_STUDIO_DIR) not in sys.path:
            sys.path.insert(0, str(_STUDIO_DIR))
        import install_lemonade_prebuilt as lemonade_installer
        import prebuilt_core as core

        runtime = self.pins["runtime"]
        asset = runtime["asset"]
        url = core.release_asset_download_url(
            runtime["repo"], f"v{runtime['version']}", asset["name"]
        )
        target = self.runtime_dir()
        target.parent.mkdir(parents = True, exist_ok = True)
        with core.install_lock(core.install_lock_path(target)):
            existing = self.installed()
            if existing is not None:
                return existing
            with tempfile.TemporaryDirectory(
                prefix = ".ryzenai-install-", dir = target.parent
            ) as scratch:
                archive = Path(scratch) / asset["name"]
                lemonade_installer._download(
                    url, archive, expected_sha256 = asset["sha256"], progress = None, cancel = cancel
                )
                payload = Path(scratch) / "payload"
                payload.mkdir()
                with zipfile.ZipFile(archive) as zf:
                    for info in zf.infolist():
                        name = Path(info.filename).name
                        if info.is_dir() or not name.lower().endswith((".dll", "_license")):
                            continue
                        with zf.open(info) as src, open(payload / name, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                (payload / _MARKER).write_text(
                    json.dumps({"version": runtime["version"], "sha256": asset["sha256"]}) + "\n",
                    encoding = "utf-8",
                )
                if target.exists():
                    shutil.rmtree(target)
                os.replace(payload, target)
        installed = self.installed()
        if installed is None:
            raise RyzenAiError(f"The RyzenAI runtime at {target} is incomplete.")
        return installed

    def validate(self, timeout: float = 120.0) -> Optional[str]:
        """None when Studio's Python can load the runtime DLLs, else why not."""
        dll_dir = self.installed()
        if dll_dir is None:
            return "The RyzenAI NPU runtime is not installed."
        code = (
            "import sys; from core.inference.ryzenai_oga import Oga; Oga(sys.argv[1]); print('ok')"
        )
        try:
            done = subprocess.run(
                [sys.executable, "-c", code, str(dll_dir)],
                cwd = str(_BACKEND_DIR),
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = timeout,
                env = utf8_child_env(child_env_without_native_path_secret()),
                **windows_hidden_subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"Checking the RyzenAI NPU runtime failed: {exc}"
        if done.returncode == 0 and done.stdout.strip().endswith("ok"):
            return None
        output = (done.stderr or done.stdout).strip()
        blocked = code_integrity_block_reason(output) or code_integrity_block_reason(
            done.returncode
        )
        if blocked:
            return code_integrity_user_message(str(dll_dir), blocked)
        return f"The RyzenAI NPU runtime did not load: {output[-400:] or done.returncode}"

    def model_dir(self, model_id: str) -> Path:
        return self.models_dir() / model_id[len(ID_PREFIX) :]

    def catalog(self) -> list[RyzenAiModel]:
        rows = []
        for entry in self.pins.get("models") or []:
            model_id = model_id_for(entry["repo"])
            rows.append(
                RyzenAiModel(
                    id = model_id,
                    repo = entry["repo"],
                    size_gb = entry.get("size_gb"),
                    labels = tuple(entry.get("labels") or ("chat",)),
                    downloaded = (self.model_dir(model_id) / _COMPLETE).is_file(),
                )
            )
        return rows

    def model(self, model_id: str) -> RyzenAiModel:
        for row in self.catalog():
            if row.id == model_id:
                return row
        raise RyzenAiError(f"{model_id} is not in the RyzenAI NPU catalog.")

    def download(
        self,
        model_id: str,
        cancel: Optional[threading.Event] = None,
    ) -> Iterator[dict[str, Any]]:
        """Hugging Face download with progress; finished files are kept for a resume."""
        from huggingface_hub import HfApi, hf_hub_download

        cancel = cancel or threading.Event()
        model = self.model(model_id)
        target = self.model_dir(model_id)
        (target / _COMPLETE).unlink(missing_ok = True)
        info = HfApi().model_info(model.repo, files_metadata = True)
        files = [(s.rfilename, s.size or 0) for s in info.siblings or []]
        total = sum(size for _, size in files) or 1
        done = 0
        yield {"event": "progress", "percent": 0}
        for name, size in files:
            if cancel.is_set():
                raise RyzenAiDownloadCancelled(f"Downloading {model_id} was cancelled.")
            hf_hub_download(model.repo, name, revision = info.sha, local_dir = target)
            done += size
            yield {"event": "progress", "percent": min(99, int(done * 100 / total)), "file": name}
        (target / _COMPLETE).write_text(
            json.dumps({"repo": model.repo, "revision": info.sha}) + "\n", encoding = "utf-8"
        )
        yield {"event": "complete", "model": model_id, "percent": 100}

    def delete(self, model_id: str) -> None:
        self.model(model_id)
        shutil.rmtree(self.model_dir(model_id), ignore_errors = True)

    def downloaded_folders(self) -> list[tuple[Path, str]]:
        return [
            (self.model_dir(m.id), m.repo) for m in self.catalog() if self.model_dir(m.id).exists()
        ]


class RyzenAiServer:
    """One ryzenai_server.py child serving one model; same port / api_key / is_alive shape as Lemonade."""

    def __init__(self, dll_dir: Path, model_dir: Path, model_id: str, context_length: int) -> None:
        self.dll_dir, self.model_dir, self.model_id = dll_dir, model_dir, model_id
        self.context_length = context_length
        self.port: Optional[int] = None
        self.api_key: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None
        self._tail: deque[str] = deque(maxlen = 60)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _drain(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                self._tail.append(line.rstrip())
        except Exception:  # noqa: BLE001 -- the pipe closes at teardown
            pass

    def start(
        self,
        timeout: float = 900.0,
        cancel: Optional[threading.Event] = None,
    ) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = int(s.getsockname()[1])
        api_key = secrets.token_urlsafe(32)
        env = utf8_child_env(child_env_without_native_path_secret())
        env["UNSLOTH_RYZENAI_API_KEY"] = api_key
        if is_process_shutting_down():
            raise RyzenAiError("Unsloth is shutting down; not starting the RyzenAI NPU server.")
        cmd = [
            sys.executable, "-m", "core.inference.ryzenai_server",
            "--dll-dir", str(self.dll_dir), "--model-dir", str(self.model_dir),
            "--model-id", self.model_id, "--port", str(port),
            "--context-length", str(self.context_length), "--parent-pid", str(os.getpid()),
        ]  # fmt: skip
        try:
            proc = spawn_on_lifetime_thread(
                lambda: subprocess.Popen(
                    cmd,
                    cwd = str(_BACKEND_DIR),
                    stdout = subprocess.PIPE,
                    stderr = subprocess.STDOUT,
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
            blocked = code_integrity_block_reason(exc)
            if blocked:
                raise RyzenAiError(code_integrity_user_message(sys.executable, blocked)) from exc
            raise RyzenAiError(f"Starting the RyzenAI NPU server failed: {exc}") from exc
        adopt_pid(proc.pid)
        self._process, self.port, self.api_key = proc, port, api_key
        # Shutdown may have swept children before this one was adopted.
        if is_process_shutting_down():
            self.stop()
            raise RyzenAiError("Unsloth is shutting down; not starting the RyzenAI NPU server.")
        threading.Thread(
            target = self._drain, args = (proc,), daemon = True, name = "ryzenai-drain"
        ).start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                self.stop()
                raise RyzenAiError(f"Loading {self.model_id} was cancelled.")
            if not self.is_alive():
                self._raise_exit()
            try:
                response = httpx.get(
                    f"http://127.0.0.1:{port}/health",
                    headers = {"Authorization": f"Bearer {api_key}"},
                    timeout = 2.0,
                    trust_env = False,  # loopback: never through HTTP_PROXY
                )
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        self.stop()
        raise RyzenAiError(f"{self.model_id} did not load on the NPU within {int(timeout)} s.")

    def _raise_exit(self) -> None:
        code = self._process.returncode if self._process is not None else None
        time.sleep(0.2)  # let the drain thread take the last lines
        tail = "\n".join(self._tail)
        self.stop()
        blocked = code_integrity_block_reason(tail) or code_integrity_block_reason(code)
        if blocked:
            raise RyzenAiError(code_integrity_user_message(str(self.dll_dir), blocked))
        last = next((line for line in reversed(tail.splitlines()) if line.strip()), "")
        raise RyzenAiError(
            f"Loading {self.model_id} on the NPU failed: {last or f'exit code {code}'}"
        )

    def stop(self) -> None:
        proc, self._process = self._process, None
        self.port = None
        if proc is None:
            return
        try:
            terminate_pid(proc.pid, timeout = 10.0, owner_verified = True)
            proc.wait(timeout = 10)
        except Exception as exc:  # noqa: BLE001 -- best-effort teardown
            logger.warning("Error stopping the RyzenAI NPU server: %s", exc)
        finally:
            forget_pid(proc.pid)

    close = stop

    def log_tail(self) -> str:
        return "\n".join(self._tail)
