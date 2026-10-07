# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decisions from a GGUF on a private loopback llama-server (``POST /v1/systemone``).

Ported from wasimysaid's native worker in #12770. The server is Studio's own llama.cpp. Clef answers
are rebuilt with the formatter the PyTorch path uses, so the public response does not depend on the
backend (llama.cpp's own confidence is a normalised formula, PyTorch's the winning probability).
GGUF-only models (Kev, lev, Nimble, OpenJev, Laya GGUFs) have no Studio reference, so llama.cpp's
answer is the reference: checked, then passed through.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import httpx

DEFAULT_CTX = 16384
LOAD_TIMEOUT_S = 1800.0
DECIDE_TIMEOUT_S = 300.0
CLOSE_TIMEOUT_S = 10.0
_POLL_S = 0.1
# llama.cpp server-context.cpp wording for inputs over -ub / -c / -b.
_OVERFLOW = (
    re.compile(
        r"input \((\d+) tokens\) is too large to process\. increase the physical batch size "
        r"\(current batch size: (\d+)\)"
    ),
    re.compile(r"input \((\d+) tokens\) is larger than the max context size \((\d+) tokens\)"),
    re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)"),
    re.compile(
        r"the question and its options \((\d+) tokens\) are too large to process\. "
        r"increase the batch size \(current batch size: (\d+)\)"
    ),
)
# server-common.cpp: an image mtmd cannot decode, answered as HTTP 500 by a healthy server.
_UNDECODABLE_MEDIA = "Failed to load image or audio file"
_SERVER_LOG_LINES = 40


class NativeError(RuntimeError):
    """The server failed or is unusable: 503, and the server is retired."""

    status = 503
    error_type = "model_unavailable"
    retire = True


class NativeIncapable(NativeError):
    """This llama-server build cannot serve decision models."""


class NativeInputError(NativeError):
    """The server refused this request; the server itself is fine."""

    status = 422
    error_type = "invalid_request_error"
    retire = False


class NativeContextOverflow(NativeInputError):
    def __init__(self, tokens: int, limit: int):
        self.tokens, self.limit = tokens, limit
        super().__init__(
            f"State and questions are {tokens} tokens; the llama.cpp backend serves at most {limit}. "
            "Shorten the state or use fewer/shorter criteria."
        )


def resolve_binary() -> str | None:
    from core.inference.llama_cpp import LlamaCppBackend
    binary = LlamaCppBackend._find_llama_server_binary()
    return (LlamaCppBackend._exec_path_for_launch(binary) or binary) if binary else None


def request_gap(questions: Mapping[str, Any]) -> str | None:
    """Why llama.cpp cannot take these questions (its parser wants 2 to 10 score levels), else None."""
    for name, question in questions.items():
        if question.get("type") == "score" and len(question.get("criteria") or ()) < 2:
            return f'llama.cpp needs at least two levels for score "{name}".'
    return None


def _reference_formatter() -> Callable[[dict, dict], dict]:
    """``systemone_answer`` from the vendored Clef reference, the function the PyTorch path answers with."""
    if (module := sys.modules.get("_unsloth_clef_reference")) is not None:
        return module.systemone_answer
    spec = importlib.util.find_spec("unsloth")
    roots = [Path(p) for p in (spec.submodule_search_locations or ())] if spec else []
    roots.append(Path(__file__).resolve().parents[4] / "unsloth")
    source = next(
        (
            path
            for root in roots
            if (path := root / "_vendor" / "clef" / "joint_schema_model.py").is_file()
        ),
        None,
    )
    if source is None:
        raise NativeError("The vendored Clef reference (unsloth/_vendor/clef) is missing.")
    # By path: importing the unsloth package would patch transformers in the Studio process.
    module_spec = importlib.util.spec_from_file_location("_unsloth_clef_reference", source)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules["_unsloth_clef_reference"] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("_unsloth_clef_reference", None)
        raise
    return module.systemone_answer


def _probabilities(question: Mapping[str, Any], answer: Any) -> dict[str, float]:
    if not isinstance(answer, Mapping):
        raise ValueError("missing answer")
    kind = question["type"]
    if kind == "noul":
        p = answer.get("noul")
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError("missing noul probability")
        values = {"true": float(p), "false": 1.0 - float(p)}
    else:
        raw = answer.get("probabilities")
        keys = (
            [str(k) for k in question["criteria"]]
            if kind == "choice"
            else [str(i) for i in range(len(question["criteria"]))]
        )
        if not isinstance(raw, Mapping) or set(raw) != set(keys):
            raise ValueError("probabilities do not name the question's options")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw.values()):
            raise ValueError("non-numeric probabilities")
        values = {key: float(raw[key]) for key in keys}
        if not math.isclose(sum(values.values()), 1.0, rel_tol = 1e-4, abs_tol = 1e-4):
            raise ValueError("probabilities do not sum to one")
    if any(not math.isfinite(v) or not -1e-6 <= v <= 1 + 1e-6 for v in values.values()):
        raise ValueError("probabilities out of range")
    return values


def _number(answer: Mapping[str, Any], key: str) -> float:
    value = answer.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"missing {key}")
    return float(value)


def _llama_answer(question: Mapping[str, Any], answer: Any) -> dict[str, Any]:
    """llama.cpp's own answer in Studio's shape, after checking it names the question's options."""
    probabilities = _probabilities(question, answer)
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": probabilities["true"]}
    if kind == "choice":
        choice = answer.get("choice")
        if choice not in probabilities:
            raise ValueError("choice is not one of the options")
        return {
            "type": "choice",
            "choice": choice,
            "confidence": _number(answer, "confidence"),
            "probabilities": probabilities,
        }
    return {
        "type": "score",
        "score": _number(answer, "score"),
        "confidence": _number(answer, "confidence"),
        "legend": {str(i): c for i, c in enumerate(question["criteria"])},
        "probabilities": probabilities,
    }


def normalise(
    data: Any,
    questions: Mapping[str, Mapping[str, Any]],
    clef_answers: bool = True,
) -> dict[str, Any]:
    """Studio's decision result from a llama.cpp /v1/systemone body; Clef answers rebuilt by the PyTorch formatter."""
    if not isinstance(data, Mapping) or not isinstance(data.get("answers"), Mapping):
        raise NativeError("llama.cpp returned an invalid decision response.")
    systemone_answer = _reference_formatter() if clef_answers else None
    answers = {}
    for name, question in questions.items():
        answer = data["answers"].get(name)
        try:
            if systemone_answer is None:
                answers[name] = _llama_answer(question, answer)
            else:
                answers[name] = systemone_answer(dict(question), _probabilities(question, answer))
        except (TypeError, ValueError, KeyError) as exc:
            raise NativeError(f'llama.cpp returned an invalid answer for "{name}": {exc}') from None
    usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else {}
    tokens = usage.get("input_tokens")
    return {
        "answers": answers,
        "input_tokens": int(tokens)
        if isinstance(tokens, int) and not isinstance(tokens, bool)
        else 0,
        # llama.cpp never cuts a state: an input over its batch is refused (NativeContextOverflow).
        "truncated": False,
    }


def _wire_questions(questions: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    wire = {}
    for name, question in questions.items():
        copied = {k: v for k, v in question.items() if v is not None}
        # The reference encoder reads a missing or empty instruction as the question id; llama.cpp refuses both.
        if question.get("instructions") in (None, ""):
            copied["instructions"] = str(name)
        wire[name] = copied
    return wire


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, Mapping):
        value = body.get("error") or body.get("detail") or body.get("message")
        if isinstance(value, Mapping):
            value = value.get("message") or value.get("detail")
        if isinstance(value, str) and value:
            return value[:1000]
    return response.text.strip()[:1000] or f"HTTP {response.status_code}"


def map_error(status: int, detail: str) -> NativeError:
    """The Studio error for a non-2xx llama.cpp answer."""
    if status not in (401, 403):
        for pattern in _OVERFLOW:
            if match := pattern.search(detail):
                return NativeContextOverflow(int(match.group(1)), int(match.group(2)))
        if 400 <= status < 500 or _UNDECODABLE_MEDIA in detail:
            return NativeInputError(detail)
    return NativeError(f"llama.cpp answered HTTP {status}: {detail}")


def _pick_device(binary: str, env: dict[str, str]) -> str:
    from core.inference.llama_cpp import LlamaCppBackend

    devices = LlamaCppBackend._enumerated_gpu_devices(binary, env)
    if not devices:
        raise NativeError("llama.cpp sees no usable GPU under this Studio's device visibility.")
    if len(devices) > 1:
        try:
            free = LlamaCppBackend._get_gpu_free_memory(binary, for_llama_server = True)
        except Exception:
            free = []
        # llama.cpp lists GPUs in CUDA_VISIBLE_DEVICES order; free rows come sorted by physical index.
        by_index = dict(free)
        try:
            order = [int(x) for x in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
        except ValueError:
            order = []
        if len(order) != len(devices):
            order = sorted(by_index)
        if len(order) == len(devices) and all(i in by_index for i in order):
            return devices[max(range(len(order)), key = lambda i: by_index[order[i]])]
    return devices[0]


def _write_key(key: str) -> Path:
    """The server's API key, readable only by this user, under Studio's auth directory."""
    from utils.paths.storage_roots import auth_root

    directory = auth_root()
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / f"decision_llama_api_key_{secrets.token_hex(8)}"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding = "utf-8") as handle:
        handle.write(key)
    return path


class NativeClefAgent:
    """One llama-server serving one decision GGUF; the Decision API keeps it where it keeps a Clef agent."""

    backend = "llama.cpp"

    def __init__(
        self,
        model: Path,
        mmproj: Path | None,
        alias: str,
        *,
        gpu: bool,
        ctx: int = DEFAULT_CTX,
        cancelled: Callable[[], bool] | None = None,
        clef_answers: bool = True,
    ):
        from core.inference.llama_cpp import LlamaCppBackend
        from utils.process_lifetime import (
            adopt_pid,
            child_popen_kwargs,
            is_process_shutting_down,
            spawn_on_lifetime_thread,
        )
        from utils.subprocess_compat import windows_hidden_subprocess_kwargs

        self.alias, self.ctx, self.gpu = alias, int(ctx), gpu
        self.clef_answers = clef_answers
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._client: httpx.Client | None = None
        self.input_modalities: tuple[str, ...] = ()
        binary = resolve_binary()
        if binary is None:
            raise NativeError("llama-server is not installed; Studio installs it with llama.cpp.")
        if not Path(model).is_file():
            raise NativeError(f"The decision GGUF is missing: {model}")
        env = LlamaCppBackend._llama_server_env_for_binary(binary)
        LlamaCppBackend._clear_device_placement_env(env)
        if gpu:
            self.device = _pick_device(binary, env)
            placement = ["-ngl", "-1", "--device", self.device]
        else:
            self.device = "cpu"
            env.update(CUDA_VISIBLE_DEVICES = "", HIP_VISIBLE_DEVICES = "-1", ROCR_VISIBLE_DEVICES = "-1")
            # The AMX CPU variant aborts loading Clef in graph_reserve unless weights stay unrepacked.
            placement = ["-ngl", "0", "--device", "none", "--no-repack"]
            if mmproj is not None:
                placement.append("--no-mmproj-offload")
        self.port = LlamaCppBackend._find_free_port()
        self._key = secrets.token_urlsafe(32)
        self._key_file: Path | None = None
        self._log_path: Path | None = None
        if is_process_shutting_down():
            raise NativeError("Studio is shutting down; llama.cpp was not started.")
        self._key_file = _write_key(self._key)
        self.command = [
            binary,
            "-m",
            str(model),
            *(["--mmproj", str(mmproj)] if mmproj is not None else []),
            "--alias",
            alias,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            # Through a file, not argv: a command line is readable by every process of this user.
            "--api-key-file",
            str(self._key_file),
            "--parallel",
            "1",
            # A decision reads every token in one ubatch: -ub bounds the longest state served.
            "-c",
            str(self.ctx),
            "-b",
            str(self.ctx),
            "-ub",
            str(self.ctx),
            *placement,
        ]
        try:
            self._log_path = self._server_log(self.port)
            log = open(self._log_path, "wb")
        except BaseException:
            self._remove_files()
            raise
        try:
            self._process = spawn_on_lifetime_thread(
                lambda: subprocess.Popen(
                    self.command,
                    stdin = subprocess.DEVNULL,
                    stdout = log,
                    stderr = subprocess.STDOUT,
                    env = env,
                    **windows_hidden_subprocess_kwargs(),
                    **child_popen_kwargs(),
                )
            )
        except BaseException:
            self._remove_files()
            raise
        finally:
            log.close()
        adopt_pid(self._process.pid)
        self._client = httpx.Client(trust_env = False, timeout = DECIDE_TIMEOUT_S)
        try:
            self._wait_ready(cancelled)
            self.input_modalities = self._capabilities()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _server_log(port: int) -> Path:
        import tempfile

        try:
            from utils.paths.storage_roots import studio_root
            root = studio_root() / "logs"
        except Exception:
            root = Path(tempfile.gettempdir())
        root.mkdir(parents = True, exist_ok = True)
        # Per server: a retired one still exiting must not write into its successor's log.
        return root / f"decision-llama-server-{os.getpid()}-{port}.log"

    def _remove_files(self, keep_log: bool = False) -> None:
        for path in (self._key_file, None if keep_log else self._log_path):
            if path is not None:
                try:
                    path.unlink(missing_ok = True)
                except OSError:
                    pass

    def _log_tail(self) -> str:
        if self._log_path is None:
            return ""
        try:
            lines = self._log_path.read_text(encoding = "utf-8", errors = "replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-_SERVER_LOG_LINES:])

    @property
    def _base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}"}

    def _wait_ready(self, cancelled: Callable[[], bool] | None) -> None:
        from utils.process_lifetime import is_process_shutting_down

        deadline = time.monotonic() + LOAD_TIMEOUT_S
        while time.monotonic() < deadline:
            if is_process_shutting_down() or (cancelled is not None and cancelled()):
                raise NativeError("The llama.cpp decision server was stopped while it loaded.")
            if not self.is_alive():
                tail = self._log_tail()
                raise NativeError(
                    f"llama.cpp exited while loading {self.alias} (code {self._process.returncode})"
                    + (f":\n{tail}" if tail else ".")
                )
            try:
                response = self._client.get(f"{self._base}/health", timeout = 2.0)
                if response.status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(_POLL_S)
        raise NativeError(f"llama.cpp did not load {self.alias} within {LOAD_TIMEOUT_S:.0f}s.")

    def _capabilities(self) -> tuple[str, ...]:
        try:
            response = self._client.get(
                f"{self._base}/v1/models", headers = self._headers, timeout = 10.0
            )
            entries = response.json().get("data") if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError, AttributeError):
            entries = None
        for entry in entries if isinstance(entries, list) else ():
            if not isinstance(entry, Mapping):
                continue
            names = {entry.get("id"), *(entry.get("aliases") or ())}
            arch = (
                entry.get("architecture") if isinstance(entry.get("architecture"), Mapping) else {}
            )
            if self.alias in names and "decisions" in (arch.get("output_modalities") or ()):
                return tuple(str(m) for m in arch.get("input_modalities") or ("text",))
        raise NativeIncapable(
            'This llama.cpp build does not serve decision models (no "decisions" output in '
            "/v1/models). Update llama.cpp in Studio."
        )

    @property
    def accepts_images(self) -> bool:
        return "image" in self.input_modalities

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def decide(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        if gap := request_gap(questions):
            raise NativeInputError(gap)
        if images and not self.accepts_images:
            raise NativeInputError(
                f"{self.alias} has no vision projector, so it cannot read images."
            )
        body = {"model": self.alias, "state": state, "questions": _wire_questions(questions)}
        if images:
            body["images"] = list(images)
        with self._lock:
            client = self._client
            if client is None or not self.is_alive():
                raise NativeError("The llama.cpp decision server is not running.")
            try:
                response = client.post(
                    f"{self._base}/v1/systemone", json = body, headers = self._headers
                )
            except httpx.HTTPError as exc:
                alive = "" if self.is_alive() else " and exited"
                raise NativeError(
                    f"The llama.cpp decision server failed{alive}: {type(exc).__name__}"
                ) from None
        if not 200 <= response.status_code < 300:
            raise map_error(response.status_code, _detail(response))
        try:
            data = response.json()
        except ValueError:
            raise NativeError("llama.cpp returned a decision that is not JSON.") from None
        return normalise(data, questions, self.clef_answers)

    def close(self) -> None:
        """Terminate, then kill, then reap; the pid is forgotten only once it is gone."""
        from utils.process_lifetime import forget_pid

        client, self._client = self._client, None
        if client is not None:
            client.close()
        process = self._process
        if process is None:
            self._remove_files()
            return
        # A server that exited by itself keeps its log for diagnosis.
        crashed = process.poll() is not None
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout = CLOSE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout = CLOSE_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is not None:
            forget_pid(process.pid)
            self._remove_files(keep_log = crashed)
