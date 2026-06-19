# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model loading and streaming shared by `inference` and `chat`."""

import os
import re
import sys
from pathlib import Path
from typing import Optional

import typer

_THINK_OPEN = "<think>"
_THINK_BLOCK = re.compile(rf"{re.escape(_THINK_OPEN)}.*?</think>", re.DOTALL)

# Cloudflare (in front of remote Studio proxies like RunPod) 403s the default
# "Python-urllib/X.Y" User-Agent as a bot; send a real one on every request.
_USER_AGENT = "unsloth-cli"


def ensure_studio_backend_path() -> None:
    backend_dir = str(Path(__file__).resolve().parents[1] / "studio" / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)


def configure_quiet_logging() -> None:
    import logging

    import structlog

    # The CLI never configures structlog, so without this every backend INFO
    # line prints. LOG_LEVEL is exported so the worker subprocess inherits it.
    level_name = os.environ.setdefault("LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    structlog.configure(wrapper_class = structlog.make_filtering_bound_logger(level))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def visible_text(text: str, show_thinking: bool) -> str:
    if show_thinking:
        return text
    text = _THINK_BLOCK.sub("", text)
    # Hold back an unclosed trailing <think> so reasoning never leaks mid-stream.
    open_idx = text.find(_THINK_OPEN)
    if open_idx != -1:
        text = text[:open_idx]
    max_prefix = min(len(text), len(_THINK_OPEN) - 1)
    for size in range(max_prefix, 0, -1):
        if _THINK_OPEN.startswith(text[-size:]):
            return text[:-size]
    return text


def stream_to_stdout(stream, show_thinking: bool) -> str:
    # Backends yield the full text-so-far on each step (llama.cpp ends with a
    # metadata dict, skipped); print the growing tail, return the raw text.
    raw = ""
    shown = ""
    for chunk in stream:
        if not isinstance(chunk, str):
            continue
        raw = chunk
        rendered = visible_text(chunk, show_thinking)
        delta = rendered[len(shown) :]
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
        shown = rendered
    sys.stdout.write("\n")
    sys.stdout.flush()
    return raw


def stream_markdown(stream, show_thinking: bool, *, console) -> str:
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.text import Text

    raw = ""
    with Live(console = console, refresh_per_second = 12, vertical_overflow = "visible") as live:
        for chunk in stream:
            if not isinstance(chunk, str):
                continue
            raw = chunk
            visible = visible_text(chunk, show_thinking)
            live.update(Markdown(visible) if visible.strip() else Text(""))
    return raw


def collect_stream(stream, show_thinking: bool) -> str:
    raw = ""
    for chunk in stream:
        if isinstance(chunk, str):
            raw = chunk
    return visible_text(raw, show_thinking)


def render_columns(
    left_label: str,
    left_text: str,
    right_label: str,
    right_text: str,
    *,
    console = None,
) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    table = Table(box = box.MINIMAL, expand = True, padding = (0, 1), pad_edge = False)
    table.add_column(left_label, header_style = "bold yellow", ratio = 1, overflow = "fold")
    table.add_column(right_label, header_style = "bold magenta", ratio = 1, overflow = "fold")
    table.add_row(left_text or "", right_text or "")
    (console or Console()).print(table)


class ChatBackend:
    """Uniform stream()/close() over the llama-server and Unsloth backends."""

    def __init__(self, kind: str, backend) -> None:
        self._kind = kind  # "gguf" | "unsloth"
        self._backend = backend

    def stream(
        self,
        messages: list,
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        repetition_penalty: float,
        enable_thinking: bool,
        use_adapter: Optional[bool] = None,
    ):
        if self._kind == "gguf":
            # llama-server takes the system prompt as the first message.
            msgs = list(messages)
            if system_prompt:
                msgs = [{"role": "system", "content": system_prompt}, *msgs]
            return self._backend.generate_chat_completion(
                messages = msgs,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                max_tokens = max_new_tokens,
                repetition_penalty = repetition_penalty,
                enable_thinking = enable_thinking,
            )
        gen_kwargs = dict(
            messages = messages,
            system_prompt = system_prompt,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            max_new_tokens = max_new_tokens,
            repetition_penalty = repetition_penalty,
            enable_thinking = enable_thinking,
        )
        if use_adapter is not None:
            return self._backend.generate_with_adapter_control(
                use_adapter = use_adapter, **gen_kwargs
            )
        return self._backend.generate_chat_response(**gen_kwargs)

    def close(self) -> None:
        # Shut the worker down directly: the graceful unload_model waits for
        # an ack that compare mode can swallow, hanging exit for minutes.
        try:
            if self._kind == "gguf":
                self._backend.unload_model()
            else:
                self._backend._shutdown_subprocess(timeout = 2.0)
        except Exception:
            pass


def resolve_model_config(model: str, *, hf_token: Optional[str]):
    ensure_studio_backend_path()
    from utils.models import ModelConfig

    model_config = ModelConfig.from_identifier(model_id = model, hf_token = hf_token)
    if not model_config:
        typer.echo("Could not resolve model config", err = True)
        raise typer.Exit(code = 1)
    return model_config


def _load_gguf_backend(model_config, *, hf_token, max_seq_length):
    ensure_studio_backend_path()
    from core.inference.llama_cpp import LlamaCppBackend

    llama_backend = LlamaCppBackend()
    common = dict(
        hf_variant = model_config.gguf_variant,
        model_identifier = model_config.identifier,
        is_vision = model_config.is_vision,
        n_ctx = max_seq_length,
    )
    if model_config.gguf_hf_repo:
        loaded = llama_backend.load_model(
            hf_repo = model_config.gguf_hf_repo, hf_token = hf_token, **common
        )
    else:
        loaded = llama_backend.load_model(
            gguf_path = model_config.gguf_file,
            mmproj_path = model_config.gguf_mmproj_file,
            mtp_draft_path = model_config.gguf_mtp_file,
            **common,
        )
    if not loaded:
        typer.echo("Model load failed", err = True)
        raise typer.Exit(code = 1)
    return ChatBackend("gguf", llama_backend)


def load_chat_backend(
    model: str,
    *,
    hf_token: Optional[str],
    max_seq_length: int,
    load_in_4bit: bool,
    model_config = None,
    fresh_backend: bool = False,
):
    """Load `model` in-process: GGUF via llama-server, else the orchestrator.

    fresh_backend uses a private orchestrator so a second model (compare's
    base column) can run alongside the main one.
    """
    if model_config is None:
        model_config = resolve_model_config(model, hf_token = hf_token)

    typer.echo(f"Loading {model}", err = True)

    if model_config.is_gguf:
        return _load_gguf_backend(model_config, hf_token = hf_token, max_seq_length = max_seq_length)

    if fresh_backend:
        ensure_studio_backend_path()
        from core.inference import InferenceOrchestrator
        backend = InferenceOrchestrator()
    else:
        ensure_studio_backend_path()
        from core.inference import get_inference_backend
        backend = get_inference_backend()
    if not backend.load_model(
        config = model_config,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        hf_token = hf_token,
    ):
        typer.echo("Model load failed", err = True)
        raise typer.Exit(code = 1)
    return ChatBackend("unsloth", backend)


_STUDIO_INSTALL_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_HEALTH_BODY_LIMIT = 64 * 1024


def _local_studio_install_id() -> Optional[str]:
    try:
        ensure_studio_backend_path()
        from utils.paths import studio_root
        token = (studio_root() / "share" / "studio_install_id").read_text(encoding = "utf-8").strip()
    except Exception:
        return None
    return token if _STUDIO_INSTALL_ID_RE.fullmatch(token) else None


def _is_studio_health(body) -> bool:
    return (
        isinstance(body, dict)
        and body.get("status") == "healthy"
        and body.get("service") == "Unsloth UI Backend"
        and body.get("supports_desktop_auth") is True
    )


def _is_loopback(host: str) -> bool:
    import ipaddress
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _same_origin(a: str, b: str) -> bool:
    from urllib.parse import urlparse
    pa, pb = urlparse(a), urlparse(b)
    return (pa.scheme, pa.hostname, pa.port) == (pb.scheme, pb.hostname, pb.port)


def find_studio_server(timeout: float = 3.0) -> Optional[str]:
    import json
    import urllib.request
    from urllib.parse import urlparse

    base = (os.environ.get("UNSLOTH_STUDIO_URL") or "http://127.0.0.1:8888").rstrip("/")
    request = urllib.request.Request(f"{base}/api/health", headers = {"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout = timeout) as response:
            final_url = response.geturl()
            body = json.loads(response.read(_HEALTH_BODY_LIMIT).decode("utf-8", "replace"))
    except Exception:
        return None
    if not _is_studio_health(body):
        return None
    if _is_loopback(urlparse(base).hostname or ""):
        # A squatter on this port could 302 the probe to the real Studio so the
        # id check passes, yet we'd still send credentials back to the squatter.
        if not _same_origin(final_url, base):
            return None
        install_id = _local_studio_install_id()
        if install_id is not None and body.get("studio_root_id") != install_id:
            return None
    return base


def _studio_token() -> Optional[str]:
    """Self-issue a JWT: the CLI runs as the same OS user as the server, so it
    signs with the same stored secret the server validates against."""
    try:
        import studio.backend.core  # noqa: F401  puts studio/backend on sys.path

        from studio.backend.auth import storage
        from studio.backend.auth.authentication import create_access_token

        row = storage.get_connection().execute("SELECT username FROM auth_user LIMIT 1").fetchone()
        return create_access_token(row[0], desktop = True) if row else None
    except Exception:
        return None


class HttpChatBackend:
    """Chat against a running Studio server over its OpenAI-compatible API.

    close() leaves the model loaded on purpose — the next session (or the
    UI) starts instantly.
    """

    def __init__(self, base_url: str, token: str) -> None:
        self._base = base_url
        self._token = token

    def _request(
        self,
        method: str,
        path: str,
        payload = None,
        timeout = None,
    ):
        import json
        import urllib.request

        request = urllib.request.Request(
            self._base + path,
            data = None if payload is None else json.dumps(payload).encode(),
            headers = {
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            },
            method = method,
        )
        return urllib.request.urlopen(request, timeout = timeout)

    def ensure_loaded(self, model: str, *, hf_token, max_seq_length, load_in_4bit) -> None:
        typer.echo(f"Loading {model} on the Studio server", err = True)
        try:
            self._request(
                "POST",
                "/api/inference/load",
                {
                    "model_path": model,
                    "hf_token": hf_token,
                    "max_seq_length": max_seq_length,
                    "load_in_4bit": load_in_4bit,
                },
            ).close()
        except Exception as exc:
            typer.echo(f"Model load failed: {exc}", err = True)
            raise typer.Exit(code = 1)

    def stream(
        self,
        messages: list,
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        repetition_penalty: float,
        enable_thinking: bool,
        use_adapter: Optional[bool] = None,
    ):
        import json

        msgs = list(messages)
        if system_prompt:
            msgs = [{"role": "system", "content": system_prompt}, *msgs]
        resp = self._request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "default",
                "messages": msgs,
                "stream": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "enable_thinking": enable_thinking,
            },
        )

        def cumulative():
            # Accumulate SSE deltas into the full-text-so-far convention the
            # stream helpers expect.
            text = ""
            with resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except ValueError:
                        continue
                    if "error" in parsed:
                        raise RuntimeError(
                            f"Server error: {parsed['error'].get('message', 'Unknown server error')}"
                        )
                    try:
                        delta = parsed["choices"][0]["delta"].get("content")
                    except (KeyError, IndexError):
                        continue
                    if not delta:
                        continue
                    text += delta
                    # An emoji can arrive split across two deltas as lone
                    # surrogate halves: hold back a trailing half, merge pairs.
                    visible = text
                    if "\ud800" <= visible[-1] <= "\udbff":
                        visible = visible[:-1]
                    yield visible.encode("utf-16", "surrogatepass").decode("utf-16", "replace")

        return cumulative()

    def close(self) -> None:
        pass


def connect_studio_server(model: str, *, hf_token, max_seq_length, load_in_4bit):
    """Backend on a running Studio server, or None (caller loads locally)."""
    base_url = find_studio_server()
    if not base_url:
        return None
    token = _studio_token()
    if not token:
        return None
    backend = HttpChatBackend(base_url, token)
    backend.ensure_loaded(
        model, hf_token = hf_token, max_seq_length = max_seq_length, load_in_4bit = load_in_4bit
    )
    return backend
