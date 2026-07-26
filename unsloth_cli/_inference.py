# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model loading and streaming shared by `inference` and `chat`."""

import asyncio
import json
import os
import re
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import List, Optional

import typer

_THINK_OPEN = "<think>"
_THINK_BLOCK = re.compile(rf"{re.escape(_THINK_OPEN)}.*?</think>", re.DOTALL)
_STREAMED_ERROR_PREFIX = "Error: "

# Cloudflare (in front of remote Unsloth proxies like RunPod) 403s the default
# "Python-urllib/X.Y" User-Agent as a bot; send a real one on every request.
_USER_AGENT = "unsloth-cli"
_MPI_ENV_PAIRS = (
    ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
    ("PMI_RANK", "PMI_SIZE"),
    ("PMIX_RANK", "PMIX_SIZE"),
    ("MPI_RANK", "MPI_WORLD_SIZE"),
    ("MV2_COMM_WORLD_RANK", "MV2_COMM_WORLD_SIZE"),
)

# Built lazily; urllib stays function-local to match this module.
_no_redirect_opener = None


def urlopen_no_redirect(request, timeout):
    """urlopen that errors on any redirect: following a 3xx would send a bearer
    token (or accept an identity proof) to a base we never vetted, letting a port
    squatter relay a real Unsloth's response."""
    global _no_redirect_opener
    if _no_redirect_opener is None:
        import urllib.error
        import urllib.request

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise urllib.error.HTTPError(
                    req.full_url, code, f"refusing redirect to {newurl}", headers, fp
                )

        _no_redirect_opener = urllib.request.build_opener(_NoRedirect)
    return _no_redirect_opener.open(request, timeout = timeout)


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


def _parse_nonnegative_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _first_mpi_env_pair() -> tuple[Optional[int], Optional[int]]:
    for rank_name, size_name in _MPI_ENV_PAIRS:
        rank = _parse_nonnegative_int(os.environ.get(rank_name))
        world_size = _parse_nonnegative_int(os.environ.get(size_name))
        if rank is not None and world_size is not None and world_size > 1 and rank < world_size:
            return rank, world_size
    return None, None


def _json_rank_count_from_env(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        if value.lstrip().startswith(("[", "{")):
            data = json.loads(value)
        else:
            with open(value, "r") as f:
                data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and isinstance(data.get("hosts"), list):
        return len(data["hosts"])
    return None


def mlx_distributed_info() -> tuple[bool, int, Optional[int]]:
    """Return launch-context metadata without initializing MLX distributed."""
    rank = _parse_nonnegative_int(os.environ.get("MLX_RANK"))
    world_size = _parse_nonnegative_int(os.environ.get("MLX_WORLD_SIZE"))
    if rank is not None:
        if (
            world_size is not None
            and world_size > 1
            and rank < world_size
            and os.environ.get("NCCL_HOST_IP")
            and os.environ.get("NCCL_PORT")
        ):
            return True, rank, world_size
        inferred_size = _json_rank_count_from_env("MLX_HOSTFILE")
        if inferred_size is not None and inferred_size > 1 and rank < inferred_size:
            return True, rank, inferred_size
        inferred_size = _json_rank_count_from_env("MLX_IBV_DEVICES")
        if (
            inferred_size is not None
            and inferred_size > 1
            and rank < inferred_size
            and os.environ.get("MLX_JACCL_COORDINATOR")
        ):
            return True, rank, inferred_size
        return False, 0, None

    mpi_rank, mpi_world_size = _first_mpi_env_pair()
    return mpi_rank is not None, mpi_rank or 0, mpi_world_size


def mlx_distributed_uses_mpi() -> bool:
    """Whether the current distributed context was launched through MPI."""
    return (
        _parse_nonnegative_int(os.environ.get("MLX_RANK")) is None
        and _first_mpi_env_pair()[0] is not None
    )


@contextmanager
def quiet_if_nonzero_mlx_rank():
    """Silence parent and child-process stdout/stderr on nonzero ranks."""
    if mlx_distributed_info()[1] == 0:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    with open(os.devnull, "w") as devnull:
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


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


def raise_on_streamed_error(stream):
    # Match real backend errors by type (GenStreamError), not the "Error:" text
    # prefix, so a completion whose text opens with "Error:" is not misread as a
    # backend failure.
    try:
        ensure_studio_backend_path()
        from core.inference.orchestrator import GenStreamError
    except Exception:
        GenStreamError = None
    for chunk in stream:
        if GenStreamError is not None and isinstance(chunk, GenStreamError):
            raise RuntimeError(str(chunk)[len(_STREAMED_ERROR_PREFIX) :].strip() or "Unknown error")
        yield chunk


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

    def share_distributed_object(
        self,
        obj,
        *,
        timeout = 300.0,
    ):
        if self._kind != "unsloth" or not hasattr(self._backend, "share_distributed_object"):
            raise RuntimeError(
                "Distributed MLX chat requires the Unsloth MLX backend; "
                f"backend '{self._kind}' cannot broadcast chat turns."
            )
        return self._backend.share_distributed_object(obj, timeout = timeout)


def resolve_model_config(model: str, *, hf_token: Optional[str]):
    ensure_studio_backend_path()
    from utils.models import ModelConfig

    model_config = ModelConfig.from_identifier(model_id = model, hf_token = hf_token)
    if not model_config:
        typer.echo("Could not resolve model config", err = True)
        raise typer.Exit(code = 1)
    return model_config


def _validate_llama_extra_args_or_exit(llama_extra_args: Optional[List[str]]) -> list[str]:
    from core.inference.llama_server_args import validate_extra_args
    try:
        return validate_extra_args(llama_extra_args)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err = True)
        raise typer.Exit(code = 1)


def _load_gguf_backend(
    model_config,
    *,
    hf_token,
    max_seq_length,
    tensor_parallel: bool = False,
    llama_extra_args: Optional[List[str]] = None,
):
    ensure_studio_backend_path()
    from core.inference.llama_cpp import LlamaCppBackend
    from core.inference.tensor_fallback import load_with_tensor_fallback

    llama_backend = LlamaCppBackend()
    extra_args = _validate_llama_extra_args_or_exit(llama_extra_args)
    common = dict(
        hf_variant = model_config.gguf_variant,
        model_identifier = model_config.identifier,
        is_vision = model_config.is_vision,
        n_ctx = max_seq_length,
    )

    async def _attempt_gguf_load(
        requested_tensor_parallel: bool, attempt_extra_args: Optional[List[str]]
    ) -> bool:
        attempt_common = dict(
            common,
            tensor_parallel = requested_tensor_parallel,
            extra_args = attempt_extra_args,
        )
        if model_config.gguf_hf_repo:
            return llama_backend.load_model(
                hf_repo = model_config.gguf_hf_repo,
                hf_token = hf_token,
                **attempt_common,
            )
        return llama_backend.load_model(
            gguf_path = model_config.gguf_file,
            mmproj_path = model_config.gguf_mmproj_file,
            mtp_draft_path = model_config.gguf_mtp_file,
            **attempt_common,
        )

    loaded = asyncio.run(
        load_with_tensor_fallback(
            _attempt_gguf_load,
            requested_tensor = tensor_parallel,
            extra_args = extra_args,
            label = model_config.identifier,
        )
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
    tensor_parallel: bool = False,
    llama_extra_args: Optional[List[str]] = None,
    model_config = None,
    fresh_backend: bool = False,
):
    """Load `model` in-process: GGUF via llama-server, else the orchestrator.

    fresh_backend uses a private orchestrator so a second model (compare's
    base column) can run alongside the main one.
    """
    with quiet_if_nonzero_mlx_rank():
        is_mlx_distributed, rank, _world_size = mlx_distributed_info()
        if model_config is None:
            model_config = resolve_model_config(model, hf_token = hf_token)

        if is_mlx_distributed and model_config.is_gguf:
            if rank == 0:
                typer.echo(
                    "Distributed MLX inference does not support GGUF/llama.cpp models. "
                    "Use a non-GGUF MLX model under mlx.launch, or run GGUF without "
                    "mlx.launch.",
                    err = True,
                )
            raise typer.Exit(code = 1)

        if rank == 0:
            typer.echo(f"Loading {model}", err = True)

        if model_config.is_gguf:
            return _load_gguf_backend(
                model_config,
                hf_token = hf_token,
                max_seq_length = max_seq_length,
                tensor_parallel = tensor_parallel,
                llama_extra_args = llama_extra_args,
            )

        if fresh_backend:
            ensure_studio_backend_path()
            from core.inference import InferenceOrchestrator
            backend = InferenceOrchestrator()
        else:
            ensure_studio_backend_path()
            from core.inference import get_inference_backend
            backend = get_inference_backend()
        try:
            loaded = backend.load_model(
                config = model_config,
                max_seq_length = max_seq_length,
                load_in_4bit = load_in_4bit,
                hf_token = hf_token,
                tensor_parallel = tensor_parallel,
                mlx_distributed = is_mlx_distributed,
            )
        except Exception as exc:
            if not is_mlx_distributed:
                raise
            if rank == 0:
                typer.echo(str(exc) or "Model load failed", err = True)
            raise typer.Exit(code = 1)
        if not loaded:
            typer.echo("Model load failed", err = True)
            raise typer.Exit(code = 1)
    return ChatBackend("unsloth", backend)


def _loopback_candidate_bases(base: str) -> list:
    """For a bare ``localhost`` base, the concrete IP bases to try, IPv4
    127.0.0.1 first (where ``unsloth studio`` binds by default). Pinning to one
    address up front means discovery, the identity check, and the credential we
    then send all target the same endpoint instead of racing IPv4/IPv6
    resolution -- which would otherwise let the health probe land on one address
    and the identity check on another. A literal IP or remote name is unchanged.
    """
    from urllib.parse import urlparse

    parsed = urlparse(base)
    if (parsed.hostname or "").lower() != "localhost":
        return [base]
    import socket

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        ips = {
            ai[4][0] for ai in socket.getaddrinfo(parsed.hostname, port, type = socket.SOCK_STREAM)
        }
    except Exception:
        return [base]
    ordered = sorted(ips, key = lambda ip: (ip != "127.0.0.1", ip))
    bases = [
        f"{parsed.scheme}://" + (f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}")
        for ip in ordered
    ]
    return bases or [base]


def find_studio_server(timeout: float = 3.0) -> Optional[str]:
    import urllib.request

    base = os.environ.get("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888").rstrip("/")
    # Try the concrete loopback addresses in order and return the first that
    # answers, so the rest of the flow talks to that exact address.
    for candidate in _loopback_candidate_bases(base):
        request = urllib.request.Request(
            f"{candidate}/api/health", headers = {"User-Agent": _USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout = timeout):
                return candidate
        except Exception:
            continue
    return None


def is_loopback_url(base: str) -> bool:
    """True only when *base* resolves to loopback. find_studio_server() trusts a
    base after only a health probe, so credentials are auto-sent only to loopback
    (a local Unsloth or an SSH tunnel on 127.0.0.1), the targets the auto flows mean."""
    from urllib.parse import urlparse

    host = (urlparse(base).hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        import ipaddress
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def verify_studio_identity(base: str, timeout: float = 3.0) -> bool:
    """Confirm `base` is really this machine's Unsloth before sending a secret.

    Send a random nonce to /api/auth/identity and check the returned HMAC against
    the one computed from the local same-user secret; an endpoint without that
    secret (port squatter, remote/fake) can't match. Fails closed on any error."""
    import base64
    import hmac as _hmac
    import json
    import secrets as _secrets
    import socket
    import urllib.request
    from urllib.parse import urlparse

    try:
        import studio.backend.core  # noqa: F401  puts studio/backend on sys.path
        from studio.backend.auth import storage
    except Exception:
        return False

    parsed = urlparse(base)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    # Resolve to one concrete address and talk to *that* address, then bind the
    # proof to (address, port). A name like localhost can resolve to a squatter on
    # ::1 while the real Unsloth is on 127.0.0.1; connecting to the resolved IP and
    # binding to it means a proof relayed from a different address/port won't match.
    try:
        ip = socket.getaddrinfo(host, port, type = socket.SOCK_STREAM)[0][4][0]
    except Exception:
        return False
    netloc = f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}"
    nonce = _secrets.token_bytes(32)
    query = base64.urlsafe_b64encode(nonce).decode()
    request = urllib.request.Request(
        f"{parsed.scheme}://{netloc}/api/auth/identity?nonce={query}",
        headers = {"User-Agent": _USER_AGENT, "Host": parsed.netloc},
    )
    try:
        # No redirects: a 302 could relay a real Unsloth's proof (see urlopen_no_redirect).
        # Cap the read: the server is still unverified, so don't trust its length.
        with urlopen_no_redirect(request, timeout = timeout) as response:
            proof = json.loads(response.read(65536).decode() or "{}").get("proof")
    except Exception:
        return False
    if not isinstance(proof, str):
        return False
    try:
        expected = storage.compute_identity_proof(nonce, ip, port)
    except Exception:
        return False
    return _hmac.compare_digest(proof, expected)


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
    """Chat against a running Unsloth server over its OpenAI-compatible API.

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
        # No redirects: this carries a bearer token (see urlopen_no_redirect).
        return urlopen_no_redirect(request, timeout = timeout)

    def ensure_loaded(
        self,
        model: str,
        *,
        hf_token,
        max_seq_length,
        load_in_4bit,
        tensor_parallel: bool = False,
        llama_extra_args: Optional[List[str]] = None,
    ) -> None:
        typer.echo(f"Loading {model} on the Unsloth server", err = True)
        payload = {
            "model_path": model,
            "hf_token": hf_token,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "tensor_parallel": tensor_parallel,
        }
        if llama_extra_args:
            payload["llama_extra_args"] = llama_extra_args
        try:
            self._request(
                "POST",
                "/api/inference/load",
                payload,
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


def connect_studio_server(
    model: str,
    *,
    hf_token,
    max_seq_length,
    load_in_4bit,
    tensor_parallel: bool = False,
    llama_extra_args: Optional[List[str]] = None,
):
    """Backend on a running Unsloth server, or None (caller loads locally)."""
    base_url = find_studio_server()
    if not base_url:
        return None

    # Explicit server (UNSLOTH_STUDIO_URL) we can't safely attach to -> fail loudly;
    # opportunistic local discovery just falls back to a local load.
    explicit = bool(os.environ.get("UNSLOTH_STUDIO_URL"))

    def _refuse(reason: str):
        if not explicit:
            return None
        typer.echo(
            f"Can't attach to the Unsloth server at {base_url}: {reason} Run Unsloth "
            "on this machine, or unset UNSLOTH_STUDIO_URL to load the model locally.",
            err = True,
        )
        raise typer.Exit(code = 1)

    # Only hand the self-issued JWT (signed with the local secret) to loopback: a
    # remote URL is unverified and a real remote Unsloth would reject it anyway.
    if not is_loopback_url(base_url):
        return _refuse(
            "it isn't a local Unsloth, so a self-issued token can't "
            "authenticate to it and must not be sent to it."
        )
    # Confirm the loopback responder is really our Unsloth (not a port squatter).
    if not verify_studio_identity(base_url):
        return _refuse(
            "its identity couldn't be verified (it may be running as a "
            "different OS user, or another process took the port)."
        )
    token = _studio_token()
    if not token:
        return _refuse("couldn't self-issue an Unsloth token (is Unsloth set up here?).")
    backend = HttpChatBackend(base_url, token)
    backend.ensure_loaded(
        model,
        hf_token = hf_token,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        tensor_parallel = tensor_parallel,
        llama_extra_args = llama_extra_args,
    )
    return backend
