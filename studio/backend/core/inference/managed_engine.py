# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""HTTP server lifecycle for optional text inference engines."""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import threading
import time
from collections import deque

import httpx

from .engine_install import engine_lease, installed, profile, support_reason


from .engine_adapters import ADAPTERS, gpu_memory_fraction, launch_arguments


def validate_load(engine: str, request) -> None:
    profile(engine)
    gpu_ids = request.gpu_ids or [0]
    if len(set(gpu_ids)) != len(gpu_ids) or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("Select distinct, non-negative GPU indices.")
    for gpu_id in gpu_ids:
        reason = support_reason(engine, gpu_id)
        if reason:
            raise ValueError(f"GPU {gpu_id}: {reason}")
    if not installed(engine):
        raise ValueError(f"Install {engine} in Settings > System > Inference engines first.")
    if request.gguf_variant or request.model_path.lower().endswith(".gguf") or request.is_lora:
        raise ValueError(
            "Optional engines currently support full text checkpoints in safetensors format."
        )
    if request.trust_remote_code or request.chat_template_override:
        raise ValueError(
            "Optional engines do not yet support custom model code or template overrides."
        )


def validate_model(
    config,
    hf_token = None,
    gpu_ids = None,
) -> None:
    """Check plain metadata before releasing the previous resident model."""
    from pathlib import Path

    if config.is_local:
        path = Path(config.path) / "config.json"
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(config.identifier, "config.json", token = hf_token))
    metadata = json.loads(path.read_text(encoding = "utf-8"))
    if metadata.get("model_type") not in {"llama", "mistral", "qwen2", "qwen3"}:
        raise ValueError(
            "This engine profile currently supports Llama, Mistral, Qwen2 and Qwen3 text checkpoints."
        )
    if metadata.get("quantization_config") or metadata.get("vision_config"):
        raise ValueError(
            "Choose a full precision text checkpoint for this optional engine profile."
        )
    size = len(gpu_ids or [0])
    if size > 1:
        for field in ("num_attention_heads", "hidden_size", "intermediate_size"):
            value = metadata.get(field)
            if isinstance(value, int) and value > 0 and value % size:
                raise ValueError(
                    f"This model's {field} ({value}) cannot be split across {size} GPUs. "
                    "Select a GPU count that divides it evenly."
                )
        kv_heads = metadata.get("num_key_value_heads", metadata.get("num_attention_heads"))
        if isinstance(kv_heads, int) and kv_heads > 0:
            # Both engines replicate KV heads when there are more ranks than
            # KV heads; otherwise each rank receives an equal number of heads.
            if max(kv_heads, size) % min(kv_heads, size):
                raise ValueError(
                    f"This model's {kv_heads} KV heads are incompatible with {size} GPUs. "
                    "Select a GPU count that divides the KV heads or is a multiple of them."
                )


class ManagedEngine:
    def __init__(self, engine: str):
        self.engine = engine
        self.adapter = ADAPTERS[engine]
        self.process = None
        self.model = None
        self.context = 0
        self.phase = "starting"
        self._cancel = threading.Event()
        self._lock = threading.RLock()
        self._lease = None
        self._reader = None
        self._tail = deque(maxlen = 30)
        self.base_url = ""
        self.key = secrets.token_urlsafe(32)

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(
        self,
        model: str,
        context: int,
        gpu_ids,
        env: dict,
        cancel_event = None,
    ):
        from utils.process_lifetime import (
            adopt_pid,
            child_popen_kwargs,
            spawn_on_lifetime_thread,
            is_process_shutting_down,
        )

        with self._lock:
            if (
                is_process_shutting_down()
                or self._cancel.is_set()
                or (cancel_event is not None and cancel_event.is_set())
            ):
                raise RuntimeError("Model load cancelled")
            self._lease = engine_lease(self.engine)
            self._lease.__enter__()
            try:
                info = installed(self.engine)
                if info is None:
                    raise RuntimeError("The selected engine is no longer installed.")
                # SGLang derives an auxiliary port by adding 10000 to HTTP.
                for _ in range(100):
                    with socket.socket() as sock:
                        sock.bind(("127.0.0.1", 0))
                        port = sock.getsockname()[1]
                    if port <= 55535:
                        break
                else:
                    raise RuntimeError("Could not allocate an inference server port.")
                self.base_url = f"http://127.0.0.1:{port}"
                self.model, self.context = model, context or 4096
                child_env = {
                    k: v
                    for k, v in env.items()
                    if not k.startswith(("PYTHON", "UV_", "PIP_", "SGLANG_", "VLLM_"))
                }
                from utils.native_path_leases import child_env_without_native_path_secret

                child_env = child_env_without_native_path_secret(child_env)
                child_env.pop("VIRTUAL_ENV", None)
                child_env.pop("LD_PRELOAD", None)
                child_env.pop("LD_LIBRARY_PATH", None)
                child_env["PATH"] = (
                    info["path"] + "/bin" + os.pathsep + child_env.get("PATH", os.defpath)
                )
                child_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in (gpu_ids or [0]))
                child_env["PYTHONNOUSERSITE"] = "1"
                from .engine_install import engine_root

                child_env["VLLM_CACHE_ROOT"] = str(engine_root() / self.engine / "cache")
                child_env["TORCHINDUCTOR_CACHE_DIR"] = str(
                    engine_root() / self.engine / "cache" / "inductor"
                )
                child_env["TRITON_CACHE_DIR"] = str(
                    engine_root() / self.engine / "cache" / "triton"
                )
                memory_fraction = gpu_memory_fraction(gpu_ids or [0])
                child_env.update(self.adapter.environment(len(gpu_ids or [0])))
                self.process = spawn_on_lifetime_thread(
                    lambda: subprocess.Popen(
                        self.adapter.command(
                            info["path"] + "/bin/python",
                            model,
                            port,
                            self.key,
                            self.context,
                            memory_fraction,
                            len(gpu_ids or [0]),
                        ),
                        env = child_env,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.STDOUT,
                        text = True,
                        encoding = "utf-8",
                        errors = "replace",
                        start_new_session = True,
                        **child_popen_kwargs(),
                    )
                )
                adopt_pid(self.process.pid)
                if is_process_shutting_down():
                    raise RuntimeError("Studio is shutting down")
                self._reader = threading.Thread(
                    target = self._drain, args = (self.process,), daemon = True
                )
                self._reader.start()
            except Exception:
                self.stop()
                raise
        deadline = time.monotonic() + 900
        try:
            with httpx.Client(trust_env = False, timeout = 2) as client:
                while time.monotonic() < deadline:
                    if self._cancel.is_set() or (
                        cancel_event is not None and cancel_event.is_set()
                    ):
                        raise RuntimeError("Model load cancelled")
                    if not self.alive():
                        raise RuntimeError("Engine failed to start. " + "\n".join(self._tail))
                    try:
                        response = client.get(self.base_url + "/health", headers = self.headers)
                        if response.status_code == 200:
                            self.phase = "ready"
                            return
                    except httpx.HTTPError:
                        pass
                    self._cancel.wait(0.25)
                raise RuntimeError(
                    "Engine startup timed out. Try a smaller model or context length."
                )
        except Exception:
            self.stop()
            raise

    @property
    def headers(self):
        return {"Authorization": "Bearer " + self.key}

    def _drain(self, proc):
        from utils.log_redaction import redact_log_text
        from utils.native_path_leases import redact_native_paths
        for line in proc.stdout:
            stage = self.adapter.progress(line)
            if stage and self.phase != "ready":
                self.phase = stage
            self._tail.append(
                redact_native_paths(redact_log_text(line.replace(self.key, "[redacted]"))).strip()[
                    -1000:
                ]
            )

    def stop(self) -> bool:
        from utils.process_lifetime import terminate_pid, forget_pid

        self._cancel.set()
        with self._lock:
            if self.process is not None:
                terminate_pid(self.process.pid, timeout = 5, owner_verified = True)
                try:
                    self.process.wait(timeout = 5)
                except subprocess.TimeoutExpired:
                    return False
                forget_pid(self.process.pid)
                if self._reader:
                    self._reader.join(timeout = 2)
                self.process.stdout.close()
                self.process = None
            if self._lease is not None:
                self._lease.__exit__(None, None, None)
                self._lease = None
        return True

    def count_tokens(
        self,
        messages,
        system_prompt = "",
        **kwargs,
    ):
        if kwargs.get("tools"):
            raise ValueError("Tools are unavailable for this optional engine profile.")
        if not self.adapter.exact_token_count:
            raise RuntimeError("Exact prompt token counting is unavailable for this engine.")
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        with httpx.Client(trust_env = False, timeout = 30) as client:
            response = client.post(
                self.base_url + "/tokenize",
                headers = self.headers,
                json = {
                    "model": self.model,
                    "messages": messages,
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response.raise_for_status()
            return int(response.json()["count"]), self.model

    def generate(
        self,
        *,
        messages,
        system_prompt = "",
        cancel_event = None,
        stats_holder = None,
        **params,
    ):
        if params.get("tools") or params.get("use_adapter") is not None:
            raise ValueError(
                "Tools and adapter comparisons are not supported by this optional engine profile."
            )
        if params.get("image") or params.get("images") or params.get("video"):
            raise ValueError("This optional engine profile supports text input only.")
        if (
            params.get("continue_final_message")
            or params.get("enable_thinking")
            or params.get("preserve_thinking")
        ):
            raise ValueError(
                "Continuation and reasoning controls are unavailable for this engine profile."
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if system_prompt:
            payload["messages"] = [{"role": "system", "content": system_prompt}, *messages]
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "stop",
            "logit_bias",
        ):
            if params.get(key) is not None:
                payload[key] = params[key]
        # Max can arrive as -1 or the whole context window. Forwarding the
        # latter leaves no room for the prompt, so the server rejects it.
        # Omit both forms and let the engine budget the remaining context.
        limit = params.get("max_new_tokens") or 0
        if limit > 0 and (not self.context or limit < self.context):
            payload["max_tokens"] = limit
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        from .engine_transport import stream_chat_events

        def cancelled():
            return self._cancel.is_set() or (cancel_event is not None and cancel_event.is_set())

        for event in stream_chat_events(self.base_url, self.headers, payload, cancelled):
            if event.get("usage") and stats_holder is not None:
                stats_holder.setdefault("stats", {})["usage"] = event["usage"]
            if event.get("error"):
                raise RuntimeError("Engine generation failed.")
            for choice in event.get("choices", []):
                if choice.get("finish_reason") and stats_holder is not None:
                    stats_holder.setdefault("stats", {})["finish_reason"] = choice["finish_reason"]
                content = choice.get("delta", {}).get("content")
                if content:
                    yield content
