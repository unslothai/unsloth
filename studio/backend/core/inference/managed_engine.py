# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""HTTP server lifecycle for optional inference engines."""

from __future__ import annotations

import json
import hashlib
import os
import secrets
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import httpx

from .engine_install import (
    engine_lease,
    installed,
    profile,
    profile_digest,
    stale,
    support_reason,
)


from .engine_adapters import (
    ADAPTERS,
    gpu_memory_fraction,
    launch_arguments,
    tool_parser_for_template,
)


def validate_load(engine: str, request) -> None:
    profile(engine)
    gpu_ids = request.gpu_ids or [0]
    if len(set(gpu_ids)) != len(gpu_ids) or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("Select distinct, non-negative GPU indices.")
    for gpu_id in gpu_ids:
        reason = support_reason(engine, gpu_id)
        if reason:
            raise ValueError(f"GPU {gpu_id}: {reason}")
    info = installed(engine)
    if not info:
        raise ValueError(f"Install {engine} in Settings > System > Inference engines first.")
    if stale(info):
        raise ValueError(
            f"Studio's packages changed since {engine} was installed. Repair it in Settings > System > Inference engines."
        )
    # The picker's rule too: an outdated profile loads only after an explicit restore.
    if info.get("profile_digest") != profile_digest(engine) and not info.get("restored"):
        raise ValueError(f"Update {engine} in Settings > System > Inference engines first.")
    # Also judges /validate requests, which carry no LoRA or template fields.
    if (
        request.gguf_variant
        or request.model_path.lower().endswith(".gguf")
        or getattr(request, "is_lora", False)
    ):
        raise ValueError(
            "Optional engines require a full model checkpoint. GGUF files and LoRA adapters use Default."
        )
    if getattr(request, "chat_template_override", None):
        raise ValueError("Optional engines do not yet support template overrides.")


def _model_chat_template(config, hf_token = None):
    """Read the same template files native tokenizers use, without loading weights."""

    def read(name):
        if config.is_local:
            path = Path(config.path) / name
            return path.read_text(encoding = "utf-8") if path.is_file() else None
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            return Path(hf_hub_download(config.identifier, name, token = hf_token)).read_text(
                encoding = "utf-8"
            )
        except EntryNotFoundError:
            return None

    template = read("chat_template.jinja")
    if template is not None:
        return template
    metadata = json.loads(read("tokenizer_config.json") or "{}")
    template = metadata.get("chat_template")
    if isinstance(template, list):
        template = {entry["name"]: entry["template"] for entry in template}
    if isinstance(template, dict):
        template = template.get("tool_use") or template.get("default")
    return template


def validate_model(
    config,
    hf_token = None,
    gpu_ids = None,
    engine = "vllm",
    precision = "auto",
    parallelism = "tensor",
) -> dict:
    """Check plain metadata before releasing the previous resident model."""
    if config.is_local:
        path = Path(config.path) / "config.json"
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(config.identifier, "config.json", token = hf_token))
    metadata = json.loads(path.read_text(encoding = "utf-8"))
    quant = (
        metadata.get("quantization_config")
        or metadata.get("text_config", {}).get("quantization_config")
        or {}
    )
    if quant and precision != "auto":
        raise ValueError(
            "This checkpoint is already quantized. Choose Model default to use its stored precision."
        )
    if (
        quant.get("quant_method") == "bitsandbytes"
        and len(gpu_ids or [0]) > 1
        and parallelism == "tensor"
    ):
        raise ValueError(
            "Prequantized BitsAndBytes checkpoints do not support tensor parallelism with this engine. Select one GPU, another multi-GPU mode, or a tensor-parallel compatible checkpoint such as AWQ or GPTQ."
        )
    if (
        quant.get("quant_method") == "bitsandbytes"
        and engine == "sglang"
        and parallelism == "pipeline"
        and len(gpu_ids or [0]) > 1
    ):
        raise ValueError(
            "SGLang cannot load prequantized BitsAndBytes checkpoints across pipeline stages. "
            "Select one GPU, use Replicas, or load an unquantized checkpoint with 4-bit precision."
        )
    options = {
        "tool_parser": tool_parser_for_template(_model_chat_template(config, hf_token), engine),
        "precision": precision,
        "parallelism": parallelism,
        "is_vision": bool(metadata.get("vision_config") or getattr(config, "is_vision", False)),
        "quantization": quant.get("quant_method"),
        # BNB's INT8 outlier extraction synchronizes on the CPU and cannot be captured.
        "disable_cuda_graph": engine == "sglang"
        and quant.get("quant_method") == "bitsandbytes"
        and quant.get("load_in_8bit", False),
        "load_format": "bitsandbytes"
        if quant.get("quant_method") == "bitsandbytes"
        or (engine == "vllm" and precision == "int4" and parallelism != "pipeline")
        else "auto",
    }
    if precision == "fp8":
        # Use eager TorchAO FP8 storage on Ampere. Triton cannot compile
        # its FP8 casts, and SGLang's online FP8 kernels produce invalid output.
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                ",".join(str(i) for i in (gpu_ids or [0])),
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
            check = True,
        )
        options["disable_cuda_graph"] = any(float(cap) < 8.9 for cap in result.stdout.splitlines())
    # Multimodal architectures keep their language-model dimensions here.
    metadata = metadata.get("text_config") or metadata
    size = len(gpu_ids or [0])
    if size > 1 and parallelism == "tensor":
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

    return options


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
        self._tail = deque(maxlen = 200)
        self.base_url = ""
        # A URL-safe token can start with '-', which native CLI parsers read as
        # another option instead of the API key.
        self.key = "studio-" + secrets.token_urlsafe(32)

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(
        self,
        model: str,
        context: int,
        gpu_ids,
        env: dict,
        cancel_event = None,
        options = None,
        trust_remote_code = False,
        model_path = None,
    ):
        """``model`` is the served name; ``model_path`` is what the engine loads when they differ."""
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
                # A shared environment also runs console scripts of the packages Studio provides.
                child_env["PATH"] = os.pathsep.join(
                    [
                        info["path"] + "/bin",
                        *([info["studio_prefix"] + "/bin"] if info.get("shared") else []),
                        child_env.get("PATH", os.defpath),
                    ]
                )
                child_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in (gpu_ids or [0]))
                child_env["PYTHONNOUSERSITE"] = "1"
                from .engine_install import engine_root

                # Pinned compilers can reuse incompatible artifacts across dtype
                # and GPU changes. Reuse only within the same launch configuration.
                policy = Path(__file__).with_name("engine_adapters.py").read_bytes()
                if self.engine == "sglang":
                    policy += Path(__file__).with_name("sglang_server.py").read_bytes()
                cache_key = hashlib.sha256(
                    policy
                    + json.dumps(
                        [info.get("profile_digest"), model, self.context, gpu_ids, options],
                        sort_keys = True,
                    ).encode()
                ).hexdigest()[:16]
                cache = engine_root() / self.engine / "cache" / cache_key
                child_env["VLLM_CACHE_ROOT"] = str(cache)
                child_env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
                child_env["TRITON_CACHE_DIR"] = str(cache / "triton")
                memory_fraction = gpu_memory_fraction(gpu_ids or [0])
                child_env.update(self.adapter.environment(len(gpu_ids or [0])))
                self.process = spawn_on_lifetime_thread(
                    lambda: subprocess.Popen(
                        self.adapter.command(
                            info["path"] + "/bin/python",
                            model_path or model,
                            port,
                            self.key,
                            self.context,
                            memory_fraction,
                            len(gpu_ids or [0]),
                            **(
                                {"options": options, "trust_remote_code": trust_remote_code}
                                if options
                                else {}
                            ),
                            **(
                                {"served_model_name": model}
                                if model_path and model_path != model
                                else {}
                            ),
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
        if params.get("video"):
            raise ValueError(
                "Video input is not yet supported by the Studio managed engine transport."
            )
        from .orchestrator import _encoded_images, InferenceOrchestrator

        request_images = _encoded_images(
            params.get("images") or ([params["image"]] if params.get("image") is not None else []),
            InferenceOrchestrator._pil_to_base64,
        )
        if request_images:
            messages = [dict(message) for message in messages]
            pending = iter(request_images)
            used_placeholders = False
            for message in messages:
                content = message.get("content", "")
                if not isinstance(content, list):
                    continue
                parts = []
                for part in content:
                    if part.get("type") == "image":
                        used_placeholders = True
                        image = next(pending)
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + image},
                            }
                        )
                    else:
                        parts.append(part)
                message["content"] = parts
            if not used_placeholders:
                for message in reversed(messages):
                    if message.get("role") == "user":
                        content = message.get("content", "")
                        parts = (
                            [{"type": "text", "text": content}]
                            if isinstance(content, str)
                            else list(content)
                        )
                        parts.extend(
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + image},
                            }
                            for image in request_images
                        )
                        message["content"] = parts
                        break
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
