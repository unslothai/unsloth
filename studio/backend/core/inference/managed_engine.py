# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""HTTP server lifecycle for optional inference engines."""

from __future__ import annotations

import json
import hashlib
import os
import re
import secrets
import socket
import subprocess

from utils.subprocess_compat import windows_hidden_subprocess_kwargs
import sys
import threading
import time
from collections import deque
from pathlib import Path

import httpx

from utils.hardware.hardware import resolve_requested_gpu_ids

from .engine_install import (
    ENGINE_NAMES,
    built_for_this_gpu,
    driver_library_path,
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
    memory_reserve_mib,
    RESERVE_SHARE,
    launch_arguments,
    tool_parser_for_template,
)


def _offline(env) -> bool:
    return any(
        str(env.get(key, "")).strip().lower() in {"1", "true", "yes", "on"}
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def validate_load(engine: str, request) -> list[int]:
    """Returns the physical GPU ids to launch on, all inside the GPUs Studio may use."""
    profile(engine)
    gpu_ids = list(request.gpu_ids or [])
    if len(set(gpu_ids)) != len(gpu_ids) or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("Select distinct, non-negative GPU indices.")
    # The engine's CUDA_VISIBLE_DEVICES replaces Studio's, so an omitted selection is the first GPU Studio sees.
    gpu_ids = resolve_requested_gpu_ids(gpu_ids) if gpu_ids else resolve_requested_gpu_ids(None)[:1]
    if not gpu_ids:
        raise ValueError("Studio has no GPU it can assign to this engine.")
    from . import wsl_host

    if len(gpu_ids) > 1 and wsl_host.active() and profile(engine)["platform"] == "rocm":
        raise ValueError(
            f"On Windows, {ENGINE_NAMES[engine]} runs on one AMD GPU. Select a single GPU."
        )
    for gpu_id in gpu_ids:
        reason = support_reason(engine, gpu_id)
        if reason:
            raise ValueError(f"GPU {gpu_id}: {reason}")
    info = installed(engine)
    if not info:
        raise ValueError(
            f"Install {ENGINE_NAMES[engine]} in Settings > System > Inference engines first."
        )
    if stale(info):
        raise ValueError(
            f"Studio's packages changed since {ENGINE_NAMES[engine]} was installed. Repair it in Settings > System > Inference engines."
        )
    if info.get("profile_digest") != profile_digest(engine) and not (
        info.get("restored") and built_for_this_gpu(engine, info)
    ):
        raise ValueError(
            f"Update {ENGINE_NAMES[engine]} in Settings > System > Inference engines first."
        )
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
    from .engine_adapters import _release_at_least

    # Refused here, before the resident model is unloaded, not when the engine command is built.
    precision = getattr(request, "engine_precision", "auto")
    loadable = profile(engine)["precisions"]
    if precision not in loadable:
        names = {"bf16": "BF16", "fp16": "FP16", "int8": "INT8", "fp8": "FP8", "int4": "4-bit"}
        offered = ", ".join(names[p] for p in loadable if p != "auto")
        raise ValueError(
            f"{ENGINE_NAMES[engine]} on this GPU cannot load weights as {names.get(precision, precision)}. "
            f"Choose {offered} or Model default."
        )
    if (
        engine == "sglang"
        and precision in ("int8", "int4")
        and _release_at_least(info.get("version"), "0.5.18")
    ):
        raise ValueError(
            f"SGLang {info.get('version')} cannot convert weights to {precision.upper()} when "
            "loading. Choose FP8 or Model default, or use vLLM."
        )
    return gpu_ids


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
    if quant.get("quant_method") == "bitsandbytes" and not profile(engine)["bitsandbytes"]:
        raise ValueError(
            f"{ENGINE_NAMES[engine]} on this GPU cannot load BitsAndBytes checkpoints. Choose an unquantized "
            "or AWQ checkpoint."
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
    if precision == "fp8" and profile(engine)["platform"] == "cuda":
        # Eager TorchAO FP8 on Ampere: Triton cannot compile its casts, SGLang online FP8 is invalid.
        from utils.hardware.nvidia import _nvidia_smi_executable
        result = subprocess.run(
            [
                _nvidia_smi_executable(),
                "--id",
                ",".join(str(i) for i in (gpu_ids or [0])),
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output = True,
            **windows_hidden_subprocess_kwargs(),
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 60,
            check = True,
        )
        options["disable_cuda_graph"] = any(float(cap) < 8.9 for cap in result.stdout.splitlines())
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
            if max(kv_heads, size) % min(kv_heads, size):
                raise ValueError(
                    f"This model's {kv_heads} KV heads are incompatible with {size} GPUs. "
                    "Select a GPU count that divides the KV heads or is a multiple of them."
                )

    return options


def _deep_gemm_unloadable(environment: str) -> bool:
    """vLLM bundles DeepGEMM's _C for one CPython only; a mismatch still reads as present, so
    Hopper/Blackwell warmup crashes even bf16 models (vllm-project/vllm#41849)."""
    for site in Path(environment).glob("lib/python3.*/site-packages"):
        if (site / "deep_gemm").is_dir():
            return False
        vendored = site / "vllm" / "third_party" / "deep_gemm"
        if vendored.is_dir():
            tag = "cpython-3" + site.parent.name.removeprefix("python3.")
            return not any(vendored.glob(f"_C.{tag}-*.so")) and not any(
                vendored.glob("_C.abi3*.so")
            )
    return False


# The engine's own torch on its devices: what vLLM budgets against on ROCm (amd-smi may be absent).
_DEVICE_MEMORY = (
    "import json, torch; print(json.dumps("
    "[torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count())]))"
)


def _memory_line(output: str) -> list:
    """The probe's JSON out of everything the engine's imports print: wsl.exe returns stderr with
    stdout, and torch can warn there after the measurement."""
    for line in reversed(output.splitlines()):
        if line.startswith("[["):
            return json.loads(line)
    raise ValueError("The GPU memory probe printed no measurement")


def _wsl_amd_usable_mib(gpu_ids) -> list[float] | None:
    """MiB each selected AMD GPU can allocate through WSL, or None when Windows cannot say.

    Through DXG the pool HIP reports is the dedicated memory plus a share of the host's RAM, and an
    allocation past what the host can back hangs instead of failing. So a discrete card keeps its
    dedicated memory, and an APU adds 80% of the host RAM Windows reports available now."""
    try:
        import psutil
        import torch
        from utils.hardware.hardware import (
            _normalize_adapter_name,
            _props_gfx_arch,
            _rocm_props_are_positively_unified,
            _torch_ordinal_physical_ids,
            _windows_amd_adapter_records_or_none,
        )

        records = list((_windows_amd_adapter_records_or_none() or {}).values())
        count = torch.cuda.device_count()
        physical = _torch_ordinal_physical_ids(count) or list(range(count))
        available = psutil.virtual_memory().available
        caps = []
        for gpu_id in gpu_ids:
            props = torch.cuda.get_device_properties(physical.index(gpu_id))
            matches = [
                record
                for record in records
                if record.get("gfx") == _props_gfx_arch(props)
                or _normalize_adapter_name(record["name"]) == _normalize_adapter_name(props.name)
            ]
            if len(matches) != 1 or "dedicated_memory_bytes" not in matches[0]:
                return None
            usable = matches[0]["dedicated_memory_bytes"]
            if _rocm_props_are_positively_unified(props):
                usable += 0.8 * available
            caps.append(usable / 2**20)
        return caps
    except Exception:
        return None


def _engine_memory_rows(
    info: dict, child_env: dict, gpu_ids: list[int]
) -> list[tuple[float, float]]:
    """(total, free) MiB of each device the engine will see, in its order."""
    from . import wsl_host

    command = [info["path"] + "/bin/python", "-I", "-c", _DEVICE_MEMORY]
    if info.get("host") == "wsl":
        output = wsl_host.guest(command, env = child_env, timeout = 180)
    else:
        output = subprocess.run(
            command,
            env = child_env,
            capture_output = True,
            **windows_hidden_subprocess_kwargs(),
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 180,
            check = True,
        ).stdout
    rows = [(total / 2**20, free / 2**20) for free, total in _memory_line(output)]
    if info.get("host") == "wsl":
        caps = _wsl_amd_usable_mib(list(gpu_ids))
        if caps is None or len(caps) != len(rows):
            # Windows did not say; the VM's own free memory is the smaller, safe bound.
            meminfo = wsl_host.guest(["cat", "/proc/meminfo"], timeout = 60)
            match = re.search(r"^MemAvailable:\s+(\d+) kB", meminfo, re.M)
            if not match:
                raise ValueError("Could not read the memory of the WSL environment")
            caps = [int(match.group(1)) / 1024] * len(rows)
        rows = [(total, min(free, cap)) for (total, free), cap in zip(rows, caps)]
    return rows


def _rocm_visibility(env: dict, gpu_ids) -> dict:
    """The child's device mask on ROCm. vLLM refuses a HIP_VISIBLE_DEVICES that differs from
    CUDA_VISIBLE_DEVICES, and HIP numbers devices after an inherited ROCR_VISIBLE_DEVICES, so the
    physical ids become ordinals into that list, or the ROCr mask goes when they cannot."""
    from utils.hardware.hardware import _rocr_relative_visibility

    physical = ",".join(str(i) for i in gpu_ids)
    mask = _rocr_relative_visibility(physical)
    if mask is None:
        env.pop("ROCR_VISIBLE_DEVICES", None)
        mask = physical
    return {"HIP_VISIBLE_DEVICES": mask, "CUDA_VISIBLE_DEVICES": mask}


# SGLang raises when its derived gRPC port (HTTP + 10000) exceeds 65535.
_PORT_LIMIT = 55535


def _free_port() -> int:
    """A free local port no higher than _PORT_LIMIT. Windows hands out ephemeral ports in sequence
    from 49152, so once its counter passes the limit the OS never offers a low enough one; free
    ports below it are then tried at random."""
    for attempt in range(100):
        with socket.socket() as sock:
            candidate = 0 if attempt == 0 else 20000 + secrets.randbelow(_PORT_LIMIT - 20000 + 1)
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            port = sock.getsockname()[1]
        if port <= _PORT_LIMIT:
            return port
    raise RuntimeError("Could not allocate an inference server port.")


# Engine silence, not total startup time: an uncached Hub model downloads inside the engine.
STARTUP_STALL_S = 900
STARTUP_LIMIT_S = 4 * 3600


def _token_file_token(env) -> str | None:
    """The `hf auth login` token a local engine would read from its inherited HF_HOME; the WSL
    guest gets its own HF_HOME, so the token has to cross over. Anonymous loads keep none."""
    if (env.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") or "").upper() in ("1", "ON", "YES", "TRUE"):
        return None
    home = env.get("HF_HOME") or os.path.join(
        env.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "huggingface"
    )
    path = env.get("HF_TOKEN_PATH") or os.path.join(
        os.path.expandvars(os.path.expanduser(home)), "token"
    )
    try:
        token = (
            Path(os.path.expandvars(os.path.expanduser(path))).read_text(encoding = "utf-8").strip()
        )
    except (OSError, UnicodeDecodeError):
        return None
    return token or None


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
        self._last_output = time.monotonic()
        self.base_url = ""
        self._guest_environment = None
        # A URL-safe token can start with '-', which CLI parsers read as an option.
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
                port = _free_port()
                self.base_url = f"http://127.0.0.1:{port}"
                self.model, self.context = model, context or 4096
                stdin = None
                if info.get("host") == "wsl":
                    command, child_env = self._wsl_command(
                        info, env, gpu_ids, options, trust_remote_code, model, model_path, port
                    )
                    # The guest runner ends the engine when this pipe closes (stop, or Studio dying).
                    stdin = subprocess.PIPE
                    self._guest_environment = info["path"]
                else:
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
                    if driver := driver_library_path(env):
                        child_env["LD_LIBRARY_PATH"] = driver
                    child_env["PATH"] = os.pathsep.join(
                        [
                            info["path"] + "/bin",
                            *([info["studio_prefix"] + "/bin"] if info.get("shared") else []),
                            child_env.get("PATH", os.defpath),
                        ]
                    )
                    child_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in (gpu_ids or [0]))
                    rocm = info.get("platform") == "rocm"
                    if rocm:
                        child_env.update(_rocm_visibility(child_env, gpu_ids or [0]))
                    child_env["PYTHONNOUSERSITE"] = "1"
                    from .engine_install import engine_root

                    cache = (
                        engine_root()
                        / self.engine
                        / "cache"
                        / self._cache_key(info, model, gpu_ids, options)
                    )
                    child_env["VLLM_CACHE_ROOT"] = str(cache)
                    child_env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
                    child_env["TRITON_CACHE_DIR"] = str(cache / "triton")
                    # FlashInfer's JIT build files name this env's sources; a shared ~/.cache outlives a replaced env.
                    child_env["FLASHINFER_WORKSPACE_BASE"] = info["path"]
                    from .engine_install import cuda_environment

                    child_env.pop("CUDA_PATH", None)
                    child_env.update(cuda_environment(info))
                    measure = (
                        (lambda ids: _engine_memory_rows(info, child_env, ids)) if rocm else None
                    )
                    memory_fraction = gpu_memory_fraction(
                        gpu_ids or [0],
                        memory_reserve_mib(self.engine, options),
                        RESERVE_SHARE,
                        measure,
                    )
                    child_env.update(self.adapter.environment(len(gpu_ids or [0])))
                    child_env.update(self.adapter.key_environment(self.key))
                    if self.engine == "vllm" and _deep_gemm_unloadable(info["path"]):
                        child_env["VLLM_USE_DEEP_GEMM"] = "0"
                    command = self.adapter.command(
                        info["path"] + "/bin/python",
                        model_path or model,
                        port,
                        self.key,
                        self.context,
                        memory_fraction,
                        len(gpu_ids or [0]),
                        **(
                            {
                                "options": {**options, "engine_version": info.get("version")},
                                "trust_remote_code": trust_remote_code,
                            }
                            if options
                            else {}
                        ),
                        **(
                            {"served_model_name": model}
                            if model_path and model_path != model
                            else {}
                        ),
                    )
                self.process = spawn_on_lifetime_thread(
                    lambda: subprocess.Popen(
                        command,
                        env = child_env,
                        stdin = stdin,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.STDOUT,
                        text = True,
                        encoding = "utf-8",
                        errors = "replace",
                        start_new_session = True,
                        creationflags = 0x08000000 if sys.platform == "win32" else 0,
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
        started = self._last_output = time.monotonic()
        try:
            with httpx.Client(trust_env = False, timeout = 2) as client:
                while (
                    time.monotonic() - self._last_output < STARTUP_STALL_S
                    and time.monotonic() - started < STARTUP_LIMIT_S
                ):
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
                    "Engine startup timed out. Try a smaller model or context length.\n"
                    + "\n".join(list(self._tail)[-20:])
                )
        except Exception:
            self.stop()
            raise

    def _cache_key(self, info, model, gpu_ids, options) -> str:
        # Compiler caches are keyed per launch config: reuse across dtype/GPU changes breaks.
        policy = Path(__file__).with_name("engine_adapters.py").read_bytes()
        policy += Path(__file__).with_name(f"{self.engine}_server.py").read_bytes()
        return hashlib.sha256(
            policy
            + json.dumps(
                [info.get("profile_digest"), model, self.context, gpu_ids, options],
                sort_keys = True,
            ).encode()
        ).hexdigest()[:16]

    def _wsl_command(
        self, info, env, gpu_ids, options, trust_remote_code, model, model_path, port
    ) -> tuple[list[str], dict]:
        """The engine inside Studio's WSL distro: only the variables it needs cross over, since
        Studio's Windows paths mean nothing there; the token crosses through WSLENV, not argv."""
        from . import wsl_host
        from hub.utils.hf_tokens import _HF_TOKEN_ENV_KEYS

        guest_root = wsl_host.GUEST_ROOT
        environment = info["path"]
        wsl_host.guest(["test", "-x", environment + "/bin/python"], timeout = 300)
        cache = f"{guest_root}/cache/{self.engine}/{self._cache_key(info, model, gpu_ids, options)}"
        rocm = info.get("platform") == "rocm"
        # NVIDIA GPUs are matched by UUID; the one AMD GPU a WSL engine may use is the same ordinal.
        devices = ",".join(
            str(i)
            for i in (
                list(gpu_ids or [0]) if rocm else wsl_host.guest_gpu_indices(list(gpu_ids or [0]))
            )
        )
        guest_env = {
            "PATH": environment
            + "/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": devices,
            **({"HIP_VISIBLE_DEVICES": devices, **wsl_host.rocm_environment()} if rocm else {}),
            "PYTHONNOUSERSITE": "1",
            # C++ links FlashInfer JIT kernels against WSL's driver, outside the usual linker paths.
            "LIBRARY_PATH": "/usr/lib/wsl/lib",
            # Weights download inside the distro's own disk; /mnt/c reads are far slower.
            "HF_HOME": f"{guest_root}/hf",
            "VLLM_CACHE_ROOT": cache,
            "TORCHINDUCTOR_CACHE_DIR": cache + "/inductor",
            "TRITON_CACHE_DIR": cache + "/triton",
            "FLASHINFER_WORKSPACE_BASE": environment,
            **({"HF_ENDPOINT": env["HF_ENDPOINT"]} if env.get("HF_ENDPOINT") else {}),
            # Cache-only mode must hold in the guest too.
            **({"HF_HUB_OFFLINE": "1"} if _offline(env) else {}),
        }
        from .engine_install import cuda_environment

        guest_env.update(cuda_environment(info))
        guest_env.update(self.adapter.environment(len(gpu_ids or [0])))
        if self.engine == "vllm" and info.get("deep_gemm_unloadable"):
            guest_env["VLLM_USE_DEEP_GEMM"] = "0"
        target = wsl_host.to_guest_path(model_path) if model_path else model
        measure = (lambda ids: _engine_memory_rows(info, guest_env, ids)) if rocm else None
        command = self.adapter.command(
            environment + "/bin/python",
            target,
            port,
            self.key,
            self.context,
            gpu_memory_fraction(
                gpu_ids or [0], memory_reserve_mib(self.engine, options), RESERVE_SHARE, measure
            ),
            len(gpu_ids or [0]),
            **(
                {
                    "options": {**options, "engine_version": info.get("version")},
                    "trust_remote_code": trust_remote_code,
                }
                if options
                else {}
            ),
            **({"served_model_name": model} if model_path and model_path != model else {}),
        )
        # The engine launchers are Studio source files; the guest reads them through /mnt.
        server = str(Path(__file__).with_name(f"{self.engine}_server.py"))
        command = [wsl_host.to_guest_path(arg) if arg == server else arg for arg in command]
        secrets = {
            key: env[key]
            for key in (*_HF_TOKEN_ENV_KEYS, "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY")
            if env.get(key)
        }
        if not any(key in secrets for key in _HF_TOKEN_ENV_KEYS):
            token = _token_file_token(env)
            if token:
                secrets["HF_TOKEN"] = token
        # The engine key goes through WSLENV like the tokens: guest_env lands on /usr/bin/env's argv.
        secrets.update(self.adapter.key_environment(self.key))
        return wsl_host.guest_command(
            [
                f"{guest_root}/bin/run-engine",
                "/usr/bin/env",
                *[f"{k}={v}" for k, v in guest_env.items()],
                *command,
            ],
            secrets = secrets,
            withhold = tuple(key.upper() for key in _HF_TOKEN_ENV_KEYS if key not in secrets),
        )

    @property
    def headers(self):
        return {"Authorization": "Bearer " + self.key}

    def _drain(self, proc):
        from utils.log_redaction import redact_log_text
        from utils.native_path_leases import redact_native_paths
        for line in proc.stdout:
            self._last_output = time.monotonic()
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
                graceful = False
                if self.process.stdin is not None:
                    # WSL: closing the pipe makes the guest runner stop the engine's process group.
                    try:
                        self.process.stdin.close()
                        self.process.wait(timeout = 15)
                        graceful = True
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                if not graceful:
                    terminate_pid(self.process.pid, timeout = 5, owner_verified = True)
                try:
                    self.process.wait(timeout = 5)
                except subprocess.TimeoutExpired:
                    return False
                if self._guest_environment and not graceful:
                    from .wsl_host import kill_environment
                    kill_environment(self._guest_environment)
                self._guest_environment = None
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
        from .engine_transport import engine_messages

        messages = engine_messages(messages)
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
        # -1 or full context leaves no room for the prompt; let the engine budget it.
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
