# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-process Laya runtime behind ``POST /v1/systemone``: one checkpoint resident at a time."""

from __future__ import annotations

import gc
import importlib.util
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .catalog import CHECKPOINTS, LOCAL_NAME, Checkpoint

logger = logging.getLogger(__name__)

LOAD_WAIT_S = 20.0
RUN_WAIT_S = 30.0
# Bounded: each waiter holds a shared threadpool worker, so an unbounded queue stalls every sync route.
MAX_PENDING = 8
FAILURE_BACKOFF_S = 60.0
_REQUIRED_DIRS = ("encoder", "tokenizer")
# laya 0.3.5 ships inside Studio (see vendor/README.md), so the Decision API never installs anything.
_VENDORED_LAYA = Path(__file__).resolve().parent.parent.parent / "vendor" / "laya"

_state_lock = threading.Lock()
_run_lock = threading.Lock()
_admission = threading.BoundedSemaphore(MAX_PENDING)
_agent = None
_loaded: Checkpoint | None = None
_device_name: str | None = None
_loader: threading.Thread | None = None
_loading: Checkpoint | None = None
_failure: tuple[Checkpoint, str, float] | None = None
_import_lock = threading.Lock()


class Unavailable(Exception):
    def __init__(
        self,
        status: int,
        error_type: str,
        message: str,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = message
        self.retry_after = retry_after


def _device() -> str:
    from utils.systemone_settings import get_device as preferred_device

    # CPU unless asked: a CUDA context opened in the backend process is never returned (see core.rag.embeddings._device).
    if preferred_device() != "gpu":
        return "cpu"
    from utils.hardware.hardware import DeviceType, get_device

    device = get_device()
    if device == DeviceType.MLX:
        if _mlx_available():
            return "mlx"
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    candidate = {DeviceType.CUDA: "cuda", DeviceType.XPU: "xpu"}.get(device, "cpu")
    if candidate == "cpu":
        return candidate
    from utils.torch_device_probe import device_can_allocate

    return candidate if device_can_allocate(candidate) else "cpu"


def _training_active() -> bool:
    try:
        from core.training import get_training_backend
        return bool(get_training_backend().is_training_active())
    except Exception:
        return False


def _mlx_available() -> bool:
    try:
        import unsloth_zoo.mlx.decision  # noqa: F401
    except ImportError:
        return False
    return True


_WEIGHT_FILES = ("rl_agent_config.json", "model.safetensors")


def _wanted(path: str, subfolder: str | None) -> bool:
    prefix = f"{subfolder}/" if subfolder else ""
    if not path.startswith(prefix):
        return False
    rest = path[len(prefix) :]
    return rest in _WEIGHT_FILES or rest.startswith(tuple(f"{name}/" for name in _REQUIRED_DIRS))


def _clef_complete(folder: Path) -> bool:
    import json

    from utils.models.model_config import clef_folder_kind

    kind = clef_folder_kind(folder)
    if kind is None:
        return False
    if kind == "adapter":
        # The base LLM the adapters sit on is fetched by the loader, like any Unsloth LoRA.
        return (folder / "adapter_model.safetensors").is_file()
    index = folder / "model.safetensors.index.json"
    if not index.is_file():
        return (folder / "model.safetensors").is_file()
    try:
        shards = set(json.loads(index.read_text(encoding = "utf-8"))["weight_map"].values())
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return all((folder / shard).is_file() for shard in shards)


def _clef_dir(checkpoint: Checkpoint, local_only: bool) -> Path:
    if checkpoint.is_local:
        root = Path(checkpoint.source).expanduser()
    else:
        from huggingface_hub import snapshot_download

        from utils.hf_cache_settings import active_hf_hub_cache
        from utils.utils import hf_env_offline

        root = Path(
            snapshot_download(
                checkpoint.source,
                cache_dir = active_hf_hub_cache(),
                local_files_only = local_only or hf_env_offline(),
            )
        )
    if not _clef_complete(root):
        raise FileNotFoundError(f"No complete Clef checkpoint at {root}")
    return root


def _checkpoint_dir(checkpoint: Checkpoint, *, local_only: bool = False) -> Path:
    if checkpoint.layout == "clef":
        return _clef_dir(checkpoint, local_only)
    if checkpoint.name == LOCAL_NAME and not checkpoint.is_local:
        raise FileNotFoundError(
            f"UNSLOTH_SYSTEMONE_MODEL={checkpoint.source} is neither a known model nor a directory"
        )
    if checkpoint.is_local:
        root = Path(checkpoint.source).expanduser()
    else:
        from huggingface_hub import snapshot_download

        from utils.hf_cache_settings import active_hf_hub_cache
        from utils.utils import hf_env_offline

        prefix = f"{checkpoint.subfolder}/" if checkpoint.subfolder else ""
        root = Path(
            snapshot_download(
                checkpoint.source,
                cache_dir = active_hf_hub_cache(),
                local_files_only = local_only or hf_env_offline(),
                allow_patterns = [prefix + name for name in _WEIGHT_FILES]
                + [f"{prefix}{name}/*" for name in _REQUIRED_DIRS],
            )
        )
    folder = root / checkpoint.subfolder if checkpoint.subfolder else root
    # Without these, laya falls back to downloading the base encoder from the Hub on every load.
    missing = [name for name in _REQUIRED_DIRS if not (folder / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Laya checkpoint at {folder} is missing {', '.join(n + '/' for n in missing)}"
        )
    return root


def _laya():
    """The vendored laya package, registered as top-level ``laya`` (its modules import each other relatively).

    Loaded by file path, not from ``sys.path``: a laya installed in the venv (Studio pinned one before
    vendoring it) must not replace this copy, since this module drives laya internals.
    """
    if (module := sys.modules.get("laya")) is not None:
        return module
    with _import_lock:
        if (module := sys.modules.get("laya")) is not None:
            return module
        init = _VENDORED_LAYA / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "laya", init, submodule_search_locations = [str(_VENDORED_LAYA)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["laya"] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for name in [n for n in sys.modules if n == "laya" or n.startswith("laya.")]:
                del sys.modules[name]
            raise
        for name in [n for n in sys.modules if n == "laya" or n.startswith("laya.")]:
            sys.modules[name].open = _utf8_open
        return module


def _utf8_open(
    file,
    mode = "r",
    buffering = -1,
    encoding = None,
    errors = None,
    newline = None,
    closefd = True,
    opener = None,
):
    """``open`` for the vendored laya modules: text mode defaults to UTF-8.

    laya reads ``rl_agent_config.json`` and ``tokenizer_config.json`` with a bare ``open()``,
    which decodes with the locale's code page (ANSI on Windows, ASCII under a C locale). A
    checkpoint whose tokenizer config holds non-ASCII special tokens then fails to read, and
    ``_fix_tokenizer_config`` swallows that and skips the repair the model needs to load.
    The vendored files stay byte-identical to the wheel (vendor/README.md), so the encoding is
    supplied here, as each laya module's own ``open``.
    """
    if encoding is None and "b" not in mode:
        encoding = "utf-8"
    return open(
        file,
        mode,
        buffering,
        encoding = encoding,
        errors = errors,
        newline = newline,
        closefd = closefd,
        opener = opener,
    )


def is_cached(checkpoint: Checkpoint) -> bool:
    try:
        root = _checkpoint_dir(checkpoint, local_only = True)
    except Exception:
        return False
    if checkpoint.layout == "clef":
        return True
    folder = root / checkpoint.subfolder if checkpoint.subfolder else root
    return all((folder / name).is_file() for name in _WEIGHT_FILES)


def download_plan(checkpoint: Checkpoint) -> dict[str, Any]:
    cached = is_cached(checkpoint)
    plan = {"repo": None, "files": [], "size_bytes": 0, "cached": cached, "error": None}
    if checkpoint.name == LOCAL_NAME or checkpoint.is_local:
        if not cached:
            kind = "Clef" if checkpoint.layout == "clef" else "Laya"
            plan["error"] = f"No complete {kind} checkpoint at {checkpoint.source}"
        return plan
    plan["repo"] = checkpoint.source
    plan["size_bytes"] = checkpoint.download_bytes
    if cached:
        return plan
    try:
        from huggingface_hub import HfApi
        entries = HfApi().list_repo_tree(checkpoint.source, recursive = True)
        files = [
            (entry.path, getattr(entry, "size", 0) or 0)
            for entry in entries
            if hasattr(entry, "size")
            and (checkpoint.layout == "clef" or _wanted(entry.path, checkpoint.subfolder))
        ]
    except Exception as exc:
        plan["error"] = f"Could not list {checkpoint.source}: {type(exc).__name__}"
        return plan
    plan["files"] = sorted(path for path, _ in files)
    plan["size_bytes"] = sum(size for _, size in files) or checkpoint.download_bytes
    return plan


def _release_memory() -> None:
    gc.collect()
    # MLX keeps freed buffers in its allocator cache until told otherwise; mlx below 0.24.1 names it mx.metal.clear_cache.
    if (mx := sys.modules.get("mlx.core")) is not None:
        clear = getattr(mx, "clear_cache", None) or getattr(
            getattr(mx, "metal", None), "clear_cache", None
        )
        if clear is not None:
            clear()
    # Torch's allocator too; only already-initialised backends, so a CPU-only Studio never opens a device context.
    if (torch := sys.modules.get("torch")) is not None:
        try:
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_initialized():
                torch.xpu.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            logger.debug("Could not clear the Decision API device cache", exc_info = True)


def _close(agent) -> None:
    # A Clef agent is a worker process; ending it returns its GPU memory.
    if agent is not None and hasattr(agent, "close"):
        agent.close()


def _evict() -> None:
    global _agent, _loaded, _device_name
    with _run_lock:
        agent, _agent, _loaded, _device_name = _agent, None, None, None
    _close(agent)
    _release_memory()


class _MLXAgent:
    """The part of ``laya.Agent`` this module drives, over the unsloth-zoo MLX decision model."""

    device = "mlx"

    def __init__(
        self,
        folder: Path,
        fp16_checkpoint: bool = False,
    ):
        import inspect
        import json

        import mlx.core as mx
        from transformers import AutoTokenizer
        from unsloth_zoo.mlx.decision import load_decision_model

        laya = _laya()
        clamp_temperature = laya.common.clamp_temperature
        self.folder = folder
        self.cfg = json.loads((folder / "rl_agent_config.json").read_text(encoding = "utf-8"))
        self.tok = AutoTokenizer.from_pretrained(str(folder / "tokenizer"))
        self.temperature = [
            clamp_temperature(t) for t in self.cfg.get("temperature", [1.0, 1.0, 1.0])
        ]
        self.temperature_by_options = {
            k: clamp_temperature(v) for k, v in self.cfg.get("temperature_by_options", {}).items()
        }
        self._to_internal = laya.agent.Agent._to_internal
        # As _precision does for fp16 checkpoints on CUDA: exact fp16 weights, fp16 matmuls, fp32 norms.
        # An unsloth-zoo without compute_dtype casts the whole model instead, so it stays fp32.
        mixed = "compute_dtype" in inspect.signature(load_decision_model).parameters
        self.dtype = mx.float16 if fp16_checkpoint and mixed and not _fp32_forced() else mx.float32
        self.model = load_decision_model(folder, **({"compute_dtype": self.dtype} if mixed else {}))


def _load_checkpoint(checkpoint: Checkpoint):
    from utils.systemone_settings import get_device as preferred_device

    root = _checkpoint_dir(checkpoint)
    if checkpoint.layout == "clef":
        from .clef_runtime import ClefAgent, ClefWorkerError

        _evict()
        agent = ClefAgent(root, cancelled = _training_active)
        # Training may have started while this loaded, and Clef has no CPU fallback.
        if _training_active():
            agent.close()
            raise ClefWorkerError(
                f"{checkpoint.name} needs the GPU, which a training run took while it loaded."
            )
        return agent, "cuda"
    _laya()

    # Evict only once the new checkpoint is on disk, so a long or failed download leaves the resident model serving.
    _evict()
    # While training holds the GPU, a GPU model loads on CPU, as dictation does; see _misplaced.
    for_training = _training_active() and preferred_device() == "gpu"
    device = "cpu" if for_training else _device()
    folder = root / checkpoint.subfolder if checkpoint.subfolder else root
    fp16_checkpoint = checkpoint.name in CHECKPOINTS or _stored_fp16(folder)
    if device == "mlx":
        return _MLXAgent(folder, fp16_checkpoint), device
    if device not in ("cuda", "cpu"):
        return _load_laya(str(root), subfolder = checkpoint.subfolder, device = device), device
    import torch

    # Built on CPU and cast before the move, so the device never holds laya's fp32 copy.
    weights, _ = _precision(torch.device(device), fp16_checkpoint)
    agent = _load_laya(
        str(root), subfolder = checkpoint.subfolder, device = "cpu", embedding_dtype = weights
    )
    # Training may have started while this built.
    if device != "cpu" and _training_active():
        device, for_training = "cpu", True
    _place(agent, torch.device(device), fp16_checkpoint)
    agent.__dict__["_unsloth_for_training"] = for_training
    return agent, str(agent.device.type)


_build_lock = threading.Lock()


def _load_laya(
    path: str,
    *,
    embedding_dtype = None,
    **kwargs,
):
    """``laya.load`` without randomly initialising the encoder's vocabulary embedding.

    laya builds a randomly initialised fp32 encoder and then loads the checkpoint over it. For mmBERT's
    256000 x 768 embedding that init alone took ~3.2 GB of scratch RAM and ~8 s, all of it overwritten
    by laya's strict load. The embedding is created with ``skip_init`` instead, in the dtype it will be
    served in; the rest of the model, including buffers the checkpoint does not carry, is built as before.
    """
    laya = _laya()
    hook = getattr(laya, "agent", None)
    if not hasattr(hook, "build_model"):
        return laya.load(path, **kwargs)
    with _build_lock:
        original = hook.build_model

        def build_model(cfg, encoder_dir = None):
            return _build_model(cfg, encoder_dir, original, embedding_dtype)

        hook.build_model = build_model
        try:
            return laya.load(path, **kwargs)
        finally:
            hook.build_model = original


def _build_model(
    cfg,
    encoder_dir,
    original,
    embedding_dtype = None,
):
    """laya.common.build_model, with the vocabulary embedding allocated but not initialised."""
    import os

    import torch
    from transformers import AutoConfig, AutoModel

    if not encoder_dir or not os.path.exists(encoder_dir):
        return original(cfg, encoder_dir = encoder_dir)
    config = AutoConfig.from_pretrained(encoder_dir)
    vocab_size, pad_token_id = config.vocab_size, getattr(config, "pad_token_id", None)
    # ModernBERT reads pad_token_id only for this embedding; others keep it (RoBERTa's position ids).
    if config.model_type != "modernbert" or vocab_size <= 1:
        return original(cfg, encoder_dir = encoder_dir)
    # A one-row placeholder, with row 0 standing in for the padding id so the check below can
    # tell the embedding was sized and padded from the config.
    placeholder_pad = None if pad_token_id is None else 0
    config.vocab_size, config.pad_token_id = 1, placeholder_pad
    try:
        encoder = AutoModel.from_config(config, attn_implementation = "sdpa")
    finally:
        config.vocab_size, config.pad_token_id = vocab_size, pad_token_id
    placeholder = encoder.get_input_embeddings()
    if (
        type(placeholder) is not torch.nn.Embedding
        or placeholder.num_embeddings != 1
        or placeholder.padding_idx != placeholder_pad
        or encoder.config.vocab_size != vocab_size
    ):
        # Not laid out like ModernBERT: build it laya's way.
        del encoder, placeholder
        return original(cfg, encoder_dir = encoder_dir)
    encoder.set_input_embeddings(
        torch.nn.utils.skip_init(
            torch.nn.Embedding,
            vocab_size,
            placeholder.embedding_dim,
            padding_idx = pad_token_id,
            max_norm = placeholder.max_norm,
            norm_type = placeholder.norm_type,
            scale_grad_by_freq = placeholder.scale_grad_by_freq,
            sparse = placeholder.sparse,
            dtype = embedding_dtype or placeholder.weight.dtype,
        )
    )
    return _laya().common.DecisionModel(
        encoder, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1
    )


def _stored_fp16(folder: Path) -> bool:
    """Whether the checkpoint's weights are saved as float16, as all three published Laya checkpoints are."""
    import json
    import math
    import struct

    try:
        with open(folder / "model.safetensors", "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    except (OSError, ValueError, struct.error):
        return False
    sizes: dict[str, int] = {}
    for name, meta in header.items():
        if name != "__metadata__" and isinstance(meta, dict):
            sizes[meta["dtype"]] = sizes.get(meta["dtype"], 0) + math.prod(meta["shape"])
    return bool(sizes) and max(sizes, key = sizes.get) == "F16"


def _fp32_forced() -> bool:
    import os
    return os.environ.get("UNSLOTH_SYSTEMONE_FP32", "") == "1"


def _precision(device, fp16_checkpoint: bool):
    """(weight dtype, compute dtype) for ``device``; ``(None, None)`` keeps laya's fp32 weights and compute.

    An fp16 checkpoint held in fp16 is exact (fp16 -> fp32 is lossless), and fp16 compute tracked fp32 about
    10x closer than bf16 on both GPU and CPU. Anything else follows the device: bf16 where it is native, else fp16.
    """
    import platform

    import torch

    if _fp32_forced():
        return None, None
    if device.type == "cuda":
        if fp16_checkpoint:
            return torch.float16, torch.float16
        from core.inference.rocm_bf16 import is_rocm_torch, rocm_bf16_supported

        if is_rocm_torch(torch):
            bf16 = rocm_bf16_supported(torch, device.index)
        else:
            # By capability: pre-Ampere NVIDIA reports is_bf16_supported() through slow emulation.
            bf16 = torch.cuda.get_device_capability(device)[0] >= 8
        dtype = torch.bfloat16 if bf16 else torch.float16
        return dtype, dtype
    # CPU only where the ISA has the format natively (AVX512-FP16 / AMX); emulated fp16 or bf16 is slower than fp32.
    # Measured on x86 only, so other CPUs keep fp32.
    if device.type == "cpu" and platform.machine().lower() in ("x86_64", "amd64"):
        mkldnn = getattr(torch.ops, "mkldnn", None)
        try:
            if fp16_checkpoint and mkldnn._is_mkldnn_fp16_supported():
                return torch.float16, torch.float16
            if not fp16_checkpoint and mkldnn._is_mkldnn_bf16_supported():
                return torch.bfloat16, torch.bfloat16
        except (AttributeError, RuntimeError):
            pass
    return None, None


def _fp32_output(module, args, output):
    return output.float()


def _cast_matmul_weights(model, dtype) -> None:
    """Cast the matmul and lookup weights only.

    Norms stay fp32 (CPU layer_norm rejects low-precision parameters) and embedding outputs return to fp32,
    so the residual stream keeps laya's precision. With bf16 compute this is bit-identical to fp32 weights,
    since autocast rounds each weight to the same bf16 either way.
    """
    import torch.nn as nn
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.to(dtype)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data = module.in_proj_weight.data.to(dtype)
            if module.in_proj_bias is not None:
                module.in_proj_bias.data = module.in_proj_bias.data.to(dtype)
        if isinstance(module, nn.Embedding) and not getattr(module, "_unsloth_fp32_output", False):
            module.register_forward_hook(_fp32_output)
            module._unsloth_fp32_output = True


def _place(agent, device, fp16_checkpoint: bool) -> None:
    """Move a CPU-built laya agent to ``device`` in the precision :func:`_precision` picks for it."""
    import torch

    agent.__dict__["_unsloth_fp16_checkpoint"] = fp16_checkpoint
    # Captured graphs point at the current weights; a move or recast replaces them.
    agent.__dict__.pop("_unsloth_graphs", None)
    weights, compute = _precision(device, fp16_checkpoint)
    # Cast on the CPU side of the move: before moving to an accelerator, after coming back from one.
    if device.type == "cpu":
        agent.model.to(device)
    if weights is None:
        agent.model.float()
    else:
        _cast_matmul_weights(agent.model, weights)
    if device.type != "cpu":
        try:
            agent.model.to(device)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # Same fallback as laya's own loader: a device that cannot hold the model leaves it on CPU.
            logger.warning("Laya could not be placed on %s; running it on CPU", device)
            _place(agent, torch.device("cpu"), fp16_checkpoint)
            _release_memory()
            return
    agent.device, agent.dtype = device, compute or torch.float32


def _hub_download_active(checkpoint: Checkpoint) -> bool:
    # A Hub download job writing the same repo would share blobs with our snapshot_download.
    if checkpoint.is_local:
        return False
    try:
        from hub.utils.download_registry import get_models_registry
        if not get_models_registry().active_job_refs(checkpoint.source):
            return False
    except Exception:
        return False
    return not is_cached(checkpoint)


def loading_repo_ids() -> tuple[str, ...]:
    with _state_lock:
        if _loading is not None and not _loading.is_local:
            return (_loading.source,)
    return ()


def _load(checkpoint: Checkpoint) -> None:
    global _agent, _loaded, _device_name, _loading, _failure
    started = time.monotonic()
    try:
        agent, device = _load_checkpoint(checkpoint)
    except Exception as exc:
        message = f"Could not load {checkpoint.name}: {type(exc).__name__}: {exc}"
        logger.warning("System One load failed: %s", message, exc_info = True)
        with _state_lock:
            _failure = (checkpoint, message, time.monotonic() + FAILURE_BACKOFF_S)
            _loading = None
        return
    with _state_lock:
        _agent, _loaded, _device_name = agent, checkpoint, str(getattr(agent, "device", device))
        _failure = _loading = None
    logger.info(
        "System One loaded %s on %s in %.1fs",
        checkpoint.name,
        _device_name,
        time.monotonic() - started,
    )


def _misplaced() -> bool:
    # The resident model follows the same rule: off the GPU while training holds it, back on it after.
    if getattr(_agent, "_unsloth_for_training", False):
        return not _training_active()
    return _device_name not in (None, "cpu") and _training_active()


def _clef_blocked_by_training(checkpoint: Checkpoint) -> None:
    if checkpoint.layout != "clef":
        return
    from .catalog import clef_unsupported_reason

    if (reason := clef_unsupported_reason()) is not None:
        raise Unavailable(400, "api_usage_error", reason)
    if not _training_active():
        return
    # Clef has no CPU fallback: it waits for the GPU instead of taking it from the run.
    if _loaded is not None and _loaded.layout == "clef":
        _evict()
    raise Unavailable(
        503,
        "model_unavailable",
        f"{checkpoint.name} needs the GPU, which a training run is using; it answers again when the run ends.",
        retry_after = 30,
    )


def _ensure_loading(checkpoint: Checkpoint) -> threading.Thread | None:
    global _loader, _loading
    # Asked before _state_lock: the training backend takes its own lock.
    _clef_blocked_by_training(checkpoint)
    misplaced = _misplaced()
    with _state_lock:
        if _loaded == checkpoint and _agent is not None and not misplaced:
            return None
        if _failure and _failure[0] == checkpoint and time.monotonic() < _failure[2]:
            raise Unavailable(
                503,
                "model_unavailable",
                _failure[1],
                retry_after = max(1.0, _failure[2] - time.monotonic()),
            )
        if _loading is not None:
            if _loading != checkpoint:
                raise Unavailable(
                    503, "model_loading", f"{_loading.name} is loading", retry_after = 5
                )
            if _loader is not None and _loader.is_alive():
                return _loader
            raise Unavailable(503, "model_loading", f"{checkpoint.name} is loading", retry_after = 5)
        # Claimed before the registry probe, so a Hub download admitted from now on sees this load.
        _loading = checkpoint
    # Outside _state_lock: the registry calls loading_repo_ids() while holding its own lock.
    if _hub_download_active(checkpoint):
        with _state_lock:
            _loading = None
        raise Unavailable(503, "model_loading", f"{checkpoint.name} is downloading", retry_after = 5)
    with _state_lock:
        _loader = threading.Thread(
            target = _load, args = (checkpoint,), name = "systemone-load", daemon = True
        )
        _loader.start()
        return _loader


def _agent_for(checkpoint: Checkpoint, wait: float):
    loader = _ensure_loading(checkpoint)
    if loader is not None:
        loader.join(wait)
        if loader.is_alive() or _ensure_loading(checkpoint) is not None:
            raise Unavailable(
                503, "model_loading", f"{checkpoint.name} is still loading", retry_after = 5
            )
    return _agent


def _to_laya(question: dict[str, Any]) -> dict[str, Any]:
    out = {"type": question["type"], "instructions": question.get("instructions") or ""}
    if question.get("criteria") is not None:
        out["criteria"] = question["criteria"]
    return out


# Long states tokenized in growing prefixes; margin keeps kept tokens clear of the cut (prefix tokenization differs).
_PREFIX_MARGIN = 64
_HEAD_CACHE_SIZE = 1024


def _state_ids(tok, state, room: int) -> tuple[list[int], bool]:
    serialize_state = _laya().common.serialize_state

    text = serialize_state(state).replace(tok.mask_token, " ")
    chars = max(4096, room * 16)
    while chars < len(text):
        ids = tok(text[:chars], add_special_tokens = False)["input_ids"]
        if len(ids) > room + _PREFIX_MARGIN:
            return ids[:room], True
        chars *= 4
    ids = tok(text, add_special_tokens = False)["input_ids"]
    return ids[:room], len(ids) > room


def _head(agent, question: dict[str, Any], max_len: int, head_max_len: int):
    import json

    common = _laya().common
    build_sequence = common.build_sequence
    render_options = common.render_options

    cache = agent.__dict__.setdefault("_unsloth_heads", {})
    key = json.dumps(question, ensure_ascii = False)
    if key not in cache:
        internal = agent._to_internal(question)
        ids, markers = build_sequence(agent.tok, "", internal, max_len, head_max_len)
        if len(markers) != len(render_options(internal)):
            raise ValueError("question options exceed head_max_len=%d" % head_max_len)
        if len(cache) >= _HEAD_CACHE_SIZE:
            cache.clear()
        cache[key] = (ids, markers, internal)
    return cache[key]


def _predict(agent, state, questions: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    import numpy as np

    common = _laya().common
    QTYPES = common.QTYPES
    confidence_from_probs = common.confidence_from_probs

    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    names = list(questions)
    heads = []
    for name in names:
        try:
            heads.append(_head(agent, questions[name], max_len, head_max_len))
        except ValueError:
            raise ValueError(
                f"Question {name!r} options exceed the Laya context window ({max_len} tokens). "
                "Use fewer or shorter criteria."
            ) from None
    room = max(0, max_len - min(len(ids) for ids, _, _ in heads))
    state_ids, state_cut = _state_ids(agent.tok, state, room)
    items, truncated = [], False
    for ids, markers, internal in heads:
        keep = max(0, max_len - len(ids))
        truncated = truncated or state_cut or len(state_ids) > keep
        items.append(
            {
                "ids": (ids[:-1] + state_ids[:keep] + ids[-1:])[:max_len],
                "markers": markers,
                "qtype": QTYPES[internal["t"]],
            }
        )
    logits, usage = _forward(agent, items)
    answers = {}
    for row, (name, (_, markers, internal)) in enumerate(zip(names, heads)):
        k = len(markers)
        p = _probabilities(agent, logits[row], k, QTYPES[internal["t"]])
        confidence = round(confidence_from_probs(p, k), 4)
        if internal["t"] == "choice":
            keys = list(internal["crit"])
            answers[name] = {
                "type": "choice",
                "choice": keys[int(p.argmax())],
                "probabilities": {key: round(float(v), 4) for key, v in zip(keys, p)},
                "confidence": confidence,
            }
        elif internal["t"] == "score":
            answers[name] = {
                "type": "score",
                "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(internal["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": confidence,
            }
        else:
            answers[name] = {
                "type": "noul",
                "noul": round(float(p[1]), 4),
                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
            }
    return {"answers": answers, "usage": {"input_tokens": usage, "output_tokens": 0}}, truncated


def _forward(agent, items: list[dict[str, Any]]):
    global _agent, _loaded, _device_name
    import numpy as np
    import torch

    batch = _collate(items, agent.tok.pad_token_id)
    if agent.device == "mlx":
        try:
            logits = _mlx_logits(agent, batch)
            if not np.isfinite(logits).all():
                import mlx.core as mx
                if agent.dtype != mx.float32:
                    logger.warning("Laya overflowed in float16; continuing in float32")
                    agent.model.set_dtype(mx.float32)
                    agent.dtype = mx.float32
                    logits = _mlx_logits(agent, batch)
            return logits, int(batch["attention_mask"].sum())
        except RuntimeError as exc:
            # MLX reports exhausted memory as "[malloc] Unable to allocate ..." or "Insufficient Memory".
            reason = str(exc).lower()
            if "memory" not in reason and "allocate" not in reason:
                raise
        # Past the handler, so the traceback no longer keeps the MLX arrays alive while the CPU copy loads.
        logger.warning("Laya ran out of GPU memory; moving it to CPU")
        agent.model = None
        _release_memory()
        try:
            cpu_device, fp16_checkpoint = torch.device("cpu"), _stored_fp16(Path(agent.folder))
            cpu = _load_laya(
                str(agent.folder),
                device = "cpu",
                embedding_dtype = _precision(cpu_device, fp16_checkpoint)[0],
            )
            _place(cpu, cpu_device, fp16_checkpoint)
        except Exception:
            # Callers hold _run_lock, so drop the half-moved agent here; the next request loads it again.
            _agent = _loaded = _device_name = None
            raise
        agent.model, agent.device, agent.dtype = cpu.model, cpu.device, cpu.dtype
        _device_name = "cpu"
    try:
        logits = _run_model(agent, batch)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        # Same fallback as laya's Agent.predict: a GPU that runs out of memory moves the model to CPU.
        reason = str(exc).lower()
        if agent.device.type == "cpu" or ("memory" not in reason and "cuda" not in reason):
            raise
        logger.warning("Laya ran out of GPU memory; moving it to CPU")
        _place(agent, torch.device("cpu"), agent.__dict__.get("_unsloth_fp16_checkpoint", False))
        _device_name = "cpu"
        _release_memory()
        logits = _run_model(agent, batch)
    if agent.dtype == torch.float16 and not bool(torch.isfinite(logits).all()):
        # fp16 activations overflowed; the fp16 weights widen to fp32 exactly, so rerun at full precision.
        logger.warning("Laya overflowed in float16; continuing in float32")
        # The graphs read the fp16 weights that .float() is about to replace.
        agent.__dict__.pop("_unsloth_graphs", None)
        agent.model.float()
        agent.dtype = torch.float32
        logits = _run_model(agent, batch)
    return logits.float().cpu().numpy(), int(batch["attention_mask"].sum())


def _chunks(batch, budget: int):
    """``batch`` as consecutive row slices of at most ``budget`` padded tokens, each trimmed to its longest row.

    ``budget=None`` keeps the whole batch."""
    rows, tokens = batch["input_ids"].shape
    step = rows if budget is None else max(1, budget // tokens)
    if rows <= step:
        yield batch
        return
    for start in range(0, rows, step):
        part = {name: batch[name][start : start + step] for name in _INPUTS}
        keep = int(part["attention_mask"].sum(1).max())
        part["input_ids"] = part["input_ids"][:, :keep]
        part["attention_mask"] = part["attention_mask"][:, :keep]
        yield part


def _mlx_logits(agent, batch):
    """One MLX request, forwarded in chunks and concatenated on the host."""
    import numpy as np
    return np.concatenate(
        [agent.model.logits(part) for part in _chunks(batch, _chunk_budget("mlx"))]
    )


def _run_model(agent, batch):
    import torch

    if not _marker_head(agent.model):
        return _run_chunk(agent, batch)
    budget = _chunk_budget(agent.device)
    # cat copies, so no result is a view into a CUDA graph's output buffer, which its next replay overwrites.
    return torch.cat([_run_chunk(agent, part) for part in _chunks(batch, budget)])


def _chunk_budget(device):
    """Padded tokens per forward, or None for one forward.

    Splitting only saves memory: each extra forward costs a fixed overhead. On a CPU host memory is rarely the
    limit and that overhead dominates (a 16-core Strix Halo ran the 64-question request 5x slower in 4096-token
    chunks), so the CPU runs one forward. A GPU splits only past a share of its memory, so a large card keeps
    one forward and its speed, and a small one still bounds its peak.
    """
    kind = getattr(device, "type", device)
    budget = _CHUNK_TOKENS.get(kind)
    if kind != "cuda" or budget is None:
        return budget
    import torch

    try:
        total = torch.cuda.get_device_properties(device).total_memory
    except (RuntimeError, AssertionError):
        return budget
    per_token = _TOKEN_BYTES["hip" if torch.version.hip else "cuda"]
    return max(budget, int(total * _GPU_SHARE) // per_token)


def _run_chunk(agent, batch):
    import os

    import torch

    device = agent.device
    fast = _marker_head(agent.model)
    if (
        fast
        and device.type == "cuda"
        and os.environ.get("UNSLOTH_SYSTEMONE_CUDA_GRAPHS", "") != "0"
        # ROCm reports "cuda" too; as in diffusion_cuda_graph.py, graphs are CUDA only.
        and not torch.version.hip
    ):
        graphs = agent.__dict__.get("_unsloth_graphs")
        if graphs is None:
            graphs = agent.__dict__["_unsloth_graphs"] = _CUDAGraphs(agent)
        with torch.inference_mode():
            logits = graphs.run(batch)
        if logits is not None:
            return logits
    with torch.inference_mode(), _autocast(agent):
        args = [batch[name].to(device) for name in _INPUTS]
        if fast:
            # No padding in the batch: no mask, so SDPA may pick its fastest kernel.
            return _decision_logits(
                agent.model, *args, padded = not bool(batch["attention_mask"].all())
            )
        logits, _ = agent.model(*args)
    return logits


_INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
# Smallest padded rows x tokens per forward; a larger request runs as several (see _chunk_budget). None: one forward.
_CHUNK_TOKENS = {"cuda": 16384, "mlx": 8192, "cpu": None}
# Activation bytes per padded token of the worst request (B200 fp16 41 KiB, gfx1151 137 KiB), with headroom.
_TOKEN_BYTES = {"cuda": 48 << 10, "hip": 160 << 10}
# Share of the GPU's total memory one Decision API forward may take before the request is split.
_GPU_SHARE = 0.05
# Padded rows x tokens above which a batch runs eagerly (B200: graphs win up to 8 x 1024, lose at 16 x 512).
_GRAPH_TOKENS = 8192


def _autocast(agent, cache_enabled = True):
    import torch
    device = agent.device
    return torch.autocast(
        device_type = device.type,
        dtype = agent.dtype,
        enabled = device.type in ("cuda", "cpu") and agent.dtype != torch.float32,
        cache_enabled = cache_enabled,
    )


def _collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    """laya's ``collate_items`` for one request (same tensors), filled with numpy instead of per-row copies."""
    import numpy as np
    import torch

    n = len(items)
    lengths = np.fromiter((len(it["ids"]) for it in items), dtype = np.int64, count = n)
    counts = np.fromiter((len(it["markers"]) for it in items), dtype = np.int64, count = n)
    tokens = np.arange(int(lengths.max()))[None, :] < lengths[:, None]
    options = np.arange(int(counts.max()))[None, :] < counts[:, None]
    ids = np.full(tokens.shape, pad_id, dtype = np.int64)
    ids[tokens] = np.fromiter(
        (t for it in items for t in it["ids"]), dtype = np.int64, count = int(lengths.sum())
    )
    positions = np.zeros(options.shape, dtype = np.int64)
    positions[options] = np.fromiter(
        (m for it in items for m in it["markers"]), dtype = np.int64, count = int(counts.sum())
    )
    return {
        "input_ids": torch.from_numpy(ids),
        "attention_mask": torch.from_numpy(tokens.astype(np.int64)),
        "marker_pos": torch.from_numpy(positions),
        "marker_mask": torch.from_numpy(options),
        "qtype": torch.tensor([it["qtype"] for it in items]),
    }


def _marker_head(model) -> bool:
    """Whether :func:`_decision_logits` can stand in for ``model(...)``: laya's pre-norm ReLU head and GELU scorer."""
    import os

    import torch.nn as nn

    if os.environ.get("UNSLOTH_SYSTEMONE_FAST", "") == "0":
        return False
    cached = model.__dict__.get("_unsloth_marker_head")
    if cached is not None:
        return cached
    head, scorer = getattr(model, "head", None), getattr(model, "scorer", None)
    ok = (
        isinstance(head, nn.TransformerEncoder)
        and len(head.layers) > 0
        and head.norm is None
        and all(
            type(layer) is nn.TransformerEncoderLayer
            and layer.norm_first
            and layer.activation_relu_or_gelu == 1
            and type(layer.self_attn) is nn.MultiheadAttention
            and layer.self_attn.batch_first
            and layer.self_attn._qkv_same_embed_dim
            and layer.self_attn.in_proj_bias is not None
            for layer in head.layers
        )
        and isinstance(scorer, nn.Sequential)
        and [type(m) for m in scorer] == [nn.LayerNorm, nn.Linear, nn.GELU, nn.Linear]
        and scorer[2].approximate == "none"
    )
    model.__dict__["_unsloth_marker_head"] = ok
    return ok


def _feed_forward(layer, x):
    import torch.nn.functional as F

    y = F.layer_norm(x, x.shape[-1:], layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)
    y = F.relu(F.linear(y, layer.linear1.weight, layer.linear1.bias), inplace = True)
    return x.add_(F.linear(y, layer.linear2.weight, layer.linear2.bias))


def _head_layer(
    layer,
    x,
    mask,
    markers = None,
):
    """One eval-mode pre-norm ``TransformerEncoderLayer``; with ``markers``, only the rows at those positions.

    Keys and values always span every token, so each marker row is exactly what the full layer gives it.
    """
    import torch
    import torch.nn.functional as F

    attn = layer.self_attn
    batch, length, width = x.shape
    heads = attn.num_heads
    y = F.layer_norm(x, (width,), layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
    if markers is None:
        q, k, v = (
            F.linear(y, attn.in_proj_weight, attn.in_proj_bias)
            .view(batch, length, 3, heads, width // heads)
            .permute(2, 0, 3, 1, 4)
        )
    else:
        index = markers[:, :, None].expand(-1, -1, width)
        k, v = (
            F.linear(y, attn.in_proj_weight[width:], attn.in_proj_bias[width:])
            .view(batch, length, 2, heads, width // heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = F.linear(
            torch.gather(y, 1, index), attn.in_proj_weight[:width], attn.in_proj_bias[:width]
        )
        q = q.view(batch, -1, heads, width // heads).transpose(1, 2)
        x = torch.gather(x, 1, index)
    # Boolean attn_mask is True where attention is allowed (the inverse of src_key_padding_mask); SDPA
    # applies dropout_p even in eval, so it is passed as 0.
    out = F.scaled_dot_product_attention(q, k, v, attn_mask = mask, dropout_p = 0.0)
    out = out.transpose(1, 2).reshape(batch, -1, width)
    x = x.add_(F.linear(out, attn.out_proj.weight, attn.out_proj.bias))
    return _feed_forward(layer, x)


def _decision_logits(
    model,
    input_ids,
    attention_mask,
    marker_pos,
    marker_mask,
    qtype,
    *,
    padded = True,
):
    """The ``logits`` of laya's ``DecisionModel.forward``, computing only what they read.

    The scorer only reads the option markers, so the last head layer runs at the markers alone, and the
    action head Studio never returns is skipped.
    """
    import torch.nn.functional as F

    h = model.encoder(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
    h = h + model.type_emb(qtype)[:, None, :]
    mask = attention_mask.bool()[:, None, None, :] if padded else None
    layers = model.head.layers
    for layer in layers[:-1]:
        h = _head_layer(layer, h, mask)
    m = _head_layer(layers[-1], h, mask, marker_pos.clamp(min = 0))
    norm, up, _, down = model.scorer
    m = F.layer_norm(m, m.shape[-1:], norm.weight, norm.bias, norm.eps)
    logits = (
        F.linear(F.gelu(F.linear(m, up.weight, up.bias)), down.weight, down.bias)
        .squeeze(-1)
        .float()
    )
    return logits.masked_fill(~marker_mask, -1e4)


class _CUDAGraphs:
    """One captured :func:`_decision_logits` per padded (rows, tokens, options) bucket, sharing one memory pool.

    A short request is bound by kernel launches (about 2.5 ms of GPU work in a 12 ms forward); replaying a
    graph launches it all at once. Past ``_GRAPH_TOKENS`` padded tokens the GPU is the bottleneck and
    padding up to a bucket costs more than launches save, so those batches run eagerly. Callers hold
    ``_run_lock``, so one replay runs at a time and each output is read before the next overwrites it.
    """

    ROWS = (1, 2, 4, 8, 16)
    TOKENS = (64, 128, 256, 512, 1024)
    OPTIONS = (2, 4, 8, 16, 32, 64, 128, 256)
    MAX_GRAPHS = 64

    def __init__(self, agent):
        import torch

        self.agent = agent
        # transformers 4.x wraps ModernBERT's embeddings and MLP in torch.compile on CUDA, which cannot run
        # under capture; 5.x dropped that path, so this is what 5.x runs anyway.
        config = getattr(getattr(agent.model, "encoder", None), "config", None)
        if getattr(config, "reference_compile", False) is not False:
            config.reference_compile = False
        self.graphs: dict[tuple[int, int, int], tuple[Any, dict[str, Any], Any]] = {}
        self.broken = False
        self.pool = torch.cuda.graph_pool_handle()
        self.pool_bytes = 0
        # One warm-up stream: the allocator caches blocks per stream, so a new stream per capture would strand them.
        self.stream = torch.cuda.Stream(agent.device)
        free, _ = torch.cuda.mem_get_info(agent.device)
        self.max_pool_bytes = min(1 << 30, free // 10)

    @staticmethod
    def _fit(n, sizes):
        return next((size for size in sizes if n <= size), None)

    def _forward(self, static):
        with _autocast(self.agent, cache_enabled = False):
            return _decision_logits(self.agent.model, *(static[name] for name in _INPUTS))

    def _capture(self, key):
        import torch

        rows, tokens, options = key
        device = self.agent.device
        static = {
            "input_ids": torch.zeros((rows, tokens), dtype = torch.long, device = device),
            "attention_mask": torch.ones((rows, tokens), dtype = torch.long, device = device),
            "marker_pos": torch.zeros((rows, options), dtype = torch.long, device = device),
            "marker_mask": torch.ones((rows, options), dtype = torch.bool, device = device),
            "qtype": torch.zeros((rows,), dtype = torch.long, device = device),
        }
        # Warm up off the default stream first, as torch.cuda.graph requires, so lazy init is not captured.
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream):
            for _ in range(2):
                self._forward(static)
        torch.cuda.current_stream(device).wait_stream(self.stream)
        graph = torch.cuda.CUDAGraph()
        # thread_local: CUDA work on other Studio threads cannot break (or be broken by) this capture.
        with torch.cuda.graph(graph, pool = self.pool, capture_error_mode = "thread_local"):
            logits = self._forward(static)
        return graph, static, logits

    def run(self, batch):
        """Logits for ``batch`` from a graph replay, or None to run it eagerly."""
        import torch

        rows, tokens = batch["input_ids"].shape
        options = batch["marker_pos"].shape[1]
        key = (
            self._fit(rows, self.ROWS),
            self._fit(tokens, self.TOKENS),
            self._fit(options, self.OPTIONS),
        )
        if self.broken or None in key or key[0] * key[1] > _GRAPH_TOKENS:
            return None
        entry = self.graphs.get(key)
        if entry is None:
            if len(self.graphs) >= self.MAX_GRAPHS or self.pool_bytes >= self.max_pool_bytes:
                return None
            before = torch.cuda.memory_reserved(self.agent.device)
            try:
                entry = self._capture(key)
            except Exception:
                # e.g. transformers 4.x builds ModernBERT's sliding-window mask on the CPU every forward.
                logger.warning(
                    "Laya cannot run in CUDA graphs here; running it eagerly", exc_info = True
                )
                self.broken = True
                return None
            self.graphs[key] = entry
            self.pool_bytes += max(0, torch.cuda.memory_reserved(self.agent.device) - before)
            if self.pool_bytes > self.max_pool_bytes:
                # Over budget: the pool is only released with every graph in it, so drop them all and stay eager.
                logger.warning("Laya CUDA graphs need more memory than allowed; running eagerly")
                self.graphs.clear()
                self.broken = True
                return None
        graph, static, logits = entry
        # Padding tokens are masked; padding rows repeat row 0 so every row has a real token to attend to.
        static["input_ids"].zero_()
        static["attention_mask"].zero_()
        static["marker_pos"].zero_()
        static["marker_mask"].zero_()
        for name in _INPUTS:
            source = batch[name]
            if source.dim() == 1:
                static[name][:rows].copy_(source)
            else:
                static[name][:rows, : source.shape[1]].copy_(source)
            if rows < key[0]:
                static[name][rows:] = static[name][:1]
        graph.replay()
        return logits[:rows, :options]


def _probabilities(agent, row, k: int, qtype: int):
    import numpy as np

    temp_bucket = _laya().common.temp_bucket

    scale = agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype])
    z = row[:k] / scale
    p = np.exp(z - z.max())
    return p / p.sum()


def _wire_answer(answer: dict[str, Any]) -> dict[str, Any]:
    kind = answer["type"]
    if kind == "noul":
        return {"type": "noul", "noul": float(answer["noul"])}
    if kind == "choice":
        return {
            "type": "choice",
            "choice": answer["choice"],
            "confidence": float(answer["confidence"]),
            "probabilities": {k: float(v) for k, v in answer["probabilities"].items()},
        }
    return {
        "type": "score",
        "score": float(answer["score"]),
        "confidence": float(answer["confidence"]),
        "legend": answer["legend"],
        "probabilities": {k: float(v) for k, v in answer["probabilities"].items()},
    }


def decide(checkpoint: Checkpoint, state, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not _admission.acquire(blocking = False):
        raise Unavailable(529, "overloaded", "System One is busy; retry shortly", retry_after = 1)
    try:
        return _decide(checkpoint, state, questions)
    finally:
        _admission.release()


def _decide(checkpoint: Checkpoint, state, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    agent = _agent_for(checkpoint, LOAD_WAIT_S)
    if not _run_lock.acquire(timeout = RUN_WAIT_S):
        raise Unavailable(529, "overloaded", "System One is busy; retry shortly", retry_after = 1)
    try:
        if agent is None or _agent is not agent or _loaded != checkpoint:
            raise Unavailable(
                503, "model_loading", f"{checkpoint.name} is reloading", retry_after = 5
            )
        if checkpoint.layout == "clef":
            return _decide_clef(checkpoint, agent, state, questions)
        laya_questions = {name: _to_laya(q) for name, q in questions.items()}
        try:
            result, truncated = _predict(agent, state, laya_questions)
        except ValueError as exc:
            raise Unavailable(422, "invalid_request_error", str(exc)) from None
    finally:
        _run_lock.release()
    return {
        "model": checkpoint.name,
        "answers": {name: _wire_answer(result["answers"][name]) for name in questions},
        "usage": {"input_tokens": int(result["usage"]["input_tokens"]), "output_tokens": 0},
        "truncated": truncated,
    }


def _decide_clef(checkpoint: Checkpoint, agent, state, questions) -> dict[str, Any]:
    from .clef_runtime import ClefWorkerError
    try:
        result = agent.decide(state, questions)
    except ValueError as exc:
        raise Unavailable(422, "invalid_request_error", str(exc)) from None
    except ClefWorkerError as exc:
        # A dead or hung worker is dropped, so the next request starts a fresh one.
        threading.Thread(target = _evict_agent, args = (agent,), daemon = True).start()
        raise Unavailable(503, "model_unavailable", str(exc), retry_after = 5) from None
    return {
        "model": checkpoint.name,
        "answers": {name: _wire_answer(result["answers"][name]) for name in questions},
        "usage": {"input_tokens": int(result["input_tokens"]), "output_tokens": 0},
        "truncated": bool(result["truncated"]),
    }


def _evict_agent(agent) -> None:
    global _agent, _loaded, _device_name
    # Waits for _decide to release _run_lock.
    with _run_lock:
        if _agent is not agent:
            return
        _agent = _loaded = _device_name = None
    _close(agent)


def status() -> dict[str, Any]:
    with _state_lock:
        failure = _failure if _failure and time.monotonic() < _failure[2] else None
        return {
            "loaded_model": _loaded.name if _loaded else None,
            "device": _device_name,
            "loading_model": _loading.name if _loading else None,
            # Kept for the settings API: laya is vendored, so there is never an install in flight.
            "installing": False,
            "error": failure[1] if failure else None,
            "error_model": failure[0].name if failure else None,
        }


def ensure_can_unload() -> None:
    with _state_lock:
        # _loading: a load claimed but not yet started would otherwise land after this unload.
        if _loading is not None or (_loader is not None and _loader.is_alive()):
            raise Unavailable(409, "model_loading", "Wait for the load to finish before unloading")


def unload() -> bool:
    global _agent, _loaded, _device_name, _failure
    ensure_can_unload()
    with _run_lock:
        agent, was_loaded = _agent, _agent is not None
        _agent = _loaded = _device_name = None
        _failure = None
    _close(agent)
    _release_memory()
    return was_loaded
