# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 Gemma 4 text encoder, in a spawned worker on the transformers 5.10 sidecar.

The server process runs Studio's default transformers, which has no ``Gemma4UnifiedForConditionalGeneration``. This
entry point runs in a fresh spawned interpreter: it prepends ``.venv_t5_510`` to ``sys.path`` BEFORE anything imports
transformers (the same activation the chat / training workers use), loads the encoder and tokenizer from the LTX-2.5
pack, and answers encode requests with the packed hidden states diffusers' ``LTX2Pipeline._get_gemma_prompt_embeds``
computes (every hidden state stacked on a last axis and flattened, left padding to ``max_sequence_length``).

Protocol (multiprocessing queues): the worker sends ``{"type": "ready", ...}`` once the model is loaded, then answers
``{"type": "encode", "id", "prompts", "max_sequence_length"}`` with ``{"type": "embeds", "id", "embeds", "mask"}``
(host tensors; bf16 ``[B, L, hidden * (layers + 1)]`` and int64 ``[B, L]``); ``{"type": "shutdown"}`` ends it. Any
failure is ``{"type": "error", "error": "..."}``. Must import nothing heavy at module level.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Optional

_BACKEND_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _version_tuple(text: str) -> tuple[int, ...]:
    out = []
    for part in str(text).split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def _activate_sidecar(venv_dir: str) -> None:
    # A transformers already imported here (a parent __main__ re-imported by spawn) shadows the sidecar; the version
    # check in _load then names the problem.
    if not venv_dir or not os.path.isdir(venv_dir):
        raise RuntimeError(f"the transformers 5.10 sidecar is missing at '{venv_dir}'")
    if venv_dir not in sys.path:
        sys.path.insert(0, venv_dir)
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = venv_dir + (os.pathsep + pythonpath if pythonpath else "")


def _single_file_folder(config: dict) -> str:
    """A scratch folder that reads like a transformers checkpoint for the single-file release's encoder: its embedded
    config and tokenizer written out, the weights linked in (renamed at load by ``key_mapping``)."""
    import json
    import tempfile

    from safetensors import safe_open

    repo, name = config["repo"], config["single_file"]
    if os.path.isdir(os.path.expanduser(repo)):
        path = os.path.join(os.path.expanduser(repo), name)
        if not os.path.isfile(path):
            raise RuntimeError(f"'{name}' is missing from '{repo}'")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo,
            name,
            token = config.get("hf_token") or None,
            cache_dir = config.get("cache_dir") or None,
            local_files_only = bool(config.get("local_files_only")),
        )
    weights = os.path.realpath(path)
    folder = tempfile.mkdtemp(prefix = "unsloth_ltx25_te_")
    try:
        _link_weights(weights, os.path.join(folder, "model.safetensors"))
    except OSError:
        # The temp dir is on another volume than the weights, so no hard link: the scratch folder goes beside them.
        os.rmdir(folder)
        folder = tempfile.mkdtemp(prefix = ".unsloth_ltx25_te_", dir = os.path.dirname(weights))
        try:
            _link_weights(weights, os.path.join(folder, "model.safetensors"))
        except OSError as exc:
            os.rmdir(folder)
            raise RuntimeError(
                f"could not link '{name}' into a scratch checkpoint folder without copying it: {exc}"
            ) from exc
    try:
        with safe_open(path, framework = "pt", device = "cpu") as handle:
            meta = handle.metadata() or {}
            if "gemma_config" not in meta:
                raise RuntimeError(f"'{name}' has no embedded encoder config")
            with open(os.path.join(folder, "config.json"), "w", encoding = "utf-8") as fh:
                json.dump(json.loads(meta["gemma_config"]), fh)
            keys = set(handle.keys())
            for tensor_name, file_name in (
                ("tokenizer_json", "tokenizer.json"),
                ("hf_asset__tokenizer_config.json", "tokenizer_config.json"),
            ):
                if tensor_name not in keys:
                    raise RuntimeError(f"'{name}' has no embedded {file_name}")
                with open(os.path.join(folder, file_name), "wb") as fh:
                    fh.write(handle.get_tensor(tensor_name).numpy().tobytes())
    except BaseException:
        import shutil
        shutil.rmtree(folder, ignore_errors = True)
        raise
    return folder


def _link_weights(src: str, dst: str) -> None:
    """Link ``src`` at ``dst`` without copying the 24 GB file: a symlink, else a hard link. Windows refuses symlinks
    without Developer Mode or administrator rights (WinError 1314); a hard link needs neither, only the same volume."""
    try:
        os.symlink(src, dst)
    except OSError:
        os.link(src, dst)


def _load(config: dict) -> tuple[Any, Any]:
    import torch
    import transformers

    minimum = config.get("min_transformers") or "5.10.1"
    if _version_tuple(transformers.__version__) < _version_tuple(minimum):
        raise RuntimeError(
            f"the sidecar provides transformers {transformers.__version__}, but LTX-2.5's Gemma 4 encoder needs "
            f">= {minimum}"
        )
    model_cls = getattr(transformers, "Gemma4UnifiedForConditionalGeneration", None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} in the sidecar has no Gemma4UnifiedForConditionalGeneration"
        )
    dtype = getattr(torch, str(config.get("dtype") or "bfloat16"))
    common = {
        "token": config.get("hf_token") or None,
        "cache_dir": config.get("cache_dir") or None,
        "local_files_only": bool(config.get("local_files_only")),
    }
    if config.get("single_file"):
        import shutil
        folder = _single_file_folder(config)
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(folder)
            model = model_cls.from_pretrained(
                folder, torch_dtype = dtype, key_mapping = config.get("key_mapping") or None
            )
        finally:
            shutil.rmtree(folder, ignore_errors = True)
    else:
        repo = config["repo"]
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            repo, subfolder = "tokenizer", **common
        )
        model = model_cls.from_pretrained(
            repo, subfolder = "text_encoder", torch_dtype = dtype, **common
        )
    model.eval()
    model.requires_grad_(False)
    _truncate_dead_tail(tokenizer, model)
    if config.get("te_quant") == TE_QUANT_INT8_CONVROT:
        quantize_int8_convrot_(model, config.get("device") or "cpu", bool(config.get("resident")))
    elif config.get("te_quant"):
        raise RuntimeError(f"unsupported text encoder precision '{config.get('te_quant')}'")
    if config.get("resident"):
        model.to(config["device"])
    return tokenizer, model


# Matches ComfyUI's gemma4 int8-convrot encoder, built from the bf16 weights on disk (no second 15 GB download).
TE_QUANT_INT8_CONVROT = "int8"
CONVROT_GROUP = 256
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _decoder_layers(model: Any) -> Any:
    inner = getattr(model, "model", model)
    language = getattr(inner, "language_model", inner)
    layers = getattr(language, "layers", None)
    if layers is None:
        raise RuntimeError("the Gemma 4 encoder has no language_model.layers to quantize")
    return layers


_MADV_DONTNEED = 4


def _file_mapping_holds(start: int, end: int) -> bool:
    """Whether ``[start, end)`` lies inside one file-backed mapping (Hub blobs have no suffix, so any file qualifies)."""
    try:
        with open("/proc/self/maps", "r", encoding = "utf-8", errors = "replace") as fh:
            for line in fh:
                fields = line.split(None, 5)
                if len(fields) < 6:
                    continue
                lo, _, hi = fields[0].partition("-")
                if int(lo, 16) <= start and end <= int(hi, 16):
                    return fields[4] != "0" and fields[5].strip().startswith("/")
    except (OSError, ValueError):
        return False
    return False


def _release_mapped_pages(tensor: Any) -> bool:
    """MADV_DONTNEED file-mapped pages only (DONTNEED zeroes anonymous memory); whether it did."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes
        import mmap

        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            return False
        start = int(tensor.data_ptr())
        end = start + tensor.numel() * tensor.element_size()
        if end <= start or not _file_mapping_holds(start, end):
            return False
        page = mmap.PAGESIZE
        lo = (start + page - 1) // page * page
        hi = end // page * page
        if hi <= lo:
            return True
        libc = ctypes.CDLL(None, use_errno = True)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        libc.madvise.restype = ctypes.c_int
        return libc.madvise(lo, hi - lo, _MADV_DONTNEED) == 0
    except Exception:  # noqa: BLE001 - best effort: the pages then stay mapped until the copy-out below
        return False


def _empty_device_cache(device: Any) -> None:
    import torch
    empty = getattr(getattr(torch, device.type, None), "empty_cache", None)
    if callable(empty):
        empty()


def _quantize_off_card(weight: Any, work: Any) -> tuple[Any, Any, Any]:
    """``(codes, scale, work)``; the card only speeds this up, so its OOM moves this and later projections to the host."""
    import torch

    from core.inference.diffusion_convrot import quantize_convrot_int8_weight

    if work.type != "cpu":
        oom = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
        try:
            return (*quantize_convrot_int8_weight(weight.detach().to(work), CONVROT_GROUP), work)
        except oom:
            pass
        # Outside the handler: the failed attempt's tensors are unreferenced by now.
        _empty_device_cache(work)
        work = torch.device("cpu")
    return (*quantize_convrot_int8_weight(weight.detach(), CONVROT_GROUP), work)


def quantize_int8_convrot_(
    model: Any,
    device: Any = "cpu",
    resident: bool = False,
) -> int:
    """Swap every decoder-layer projection for the ConvRot INT8 linear in place; the number swapped. ``resident``
    quantizes layer by layer on the card. Otherwise mapped bf16 pages are released per projection; if any cannot be,
    host tensors are copied out of the map so it can close."""
    import torch
    from torch import nn

    from core.inference.diffusion_convrot import quantize_convrot_int8_weight
    from core.inference.video_minimax_h3_te import _int8_convrot_linear_class

    cls = _int8_convrot_linear_class()
    work = torch.device(str(device))
    if work.type not in ("cuda", "xpu") or not getattr(torch, work.type).is_available():
        work = torch.device("cpu")
    on_card = resident and work.type != "cpu"
    swapped = 0
    released_all = True
    quantized: set = set()
    for layer in _decoder_layers(model):
        if on_card:
            layer.to(work)
        for parent in (getattr(layer, "self_attn", None), getattr(layer, "mlp", None)):
            if parent is None:
                continue
            for name in _PROJECTIONS:
                linear = getattr(parent, name, None)
                if not isinstance(linear, nn.Linear):
                    continue
                home = linear.weight.device
                if on_card:
                    codes, scale = quantize_convrot_int8_weight(
                        linear.weight.detach().to(work), CONVROT_GROUP
                    )
                else:
                    codes, scale, work = _quantize_off_card(linear.weight, work)
                if not on_card:
                    released_all = _release_mapped_pages(linear.weight) and released_all
                new = cls(
                    linear.in_features, linear.out_features, linear.bias is not None, CONVROT_GROUP
                )
                new.weight = codes.to(home)
                new.weight_scale = scale.to(home)
                if linear.bias is not None:
                    new.bias = linear.bias.detach().to(device = home, dtype = torch.bfloat16)
                setattr(parent, name, new)
                quantized.add(id(new))
                swapped += 1
                del linear, codes, scale
    if swapped == 0:
        raise RuntimeError("no Gemma 4 decoder projection could be quantized")
    if not on_card and work.type != "cpu":
        # The plan never priced this transient.
        _empty_device_cache(work)
    if not on_card and not released_all:
        with torch.no_grad():
            for module in model.modules():
                if id(module) in quantized:
                    continue
                for table in (module._parameters, module._buffers):
                    for key, tensor in table.items():
                        if tensor is not None and tensor.device.type == "cpu":
                            tensor.data = tensor.data.clone()
    return swapped


# Room an onloaded encode needs beyond the weights: every hidden state stacked for the packed embeddings (49 x 1024 x
# 3840 bf16 is ~0.4 GB per prompt, twice over while stacking) plus the forward's own activations.
ENCODE_MARGIN_MIB = 4096
# This process's own CUDA context, which the budget must also hold.
WORKER_CONTEXT_MIB = 768


def _weights_mib(model: Any) -> int:
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total += tensor.numel() * tensor.element_size()
    return total >> 20


def _weights_mib_or_none(model: Any) -> Optional[int]:
    try:
        return _weights_mib(model)
    except Exception:  # noqa: BLE001 -- reporting only
        return None


def _encode_device(
    config: dict,
    model: Any,
    server_reserved_mib: Optional[int] = None,
) -> str:
    """Where a non-resident encoder runs this encode: the configured accelerator when it has room for the weights plus
    ``ENCODE_MARGIN_MIB``, else the CPU. Room is the card's free memory right now AND, when the server sent them, the
    load's memory budget minus what the server holds at this moment: the plan did not budget the encoder on the card,
    so it may only borrow budget the server is not using (encodes run before the denoise). An unreadable free count
    falls back to the budget alone; an XPU with neither reading encodes on the CPU."""
    device = str(config["device"])
    kind = device.split(":", 1)[0]
    # MPS shares host memory, so there is no card to keep the encoder off.
    if config.get("resident") or kind not in ("cuda", "xpu"):
        return device
    import torch

    need = _weights_mib(model) + ENCODE_MARGIN_MIB
    room: Optional[int] = None
    try:
        free, _total = getattr(torch, kind).mem_get_info(torch.device(device))
        room = free >> 20
    except Exception:  # noqa: BLE001 -- judged by the budget alone
        pass
    budget = config.get("budget_mib")
    if budget is not None and server_reserved_mib is not None:
        allowed = int(budget) - int(server_reserved_mib) - WORKER_CONTEXT_MIB
        room = allowed if room is None else min(room, allowed)
    if room is None:
        return device if kind == "cuda" else "cpu"
    return device if room >= need else "cpu"


def _memory_stats(device: str) -> dict:
    """This worker's own peaks: accelerator bytes since the last reset and the host resident set."""
    stats: dict = {}
    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            dev = torch.device(device)
            stats["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated(dev) >> 20
            stats["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved(dev) >> 20
            stats["gpu_allocated_mib"] = torch.cuda.memory_allocated(dev) >> 20
            torch.cuda.reset_peak_memory_stats(dev)
    except Exception:  # noqa: BLE001 -- stats are best-effort
        pass
    try:
        import resource
        stats["host_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 10
        with open("/proc/self/statm", encoding = "ascii") as fh:
            stats["host_rss_mib"] = (int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")) >> 20
    except Exception:  # noqa: BLE001 -- not Linux
        pass
    return stats


def _pack_hidden_states(model: Any, ids: Any, mask: Any) -> Any:
    """``LTX2Pipeline._get_gemma_prompt_embeds``: every hidden state stacked on a last axis, flattened."""
    import torch

    out = model(input_ids = ids, attention_mask = mask, output_hidden_states = True)
    return torch.stack(out.hidden_states, dim = -1).flatten(2, 3).to(dtype = model.dtype)


def _truncate_dead_tail(tokenizer: Any, model: Any) -> dict:
    """All hidden states are read, so only the unused 262k-vocab logits head goes."""
    from core.inference.diffusion_te_truncate import truncate_text_encoders

    class _Probe:
        text_encoder = model

        @staticmethod
        def encode_prompt(prompt: str, device: Any = None) -> Any:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            inputs = tokenizer(
                [prompt],
                padding = "max_length",
                max_length = 32,
                truncation = True,
                add_special_tokens = True,
                return_tensors = "pt",
            )
            return _pack_hidden_states(
                model, inputs.input_ids.to(device), inputs.attention_mask.to(device)
            )

    return truncate_text_encoders(_Probe())


def _encode(
    tokenizer: Any, model: Any, config: dict, prompts: list, max_len: int
) -> tuple[Any, Any]:
    embeds, mask, _device = _encode_on(tokenizer, model, config, prompts, max_len)
    return embeds, mask


def _encode_on(
    tokenizer: Any,
    model: Any,
    config: dict,
    prompts: list,
    max_len: int,
    server_reserved_mib: Optional[int] = None,
) -> tuple[Any, Any, str]:
    import torch

    resident = bool(config.get("resident"))
    device = _encode_device(config, model, server_reserved_mib)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        [str(p).strip() for p in prompts],
        padding = "max_length",
        max_length = int(max_len),
        truncation = True,
        add_special_tokens = True,
        return_tensors = "pt",
    )
    onloaded = not resident and device != "cpu"
    if onloaded:
        model.to(device)
    try:
        with torch.inference_mode():
            ids = inputs.input_ids.to(device)
            mask = inputs.attention_mask.to(device)
            embeds = _pack_hidden_states(model, ids, mask)
            embeds_host = embeds.to("cpu")
            mask_host = mask.to("cpu")
            del embeds
    finally:
        if onloaded:
            model.to("cpu")
        # Return the encode's activations (and, onloaded, the weights) to the card: a resident worker otherwise keeps
        # ~1.7 GB cached through the denoise, an onloaded one the whole encoder.
        kind = str(device).split(":", 1)[0]
        accel = getattr(torch, kind, None) if kind in ("cuda", "xpu") else None
        if accel is not None and accel.is_available():
            accel.empty_cache()
    return embeds_host, mask_host, device


def run_text_encoder_worker(*, cmd_queue: Any, resp_queue: Any, config: dict) -> None:
    if _BACKEND_PATH not in sys.path:
        sys.path.insert(0, _BACKEND_PATH)
    try:
        _activate_sidecar(config.get("venv_dir") or "")
        tokenizer, model = _load(config)
        import transformers

        resp_queue.put(
            {
                "type": "ready",
                "transformers": transformers.__version__,
                "model_type": getattr(model.config, "model_type", None),
                "hidden_size": getattr(
                    getattr(model.config, "text_config", model.config), "hidden_size", None
                ),
                "resident": bool(config.get("resident")),
                "te_quant": config.get("te_quant") or None,
                "weights_mib": _weights_mib_or_none(model),
                **_memory_stats(config.get("device") or ""),
            }
        )
    except Exception as exc:  # noqa: BLE001 -- every failure goes back to the server as a message
        resp_queue.put(
            {
                "type": "error",
                "error": f"LTX-2.5 text encoder failed to load: {exc}",
                "stack": traceback.format_exc(limit = 20),
            }
        )
        return
    while True:
        msg = cmd_queue.get()
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "shutdown" or kind is None:
            return
        if kind != "encode":
            continue
        try:
            embeds, mask, device = _encode_on(
                tokenizer,
                model,
                config,
                msg["prompts"],
                msg.get("max_sequence_length") or 1024,
                msg.get("server_reserved_mib"),
            )
            resp_queue.put(
                {
                    "type": "embeds",
                    "id": msg.get("id"),
                    "embeds": embeds,
                    "mask": mask,
                    "device": device,
                    **_memory_stats(config.get("device") or ""),
                }
            )
        except Exception as exc:  # noqa: BLE001 -- report, keep serving
            resp_queue.put(
                {
                    "type": "error",
                    "id": msg.get("id"),
                    "error": f"LTX-2.5 text encoder failed: {exc}",
                    "stack": traceback.format_exc(limit = 20),
                }
            )
