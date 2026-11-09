# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 Gemma 4 encoder worker; prepends .venv_t5_510 to sys.path before any transformers import;
must import nothing heavy at module level"""

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
            **({"revision": config["revision"]} if config.get("revision") else {}),
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
        # The load plan's commit for the base, when it pinned one.
        **({"revision": config["revision"]} if config.get("revision") else {}),
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
        model = None
        if config.get("te_quant") == TE_QUANT_INT8_CONVROT and config.get("te_prequant"):
            try:
                model = _load_hosted(config, model_cls, dtype, common)
                config["_te_source"] = "hosted"
            except Exception as exc:  # noqa: BLE001 -- the dense encoder quantized at load is the same build
                _require_room_for_dense(config, exc)
                print(
                    f"video.ltx25_te: hosted int8 ConvRot encoder unavailable ({exc}); quantizing the dense "
                    "encoder at load",
                    file = sys.stderr,
                    flush = True,
                )
                model = None
        if model is None:
            model = _load_dense(config, model_cls, dtype, common)
    model.eval()
    model.requires_grad_(False)
    _truncate_dead_tail(tokenizer, model)
    if config.get("te_quant") == TE_QUANT_INT8_CONVROT:
        if config.get("_te_source") != "hosted":
            quantize_int8_convrot_(
                model, config.get("device") or "cpu", bool(config.get("resident"))
            )
            config["_te_source"] = "quantized_at_load"
    elif config.get("te_quant"):
        raise RuntimeError(f"unsupported text encoder precision '{config.get('te_quant')}'")
    if config.get("resident"):
        model.to(config["device"])
    return tokenizer, model


# Matches ComfyUI's gemma4 int8-convrot encoder. Loaded from the hosted build of this same quantization
# (scripts/build_ltx25_te_int8_convrot.py) when the server hands one over, else built from the bf16 weights at load.
TE_QUANT_INT8_CONVROT = "int8"
CONVROT_GROUP = 256
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _available_host_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 -- no psutil in this venv
        pass
    try:
        with open("/proc/meminfo", encoding = "utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


_DTYPE_SCALE_VS_BF16 = {"float32": 2.0, "float": 2.0, "float64": 4.0, "double": 4.0}


def _dense_scale_vs_bf16(dtype: Any) -> float:
    return _DTYPE_SCALE_VS_BF16.get(
        str(dtype or "bfloat16").replace("torch.", "").strip().lower(), 1.0
    )


def _require_room_for_dense(config: dict, exc: Exception) -> None:
    """The load was planned at the hosted file's size: refuse the dense fallback when free memory cannot hold the
    dense encoder, rather than let the OS kill the process mid-read (on Apple Silicon the same pool as the DiT).
    ``dense_te_gb`` is bf16; a float32 load (fp16 promoted without bf16) needs twice that."""
    need = (
        float(config.get("dense_te_gb") or 0.0)
        * 1000**3
        * _dense_scale_vs_bf16(config.get("dtype"))
    )
    free = _available_host_bytes()
    if need and free is not None and free < need:
        raise RuntimeError(
            f"The hosted int8 text encoder could not be loaded ({exc}), and quantizing the dense encoder at load "
            f"needs about {need / 1000**3:.0f} GB of free memory ({free / 1000**3:.0f} GB free)."
        ) from exc


def _load_dense(config: dict, model_cls: Any, dtype: Any, common: dict) -> Any:
    """The pack's bf16 encoder. A scoped snapshot whose dense shards the plan dropped for the hosted file has none to
    open, so a failed read retries the pack's Hub id (``hub_repo``) the way a fresh load would fetch it."""
    repo, hub_repo = config["repo"], config.get("hub_repo")
    try:
        return model_cls.from_pretrained(
            repo, subfolder = "text_encoder", torch_dtype = dtype, **common
        )
    except Exception:
        # Offline too: a complete dense snapshot may already be cached under the Hub id.
        if not hub_repo or hub_repo == repo:
            raise
    return model_cls.from_pretrained(
        hub_repo, subfolder = "text_encoder", torch_dtype = dtype, **common
    )


def _hosted_path(spec: dict, common: dict) -> str:
    mirror = spec.get("mirror_path")
    if mirror and os.path.isfile(mirror):
        return mirror
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    # ``common`` may pin the BASE repo's commit; the hosted file lives in its own repo, so that revision never applies.
    kwargs = {k: v for k, v in common.items() if k != "revision"}
    # The server stages this file with reuse_other_cache_root (hf_hub_download_with_xet_fallback), so a copy cached
    # only under huggingface_hub's own root counts as staged and the dense shards are dropped. Resolve it the same way,
    # else a local_files_only load misses it here and has no dense encoder to fall back to.
    if kwargs.get("cache_dir"):
        try:
            here = try_to_load_from_cache(
                spec["repo"], spec["filename"], cache_dir = kwargs["cache_dir"]
            )
            if not isinstance(here, str):
                elsewhere = try_to_load_from_cache(spec["repo"], spec["filename"], cache_dir = None)
                if isinstance(elsewhere, str) and os.path.isfile(elsewhere):
                    kwargs["cache_dir"] = None
        except Exception:  # noqa: BLE001 -- an unreadable cache keeps the configured root
            pass
    return hf_hub_download(spec["repo"], spec["filename"], **kwargs)


def _load_hosted(config: dict, model_cls: Any, dtype: Any, common: dict) -> Any:
    """The hosted int8 ConvRot encoder: the skeleton meta-initialised from the pack's text_encoder config, every
    decoder projection swapped for ``Int8ConvRotLinear``, then the file assigned in strictly (the tied lm_head is the
    one key it lacks, re-tied to the embedding). Raises on any mismatch; the caller falls back to the at-load build."""
    import torch
    import transformers
    from accelerate import init_empty_weights
    from torch import nn

    from core.inference.prequant_safetensors import load_plain_prequant_safetensors
    from core.inference.video_ltx25 import LTX25_TE_HOSTED_FORMAT
    from core.inference.video_minimax_h3_te import _int8_convrot_linear_class, _meta_tensor_names

    spec = config["te_prequant"]
    ckpt = load_plain_prequant_safetensors(_hosted_path(spec, common))
    if ckpt.get("format") != LTX25_TE_HOSTED_FORMAT:
        raise ValueError(f"not an LTX-2.5 int8 ConvRot encoder (format {ckpt.get('format')!r})")
    meta = ckpt.get("metadata") or {}
    quant = meta.get("quant") or {}
    if str(meta.get("base_model_id") or "").lower() != str(spec.get("base") or "").lower():
        raise ValueError(f"built from {meta.get('base_model_id')!r}, not {spec.get('base')!r}")
    if meta.get("scheme") != "int8" or quant.get("group_size") != CONVROT_GROUP:
        raise ValueError(
            f"unexpected scheme {meta.get('scheme')!r} / group {quant.get('group_size')!r}"
        )
    state_dict = ckpt["state_dict"]
    for key, tensor in list(state_dict.items()):
        # The int8 codes and fp32 scales keep their storage; dense tensors follow the load dtype, as from_pretrained.
        if (
            tensor.is_floating_point()
            and not key.endswith(".weight_scale")
            and tensor.dtype != dtype
        ):
            state_dict[key] = tensor.to(dtype)
    repo, hub_repo = config["repo"], config.get("hub_repo")
    try:
        cfg = transformers.AutoConfig.from_pretrained(repo, subfolder = "text_encoder", **common)
    except Exception:
        if not hub_repo or hub_repo == repo:
            raise
        cfg = transformers.AutoConfig.from_pretrained(hub_repo, subfolder = "text_encoder", **common)
    previous = torch.get_default_dtype()
    # from_pretrained(torch_dtype = bf16) builds under the same default, so buffers made in __init__ match it.
    torch.set_default_dtype(dtype)
    try:
        with init_empty_weights(include_buffers = False):
            model = model_cls(cfg)
    finally:
        torch.set_default_dtype(previous)
    cls = _int8_convrot_linear_class()
    for layer in _decoder_layers(model):
        for parent in (getattr(layer, "self_attn", None), getattr(layer, "mlp", None)):
            if parent is None:
                continue
            for name in _PROJECTIONS:
                linear = getattr(parent, name, None)
                if isinstance(linear, nn.Linear):
                    setattr(
                        parent,
                        name,
                        cls(
                            linear.in_features,
                            linear.out_features,
                            linear.bias is not None,
                            CONVROT_GROUP,
                        ),
                    )
    result = model.load_state_dict(state_dict, strict = False, assign = True)
    tied = {"lm_head.weight"} if getattr(cfg, "tie_word_embeddings", False) else set()
    missing = set(result.missing_keys) - tied
    if missing or result.unexpected_keys:
        raise ValueError(
            f"hosted encoder does not match the model: missing {sorted(missing)[:3]}, "
            f"unexpected {sorted(result.unexpected_keys)[:3]}"
        )
    tie = getattr(model, "tie_weights", None)
    if callable(tie):
        tie()
    stranded = _meta_tensor_names(model)
    if stranded:
        raise ValueError(f"{len(stranded)} tensor(s) still on the meta device, e.g. {stranded[0]}")
    swapped = sum(1 for m in model.modules() if isinstance(m, cls))
    if swapped != int(quant.get("linears") or -1):
        raise ValueError(
            f"{swapped} projections swapped, the file quantized {quant.get('linears')}"
        )
    return model


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
    need = _weights_mib(model) + ENCODE_MARGIN_MIB
    room = _encode_room_mib(config, server_reserved_mib)
    if room is None:
        return device if kind == "cuda" else "cpu"
    return device if room >= need else "cpu"


def _encode_room_mib(config: dict, server_reserved_mib: Optional[int] = None) -> Optional[int]:
    """MiB an encode may use on the configured card (see ``_encode_device``), or None when nothing could be read."""
    import torch

    device = str(config["device"])
    kind = device.split(":", 1)[0]
    room: Optional[int] = None
    try:
        free, total = getattr(torch, kind).mem_get_info(torch.device(device))
        share = float(config.get("vram_reserve_share") or 0.0)
        room = (int(free) - int(int(total) * share)) >> 20
    except Exception:  # noqa: BLE001 -- judged by the budget alone
        pass
    budget = config.get("budget_mib")
    if budget is not None and server_reserved_mib is not None:
        allowed = int(budget) - int(server_reserved_mib) - WORKER_CONTEXT_MIB
        room = allowed if room is None else min(room, allowed)
    return room


# Towers a text encode never runs; a streamed encode leaves them on the host.
_TOWER_TOKENS = ("visual", "vision", "audio")


def _tensor_tables(module: Any) -> tuple:
    return (module._parameters, module._buffers)


def _stream_parts(model: Any) -> Optional[tuple[list, list, list]]:
    """``(resident, containers, layers)`` for a layer-streamed encode, or None when the encoder cannot stream.
    A module overriding ``_apply`` may keep tensors outside its tables, so it declines streaming."""
    import torch

    from core.inference.diffusion_te_truncate import is_passthrough

    try:
        stack = _decoder_layers(model)
    except Exception:  # noqa: BLE001 -- not the expected Gemma 4 layout
        return None
    if not isinstance(model, torch.nn.Module) or not isinstance(stack, torch.nn.ModuleList):
        return None
    layers = [m for m in stack if not is_passthrough(m)]
    if not layers:
        return None
    resident: list = []
    containers: list = []

    def walk(module: Any) -> None:
        containers.append(module)
        for name, child in module.named_children():
            if child is stack or any(token in name.lower() for token in _TOWER_TOKENS):
                continue
            if any(m is stack for m in child.modules()):
                walk(child)
            else:
                resident.append(child)

    walk(model)
    for module in [*containers, *[s for m in (*resident, *layers) for s in m.modules()]]:
        if type(module)._apply is not torch.nn.Module._apply:
            return None
    return resident, containers, layers


def _tensors_mib(modules: list, recurse: bool = True) -> int:
    seen: set = set()
    total = 0
    for module in modules:
        for sub in module.modules() if recurse else (module,):
            for table in _tensor_tables(sub):
                for tensor in table.values():
                    if tensor is None or id(tensor) in seen:
                        continue
                    seen.add(id(tensor))
                    total += tensor.numel() * tensor.element_size()
    return total >> 20


def _stream_need_mib(parts: tuple[list, list, list]) -> int:
    """Resident modules + two layers (allocator may still hold the last) + the encode margin."""
    resident, containers, layers = parts
    biggest = max(_tensors_mib([layer]) for layer in layers)
    return (
        _tensors_mib(resident)
        + _tensors_mib(containers, recurse = False)
        + 2 * biggest
        + ENCODE_MARGIN_MIB
    )


def _encode_plan(
    config: dict,
    model: Any,
    server_reserved_mib: Optional[int] = None,
) -> tuple[str, Optional[tuple[list, list, list]]]:
    """``(device, parts)``: an encoder too big to onload whole streams its layers (``parts`` set) instead of CPU."""
    device = _encode_device(config, model, server_reserved_mib)
    configured = str(config["device"])
    if device != "cpu" or configured.split(":", 1)[0] not in ("cuda", "xpu"):
        return device, None
    if config.get("resident") or (os.environ.get(STREAM_ENCODE_ENV) or "").strip() == "0":
        return device, None
    try:
        parts = _stream_parts(model)
        if parts is None:
            return device, None
        room = _encode_room_mib(config, server_reserved_mib)
        if room is None or room < _stream_need_mib(parts):
            return device, None
    except Exception:  # noqa: BLE001 -- any doubt keeps the CPU encode
        return device, None
    return configured, parts


def _stream_need_mib_or_none(config: dict, model: Any) -> Optional[int]:
    """``_encode_plan``'s streamed need, or None when this encoder never streams."""
    if config.get("resident") or (os.environ.get(STREAM_ENCODE_ENV) or "").strip() == "0":
        return None
    if str(config.get("device") or "").split(":", 1)[0] not in ("cuda", "xpu"):
        return None
    try:
        parts = _stream_parts(model)
        return None if parts is None else _stream_need_mib(parts)
    except Exception:  # noqa: BLE001 -- reporting only
        return None


# "0" turns the layer-streamed encode off (a card too small for the whole encoder then encodes on the CPU again).
STREAM_ENCODE_ENV = "UNSLOTH_LTX25_TE_STREAM"


def _to_device(tensor: Any, device: Any) -> Any:
    """Device copy; non-blocking from pinned memory (same-stream reader, host copy never written)."""
    try:
        pinned = tensor.device.type == "cpu" and tensor.is_pinned()
    except Exception:  # noqa: BLE001 -- unknown: the blocking copy
        pinned = False
    return tensor.to(device, copy = True, non_blocking = pinned)


def _onloads_whole(model: Any) -> bool:
    """Whether ``_Onload`` reaches every tensor ``model.to`` would move: no module overrides ``_apply``."""
    import torch
    try:
        return all(type(m)._apply is torch.nn.Module._apply for m in model.modules())
    except Exception:  # noqa: BLE001 -- unknown: keep ``model.to``
        return False


def _discrete_card(device: Any) -> bool:
    """A CUDA card with its own memory. On unified memory (Apple, Jetson / GB10, Strix Halo) a kept host copy would
    hold the encoder twice in the same RAM during the encode, so those keep ``model.to``."""
    if str(device).split(":", 1)[0] != "cuda":
        return False
    try:
        from core.inference.host_pinning import host_pinning_allowed
        return bool(host_pinning_allowed(device))
    except Exception:  # noqa: BLE001 -- unclassifiable: keep ``model.to``
        return False


# "0" keeps the non-resident encoder's host copy pageable.
PIN_HOST_ENV = "UNSLOTH_LTX25_TE_PIN"


def _pin_host_copy(model: Any, config: dict) -> int:
    """Copy a non-resident encoder's host weights into registered (pinned) chunks once, Linux CUDA only and only with
    room for the transient second copy, so uploads run at DMA speed; returns bytes pinned, 0 when skipped."""
    if config.get("resident") or (os.environ.get(PIN_HOST_ENV) or "").strip() == "0":
        return 0
    device = str(config.get("device") or "")
    if not device.startswith("cuda"):
        return 0
    try:
        from core.inference.diffusion_memory import (
            _exact_pin_supported,
            _named_buffers,
            _pin_host_weights,
        )

        if not _exact_pin_supported(device):
            return 0
        host = {name: p.data for name, p in model.named_parameters() if p.device.type == "cpu"}
        buffers = {name: b for name, b in _named_buffers(model) if b.device.type == "cpu"}
        if not _pin_fits_twice(list(host.values()) + list(buffers.values())):
            return 0
        return int(
            _pin_host_weights(model, host, None, buffer_host = buffers, exact = True, device = device)
            or 0
        )
    except Exception as exc:  # noqa: BLE001 -- an optimisation: the encode reads the pageable copy
        print(f"video.ltx25_te: host copy left pageable ({exc})", file = sys.stderr, flush = True)
        return 0


def _own_host_copy(model: Any) -> bool:
    """Copy the host weights out of the checkpoint's file mapping into process memory, once, before the first card
    encode. Base did this implicitly (``model.to("cpu")`` after each encode); kept mapped, the pages are page cache the
    server's pins evict, and each onload then re-reads 23 GB from disk (70-80 s per encode on an A100 host)."""
    import torch

    if not _host_fits(list(model.parameters()) + list(model.buffers()), copies = 1):
        # Base's streamed and CPU encodes kept the mapping too; a short host keeps it rather than risk the OOM killer.
        return False
    try:
        # Keyed by identity: a tied tensor is copied once and stays shared.
        done: dict = {}
        with torch.no_grad():
            for module in model.modules():
                for table in (module._parameters, module._buffers):
                    for key, tensor in list(table.items()):
                        if tensor is None or tensor.device.type != "cpu":
                            continue
                        if isinstance(tensor, torch.nn.Parameter):
                            if id(tensor) not in done:
                                done[id(tensor)] = tensor
                                tensor.data = tensor.data.clone()
                            continue
                        if id(tensor) not in done:
                            done[id(tensor)] = tensor.clone()
                        table[key] = done[id(tensor)]
        return True
    except Exception as exc:  # noqa: BLE001 -- the mapped copy still encodes, only slower when evicted
        print(f"video.ltx25_te: host copy left mapped ({exc})", file = sys.stderr, flush = True)
        return False


def _pin_fits_twice(tensors: list) -> bool:
    """Whether host RAM holds a second copy of ``tensors`` above the pin reserve. The loaded encoder's tensors share
    storages, so the pinned chunks fill before the pageable copy is freed (24.9 -> 42.7 GB on an A100 host); on a host
    without that room the pin pushed the box into reclaim and encodes took 70-80 s instead of 22 s."""
    return _host_fits(tensors, copies = 2)


def _host_fits(tensors: list, copies: int) -> bool:
    """Whether host RAM holds ``copies`` more copies of ``tensors`` above the pin reserve."""
    from core.inference.diffusion_memory import (
        _PIN_RESERVE_FRACTION,
        _PIN_RESERVE_MIN_BYTES,
        _available_system_memory_mib,
        _host_ram_capacity_mib,
    )

    available, total = _available_system_memory_mib(), _host_ram_capacity_mib()
    if available is None or total is None:
        return False
    need = sum(int(t.numel()) * int(t.element_size()) for t in tensors)
    reserve = max(_PIN_RESERVE_MIN_BYTES, int((int(total) << 20) * _PIN_RESERVE_FRACTION))
    return (int(available) << 20) - int(copies) * need >= reserve


class _Onload:
    """Moves the tensors of some modules to a device and back. Parameters keep their identity (``.data`` swaps, so a
    tied weight moves once); the host tensors are kept and restored, so returning costs no copy."""

    def __init__(
        self,
        device: Any,
        ready: Optional[dict] = None,
    ) -> None:
        self.device = device
        self.saved: list = []
        self.moved: dict = {}
        # id(tensor) -> (host data_ptr, device copy); a moved host storage falls back to a plain copy.
        self.ready = ready or {}

    def _copy(self, key: int, tensor: Any) -> Any:
        hit = self.ready.get(key)
        if hit is not None and hit[0] == tensor.data_ptr():
            return hit[1]
        return _to_device(tensor, self.device)

    def add(
        self,
        modules: list,
        recurse: bool = True,
    ) -> None:
        import torch
        for module in modules:
            for sub in module.modules() if recurse else (module,):
                for table in _tensor_tables(sub):
                    for key, tensor in list(table.items()):
                        if tensor is None:
                            continue
                        if isinstance(tensor, torch.nn.Parameter):
                            if id(tensor) not in self.moved:
                                self.moved[id(tensor)] = tensor
                                self.saved.append((tensor, None, None, tensor.data))
                                tensor.data = self._copy(id(tensor), tensor.data)
                            continue
                        moved = self.moved.get(id(tensor))
                        if moved is None:
                            moved = self._copy(id(tensor), tensor)
                            self.moved[id(tensor)] = moved
                        self.saved.append((None, table, key, tensor))
                        table[key] = moved

    def restore(self) -> None:
        while self.saved:
            param, table, key, original = self.saved.pop()
            if param is not None:
                param.data = original
            else:
                table[key] = original
        self.moved.clear()


# "0" uploads each streamed layer on the compute stream when it starts, as before the one-layer lookahead.
PREFETCH_ENV = "UNSLOTH_LTX25_TE_PREFETCH"


def _host_sources(module: Any) -> Optional[list]:
    """``(id, host data)`` per tensor ``_Onload`` moves, or None unless all are pinned (only pinned copies async)."""
    import torch

    seen: set = set()
    sources: list = []
    for sub in module.modules():
        for table in _tensor_tables(sub):
            for tensor in table.values():
                if tensor is None or id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                data = tensor.data if isinstance(tensor, torch.nn.Parameter) else tensor
                if data.device.type != "cpu" or not data.is_pinned():
                    return None
                sources.append((id(tensor), data))
    return sources or None


class _CudaUploads:
    def __init__(self, device: Any) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self._stream: Any = None

    @property
    def stream(self) -> Any:
        # Lazy: the stream pool's first use costs ~20 MiB of card memory.
        if self._stream is None:
            self._stream = self.torch.cuda.Stream(device = self.device)
        return self._stream

    def _compute(self) -> Any:
        return self.torch.cuda.current_stream(self.device)

    def fence(self, events: list) -> None:
        for event in events:
            self.stream.wait_event(event)

    def upload(self, sources: list) -> tuple[dict, Any]:
        torch = self.torch
        with torch.cuda.stream(self.stream):
            copies = {
                key: (data.data_ptr(), data.to(self.device, non_blocking = True))
                for key, data in sources
            }
            ready = torch.cuda.Event()
            ready.record(self.stream)
        return copies, ready

    def wait_ready(self, ready: Any) -> None:
        self._compute().wait_event(ready)

    def done(self) -> Any:
        event = self.torch.cuda.Event()
        event.record(self._compute())
        return event

    def drain(self) -> None:
        if self._stream is None:
            return
        self._stream.wait_stream(self._compute())
        self._stream.synchronize()


def _lookahead_ops(device: str) -> Any:
    if (os.environ.get(PREFETCH_ENV) or "").strip() == "0":
        return None
    if str(device).split(":", 1)[0] != "cuda":
        return None
    try:
        import torch

        from .rocm_bf16 import is_rocm_torch

        if is_rocm_torch(torch) or not torch.cuda.is_available():
            return None
        return _CudaUploads(device)
    except Exception:  # noqa: BLE001 -- the synchronous upload still works
        return None


class _Lookahead:
    """Layer i+1 uploads while layer i computes. Layer i's buffers are freed only after the copy stream waits on its
    compute, so the allocator cannot reuse them early. At most two layers resident (priced by ``_stream_need_mib``)."""

    def __init__(self, layers: list, ops: Any) -> None:
        self.ops = ops
        self.order = {id(layer): i for i, layer in enumerate(layers)}
        self.sources = [_host_sources(layer) for layer in layers]
        self.pending: dict = {}
        self.retired: list = []
        self.layers = 0

    def issue(self, index: int) -> None:
        if index >= len(self.sources) or self.pending or self.sources[index] is None:
            return
        self.ops.fence([done for done, _copies in self.retired])
        self.retired.clear()
        self.pending[index] = self.ops.upload(self.sources[index])
        self.layers += 1

    def take(self, module: Any) -> Optional[dict]:
        index = self.order.get(id(module))
        entry = self.pending.pop(index, None) if index is not None else None
        # An out-of-order layer's upload is dropped, never read.
        self.pending.clear()
        if entry is not None:
            self.ops.wait_ready(entry[1])
        if index is not None:
            self.issue(index + 1)
        return entry[0] if entry is not None else None

    def retire(self, copies: dict) -> None:
        self.retired.append((self.ops.done(), copies))

    def close(self) -> None:
        self.ops.drain()
        self.pending.clear()
        self.retired.clear()


def _streamed(
    parts: tuple[list, list, list],
    device: str,
    stats: Optional[dict] = None,
) -> Any:
    """Context: the resident modules on ``device`` for the encode, each decoder layer only while it runs."""
    import contextlib

    resident, containers, layers = parts

    @contextlib.contextmanager
    def _ctx():
        base = _Onload(device)
        handles: list = []
        live: dict = {}
        ahead: Optional[_Lookahead] = None
        try:
            base.add(resident)
            base.add(containers, recurse = False)
            ops = _lookahead_ops(device) if stats is None or not stats.get("no_prefetch") else None
            if ops is not None:
                ahead = _Lookahead(layers, ops)
                if all(s is None for s in ahead.sources):
                    ahead = None
                else:
                    ahead.issue(0)

            def before(module: Any, _args: Any) -> None:
                copies = ahead.take(module) if ahead is not None else None
                onload = _Onload(device, copies)
                live[id(module)] = onload
                onload.add([module])

            def after(module: Any, _args: Any, _output: Any) -> None:
                onload = live.pop(id(module), None)
                if onload is not None:
                    onload.restore()
                    if ahead is not None and onload.ready:
                        ahead.retire(onload.ready)

            for layer in layers:
                handles.append(layer.register_forward_pre_hook(before))
                handles.append(layer.register_forward_hook(after, always_call = True))
            yield
        finally:
            for handle in handles:
                handle.remove()
            for onload in live.values():
                onload.restore()
            try:
                if ahead is not None:
                    ahead.close()
                    if stats is not None:
                        stats["prefetched_layers"] = ahead.layers
            finally:
                base.restore()

    return _ctx()


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
        import psutil

        info = psutil.Process().memory_info()
        stats["host_rss_mib"] = info.rss >> 20
        if getattr(info, "peak_wset", None):  # Windows has no getrusage
            stats["host_peak_rss_mib"] = info.peak_wset >> 20
    except Exception:  # noqa: BLE001 -- stats are best-effort
        pass
    try:
        import resource

        # ru_maxrss is KiB on Linux but bytes on macOS.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        stats["host_peak_rss_mib"] = peak >> (20 if sys.platform == "darwin" else 10)
        if "host_rss_mib" not in stats:
            with open("/proc/self/statm", encoding = "ascii") as fh:
                stats["host_rss_mib"] = (
                    int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                ) >> 20
    except Exception:  # noqa: BLE001 -- Windows, or no /proc
        pass
    return stats


def _pack_hidden_states(model: Any, ids: Any, mask: Any) -> Any:
    """``LTX2Pipeline._get_gemma_prompt_embeds``: every hidden state stacked on a last axis, flattened. No KV cache:
    the encoder config sets ``use_cache``, which would build and hold a DynamicCache this single forward never reads."""
    import torch

    out = model(input_ids = ids, attention_mask = mask, output_hidden_states = True, use_cache = False)
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
    resident = bool(config.get("resident"))
    device, stream = _encode_plan(config, model, server_reserved_mib)
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
    config["_streamed"] = stream is not None
    config["_stream_stats"] = {}
    oom: Optional[str] = None
    try:
        embeds_host, mask_host = _run_encode(model, config, inputs, device, stream)
    except Exception as exc:
        if resident or device == "cpu" or not _is_out_of_memory(exc):
            raise
        oom = str(exc)
    if oom is not None and config["_stream_stats"].get("prefetched_layers"):
        # Retry without the lookahead's second layer before falling back to CPU.
        _empty_accelerator_cache(device)
        print(
            f"video.ltx25_te: out of memory with the layer lookahead on {device} ({oom}); retrying without it",
            file = sys.stderr,
            flush = True,
        )
        config["_stream_stats"] = {"no_prefetch": True}
        oom = None
        try:
            embeds_host, mask_host = _run_encode(model, config, inputs, device, stream)
        except Exception as exc:
            if not _is_out_of_memory(exc):
                raise
            oom = str(exc)
    if oom is not None:
        # Outside the handler, so the failed attempt's card tensors are gone before the CPU encode; then hand the
        # card back, since the server plans the denoise against it.
        _empty_accelerator_cache(device)
        print(
            f"video.ltx25_te: out of memory encoding on {device} ({oom}); encoding on the CPU",
            file = sys.stderr,
            flush = True,
        )
        device, config["_streamed"] = "cpu", False
        embeds_host, mask_host = _run_encode(model, config, inputs, device, None)
    return embeds_host, mask_host, device


def _empty_accelerator_cache(device: Any) -> None:
    import torch

    kind = str(device).split(":", 1)[0]
    accel = getattr(torch, kind, None) if kind in ("cuda", "xpu") else None
    if accel is not None and accel.is_available():
        accel.empty_cache()


def _is_out_of_memory(exc: BaseException) -> bool:
    import torch
    kinds = tuple(
        k
        for k in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
        )
        if isinstance(k, type)
    )
    return (bool(kinds) and isinstance(exc, kinds)) or "out of memory" in str(exc).lower()


def _run_encode(model: Any, config: dict, inputs: Any, device: str, stream: Any) -> tuple[Any, Any]:
    import torch

    resident = bool(config.get("resident"))
    onloaded = not resident and device != "cpu" and stream is None
    if not resident and device != "cpu" and _discrete_card(device):
        if config.get("_pin_host") and "_pinned_bytes" not in config:
            config["_pinned_bytes"] = _pin_host_copy(model, config)
        if not config.get("_pinned_bytes") and not config.get("_owned_host"):
            config["_owned_host"] = _own_host_copy(model)
    # Onload copies host tensors up and restores by repointing: no 23 GB device-to-host copy per prompt.
    whole = (
        _Onload(device) if onloaded and _discrete_card(device) and _onloads_whole(model) else None
    )
    if whole is None and onloaded:
        model.to(device)
    try:
        if whole is not None:
            whole.add([model])
        with (
            _streamed(stream, device, config.setdefault("_stream_stats", {}))
            if stream is not None
            else _no_context(),
            torch.inference_mode(),
        ):
            ids = inputs.input_ids.to(device)
            mask = inputs.attention_mask.to(device)
            embeds = _pack_hidden_states(model, ids, mask)
            embeds_host = embeds.to("cpu")
            mask_host = mask.to("cpu")
            del embeds
    finally:
        if whole is not None:
            whole.restore()
        elif onloaded:
            model.to("cpu")
        # Return the encode's activations (and, onloaded, the weights) to the card: a resident worker otherwise keeps
        # ~1.7 GB cached through the denoise, an onloaded one the whole encoder.
        _empty_accelerator_cache(device)
    return embeds_host, mask_host


def _no_context() -> Any:
    import contextlib
    return contextlib.nullcontext()


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
                # "hosted" (the unsloth file), "quantized_at_load" (built from the dense encoder) or None (bf16).
                "te_source": config.get("_te_source"),
                "weights_mib": _weights_mib_or_none(model),
                # video_ltx25.lend_resident_dit_for_encode lends resident denoiser blocks up to this.
                "stream_need_mib": _stream_need_mib_or_none(config, model),
                **_memory_stats(config.get("device") or ""),
            }
        )
    except Exception as exc:  # noqa: BLE001 -- every failure goes back to the server as a message
        resp_queue.put(
            {
                "type": "error",
                "error": f"LTX-2.5 text encoder failed to load: {exc}",
                "cache_miss": isinstance(exc, OSError) or isinstance(exc.__cause__, OSError),
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
            config["_pin_host"] = bool(msg.get("pin_host"))
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
                    "streamed": bool(config.get("_streamed")),
                    "prefetched_layers": int(
                        (config.get("_stream_stats") or {}).get("prefetched_layers") or 0
                    ),
                    "pinned_mib": int(config.get("_pinned_bytes") or 0) >> 20,
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
