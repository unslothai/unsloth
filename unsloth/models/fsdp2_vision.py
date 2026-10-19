# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Mixed image / text micro-batches under native FSDP2.

When one rank's micro-batch has images and another's does not, only the image rank runs the vision
tower: its units all-gather on that rank alone and NCCL pairs the collectives wrongly (hang), and
FSDP2 reduce-scatters only parameters whose gradient is not None, so even a vision tower held in the
root unit (fit layout) desyncs at the reduce-scatter. Here a rank without images runs the vision
encoder on a minimal dummy input before the language model (same collective order as an image rank)
and adds `0.0 * sum(outputs)` to the multimodal model's output, so every vision parameter gets a zero
gradient on that rank. No collective and no host sync is added (the decision is local: is
`pixel_values` given). Installed only by `fsdp2_native.shard_model`, and only when some vision
parameter is managed by FSDP2. `UNSLOTH_FSDP2_VISION_DUMMY=0` turns it off (then every rank's
micro-batch must carry an image).
"""

import inspect
import os

import torch

__all__ = ["install_vision_dummy"]

_INSTALLED = "_unsloth_fsdp2_vision_dummy"
_VISION_INPUTS = ("pixel_values", "pixel_values_videos")
_TOWER_NAMES = ("visual", "vision_tower", "vision_model", "vision_encoder", "vision")


def _rank0():
    try:
        import torch.distributed as dist
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


def _print_rank0(message):
    if _rank0():
        print(message, flush = True)


def _takes(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _forward_signature(module):
    """The forward's signature; a `(*args, **kwargs)` wrapper (zoo temporary patches wrap Gemma4Model's)
    is looked through to the function it wraps when that one names `pixel_values`."""
    forward = type(module).forward
    try:
        unwrapped = inspect.unwrap(forward)
        if unwrapped is not forward and _takes(unwrapped, "pixel_values"):
            return inspect.signature(unwrapped)
        return inspect.signature(forward)
    except (TypeError, ValueError):
        return None


def _takes_any_kwargs(signature):
    return signature is not None and any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )


def _multimodal_module(model):
    """The innermost module with `get_image_features` and a vision tower whose forward takes
    `pixel_values` (or `**kwargs`, a wrapped forward)."""
    found = None
    for module in model.modules():
        if not callable(getattr(module, "get_image_features", None)):
            continue
        signature = _forward_signature(module)
        takes = signature is not None and "pixel_values" in signature.parameters
        if takes or (_takes_any_kwargs(signature) and _vision_tower(module) is not None):
            found = module  # pre-order: a later match is nested deeper (CG -> Model)
    return found


def _vision_tower(mm):
    for name in _TOWER_NAMES:
        tower = getattr(mm, name, None)
        if isinstance(tower, torch.nn.Module):
            return tower
    return None


def _is_params4bit(p):
    return any(c.__name__ == "Params4bit" for c in type(p).__mro__)


def _tower_dtype(tower, mm):
    for p in tower.parameters():
        if p.is_floating_point() and not _is_params4bit(p):
            try:
                return p.to_local().dtype if hasattr(p, "to_local") else p.dtype
            except Exception:
                return p.dtype
    return getattr(mm, "dtype", torch.float32)


def _tower_device(tower):
    for p in tower.parameters():
        try:
            local = p.to_local() if hasattr(p, "to_local") else p
        except Exception:
            local = p
        if local.device.type != "meta":
            return local.device
    return torch.device("cpu")


def _dummy_inputs(mm, tower):
    """Smallest valid image input for this family, or None when it cannot be built."""
    config = getattr(mm, "config", None)
    vision = getattr(config, "vision_config", None) or getattr(tower, "config", None)
    if vision is None:
        return None
    dtype = _tower_dtype(tower, mm)
    device = _tower_device(tower)
    get = mm.get_image_features
    patch = getattr(vision, "patch_size", None) or getattr(vision, "spatial_patch_size", None)
    channels = getattr(vision, "in_channels", None) or getattr(vision, "num_channels", None) or 3
    if _takes(get, "image_grid_thw"):
        m = getattr(vision, "spatial_merge_size", None) or 1
        temporal = getattr(vision, "temporal_patch_size", None) or 1
        if not patch:
            return None
        pixel = torch.zeros(m * m, channels * temporal * patch * patch, dtype = dtype, device = device)
        grid = torch.tensor([[1, m, m]], dtype = torch.long, device = device)
        return {"pixel_values": pixel, "image_grid_thw": grid}
    if _takes(get, "image_position_ids"):
        # Gemma4: flattened patches + (x, y) patch positions; one pooling window = one soft token.
        k = getattr(vision, "pooling_kernel_size", None) or 1
        if not patch:
            return None
        n = k * k
        pixel = torch.zeros(1, n, channels * patch * patch, dtype = dtype, device = device)
        ys, xs = torch.meshgrid(torch.arange(k), torch.arange(k), indexing = "ij")
        pos = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim = -1)[None].to(device)
        return {"pixel_values": pixel, "image_position_ids": pos}
    if _takes(get, "image_sizes"):
        merge = (
            getattr(config, "spatial_merge_size", None)
            or getattr(vision, "spatial_merge_size", None)
            or 1
        )
        if not patch:
            return None
        s = patch * merge
        pixel = torch.zeros(1, channels, s, s, dtype = dtype, device = device)
        return {"pixel_values": pixel, "image_sizes": torch.tensor([[s, s]], device = device)}
    size = getattr(vision, "image_size", None)
    if isinstance(size, (tuple, list)):
        size = size[0]
    if not size:
        return None
    return {"pixel_values": torch.zeros(1, channels, size, size, dtype = dtype, device = device)}


def _float_sum(obj, acc = None):
    """Sum of every floating tensor in a (nested) output, as one scalar kept in the autograd graph."""
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and obj.numel():
            s = obj.float().sum()
            return s if acc is None else acc + s
        return acc
    if isinstance(obj, dict):
        values = obj.values()
    elif isinstance(obj, (list, tuple)):
        values = obj
    elif hasattr(obj, "to_tuple"):
        values = obj.to_tuple()
    else:
        return acc
    for value in values:
        acc = _float_sum(value, acc)
    return acc


def _managed_by_fsdp(tower):
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        try:
            from torch.distributed._tensor import DTensor
        except ImportError:
            return False
    return any(isinstance(p, DTensor) for p in tower.parameters())


def _in_generate():
    try:
        from .fsdp2_native import _GENERATE_DEPTH
        return _GENERATE_DEPTH[0] > 0
    except Exception:
        return False


def install_vision_dummy(model):
    """Register the dummy vision pass on the multimodal module of a natively sharded `model`.
    Returns the module the hooks are on, or None (text model, vision not managed by FSDP2, or a family
    whose dummy input cannot be built: a warning is printed then)."""
    if os.environ.get("UNSLOTH_FSDP2_VISION_DUMMY", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return None
    mm = _multimodal_module(model)
    if mm is None or getattr(mm, _INSTALLED, False):
        return None
    tower = _vision_tower(mm)
    if tower is None:
        _print_rank0(
            f"Unsloth: native FSDP2 found no vision tower on {type(mm).__name__}; ranks whose micro-batch "
            "has no image must still see the same vision collectives (give every rank an image)."
        )
        return None
    if not _managed_by_fsdp(tower):
        return None  # replicated / ignored tower: no collectives to desync
    if _dummy_inputs(mm, tower) is None:
        _print_rank0(
            f"Unsloth: native FSDP2 cannot build a dummy image input for {type(mm).__name__}; a micro-batch "
            "without images on one rank while another has them will hang. Give every rank an image."
        )
        return None
    signature = _forward_signature(mm)
    state = {"pending": None}

    def _has_vision_input(args, kwargs):
        if any(kwargs.get(name) is not None for name in _VISION_INPUTS):
            return True
        if not args or signature is None:
            return False
        try:
            bound = signature.bind_partial(mm, *args, **kwargs).arguments
        except TypeError:
            return False
        return any(bound.get(name) is not None for name in _VISION_INPUTS)

    def _zero_like(total, ref):
        return (0.0 * total).to(ref.dtype)

    def pre_hook(module, args, kwargs):
        state["pending"] = None
        if _in_generate() or _has_vision_input(args, kwargs):
            return None
        dummy = _dummy_inputs(mm, tower)
        total = _float_sum(mm.get_image_features(**dummy))
        if total is None:
            return None
        embeds = kwargs.get("inputs_embeds")
        if isinstance(embeds, torch.Tensor):
            kwargs["inputs_embeds"] = embeds + _zero_like(total, embeds)
            return args, kwargs
        state["pending"] = total
        return None

    def embed_hook(module, args, output):
        # Into the language model's input, like an image rank's masked_scatter: the decoder units' inputs
        # then require grad on every rank, so FSDP2 runs their post-backward (reduce-scatter) at the same
        # point of the backward everywhere instead of deferring it to the final callback on this rank.
        total = state["pending"]
        if total is None or not isinstance(output, torch.Tensor):
            return None
        state["pending"] = None
        return output + _zero_like(total, output)

    def post_hook(module, args, kwargs, output):
        total = state["pending"]
        state["pending"] = None
        if total is None:
            return None
        zero = 0.0 * total  # inputs_embeds was not built from the input embeddings: add here
        hidden = getattr(output, "last_hidden_state", None)
        if isinstance(hidden, torch.Tensor):
            output.last_hidden_state = hidden + zero.to(hidden.dtype)
            return output
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (output[0] + zero.to(output[0].dtype),) + tuple(output[1:])
        if isinstance(output, torch.Tensor):
            return output + zero.to(output.dtype)
        return None

    embeddings = mm.get_input_embeddings() if hasattr(mm, "get_input_embeddings") else None
    mm.register_forward_pre_hook(pre_hook, with_kwargs = True)
    if isinstance(embeddings, torch.nn.Module):
        embeddings.register_forward_hook(embed_hook)
    mm.register_forward_hook(post_hook, with_kwargs = True)
    setattr(mm, _INSTALLED, True)
    _print_rank0(
        f"Unsloth: native FSDP2 runs {type(mm).__name__}'s vision tower on a dummy image for micro-batches "
        "without images, so every rank issues the same vision collectives (output added times 0)."
    )
    return mm
