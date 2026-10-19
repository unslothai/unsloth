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
`pixel_values` given). Installed only by `fsdp2_native.shard_model`, and only when some vision tower
or projector parameter issues a gradient collective (FSDP2-managed, or replicated with an all-reduce
hook). `UNSLOTH_FSDP2_VISION_DUMMY=0` turns it off (then every rank's micro-batch must carry an image).
"""

import inspect
import os

import torch

__all__ = ["install_vision_dummy"]

_INSTALLED = "_unsloth_fsdp2_vision_dummy"
_VISION_INPUTS = ("pixel_values", "pixel_values_videos")
_TOWER_NAMES = ("visual", "vision_tower", "vision_model", "vision_encoder", "vision")
_PROJECTOR_NAMES = (
    "multi_modal_projector",
    "mm_projector",
    "embed_vision",
    "connector",
    "vision_projection",
    "visual_projection",
)


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


def _fixed_resolution(get, tower):
    """`get_image_features` takes `image_sizes` only optionally (Llava on recent transformers forwards
    it to backbones that need it) and the tower's forward does not take it (CLIP / SigLIP): the encoder
    expects `vision_config.image_size`, so a patch-sized dummy raises there. A tower that takes
    `image_sizes` (Pixtral) or a `get_image_features` that requires it (LlavaNext) is variable
    resolution."""
    try:
        param = inspect.signature(get).parameters["image_sizes"]
    except (TypeError, ValueError, KeyError):
        return False
    if param.default is inspect.Parameter.empty:
        return False
    tower_signature = _forward_signature(tower)
    return tower_signature is not None and "image_sizes" not in tower_signature.parameters


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
    size = getattr(vision, "image_size", None)
    if isinstance(size, (tuple, list)):
        size = size[0]
    if _takes(get, "image_sizes") and not (size and _fixed_resolution(get, tower)):
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
    fields = getattr(obj, "__dataclass_fields__", None)
    if fields or isinstance(obj, dict):
        # ModelOutput: Gemma4 assigns pooler_output (the embed_vision projection) to a
        # BaseModelOutputWithPast after building it, so it is neither a field nor a dict item, only an
        # instance attribute. Read the dict items, the fields and the instance attributes (a tensor
        # counted twice only adds another 0 times it).
        values = list(obj.values()) if isinstance(obj, dict) else []
        values += [getattr(obj, name, None) for name in fields or ()]
        values += list(getattr(obj, "__dict__", {}).values())
    elif isinstance(obj, (list, tuple)):
        values = obj
    elif hasattr(obj, "to_tuple"):
        values = obj.to_tuple()
    else:
        return acc
    for value in values:
        acc = _float_sum(value, acc)
    return acc


def _vision_modules(mm, tower):
    """The tower plus the connectors `get_image_features` runs after it (Llava / Gemma3 / Mllama
    `multi_modal_projector`, Gemma4 `embed_vision`, Idefics3 `connector`, ...); Qwen-VL's merger is
    inside the tower."""
    modules = [tower]
    for name in _PROJECTOR_NAMES:
        module = getattr(mm, name, None)
        if isinstance(module, torch.nn.Module) and all(module is not m for m in modules):
            modules.append(module)
    return modules


def _managed_by_fsdp(*modules):
    """Some parameter of `modules` issues a gradient collective: a DTensor (FSDP2 reduce-scatters it,
    but only where its gradient is not None), or a replicated trainable one whose gradient is averaged
    by the all-reduce hook `fsdp2_save.reduce_unsharded_trainable_grads` installed."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        try:
            from torch.distributed._tensor import DTensor
        except ImportError:
            DTensor = None
    for module in modules:
        for p in module.parameters():
            if DTensor is not None and isinstance(p, DTensor):
                return True
            if getattr(p, "_unsloth_grad_all_reduce", False):
                return True
    return False


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
    if not _managed_by_fsdp(*_vision_modules(mm, tower)):
        return None  # replicated / ignored tower and projector: no collectives to desync
    if _dummy_inputs(mm, tower) is None:
        _print_rank0(
            f"Unsloth: native FSDP2 cannot build a dummy image input for {type(mm).__name__}; a micro-batch "
            "without images on one rank while another has them will hang. Give every rank an image."
        )
        return None
    signature = _forward_signature(mm)
    state = {"pending": None, "armed": False}
    # Run the dummy where an image rank runs its tower: just before the language model (after the input
    # embeddings and Gemma4's per-layer embedding unit), so per-block layouts gather units in the same
    # order on every rank. Without a `language_model` child, before the multimodal forward.
    language_model = getattr(mm, "language_model", None)
    if not isinstance(language_model, torch.nn.Module) or language_model is tower:
        language_model = None

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

    def _dummy_total():
        return _float_sum(mm.get_image_features(**_dummy_inputs(mm, tower)))

    def lm_pre_hook(module, args, kwargs):
        if not state["armed"]:
            return None
        state["armed"] = False
        total = _dummy_total()
        if total is None:
            return None
        embeds = kwargs.get("inputs_embeds")
        if isinstance(embeds, torch.Tensor):
            # Into the decoder input, like an image rank's masked_scatter (see embed_hook).
            kwargs["inputs_embeds"] = embeds + _zero_like(total, embeds)
            return args, kwargs
        state["pending"] = total  # added to the multimodal output by post_hook
        return None

    def pre_hook(module, args, kwargs):
        state["pending"] = None
        state["armed"] = False
        if _in_generate() or _has_vision_input(args, kwargs):
            return None
        if language_model is not None:
            state["armed"] = True
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
        state["armed"] = False
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
    if language_model is not None:
        language_model.register_forward_pre_hook(lm_pre_hook, with_kwargs = True)
    elif isinstance(embeddings, torch.nn.Module):
        embeddings.register_forward_hook(embed_hook)
    mm.register_forward_hook(post_hook, with_kwargs = True)
    setattr(mm, _INSTALLED, True)
    _print_rank0(
        f"Unsloth: native FSDP2 runs {type(mm).__name__}'s vision tower on a dummy image for micro-batches "
        "without images, so every rank issues the same vision collectives (output added times 0)."
    )
    return mm
