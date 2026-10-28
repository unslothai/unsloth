# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Image-less ranks run a dummy vision pass (else NCCL hangs); UNSLOTH_FSDP2_VISION_DUMMY=0
disables, then every rank micro-batch must carry an image."""

import inspect
import os
import re

import torch

__all__ = ["install_vision_dummy"]

_INSTALLED = "_unsloth_fsdp2_vision_dummy"
_VISION_INPUTS = (
    "pixel_values",
    "pixel_values_videos",
    "pixel_values_images",
    "image_pixel_values",
)
# Audio encoders (Gemma3n / Gemma4 `input_features`, Qwen2.5-Omni / Qwen3-Omni `input_features` with
# `feature_attention_mask`, PE Audio / PeAudioVideo `input_values`, others `audio_values`): no dummy
# is built, a mixed batch raises. The media keys unsloth/trainer.py and utils/token_budget.py batch.
_AUDIO_INPUTS = (
    "input_features",
    "input_values",
    "audio_values",
    "input_audio_embeds",
    "audio_input_features",
)
# Named per speaker / stream: Moshi's `user_input_values` / `moshi_input_values`
# (transformers/models/moshi/modeling_moshi.py MoshiForConditionalGeneration.forward), each encoded by
# its own audio_encoder.encode call.
_AUDIO_INPUT_RE = re.compile(r"^\w+_(input_values|audio_values)$")


def _audio_inputs(signature):
    """The audio inputs a forward takes: the known names, then any `*_input_values` /
    `*_audio_values` parameter. All known names for a signature that cannot be read."""
    if signature is None:
        return ()
    names = [n for n in _AUDIO_INPUTS if n in signature.parameters]
    names += [n for n in signature.parameters if _AUDIO_INPUT_RE.match(n) and n not in names]
    return tuple(names)


_AUDIO_TOWER_NAMES = ("audio_tower", "audio_model", "audio_encoder", "audio")
_AUDIO_PROJECTOR_NAMES = (
    "embed_audio",
    "audio_projector",
    "audio_projection",
    "audio_multi_modal_projector",
)
# `visual` (Qwen-VL) also runs the video path; VideoLlava has separate `image_tower` / `video_tower`.
_TOWER_NAMES = ("visual", "vision_tower", "vision_model", "vision_encoder", "vision", "image_tower")
_VIDEO_TOWER_NAMES = ("video_tower",)
# Generic encoder attribute names (Phi-4 multimodal `image_embed` / `audio_embed`, `*_encoder`,
# `*_model`), matched on registered children after the known names above.
_SUFFIXES = r"(_(embed|embedding|embeddings|encoder|tower|model|processor))?"
_VISION_TOWER_RE = re.compile(r"^(image|vision|visual|img)" + _SUFFIXES + "$")
_AUDIO_TOWER_RE = re.compile(r"^(audio|speech)" + _SUFFIXES + "$")
_VIDEO_TOWER_RE = re.compile(r"^video_(embed|embedding|encoder|tower|model)$")
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
        takes = signature is not None and any(n in signature.parameters for n in _VISION_INPUTS)
        if takes or (_takes_any_kwargs(signature) and _vision_tower(module) is not None):
            found = module  # pre-order: a later match is nested deeper (CG -> Model)
    return found


def _inline_multimodal_module(model):
    """The outermost module whose forward takes a vision input (pixel_values, image_pixel_values, ...)
    with no `get_image_features`: its encoder runs inline and conditionally (Mllama's vision_model +
    multi_modal_projector, then cross-attention layers skipped without images; Phi-4 multimodal's
    `image_embed` inside its feature embedding). Outermost, because an inner module taking the input may
    run only on ranks that have it, and the agreement must be on a forward every rank runs."""
    for module in model.modules():
        if callable(getattr(module, "get_image_features", None)):
            continue
        signature = _forward_signature(module)
        if signature is None or not any(n in signature.parameters for n in _VISION_INPUTS):
            continue
        return module
    return None


def _video_modality(host, image_tower):
    """A video tower separate from the image tower (VideoLlava `video_tower`) is its own modality
    (`pixel_values_videos`), with no dummy. Returns (image inputs, [video modality]) or the shared
    form (all vision inputs, []) when video runs through the image tower (Qwen-VL `visual`)."""
    children = host.__dict__.get("_modules", {})
    names = list(_VIDEO_TOWER_NAMES) + [n for n in children if _VIDEO_TOWER_RE.match(n)]
    for name in names:
        video = children.get(name, None)
        if isinstance(video, torch.nn.Module) and video is not image_tower:
            inputs = tuple(n for n in _VISION_INPUTS if n != "pixel_values_videos")
            if not _managed_by_fsdp(video):
                return inputs, []
            why = "runs a separate video tower Unsloth has no dummy for"
            return inputs, [_Modality("videos", ("pixel_values_videos",), why)]
    return _VISION_INPUTS, []


def _image_only_modules(mm, tower = None):
    """Modules only run with an image (Mllama's cross-attention layers), outside the vision tower."""
    skip = {id(m) for m in tower.modules()} if isinstance(tower, torch.nn.Module) else set()
    return [m for m in mm.modules() if "CrossAttention" in type(m).__name__ and id(m) not in skip]


def _is_media_value(value):
    """A tensor, or a non-empty list / tuple holding one (batched per-sample images)."""
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (list, tuple)):
        return any(_is_media_value(v) for v in value)
    return False


def _given(
    signature,
    module,
    args,
    kwargs,
    names,
    tensors_only = False,
):
    """Whether this forward call passes a non-None value (a tensor value with `tensors_only`) for any
    of `names`."""
    check = _is_media_value if tensors_only else (lambda v: v is not None)
    if any(check(kwargs.get(name)) for name in names):
        return True
    if not args or signature is None:
        return False
    try:
        bound = signature.bind_partial(module, *args, **kwargs).arguments
    except TypeError:
        return False
    return any(check(bound.get(name)) for name in names)


def _agree_counts(
    flags,
    device,
    mesh = None,
):
    """Elementwise sum of `flags` over the ranks of `mesh` (the model's; the world when None), and
    their number: one small all-reduce per mesh dimension."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return [int(f) for f in flags], 1
    if "nccl" in str(dist.get_backend()).lower() and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    vector = torch.tensor([float(f) for f in flags], device = device)
    from .fsdp2_save import mesh_all_reduce

    world = mesh_all_reduce(vector, mesh, dist)
    return [int(round(v)) for v in vector.tolist()], world


def _missing_required(fn, inputs):
    """Parameters of `fn` with no default (not *args / **kwargs) that `inputs` does not supply. A
    signature that cannot be read counts as unknown: ["?"]."""
    try:
        signature = inspect.signature(inspect.unwrap(fn))
    except (TypeError, ValueError):
        return ["?"]
    missing = []
    for name, param in signature.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is param.empty and name not in inputs:
            missing.append(name)
    return missing


# A forward that takes a media input but no recognised encoder under it: the conservative fallback.
_NO_ENCODER = (
    "takes this input but Unsloth found no encoder it could run on a dummy (whatever runs only for it "
    "would leave the ranks without it out of its collectives)"
)


class _Modality:
    """One encoder input kind on a host module: which forward arguments carry it, and how a rank
    without it can still run the encoder's collectives (`ready()` -> bool), or None (it cannot)."""

    def __init__(
        self,
        label,
        inputs,
        why,
        ready = None,
    ):
        self.label = label
        self.inputs = inputs
        self.why = why
        self.ready = ready
        self.run_dummy = False
        self.tensors_only = False

    def groups(self, signature):
        """The inputs, one group per encoder call: each input the host takes is encoded on its own
        (Qwen-VL runs `visual` once for pixel_values and once more for pixel_values_videos; Moshi
        encodes user_input_values and moshi_input_values separately), so presence is agreed per input.
        The image inputs a dummy stands in for (`ready`) are one group, the first."""
        taken = [
            n
            for n in self.inputs
            if signature is None or n in signature.parameters or _takes_any_kwargs(signature)
        ] or list(self.inputs)
        if self.ready is None:
            return [(n,) for n in taken]
        covered = tuple(n for n in taken if n != "pixel_values_videos")
        return ([covered] if covered else []) + [(n,) for n in taken if n not in covered]


def _install_presence_agreement(
    host,
    modalities,
    device_of,
    mesh = None,
):
    """Every forward of `host`, before any encoder runs: one all-reduce over [present per input,
    cannot-run-a-dummy per modality]. Each input is encoded by its own call, so an input present on
    some ranks only makes the ranks without it run the modality's dummy (`run_dummy`, the image inputs
    a dummy stands in for), or, when there is no dummy for that input or it cannot be built on some
    rank lacking it, raises on every rank alike (the ranks with it would otherwise wait in
    collectives the others never enter). A dummy that cannot be built is found before the
    collective, so that failure is agreed too."""
    signature = _forward_signature(host)
    plan = [(m, m.groups(signature)) for m in modalities]
    slots = [
        (m, g, j == 0 and m.ready is not None and "pixel_values_videos" not in g)
        for m, groups in plan
        for j, g in enumerate(groups)
    ]
    pairs = [
        (l, c)
        for l, (ml, _, _) in enumerate(slots)
        if ml.tensors_only
        for c, (_, _, covered) in enumerate(slots)
        if covered
    ]

    def agree(module, args, kwargs):
        for m in modalities:
            m.run_dummy = False
        if _in_generate():
            return None
        present = [_given(signature, host, args, kwargs, g, m.tensors_only) for m, g, _ in slots]
        unready = []
        for m, groups in plan:
            ok = True
            if m.ready is not None and groups:
                ok = _given(signature, host, args, kwargs, groups[0])
                if not ok:
                    try:
                        ok = bool(m.ready())
                    except Exception:
                        ok = False
            unready.append(not ok)
        # A generic input arriving with a dummy-covered group on exactly the same ranks is that
        # group's companion (image_grid_thw with pixel_values): the dummy stands in for both.
        xor = [present[l] != present[c] for l, c in pairs]
        counts, world = _agree_counts(present + unready + xor, device_of(), mesh)
        n = len(slots)
        mismatched = []
        for i, (m, group, covered) in enumerate(slots):
            if not 0 < counts[i] < world:
                continue  # every rank has it, or none does: the collectives already match
            if covered and not counts[n + modalities.index(m)]:
                m.run_dummy = not present[i]
                continue
            companion = any(
                l == i
                and 0 < counts[c] < world
                and not counts[n + modalities.index(slots[c][0])]
                and counts[n + len(plan) + k] == 0
                for k, (l, c) in enumerate(pairs)
            )
            if not companion:
                mismatched.append((i, m, group, covered))
        if mismatched:
            i, m, group, covered = mismatched[0]
            why = (
                m.why
                if covered or m.ready is None
                else (f"encodes `{group[0]}` in a call of its own, which Unsloth has no dummy for")
            )
            single = len(plan[modalities.index(m)][1]) == 1 and not m.tensors_only
            which = "" if single else f" (input `{group[0]}`)"
            names = sorted({g[0] for _, _, g, _ in mismatched})
            raise RuntimeError(
                f"Unsloth: native FSDP2 got a micro-batch with {m.label} on {counts[i]} of {world} "
                f"ranks{which}. {type(host).__name__} {why or 'cannot build a dummy input here'}, "
                f"so the ranks without it would skip collectives the others run and the job would "
                f"hang. Mismatched inputs: {', '.join(names)}. Give every rank's micro-batch "
                f"{m.label} (or none), for example by grouping samples by modality into batches."
            )
        return None

    host.register_forward_pre_hook(agree, with_kwargs = True)
    setattr(host, _INSTALLED, True)
    for m in modalities:
        if m.ready is None:
            _print_rank0(
                f"Unsloth: native FSDP2 cannot run {type(host).__name__}'s {m.label} path on a dummy "
                f"input ({m.why}); every rank's micro-batch must carry {m.label} or none must (checked "
                "each forward, mixed raises on every rank)."
            )
    return host


def _audio_tower(module):
    return _named_tower(module, _AUDIO_TOWER_NAMES, _AUDIO_TOWER_RE)


def _audio_host(model, vision_host = None):
    """The module whose forward receives the audio input on every rank: the outermost one taking an
    audio input with an audio tower in its subtree. An inner module taking it too may run only on some
    ranks (PeAudioVideoModel calls `audio_model` only when no video is given), and an agreement hook
    there would leave the other ranks out of its collective. The vision host is reused when it takes an
    audio input and holds the tower (Gemma3n / Gemma4), so one collective covers both."""
    candidates = [vision_host] if vision_host is not None else []
    candidates += list(model.modules())
    for module in candidates:
        signature = _forward_signature(module)
        if not _audio_inputs(signature):
            continue
        if any(_audio_tower(m) is not None for m in module.modules()):
            return module
    return None


def _audio_modules(host):
    """Every audio tower and audio projector in the host's subtree (they may sit below the host)."""
    modules = []
    for owner in host.modules():
        children = owner.__dict__.get("_modules", {})
        names = list(_AUDIO_TOWER_NAMES + _AUDIO_PROJECTOR_NAMES)
        names += [n for n in children if _AUDIO_TOWER_RE.match(n) and n not in names]
        for name in names:
            module = children.get(name, None)
            if isinstance(module, torch.nn.Module) and all(module is not m for m in modules):
                modules.append(module)
    return modules


def _named_tower(module, names, pattern):
    """A known encoder attribute (properties included), else a registered child whose name matches."""
    for name in names:
        tower = getattr(module, name, None)
        if isinstance(tower, torch.nn.Module):
            return tower
    for name, child in (getattr(module, "_modules", None) or {}).items():
        if isinstance(child, torch.nn.Module) and pattern.match(name):
            return child
    return None


def _vision_tower(mm):
    return _named_tower(mm, _TOWER_NAMES, _VISION_TOWER_RE)


def _subtree_tower(host, find):
    """The first encoder `find` recognises anywhere under `host` (pre-order)."""
    for module in host.modules():
        tower = find(module)
        if tower is not None:
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


def _vision_plan(model):
    """(host, image modality or None, dummy tower or None, [separate video modality]) for the vision
    encoders, or None when none issues collectives."""
    mm = _multimodal_module(model)
    if mm is None:
        inline = _inline_multimodal_module(model)
        if inline is None:
            return None
        tower = _subtree_tower(inline, _vision_tower)
        if tower is None:
            return inline, _Modality("images", _VISION_INPUTS, _NO_ENCODER), None, []
        managed = _managed_by_fsdp(*_vision_modules(inline, tower))
        if not (managed or _managed_by_fsdp(*_image_only_modules(inline, tower))):
            return None  # nothing image-only issues a collective
        why = (
            "runs its vision tower inline (no get_image_features)"
            if managed
            else "skips its cross-attention layers without images"
        )
        inputs, extra = _video_modality(inline, tower)
        return inline, _Modality("images", inputs, why), None, extra
    tower = _vision_tower(mm)
    if tower is None:
        return mm, _Modality("images", _VISION_INPUTS, _NO_ENCODER), None, []
    inputs, extra = _video_modality(mm, tower)
    if not _managed_by_fsdp(*_vision_modules(mm, tower)):
        if extra:  # only the separate video tower issues collectives
            return mm, None, None, extra
        return None  # replicated / ignored tower and projector: no collectives to desync
    dummy = _dummy_inputs(mm, tower)
    missing = _missing_required(mm.get_image_features, dummy or {})
    if dummy is None or missing:
        why = (
            "has no dummy image input Unsloth can build"
            if dummy is None
            else f"needs {', '.join(missing)} for get_image_features, which no dummy supplies"
        )
        return mm, _Modality("images", inputs, why), None, extra

    def ready():
        inputs = _dummy_inputs(mm, tower)
        return inputs is not None and not _missing_required(mm.get_image_features, inputs)

    return mm, _Modality("images", inputs, "", ready), tower, extra


def _audio_plan(model, vision_host = None):
    host = _audio_host(model, vision_host)
    if host is None:
        # Takes an audio input but no encoder is recognised: agree anyway, raise on mixed batches.
        for module in (
            [vision_host] + list(model.modules()) if vision_host is not None else model.modules()
        ):
            names = _audio_inputs(_forward_signature(module))
            if names:
                return module, _Modality("audio", names, _NO_ENCODER)
        return None
    if not _managed_by_fsdp(*_audio_modules(host)):
        return None
    names = _audio_inputs(_forward_signature(host))
    return host, _Modality("audio", names, "runs an audio tower Unsloth has no dummy for")


# Forward arguments that are text (or control) inputs, never a media encoder's: everything else a
# multimodal model's forward takes is a candidate media input of the generic agreement.
_TEXT_KWARGS = frozenset(
    (
        "input_ids",
        "attention_mask",
        "labels",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "token_type_ids",
        "cache_position",
        "use_cache",
        "logits_to_keep",
        "num_logits_to_keep",
        "num_items_in_batch",
        "output_attentions",
        "output_hidden_states",
        "output_router_logits",
        "return_dict",
        "return_loss",
        "head_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "decoder_inputs_embeds",
        "decoder_head_mask",
        "cross_attn_head_mask",
        "encoder_outputs",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "task_ids",
        "kwargs",
        "args",
    )
)
# Media inputs by name (the ones above in this module, Fuyu's image_patches, ...) or by shape of the
# name, for models whose config carries no media sub-config.
_MEDIA_KWARG_RE = re.compile(
    r"(pixel|image|img|video|audio|speech|patch|wave|mel|spectrogram|_features$|_values$)"
)
_MEDIA_CONFIG_RE = re.compile(r"(vision|visual|image|video|audio|speech)\w*_config$")
_BACKBONE_NAMES = ("language_model", "text_model", "text_decoder")


def _forward_chain(model):
    """The outermost module and the wrapped models its forward hands kwargs to (PeftModel ->
    LoraModel -> the transformers model), outermost first."""
    chain = [model]
    for _ in range(6):
        current = chain[-1]
        inner = None
        children = current.__dict__.get("_modules", {})
        for attr in ("base_model", "model", "module", "_orig_mod"):
            child = children.get(attr)  # registered children only (PeftModel forwards attributes)
            if isinstance(child, torch.nn.Module) and all(child is not c for c in chain):
                if attr == "model" and getattr(child, "config", None) is None:
                    continue  # a decoder stack, not a wrapped model
                inner = child
                break
        if inner is None:
            break
        chain.append(inner)
    return chain


def _media_candidates(chain):
    """Sorted forward argument names, over the wrapper chain, that may carry a media input."""
    names = set()
    for module in chain:
        signature = _forward_signature(module)
        if signature is None:
            continue
        for name, param in signature.parameters.items():
            if name == "self" or name in _TEXT_KWARGS:
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            names.add(name)
    return sorted(names)


def _is_multimodal(chain, candidates):
    """A media sub-config (vision_config, audio_config, audio_encoder_config, ...) on the model's
    config, or a media-looking input in its forward."""
    for module in chain:
        config = getattr(module, "config", None)
        if config is None:
            continue
        for name in list(getattr(config, "__dict__", {})):
            if _MEDIA_CONFIG_RE.search(name) and getattr(config, name, None) is not None:
                return True
    known = set(_VISION_INPUTS) | set(_AUDIO_INPUTS)
    return any(n in known or _MEDIA_KWARG_RE.search(n) for n in candidates)


def _outside_backbone(model, chain):
    """Parameters outside the language model that are trainable or FSDP-managed (a DTensor): the
    ones a media input alone reaches. The text backbone (`language_model` / `text_model`, else the
    decoder) and the input / output embeddings do not count."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        try:
            from torch.distributed._tensor import DTensor
        except ImportError:
            DTensor = None
    inside = set()
    backbones = [m for n, m in model.named_modules() if n.rsplit(".", 1)[-1] in _BACKBONE_NAMES]
    if not backbones:
        for module in chain:
            get = getattr(module, "get_decoder", None)
            try:
                decoder = get() if callable(get) else None
            except Exception:
                decoder = None
            if isinstance(decoder, torch.nn.Module) and decoder is not module:
                backbones.append(decoder)
                break
    for module in chain:
        for getter in ("get_input_embeddings", "get_output_embeddings"):
            try:
                embedding = getattr(module, getter)()
            except Exception:
                embedding = None
            if isinstance(embedding, torch.nn.Module):
                backbones.append(embedding)
    for module in backbones:
        inside.update(id(p) for p in module.parameters())
    return [
        n
        for n, p in model.named_parameters()
        if id(p) not in inside
        and (p.requires_grad or (DTensor is not None and isinstance(p, DTensor)))
    ]


def _generic_media_plan(model, named):
    """The outermost forward agrees on every media input whenever a multimodal model has trainable
    or FSDP-managed parameters outside its language model: one bit per candidate input (a fixed,
    sorted list of forward argument names read now). Inputs a named modality covers keep its dummy;
    the others have none, so a mismatch raises on every rank. Returns (host, leftover modality or
    None), or None (text-only model, or nothing outside the backbone)."""
    chain = _forward_chain(model)
    candidates = _media_candidates(chain)
    if not candidates or not _is_multimodal(chain, candidates):
        return None
    outside = _outside_backbone(model, chain)
    if not outside:
        return None
    covered = {n for m in named for n in m.inputs}
    leftover = tuple(n for n in candidates if n not in covered)
    modality = None
    if leftover:
        modality = _Modality(
            "media",
            leftover,
            f"runs parameters outside its language model for this input (e.g. {outside[0]}) and "
            "Unsloth has no dummy for it",
        )
        modality.tensors_only = True
    return model, modality


def install_vision_dummy(model):
    """Per-modality presence agreement on a natively sharded `model` (vision and audio encoders that
    issue gradient / gather collectives). Every forward agrees over ranks, with one small all-reduce,
    which encoder inputs each rank has; a modality present on some ranks only runs a dummy on the
    others (images, when a dummy input with every required argument can be built) or raises on every
    rank. Models with no such encoder get no hook and pay nothing. Returns the host module (the vision
    one when there is one), or None."""
    if os.environ.get("UNSLOTH_FSDP2_VISION_DUMMY", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return None
    vision = _vision_plan(model)
    audio = _audio_plan(model, vision[0] if vision is not None else None)
    hosts = {}
    for plan in (vision, audio):
        if plan is not None and not getattr(plan[0], _INSTALLED, False):
            modalities = [plan[1]] if plan[1] is not None else []
            modalities += list(plan[3]) if len(plan) > 3 else []
            hosts.setdefault(id(plan[0]), (plan[0], []))[1].extend(modalities)
    generic = (
        None
        if getattr(model, _INSTALLED, False)
        else _generic_media_plan(model, [m for _, ms in hosts.values() for m in ms])
    )
    if generic is not None:
        # One agreement, on the outermost forward (every rank runs it, and the media kwargs pass
        # through it by name): the named modalities (and their dummies) plus every other media input.
        named = [m for _, ms in hosts.values() for m in ms]
        hosts = {id(generic[0]): (generic[0], named + ([generic[1]] if generic[1] else []))}
    if not hosts:
        return None
    from .fsdp2_save import model_mesh

    for host, modalities in hosts.values():
        encoders = [_subtree_tower(host, _vision_tower)] + _audio_modules(host)
        encoders += [host.__dict__.get("_modules", {}).get(n) for n in _VIDEO_TOWER_NAMES]
        ref = next((e for e in encoders if e is not None), host)
        _install_presence_agreement(
            host, modalities, lambda ref = ref: _tower_device(ref), model_mesh(model)
        )
    if vision is None or vision[2] is None or getattr(vision[0], _INSTALLED + "_dummy", False):
        return vision[0] if vision is not None else next(iter(hosts.values()))[0]
    mm, modality, tower = vision[:3]
    state = {"pending": None, "armed": False}
    # Run the dummy where an image rank runs its tower: just before the language model (after the input
    # embeddings and Gemma4's per-layer embedding unit), so per-block layouts gather units in the same
    # order on every rank. Without a `language_model` child, before the multimodal forward.
    language_model = getattr(mm, "language_model", None)
    if not isinstance(language_model, torch.nn.Module) or language_model is tower:
        language_model = None

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
        # After the agreement hook (registered first): run only when images are mixed and absent here.
        state["pending"] = None
        state["armed"] = False
        if not modality.run_dummy:
            return None
        if language_model is not None:
            state["armed"] = True
            return None
        total = _dummy_total()
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
    setattr(mm, _INSTALLED + "_dummy", True)
    _print_rank0(
        f"Unsloth: native FSDP2 runs {type(mm).__name__}'s vision tower on a dummy image for micro-batches "
        "when another rank has images, so every rank issues the same vision collectives (output added "
        "times 0)."
    )
    return mm
