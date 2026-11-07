# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""gradient_checkpointing_enable(every_n_layers, offload) for both import routes (GPU _utils.py, CPU _cpu_init.py).
Imports transformers lazily so the CPU route avoids the GPU model stack."""

import functools
import inspect


def _skip_aware_flag(cls):
    # for_training, gradient_checkpointing_enable and the trainer rewrite the flag; a marked layer keeps reading False.
    if cls.__dict__.get("_unsloth_skip_aware", False):
        return

    def _get(self):
        if self.__dict__.get("_unsloth_skip_checkpoint", False) or self.__dict__.get(
            "_unsloth_every_n_skip", False
        ):
            return False
        # Unmarked layers of the class still hold the flag in their own dict.
        return self.__dict__.get(
            "_unsloth_gc_flag", self.__dict__.get("gradient_checkpointing", False)
        )

    def _set(self, value):
        self.__dict__["_unsloth_gc_flag"] = value

    cls.gradient_checkpointing = property(_get, _set)
    cls._unsloth_skip_aware = True


def _apply_every_n_layers(model, every_n_layers):
    """Checkpoint GradientCheckpointingLayer 0, N, 2N...; held by the skip-aware flag so for_training keeps it."""
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
    except Exception:
        return []
    layers = [
        m
        for m in model.modules()
        if isinstance(m, GradientCheckpointingLayer)
        or "_unsloth_gradient_checkpointing_holder" in m.__dict__
    ]
    skipped = []
    for i, layer in enumerate(layers):
        if every_n_layers > 1 and i % every_n_layers != 0:
            if not isinstance(layer, GradientCheckpointingLayer):
                layer.__dict__["_unsloth_every_n_skip"] = True
                skipped.append(i)
                continue
            if not hasattr(layer, "gradient_checkpointing"):
                continue
            current = layer.__dict__.pop("gradient_checkpointing", None)
            _skip_aware_flag(type(layer))
            if current is not None:
                layer.__dict__["_unsloth_gc_flag"] = current
            layer.__dict__["_unsloth_every_n_skip"] = True
            skipped.append(i)
        else:
            layer.__dict__.pop("_unsloth_every_n_skip", None)
    return skipped


_OFFLOADING_CHECKPOINT_SHIMS = frozenset(
    {"unsloth_checkpoint", "unsloth_offloaded_gradient_checkpoint"}
)


def _checkpoint_offloads(fn):
    # Unsloth checkpoint functions that offload set `_unsloth_offloads`; the zoo shims are known by name.
    return bool(getattr(fn, "_unsloth_offloads", False)) or (
        getattr(fn, "__name__", "") in _OFFLOADING_CHECKPOINT_SHIMS
    )


_GC_KWARGS_WARNED = set()


def _gc_kwargs_note(key, message):
    if key not in _GC_KWARGS_WARNED:
        _GC_KWARGS_WARNED.add(key)
        print(message)


def patch_gradient_checkpointing_enable_kwargs():
    """Pop every_n_layers / offload from gradient_checkpointing_kwargs; offload under Unsloth's offloading shim would double-offload."""
    try:
        from transformers import PreTrainedModel
        import transformers.modeling_utils as hf_modeling_utils
    except Exception:
        return
    # transformers >= 5.23: both public names and instance wrappers of the old one end here.
    name = next(
        (
            n
            for n in ("_activation_checkpointing_enable", "activation_checkpointing_enable")
            if hasattr(PreTrainedModel, n)
        ),
        "gradient_checkpointing_enable",
    )
    original = getattr(PreTrainedModel, name)
    if getattr(original, "_unsloth_gc_kwargs", False):
        return
    try:
        params = list(inspect.signature(original).parameters)
    except Exception:
        return
    kwargs_name = name.lstrip("_").replace("_enable", "_kwargs")
    positional = [p for p in params if p not in ("self", kwargs_name)]
    native_every = "every_n_layers" in params
    native_offload = "offload" in params
    # transformers >= 5.23 always passes the defaults (1, False) here, so they cannot override the dict.
    forwards_defaults = name == "_activation_checkpointing_enable"

    @functools.wraps(original)
    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs = None,
        *args,
        **kwargs,
    ):
        gradient_checkpointing_kwargs = kwargs.pop(kwargs_name, gradient_checkpointing_kwargs)
        kwargs.update(zip(positional, args))
        every = kwargs.pop("every_n_layers", None)
        offload = kwargs.pop("offload", None)
        if gradient_checkpointing_kwargs is not None:
            gc_kwargs = dict(gradient_checkpointing_kwargs)
            dict_every = gc_kwargs.pop("every_n_layers", None)
            dict_offload = gc_kwargs.pop("offload", None)
            if forwards_defaults:
                every = None if every == 1 else every
                offload = offload or None
            every = dict_every if every is None else every
            offload = dict_offload if offload is None else offload
            if len(gc_kwargs) != len(gradient_checkpointing_kwargs):
                gradient_checkpointing_kwargs = gc_kwargs or None
        every = 1 if every is None else int(every)
        if every < 1:
            raise ValueError(f"Unsloth: every_n_layers must be >= 1, got {every}.")
        if offload:
            shim = getattr(hf_modeling_utils, "checkpoint", None)
            if _checkpoint_offloads(shim):
                _gc_kwargs_note(
                    "offload",
                    "Unsloth: gradient checkpointing already offloads activations to host memory, "
                    "so transformers' offload=True is ignored.",
                )
                offload = False
            elif not native_offload:
                _gc_kwargs_note(
                    "offload_native",
                    "Unsloth: gradient checkpointing offload=True needs transformers >= 5.17; ignored. "
                    'use_gradient_checkpointing = "unsloth" offloads activations on any version.',
                )
                offload = False
        if native_offload and offload:
            kwargs["offload"] = True
        if native_every:
            kwargs["every_n_layers"] = 1
        out = original(self, gradient_checkpointing_kwargs, **kwargs)
        if every > 1 or any("_unsloth_every_n_skip" in m.__dict__ for m in self.modules()):
            skipped = _apply_every_n_layers(self, every)
            if every > 1 and not skipped:
                _gc_kwargs_note(
                    "every_n_layers",
                    f"Unsloth: every_n_layers = {every} found no GradientCheckpointingLayer to skip; "
                    "every layer stays checkpointed.",
                )
        return out

    gradient_checkpointing_enable._unsloth_gc_kwargs = True
    setattr(PreTrainedModel, name, gradient_checkpointing_enable)
