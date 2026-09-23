# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""transformers 5 gives non-persistent buffers `torch.empty_like` storage and leaves them to
`_init_weights`; 4.x-era remote code computes them in `__init__` (RoPE inv_freq, decay slopes),
so they load as garbage. Rebuild each such module on CPU with meta parameters and copy its
buffers into the loaded ones in place, keeping device placement and dispatch hooks."""

import inspect

import torch

__all__ = ["restore_remote_code_non_persistent_buffers"]

_SCALAR_TYPES = (bool, int, float, str, type(None))
# Arguments that only choose where buffers are built, not their values.
_PLACEMENT_ARGUMENTS = ("device",)


def _transformers_builds_on_meta():
    try:
        import transformers
        from packaging.version import Version
        return Version(Version(transformers.__version__).base_version).major >= 5
    except Exception:
        return False


def _is_remote_code_module(module):
    return type(module).__module__.startswith("transformers_modules")


def _constructor_kwargs(module):
    """Arguments to rebuild ``module`` from its own attributes, or None when a required
    one cannot be recovered."""
    try:
        signature = inspect.signature(type(module).__init__)
    except (TypeError, ValueError):
        return None
    kwargs = {}
    for name, parameter in list(signature.parameters.items())[1:]:
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            # Whatever went through *args / **kwargs cannot be recovered from the instance.
            return None
        if name == "config":
            # A composite child may have been built with a sub-config: trust only a kept one.
            value = module.__dict__.get("config", None)
            if value is None:
                return None
        elif name in _PLACEMENT_ARGUMENTS:
            # A stored `self.device` can be the meta device transformers built on.
            value = None if parameter.default is inspect.Parameter.empty else parameter.default
        elif name in module.__dict__:
            value = module.__dict__[name]
        else:
            # Guessing the default for an unkept argument could build plausible but wrong buffers.
            return None
        if value is inspect.Parameter.empty:
            return None
        if isinstance(value, (torch.Tensor, torch.nn.Module)):
            # The buffers may be derived from it and it cannot be reproduced safely.
            return None
        kwargs[name] = value
    return kwargs


def _modules_with_init_weights(model):
    """Each module of ``model`` once, with the ``_init_weights`` transformers 5 runs on it:
    that of the nearest enclosing PreTrainedModel, as `initialize_weights` dispatches it."""
    try:
        from transformers import PreTrainedModel
    except Exception:
        PreTrainedModel = ()
    seen = set()
    stack = [(model, getattr(model, "_init_weights", None))]
    while stack:
        module, init_weights = stack.pop()
        if isinstance(module, PreTrainedModel):
            init_weights = getattr(module, "_init_weights", None)
        if id(module) in seen:
            continue
        seen.add(id(module))
        yield module, init_weights
        stack.extend((child, init_weights) for child in reversed(list(module.children())))


def _cache_key(module, kwargs, init_weights):
    parts = []
    for name in sorted(kwargs):
        value = kwargs[name]
        # Typed repr: True and 1, or 0.0 and -0.0, compare equal but may build different buffers.
        if isinstance(value, _SCALAR_TYPES):
            parts.append((name, type(value), repr(value)))
        else:
            parts.append((name, id(value)))
    # The probe result depends on which `_init_weights` runs, so sub-models do not share it.
    owner = getattr(init_weights, "__self__", init_weights)
    return type(module), tuple(parts), id(owner)


def _written_by_init_weights(fresh, buffers, init_weights):
    """Buffers ``init_weights`` writes on ``fresh``, probed with sentinels (NaN for floats, 0
    then 1 otherwise); None when it raises, since its live writes then cannot be told apart."""
    if init_weights is None or not buffers:
        return set()
    others = [name for name, buffer in buffers.items() if not buffer.is_floating_point()]
    written = set()
    with torch.no_grad():
        for sentinel in (0, 1) if others else (0,):
            for name, buffer in buffers.items():
                fill = float("nan") if buffer.is_floating_point() else sentinel
                fresh._buffers[name] = torch.full_like(buffer, fill)
            try:
                init_weights(fresh)
            except Exception:
                return None
            for name, buffer in buffers.items():
                after = fresh._buffers.get(name, None)
                if after is None or after.is_meta:
                    written.add(name)
                elif buffer.is_floating_point():
                    if not torch.isnan(after).all():
                        written.add(name)
                elif not after.eq(sentinel).all():
                    written.add(name)
    return written


def _fresh_non_persistent_buffers(
    module,
    kwargs,
    dtype,
    init_weights = None,
):
    """Rebuild ``module`` on CPU with meta parameters; return its non-persistent buffers,
    the attributes aliasing them, and the ones ``init_weights`` fills itself."""
    from accelerate import init_empty_weights

    previous_dtype = torch.get_default_dtype()
    try:
        # transformers constructs under the load dtype as the default dtype.
        if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
        with torch.device("cpu"), init_empty_weights(include_buffers = False):
            fresh = type(module)(**kwargs)
    finally:
        torch.set_default_dtype(previous_dtype)
    buffers = {}
    for name in getattr(fresh, "_non_persistent_buffers_set", ()):
        buffer = fresh._buffers.get(name, None)
        if buffer is not None and not buffer.is_meta:
            buffers[name] = buffer.clone()
    aliases = {}
    for attribute, value in fresh.__dict__.items():
        if not isinstance(value, torch.Tensor) or attribute in ("_buffers", "_parameters"):
            continue
        for name in buffers:
            if value is fresh._buffers.get(name, None):
                aliases[attribute] = name
    # After the aliases are read: the probe below replaces fresh's buffers.
    written = _written_by_init_weights(fresh, buffers, init_weights)
    return buffers, aliases, written


def restore_remote_code_non_persistent_buffers(model):
    """Recompute the non-persistent buffers of remote-code modules that transformers 5
    left uninitialised. Returns the number of buffers restored; a no-op on 4.x, where
    the model is built with real buffers."""
    if model is None or not _transformers_builds_on_meta():
        return 0
    dtype = getattr(model, "dtype", None)
    cache = {}
    restored = 0
    # transformers 5 runs the nearest PreTrainedModel's `_init_weights` on every module after
    # building on meta; a buffer it writes already holds the right value.
    for module, init_weights in _modules_with_init_weights(model):
        own = getattr(module, "_non_persistent_buffers_set", None)
        if not own or not _is_remote_code_module(module):
            continue
        kwargs = _constructor_kwargs(module)
        if kwargs is None:
            continue
        key = _cache_key(module, kwargs, init_weights)
        if key not in cache:
            try:
                cache[key] = _fresh_non_persistent_buffers(module, kwargs, dtype, init_weights)
            except Exception:
                cache[key] = None
        if cache[key] is None:
            continue
        buffers, aliases, written = cache[key]
        if written is None:
            continue
        for name, fresh in buffers.items():
            if name in written:
                continue
            live = module._buffers.get(name, None)
            if live is None or live.is_meta or live.shape != fresh.shape:
                continue
            with torch.no_grad():
                live.copy_(fresh.to(dtype = live.dtype))
            restored += 1
        for attribute, name in aliases.items():
            live = module._buffers.get(name, None)
            if live is not None and not live.is_meta:
                module.__dict__[attribute] = live
    if restored:
        print(
            f"Unsloth: Recomputed {restored} non-persistent buffers (RoPE frequencies and similar) "
            "that transformers 5 does not initialise for remote code."
        )
    return restored
