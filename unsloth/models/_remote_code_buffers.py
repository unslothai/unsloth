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

"""Rebuild the non-persistent buffers of remote-code modules after a transformers 5 load.

transformers 5 builds the model under the meta device, gives every non-persistent buffer
fresh `torch.empty_like` storage and relies on the model's `_init_weights` to fill it.
Remote code written against transformers 4.x computed those buffers in `__init__` (RoPE
`inv_freq`, lightning-attention decay slopes, cos / sin caches) and its `_init_weights`
only touches Linear and Embedding weights, so they come back holding whatever the
allocator returned: zeros on a fresh card, which silently removes the rotary embedding
and the attention decay, or garbage. The model still trains, just as a different model.

Each such module is rebuilt once on the CPU with its parameters on the meta device, and
its non-persistent buffers are copied into the loaded ones in place (so device placement
and dispatch hooks are kept). Native transformers modules are left alone: their
`_init_weights` already recomputes these buffers.
"""

import ast
import inspect
import textwrap

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
            # A child of a composite model may have been built with a sub-config; only the
            # config the module kept is known to be the one it was built with.
            value = module.__dict__.get("config", None)
            if value is None:
                return None
        elif name in _PLACEMENT_ARGUMENTS:
            # A stored `self.device` can be the meta device transformers built on.
            value = None if parameter.default is inspect.Parameter.empty else parameter.default
        elif name in module.__dict__:
            value = module.__dict__[name]
        else:
            # An argument the module did not keep may have had a non-default value; guessing
            # the default could rebuild plausible but wrong buffers, so skip the module.
            return None
        if value is inspect.Parameter.empty:
            return None
        if isinstance(value, (torch.Tensor, torch.nn.Module)):
            # The buffers may be derived from it and it cannot be reproduced safely.
            return None
        kwargs[name] = value
    return kwargs


def _cache_key(module, kwargs):
    parts = []
    for name in sorted(kwargs):
        value = kwargs[name]
        # Typed repr: True and 1, or 0.0 and -0.0, compare equal but may build different buffers.
        if isinstance(value, _SCALAR_TYPES):
            parts.append((name, type(value), repr(value)))
        else:
            parts.append((name, id(value)))
    return type(module), tuple(parts)


def _remote_init_weights_identifiers(model):
    """The identifiers of each remote ``_init_weights`` found on the model, one set per
    implementation."""
    found = []
    seen = set()
    for module in model.modules():
        for cls in type(module).__mro__:
            init_weights = cls.__dict__.get("_init_weights")
            if (
                init_weights is None
                or cls in seen
                or not cls.__module__.startswith("transformers_modules")
            ):
                continue
            seen.add(cls)
            try:
                tree = ast.parse(textwrap.dedent(inspect.getsource(init_weights)))
            except (OSError, TypeError, SyntaxError):
                continue
            found.append(_code_identifiers(tree))
    return found


def _code_identifiers(tree):
    """Names, attributes and string arguments the code uses; comments and the docstring are
    not part of the tree walked, so prose that mentions a buffer does not count."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            # getattr(module, "inv_freq") / register_buffer("inv_freq", ...)
            names.add(node.value)
    return names


def _initialised_by_remote_code(module, buffer_name, init_identifiers):
    """Whether a remote ``_init_weights`` names both ``module``'s class (or a base) and the
    buffer, i.e. fills that buffer after construction for this kind of module."""
    classes = {
        cls.__name__
        for cls in type(module).__mro__
        if not cls.__module__.startswith(("torch", "builtins"))
    }
    return any(buffer_name in names and classes & names for names in init_identifiers)


def _fresh_non_persistent_buffers(module, kwargs, dtype):
    """Construct ``type(module)(**kwargs)`` on the CPU with parameters on meta and
    return its non-persistent buffers, plus the attributes that alias them."""
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
            buffers[name] = buffer
    aliases = {}
    for attribute, value in fresh.__dict__.items():
        if not isinstance(value, torch.Tensor) or attribute in ("_buffers", "_parameters"):
            continue
        for name, buffer in buffers.items():
            if value is buffer:
                aliases[attribute] = name
    return buffers, aliases


def restore_remote_code_non_persistent_buffers(model):
    """Recompute the non-persistent buffers of remote-code modules that transformers 5
    left uninitialised. Returns the number of buffers restored; a no-op on 4.x, where
    the model is built with real buffers."""
    if model is None or not _transformers_builds_on_meta():
        return 0
    dtype = getattr(model, "dtype", None)
    cache = {}
    restored = 0
    init_identifiers = _remote_init_weights_identifiers(model)
    for module in model.modules():
        own = getattr(module, "_non_persistent_buffers_set", None)
        if not own or not _is_remote_code_module(module):
            continue
        kwargs = _constructor_kwargs(module)
        if kwargs is None:
            continue
        key = _cache_key(module, kwargs)
        if key not in cache:
            try:
                cache[key] = _fresh_non_persistent_buffers(module, kwargs, dtype)
            except Exception:
                cache[key] = None
        if cache[key] is None:
            continue
        buffers, aliases = cache[key]
        for name, fresh in buffers.items():
            if _initialised_by_remote_code(module, name, init_identifiers):
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
