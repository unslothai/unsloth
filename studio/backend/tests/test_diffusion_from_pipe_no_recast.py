# SPDX-License-Identifier: Apache-2.0
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""#9186: the bundled diffusers resolves ``torch_dtype=None`` inside
``from_pipe`` to float32 and then calls ``.to(dtype)`` on every component,
which hard-crashes GGUF-quantized transformers. ``_from_pipe_no_recast``
keeps the ``from_pipe`` fast path and falls back to re-wiring the resident
components without any cast when that crash fires.

The helper is extracted from diffusion.py via its AST and executed in an
isolated namespace (the same convention test_chat_completions_sanitize_floats
established), so this file imports neither torch nor diffusers and runs
wherever the module parses."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

SOURCE = (Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py").read_text()


def _extract_static_method(name: str):
    """Compile the whole FunctionDef at module level so `return` keeps its
    function context; the resulting namespace holds the function itself."""
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            node.decorator_list = []
            code = compile(
                ast.Module(body = [node], type_ignores = []), filename = str(SOURCE), mode = "exec"
            )
            namespace: dict = {"__builtins__": __builtins__}
            exec(code, namespace)
            return namespace[name]
    raise AssertionError(f"{name} not found in diffusion.py")


_from_pipe_no_recast = _extract_static_method("_from_pipe_no_recast")


class _QuantizedCaster:
    """A from_pipe that mimics the bundled diffusers' fp32 recast crash."""

    calls: list[dict] = []

    def __init__(self, **components):
        self.components = components

    @classmethod
    def from_pipe(cls, base_pipe, **kwargs):
        cls.calls.append(kwargs)
        raise ValueError(
            "Casting a quantized model to a new `dtype` is unsupported. "
            "To set the dtype of unquantized layers, please use the "
            "`torch_dtype` argument when loading the model."
        )


class _FriendlyPipe:
    """A from_pipe that succeeds — the normal path."""

    def __init__(self, **components):
        self.components = components

    @classmethod
    def from_pipe(cls, base_pipe, **kwargs):
        cls.calls.append(kwargs)
        return cls(**base_pipe.components)


class _Base:
    def __init__(self):
        self.components = {
            "transformer": object(),
            "vae": object(),
            "text_encoder": object(),
        }


def test_fast_path_still_uses_from_pipe():
    _FriendlyPipe.calls = []
    base = _Base()
    _from_pipe_no_recast(base, _FriendlyPipe)
    assert _FriendlyPipe.calls == [{"torch_dtype": None}]


def test_quantized_crash_falls_back_to_component_rewiring():
    _QuantizedCaster.calls = []
    base = _Base()
    pipe = _from_pipe_no_recast(base, _QuantizedCaster)
    # from_pipe WAS tried with the no-recast intent first...
    assert _QuantizedCaster.calls == [{"torch_dtype": None}]
    # ...and the fallback reuses the same resident module references.
    assert pipe.components["transformer"] is base.components["transformer"]
    assert pipe.components["vae"] is base.components["vae"]


def test_unrelated_errors_still_raise():
    class _Broken:
        @classmethod
        def from_pipe(cls, base_pipe, **kwargs):
            raise ValueError("some other failure")

    with pytest.raises(ValueError, match = "some other failure"):
        _from_pipe_no_recast(_Base(), _Broken)


def test_extra_components_forward_to_the_fallback():
    class _StrictSig:
        def __init__(
            self,
            transformer = None,
            vae = None,
            text_encoder = None,
            controlnet = None,
        ):
            self.transformer = transformer
            self.vae = vae
            self.controlnet = controlnet

        @classmethod
        def from_pipe(cls, base_pipe, **kwargs):
            raise ValueError("Casting a quantized model to a new dtype is unsupported")

    base = _Base()
    cn = object()
    pipe = _from_pipe_no_recast(base, _StrictSig, controlnet = cn)
    assert pipe.controlnet is cn
    assert pipe.transformer is base.components["transformer"]
