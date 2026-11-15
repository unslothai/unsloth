# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prompt encodes that read a few hidden states keep only those (UNSLOTH_DIFFUSION_TE_SELECT_HIDDEN=0 disables).

The encoder runs with ``output_hidden_states=False`` and hooks record only the requested layers; reading any other
index raises ``SelectionError`` so the caller re-runs stock."""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator, Optional

SELECT_ENV = "UNSLOTH_DIFFUSION_TE_SELECT_HIDDEN"
_CALLS = [0]


class SelectionError(RuntimeError):
    """The selective encode could not answer what the caller read: re-run stock."""


def enabled() -> bool:
    return (os.environ.get(SELECT_ENV) or "1").strip().lower() not in ("0", "off", "false", "no")


def call_count() -> int:
    return _CALLS[0]


class SelectedHiddenStates:
    """The requested entries of the stock ``hidden_states`` tuple (length ``n + 1``: embeddings, then one per
    decoder layer, the last per the caller's contract); reading any other entry raises ``SelectionError``."""

    __slots__ = ("_length", "_states")

    def __init__(self, length: int, states: dict) -> None:
        self._length = length
        self._states = states

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: Any) -> Any:
        if not isinstance(index, int) or isinstance(index, bool):
            raise SelectionError(f"hidden_states[{index!r}] is not a plain index")
        k = index + self._length if index < 0 else index
        if k not in self._states:
            raise SelectionError(f"hidden_states[{index}] was not recorded")
        return self._states[k]

    def __iter__(self) -> Iterator:
        raise SelectionError("the selective hidden states cannot be iterated")


def _text_config(encoder: Any) -> Any:
    config = getattr(encoder, "config", None)
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001
            pass
    return getattr(config, "text_config", None) or config


def _stack(encoder: Any) -> Optional[tuple[str, Any]]:
    from .diffusion_te_truncate import _find_stack
    return _find_stack(encoder)


def _parent(root: Any, dotted: str) -> Any:
    parent = root
    for part in dotted.split(".")[:-1]:
        parent = getattr(parent, part, None)
        if parent is None:
            return None
    return parent


def plan(
    encoder: Any,
    indices: Any,
    *,
    last_pre_norm: bool = False,
) -> Optional[dict]:
    """``{"length", "layers": {k: stack slot}, "norm"}`` for the requested indices, or None (run stock).

    Entry ``k`` (1 <= k < n) is decoder layer ``k - 1``'s output; entry ``n`` (or -1) only with ``last_pre_norm``,
    the caller's guarantee that its stock reading is the final norm's input. Entry 0 (embeddings) is not handled."""
    import torch

    if not enabled() or not isinstance(encoder, torch.nn.Module):
        return None
    if type(encoder).__name__ == "OptimizedModule":
        return None
    found = _stack(encoder)
    if found is None:
        return None
    name, stack = found
    n = len(stack)
    expected = getattr(_text_config(encoder), "num_hidden_layers", None)
    if not isinstance(expected, int) or expected != n:
        return None
    try:
        wanted = [int(k) for k in indices]
    except (TypeError, ValueError):
        return None
    if not wanted:
        return None
    layers: dict[int, int] = {}
    norm = None
    for raw in wanted:
        k = raw + n + 1 if raw < 0 else raw
        if 1 <= k < n:
            layers[k] = k - 1
        elif k == n and last_pre_norm:
            norm = getattr(_parent(encoder, name), "norm", None)
            if not isinstance(norm, torch.nn.Module):
                return None
        else:
            return None
    return {"length": n + 1, "layers": layers, "norm": norm, "stack": stack, "n": n}


@contextlib.contextmanager
def selected_hidden_states(
    encoder: Any,
    indices: Any,
    *,
    last_pre_norm: bool = False,
) -> Iterator[bool]:
    """Yields True when armed (the next ``output_hidden_states=True`` forward of ``encoder`` records only
    ``indices``), False when the caller runs stock. Hooks are removed on exit."""
    p = plan(encoder, indices, last_pre_norm = last_pre_norm)
    if p is None:
        yield False
        return
    state: dict = {"active": False, "states": {}}
    handles = []

    def pre(_module: Any, args: tuple, kwargs: dict) -> Any:
        if state["active"] or not kwargs.get("output_hidden_states"):
            return None
        state["active"] = True
        state["states"] = {}
        kwargs = dict(kwargs)
        kwargs["output_hidden_states"] = False
        return args, kwargs

    def layer_hook(k: int) -> Any:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            if state["active"]:
                state["states"][k] = output[0] if isinstance(output, tuple) else output

        return hook

    def norm_hook(_module: Any, args: Any, _output: Any) -> None:
        if state["active"] and args:
            state["states"][p["n"]] = args[0]

    def post(_module: Any, _args: Any, _kwargs: dict, output: Any) -> Any:
        if not state["active"]:
            return None
        state["active"] = False
        states, state["states"] = state["states"], {}
        want = set(p["layers"]) | ({p["n"]} if p["norm"] is not None else set())
        if set(states) != want or getattr(output, "hidden_states", None) is not None:
            # a recorder that did not fire, or a forward that still built the full tuple: refuse the result
            raise SelectionError("the selective encode did not record exactly the requested states")
        selected = SelectedHiddenStates(p["length"], states)
        try:
            output["hidden_states"] = selected
        except Exception:  # noqa: BLE001 - not a ModelOutput
            try:
                output.hidden_states = selected
            except Exception as exc:  # noqa: BLE001
                raise SelectionError(
                    f"cannot attach hidden states to {type(output).__name__}"
                ) from exc
        _CALLS[0] += 1
        return output

    try:
        handles.append(encoder.register_forward_pre_hook(pre, with_kwargs = True))
        for k, slot in p["layers"].items():
            handles.append(p["stack"][slot].register_forward_hook(layer_hook(k)))
        if p["norm"] is not None:
            handles.append(p["norm"].register_forward_hook(norm_hook))
        handles.append(encoder.register_forward_hook(post, with_kwargs = True))
        yield True
    finally:
        state["active"] = False
        state["states"] = {}
        for handle in handles:
            handle.remove()


def run_selected(
    encoder: Any,
    indices: Any,
    call: Any,
    *,
    last_pre_norm: bool = False,
) -> Any:
    """``call()`` under ``selected_hidden_states``; a ``SelectionError`` re-runs ``call()`` stock."""
    try:
        with selected_hidden_states(encoder, indices, last_pre_norm = last_pre_norm) as armed:
            if armed:
                return call()
    except SelectionError:
        pass
    return call()


__all__ = [
    "SELECT_ENV",
    "SelectedHiddenStates",
    "SelectionError",
    "call_count",
    "enabled",
    "plan",
    "run_selected",
    "selected_hidden_states",
]
