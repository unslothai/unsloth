# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generic text-encoder dead-tail truncation, decided by the pipeline's own ``encode_prompt``.

Many pipelines read an intermediate hidden state of a decoder-only text encoder and never the rest:
FLUX.2-klein stacks ``hidden_states[9, 18, 27]`` of a 36-layer Qwen3, Z-Image reads ``hidden_states[-2]``,
HunyuanVideo-1.5 reads ``hidden_states[-3]``, Qwen-Image reads ``hidden_states[-1]`` and never the ``lm_head``.
Layers after the last consumed one, and an unread head, cannot influence the conditioning, yet transformers
runs (and Studio places, casts and offloads) all of them. Instead of a per-family table, one probe asks the
pipeline which ones it reads:

1. Every slot of each encoder's decoder stack is swapped for a WEIGHTLESS passthrough that is a subclass of the
   real layer class with the same name, so transformers' hidden-state recorders still fire on it and
   ``len(hidden_states)`` (hence every negative index) is unchanged. The head becomes a free zero view.
   One discovery call runs the real slot 0 once to learn the layer's output format (tensor or tuple).
2. The pipeline's own ``encode_prompt`` runs on a short prompt. Slots are identities, so the call costs the
   embedding lookup, the final norm and the pipeline's post-processing only.
3. NaN taint: poisoning slot ``j`` poisons every hidden state after it. If no returned tensor changes, nothing
   after slot ``j - 1`` is read. Monotone in ``j``, so a bisection finds the last consumed slot ``L``; the head is
   tainted on its own.
4. Slots ``L+1..N-1`` become passthroughs (their weights are released) and an unread head becomes
   :class:`NoLogitsHead`. The kept layers run the same ops on the same inputs, so every hidden state the pipeline
   reads is bit-identical by construction.

Any surprise (no ``encode_prompt``, a required argument it cannot fill, a non-finite reference, an encoder the
probe never reached, an exception) keeps the full encoder. Kill switch: ``UNSLOTH_DIFFUSION_TE_TRUNCATE=0``.
"""

from __future__ import annotations

import inspect
import os
from functools import lru_cache
from typing import Any, Optional

TRUNCATE_ENV = "UNSLOTH_DIFFUSION_TE_TRUNCATE"
TRUNCATION_ATTR = "_unsloth_te_truncation"
PASSTHROUGH_FLAG = "_unsloth_te_passthrough"

_TEXT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4")
# Module-path tokens of towers that are not the language stack the pipeline reads hidden states of.
_NON_LANGUAGE_TOKENS = ("visual", "vision", "audio", "image_encoder", "patch_embed")
_PROBE_PROMPT = "a photo of a cat"
# nn.Module bookkeeping that a passthrough must own fresh, never share with the layer it replaces.
_MODULE_INTERNALS = frozenset(
    {
        "_parameters",
        "_buffers",
        "_modules",
        "_non_persistent_buffers_set",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_is_full_backward_hook",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
        "_hf_hook",
        "_old_forward",
        "forward",
    }
)
# Forward-hook registries a passthrough SHARES with the layer it stands in for. transformers >= 5 installs its
# hidden-state recorders once per model as forward hooks on each decoder layer (``output_capturing.py``) and marks the
# model hooked; sharing the dicts means a recorder installed on either object is seen by both, so the probe can never
# leave a restored layer unhooked and a dead-tail slot keeps recording its (passthrough) hidden state.
_SHARED_HOOK_DICTS = ("_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called")


def truncate_enabled() -> bool:
    return os.environ.get(TRUNCATE_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def is_passthrough(module: Any) -> bool:
    return bool(getattr(type(module), PASSTHROUGH_FLAG, False))


@lru_cache(maxsize = None)
def _passthrough_class(layer_cls: type) -> type:
    """A weightless subclass of ``layer_cls`` that returns its input. Same ``__name__``/``__qualname__`` so name- and
    isinstance-based output recorders treat it as the layer; built once per class (stable dynamo type guard)."""
    import torch

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        hidden = args[0] if args else kwargs.get("hidden_states")
        if self._unsloth_taint:
            hidden = torch.full_like(hidden, float("nan"))
        width = self._unsloth_fmt.get("tuple_len")
        if width is None:
            return hidden
        return (hidden,) + (None,) * (width - 1)

    def extra_repr(self) -> str:
        return "dead tail slot: passthrough, no weights"

    return type(
        layer_cls.__name__,
        (layer_cls,),
        {
            "forward": forward,
            "extra_repr": extra_repr,
            PASSTHROUGH_FLAG: True,
            "__module__": layer_cls.__module__,
            "__qualname__": layer_cls.__qualname__,
        },
    )


def _make_passthrough(layer: Any, fmt: dict, *, taint: bool = False) -> Any:
    import torch

    cls = _passthrough_class(type(layer))
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    for key, value in layer.__dict__.items():
        if key in _MODULE_INTERNALS or isinstance(value, (torch.nn.Module, torch.Tensor)):
            continue
        obj.__dict__[key] = value
    for key in _SHARED_HOOK_DICTS:
        if key in layer.__dict__:
            obj.__dict__[key] = layer.__dict__[key]
    obj.training = layer.training
    obj._unsloth_fmt = fmt
    obj._unsloth_taint = taint
    return obj


def _find_stack(encoder: Any) -> Optional[tuple[str, Any]]:
    """The language decoder stack: the homogeneous ModuleList with the most parameters outside vision towers."""
    import torch

    best: Optional[tuple[str, Any]] = None
    best_params = 0
    for name, module in encoder.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) < 2:
            continue
        if any(token in name.lower() for token in _NON_LANGUAGE_TOKENS):
            continue
        kinds = {type(m) for m in module if not is_passthrough(m)}
        if len(kinds) != 1:
            continue
        params = sum(p.numel() for p in module.parameters())
        if params > best_params:
            best, best_params = (name, module), params
    return best


def _find_head(encoder: Any) -> Optional[tuple[str, Any]]:
    import torch

    getter = getattr(encoder, "get_output_embeddings", None)
    head = None
    try:
        head = getter() if callable(getter) else None
    except Exception:  # noqa: BLE001
        head = None
    if head is None:
        head = getattr(encoder, "lm_head", None)
    if not isinstance(head, torch.nn.Module):
        return None
    for name, module in encoder.named_children():
        if module is head:
            return name, head
    return None


def _set_child(root: Any, dotted: str, value: Any) -> None:
    parent = root
    *path, leaf = dotted.split(".")
    for part in path:
        parent = getattr(parent, part)
    setattr(parent, leaf, value)


@lru_cache(maxsize = None)
def _stub_head_class() -> type:
    import torch

    class ProbeHead(torch.nn.Module):
        """Free stand-in head for the probe: an expanded zero (or NaN) view of logits shape."""

        def __init__(self, out_features: int) -> None:
            super().__init__()
            self.out_features = out_features
            self.taint = False

        def forward(self, hidden_states: Any) -> Any:  # noqa: D102
            fill = float("nan") if self.taint else 0.0
            return hidden_states.new_full((1,), fill).expand(*hidden_states.shape[:-1], self.out_features)

    return ProbeHead


def _collect_tensors(value: Any, out: list) -> list:
    import torch

    if isinstance(value, torch.Tensor):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_tensors(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_tensors(item, out)
    elif hasattr(value, "to_tuple"):
        _collect_tensors(value.to_tuple(), out)
    return out


def _encode_kwargs(pipe: Any, device: Any) -> Optional[dict]:
    encode = getattr(pipe, "encode_prompt", None)
    if not callable(encode):
        return None
    try:
        params = inspect.signature(encode).parameters
    except (TypeError, ValueError):
        return None
    kwargs: dict = {}
    for name, param in params.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in ("prompt", "prompt_2", "prompt_3", "prompt_4"):
            kwargs[name] = _PROBE_PROMPT
        elif name == "device":
            kwargs[name] = device
        elif name in ("num_images_per_prompt", "num_videos_per_prompt"):
            kwargs[name] = 1
        elif name == "do_classifier_free_guidance":
            kwargs[name] = False
        elif param.default is inspect.Parameter.empty:
            return None
    return kwargs if "prompt" in kwargs else None


class _Target:
    def __init__(self, attr: str, encoder: Any, stack_name: str, stack: Any, head: Optional[tuple[str, Any]]):
        self.attr = attr
        self.encoder = encoder
        self.stack_name = stack_name
        self.stack = stack
        self.originals = list(stack)
        self.head_name, self.head = head if head is not None else (None, None)
        self.fmt: dict = {}
        self.called = 0


def _head_out_features(head: Any) -> Optional[int]:
    weight = getattr(head, "weight", None)
    if getattr(weight, "ndim", 0) == 2:
        return int(weight.shape[0])
    out = getattr(head, "out_features", None)
    return int(out) if isinstance(out, int) else None


def _bytes_of(modules: list) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for tensor in list(module.parameters()) + list(module.buffers()):
            key = id(tensor)
            if key in seen:
                continue
            seen.add(key)
            total += tensor.numel() * tensor.element_size()
    return total


def probe_consumption(pipe: Any, encode_kwargs: Optional[dict] = None) -> dict:
    """Run the NaN-taint probe; returns ``{attr: {"layers": N, "last_consumed": L, "head_consumed": bool}}``.

    Restores every swapped module before returning, also on failure. Raises on any probe failure."""
    import torch

    targets: list[_Target] = []
    for attr in _TEXT_ENCODER_ATTRS:
        encoder = getattr(pipe, attr, None)
        if not isinstance(encoder, torch.nn.Module) or getattr(encoder, TRUNCATION_ATTR, None):
            continue
        found = _find_stack(encoder)
        if found is None or any(is_passthrough(m) for m in found[1]):
            continue
        targets.append(_Target(attr, encoder, found[0], found[1], _find_head(encoder)))
    if not targets:
        return {}
    devices = {p.device for t in targets for p in t.encoder.parameters()}
    if any(d.type == "meta" for d in devices):
        raise RuntimeError("an encoder holds meta tensors")
    device = next(iter(targets[0].encoder.parameters())).device
    kwargs = encode_kwargs if encode_kwargs is not None else _encode_kwargs(pipe, device)
    if kwargs is None:
        raise RuntimeError(f"{type(pipe).__name__}.encode_prompt cannot be called generically")

    ProbeHead = _stub_head_class()
    swapped = list(targets)
    handles = []
    try:
        # Discovery: real slot 0 records the layer's output format, every later slot is already a passthrough.
        for target in targets:
            fmt = target.fmt

            def record(_m: Any, _a: Any, output: Any, fmt: dict = fmt) -> None:
                fmt["tuple_len"] = len(output) if isinstance(output, tuple) else None
                fmt["seen"] = True

            def count(_m: Any, _a: Any, target: _Target = target) -> None:
                target.called += 1

            handles.append(target.originals[0].register_forward_hook(record))
            handles.append(target.encoder.register_forward_pre_hook(count))
            for i in range(1, len(target.stack)):
                target.stack[i] = _make_passthrough(target.originals[i], fmt)
            if target.head is not None:
                out_features = _head_out_features(target.head)
                if out_features is None:
                    target.head = None
                else:
                    _set_child(target.encoder, target.head_name, ProbeHead(out_features))

        def run() -> list:
            with torch.inference_mode():
                return [t.detach() for t in _collect_tensors(pipe.encode_prompt(**kwargs), [])]

        run()
        for handle in handles:
            handle.remove()
        handles.clear()
        targets = [t for t in targets if t.called and t.fmt.get("seen")]
        if not targets:
            return {}
        for target in targets:
            target.stack[0] = _make_passthrough(target.originals[0], target.fmt)
        reference = run()
        if not reference or not all(bool(torch.isfinite(t.float()).all()) for t in reference if t.is_floating_point()):
            raise RuntimeError("probe reference is empty or non-finite")

        def unchanged() -> bool:
            got = run()
            if len(got) != len(reference):
                return False
            for a, b in zip(got, reference):
                if a.shape != b.shape or a.dtype != b.dtype:
                    return False
                if a.is_floating_point() and not bool(torch.isfinite(a.float()).all()):
                    return False
                if not torch.equal(a, b):
                    return False
            return True

        result: dict = {}
        for target in targets:
            n = len(target.stack)

            def consumed(j: int, target: _Target = target) -> bool:
                target.stack[j]._unsloth_taint = True
                try:
                    return not unchanged()
                finally:
                    target.stack[j]._unsloth_taint = False

            if consumed(n - 1):
                last = n - 1
            elif not consumed(0):
                last = -1
            else:
                lo, hi = 0, n - 1  # consumed(lo) and not consumed(hi)
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if consumed(mid):
                        lo = mid
                    else:
                        hi = mid
                last = lo
            head_consumed = True
            if target.head is not None:
                stub = getattr(target.encoder, target.head_name)
                stub.taint = True
                try:
                    head_consumed = not unchanged()
                finally:
                    stub.taint = False
            result[target.attr] = {
                "layers": n,
                "last_consumed": last,
                "head_consumed": head_consumed,
                "head": target.head_name,
                "stack": target.stack_name,
                "tuple_len": target.fmt.get("tuple_len"),
            }
        return result
    finally:
        for handle in handles:
            handle.remove()
        for target in swapped:
            for i, layer in enumerate(target.originals):
                target.stack[i] = layer
            if target.head is not None:
                _set_child(target.encoder, target.head_name, target.head)


def apply_truncation(pipe: Any, plan: dict) -> dict:
    """Swap dead slots for passthroughs and an unread head for ``NoLogitsHead``; returns per-encoder savings."""
    from .diffusion_text_encoder_trim import _no_logits_head_class

    report: dict = {}
    for attr, info in plan.items():
        encoder = getattr(pipe, attr, None)
        found = _find_stack(encoder) if encoder is not None else None
        if found is None or found[0] != info["stack"] or len(found[1]) != info["layers"]:
            continue
        stack = found[1]
        n, last = info["layers"], info["last_consumed"]
        dropped_layers: list = []
        if last < 0:
            # The probe reached the encoder but read nothing from its stack: an unexpected contract, keep it whole.
            last = n - 1
        fmt = {"tuple_len": info.get("tuple_len"), "seen": True}
        for i in range(last + 1, n):
            dropped_layers.append(stack[i])
            stack[i] = _make_passthrough(stack[i], fmt)
        head_bytes, head_dropped = 0, False
        head = getattr(encoder, info["head"], None) if info.get("head") else None
        if head is not None and not info["head_consumed"] and type(head).__name__ != "NoLogitsHead":
            out_features = _head_out_features(head)
            weight = getattr(head, "weight", None)
            if out_features is not None and getattr(weight, "ndim", 0) == 2:
                embed = None
                try:
                    embed = encoder.get_input_embeddings()
                except Exception:  # noqa: BLE001
                    pass
                tied = embed is not None and getattr(embed, "weight", None) is not None and (
                    embed.weight is weight or embed.weight.data_ptr() == weight.data_ptr()
                )
                head_bytes = 0 if tied else _bytes_of([head])
                setattr(encoder, info["head"], _no_logits_head_class()(int(weight.shape[1]), out_features))
                head_dropped = True
        layer_bytes = _bytes_of(dropped_layers)
        entry = {
            "layers": n,
            "kept_layers": last + 1,
            "dropped_layers": len(dropped_layers),
            "head_dropped": head_dropped,
            "bytes_freed": layer_bytes + head_bytes,
        }
        if dropped_layers or head_dropped:
            setattr(encoder, TRUNCATION_ATTR, entry)
            report[attr] = entry
    return report


def truncate_text_encoders(pipe: Any, *, logger: Any = None, encode_kwargs: Optional[dict] = None) -> dict:
    """Probe + apply. Never raises: any failure keeps every encoder whole and returns ``{"skipped": reason}``."""
    if pipe is None:
        return {"skipped": "no pipeline"}
    if not truncate_enabled():
        return {"skipped": TRUNCATE_ENV}
    try:
        plan = probe_consumption(pipe, encode_kwargs)
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.info("diffusion.text_encoder: dead-tail probe declined (%s); encoders kept whole", exc)
        return {"skipped": f"probe: {exc}"}
    if not plan:
        return {"skipped": "nothing to probe"}
    try:
        report = apply_truncation(pipe, plan)
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.warning("diffusion.text_encoder: dead-tail truncation failed (%s)", exc)
        return {"skipped": f"apply: {exc}", "plan": plan}
    if logger is not None:
        for attr, entry in report.items():
            logger.info(
                "diffusion.text_encoder: %s runs %d/%d decoder layers%s; %.2f GiB never placed "
                "(probe-verified unread, outputs bit-identical)",
                attr,
                entry["kept_layers"],
                entry["layers"],
                ", lm_head dropped" if entry["head_dropped"] else "",
                entry["bytes_freed"] / 2**30,
            )
    return {"plan": plan, "truncated": report}


def live_slots(stack: Any) -> list[int]:
    """Indices of a ModuleList that still run real weights (for keep-lists that count from the end)."""
    return [i for i, m in enumerate(stack) if not is_passthrough(m)]


__all__: list[str] = [
    "TRUNCATE_ENV",
    "truncate_enabled",
    "is_passthrough",
    "probe_consumption",
    "apply_truncation",
    "truncate_text_encoders",
    "live_slots",
]
