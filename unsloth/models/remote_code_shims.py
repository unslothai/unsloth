# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Repairs for remote modeling code (vLLM ports such as Step-3.7-Flash) that breaks training:

* `get_input_embeddings(self, input_ids)` returns embedded tokens, not the embedding module.
* `forward(..., labels=...)` accepts labels but returns no loss, or fails in its loss code.

Only the checkpoint's own classes are patched, and the original behaviour stays reachable.
"""

import functools
import inspect

import torch

__all__ = [
    "apply_remote_code_shims",
    "accessor_requires_arguments",
    "find_embedding_module",
]

_EMBEDDING_ATTRIBUTES = ("embed_tokens", "wte", "word_embeddings", "tok_embeddings", "embeddings")


def accessor_requires_arguments(function):
    """True when `function(self)` cannot be called: a required parameter follows self."""
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return False
    return any(
        p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in parameters[1:]
    )


def _vocab_sizes(module):
    config = getattr(module, "config", None)
    sizes = set()
    for holder in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "get_text_config", lambda: None)(),
    ):
        size = getattr(holder, "vocab_size", None)
        if isinstance(size, int):
            sizes.add(size)
    return sizes


def find_embedding_module(module):
    """The token embedding of `module` without going through its accessor."""
    for name in _EMBEDDING_ATTRIBUTES:
        child = getattr(module, name, None)
        if isinstance(child, torch.nn.Embedding):
            return child
    from transformers import PreTrainedModel

    for child in module.children():
        if isinstance(child, PreTrainedModel) and not accessor_requires_arguments(
            type(child).get_input_embeddings
        ):
            try:
                embedding = child.get_input_embeddings()
            except Exception:
                continue
            if isinstance(embedding, torch.nn.Module):
                return embedding
    sizes = _vocab_sizes(module)
    for child in module.modules():
        if isinstance(child, torch.nn.Embedding) and child.num_embeddings in sizes:
            return child
    return None


def _repair_accessor(cls):
    original = cls.__dict__.get("get_input_embeddings")
    if original is None or "_unsloth_original_get_input_embeddings" in cls.__dict__:
        return False

    @functools.wraps(original)
    def get_input_embeddings(self, *args, **kwargs):
        if args or kwargs:
            return original(self, *args, **kwargs)
        embedding = find_embedding_module(self)
        if embedding is None:
            raise NotImplementedError(
                f"Unsloth: `{type(self).__name__}.get_input_embeddings` embeds tokens instead of returning "
                "the embedding module, and no token embedding could be located to stand in for it."
            )
        return embedding

    cls.get_input_embeddings = get_input_embeddings
    cls._unsloth_original_get_input_embeddings = original
    return True


def _is_remote_code(cls):
    return "transformers_modules" in (getattr(cls, "__module__", "") or "")


_OUTPUT_HEAD_ATTRIBUTES = ("lm_head", "output", "embed_out", "output_layer")


def find_output_head(module):
    for name in _OUTPUT_HEAD_ATTRIBUTES:
        child = getattr(module, name, None)
        if isinstance(child, torch.nn.Module) and hasattr(child, "weight"):
            return child
    return None


def _repair_output_accessor(cls):
    """`get_output_embeddings` that returns None while the class owns an `lm_head` (Step-3.7 delegates to a headless inner model)."""
    original = cls.__dict__.get("get_output_embeddings")
    if original is None or "_unsloth_original_get_output_embeddings" in cls.__dict__:
        return False

    @functools.wraps(original)
    def get_output_embeddings(self, *args, **kwargs):
        try:
            head = original(self, *args, **kwargs)
        except (AttributeError, TypeError, NotImplementedError):
            head = None
        if head is None and not args and not kwargs:
            head = find_output_head(self)
        return head

    cls.get_output_embeddings = get_output_embeddings
    cls._unsloth_original_get_output_embeddings = original
    return True


def _output_accessor_is_broken(model):
    if find_output_head(model) is None:
        return False
    try:
        return model.get_output_embeddings() is None
    except (AttributeError, TypeError, NotImplementedError):
        return True


def _fill_missing_loss(cls):
    """Wrap `cls.forward` so a call with labels always yields a loss.

    The first labelled call probes the original; if it gives no loss, labels are withheld
    from then on and the causal LM loss is computed from its logits.
    """
    original = cls.__dict__.get("forward")
    # Own dict only: a subclass of an already repaired class has its own unwrapped forward.
    if original is None or "_unsloth_original_forward" in cls.__dict__:
        return False
    try:
        parameters = inspect.signature(original).parameters
    except (TypeError, ValueError):
        return False
    if "labels" not in parameters:
        return False

    # Per instance: two configs of one remote class can differ in whether the loss works.
    state_key = "_unsloth_returns_loss"

    def _loss_from(output, labels, kwargs):
        logits = (
            output["logits"]
            if isinstance(output, dict)
            else output[0]
            if isinstance(output, tuple)
            else None
        )
        if logits is None:
            return None
        from transformers.loss.loss_utils import ForCausalLMLoss

        return ForCausalLMLoss(
            logits = logits,
            labels = labels,
            vocab_size = logits.shape[-1],
            num_items_in_batch = kwargs.get("num_items_in_batch", None),
        )

    def _has_own_loss(output):
        if isinstance(output, dict):
            return output.get("loss", None) is not None
        # return_dict = False: (loss, logits, ...) when labels were given; logits are never 0-d.
        if isinstance(output, (tuple, list)) and output:
            first = output[0]
            return isinstance(first, torch.Tensor) and first.ndim == 0 and first.is_floating_point()
        return False

    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        signature = None
    self_placeholder = object()

    def _bind(args, kwargs):
        """Move a positional `labels` into kwargs; anything unbindable is left as it came."""
        if not args or signature is None:
            return args, kwargs
        try:
            bound = signature.bind_partial(self_placeholder, *args, **kwargs)
        except TypeError:
            return args, kwargs
        flat = {}
        for name, parameter in signature.parameters.items():
            if name not in bound.arguments:
                continue
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                flat.update(bound.arguments[name])
            elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                return args, kwargs
            else:
                flat[name] = bound.arguments[name]
        flat.pop(next(iter(signature.parameters)), None)
        return (), flat

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        args, kwargs = _bind(args, kwargs)
        labels = kwargs.get("labels", None)
        returns_loss = self.__dict__.get(state_key, None)
        if labels is None or returns_loss is True:
            return original(self, *args, **kwargs)
        if returns_loss is None:
            try:
                output = original(self, *args, **kwargs)
                if _has_own_loss(output):
                    self.__dict__[state_key] = True
                    return output
            except (AttributeError, TypeError, KeyError):
                pass
            self.__dict__[state_key] = False
            print(
                f"Unsloth: `{cls.__name__}.forward` accepts `labels` but returns no loss, "
                "so the causal LM loss is computed from its logits."
            )
        kwargs = dict(kwargs)
        kwargs.pop("labels")
        output = original(self, *args, **kwargs)
        loss = _loss_from(output, labels, kwargs)
        if loss is None:
            raise RuntimeError(
                f"Unsloth: `{cls.__name__}.forward` returned neither a loss nor logits, so no loss can be trained on."
            )
        if isinstance(output, dict):
            # `loss` must come first: positional readers take output[0] as the loss.
            fields = {k: v for k, v in output.items() if k != "loss"}
            try:
                return type(output)(loss = loss, **fields)
            except Exception:
                rebuilt = type(output)()
                rebuilt["loss"] = loss
                for k, v in fields.items():
                    rebuilt[k] = v
                return rebuilt
        return (loss,) + tuple(output)

    cls.forward = forward
    cls._unsloth_original_forward = original
    return True


def _rebind_accelerate_hook(model):
    """Point an accelerate hook attached during loading at the repaired forward.

    `device_map` loading keeps the bound original as `model._old_forward`, bypassing class repairs.
    """
    if getattr(type(model), "_unsloth_original_forward", None) is None:
        return
    if "_old_forward" not in vars(model):
        return
    import types

    model._old_forward = types.MethodType(type(model).forward, model)


def apply_remote_code_shims(model):
    from transformers import PreTrainedModel

    repaired = []
    # Deepest first, so a parent's accessor can delegate to an already repaired child.
    modules = [m for m in model.modules() if isinstance(m, PreTrainedModel)]
    seen = set()
    for module in reversed(modules):
        cls = type(module)
        if cls in seen:
            continue
        seen.add(cls)
        if accessor_requires_arguments(cls.get_input_embeddings) and _repair_accessor(cls):
            repaired.append(f"{cls.__name__}.get_input_embeddings")
    cls = type(model)
    if _is_remote_code(cls) and _output_accessor_is_broken(model) and _repair_output_accessor(cls):
        repaired.append(f"{cls.__name__}.get_output_embeddings")
    if _is_remote_code(cls) and _fill_missing_loss(cls):
        repaired.append(f"{cls.__name__}.forward")
    _rebind_accelerate_hook(model)
    if repaired:
        print("Unsloth: Repaired remote modeling code so it trains: " + ", ".join(repaired) + ".")
    return repaired
