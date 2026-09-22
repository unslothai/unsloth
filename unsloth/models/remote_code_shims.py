# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Repairs for remote modeling code that breaks the transformers contract.

Ports written against vLLM keep two habits that stop training in transformers:

* `get_input_embeddings(self, input_ids)` returns the embedded tokens instead of
  the embedding module. Everything on the training side (PEFT, gradient
  checkpointing, `enable_input_require_grads`, embedding offload, resizing) calls
  it with no arguments and expects an `nn.Module`.
* `forward(..., labels=...)` accepts labels but never returns a loss, or trips
  over its own loss code (`self.config.vocab_size` on a composite config).

Both are fixed on the checkpoint's dynamically created classes, never on
transformers' own, and the original behaviour stays reachable: the accessor
still embeds when called with arguments, and a forward that does return a loss
is left alone after one probe. Step-3.7-Flash is the case that surfaced both.
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
    for holder in (config, getattr(config, "text_config", None), getattr(config, "get_text_config", lambda: None)()):
        size = getattr(holder, "vocab_size", None)
        if isinstance(size, int):
            sizes.add(size)
    return sizes


def find_embedding_module(module):
    """The token embedding of `module` without going through its accessor.

    Order: a conventionally named `nn.Embedding` attribute, then a child model
    whose own accessor works (already repaired if it needed to be), then the
    first `nn.Embedding` sized like the vocabulary.
    """
    for name in _EMBEDDING_ATTRIBUTES:
        child = getattr(module, name, None)
        if isinstance(child, torch.nn.Embedding):
            return child
    from transformers import PreTrainedModel
    for child in module.children():
        if isinstance(child, PreTrainedModel) and not accessor_requires_arguments(type(child).get_input_embeddings):
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
    if original is None or getattr(cls, "_unsloth_original_get_input_embeddings", None) is not None:
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


def _fill_missing_loss(cls):
    """Wrap `cls.forward` so a call with labels always yields a loss.

    The first call with labels probes the original: if it returns a loss the
    wrapper steps aside for good. If it raises or returns no loss, from then on
    the labels are withheld from the original and the causal LM loss is
    computed from its logits, the same formula transformers uses.
    """
    original = cls.__dict__.get("forward")
    if original is None or getattr(cls, "_unsloth_original_forward", None) is not None:
        return False
    try:
        parameters = inspect.signature(original).parameters
    except (TypeError, ValueError):
        return False
    if "labels" not in parameters:
        return False

    state = {"returns_loss": None}

    def _loss_from(output, labels, kwargs):
        logits = output["logits"] if isinstance(output, dict) else output[0] if isinstance(output, tuple) else None
        if logits is None:
            return None
        from transformers.loss.loss_utils import ForCausalLMLoss
        return ForCausalLMLoss(
            logits = logits,
            labels = labels,
            vocab_size = logits.shape[-1],
            num_items_in_batch = kwargs.get("num_items_in_batch", None),
        )

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        labels = kwargs.get("labels", None)
        if labels is None or state["returns_loss"] is True:
            return original(self, *args, **kwargs)
        if state["returns_loss"] is None:
            try:
                output = original(self, *args, **kwargs)
                if isinstance(output, dict) and output.get("loss", None) is not None:
                    state["returns_loss"] = True
                    return output
            except (AttributeError, TypeError, KeyError):
                pass
            state["returns_loss"] = False
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
            output["loss"] = loss
            return output
        return (loss,) + tuple(output)

    cls.forward = forward
    cls._unsloth_original_forward = original
    return True


def apply_remote_code_shims(model):
    """Repair the checkpoint-defined classes in `model`. Returns the repaired class names."""
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
    if _is_remote_code(cls) and _fill_missing_loss(cls):
        repaired.append(f"{cls.__name__}.forward")
    if repaired:
        print("Unsloth: Repaired remote modeling code so it trains: " + ", ".join(repaired) + ".")
    return repaired
