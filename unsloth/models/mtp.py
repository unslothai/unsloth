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

import inspect
import types
import torch


def get_mtp_modules(model):
    for name in ("mtp", "mtp_layers", "mtp_heads", "mtp_modules"):
        mtp = getattr(model, name, None)
        if mtp is None:
            continue
        if isinstance(mtp, (list, tuple, torch.nn.ModuleList)):
            return list(mtp)
        return [mtp]
    return []


def call_mtp_module(module, hidden_states, **kwargs):
    forward_fn = getattr(module, "forward", module)
    try:
        signature = inspect.signature(forward_fn)
    except (TypeError, ValueError):
        return module(hidden_states, **kwargs)

    parameters = signature.parameters
    has_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    filtered_kwargs = (
        kwargs if has_kwargs else {key: value for key, value in kwargs.items() if key in parameters}
    )
    has_positional = any(
        parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for parameter in parameters.values()
    )

    if has_positional:
        return module(hidden_states, **filtered_kwargs)
    if "hidden_states" in parameters:
        return module(hidden_states = hidden_states, **filtered_kwargs)
    return module(hidden_states, **filtered_kwargs)


def unwrap_mtp_output(output):
    if hasattr(output, "logits"):
        return output.logits
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        if "last_hidden_state" in output:
            return output["last_hidden_state"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def iter_mtp_outputs(output):
    if isinstance(output, (tuple, list)):
        outputs = []
        for item in output:
            item = unwrap_mtp_output(item)
            if item is not None and hasattr(item, "shape"):
                outputs.append(item)
        if len(outputs) != 0:
            return outputs
        output = output[0] if len(output) != 0 else None

    output = unwrap_mtp_output(output)
    if output is None:
        return []
    return [output]


def mask_mtp_packed_sequence_boundaries(
    shift_labels,
    seq_lengths,
    offset,
    ignore_index = -100,
):
    if seq_lengths is None:
        return False
    if isinstance(seq_lengths, torch.Tensor):
        lengths = seq_lengths.to(device = shift_labels.device, dtype = torch.int64).reshape(-1)
    else:
        lengths = torch.tensor(seq_lengths, device = shift_labels.device, dtype = torch.int64).reshape(
            -1
        )
    if lengths.numel() == 0:
        return False

    flat = shift_labels.view(-1)
    total_tokens = flat.shape[0]
    boundary_ends = torch.cumsum(lengths, dim = 0)
    changed = False
    for backoff in range(1, offset + 1):
        positions = boundary_ends - backoff
        valid = (positions >= 0) & (positions < total_tokens)
        positions = positions[valid]
        if positions.numel() != 0:
            flat[positions] = ignore_index
            changed = True
    return changed


def make_mtp_shift_labels(
    labels,
    offset,
    packed_seq_lengths = None,
):
    shift_labels = torch.empty(labels.shape, dtype = labels.dtype, device = labels.device)
    if offset < labels.shape[-1]:
        shift_labels[..., :-offset] = labels[..., offset:]
        shift_labels[..., -offset:] = -100
    else:
        shift_labels.fill_(-100)
    mask_mtp_packed_sequence_boundaries(
        shift_labels,
        packed_seq_lengths,
        offset,
    )
    return shift_labels


def should_use_mtp_loss(
    model,
    use_mtp_loss = None,
    train_mtp = None,
):
    if use_mtp_loss is None:
        use_mtp_loss = train_mtp
    if use_mtp_loss is None:
        use_mtp_loss = getattr(getattr(model, "config", None), "use_mtp_loss", None)
    if use_mtp_loss is None:
        model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
        use_mtp_loss = model_type.startswith(("qwen3_5", "qwen3.5"))
    return bool(use_mtp_loss)


def filter_mtp_kwargs(kwargs):
    return {
        key: value
        for key, value in kwargs.items()
        if key
        not in {
            "use_mtp_loss",
            "train_mtp",
            "mtp_loss_weight",
            "packed_seq_lengths",
        }
    }


def compute_mtp_loss(
    model,
    hidden_states,
    labels,
    *,
    loss_fn,
    n_items = None,
    logit_softcapping = 0,
    logit_scaling = 0,
    mtp_loss_weight = None,
    use_mtp_loss = None,
    train_mtp = None,
    packed_seq_lengths = None,
    input_ids = None,
    position_ids = None,
    attention_mask = None,
    inputs_embeds = None,
    cache_position = None,
    position_embeddings = None,
    embed_tokens = None,
    embed_fn = None,
    **kwargs,
):
    mtp_modules = get_mtp_modules(model)
    if not mtp_modules or not should_use_mtp_loss(model, use_mtp_loss, train_mtp):
        return None

    if mtp_loss_weight is None:
        mtp_loss_weight = getattr(getattr(model, "config", None), "mtp_loss_weight", 1.0)

    mtp_kwargs = filter_mtp_kwargs(kwargs)
    if input_ids is not None:
        mtp_kwargs.setdefault("input_ids", input_ids)
    if position_ids is not None:
        mtp_kwargs.setdefault("position_ids", position_ids)
    if attention_mask is not None:
        mtp_kwargs.setdefault("attention_mask", attention_mask)
    if inputs_embeds is not None:
        mtp_kwargs.setdefault("inputs_embeds", inputs_embeds)
    if cache_position is not None:
        mtp_kwargs.setdefault("cache_position", cache_position)
    if position_embeddings is not None:
        mtp_kwargs.setdefault("position_embeddings", position_embeddings)
    if embed_tokens is None:
        embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
    if embed_fn is None:
        embed_fn = embed_tokens
    if embed_tokens is not None:
        mtp_kwargs.setdefault("embed_tokens", embed_tokens)
    if embed_fn is not None:
        mtp_kwargs.setdefault("embed_fn", embed_fn)

    losses = []
    offset = 2
    for mtp_module in mtp_modules:
        try:
            mtp_output = call_mtp_module(mtp_module, hidden_states, **mtp_kwargs)
        except Exception as error:
            raise RuntimeError(
                "Unsloth: Failed to run an MTP head while computing the Qwen3.5 "
                "MTP fine-tuning loss. Pass `use_mtp_loss = False` to disable "
                "the auxiliary MTP objective for this run."
            ) from error

        for mtp_logits_or_hidden in iter_mtp_outputs(mtp_output):
            if offset >= labels.shape[-1]:
                continue
            vocab_size = getattr(model, "vocab_size", None)
            if vocab_size is None:
                vocab_size = getattr(getattr(model, "config", None), "vocab_size", -1)
            if mtp_logits_or_hidden.shape[-1] == vocab_size:
                mtp_logits = mtp_logits_or_hidden
            else:
                lm_head = getattr(model, "lm_head", None)
                if lm_head is None:
                    raise AttributeError(
                        "Unsloth: MTP hidden states require a model.lm_head module."
                    )
                lm_head_weight = getattr(lm_head, "weight", None)
                if lm_head_weight is not None:
                    mtp_logits_or_hidden = mtp_logits_or_hidden.to(
                        device = lm_head_weight.device,
                        dtype = lm_head_weight.dtype,
                    )
                mtp_logits = lm_head(mtp_logits_or_hidden)

            mtp_shift_labels = make_mtp_shift_labels(
                labels,
                offset,
                packed_seq_lengths = packed_seq_lengths,
            )
            losses.append(
                loss_fn(
                    logits = mtp_logits,
                    labels = mtp_shift_labels,
                    logit_softcapping = logit_softcapping,
                    logit_scaling = logit_scaling,
                    n_items = n_items,
                )
            )
            offset += 1

    if len(losses) == 0:
        return None
    mtp_loss = torch.stack(losses).mean()
    return mtp_loss * mtp_loss_weight


def bind_forward_arguments(forward_fn, args, kwargs):
    try:
        signature = inspect.signature(forward_fn)
        bound = signature.bind_partial(*args, **kwargs)
        return bound.arguments
    except (TypeError, ValueError):
        return {}


def get_forward_argument(
    bound_arguments,
    kwargs,
    name,
    default = None,
):
    if name in kwargs:
        return kwargs[name]
    return bound_arguments.get(name, default)


def get_tuple_hidden_states(outputs):
    for item in reversed(outputs):
        if isinstance(item, (tuple, list)) and len(item) != 0 and hasattr(item[-1], "shape"):
            return item
    return None


def get_output_loss_and_hidden_states(outputs):
    if isinstance(outputs, tuple):
        if len(outputs) == 0:
            return None, None
        return outputs[0], get_tuple_hidden_states(outputs)
    return getattr(outputs, "loss", None), getattr(outputs, "hidden_states", None)


def set_output_loss_and_hidden_states(outputs, loss, hidden_states):
    if isinstance(outputs, tuple):
        return (loss,) + outputs[1:]
    outputs.loss = loss
    outputs.hidden_states = hidden_states
    return outputs


def patch_mtp_loss(model, loss_fn):
    if getattr(model, "_unsloth_mtp_loss_patched", False):
        return model
    if not get_mtp_modules(model):
        return model

    original_forward = model.forward
    original_signature = inspect.signature(original_forward)
    if "self" in original_signature.parameters:
        wrapper_signature = original_signature
    else:
        wrapper_signature = original_signature.replace(
            parameters = [
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                *original_signature.parameters.values(),
            ]
        )

    def _forward_with_mtp_loss(self, *args, **kwargs):
        bound_arguments = bind_forward_arguments(original_forward, args, kwargs)
        labels = get_forward_argument(bound_arguments, kwargs, "labels")
        if labels is None or not should_use_mtp_loss(
            self,
            kwargs.get("use_mtp_loss", None),
            kwargs.get("train_mtp", None),
        ):
            return original_forward(*args, **kwargs)

        original_return_dict = get_forward_argument(bound_arguments, kwargs, "return_dict")
        original_output_hidden_states = get_forward_argument(
            bound_arguments,
            kwargs,
            "output_hidden_states",
        )
        kwargs["output_hidden_states"] = True
        if original_return_dict is False:
            kwargs["return_dict"] = True
        outputs = original_forward(*args, **kwargs)
        base_loss, hidden_states = get_output_loss_and_hidden_states(outputs)
        if base_loss is None or hidden_states is None:
            return outputs

        last_hidden_state = hidden_states[-1]
        mtp_loss = compute_mtp_loss(
            self,
            last_hidden_state,
            labels.to(last_hidden_state.device),
            loss_fn = loss_fn,
            n_items = kwargs.get("num_items_in_batch", kwargs.get("n_items", None)),
            mtp_loss_weight = kwargs.get("mtp_loss_weight", None),
            use_mtp_loss = kwargs.get("use_mtp_loss", None),
            train_mtp = kwargs.get("train_mtp", None),
            packed_seq_lengths = kwargs.get("packed_seq_lengths", None),
            input_ids = get_forward_argument(bound_arguments, kwargs, "input_ids"),
            position_ids = get_forward_argument(bound_arguments, kwargs, "position_ids"),
            attention_mask = get_forward_argument(bound_arguments, kwargs, "attention_mask"),
            inputs_embeds = get_forward_argument(bound_arguments, kwargs, "inputs_embeds"),
            cache_position = get_forward_argument(bound_arguments, kwargs, "cache_position"),
        )
        if mtp_loss is None:
            return outputs

        output_loss = base_loss + mtp_loss.to(base_loss.device)
        if original_output_hidden_states is not True and not getattr(
            getattr(self, "config", None), "output_hidden_states", False
        ):
            hidden_states = None
        outputs = set_output_loss_and_hidden_states(outputs, output_loss, hidden_states)
        if original_return_dict is False and hasattr(outputs, "to_tuple"):
            return outputs.to_tuple()
        return outputs

    model._unsloth_old_forward_before_mtp_loss = original_forward
    _forward_with_mtp_loss.__signature__ = wrapper_signature
    _forward_with_mtp_loss.__wrapped__ = original_forward
    model.forward = types.MethodType(_forward_with_mtp_loss, model)
    model._unsloth_mtp_loss_patched = True
    return model
