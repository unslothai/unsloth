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
    try:
        return module(hidden_states, **kwargs)
    except TypeError:
        try:
            return module(hidden_states = hidden_states, **kwargs)
        except TypeError:
            return module(hidden_states)


def unwrap_mtp_output(output):
    if hasattr(output, "logits"):
        return output.logits
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


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
        lengths = torch.tensor(seq_lengths, device = shift_labels.device, dtype = torch.int64).reshape(-1)
    if lengths.numel() == 0:
        return False

    flat = shift_labels.reshape(-1)
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


def make_mtp_shift_labels(labels, offset, packed_seq_lengths = None):
    shift_labels = torch.empty_like(labels)
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


def should_use_mtp_loss(model, use_mtp_loss = None, train_mtp = None):
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
    **kwargs,
):
    mtp_modules = get_mtp_modules(model)
    if not mtp_modules or not should_use_mtp_loss(model, use_mtp_loss, train_mtp):
        return None

    if mtp_loss_weight is None:
        mtp_loss_weight = getattr(getattr(model, "config", None), "mtp_loss_weight", 1.0)

    mtp_kwargs = filter_mtp_kwargs(kwargs)
    losses = []
    for mtp_idx, mtp_module in enumerate(mtp_modules):
        offset = mtp_idx + 2
        if offset >= labels.shape[-1]:
            continue
        try:
            mtp_output = call_mtp_module(mtp_module, hidden_states, **mtp_kwargs)
        except Exception as error:
            raise RuntimeError(
                "Unsloth: Failed to run an MTP head while computing the Qwen3.5 "
                "MTP fine-tuning loss. Pass `use_mtp_loss = False` to disable "
                "the auxiliary MTP objective for this run."
            ) from error

        mtp_logits_or_hidden = unwrap_mtp_output(mtp_output)
        if mtp_logits_or_hidden is None:
            continue
        vocab_size = getattr(model, "vocab_size", None)
        if vocab_size is None:
            vocab_size = getattr(getattr(model, "config", None), "vocab_size", -1)
        if mtp_logits_or_hidden.shape[-1] == vocab_size:
            mtp_logits = mtp_logits_or_hidden
        else:
            mtp_logits = model.lm_head(mtp_logits_or_hidden.to(model.lm_head.weight.dtype))

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

    if len(losses) == 0:
        return None
    mtp_loss = torch.stack(losses).mean()
    return mtp_loss * mtp_loss_weight


def patch_mtp_loss(model, loss_fn):
    if getattr(model, "_unsloth_mtp_loss_patched", False):
        return model
    if not get_mtp_modules(model):
        return model

    original_forward = model.forward

    def _forward_with_mtp_loss(self, *args, **kwargs):
        labels = kwargs.get("labels", None)
        if labels is None or not should_use_mtp_loss(
            self,
            kwargs.get("use_mtp_loss", None),
            kwargs.get("train_mtp", None),
        ):
            return original_forward(*args, **kwargs)

        original_output_hidden_states = kwargs.get("output_hidden_states", None)
        kwargs["output_hidden_states"] = True
        outputs = original_forward(*args, **kwargs)
        base_loss = getattr(outputs, "loss", None)
        hidden_states = getattr(outputs, "hidden_states", None)
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
        )
        if mtp_loss is None:
            return outputs

        outputs.loss = base_loss + mtp_loss.to(base_loss.device)
        if (
            original_output_hidden_states is not True
            and not getattr(getattr(self, "config", None), "output_hidden_states", False)
        ):
            outputs.hidden_states = None
        return outputs

    model._unsloth_old_forward_before_mtp_loss = original_forward
    model.forward = types.MethodType(_forward_with_mtp_loss, model)
    model._unsloth_mtp_loss_patched = True
    return model
