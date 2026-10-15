# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Fine-tune a model's native MTP head so its speculative drafts match the fine-tuned model.

transformers drops `mtp.*` on load, and the merged export copies the stock head through
unchanged, so after fine-tuning the head drafts for the old model. `get_mtp_head` loads the
head from the checkpoint and attaches it; `UnslothMTPTrainer` trains it against the frozen
model; every merged / GGUF save then writes the trained head instead of the stock one.
"""

# The trainer names resolve lazily through __getattr__ (they need a recent unsloth_zoo).
__all__ = ["get_mtp_head", "attach_saved_mtp_head"]

import functools
import inspect
import os
import time


def _zoo_speculative():
    try:
        import unsloth_zoo.speculative as speculative
        from unsloth_zoo.speculative import mtp  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Unsloth: MTP head training needs a newer unsloth_zoo. "
            "Update via `pip install --upgrade --no-cache-dir unsloth_zoo`."
        ) from error
    return speculative


def _checkpoint_source(model):
    """The repo id / folder the weights came from, looked up through PEFT wrappers."""
    holders, seen, stack = [], set(), [model]
    while stack:
        holder = stack.pop()
        if holder is None or id(holder) in seen:
            continue
        seen.add(id(holder))
        holders.append(holder)
        for attribute in ("base_model", "model"):
            child = holder.__dict__.get("_modules", {}).get(attribute) or holder.__dict__.get(
                attribute
            )
            if child is not None:
                stack.append(child)
    # ModelScope loads keep the repo id in the config and the downloaded weights on the inner
    # model, under any PEFT wrapper: the snapshot wins over every holder's config.
    for holder in holders:
        snapshot = holder.__dict__.get("_unsloth_modelscope_snapshot")
        if isinstance(snapshot, str) and os.path.isdir(snapshot):
            return snapshot
    for holder in holders:
        name = getattr(getattr(holder, "config", None), "_name_or_path", None)
        if name:
            return name
    return None


def _loaded_revision(model, source, revision):
    """The commit the weights were loaded from, so a Hub lookup reads that head, not the default branch's."""
    if revision is not None or not isinstance(source, str) or os.path.isdir(source):
        return revision
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def get_mtp_head(
    model,
    source = None,
    revision = None,
    random_init = False,
    dtype = None,
):
    """Load the model's native MTP head, attach it to `model`, and return it.

    source: repo id or folder holding the original `mtp.*` tensors. Defaults to the
        checkpoint the model was loaded from; pass the original repo when that checkpoint
        was re-saved without its head.
    random_init: start from random weights instead (for checkpoints without a head).
    dtype: head parameter dtype, float32 by default (the head is small; the target stays as loaded).
    Without `source` / `random_init`, a head already attached (e.g. re-attached from beside a
    reloaded adapter) is returned as is.
    """
    import torch

    if source is None and not random_init:
        from ..save import _attached_mtp_head
        attached = _attached_mtp_head(model)
        if attached is not None:
            return attached if dtype is None else attached.to(dtype = dtype)
    speculative = _zoo_speculative()
    from unsloth_zoo.speculative.mtp import build_mtp_head, load_mtp_state_dict

    parts = speculative.resolve_text_parts(model)
    if source is None:
        source = _checkpoint_source(model)
        revision = _loaded_revision(model, source, revision)
    family = speculative.mtp_family_name(parts["config"])
    if family is None:
        raise NotImplementedError(
            f"Unsloth: `{getattr(parts['config'], 'model_type', None)}` has no native MTP head Unsloth can train."
        )
    if random_init:
        state_dict, info = (
            None,
            {"prefix": "mtp." if family == "qwen3_5" else "", "num_layers": 1, "source": source},
        )
    elif isinstance(source, str) and os.path.isfile(os.path.join(source, "mtp_head.safetensors")):
        # A head trained earlier and saved beside a LoRA adapter.
        from unsloth_zoo.speculative.stitch import load_saved_mtp_head
        state_dict, info = load_saved_mtp_head(source)
    else:
        if source is None:
            raise ValueError(
                "Unsloth: pass `source=` (the repo id or folder that holds the `mtp.*` tensors)."
            )
        state_dict, info = load_mtp_state_dict(source, revision = revision, config = parts["config"])
        # `files` are local cache paths; never write them into a folder that may be pushed.
        info = {k: v for k, v in info.items() if k != "files"}
        info.update(source = source, revision = revision)
    device = parts["embed_tokens"].weight.device
    extra = {}
    if family != "qwen3_5":
        # Head-serialized embedding / lm_head copies equal to the target's are not held twice.
        extra = dict(
            target_embed = parts["embed_tokens"].weight,
            target_lm_head = parts["lm_head"].weight,
            ckpt_prefix = info.get("ckpt_prefix"),
        )
    head = build_mtp_head(
        parts["config"],
        state_dict = state_dict,
        dtype = dtype or torch.float32,
        device = device,
        **extra,
    )
    status = getattr(getattr(head, "family", None), "llama_cpp", "drafts")
    if status != "drafts":
        print(
            f"Unsloth: llama.cpp support for `{family}` MTP drafting is `{status}`; the trained head "
            "is still written into merged / GGUF exports."
        )
    head._unsloth_mtp_info = info
    # Not a submodule: keeps the head out of model.state_dict(), PEFT and the target's optimizer.
    object.__setattr__(model, "_unsloth_mtp_head", head)
    _wrap_save_pretrained(model)
    return head


def _takes_push_to_hub(fn):
    # Unsloth's loaders wrap save_pretrained as (self, *args, **kwargs); read the wrapped one.
    if getattr(fn, "__name__", "") == "unsloth_model_save_pretrained":
        fn = getattr(getattr(fn, "__self__", None), "original_model_save_pretrained", fn)
    try:
        return "push_to_hub" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _wrap_save_pretrained(model):
    """Plain `save_pretrained` keeps the trained head too: beside a LoRA adapter, else in the checkpoint."""
    original = model.save_pretrained
    if getattr(original, "_unsloth_saves_mtp_head", False):
        return

    @functools.wraps(original)
    def save_pretrained(save_directory, *args, **kwargs):
        from ..save import (
            _INTERNAL_MTP_SAVE,
            _refuse_mtp_variant,
            _save_mtp_head_beside_adapter,
            _stitch_mtp_head,
        )

        internal = _INTERNAL_MTP_SAVE.get()
        if (
            not internal
            and kwargs.get("safe_serialization", True) is False
            and not hasattr(model, "peft_config")
        ):
            # The head is stitched into safetensors checkpoints only; refuse before writing.
            raise ValueError(
                "Unsloth: a model with a trained MTP head saves as safetensors; drop `safe_serialization = False`."
            )
        # PEFT's save_pretrained has no push_to_hub (its push_to_hub saves into a folder, then uploads it).
        push = bool(kwargs.get("push_to_hub")) and _takes_push_to_hub(original)
        if not internal:
            _refuse_mtp_variant(model, kwargs.get("variant"))
            if push and kwargs.get("create_pr"):
                # transformers opens the PR and keeps its ref; a second upload would open another.
                raise ValueError(
                    "Unsloth: `create_pr = True` cannot carry the trained MTP head. Save locally, then "
                    "upload the folder with `HfApi().upload_folder(..., create_pr = True)`."
                )
        started = time.time()
        result = original(save_directory, *args, **kwargs)
        if internal or not os.path.isdir(save_directory):
            return result
        adapter_config = os.path.join(save_directory, "adapter_config.json")
        # Written by this save, not left over from an adapter saved here before.
        if os.path.isfile(adapter_config) and os.path.getmtime(adapter_config) >= started - 1:
            files = (
                ["mtp_head.safetensors"]
                if _save_mtp_head_beside_adapter(model, save_directory)
                else []
            )
        else:
            report = _stitch_mtp_head(model, save_directory)
            files = list(report["files_touched"]) if report else []
        if push and files:
            # transformers uploaded the folder before the head existed; send the head files after it.
            from huggingface_hub import HfApi
            HfApi(token = kwargs.get("token")).upload_folder(
                repo_id = kwargs.get("repo_id")
                or str(save_directory).rstrip(os.sep).split(os.sep)[-1],
                folder_path = save_directory,
                allow_patterns = files,
                commit_message = "Add the trained MTP head",
            )
        return result

    save_pretrained._unsloth_saves_mtp_head = True
    object.__setattr__(model, "save_pretrained", save_pretrained)


def attach_saved_mtp_head(model, directory):
    """Re-attach a head saved beside an adapter folder; None when the folder has none."""
    if not isinstance(directory, str) or not os.path.isfile(
        os.path.join(directory, "mtp_head.safetensors")
    ):
        return None
    return get_mtp_head(model, source = directory)


def __getattr__(name):
    if name in ("UnslothMTPTrainer", "MTPTrainingArguments", "MTPDataCollator"):
        _zoo_speculative()
        from unsloth_zoo.speculative import trainer
        return getattr(trainer, name)
    if name in ("UnslothDSparkTrainer", "DSparkTrainingArguments", "Qwen3DSparkDrafter"):
        _zoo_speculative()
        from unsloth_zoo.speculative import dspark
        return getattr(dspark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
