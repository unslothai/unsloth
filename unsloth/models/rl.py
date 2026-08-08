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

__all__ = [
    "PatchFastRL",
    "vLLMSamplingParams",
]

import torch
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import importlib
import inspect
import os
import re
import sys
from contextlib import contextmanager
from unsloth_zoo.compiler import create_new_function
from unsloth_zoo.log import logger
from unsloth_zoo.logging_utils import PatchRLStatistics
from unsloth_zoo.rl_replacements import RL_REPLACEMENTS
from ..device_type import DEVICE_TYPE
from .rl_replacements import (
    RL_EXTRA_ARGS,
    RL_FUNCTIONS,
    RL_PRE_ITEMS,
    RL_CONFIG_CHANGES,
    RL_METRICS_CHANGES,
    RL_ADDITIONAL_FUNCTIONS,
)

torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": False,  # Disable Triton mm kernels
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}

# vLLM compatibility shim (TRL expects GuidedDecodingParams even if vLLM doesn't provide it)
try:
    import vllm.sampling_params as _unsloth_vllm_sp
    if not hasattr(_unsloth_vllm_sp, "GuidedDecodingParams"):

        class GuidedDecodingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        _unsloth_vllm_sp.GuidedDecodingParams = GuidedDecodingParams
except Exception:
    pass

from trl import __version__ as trl_version_raw
from importlib.metadata import version as importlib_version
from unsloth_zoo.utils import Version

try:
    trl_version = Version(trl_version_raw)
except Exception:
    try:
        trl_version = Version(importlib_version("trl"))
    except Exception:
        trl_version = Version("0.0.0")

# Get PyTorch version for feature detection
try:
    torch_version = Version(torch.__version__.split("+")[0].split("a")[0].split("b")[0])
except Exception:
    torch_version = Version("0.0.0")

# Get transformers version for feature detection
try:
    from transformers import __version__ as _transformers_version_raw
    transformers_version = Version(_transformers_version_raw)
except Exception:
    transformers_version = Version("0.0.0")


def vLLMSamplingParams(**kwargs):
    from vllm import SamplingParams

    sampling_params = SamplingParams(**kwargs)
    sampling_params._set_kwargs = kwargs
    return sampling_params


def _maybe_prepare_vllm_for_resume(trainer):
    if not torch.cuda.is_available():
        return

    llm = getattr(trainer, "llm", None)
    if llm is None:
        llm = getattr(getattr(trainer, "model", None), "vllm_engine", None)
    if llm is None:
        return

    model_config = getattr(
        getattr(getattr(llm, "llm_engine", None), "vllm_config", None),
        "model_config",
        None,
    )
    if not getattr(model_config, "enable_sleep_mode", False):
        return

    try:
        llm.sleep(1)
    except Exception:
        pass

    import gc

    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()


def _patch_resume_from_checkpoint_memory(trainer_class):
    original_train = getattr(trainer_class, "train", None)
    if original_train is None:
        return
    if getattr(original_train, "_unsloth_resume_guard", False):
        return

    def _unsloth_train_with_resume_guard(self, *args, **kwargs):
        resume_from_checkpoint = kwargs.get("resume_from_checkpoint", None)
        if resume_from_checkpoint is None:
            resume_from_checkpoint = kwargs.get("model_path", None)
        if resume_from_checkpoint is None and len(args) != 0:
            resume_from_checkpoint = args[0]

        if resume_from_checkpoint:
            _maybe_prepare_vllm_for_resume(self)
        return original_train(self, *args, **kwargs)

    _unsloth_train_with_resume_guard._unsloth_resume_guard = True
    trainer_class.train = _unsloth_train_with_resume_guard


def PatchRL(FastLanguageModel):
    try:
        from trl.models.utils import unwrap_model_for_generation
    except ImportError:
        try:
            from trl.models import unwrap_model_for_generation
        except ImportError:
            # Local fallback -- TRL removed or moved this symbol
            from contextlib import contextmanager as _cm

            @_cm
            def unwrap_model_for_generation(
                model,
                accelerator,
                gather_deepspeed3_params = True,
            ):
                unwrapped_model = accelerator.unwrap_model(model)
                is_gc = getattr(unwrapped_model, "is_gradient_checkpointing", False)
                if is_gc:
                    unwrapped_model.gradient_checkpointing_disable()
                if (
                    getattr(accelerator, "state", None) is not None
                    and getattr(accelerator.state, "deepspeed_plugin", None) is not None
                    and accelerator.state.deepspeed_plugin.zero_stage == 3
                ):
                    if not gather_deepspeed3_params:
                        yield accelerator.unwrap_model(model)
                    else:
                        import deepspeed
                        with deepspeed.zero.GatheredParameters(model.parameters()):
                            yield accelerator.unwrap_model(model)
                else:
                    yield unwrapped_model
                if is_gc:
                    unwrapped_model.gradient_checkpointing_enable()

    from contextlib import contextmanager

    @contextmanager
    def unsloth_unwrap_model_for_generation(model, *args, **kwargs):
        # why: snapshot before TRL's unwrap context manager, which calls
        # gradient_checkpointing_disable() before yielding; preserve the actual
        # mode value (e.g. "unsloth") rather than collapsing it to a bool, so
        # the finally restore matches the caller's configured GC mode.
        use_gradient_checkpointing = next(
            (
                v
                for v in (getattr(m, "gradient_checkpointing", False) for m in model.modules())
                if v
            ),
            False,
        )
        with unwrap_model_for_generation(model, *args, **kwargs) as unwrapped_model:
            # Put the model in inference mode.
            FastLanguageModel.for_inference(model)

            # We must use .clone for Unsloth since we force inference_mode
            # Rather we should have used no_grad
            original_generate = unwrapped_model.generate

            def generate_with_clone(*args, **kwargs):
                out = original_generate(*args, **kwargs)
                if isinstance(out, torch.Tensor):
                    return out.clone()
                return out

            unwrapped_model.generate = generate_with_clone

            try:
                yield unwrapped_model
            finally:
                # Restore generate and return
                unwrapped_model.generate = original_generate
                FastLanguageModel.for_training(
                    model,
                    use_gradient_checkpointing = use_gradient_checkpointing,
                )

    from transformers import Trainer
    from transformers.trainer_pt_utils import nested_detach

    @torch.no_grad()
    def unsloth_prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    try:
                        num_items_in_batch = self._get_num_items_in_batch(
                            [inputs], self.args.device
                        )
                    except (AttributeError, TypeError):
                        num_items_in_batch = None
                    loss, outputs = self.compute_loss(
                        model,
                        inputs,
                        return_outputs = True,
                        num_items_in_batch = num_items_in_batch,
                    )
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    tokenized_output = self.processing_class(
                        inputs["prompt"],
                        padding = True,
                        truncation = True,
                        return_tensors = "pt",
                    ).to(model.device)
                    outputs = model(**tokenized_output)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]
        os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    import trl.trainer

    trainers = dir(trl.trainer)
    trainers = [x for x in trainers if x.endswith("_trainer")]
    unwrap = "unwrap_model_for_generation"
    for trainer in trainers:
        try:
            current_trainer = getattr(trl.trainer, trainer)
        except:
            continue
        if hasattr(current_trainer, unwrap):
            try:
                setattr(current_trainer, unwrap, unsloth_unwrap_model_for_generation)
            except:
                continue
    Trainer.prediction_step = unsloth_prediction_step


grpo_selective_log_softmax = RL_REPLACEMENTS["grpo_selective_log_softmax"]
selective_log_softmax = RL_REPLACEMENTS["selective_log_softmax"]
calculate_pad_tokens_in_prompt = RL_REPLACEMENTS["calculate_pad_tokens_in_prompt"]
create_completion_attention_mask = RL_REPLACEMENTS["create_completion_attention_mask"]
left_pack_padding = RL_REPLACEMENTS["left_pack_padding"]
align_logprobs_with_mask = RL_REPLACEMENTS["align_logprobs_with_mask"]
align_completion_tool_mask = RL_REPLACEMENTS.get("align_completion_tool_mask")
if align_completion_tool_mask is None:

    def align_completion_tool_mask(
        tool_mask: torch.Tensor, completion_mask: torch.Tensor
    ) -> torch.Tensor:
        if tool_mask is None:
            return completion_mask
        raise RuntimeError(
            "env_mask/tool_mask GRPO requires an unsloth_zoo build whose "
            "grpo_accumulated_loss handles tool_mask. Please upgrade "
            "unsloth_zoo."
        )


autotune_batch_and_chunks = RL_REPLACEMENTS["grpo_autotune_batch_and_chunks"]
sanitize_logprob = RL_REPLACEMENTS["sanitize_logprob"]

RLTrainer_replacement = '''
import os
import math
import logging
from typing import *
from dataclasses import dataclass, field
from packaging.version import Version
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
import inspect
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling as TransformersDataCollatorForLanguageModeling
from transformers.training_args import ParallelMode
from unsloth_zoo.device_type import DEVICE_TYPE, device_synchronize

# Wrap trainer with padding to right and enable training mode
import functools
from types import MethodType
try:
    from unsloth_zoo.gradient_checkpointing import reset_unsloth_gradient_checkpointing_buffers
except:
    def reset_unsloth_gradient_checkpointing_buffers(): pass
# Canonical reset lives in unsloth.models._utils so the SFT auto-packing wrapper and the plain
# Trainer loop can import the same helper; fall back to a no-op only if it can't be imported.
try:
    from unsloth.models._utils import _unsloth_reset_stray_compile_cache
except Exception:
    def _unsloth_reset_stray_compile_cache(self): pass
def prepare_for_training_mode(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        # Drop any torch.compile graph cache poisoned by a stray pre-train forward.
        try:
            _unsloth_reset_stray_compile_cache(self)
        except Exception:
            pass
        # Finish the previous W&B run if this is a subsequent train() call.
        # We do this at the START of train() (not the end) so that
        # evaluate() / log() still work after train() completes.
        # HF's WandbCallback.setup() will call wandb.init() for the new run.
        # See: https://github.com/unslothai/unsloth/issues/3954
        if getattr(self, '_unsloth_training_completed', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
                    # Reset HF's WandbCallback so it calls wandb.init() for the new run
                    for cb in self.callback_handler.callbacks:
                        if type(cb).__name__ == 'WandbCallback':
                            cb._initialized = False
                            break
            except:
                pass
        # Enable training mode
        _was_training = None
        # Restore the GC mode the model was configured with at setup; fall back to
        # the training args only when it wasn't recorded (issue #4735). Use hasattr,
        # not a None sentinel, so a deliberately-recorded None is restored verbatim.
        _model = getattr(self, 'model', None)
        if hasattr(_model, '_unsloth_gradient_checkpointing'):
            use_gc = _model._unsloth_gradient_checkpointing
        else:
            use_gc = getattr(self.args, 'gradient_checkpointing', True)
        if hasattr(self, 'model') and hasattr(self.model, "training"):
            _was_training = self.model.training
        if hasattr(self, 'model') and hasattr(self.model, "for_training"):
            self.model.for_training(use_gradient_checkpointing=use_gc)
        output = f(self, *args, **kwargs)
        # Restore previous mode when possible
        if hasattr(self, 'model') and hasattr(self.model, "for_inference"):
            if _was_training is False:
                self.model.for_inference()
            elif _was_training is True and hasattr(self.model, "for_training"):
                self.model.for_training(use_gradient_checkpointing=use_gc)
        # Reset gradient checkpointing buffers to free memory while staying ready for next run
        try:
            reset_unsloth_gradient_checkpointing_buffers()
        except:
            pass
        # Mark that training completed so the next train() call can
        # finish this W&B run before starting a new one
        self._unsloth_training_completed = True
        return output
    return wrapper
pass

torch_compile_options = {{
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}}

{grpo_selective_log_softmax_code}
{selective_log_softmax_code}
{calculate_pad_tokens_in_prompt_code}
{create_completion_attention_mask_code}
{left_pack_padding_code}
{align_logprobs_with_mask_code}
{align_completion_tool_mask_code}
{autotune_batch_and_chunks_code}
{sanitize_logprob_code}

{RL_pre}

@dataclass
class Unsloth{RLConfig_name}({RLConfig_name}):
    """
    {__RLConfig_doc__}
    """
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {{'help': 'vLLM SamplingParams'}},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {{'help': 'Chunk size to reduce memory usage. -1 is most efficient.'}},
    )
    unsloth_logit_chunk_multiplier : Optional[int] = field(
            default = None,
            metadata = {{'help': 'Multiplier for chunked logit computations.'}},
        )
    unsloth_grpo_mini_batch : Optional[int] = field(
        default = None,
        metadata = {{'help': 'Mini batch size for GRPO hidden state accumulation. Default is None unless user defines it.'}},
    )
    {max_seq_length_pre}
    def __init__({RLConfig_arguments},
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        unsloth_logit_chunk_multiplier = None,
        unsloth_grpo_mini_batch = None,
        {max_seq_length_call}
        **kwargs,
    ):
{RLConfig_extra_args}
        super().__init__({RLConfig_call_args}{RLConfig_kwargs})
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
        if unsloth_grpo_mini_batch is not None:
            if self.generation_batch_size >= unsloth_grpo_mini_batch:
                self.unsloth_grpo_mini_batch = unsloth_grpo_mini_batch
            else:
                raise ValueError(
                    f"Unsloth GRPO mini batch size needs to be less than or equal to the effective generation batch size, "
                    f"which is self.per_device_train_batch_size * gradient_accumulation_steps."
                )
        self.unsloth_logit_chunk_multiplier = unsloth_logit_chunk_multiplier
        {max_seq_length_post}
{RLConfig_post}
pass

{RLTrainer_extras}

class Unsloth{RLTrainer_name}(_Unsloth{RLTrainer_name}):
    """
    {__RLTrainer_doc__}
    """
    def __init__({RLTrainer_arguments},
        **kwargs
    ):
        if args is None: args = Unsloth{RLConfig_name}()
{RLTrainer_extra_args}
        # [TODO] Fix up DataParallel multiplying batch sizes
        # [TODO] DDP works, but DP seems to not work? [TODO]
        if getattr(args, "parallel_mode", None) == ParallelMode.NOT_DISTRIBUTED and args.n_gpu > 1:
            if getattr(args, "_n_gpu", 1) != 1:
                args._n_gpu = 1
        if "model" in locals() and hasattr(model, "for_training"):
            _use_gc = model._unsloth_gradient_checkpointing if hasattr(model, '_unsloth_gradient_checkpointing') else getattr(args, 'gradient_checkpointing', True)
            model.for_training(use_gradient_checkpointing=_use_gc)
        super().__init__({RLTrainer_call_args}{RLTrainer_kwargs})
        if "model" in locals() and hasattr(model, "for_inference"):
            model.for_inference()
{RLTrainer_post}
pass
'''


def _wrap_grpo_generate_and_score(trainer_cls):
    if not hasattr(trainer_cls, "_generate_and_score_completions"):
        return
    original = trainer_cls._generate_and_score_completions
    if getattr(original, "_unsloth_restore_training_wrapped", False):
        return

    def wrapped(self, *args, **kwargs):
        was_training = getattr(getattr(self, "model", None), "training", None)
        try:
            return original(self, *args, **kwargs)
        finally:
            if (
                was_training is False
                and hasattr(self, "model")
                and hasattr(self.model, "for_inference")
            ):
                try:
                    self.model.for_inference()
                except Exception:
                    pass

    wrapped._unsloth_restore_training_wrapped = True
    trainer_cls._generate_and_score_completions = wrapped


_PER_TOKEN = (
    "input_ids",
    "attention_mask",
    "labels",
    "completion_mask",
    "assistant_masks",
    "token_type_ids",
    "position_ids",
)


def _column_names(dataset):
    """The split's columns, from metadata AND from a row.

    Both, because either alone is wrong. A `torch.utils.data.Dataset`, a list
    or any custom map-style split carries no `column_names`, and reading that
    as "raw text, prep will tokenize it" left a pre-tokenized one uncapped on
    a path where `args.max_length` is already None. And a `with_transform`
    dataset reports its BACKING columns (`text`) while yielding `input_ids`,
    so trusting the metadata alone misses it in the other direction.

    Returns the split to actually USE alongside the names. `iter(gen) is gen`
    for a bare generator or any other single-pass iterator, so reading a row off
    it consumes that row for good and the split silently evaluates one example
    short. Those get the probed row chained back on the front instead.
    """  # noqa: D208
    names = set(getattr(dataset, "column_names", None) or ())
    source = dataset
    try:
        iterator = iter(dataset)
        # `iterator is dataset` catches a bare generator and misses an
        # `IterableDataset` whose `__iter__` hands back one stored generator:
        # that is not the dataset, but it is still the only pass there is. Two
        # `iter()` calls returning the same object covers both, and a
        # `datasets.IterableDataset` restarts, so it answers False and is
        # rewound rather than chained. Both calls happen before anything is
        # read.
        single_pass = iterator is dataset or iterator is iter(dataset)
        row = next(iterator, None)
        if single_pass and row is not None:
            import itertools
            source = itertools.chain([row], iterator)
    except Exception:
        row = None
    if isinstance(row, dict):
        names.update(row.keys())
    # The row too: it was read either way, and on a one-shot stream it is the
    # ONLY row anything may look at. Without it `_sliceable_per_token` had no
    # widths to compare and fell back to `input_ids` alone, leaving `labels`
    # and `attention_mask` at their overlength size -- rows whose supervision
    # no longer lines up with the tokens they describe.
    return tuple(names), source, (row if isinstance(row, dict) else None)


class _CappedBase:
    """A read-side cap for a split that cannot be rewritten in place.

    `map`/`filter` belong to `datasets`; a plain `torch.utils.data.Dataset` or
    a list has neither, and a `with_transform` dataset has them but rebuilds
    its rows on every read, so mapping it writes the backing table while the
    reader keeps handing back the untruncated row. Both reach the collator
    through iteration, and the map-style subclass below adds indexing.
    """

    def __init__(self, inner, cut, supervision, per_token):
        self._inner = inner
        self._cut = cut
        self._supervision = tuple(supervision)
        self._per_token = tuple(per_token)

    def _slice(self, row):
        # Per value, per row, exactly as the `map` path does it.
        # `_sliceable_per_token` judges alignment from ONE probed row, and an
        # optional column that is a list there can be None -- or a different
        # width -- further in. Slicing that raised inside the dataloader, which
        # is a failure the caller would not have had without the cap. Only
        # `input_ids` is cut unconditionally: it is the column the cap exists
        # for, and the width every other one is measured against.
        if not isinstance(row, dict):
            return row
        try:
            width = len(row["input_ids"])
        except Exception:
            width = None
        out = {}
        for key, value in row.items():
            if key not in self._per_token:
                out[key] = value
                continue
            if key == "input_ids":
                out[key] = value[self._cut]
                continue
            try:
                aligned = width is not None and len(value) == width
            except Exception:
                aligned = False
            out[key] = value[self._cut] if aligned else value
        return out

    def _keep(self, row):
        if not self._supervision or not isinstance(row, dict):
            return True
        columns = [c for c in self._supervision if c in row]
        if not columns:
            return True
        return any(
            all((x != -100) if n == "labels" else x for n, x in zip(columns, values))
            for values in zip(*[row[c] for c in columns])
        )

    def __iter__(self):
        for row in self._inner:
            cut = self._slice(row)
            if self._keep(cut):
                yield cut

    def __getattr__(self, attribute):
        # Anything else the trainer asks for (column_names, features, ...) is
        # the wrapped split's own answer. Never a dunder, and never our own
        # state: a DataLoader worker pickles the split, and `__setstate__`
        # looked up before `__init__` has run would recurse on `_inner`
        # forever.
        inner = self.__dict__.get("_inner")
        if inner is None or attribute.startswith("__"):
            raise AttributeError(attribute)
        return getattr(inner, attribute)


class _CappedRows(_CappedBase):
    """The map-style flavour: a split with a length and an index.

    Rows left with no supervised token are dropped, which changes the length,
    so the surviving indices are resolved once up front.
    """

    def __init__(self, inner, cut, supervision, per_token):
        super().__init__(inner, cut, supervision, per_token)
        # No supervision columns means `_keep` is True for every row, so the
        # index would be `range(len(inner))` -- and building it read and
        # transformed every item, which for a `with_transform` split is a whole
        # extra tokenization pass before the dataloader can even start, plus an
        # O(n) list. Stay lazy and let `__getitem__` map straight through.
        self._index = (
            None
            if not self._supervision
            else [i for i in range(len(inner)) if self._keep(self._slice(inner[i]))]
        )

    def __len__(self):
        return len(self._inner) if self._index is None else len(self._index)

    def __getitem__(self, i):
        return self._slice(self._inner[i if self._index is None else self._index[i]])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


try:
    from torch.utils.data import IterableDataset as _IterableDatasetBase
except Exception:  # torch's data stack is optional at import time
    _IterableDatasetBase = object


class _CappedStream(_CappedBase, _IterableDatasetBase):
    """The iterable-style flavour, and it has to BE one.

    Trainer and DataLoader both split map-style from iterable-style with
    `isinstance(dataset, IterableDataset)`, not by looking for `__iter__`, so
    wrapping a stream in a plain object got it a `SequentialSampler` asking for
    the `len()` a stream never had. Declared here rather than built with `type()`
    inside a function: a DataLoader worker under `spawn` pickles the split by
    module and qualified name, and a class with neither is unpicklable.
    """


def _capped_stream(inner, cut, supervision, per_token):
    return _CappedStream(inner, cut, supervision, per_token)


def _is_stream(dataset):
    """Iterable-style, by the same test the DataLoader applies."""
    try:
        from torch.utils.data import IterableDataset
        if isinstance(dataset, IterableDataset):
            return True
    except Exception:
        pass
    return not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__")


_SCAN_ROWS = 1024


def pretokenized_within_cap(dataset, cap):
    """Whether every pre-tokenized row in `dataset` already fits `cap`.

    The generated `__init__` carries its own copy of this, inlined, because that
    module is standalone and cannot import from here. This one is for callers
    that need the same answer when the generated block was never inserted -- see
    `trainer.py`'s padding-free fallback. The two must agree, and a test pins
    them to the same verdict on the shapes below.

    Unverifiable reads FALSE, never true. A single-pass stream cannot be scanned
    without consuming it, an unexhausted one is only proof about its prefix, and
    a split that raises mid-scan has told us nothing: in every one of those cases
    the caller is about to decide whether anything downstream enforces the cap,
    and guessing yes is the silently-uncapped run.
    """
    if dataset is None:
        return True
    try:
        try:
            n = len(dataset)
        except Exception:
            n = None
        rows = iter(dataset)
        if rows is iter(dataset):
            return False
        seen = 0
        for row in rows:
            if "input_ids" not in row:
                return True
            if len(row["input_ids"]) > cap:
                return False
            seen += 1
            if n is None and seen >= _SCAN_ROWS:
                return False
    except Exception:
        return False
    return True


def splits_within_cap(splits, cap):
    """`pretokenized_within_cap` over a split or a dict of them. Each one counts."""
    every = splits.values() if isinstance(splits, dict) else [splits]
    return all(pretokenized_within_cap(s, cap) for s in every)


_CAP_SIGNATURE_ATTR = "_unsloth_cap_signature"
_EVAL_CAP_MEMO_MAX = 8


def _cap_signature(dataset):
    """What this split was already capped to, or None if we did not cap it.

    Read through `__dict__` for our own wrappers: `_CappedBase.__getattr__`
    forwards anything it does not hold to the split inside, so a plain `getattr`
    on an unmarked wrapper asks the INNER split, and an inner split we happen to
    have capped earlier would answer for the outer one.
    """
    own = getattr(dataset, "__dict__", None)
    if isinstance(own, dict) and _CAP_SIGNATURE_ATTR in own:
        return own[_CAP_SIGNATURE_ATTR]
    return None


def _mutation_token(dataset):
    """What moves when this split's ROWS move, or None if nothing does.

    `datasets` splits are content-addressed by `_fingerprint`. A `with_transform`
    split has one too, but it covers the backing table and not the transform, so
    a transform closing over mutable state yields different rows under an
    unchanged fingerprint; those answer None. Anything else has no answer either.
    """
    try:
        fmt = getattr(dataset, "format", None)
        kind = fmt.get("type") if isinstance(fmt, dict) else None
        if (kind or getattr(dataset, "_format_type", None)) == "custom":
            return None
    except Exception:
        return None
    return getattr(dataset, "_fingerprint", None)


def _mark_capped(dataset, cap, drop_unsupervised):
    try:
        setattr(dataset, _CAP_SIGNATURE_ATTR, (cap, drop_unsupervised, _mutation_token(dataset)))
    except Exception:
        pass  # a split that refuses attributes just gets scanned twice
    return dataset


def _cap_still_holds(dataset, cap, drop_unsupervised):
    """Whether an earlier mark of ours still describes this split.

    Three of the four `_cap` outcomes mark the CALLER'S OWN object and hand it
    back -- no tokens, packed, already short -- and that object can be mutated
    between two `evaluate()` calls. A `set_transform` that starts yielding
    longer `input_ids` is enough, and the mark alone then skipped the rescan and
    let the new rows through uncapped. Our own wrappers hold a fixed slice and
    cannot drift, so those are trusted outright; anything else has to still
    fingerprint the way it did when marked. An unfingerprintable split answers
    None both times and is simply rescanned, which is the same conclusion the
    memo reaches for the same reason, and is not destructive: the second pass
    reads through `_column_names`, which chains its probed row back on.
    """
    signature = _cap_signature(dataset)
    if signature is None:
        return False
    if isinstance(dataset, _CappedBase):
        return tuple(signature[:2]) == (cap, drop_unsupervised)
    token = _mutation_token(dataset)
    return token is not None and tuple(signature) == (cap, drop_unsupervised, token)


def _first_row_without_consuming(dataset):
    """Row 0, or None when reading one would cost the caller that row."""
    if not _is_stream(dataset):
        try:
            return dataset[0]
        except Exception:
            return None
    # A stream whose `iter` hands back the same exhausting generator cannot
    # spare a row, and nothing here chains it back.
    probe = iter(dataset)
    if probe is dataset or probe is iter(dataset):
        return None
    return next(probe, None)


def _is_token_vector(value, width):
    """Whether `value` reads as one number per token, for a row of `width`.

    Only used to decide whether a column NOT on the allow-list rides along with
    the slice, so it is deliberately narrow: a string is as long as its
    characters, and a list of messages or of strings can match a row length by
    coincidence. A flat vector of scalars is what a per-token field actually is.
    """
    if isinstance(value, (str, bytes, dict)):
        return False
    try:
        if len(value) != width or width == 0:
            return False
    except Exception:
        return False
    # By what the entries are NOT, so a numpy or torch scalar still counts.
    for item in value:
        return not isinstance(item, (str, bytes, dict, list, tuple, set))
    return False


def _sliceable_per_token(
    dataset,
    names,
    cap,
    probed = None,
):
    """The token columns whose VALUES can be sliced alongside `input_ids`.

    Presence is not enough. An optional column stored as `token_type_ids = None`
    makes the late cap's `map` raise, and the broad catch around it hands the
    caller its uncapped split straight back; a 2-D `position_ids` slices on the
    wrong axis and comes out misaligned with the truncated `input_ids`. Either
    defeats the cap through one auxiliary column, so judge by a row rather than
    by a name, the way the construction-time truncation already does.

    A row that cannot be read without costing it leaves `input_ids` alone: the
    column the cap exists for, and the one every other is measured against.
    """
    # `input_ids` first, and the rest in a fixed order: `names` arrives as a set,
    # and the `map` path reads the width off `input_ids` as it walks this list,
    # so a run that happened to order `labels` ahead of it sliced the labels
    # without ever comparing them to anything.
    known = [c for c in _PER_TOKEN if c in names]
    # A user-defined per-token field -- `loss_mask`, `token_weights`, a custom
    # model input -- is not in the allow-list, so it kept its full length while
    # `input_ids` was cut. A custom collator (the case where padding-free is off
    # and this wrapper still runs) then gets mismatched lengths and either fails
    # or supervises the wrong tokens. Judge those the way the row check below
    # judges everything else, by alignment, but only accept a flat vector of
    # scalars: that is what a per-token field is, and it keeps `messages` or a
    # column of strings that happens to be as long as the row out of the slice.
    custom = sorted(c for c in names if c not in _PER_TOKEN)
    per_token = known + custom
    if len(known) < 2 and not custom:
        return known
    # `probed` is the row `_column_names` already read. Preferring it is what
    # lets a one-shot stream align every per-token column: reading another row
    # would cost the caller that example, so without it this fell back to
    # `input_ids` alone and left the masks and labels overlength.
    row = probed if isinstance(probed, dict) else _first_row_without_consuming(dataset)
    if not isinstance(row, dict):
        return ["input_ids"] if "input_ids" in names else []
    try:
        width = len(row.get("input_ids"))
    except Exception:
        # Nothing to measure against, so a custom column has no evidence at all
        # behind it. Fall back to the named ones only.
        return known
    kept = []
    for name in per_token:
        value = row.get(name)
        if name in custom and not _is_token_vector(value, width):
            continue
        try:
            # As long as `input_ids`, which is what makes the FIRST axis the
            # token axis: a `[seq_len, channels]` field slices correctly there,
            # and a channel-major one (`position_ids` under mrope is
            # `[3, seq_len]`) fails this check and is left alone, which is the
            # shape the nested test used to be aimed at.
            if len(value) != width:
                continue
        except Exception:
            continue
        kept.append(name)
    return kept


def _eval_packing_on(args):
    """TRL's own resolution: `args.packing` unless `eval_packing` overrides it.

    Kept identical to the generated block's `_unsloth_eval_packing`, because the
    late cap and the construction-time one have to answer this the same way.
    """
    eval_packing = getattr(args, "eval_packing", None)
    if eval_packing is None:
        return bool(getattr(args, "packing", False))
    return bool(eval_packing)


def _trl_prepares_late_evals(trainer_cls):
    """Does TRL's own `evaluate` prepare a split handed straight to it?

    False up to TRL 1.6, where `evaluate` was the base Trainer's and
    `_prepare_dataset` ran only from `__init__`. True from 1.7.0, whose
    `SFTTrainer.evaluate` calls `_prepare_dataset` with
    `packing = args.packing if args.eval_packing is None else args.eval_packing`,
    so on those versions the packer owns a late eval split and cutting its rows
    at the cap first throws away the overflow packing exists to redistribute.

    Read off the class rather than off a version number, so a TRL that moves the
    behaviour again is answered correctly. The first class in the MRO that
    defines `evaluate` is the one that runs; anything unreadable answers False,
    which is what every TRL did before 1.7.
    """
    for klass in getattr(trainer_cls, "__mro__", ()):
        method = klass.__dict__.get("evaluate")
        if method is None or getattr(method, "_unsloth_eval_cap_wrapped", False):
            continue
        try:
            return "_prepare_dataset" in inspect.getsource(method)
        except Exception:
            return False
    return False


def _wrap_sft_evaluate_cap(trainer_cls):
    """Cap a pre-tokenized split handed to `evaluate()`/`predict()` later on.

    The padding-free branch caps the init-time splits itself and then clears
    `args.max_length`, because that is what TRL's guard demands. Only the splits
    present at construction went through it, so a split supplied afterwards is
    prepared with `max_length = None`, and Zoo's prep leaves rows that already
    carry `input_ids` alone: overlength rows reach the collator with nothing
    enforcing the cap. The cap itself survives on `args.max_seq_length`, so
    apply it here to exactly the rows nothing else will truncate.

    Both entry points, not just `evaluate`: `predict(test_dataset = ...)` comes
    from the base Trainer and reaches the same collator by the same route.

    Whether anything TRL owns runs on a late split depends on the TRL. Up to 1.6
    nothing did: `_prepare_dataset` was called from `__init__` and nowhere else,
    and SFTTrainer overrode neither `evaluate` nor `predict` nor
    `get_eval_dataloader`, so a split handed over afterwards was never tokenized,
    never packed and never truncated. From 1.7.0 `SFTTrainer.evaluate` prepares a
    split passed straight to it, packing included, so on those versions the packer
    DOES own a late eval split and `eval_packing` has to be honoured here exactly
    as it is at construction. `_trl_prepares_late_evals` reads that off the class.

    This has to agree with the construction-time cap on every detail, because it
    is the same cap arriving late. It honours `truncation_mode`, refuses a packed
    split, drops rows left with no supervised token, and handles a stream.
    """

    def _supervision_columns(args, names):
        """Columns that decide whether a row still has a supervised token.

        `labels` when present, `assistant_masks` on presence alone,
        `completion_mask` only under `completion_only_loss`. That mode is the
        trainer's, resolved once from the training sample, so read the collator's
        effective value rather than guessing from this split's own columns: TRL
        does the same and the two must agree.
        """
        columns = ["labels"] if "labels" in names else []
        # The TRAINER's resolved value first. It is the one the collator uses,
        # and it is present whenever TRL resolved it -- including the cases the
        # generated block never runs in: a TRL whose guard did not match, or
        # padding-free off from the start. Falling straight through to this
        # split's own schema then read False off a late pre-tokenized split
        # carrying only `input_ids` and `completion_mask`, kept the rows whose
        # completion was cut away entirely, and the collator turned them into
        # all -100, i.e. a NaN eval loss.
        only = getattr(args, "_unsloth_resolved_completion_only", None)
        if only is None:
            only = getattr(args, "_unsloth_completion_only_loss", None)
        if only is None:
            only = getattr(args, "completion_only_loss", None)
        if only is None:
            only = "prompt" in names and "completion" in names
        if only and "completion_mask" in names:
            columns.append("completion_mask")
        if "assistant_masks" in names:
            columns.append("assistant_masks")
        return columns

    def _cap(
        dataset,
        cap,
        args,
        drop_unsupervised = True,
        packs_late = False,
    ):
        # `evaluate()` caps the split, stores it on the trainer, and Transformers
        # then calls `get_eval_dataloader`, which this module also wraps -- so the
        # paired wrappers reach here twice for one call. Re-capping is a no-op at
        # best, and over a one-shot stream it is destructive: `_column_names` and
        # `_sliceable_per_token` each read a row to probe it, and `_CappedStream`
        # hands out a fresh generator over the SAME exhausted source rather than
        # rewinding, so the second pass eats the first rows instead of replaying
        # them. Our own wrapper, capped the same way, is already the answer.
        if _cap_still_holds(dataset, cap, drop_unsupervised):
            return dataset
        names, dataset, probed = _column_names(dataset)
        if "input_ids" not in names:
            # No tokens here yet, so there is nothing to cut. TRL does not
            # tokenize a late split either -- `_prepare_dataset` runs only from
            # `__init__` -- but that is its own gap and not one a truncation can
            # close.
            return _mark_capped(dataset, cap, drop_unsupervised)
        # `eval_packing` is consulted only where the packer can actually reach the
        # split, which is why `packs_late` is passed in per entry point rather than
        # read here. Up to TRL 1.6 nothing packs a late split: `evaluate()` and
        # `predict()` are the base Trainer's, they call `get_eval_dataloader` /
        # `get_test_dataloader` straight through, and `_prepare_dataset` never runs
        # again, so skipping the cap there handed the collator the raw overlength
        # rows with `max_length` already cleared. From TRL 1.7.0 the opposite is
        # true for `evaluate` alone: it prepares the split itself under
        # `eval_packing`, and every strategy owns the overflow -- `wrapped`
        # concatenates the token stream before chunking, `bfd_split` splits an
        # overlength example into more chunks -- so cutting rows at the cap first
        # throws that away. `predict` and the two dataloader builders stay the base
        # Trainer's on every TRL, so they always cap.
        if packs_late and _eval_packing_on(args):
            return _mark_capped(dataset, cap, drop_unsupervised)
        # A packed split carries document lengths, not tokens. Slicing `input_ids`
        # under a `seq_lengths` that still describes the longer row makes
        # padding-free build position ids for tokens the row no longer has, which
        # is worse than not cutting: the construction-time cap refuses this shape
        # too.
        if "seq_lengths" in names:
            return _mark_capped(dataset, cap, drop_unsupervised)
        try:
            # TRL slices [-max_length:] for `keep_end`, and so does the
            # construction-time cap. Always keeping the prefix evaluates the wrong
            # half of every long row for callers whose completion sits at the tail.
            mode = getattr(args, "truncation_mode", "keep_start")
            # keep_start and keep_end are the only two slices there are, and a
            # third value silently became keep_start here, cutting every late
            # row from the side the caller asked us not to. The construction
            # path already refuses those; this one has to as well, and refusing
            # means handing the split back untouched so the caller still has it.
            if mode not in ("keep_start", "keep_end"):
                print(
                    f"Unsloth: `truncation_mode = {mode}` is not one of "
                    "keep_start / keep_end, so this split is left uncapped."
                )
                return dataset
            cut = slice(-cap, None) if mode == "keep_end" else slice(None, cap)
            per_token = _sliceable_per_token(dataset, names, cap, probed)
            # Never on the predict path. Dropping rows is right for a loss, which
            # is meaningless over a row with no supervised token, and wrong for
            # `predict`, whose contract is one prediction per row IN ORDER: a
            # caller zipping the output back onto its own dataframe would silently
            # get a shorter, shifted column.
            supervision = _supervision_columns(args, names) if drop_unsupervised else []
            # A stream has no length and cannot be rewound, so there is no prefix
            # scan to skip the work with, and indexing it does not even fail
            # loudly: on datasets 4.x `dataset[0]` reads 0 as a COLUMN name and
            # returns an IterableColumn, whose len() then raised TypeError into
            # the catch below and handed the eval call its uncapped stream back.
            # map() is lazy and applies to every row it will ever yield.
            overlength = True
            if not _is_stream(dataset):
                try:
                    overlength = max(len(r) for r in dataset["input_ids"]) > cap
                except Exception:
                    # The scan only exists to skip a pointless map. A split with no
                    # column access (a custom map-style dataset) cannot answer it, and
                    # that is not a reason to hand it back uncapped.
                    pass
            # A split already under the cap still goes through the supervision
            # filter below. Being short is not the same as being supervised: a row
            # whose labels are all -100, or whose active mask is all zeros, is a
            # NaN loss whether or not anything had to be cut off it, and the
            # construction-time cap filters those rows unconditionally too.
            # Returning early on the length alone was the one way the two paths
            # disagreed.
            if not overlength and not supervision:
                return _mark_capped(dataset, cap, drop_unsupervised)
            # A split we cannot rewrite is capped on read instead. `map` belongs to
            # `datasets`, and a `with_transform` split has it but recreates its rows
            # on every read, so mapping writes a table nobody reads.
            transform = str((getattr(dataset, "format", None) or {}).get("type", "")).lower()
            if (
                not hasattr(dataset, "map")
                or not hasattr(dataset, "filter")
                or transform == "custom"
            ):
                if _is_stream(dataset):
                    return _mark_capped(
                        _capped_stream(dataset, cut, supervision, per_token), cap, drop_unsupervised
                    )
                return _mark_capped(
                    _CappedRows(dataset, cut, supervision, per_token), cap, drop_unsupervised
                )

            def _slice_row(
                example,
                _cut = cut,
                _cols = tuple(per_token),
            ):
                # Per value, not per column. `_sliceable_per_token` judges by ONE
                # row, so an optional column that is a list there and None (or a
                # different width) three rows later raised inside `map` -- and
                # the broad catch below then returned the UNCAPPED split, losing
                # the `input_ids` truncation too. `input_ids` is always cut; an
                # auxiliary value that cannot take the same slice is left alone.
                out = {}
                width = None
                for name in _cols:
                    value = example[name]
                    try:
                        length = len(value)
                    except Exception:
                        continue  # not sliceable: leave it
                    if name == "input_ids":
                        width = length
                    elif width is not None and length != width:
                        continue  # not aligned: leave it
                    out[name] = value[_cut]
                return out

            new = dataset if not overlength else dataset.map(_slice_row)
            # Truncating can leave a row with every label at -100, or a mask that is
            # now all zeros, which the collator turns into all -100. A batch of
            # those has no supervised token and reports a NaN loss. TRL filters them
            # right after its own truncation; here `args.max_length` is already None
            # so TRL does not, and the construction-time cap filters them for the
            # splits it saw. Masks are applied one after another onto the same
            # labels, so what survives is their intersection.
            # One intersection over labels AND every active mask, not a filter each:
            # the collator applies the masks ONTO the labels, so a row whose only
            # supervised label sits where the mask is zero passes both filters
            # separately and still goes out all -100.
            if supervision:
                kept = new.filter(
                    lambda e, c = tuple(supervision): any(
                        all((x != -100) if n == "labels" else x for n, x in zip(c, v))
                        for v in zip(*[e[n] for n in c])
                    )
                )
                # Hand back the caller's own split when the filter dropped
                # nothing: a copy of an unchanged dataset is a new object for the
                # trainer to cache and reload for no reason at all.
                try:
                    new = new if len(kept) == len(new) else kept
                except Exception:
                    new = kept
            return _mark_capped(new, cap, drop_unsupervised)
        except Exception:
            return dataset  # never turn an eval call into a hard error

    def _memo_token(dataset):
        """What makes this split's cap reusable, or None if nothing does.

        Identity alone is not enough: the same list or custom map-style split can
        be appended to or shortened between two `evaluate()` calls, and the memo
        would hand back a wrapper whose snapshotted indices no longer describe
        it -- rows silently missing, or an index that raises during loading.
        `datasets` splits are content-addressed by `_fingerprint`, which moves
        whenever the rows do, so those are safe to keep. Anything else is
        recomputed, which is the cheap case anyway: the scan this memo exists to
        skip needs a materialised `input_ids` column that these do not have.

        A `with_transform` split is excluded even though it has a fingerprint:
        that covers the backing table, not the transform, so a transform closing
        over mutable state yields different rows under an unchanged fingerprint
        and the memo would replay a filter decided against the old ones. That is
        the same question the cap mark asks, so it is the same helper.
        """
        return _mutation_token(dataset)

    def _cap_cached(
        trainer,
        dataset,
        cap,
        drop_unsupervised = True,
        packs_late = False,
    ):
        """Cap a split once per object.

        `evaluate()` runs at every eval step of a training run, and the scan that
        decides whether anything needs cutting materialises the whole `input_ids`
        column each time. Keep the answer, keyed on the split object and holding a
        reference to it, so a later split cannot inherit its `id()`.
        """
        # Carried onto args because `_cap` only ever sees those. `is not None`
        # rather than truthiness: False is TRL's answer just as much as True.
        resolved = getattr(trainer, "completion_only_loss", None)
        if resolved is not None:
            try:
                trainer.args._unsloth_resolved_completion_only = resolved
            except Exception:
                pass
        token = _memo_token(dataset)
        if token is None:
            return _cap(dataset, cap, trainer.args, drop_unsupervised, packs_late)
        memo = getattr(trainer, "_unsloth_eval_cap_memo", None)
        if memo is None:
            memo = {}
            try:
                trainer._unsloth_eval_cap_memo = memo
            except Exception:
                return _cap(dataset, cap, trainer.args, drop_unsupervised, packs_late)
        # `truncation_mode` shapes the SLICE, so it belongs in the key: without
        # it, evaluating once with keep_start and again with keep_end handed
        # back the cached prefixes for both.
        key = (
            id(dataset),
            drop_unsupervised,
            getattr(trainer.args, "truncation_mode", "keep_start"),
            # `evaluate` and `get_eval_dataloader` share `drop_unsupervised` and
            # see the same object in one call, but only the first may skip the cut
            # under `eval_packing`, so without this the second reused the first's
            # answer.
            packs_late,
        )
        seen = memo.get(key)
        if seen is not None and seen[0] is dataset and seen[1] == cap and seen[3] == token:
            memo[key] = memo.pop(key)  # most recently used goes last
            return seen[2]
        capped = _cap(dataset, cap, trainer.args, drop_unsupervised, packs_late)
        memo[key] = (dataset, cap, capped, token)
        # Bounded, because every entry pins BOTH the original split and the
        # capped copy for the trainer's whole lifetime -- deliberately, so a
        # later split cannot inherit a freed `id()`. A caller that builds a
        # fresh validation subset each epoch therefore accumulated Arrow tables
        # until the host ran out. Evicting the least recently used costs nothing
        # real: a run evaluates over a handful of splits, which is what the
        # bound leaves room for.
        while len(memo) > _EVAL_CAP_MEMO_MAX:
            memo.pop(next(iter(memo)))
        return capped

    def _cap_splits(
        trainer,
        given,
        cap,
        drop_unsupervised = True,
        packs_late = False,
    ):
        # `evaluate(eval_dataset = "validation")` is the supported way to pick one
        # split out of a stored dict: `get_eval_dataloader` resolves it as
        # `self.eval_dataset[eval_dataset]`. Capping the KEY is a no-op, so the
        # split it names reached the collator uncapped. Cap it where it is stored
        # and hand the key straight back.
        if isinstance(given, str):
            stored = getattr(trainer, "eval_dataset", None)
            if isinstance(stored, dict) and given in stored:
                capped = _cap_cached(trainer, stored[given], cap, drop_unsupervised, packs_late)
                # Staged for the caller to swap in and OUT, never written
                # through. Overwriting `stored[given]` destroyed the uncapped
                # original, so a later `truncation_mode = "keep_end"` -- which
                # the memo key now distinguishes on purpose -- could only re-cap
                # the saved prefix and never produce the suffix asked for.
                if capped is not stored[given]:
                    trainer._unsloth_pending_split_swap = (stored, given, capped)
            return given
        if isinstance(given, dict):
            capped = {
                k: _cap_cached(trainer, v, cap, drop_unsupervised, packs_late)
                for k, v in given.items()
            }
            if all(capped[k] is v for k, v in given.items()):
                return given
            return capped
        return _cap_cached(trainer, given, cap, drop_unsupervised, packs_late)

    def _make(original, keyword, drop_unsupervised, packs_late):
        def wrapped(self, *args, **kwargs):
            cap = getattr(getattr(self, "args", None), "max_seq_length", None)
            retained = getattr(getattr(self, "args", None), "max_length", None)
            # A retained `max_length` is NOT proof the cap is enforced
            # downstream. It is exactly what the construction block leaves
            # behind when it turns padding-free OFF instead of clearing the
            # cap, and TRL's collator does not truncate rows that already carry
            # `input_ids`. `_prepare_dataset` runs only from `__init__`, so a
            # split handed over later is never prepared either: skipping on a
            # retained `max_length` let an overlength late split reach the model
            # with nothing enforcing anything. Cap to whichever value is set.
            if retained is not None:
                cap = retained
            if not cap:
                return original(self, *args, **kwargs)
            given = kwargs.get(keyword, args[0] if args else None)
            if given is not None:
                self._unsloth_pending_split_swap = None
                capped = _cap_splits(self, given, cap, drop_unsupervised, packs_late)
                if keyword in kwargs:
                    kwargs[keyword] = capped
                else:
                    args = (capped,) + tuple(args[1:])
                swap = getattr(self, "_unsloth_pending_split_swap", None)
                if swap is None:
                    return original(self, *args, **kwargs)
                # A named split: swap the capped copy in for this call only, so
                # the caller keeps the uncapped original for the next mode.
                container, key, replacement = swap
                self._unsloth_pending_split_swap = None
                previous = container[key]
                container[key] = replacement
                try:
                    return original(self, *args, **kwargs)
                finally:
                    container[key] = previous
            # `evaluate()` with no split passed falls back to the one stored on
            # the trainer, and a caller can install or replace that after
            # construction -- which is exactly where the constructor's cap can no
            # longer see it. Every eval during training comes through here too.
            stored = getattr(self, keyword, None) if keyword == "eval_dataset" else None
            if stored is None:
                return original(self, *args, **kwargs)
            capped = _cap_splits(self, stored, cap, drop_unsupervised, packs_late)
            if capped is stored:
                return original(self, *args, **kwargs)
            # Swapped onto the trainer rather than passed down as an argument,
            # because Trainer.evaluate recurses over a dict of splits by NAME when
            # nothing was passed:
            #     eval_dataset = _eval_dataset if override else eval_dataset_name
            # Passing the dict makes that an override, which changes what
            # `get_eval_dataloader` is handed and what it caches.
            setattr(self, keyword, capped)
            try:
                return original(self, *args, **kwargs)
            finally:
                setattr(self, keyword, stored)

        wrapped._unsloth_eval_cap_wrapped = True
        return wrapped

    # `predict` keeps every row: its contract is one prediction per row in order.
    #
    # The two dataloader builders are public API and neither goes through
    # `evaluate`/`predict`, so a caller doing `get_eval_dataloader(late)` reached
    # the padding-free collator with `args.max_length` already cleared and
    # nothing capping the split. Same wrapper, same argument position; the
    # `drop_unsupervised` split follows the method it serves, since
    # `get_test_dataloader` feeds `predict`.
    # Only `evaluate` can hand its split to TRL's own prep, and only from 1.7.0.
    # Read once, before anything is wrapped, so the probe sees TRL's method rather
    # than ours.
    prepares_late = _trl_prepares_late_evals(trainer_cls)
    for name, keyword, drop_unsupervised, packs_late in (
        ("evaluate", "eval_dataset", True, prepares_late),
        ("predict", "test_dataset", False, False),
        ("get_eval_dataloader", "eval_dataset", True, False),
        ("get_test_dataloader", "test_dataset", False, False),
    ):
        original = getattr(trainer_cls, name, None)
        if original is None or getattr(original, "_unsloth_eval_cap_wrapped", False):
            continue
        setattr(trainer_cls, name, _make(original, keyword, drop_unsupervised, packs_late))


_UNSLOTH_RETURN_HIDDEN_STATES_SUPPORT_MARKER = "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__"
_UNSLOTH_GRPO_HIDDEN_STATES_WRAPPED_ATTR = "_unsloth_grpo_hidden_states_forward_wrapped"
_UNSLOTH_GRPO_HIDDEN_STATES_WARNING_ATTR = "_unsloth_grpo_hidden_states_warning_issued"


def _grpo_hidden_states_wrap_target(model):
    if model is None:
        return None
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        if base_model is not None and base_model is not model:
            return base_model
    for attr in ("base_model", "model"):
        child = getattr(model, attr, None)
        if child is not None and child is not model and hasattr(child, "forward"):
            return child
    return model


def _model_supports_unsloth_return_hidden_states(model):
    target_model = _grpo_hidden_states_wrap_target(model)
    for candidate in (model, target_model):
        if candidate is None:
            continue
        if getattr(candidate, _UNSLOTH_RETURN_HIDDEN_STATES_SUPPORT_MARKER, False):
            return True
        if getattr(type(candidate), _UNSLOTH_RETURN_HIDDEN_STATES_SUPPORT_MARKER, False):
            return True
    return False


def _drop_forward_kwargs_consumed_positionally(forward_signature, args, kwargs):
    if len(args) == 0 or len(kwargs) == 0:
        return kwargs

    consumed_names = []
    for parameter in forward_signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            break
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            consumed_names.append(parameter.name)
        if len(consumed_names) >= len(args):
            break

    if len(consumed_names) == 0:
        return kwargs

    kwargs = dict(kwargs)
    for name in consumed_names:
        kwargs.pop(name, None)
    return kwargs


def _get_num_logits_to_keep(forward_signature, args, kwargs):
    try:
        bound = forward_signature.bind_partial(*args, **kwargs)
        arguments = bound.arguments
        num_logits_to_keep = arguments.get("num_logits_to_keep", 0) or 0
        logits_to_keep = arguments.get("logits_to_keep", 0) or 0
        for parameter in forward_signature.parameters.values():
            if parameter.kind != inspect.Parameter.VAR_KEYWORD:
                continue
            extra_kwargs = arguments.get(parameter.name, {})
            num_logits_to_keep = max(
                num_logits_to_keep,
                extra_kwargs.get("num_logits_to_keep", 0) or 0,
            )
            logits_to_keep = max(
                logits_to_keep,
                extra_kwargs.get("logits_to_keep", 0) or 0,
            )
            break
        return max(num_logits_to_keep, logits_to_keep)
    except TypeError:
        logger.debug(
            "Unsloth: Could not bind forward arguments for GRPO hidden-state fallback.",
            exc_info = True,
        )

    num_logits_to_keep = kwargs.get("num_logits_to_keep", 0) or 0
    logits_to_keep = kwargs.get("logits_to_keep", 0) or 0
    return max(num_logits_to_keep, logits_to_keep)


def _warn_grpo_hidden_states_fallback_once(model, message):
    if getattr(model, _UNSLOTH_GRPO_HIDDEN_STATES_WARNING_ATTR, False):
        return
    setattr(model, _UNSLOTH_GRPO_HIDDEN_STATES_WARNING_ATTR, True)
    logger.warning(message)


def _replace_outputs_logits(outputs, hidden_states):
    if hasattr(outputs, "logits"):
        outputs.logits = hidden_states
        return outputs
    if isinstance(outputs, dict):
        outputs["logits"] = hidden_states
        return outputs
    if isinstance(outputs, tuple) and len(outputs) != 0:
        return (hidden_states,) + tuple(outputs[1:])
    raise TypeError(f"Unsupported output type for GRPO hidden-state fallback: {type(outputs)}")


def _install_grpo_hidden_states_forward_wrapper(model):
    if model is None or getattr(model, _UNSLOTH_GRPO_HIDDEN_STATES_WRAPPED_ATTR, False):
        return False
    if _model_supports_unsloth_return_hidden_states(model):
        return False

    target_model = _grpo_hidden_states_wrap_target(model)
    if getattr(target_model, _UNSLOTH_GRPO_HIDDEN_STATES_WRAPPED_ATTR, False):
        setattr(model, _UNSLOTH_GRPO_HIDDEN_STATES_WRAPPED_ATTR, True)
        return False

    original_forward = target_model.forward
    forward_signature = inspect.signature(original_forward)
    model_name = type(target_model).__name__

    def wrapped_forward(*args, **kwargs):
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") != "1":
            return original_forward(*args, **kwargs)

        forward_kwargs = _drop_forward_kwargs_consumed_positionally(forward_signature, args, kwargs)
        num_logits_to_keep = _get_num_logits_to_keep(forward_signature, args, forward_kwargs)
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True
        try:
            outputs = original_forward(*args, **forward_kwargs)
        except TypeError as error:
            if "output_hidden_states" not in str(error) and "return_dict" not in str(error):
                raise
            _warn_grpo_hidden_states_fallback_once(
                target_model,
                f"Unsloth: GRPO fallback could not request hidden states for unsupported model {model_name}; using logits directly.",
            )
            return original_forward(*args, **kwargs)

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None or len(hidden_states) == 0:
            _warn_grpo_hidden_states_fallback_once(
                target_model,
                f"Unsloth: GRPO fallback did not receive hidden states for unsupported model {model_name}; using logits directly.",
            )
            return outputs

        hidden_states = hidden_states[-1]
        if num_logits_to_keep != 0:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]
        return _replace_outputs_logits(outputs, hidden_states)

    wrapped_forward._unsloth_grpo_hidden_states_forward_wrapped = True
    target_model.forward = wrapped_forward
    setattr(target_model, _UNSLOTH_GRPO_HIDDEN_STATES_WRAPPED_ATTR, True)
    setattr(model, _UNSLOTH_GRPO_HIDDEN_STATES_WRAPPED_ATTR, True)
    return True


def _wrap_grpo_hidden_states_fallback(trainer_cls):
    original_init = trainer_cls.__init__
    if getattr(original_init, "_unsloth_grpo_hidden_states_init_wrapped", False):
        return

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _install_grpo_hidden_states_forward_wrapper(getattr(self, "model", None))
        _install_grpo_hidden_states_forward_wrapper(getattr(self, "ref_model", None))

    wrapped_init._unsloth_grpo_hidden_states_init_wrapped = True
    trainer_cls.__init__ = wrapped_init


def _backport_vision_dataset_gate(RLTrainer_source):
    """Make TRL 0.22.x decide by DATASET, not by model, for SFT vision paths.

    0.22.x keys "skip preparation" and "vision collator" off `_is_vlm` alone, so
    a VLM fine-tuned on text-only data reaches transformers with no tokenized
    columns: "No columns in the dataset match the model's forward method
    signature". Merging the signature columns above is not enough, since skipped
    preparation never creates those columns. Hit by
    Magistral_(24B)-Reasoning-Conversational, which pins trl==0.22.2.

    Back-ports TRL 0.24.0's `_is_vision_dataset` keying; no-op once TRL defines
    the flag itself. Returns the source, patched or unchanged."""
    if 'self._is_vision_dataset = "image" in dataset_sample' in RLTrainer_source:
        return RLTrainer_source
    anchor = "        dataset_sample = next(iter(train_dataset))\n"
    if anchor not in RLTrainer_source:
        return RLTrainer_source

    RLTrainer_source = RLTrainer_source.replace(
        anchor,
        anchor + "        # Unsloth: back-port of TRL 0.24.0's dataset-based check, so a\n"
        "        # text-only fine-tune of a VLM is prepared and collated as text.\n"
        '        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample\n',
        1,
    )
    # Text collator whenever the data is not actually vision data.
    RLTrainer_source = RLTrainer_source.replace(
        "if data_collator is None and not self._is_vlm:",
        "if data_collator is None and not (self._is_vlm and self._is_vision_dataset):",
    )
    RLTrainer_source = RLTrainer_source.replace(
        "elif data_collator is None and self._is_vlm:",
        "elif data_collator is None and self._is_vlm and self._is_vision_dataset:",
    )
    # Tokenize it too: skipping preparation only saves image-processing cost.
    RLTrainer_source = RLTrainer_source.replace(
        'args.dataset_kwargs.get("skip_prepare_dataset", False) or self._is_vlm',
        'args.dataset_kwargs.get("skip_prepare_dataset", False)'
        " or (self._is_vlm and self._is_vision_dataset)",
    )
    return RLTrainer_source


def _patch_trl_rl_trainers(trainer_file = "grpo_trainer"):
    # Defensive wrapper: matches patch_trl_rl_trainers()'s try/except so
    # direct callers don't see exceptions from the impl on TRL versions
    # that rename or move classes (e.g. TRL 1.x trl.experimental).
    try:
        return _patch_trl_rl_trainers_impl(trainer_file)
    except Exception as e:
        # Warning, not info. The impl RETURNS for the benign case this swallow
        # exists for (a trainer this TRL does not ship), so anything reaching
        # here means the module imported and generation itself failed, and the
        # run silently falls back to trl's trainer, losing Unsloth's
        # compute_loss, bf16/fp16 fixup and dataset handling at once.
        logger.warning_once(
            f"Unsloth: Could not build the patched trl.trainer.{trainer_file}, "
            f"so training will use trl's own trainer instead: "
            f"{type(e).__name__}: {e}"
        )
        return


def _patch_trl_rl_trainers_impl(trainer_file = "grpo_trainer"):
    # Patch for vLLM and Unsloth PEFT
    import trl
    import trl.trainer

    try:
        trainer = eval(f"trl.trainer.{trainer_file}")
    except Exception as error:
        logger.info(f"Unsloth: Could not import trl.trainer.{trainer_file}: {error}")
        return

    # Get SFTTrainer and SFTConfig names
    name = [
        x
        for x in dir(trainer)
        if x.endswith("Trainer")
        and x != "Trainer"
        and not x.startswith("_")
        and trainer_file.split("_")[0] in x.lower()
    ]
    config = [
        x
        for x in dir(trainer)
        if x.endswith("Config")
        and x != "Config"
        and not x.startswith("_")
        and trainer_file.split("_")[0] in x.lower()
    ]
    if len(name) != 1:
        logger.info(
            f"Unsloth: Could not find Trainer class in trl.trainer.{trainer_file}. Found: {name}"
        )
        return
    if len(config) != 1:
        # TRL 0.26+: Config may be in a separate *_config.py module
        config_module_name = trainer_file.replace("_trainer", "_config")
        try:
            config_mod = eval(f"trl.trainer.{config_module_name}")
            config = [
                x
                for x in dir(config_mod)
                if x.endswith("Config")
                and x != "Config"
                and not x.startswith("_")
                and trainer_file.split("_")[0] in x.lower()
            ]
        except Exception:
            pass
    if len(config) != 1 and len(name) == 1:
        # Thin wrapper fallback: walk the Trainer's MRO to find Config
        # in the real implementation module (e.g., trl.experimental.bco)
        try:
            _temp_cls = eval(f"trl.trainer.{trainer_file}.{name[0]}")
            for _parent in _temp_cls.__mro__[1:]:
                if _parent is object:
                    continue
                _parent_mod = inspect.getmodule(_parent)
                if _parent_mod is None or _parent_mod.__name__ == f"trl.trainer.{trainer_file}":
                    continue
                config = [
                    x
                    for x in dir(_parent_mod)
                    if x.endswith("Config")
                    and x != "Config"
                    and not x.startswith("_")
                    and trainer_file.split("_")[0] in x.lower()
                ]
                if len(config) == 1:
                    break
        except Exception:
            pass
    if len(config) != 1:
        logger.info(
            f"Unsloth: Could not find Config class in trl.trainer.{trainer_file}. Found: {config}"
        )
        return

    # Get SFTTrainer, SFTConfig
    RLTrainer_name = name[0]
    RLConfig_name = config[0]
    try:
        RLTrainer = eval(f"trl.trainer.{trainer_file}.{RLTrainer_name}")
    except Exception as e:
        logger.info(
            f"Unsloth: Could not load {RLTrainer_name} from trl.trainer.{trainer_file}: {e}"
        )
        return
    _config_resolved_module = None
    try:
        RLConfig = eval(f"trl.trainer.{trainer_file}.{RLConfig_name}")
    except Exception:
        # TRL 0.26+: Config may be in a separate *_config.py module
        try:
            config_module_name = trainer_file.replace("_trainer", "_config")
            RLConfig = eval(f"trl.trainer.{config_module_name}.{RLConfig_name}")
        except Exception:
            # Thin wrapper fallback: load Config from parent trainer's module
            _config_loaded = False
            try:
                _temp_cls = eval(f"trl.trainer.{trainer_file}.{name[0]}")
                for _parent in _temp_cls.__mro__[1:]:
                    if _parent is object:
                        continue
                    _parent_mod = inspect.getmodule(_parent)
                    if _parent_mod is None or _parent_mod.__name__ == f"trl.trainer.{trainer_file}":
                        continue
                    if hasattr(_parent_mod, RLConfig_name):
                        RLConfig = getattr(_parent_mod, RLConfig_name)
                        _config_resolved_module = _parent_mod
                        _config_loaded = True
                        break
            except Exception:
                pass
            if not _config_loaded:
                logger.info(f"Unsloth: Could not load {RLConfig_name}")
                return

    # Check name
    if RLTrainer.__name__.startswith("Unsloth"):
        print(f"Unsloth: {RLTrainer.__name__} is already patched.")
        return
    if RLConfig.__name__.startswith("Unsloth"):
        print(f"Unsloth: {RLConfig.__name__} is already patched.")
        return

    # TRL 0.26+: Resolve thin wrappers to their experimental parent class.
    # Thin wrappers are deprecation shims in trl.trainer that just forward
    # *args/**kwargs to the real implementation in trl.experimental.
    # Only resolve if a parent class actually lives in a trl.experimental module.
    _trainer_resolved_module = None
    try:
        _trainer_src = inspect.getsource(RLTrainer)
        _trainer_module = inspect.getmodule(RLTrainer)
        _trainer_module_src = inspect.getsource(_trainer_module) if _trainer_module else ""
        if "trl.experimental" in _trainer_src or "trl.experimental" in _trainer_module_src:
            for _parent in RLTrainer.__mro__[1:]:
                if _parent is object:
                    continue
                _parent_mod = inspect.getmodule(_parent)
                if _parent_mod is None:
                    continue
                # Only resolve to a parent that lives in trl.experimental
                if "trl.experimental" in _parent_mod.__name__:
                    RLTrainer = _parent
                    _trainer_resolved_module = _parent_mod
                    break
    except Exception:
        pass

    try:
        _config_src = inspect.getsource(RLConfig)
        _config_module = inspect.getmodule(RLConfig)
        _config_module_src = inspect.getsource(_config_module) if _config_module else ""
        if "trl.experimental" in _config_src or "trl.experimental" in _config_module_src:
            for _parent in RLConfig.__mro__[1:]:
                if _parent is object:
                    continue
                _parent_mod = inspect.getmodule(_parent)
                if _parent_mod is None:
                    continue
                # Only resolve to a parent that lives in trl.experimental
                if "trl.experimental" in _parent_mod.__name__:
                    RLConfig = _parent
                    break
    except Exception:
        pass

    # Get old source
    old_RLTrainer_source = inspect.getsource(RLTrainer)
    old_RLConfig_source = inspect.getsource(RLConfig)

    if _trainer_resolved_module is not None:
        all_imports = dir(_trainer_resolved_module)
    elif _config_resolved_module is not None:
        all_imports = dir(_config_resolved_module)
    else:
        all_imports = dir(trainer)
    # Fix _deprecate_arguments not getting imported so stop __ but not _
    imports = [x for x in all_imports if not x.startswith("__")]

    # Get default arguments
    EMPTY = inspect.Parameter.empty
    processed = []
    for RLobject in [RLTrainer, RLConfig]:
        parameters = inspect.signature(RLobject.__init__).parameters
        types = (
            bool,
            type(None),
            int,
            float,
            str,
        )
        arguments = ["self"]
        call_args = []
        for k, v in parameters.items():
            if k == "self":
                continue
            v = v.default
            if v == "\n":
                v = re.escape("\n")
            if v is EMPTY:
                arguments.append(k)
            elif type(v) is str:
                arguments.append(f"{k} = '{v}'")
            elif type(v) in types:
                arguments.append(f"{k} = {v}")
            else:
                continue
            call_args.append(f"{k} = {k}")
        arguments = f"\n{' ' * 8}" + f",\n{' ' * 8}".join(arguments)
        call_args = f"\n{' ' * 12}" + f",\n{' ' * 12}".join(call_args)
        processed.append(
            (
                arguments,
                call_args,
            )
        )

    # Process RLTrainer first
    arguments, call_args = processed[0]
    RLTrainer_post = ""

    # Add tokenizer if not seen
    if "tokenizer" not in parameters and "processing_class" in parameters:
        arguments += f",\n{' ' * 8}tokenizer = None"
        call_args = call_args.replace(
            "processing_class = processing_class",
            "processing_class = tokenizer if tokenizer is not None else processing_class",
        )

    # Edit bf16, fp16 by checking model's dtype/torch_dtype directly
    extra_args = ""
    if "args" in call_args and "model" in call_args:
        mixed_precision = (
            "use_bf16 = getattr(args, 'bf16', False)\n"
            "if type(use_bf16) is not bool: use_bf16 = False\n"
            "use_fp16 = getattr(args, 'fp16', False)\n"
            "if type(use_fp16) is not bool: use_fp16 = False\n"
            "force_float32 = False\n"
            # device-aware bf16 check (CUDA/XPU/HIP), so V100/T4 never pick bf16
            # but AMD/Intel are unaffected; fall back on older unsloth_zoo.
            "try:\n"
            "    from unsloth_zoo.device_type import device_is_bf16_supported as _bf16_supported\n"
            "except Exception:\n"
            "    _bf16_supported = torch.cuda.is_bf16_supported\n"
            # FORCE_FLOAT32 models (Gemma3, gpt_oss, ...) cannot use float16. On a GPU without
            # bf16 (V100/T4) keep them in float32 so they never autocast to fp16. On a bf16 GPU,
            # full finetuning can still use bf16 autocast (master weights stay float32), which is
            # faster and uses less memory; LoRA/QLoRA keep float32 when forced.
            "full_finetuning = os.environ.get('UNSLOTH_ENABLE_FULL_FINETUNING', '0') == '1'\n"
            "if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1' and not (full_finetuning and _bf16_supported()):\n"
            "    print('Unsloth: Switching to float32 training since model cannot work with float16')\n"
            "    force_float32 = True\n"
            "mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')\n"
            "dtype = getattr(model.config, 'dtype', None) or getattr(model.config, 'torch_dtype', None)\n"
            "if dtype is None: dtype = model.get_input_embeddings().weight.dtype\n"
            "from unsloth_zoo.utils import _get_dtype\n"
            "dtype = _get_dtype(dtype)\n"
            "float16 = dtype == torch.float16\n"
            "bfloat16 = dtype == torch.bfloat16\n"
            "if full_finetuning:\n"
            "    if bfloat16 and use_fp16: use_fp16 = False\n"
            "    if float16 and use_bf16: use_bf16 = False\n"
            "if not force_float32 and (float16 and use_bf16): raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')\n"
            "if not force_float32 and (bfloat16 and use_fp16): raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')\n"
            "if force_float32:\n"
            "    # Forced float32 training\n"
            "    args.fp16 = False\n"
            "    args.bf16 = False\n"
            "    os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'\n"
            "    if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'\n"
            "    # args.mixed_precision is a new argument which needs to be set now\n"
            "elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32':\n"
            "    # Mixed precision training. bf16 only if the GPU supports it; V100/T4 use fp16.\n"
            "    use_bf16_amp = (not float16) and _bf16_supported()\n"
            "    args.fp16 = not use_bf16_amp\n"
            "    args.bf16 = use_bf16_amp\n"
            "    os.environ['ACCELERATE_MIXED_PRECISION'] = 'bf16' if use_bf16_amp else 'fp16'\n"
            "    if hasattr(args, 'mixed_precision'): args.mixed_precision = 'bf16' if use_bf16_amp else 'fp16'\n"
            "    # args.mixed_precision is a new argument which needs to be set now\n"
            "elif mixed_precision_dtype == 'bfloat16':\n"
            "    # Both False since bfloat16 full finetuning doesn't do any autocasting.\n"
            "    args.fp16 = False\n"
            "    args.bf16 = False\n"
            "    os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'\n"
            "    if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'\n"
            "    # args.mixed_precision is a new argument which needs to be set now\n"
            "elif use_bf16 or use_fp16:\n"
            "    # transformers <5 exported this itself from fp16/bf16; 5.x dropped the write, so an\n"
            "    # explicit flag left it unset and GRPO readers defaulted to 'fp16', wrapping a\n"
            "    # bfloat16 model in a float16 autocast. See unslothai/unsloth#4891.\n"
            "    os.environ['ACCELERATE_MIXED_PRECISION'] = 'bf16' if use_bf16 else 'fp16'\n"
            "    if hasattr(args, 'mixed_precision'): args.mixed_precision = 'bf16' if use_bf16 else 'fp16'\n"
            "\n"
        )
        extra_args += mixed_precision

    # Check if per_device_eval_batch_size (default 8) bigger than bsz
    # Also use FP16 / BF16 evaluation
    if "args" in call_args:
        # Check eval_dataset first
        if "eval_dataset" in call_args:
            check_eval_dataset = (
                "if getattr(args, 'eval_dataset', None) is not None and "
                "getattr(args, 'eval_strategy', 'no') == 'no':\n"
                "    args.eval_strategy = 'steps'\n"
                "    if getattr(args, 'eval_steps', None) is None: args.eval_steps = 0.1\n"
            )
            extra_args += check_eval_dataset

        # Check if gradient accumulation bug fix is applied
        check_ga = (
            "ga_steps = getattr(args, 'gradient_accumulation_steps', None)\n"
            "if ga_steps is not None and ga_steps > 1:\n"
            "    from transformers import __version__ as transformers_version\n"
            "    if Version(transformers_version) <= Version('4.45.2'):\n"
            "        print('**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\\n'\n"
            "              '`pip install --upgrade --no-cache-dir --force-reinstall --no-deps unsloth transformers trl unsloth_zoo`')\n"
        )
        extra_args += check_ga

        eval_changes = (
            "if getattr(args, 'eval_strategy', 'no') != 'no':\n"
            "    eval_bsz = getattr(args, 'per_device_eval_batch_size', 8)\n"
            "    if eval_bsz == 8 and args.per_device_train_batch_size < eval_bsz: args.per_device_eval_batch_size = args.per_device_train_batch_size\n"
            "    if getattr(args, 'eval_accumulation_steps', None) is None and ga_steps is not None: args.eval_accumulation_steps = ga_steps\n"
            "fp16_full_eval = getattr(args, 'fp16_full_eval', False)\n"
            "if type(fp16_full_eval) is not bool: fp16_full_eval = False\n"
            "bf16_full_eval = getattr(args, 'bf16_full_eval', False)\n"
            "if type(bf16_full_eval) is not bool: bf16_full_eval = False\n"
            "if args.fp16 and bf16_full_eval: args.bf16_full_eval = False; args.fp16_full_eval = True\n"
            "if args.bf16 and fp16_full_eval: args.bf16_full_eval = True; args.fp16_full_eval = False\n"
            "if force_float32:\n"
            "    args.bf16_full_eval = False\n"
            "    args.fp16_full_eval = False\n"
            "elif os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32') == 'bfloat16':\n"
            "    args.bf16_full_eval = True\n"
            "    args.fp16_full_eval = False\n"
            "elif not bf16_full_eval and not fp16_full_eval:\n"
            "    args.bf16_full_eval = args.bf16\n"
            "    args.fp16_full_eval = args.fp16\n"
        )
        extra_args += eval_changes

    # Force logits to be produced if preprocess_logits_for_metrics or compute_metrics is used
    if "model" in call_args:
        logits_check = (
            "_output_logits = False\n"
            "if locals().get('compute_metrics', None) is not None: _output_logits = True\n"
            "if locals().get('preprocess_logits_for_metrics', None) is not None: _output_logits = True\n"
            "if _output_logits:\n"
            "    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"
        )
        extra_args += logits_check
        warnings_issued_check = (
            "if model is not None:\n"
            "    _warnings_issued = getattr(model, 'warnings_issued', None)\n"
            "    if _warnings_issued is None:\n"
            "        model.warnings_issued = {}\n"
            "    elif not isinstance(_warnings_issued, dict):\n"
            "        try:\n"
            "            model.warnings_issued = dict(_warnings_issued)\n"
            "        except Exception:\n"
            "            model.warnings_issued = {}\n"
        )
        extra_args += warnings_issued_check

    # Check max_seq_length
    if "model" in call_args:
        length_check = (
            "if 'max_seq_length' not in locals() and not hasattr(args, 'max_seq_length'):\n"
            "    pass\n"
            "else:\n"
            "    model_max_seq_length = getattr(model, 'max_seq_length', None)\n"
            "    args_max_seq_length  = getattr(args,  'max_seq_length', None)\n"
            "    if args_max_seq_length is None and model_max_seq_length is not None:\n"
            "        max_seq_length = model.max_seq_length\n"
            "        if hasattr(args, 'max_seq_length'): args.max_seq_length = max_seq_length\n"
            "    elif args_max_seq_length is not None and model_max_seq_length is not None:\n"
            "        if args_max_seq_length > model_max_seq_length:\n"
            "            print('Unsloth: You set `max_seq_length` as ' + str(args_max_seq_length) + ' but '\n"
            "                   'the maximum the model supports is ' + str(model_max_seq_length) + '. We shall reduce it.')\n"
            "            args.max_seq_length = model_max_seq_length\n"
        )
        extra_args += length_check

        # At this point max_seq_length might be set, but trl is moving to max_length
        if trainer_file == "sft_trainer":
            max_length_check = (
                "if 'max_length' not in locals() and not hasattr(args, 'max_length'):\n"
                "    pass\n"
                "else:\n"
                "    if hasattr(args, 'max_seq_length') and args.max_seq_length is not None and args.max_seq_length > 0:\n"
                "        if hasattr(args, 'max_length'):\n"
                "            args.max_length = args.max_seq_length\n"
                "            max_length = args.max_length\n"
                "    else:\n"
                "        model_max_length = getattr(model, 'max_seq_length', None)\n"
                "        if model_max_length is None: model_max_length = getattr(model, 'max_length', None)\n"
                "        if model_max_length is not None:\n"
                "            args.max_length = model_max_length\n"
                "            max_length = args.max_length\n"
                "        elif hasattr(args, 'max_length') and args.max_length is not None:\n"
                "            max_length = args.max_length\n"
                "            # if we are here, then we are in a weird case where max_length is set but max_seq_length is not set\n"
                "            setattr(model, 'max_seq_length', max_length)\n"
                "        else:\n"
                "            print('Unsloth: We did not find `max_seq_length` or `max_length` in the model or args. We will set it to 1024.')\n"
                "            args.max_length = 1024\n"
            )
            # TRL >= 1.0.0 refuses padding-free without packing while `max_length` is
            # set, since it cannot truncate a flattened batch. Unsloth auto-enables
            # padding-free and the block above always writes `args.max_length`, so
            # essentially every SFT user tripped that guard.
            #
            # Keep the length that block resolved (`max_seq_length`, capped by the
            # model, wins) and move it to wherever it will actually be enforced:
            #   * Unsloth's dataset prep will tokenize -> park it in `max_seq_length`,
            #     which prep reads, and hand TRL the None it asks for.
            #   * it will not (`skip_prepare_dataset`, or rows that already carry
            #     `input_ids` / `labels`) -> nothing would truncate, so turn
            #     padding-free off and keep `max_length` for TRL's own collator.
            #
            # The copy is unconditional on purpose: no TRL from 0.22.2 to 1.9.2
            # declares `max_seq_length` on SFTConfig (only UnslothSFTConfig re-adds
            # it), so a hasattr() gate would skip it for every pristine
            # `trl.SFTConfig` and the clear below would drop the cap. `args` is a
            # dataclass instance, so the assignment just adds the attribute and
            # `to_dict()` never sees it.
            #
            # It must be None, not 0: TRL's guard reads `args.max_length is not None`,
            # and rl_replacements.py normalises the None back to 0 inside the Zoo.
            # The schema comes off the first yielded row like the Zoo does, falling
            # back to `column_names` (a `with_transform` dataset reports its backing
            # columns there while yielding `input_ids`).
            if "`max_length` is not enforced" in old_RLTrainer_source:
                max_length_check += (
                    "if getattr(args, 'padding_free', False) is True and not getattr(args, 'packing', False) "
                    "and getattr(args, 'max_length', None) is not None:\n"
                    "    _unsloth_prep_truncates = True\n"
                    "    _unsloth_skip_prepare = False\n"
                    "    try:\n"
                    "        _unsloth_dataset_kwargs = getattr(args, 'dataset_kwargs', None)\n"
                    "        if _unsloth_dataset_kwargs is not None and _unsloth_dataset_kwargs.get('skip_prepare_dataset', False):\n"
                    "            _unsloth_prep_truncates = False\n"
                    "            _unsloth_skip_prepare = True\n"
                    "    except Exception:\n"
                    "        pass\n"
                    # Metadata first and a row only as a fallback, the other way round
                    # from before: reading a row off a one-shot training stream consumes
                    # it, and nothing here chains it back, so the run began at row 2.
                    # `iter(x) is iter(x)` marks the streams that cannot spare one --
                    # the same signal the cap scan and the schema probe use.
                    # A `with_transform` split reports its BACKING columns, so a
                    # transform yielding `input_ids` over a stored `text` answered
                    # "raw" and the cap was cleared for rows nothing then truncates.
                    # Its rows are rebuilt on every read, so probing one is free.
                    # An unprobeable stream cannot be ruled tokenized either, and
                    # `True` there clears the cap on the same guess: refuse instead
                    # and keep `max_length` for TRL's collator.
                    # One rule, read twice: the late `_unsloth_pretokenized` probe asks the
                    # same question of an eval split and must not trust its columns either.
                    "    def _unsloth_is_transformed(_ds):\n"
                    "        _f = getattr(_ds, 'format', None)\n"
                    "        _f = _f.get('type') if isinstance(_f, dict) else None\n"
                    "        return (_f or getattr(_ds, '_format_type', None)) == 'custom'\n"
                    "    try:\n"
                    "        _unsloth_transformed = _unsloth_is_transformed(train_dataset)\n"
                    "        _unsloth_columns = None if _unsloth_transformed else getattr(train_dataset, 'column_names', None)\n"
                    "        if _unsloth_columns is None and train_dataset is not None:\n"
                    "            _unsloth_probe_cols = iter(train_dataset)\n"
                    "            if _unsloth_probe_cols is train_dataset or _unsloth_probe_cols is iter(train_dataset):\n"
                    "                _unsloth_prep_truncates = False\n"
                    "            else:\n"
                    "                _unsloth_first_row = next(_unsloth_probe_cols, None)\n"
                    "                if isinstance(_unsloth_first_row, dict): _unsloth_columns = list(_unsloth_first_row.keys())\n"
                    "        if _unsloth_columns is None and not _unsloth_transformed:\n"
                    "            _unsloth_columns = getattr(train_dataset, 'column_names', None)\n"
                    "            if isinstance(_unsloth_columns, dict):\n"
                    "                _unsloth_columns = [_c for _v in _unsloth_columns.values() for _c in (_v or [])]\n"
                    "        if _unsloth_columns is not None and ('input_ids' in _unsloth_columns or 'labels' in _unsloth_columns):\n"
                    "            _unsloth_prep_truncates = False\n"
                    "    except Exception:\n"
                    "        _unsloth_prep_truncates = False\n"
                    # Already-tokenized rows are not a dead end. TRL's own _prepare_dataset
                    # truncates them (sft_trainer.py: `elif args.max_length is not None:
                    # truncate_dataset(...)`), and the LM collator it builds passes no
                    # max_length, so truncation there is the ONLY thing enforcing the cap. The
                    # Zoo's replacement returns pre-tokenized rows untouched, so doing it here
                    # restores TRL's contract rather than inventing one, and padding-free can
                    # stay on. skip_prepare_dataset is the exception: the user asked for the
                    # dataset to be left alone, and TRL skips truncation there too.
                    # Only a MATERIALISED tokenized dataset. A with_transform dataset yields
                    # input_ids on access while column_names still says ["text"], so mapping it
                    # would truncate the wrong thing; that case keeps the fallback below.
                    # One predicate, applied per DATASET, because the train split and each eval
                    # split can be in different shapes. Truncating is only safe for a materialised
                    # tokenized table: raw conversational rows (messages: list[dict]) are per-row
                    # sequences too and would be sliced into corrupted turns, and a with_transform
                    # dataset recreates its rows on read, so map() writes the backing table while
                    # the reader keeps handing back overlength input_ids.
                    "    def _unsloth_truncatable(_ds):\n"
                    "        if _ds is None or not hasattr(_ds, 'map'): return False\n"
                    "        try:\n"
                    "            if str((getattr(_ds, 'format', None) or {}).get('type', '')).lower() == 'custom':\n"
                    "                return False\n"
                    "        except Exception:\n"
                    "            return False\n"
                    "        try:\n"
                    "            _cols = getattr(_ds, 'column_names', None)\n"
                    "            if isinstance(_cols, dict):\n"
                    "                _cols = [_c for _v in _cols.values() for _c in (_v or [])]\n"
                    # A packed split is out: TRL skips truncation when packing
                    # (`if args.max_length is not None and not packing`), and cutting
                    # `input_ids` under a `seq_lengths` that still describes the old
                    # row is worse than not cutting at all.
                    "            if 'seq_lengths' in (_cols or ()): return False\n"
                    "            return bool(_cols) and 'input_ids' in _cols\n"
                    "        except Exception:\n"
                    "            return False\n"
                    # Read a row BACK. Every predicate above is a guess about what the dataset
                    # will do; this is the one check that observes it. A split with no
                    # `input_ids` is raw, so prep tokenizes it with the cap and it is fine.
                    "    _unsloth_cap = args.max_length\n"
                    # TRL slices [-max_length:] for `keep_end`, which callers use when
                    # the completion sits at the tail of a long prompt. Consuming
                    # `max_length` while always keeping the prefix trained on the wrong
                    # half of every row.
                    "    _unsloth_truncation_mode = getattr(args, 'truncation_mode', 'keep_start') or 'keep_start'\n"
                    # keep_start and keep_end are the only two slices there are. TRL's SFT path
                    # never reads this attribute at all -- it belongs to the preference trainers --
                    # so nothing downstream would catch a third value, and mapping it to the
                    # default would silently cut from the side the caller asked us not to. Refuse
                    # the enforcement claim instead, the same answer this block gives for any
                    # split it cannot honour.
                    "    _unsloth_keep_end = _unsloth_truncation_mode == 'keep_end'\n"
                    "    _unsloth_known_mode = _unsloth_truncation_mode in ('keep_start', 'keep_end')\n"
                    "    _unsloth_slice = slice(-_unsloth_cap, None) if _unsloth_keep_end else slice(None, _unsloth_cap)\n"
                    # Resolved here, one level out from the truncation block, because the fallback
                    # below reads it even when `skip_prepare_dataset` skips that block entirely.
                    # `eval_packing` is separate from `packing`:
                    #     packing = args.packing if args.eval_packing is None else args.eval_packing
                    # (sft_trainer.py), so packing = False with eval_packing = True reaches this
                    # branch, which is gated on `not args.packing`. TRL then PACKS the eval split
                    # rather than truncating it, and every strategy owns the overflow: `wrapped`
                    # concatenates the stream before chunking, `bfd_split` splits an overlength
                    # example into more chunks. Cutting rows at the cap first throws that away.
                    "    _unsloth_eval_packing = getattr(args, 'packing', False) if getattr(args, 'eval_packing', None) is None else getattr(args, 'eval_packing')\n"
                    "    _unsloth_completion_only = getattr(args, 'completion_only_loss', None)\n"
                    # Column names first, and a row only if reading one is free. On a
                    # one-shot stream this probe consumed the first TRAINING example --
                    # nothing chains it back here, so the run started at row 2. The
                    # same `iter(x) is iter(x)` signal the cap scan uses marks those,
                    # and an unreadable schema resolves to False, which is what TRL
                    # itself answers for a split with no `prompt`/`completion`.
                    # A `with_transform` split answers `column_names` with its BACKING
                    # table, so a transform storing `text` but yielding `prompt` /
                    # `completion` resolved to False here while TRL, which reads a
                    # yielded row, resolved True and applied `completion_mask`. The
                    # cap filters then kept rows whose completion was cut away
                    # entirely, and the collator turned those into all -100.
                    # Same rule as the two schema probes above: transformed columns
                    # are not evidence, so ignore them and read a row instead.
                    "    if _unsloth_completion_only is None:\n"
                    "        try:\n"
                    # A `set_format(columns = [...], output_all_columns = False)`
                    # split yields only the named columns while `column_names`
                    # still answers with the whole backing table. Reading the
                    # table resolved completion-only True off a `completion` the
                    # rows never hand over, TRL resolved False from a yielded
                    # row, and the cap then filtered on a `completion_mask` the
                    # collator ignores -- dropping valid rows, up to emptying the
                    # split. The format's own column list is what is yielded.
                    "            _unsloth_fmt = getattr(train_dataset, 'format', None)\n"
                    "            _unsloth_fmt = _unsloth_fmt if isinstance(_unsloth_fmt, dict) else {}\n"
                    "            _unsloth_shown = _unsloth_fmt.get('columns')\n"
                    "            if _unsloth_fmt.get('output_all_columns') or not _unsloth_shown:\n"
                    "                _unsloth_shown = getattr(train_dataset, 'column_names', None)\n"
                    "            _unsloth_train_sample = {} if _unsloth_is_transformed(train_dataset) else dict.fromkeys(\n"
                    "                _unsloth_shown or [])\n"
                    "            if not _unsloth_train_sample:\n"
                    "                _unsloth_probe = iter(train_dataset)\n"
                    "                if _unsloth_probe is train_dataset or _unsloth_probe is iter(train_dataset):\n"
                    "                    _unsloth_train_sample = {}\n"
                    "                else:\n"
                    "                    _unsloth_train_sample = next(_unsloth_probe, None) or {}\n"
                    "        except Exception:\n"
                    "            _unsloth_train_sample = {}\n"
                    "        _unsloth_completion_only = ('prompt' in _unsloth_train_sample and 'completion' in _unsloth_train_sample)\n"
                    # Parked on args so the late evaluate()/predict() cap reads the SAME value.
                    # It resolves from the train schema, which a split handed over later cannot
                    # see, and disagreeing with the collator is what leaves an all -100 row in.
                    "    args._unsloth_completion_only_loss = _unsloth_completion_only\n"
                    # EVERY row, not the first. A short row 0 in front of a long
                    # row 5000 read as "within the cap", and in the fallback branch
                    # nothing downstream truncates it. A map-style split is read in
                    # full; a stream cannot be rewound, so a bounded prefix is all
                    # there is and the check says so rather than pretending.
                    "    _UNSLOTH_SCAN_ROWS = 1024\n"
                    "    def _unsloth_within_cap(_ds):\n"
                    "        if _ds is None: return True\n"
                    "        try:\n"
                    "            try:    _n = len(_ds)\n"
                    "            except Exception: _n = None\n"
                    # A single-pass stream cannot be scanned at all: reading it here IS
                    # consuming it, and the trainer would then get an exhausted split
                    # (or one short by up to 1024 rows). Two `iter()` calls handing back
                    # the same object is what says so -- true for a bare generator and
                    # for a `torch.utils.data.IterableDataset` that returns shared
                    # iterator state, false for a `datasets.IterableDataset`, which
                    # restarts. Unverifiable, so answer as the prefix case below does:
                    # not proven within the cap, which drops the enforcement claim and
                    # leaves every row where it is.
                    "            _unsloth_rows = iter(_ds)\n"
                    "            if _unsloth_rows is iter(_ds): return False\n"
                    "            _seen = 0\n"
                    "            for _row in _unsloth_rows:\n"
                    "                if 'input_ids' not in _row: return True\n"
                    "                if len(_row['input_ids']) > _unsloth_cap: return False\n"
                    # An unexhausted stream is UNVERIFIED, not verified. Calling the
                    # first 1024 fitting rows proof let a later overlength row through,
                    # and in the fallback branch nothing truncates a pre-tokenized row.
                    # A stream that can be rewritten is capped by the lazy map above and
                    # never reaches this; one that cannot is refused.
                    "                _seen += 1\n"
                    "                if _n is None and _seen >= _UNSLOTH_SCAN_ROWS: return False\n"
                    "        except Exception:\n"
                    "            return False\n"
                    "        return True\n"
                    # Each eval split counts. A tokenized eval split in a shape the truncation
                    # cannot rewrite (with_transform, or an iterable with no column_names) is
                    # left alone above, and prep will not re-tokenize rows that already carry
                    # `input_ids`, so consuming `max_length` on the strength of the train split
                    # alone let evaluation run over the cap.
                    "    def _unsloth_splits_within_cap(_ev):\n"
                    "        _splits = list(_ev.values()) if isinstance(_ev, dict) else [_ev]\n"
                    "        return all(_unsloth_within_cap(_s) for _s in _splits)\n"
                    # Not train-only. `_unsloth_prep_truncates` is decided from the train
                    # split, so a raw train beside a pre-tokenized eval set skipped this
                    # whole block, consumed `max_length`, and left evaluation uncapped:
                    # prep does not re-tokenize rows that already carry `input_ids`.
                    "    if not _unsloth_skip_prepare:\n"
                    # TRL forwards its preparation map_kwargs; honour the same setting so a large
                    # pre-tokenized dataset is not rewritten single-process. Resolved through the
                    # same helper as every other map site: the config layer writes "run
                    # in-process" as `dataset_num_proc = 1`, and datasets >= 4.1 builds a Pool(1)
                    # for it, forking a tokenizer worker on the host that asked for none.
                    "        _unsloth_map_kw = {}\n"
                    "        try:\n"
                    "            try:\n"
                    "                from unsloth_zoo.dataset_num_proc import get_dataset_num_proc as _unsloth_get_nproc\n"
                    "            except ImportError:\n"
                    "                from unsloth.dataset_num_proc import get_dataset_num_proc as _unsloth_get_nproc\n"
                    "            _unsloth_nproc = _unsloth_get_nproc(getattr(args, 'dataset_num_proc', None))\n"
                    "        except Exception:\n"
                    "            _unsloth_nproc = None\n"
                    "        if _unsloth_nproc: _unsloth_map_kw['num_proc'] = _unsloth_nproc\n"
                    # Same rule as TRL's truncate_dataset: slice every per-row list column, so
                    # input_ids, labels, attention_mask and the masks stay aligned. Written out
                    # rather than imported, because trl.data_utils drags in the processor stack
                    # and an ImportError there would silently drop the cap. A torch/numpy format
                    # hands batched map() tensors, where `if _col` raises on the ambiguous truth
                    # value, so ask for a per-row sequence and exclude str/bytes.
                    "        def _unsloth_is_sequence_column(_col):\n"
                    "            try:\n"
                    "                if len(_col) == 0: return False\n"
                    "                _first = _col[0]\n"
                    "            except Exception:\n"
                    "                return False\n"
                    "            if isinstance(_first, (str, bytes)): return False\n"
                    # len(), not hasattr('__len__'): under set_format('torch') a scalar
                    # column batches to a 1-D tensor whose element is 0-dim, which HAS
                    # __len__ and raises on it. The later len(_v) then threw TypeError,
                    # the outer catch restored the overlength dataset, and a truncatable
                    # run died on "cannot be enforced".
                    "            try:    len(_first)\n"
                    "            except Exception: return False\n"
                    "            return True\n"
                    # Per-token columns only, matched by row length against `input_ids`.
                    # A packed split carries `seq_lengths` -- document lengths, not tokens
                    # -- and slicing that by the cap left it stale, so padding-free built
                    # position ids for more tokens than the row now holds.
                    # Per VALUE, because `_unsloth_is_sequence_column` judges the
                    # column from its first row. An optional field that is a list
                    # there and None (or a scalar) further in raised TypeError out
                    # of `len`, the enclosing handler restored the overlength split,
                    # and a truncatable run died on "cannot be enforced". The late
                    # cap validates each row for the same reason.
                    "        def _unsloth_cut_value(_v, _r):\n"
                    "            try:\n"
                    "                if len(_v) != len(_r): return _v\n"
                    "            except Exception: return _v\n"
                    "            return _v[_unsloth_slice]\n"
                    "        def _unsloth_truncate_rows(_batch):\n"
                    "            _ids = _batch.get('input_ids')\n"
                    "            _out = {}\n"
                    "            for _k, _col in _batch.items():\n"
                    "                if not _unsloth_is_sequence_column(_col):\n"
                    "                    _out[_k] = _col\n"
                    "                elif _ids is None:\n"
                    "                    _out[_k] = [_v[_unsloth_slice] for _v in _col]\n"
                    "                else:\n"
                    "                    _out[_k] = [_unsloth_cut_value(_v, _r) for _v, _r in zip(_col, _ids)]\n"
                    "            return _out\n"
                    # A stream has no length, and `IterableDataset.map` takes no `num_proc`:
                    # passing one raised TypeError, the catch below restored the original,
                    # and the run died on "cannot be enforced" instead of being truncated.
                    "        def _unsloth_is_stream(_ds):\n"
                    "            try:    return not hasattr(_ds, '__len__')\n"
                    "            except Exception: return True\n"
                    # One split, capped and then checked. A stream's map is lazy and applies
                    # to EVERY row it will ever yield, which is a stronger guarantee than the
                    # bounded prefix scan: reading 1024 rows and calling the rest verified let
                    # a later overlength row through, since nothing else truncates a
                    # pre-tokenized stream.
                    # Enforcement, not observation. A `with_transform` split that happens
                    # to sit under the cap is not enforced -- it rebuilds its rows on every
                    # read -- so it keeps the old answer: hold `max_length` and turn
                    # padding-free off. Only a split we rewrote, or a raw one the tokenizer
                    # pass will cut, counts.
                    # Schema first, and a row only when one is free. `next(iter(_ds))`
                    # on a single-pass stream is a row the run then trains without:
                    # if it reads raw the split is declared safe and training starts
                    # at row 2, if it reads tokenized construction rejects a stream
                    # the caller still owns and has already been mutated. The
                    # `iter(x) is iter(x)` test is the one the cap scan and the
                    # schema probe already use; an unprobeable stream answers True,
                    # which holds `max_length` rather than clearing it on a guess.
                    "        def _unsloth_pretokenized(_ds):\n"
                    "            try:\n"
                    # Columns first, EXCEPT for a transform: a `with_transform` split reports
                    # its backing table, so one storing `text` and yielding overlength
                    # `input_ids` answered "raw" here. `_unsloth_truncatable` already refuses
                    # to rewrite it, and this answer then marked it safe anyway and cleared
                    # `max_length`, leaving padding-free with rows nothing truncates. Its rows
                    # are rebuilt on every read, so probing one below costs nothing.
                    "                _cols = None if _unsloth_is_transformed(_ds) else getattr(_ds, 'column_names', None)\n"
                    "                if isinstance(_cols, dict):\n"
                    "                    _cols = [_c for _v in _cols.values() for _c in (_v or [])]\n"
                    "                if _cols is not None: return 'input_ids' in _cols\n"
                    "                _probe = iter(_ds)\n"
                    "                if _probe is _ds or _probe is iter(_ds): return True\n"
                    "                _row = next(_probe, None)\n"
                    "            except Exception: return True\n"
                    "            return isinstance(_row, dict) and 'input_ids' in _row\n"
                    # Every rank reaches this before TRL's `_prepare_dataset`, and
                    # TRL runs its own preparation maps under `main_process_first`.
                    # Without the same window, eight ranks each start `num_proc`
                    # workers against one Arrow cache -- 64 processes doing the same
                    # work, and writing the same cache files at the same time. One
                    # rank materialises, the rest read what it wrote. A single
                    # process gets a no-op context manager, so nothing changes there.
                    "        def _unsloth_rank_first():\n"
                    "            try:\n"
                    "                from accelerate import PartialState\n"
                    "                return PartialState().main_process_first()\n"
                    "            except Exception:\n"
                    "                import contextlib\n"
                    "                return contextlib.nullcontext()\n"
                    "        def _unsloth_cap_split(_ds):\n"
                    "            with _unsloth_rank_first():\n"
                    "                return _unsloth_cap_one(_ds)\n"
                    "        def _unsloth_cap_one(_ds):\n"
                    "            if _ds is None: return _ds, True\n"
                    "            if not _unsloth_truncatable(_ds): return _ds, not _unsloth_pretokenized(_ds)\n"
                    "            _kw = {} if _unsloth_is_stream(_ds) else _unsloth_map_kw\n"
                    "            _new = _ds.map(_unsloth_truncate_rows, batched = True, **_kw)\n"
                    # TRL filters these immediately after truncating: a row whose
                    # prompt alone fills the cap has every label at -100 and
                    # contributes no loss, so leaving them in feeds batches with no
                    # supervised tokens.
                    # `labels` is only one of the three ways a row says which tokens
                    # are supervised. A completion-only or assistant-only row carries
                    # `completion_mask` / `assistant_masks` instead, and truncating a
                    # long prompt can leave that mask all zeros; TRL's collator turns
                    # those into all -100 and a batch made of them has no supervised
                    # token at all, which is a NaN loss rather than a small one.
                    # A mask is supervision when TRL will actually apply it, and the
                    # two masks are not symmetric there. DataCollatorForLanguageModeling
                    # guards `completion_mask` behind `self.completion_only_loss` but
                    # applies `assistant_masks` on presence alone:
                    #     if self.completion_only_loss and "completion_mask" in examples[0]:
                    #     if "assistant_masks" in examples[0]:
                    # (trl/trainer/sft_trainer.py, checked on 0.25.1 and v1.9.2). TRL only
                    # ever BUILDS that column under `assistant_only_loss`, which is why the
                    # flag looks like the gate, but a pre-tokenized split -- the only kind
                    # this branch handles -- carries whichever columns the caller put there.
                    # So gating on the flag left an all-zero assistant mask in place, and
                    # the collator turned the row into all -100: no supervised token, and a
                    # batch of them is a NaN loss.
                    # A None `completion_only_loss` is NOT "on". TRL resolves it from
                    # the dataset shape:
                    #     if args.completion_only_loss is None:
                    #         self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample
                    # (sft_trainer.py:736). A pre-tokenized split has neither column, so
                    # the effective mode is False and the collator ignores the mask
                    # entirely; treating None as enabled deleted rows that still had
                    # valid full-sequence supervision, and could empty the split.
                    "            _unsloth_cols = getattr(_new, 'column_names', None) or ()\n"
                    # One mode for every split, resolved from the TRAIN sample, because that is
                    # what the collator uses:
                    #     dataset_sample = next(iter(train_dataset))
                    #     if args.completion_only_loss is None:
                    #         self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample
                    # (sft_trainer.py). Resolving it per split disagreed with the collator whenever
                    # the schemas differ: prompt/completion training data makes the collator apply
                    # `completion_mask`, while a pre-tokenized eval split carrying only
                    # `input_ids` and `completion_mask` read here as full-sequence loss, so rows
                    # whose mask truncated to all zeros survived and went all -100 at eval.
                    "            _unsloth_masks = []\n"
                    "            if _unsloth_completion_only and 'completion_mask' in _unsloth_cols:\n"
                    "                _unsloth_masks.append('completion_mask')\n"
                    "            if 'assistant_masks' in _unsloth_cols:\n"
                    "                _unsloth_masks.append('assistant_masks')\n"
                    "            try:\n"
                    # `labels` is unconditional: it IS the supervision.
                    # One intersection over labels AND every active mask, not a filter each.
                    # The collator applies the masks ONTO the labels, so a row whose only
                    # supervised label sits where the mask is zero passes both filters
                    # separately and still goes out all -100.
                    "                _unsloth_supervision = (['labels'] if 'labels' in _unsloth_cols else []) + _unsloth_masks\n"
                    # The masks are applied one after another onto the same labels, so
                    # what survives is their INTERSECTION. Filtering each on its own kept
                    # rows whose two masks light up in different positions, which TRL then
                    # labels all -100 -- the very rows this filter exists to drop. zip
                    # stops at the shorter, which is what an intersection means for a
                    # ragged pair.
                    "                if _unsloth_supervision:\n"
                    "                    _new = _new.filter(lambda _e, _c = tuple(_unsloth_supervision): any(all((_x != -100) if _n == 'labels' else _x for _n, _x in zip(_c, _v)) for _v in zip(*[_e[_n] for _n in _c])), **_kw)\n"
                    "            except Exception:\n"
                    "                pass\n"
                    # Recorded, not raised: the caller wraps these calls in a broad
                    # `except Exception` that would turn a raise into "could not
                    # truncate". The raise happens past that handler.
                    "            try:\n"
                    "                if _unsloth_supervision and len(_new) == 0: _unsloth_emptied.append(1)\n"
                    "            except TypeError:\n"
                    "                pass\n"
                    "            return _new, (True if _unsloth_is_stream(_new) else _unsloth_within_cap(_new))\n"
                    # Resolved BEFORE the try, because the fallback below needs it even when
                    # the try never ran. `eval_packing` is separate from `packing`:
                    #     packing = args.packing if args.eval_packing is None else args.eval_packing
                    # (sft_trainer.py), so packing = False with eval_packing = True reaches this
                    # branch, which is gated on `not args.packing`. TRL then PACKS the eval split
                    # rather than truncating it, and every strategy owns the overflow: `wrapped`
                    # concatenates the stream before chunking, `bfd_split` splits an overlength
                    # example into more chunks. Cutting rows at the cap first throws that away.
                    # So the packer keeps the split, and the enforcement claim is dropped rather
                    # than the tokens: `max_length` stays and padding-free turns off. That is
                    # required anyway, since packing raises on a None `max_length`, and it has to
                    # hold even with no eval split at construction, because one handed to
                    # `evaluate()` later would find `max_length` already cleared.
                    "        _unsloth_emptied = []\n"
                    "        _unsloth_orig_train = train_dataset\n"
                    "        _unsloth_orig_eval = eval_dataset if 'eval_dataset' in locals() else None\n"
                    "        try:\n"
                    "            _unsloth_capped = _unsloth_known_mode\n"
                    "            if not _unsloth_known_mode:\n"
                    "                print('Unsloth: `truncation_mode = ' + str(_unsloth_truncation_mode) + '` is not one of keep_start / keep_end, so `max_length` is not being enforced here.')\n"
                    # A raw train split is tokenized with the cap by prep, so leave it alone.
                    # `and`, like the eval splits below. A plain assignment threw away the
                    # unknown-mode refusal seeded above, so a `truncation_mode` this cannot
                    # honour was warned about and then silently served as keep_start, with
                    # `max_length` cleared and padding-free left on.
                    # `_unsloth_known_mode` too: seeding `_unsloth_capped` false only drops the
                    # ENFORCEMENT claim. The slice still ran, with `_unsloth_keep_end` false for
                    # any unknown value, so the fallback scanned an already-trimmed split, found
                    # it within the cap and merely turned padding-free off -- every row silently
                    # cut from the start right after warning that the mode could not be honoured.
                    # Leaving the split alone lets that scan see the real lengths and say so.
                    "            if _unsloth_known_mode and not _unsloth_prep_truncates:\n"
                    "                train_dataset, _unsloth_split_ok = _unsloth_cap_split(train_dataset)\n"
                    "                _unsloth_capped = _unsloth_capped and _unsloth_split_ok\n"
                    # An eval split TRL will PACK must not be truncated first. The branch
                    # is gated on `not args.packing`, but `eval_packing` is resolved
                    # separately:
                    #     packing = args.packing if args.eval_packing is None else args.eval_packing
                    # (sft_trainer.py), so packing = False with eval_packing = True reaches
                    # here. TRL then packs the eval split instead of truncating it
                    # (`if packing: ... elif args.max_length is not None: truncate_dataset`),
                    # and the wrapped strategy concatenates the whole token stream before
                    # chunking it, so cutting each row at the cap first throws away every
                    # token past it and evaluates on a truncated corpus.
                    # Leaving it uncut also means the cap is not enforced for that split, and
                    # packing needs `max_length` anyway ("When packing is enabled,
                    # `max_length` can't be `None`"), so this drops the enforcement claim
                    # rather than the split: `max_length` stays and padding-free turns off,
                    # which is the same answer this block already gives for a split it
                    # cannot rewrite.
                    # Each eval split on its own: a raw one stays raw for the tokenizer pass
                    # that follows, and only a materialised tokenized one is cut.
                    "            if _unsloth_eval_packing or not _unsloth_known_mode:\n"
                    "                _unsloth_capped = False\n"
                    "            elif 'eval_dataset' in locals() and eval_dataset is not None:\n"
                    "                if isinstance(eval_dataset, dict):\n"
                    "                    _unsloth_new_eval = {}\n"
                    "                    for _k, _v in eval_dataset.items():\n"
                    "                        _unsloth_new_eval[_k], _unsloth_split_ok = _unsloth_cap_split(_v)\n"
                    "                        _unsloth_capped = _unsloth_capped and _unsloth_split_ok\n"
                    "                    eval_dataset = _unsloth_new_eval\n"
                    "                else:\n"
                    "                    eval_dataset, _unsloth_split_ok = _unsloth_cap_split(eval_dataset)\n"
                    "                    _unsloth_capped = _unsloth_capped and _unsloth_split_ok\n"
                    "            _unsloth_prep_truncates = _unsloth_capped\n"
                    # The splits that WERE rewritten keep their truncation: it is the cap
                    # the caller asked for, applied exactly as TRL's own truncate_dataset
                    # would. Rolling them back because a different split cannot be
                    # rewritten put an overlength train set back and turned a healthy run
                    # into "cannot be enforced". Only the claim of enforcement is dropped.
                    "            if not _unsloth_capped:\n"
                    "                print('Unsloth: `max_length` cannot be enforced for every split here, so padding-free batching is being turned off instead.')\n"
                    "        except Exception as _unsloth_truncate_error:\n"
                    "            train_dataset = _unsloth_orig_train\n"
                    "            if 'eval_dataset' in locals(): eval_dataset = _unsloth_orig_eval\n"
                    # The flag is decided from the train split, so a failure while capping
                    # an eval split would otherwise leave it reading "cap enforced".
                    "            _unsloth_prep_truncates = False\n"
                    # Never silent: a swallowed failure here reads as the cap being enforced.
                    "            print('Unsloth: could not truncate the pre-tokenized dataset to `max_length` (' + str(_unsloth_truncate_error) + ').')\n"
                    # Outside the handler on purpose. Every row losing its supervised
                    # tokens means the cap sits below where the supervision starts, and
                    # an empty split is not something to hand onwards: every TRL 1.x
                    # reads `next(iter(train_dataset))` in `__init__` to resolve
                    # `completion_only_loss` and `_is_vision_dataset`, so it comes out
                    # as a bare `StopIteration` naming nothing at all.
                    "        if _unsloth_emptied:\n"
                    "            raise ValueError('Unsloth: truncating to `max_length = ' + str(args.max_length) + '` left every row with no supervised token, so there is nothing to train on. The supervised part of your rows starts past that length: raise `max_length`, or set `truncation_mode = \"keep_end\"` if the completion sits at the end of each row.')\n"
                    "    if _unsloth_prep_truncates:\n"
                    "        args.max_seq_length = args.max_length\n"
                    "        args.max_length = None\n"
                    "        max_length = None\n"
                    "    else:\n"
                    # Turning padding-free off keeps `max_length` for TRL's collator, and that
                    # collator does not truncate: for rows that already carry `input_ids`
                    # nothing downstream enforces the cap. Before this block existed TRL's own
                    # guard made the same configuration a hard error, so an observed overlength
                    # row must stay one rather than become a silently uncapped run.
                    # `skip_prepare_dataset` used to exempt this, which made it the one
                    # way to get a silently uncapped run: TRL then neither truncates nor
                    # builds its collator with a truncation length, so the oversized rows
                    # reach the model. The check is what decides, not the flag.
                    # An eval split left for the packer is overlength ON PURPOSE, so scanning it
                    # here turned a working eval-packing run into a hard error and denied
                    # `wrapped` / `bfd_split` the overflow they exist to handle. The train split
                    # is still scanned: nothing packs that one.
                    "        _unsloth_scan_eval = None if _unsloth_eval_packing else (eval_dataset if 'eval_dataset' in locals() else None)\n"
                    "        if not (_unsloth_within_cap(train_dataset) and _unsloth_splits_within_cap(_unsloth_scan_eval)):\n"
                    "            raise ValueError('Unsloth: `max_length = ' + str(args.max_length) + '` cannot be enforced. Your dataset already carries `input_ids` and holds rows longer than that, and nothing downstream truncates pre-tokenized rows. Truncate it yourself before passing it in, or drop `max_length`.')\n"
                    "        print('Unsloth: Turning padding-free batching off, since your dataset is already tokenized and cannot be truncated here. Padding-free batching cannot enforce a `max_length` of ' + str(args.max_length) + '.')\n"
                    "        args.padding_free = False\n"
                )
            extra_args += max_length_check

    # Enable for training and move padding side of tokenizer to right
    if "model" in call_args:
        training_check = (
            "if model is not None and hasattr(model, 'for_training'):\n"
            "    _use_gc = model._unsloth_gradient_checkpointing if hasattr(model, '_unsloth_gradient_checkpointing') else getattr(args, 'gradient_checkpointing', True)\n"
            "    model.for_training(use_gradient_checkpointing=_use_gc)\n"
            "if 'tokenizer' in locals() and hasattr(tokenizer, 'padding_side'): tokenizer.padding_side = 'right'\n"
            "if 'processing_class' in locals():\n"
            "    if hasattr(processing_class, 'padding_side'): processing_class.padding_side = 'right'\n"
            "    if hasattr(processing_class, 'tokenizer') and hasattr(processing_class.tokenizer, 'padding_side'): "
            "processing_class.tokenizer.padding_side = 'right'\n"
        )
        extra_args += training_check

    # Check data collator if it's correct!
    if "data_collator" in call_args and "train_dataset" in call_args:
        data_collator_check = (
            "__tokenizer = processing_class if 'processing_class' in locals() else tokenizer\n"
            "from unsloth_zoo.vision_utils import UnslothVisionDataCollator\n"
            "if not isinstance(data_collator, UnslothVisionDataCollator):\n"
            "    if isinstance(data_collator, DataCollatorForSeq2Seq) and 'labels' not in train_dataset.column_names:\n"
            "        data_collator = TransformersDataCollatorForLanguageModeling(\n"
            "            __tokenizer,\n"
            "            mlm = False,\n"
            "            mlm_probability = 0.0,\n"
            "            pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),\n"
            "        )\n"
            "    elif isinstance(data_collator, TransformersDataCollatorForLanguageModeling) and 'labels' in train_dataset.column_names:\n"
            "        data_collator = DataCollatorForSeq2Seq(\n"
            "            __tokenizer,\n"
            "            pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),\n"
            "        )\n"
            "else:\n"
            "    if hasattr(args, 'remove_unused_columns'): args.remove_unused_columns = False\n"
            "    if hasattr(args, 'dataset_text_field'): args.dataset_text_field = ''\n"
            "    if hasattr(args, 'dataset_kwargs'): args.dataset_kwargs = {'skip_prepare_dataset': True}\n"
        )
        extra_args += data_collator_check

        # Also check if .pad exists -> if not, and is VLM, then change it!
        # Only swap LM/Seq2Seq collators; leave preference collators
        # (DPODataCollatorWithPadding etc.) alone so ORPO/DPO/CPO/KTO keep
        # their own prompt/chosen/rejected handling.
        pad_check = (
            "if not isinstance(data_collator, UnslothVisionDataCollator):\n"
            "    if not hasattr(__tokenizer, 'pad') and hasattr(__tokenizer, 'tokenizer'):\n"
            "        if isinstance(data_collator, DataCollatorForSeq2Seq):\n"
            "            data_collator = DataCollatorForSeq2Seq(\n"
            "                __tokenizer.tokenizer,\n"
            "                pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),\n"
            "            )\n"
            "        elif isinstance(data_collator, TransformersDataCollatorForLanguageModeling):\n"
            "            data_collator = TransformersDataCollatorForLanguageModeling(\n"
            "                __tokenizer.tokenizer,\n"
            "                mlm = False,\n"
            "                mlm_probability = 0.0,\n"
            "                pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),\n"
            "            )\n"
        )
        extra_args += pad_check

    # Check NEFTune
    if "model" in call_args:
        neftune_check = (
            "if hasattr(self, 'neftune_hook_handle'):\n"
            "    self.neftune_hook_handle.remove()\n"
            "    if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle\n"
            "if getattr(args, 'neftune_noise_alpha', None) is not None:\n"
            "    model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha\n"
            "pass\n"
        )
        RLTrainer_post += neftune_check

    # Add accelerator scaler to model
    if "model" in call_args:
        accelerator_check = (
            "if hasattr(self, 'accelerator'):\n"
            "    scaler = self.accelerator.scaler\n"
            "    current_model = model\n"
            "    while hasattr(current_model, 'model'):\n"
            "        current_model.accelerator_scaler = scaler\n"
            "        current_model = current_model.model\n"
            "    current_model.accelerator_scaler = scaler\n"
            "pass\n"
        )
        RLTrainer_post += accelerator_check

    # Add enabling and disabling training modes
    if "model" in call_args:
        training_check = (
            "if hasattr(self, 'train'):\n"
            "    self.train = MethodType(prepare_for_training_mode(self.__class__.train), self)\n"
            "pass\n"
        )
        RLTrainer_post += training_check

    # Sync chat_template from processing_class to vLLM's tokenizer
    # This fixes base models that have custom chat templates applied after loading
    if "model" in call_args:
        vllm_chat_template_sync = (
            "if hasattr(self, 'llm') and self.llm is not None and hasattr(self.llm, 'get_tokenizer'):\n"
            "    _vllm_tok = self.llm.get_tokenizer()\n"
            "    _pc = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)\n"
            "    if _vllm_tok is not None and _pc is not None and getattr(_pc, 'chat_template', None) is not None and getattr(_vllm_tok, 'chat_template', None) is None:\n"
            "        _vllm_tok.chat_template = _pc.chat_template\n"
            "pass\n"
        )
        RLTrainer_post += vllm_chat_template_sync

    # Edit optional metrics
    other_metrics_processor = ""
    if trainer_file in RL_METRICS_CHANGES:
        process_extra_args = RL_METRICS_CHANGES[trainer_file]
        for process_extra_arg in process_extra_args:
            other_metrics_processor += process_extra_arg(old_RLTrainer_source, old_RLConfig_source)

    # Add statistics as well!
    extra_args += (
        "other_metrics = []\n"
        f"{other_metrics_processor}\n"
        "from unsloth_zoo.logging_utils import PatchRLStatistics\n"
        f"PatchRLStatistics('{trainer_file}', other_metrics)\n"
    )

    # Patch optional args
    if trainer_file in RL_EXTRA_ARGS:
        process_extra_args = RL_EXTRA_ARGS[trainer_file]
        for process_extra_arg in process_extra_args:
            extra_args += process_extra_arg(call_args, extra_args)

    # Create RLTrainer args
    extra_args = extra_args.split("\n")
    extra_args = "\n".join(" " * 8 + x for x in extra_args)
    RLTrainer_post = RLTrainer_post.split("\n")
    RLTrainer_post = "\n".join(" " * 8 + x for x in RLTrainer_post)
    RLTrainer_arguments = arguments
    RLTrainer_extra_args = extra_args
    RLTrainer_call_args = call_args

    # Fix RLConfig next
    arguments, call_args = processed[1]
    extra_args = ""

    # Edit GA / bsz and weight_decay
    replacements = {
        "output_dir": None,
        "logging_nan_inf_filter": False,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        # LoRA decays A and B toward 0 so effective W = W_init + (alpha/r) * B @ A is pulled toward W_init, not 0 as in full FT.
        # 0.001 keeps a small Frobenius prior |A|_F^2 + |B|_F^2 without measurably dragging the merged adapter back to base.
        "weight_decay": 0.001,
        "seed": 3407,
        "optim": "adamw_8bit",
        "learning_rate": 5e-05,
        "per_device_eval_batch_size": 4,
        "eval_accumulation_steps": 2,
        "torch_empty_cache_steps": 250,
        "logging_steps": 1,
        "max_seq_length": None,
        "num_generations": 8,
        # "steps_per_generation"          : 1, # Otherwise defaults to ga_steps which is wrong
        # "generation_batch_size"         : None, # Useless. If steps_per_generation set, generation_batch_size clashes
        "top_k": None,
        "vllm_mode": "colocate",
        "generation_kwargs": {},
        "bf16": False,
        "fp16": False,
        "report_to": "none",
        "include_tokens_per_second": False,
        "include_num_input_tokens_seen": False,
        "auto_find_batch_size": False,  # Auto /2 batch size - too many people complained so removing
        "dataloader_pin_memory": True,
        "padding_free": None,  # None = user didn't set it, allows auto-enable detection
        # Might fail so disable for now
        # "dataloader_persistent_workers" : True, # Keeps dataloader in RAM
        # "dataloader_prefetch_factor"    : 2,
        # "dataloader_num_workers"        : 2, # Default is 0 means 1
    }
    # warmup_ratio deprecated in transformers >= 5.0; warmup_steps accepts float
    if transformers_version >= Version("5.0.0"):
        replacements["warmup_steps"] = 0.1
    else:
        replacements["warmup_ratio"] = 0.1

    for k, v in replacements.items():
        x = f"{k}( = [^,\n]{{1,}})?,\n"
        y = f"'{v}'" if type(v) is str else f"{v}"
        y = f"{k} = {y},\n"
        arguments = re.sub(x, y, arguments)

    # Fix GRPO beta default as 0.001 TRL used to be 0.04, now 0.00!
    # https://github.com/huggingface/trl/pull/3516
    # https://verl.readthedocs.io/en/latest/examples/config.html
    if trainer_file == "grpo_trainer":
        replacements = {
            "loss_type": "bnpo",  # Default GRPO paper
            "beta": 0.001,  # Recommended as seen in verl
            "auto_find_batch_size": False,  # Cannot work on GRPO
            # [TODO] See https://fengyao.notion.site/off-policy-rl
            # https://github.com/huggingface/trl/pull/3867 (August 7th)
            "vllm_importance_sampling_correction": False,
            # TRL >= 1.7.0 enables the MoE router aux loss by default (0.001); the optimized
            # GRPO forward does not compute it, so default off. Opt in via router_aux_loss_coef > 0.
            "router_aux_loss_coef": 0.0,
        }
        for k, v in replacements.items():
            x = f"{k}( = [^,\n]{{1,}})?,\n"
            y = f"'{v}'" if type(v) is str else f"{v}"
            y = f"{k} = {y},\n"
            arguments = re.sub(x, y, arguments)

    # Warn on too large or too small learning rate
    if "learning_rate" in call_args:
        learning_rate_check = (
            "if learning_rate < 1e-7: print(f'Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! "
            "Consider increasing it, otherwise gradient updates will be close to 0!')\n"
            "if learning_rate > 1: print(f'Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! "
            "Consider decreasing it to 1e-1, otherwise gradient updates will explode!')\n"
        )
        extra_args += learning_rate_check

    # Fix num_train_epochs = None causing TypeError in Trainer.__init__
    # Trainer does `args.num_train_epochs > 0` which fails when None
    if "num_train_epochs" in call_args:
        num_train_epochs_check = (
            "if num_train_epochs is None:\n"
            "    num_train_epochs = 3.0  # Default to 3 epochs if None, max_steps will override\n"
        )
        extra_args += num_train_epochs_check

    # Check if max_seq_length is NOT defined (max_length is now default)
    if "max_seq_length" not in call_args and "max_length" in call_args:
        max_seq_length_pre = """max_seq_length : Optional[int] = field(
        default = None,
        metadata = {'help': 'Maximum sequence length to truncate to.'},
    )"""
        max_seq_length_call = "max_seq_length = None,"
        max_seq_length_post = "self.max_seq_length = max_seq_length"
    else:
        max_seq_length_pre = ""
        max_seq_length_call = ""
        max_seq_length_post = ""

    # Add output_dir saving
    if "output_dir" in call_args:
        # Default checks
        saving_check = (
            "if output_dir is None and save_strategy == 'steps' and save_steps == 500:\n"
            "    output_dir = 'unsloth_training_checkpoints'\n"
            "    save_strategy = 'no'\n"
        )
        extra_args += saving_check

    # Edit dataset_num_proc
    # The policy lives in unsloth_zoo.dataset_num_proc: it had drifted into four
    # inline copies, two wrong (stdlib `multiprocessing` asked about a start
    # method `datasets` takes from `multiprocess`, and `1` used as the serial
    # sentinel when datasets >= 4.1 builds a Pool(1) for it). The zoo rather than
    # unsloth, so generated source never imports back into its generator;
    # unsloth.dataset_num_proc is the fallback for an older zoo, and the
    # ladder is guarded so a stale generated file degrades to the caller's value.
    # serial_as_none depends on who reads the value back. Only SFT has a
    # downstream auto-sizer: unsloth_zoo.sft_prepare_dataset reads a config
    # `None` as "auto-size me", so serial has to be written as `1` there and the
    # map() call site (rl_replacements.py) turns it back into None. DPO, KTO,
    # CPO, ORPO, Reward and PRM hand args.dataset_num_proc straight to
    # Dataset.map, where nothing can inflate a None but a `1` is a Pool(1) on
    # datasets >= 4.1 -- one worker with its own tokenizer copy, on a host the
    # memory clamp had just declared too small for workers.
    if "dataset_num_proc" in call_args:
        _serial_as_none = "False" if trainer_file == "sft_trainer" else "True"
        num_proc_check = (
            "try:\n"
            "    from unsloth_zoo.dataset_num_proc import get_dataset_num_proc as _unsloth_get_dataset_num_proc\n"
            "except Exception:\n"
            "    try:\n"
            "        from unsloth.dataset_num_proc import get_dataset_num_proc as _unsloth_get_dataset_num_proc\n"
            "    except Exception:\n"
            "        _unsloth_get_dataset_num_proc = None\n"
            "if _unsloth_get_dataset_num_proc is not None:\n"
            "    dataset_num_proc = _unsloth_get_dataset_num_proc("
            f"dataset_num_proc, serial_as_none = {_serial_as_none})\n"
        )
        extra_args += num_proc_check

    # Add padding if flex attention is added
    if "pad_to_multiple_of" in call_args:
        pad_to_multiple_of = (
            "if os.environ.get('UNSLOTH_ENABLE_FLEX_ATTENTION', '0') == '1':\n"
            "    from unsloth_zoo.flex_attention import HAS_FLEX_ATTENTION\n"
            "    if HAS_FLEX_ATTENTION and pad_to_multiple_of is None:\n"
            "        from unsloth_zoo.flex_attention import FLEX_ATTENTION_BLOCK_SIZE\n"
            "        pad_to_multiple_of = FLEX_ATTENTION_BLOCK_SIZE\n"
            "\n"
        )
        extra_args += pad_to_multiple_of

    # Check for loss_type = dr_grpo and scale_rewards for GRPO
    if "loss_type" in call_args and "scale_rewards" in call_args:
        # See https://github.com/huggingface/trl/issues/3130#issuecomment-2746947835
        # DAPO uses per token loss so BNPO loss used
        check_dr_grpo = (
            "if loss_type.lower() == 'dr_grpo':\n"
            "    loss_type = 'dr_grpo'\n"
            "elif loss_type.lower() == 'dapo':\n"
            "    loss_type = 'dapo'\n"
            "if loss_type.lower() == 'dr_grpo':\n"
            "    if scale_rewards == None:\n"
            "        scale_rewards = True\n"
            "    elif scale_rewards == True:\n"
            "        print('Unsloth: The Dr GRPO paper recommends setting `scale_rewards` to False! Will override. Set it to `None` to force False.')\n"
            "        scale_rewards = False\n"
            "elif loss_type.lower() == 'dapo':\n"
            "    if mask_truncated_completions != True:\n"
            "        print('Unsloth: The DAPO paper recommends `mask_truncated_completions = True` - we will set it.')\n"
            "    if epsilon_high != 0.28:\n"
            "        print('Unsloth: The DAPO paper recommends `epsilon_high = 0.28` - we will set it.')\n"
            "    if beta != 0.0:\n"
            "        print(f'[WARNING] Unsloth: The DAPO paper recommends setting `beta = 0.0` to remove the KL term - You have set it to {beta}.')\n"
            "    mask_truncated_completions = True\n"
            "    epsilon_high = 0.28\n"
            "\n"
        )
        extra_args += check_dr_grpo

    # Check GRPO num_generations mismatch
    if (
        "per_device_train_batch_size" in call_args
        and "num_generations" in call_args
        and "steps_per_generation" in call_args
        and "generation_batch_size" in call_args
    ):
        # if world size is not set by accelerate or torchrun at this point it will be 1
        check_num_generations = (
            "if steps_per_generation is None and generation_batch_size is None:\n"
            "    ga = gradient_accumulation_steps\n"
            "    world_size = int(os.environ.get('WORLD_SIZE', '1'))\n"
            "    if (ga * world_size * per_device_train_batch_size) % num_generations != 0:\n"
            "        print('Unsloth: We now expect `per_device_train_batch_size` * `gradient_accumulation_steps` * `world_size` to be a multiple of `num_generations`.\\n"
            "We will change the batch size of ' + str(per_device_train_batch_size) + ' to the `num_generations` of ' + str(num_generations))\n"
            "        per_device_train_batch_size = num_generations\n"
            "\n"
        )
        extra_args += check_num_generations
    elif "per_device_train_batch_size" in call_args and "num_generations" in call_args:
        if "steps_per_generation" not in call_args:
            print(f"Unsloth: Could not find `steps_per_generation` in {trainer_file}")
        if "generation_batch_size" not in call_args:
            print(f"Unsloth: Could not find `generation_batch_size` in {trainer_file}")

        check_num_generations = (
            "if (per_device_train_batch_size // num_generations) * num_generations != per_device_train_batch_size:\n"
            "    print('Unsloth: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.\\n"
            "We will change the batch size of ' + str(per_device_train_batch_size) + ' to the `num_generations` of ' + str(num_generations))\n"
            "    per_device_train_batch_size = num_generations\n"
            "\n"
        )
        extra_args += check_num_generations

    # Check temperature must not be <= 0. Also stop if >= 10
    if "temperature" in call_args:
        check_temperature = (
            "if temperature <= 0:\n"
            "    raise ValueError('Unsloth: Please set a positive non-zero temperature since your results will be wrong.')\n"
            "elif temperature >= 10:\n"
            "    raise ValueError('Unsloth: Please set a positive non-zero temperature less than 10, since sampling will be quite erratic.')\n"
            "\n"
        )
        extra_args += check_temperature

    # Edit config with anything extra
    if trainer_file in RL_CONFIG_CHANGES:
        process_extra_args = RL_CONFIG_CHANGES[trainer_file]
        for process_extra_arg in process_extra_args:
            extra_args += process_extra_arg(old_RLTrainer_source, old_RLConfig_source)

    # Create RLConfig args
    extra_args = extra_args.split("\n")
    extra_args = "\n".join(" " * 8 + x for x in extra_args)
    RLConfig_arguments = arguments
    RLConfig_extra_args = extra_args
    RLConfig_call_args = call_args

    # TRL 0.27.0+ forces use_reentrant=False in gradient_checkpointing_kwargs.
    # Unsloth gradient checkpointing requires use_reentrant=True, so we remove
    # the setting after super().__init__() when it gets auto-applied.
    RLConfig_post = ""
    if trl_version >= Version("0.27.0"):
        RLConfig_post = (
            "        # Unsloth: Remove use_reentrant=False forced by TRL 0.27.0+\n"
            "        if getattr(self, 'gradient_checkpointing_kwargs', None) is not None:\n"
            "            if 'use_reentrant' in self.gradient_checkpointing_kwargs:\n"
            "                del self.gradient_checkpointing_kwargs['use_reentrant']\n"
        )

    # Patch vLLM and other functions
    RLTrainer_extras = patch_functions(
        RLTrainer, trainer_file, RLTrainer_name, all_imports, imports
    )
    if RLTrainer_extras is None:
        RLTrainer_extras = f"_Unsloth{RLTrainer_name} = {RLTrainer_name}"

    # Create full module
    exec(f"from trl.trainer import ({RLTrainer_name}, {RLConfig_name},)")
    __RLTrainer_doc__ = eval(f"trl.trainer.{RLTrainer_name}").__doc__
    if __RLTrainer_doc__ is None:
        __RLTrainer_doc__ = ""
    __RLConfig_doc__ = eval(f"trl.trainer.{RLConfig_name}").__doc__
    if __RLConfig_doc__ is None:
        __RLConfig_doc__ = ""

    # Get all pre-modules
    if trainer_file in RL_PRE_ITEMS:
        RL_pre = "\n".join(RL_PRE_ITEMS[trainer_file])
    else:
        RL_pre = ""

    # Check if SamplingParams is in there
    if "SamplingParams" in old_RLTrainer_source:
        RL_pre = RL_pre + "\n" + inspect.getsource(vLLMSamplingParams)

    # Selective log softmax and other functions
    selective_log_softmax_code = inspect.getsource(selective_log_softmax)
    grpo_selective_log_softmax_code = inspect.getsource(grpo_selective_log_softmax)
    calculate_pad_tokens_in_prompt_code = inspect.getsource(calculate_pad_tokens_in_prompt)
    create_completion_attention_mask_code = inspect.getsource(create_completion_attention_mask)
    left_pack_padding_code = inspect.getsource(left_pack_padding)
    align_logprobs_with_mask_code = inspect.getsource(align_logprobs_with_mask)
    align_completion_tool_mask_code = inspect.getsource(align_completion_tool_mask)
    autotune_batch_and_chunks_code = inspect.getsource(autotune_batch_and_chunks)
    sanitize_logprob_code = inspect.getsource(sanitize_logprob)
    # Get final source code
    RLTrainer_source = RLTrainer_replacement.format(
        RLTrainer_name = RLTrainer_name,
        __RLTrainer_doc__ = __RLTrainer_doc__,
        RLTrainer_arguments = RLTrainer_arguments,
        RLTrainer_extra_args = RLTrainer_extra_args,
        RLTrainer_call_args = RLTrainer_call_args,
        RLTrainer_kwargs = ",**kwargs"[1 if RLTrainer_call_args.endswith(",") else 0 :],
        RLConfig_name = RLConfig_name,
        __RLConfig_doc__ = __RLConfig_doc__,
        RLConfig_arguments = RLConfig_arguments,
        RLConfig_extra_args = RLConfig_extra_args,
        RLConfig_call_args = RLConfig_call_args,
        RLConfig_kwargs = ",**kwargs"[1 if RLConfig_call_args.endswith(",") else 0 :],
        RLConfig_post = RLConfig_post,
        RLTrainer_extras = RLTrainer_extras,
        RLTrainer_post = RLTrainer_post,
        RL_pre = RL_pre,
        max_seq_length_pre = max_seq_length_pre,
        max_seq_length_call = max_seq_length_call,
        max_seq_length_post = max_seq_length_post,
        selective_log_softmax_code = selective_log_softmax_code,
        grpo_selective_log_softmax_code = grpo_selective_log_softmax_code,
        calculate_pad_tokens_in_prompt_code = calculate_pad_tokens_in_prompt_code,
        create_completion_attention_mask_code = create_completion_attention_mask_code,
        autotune_batch_and_chunks_code = autotune_batch_and_chunks_code,
        left_pack_padding_code = left_pack_padding_code,
        align_logprobs_with_mask_code = align_logprobs_with_mask_code,
        align_completion_tool_mask_code = align_completion_tool_mask_code,
        sanitize_logprob_code = sanitize_logprob_code,
    )

    if RLTrainer_name == "GRPOTrainer":
        # Base torch_compile_options shared by all device types
        base_options = """torch_compile_options = {
            "epilogue_fusion"   : True,
            "max_autotune"      : False,
            "shape_padding"     : True,
            "trace.enabled"     : False,"""

        # Generate torch_compile_options based on device type
        if DEVICE_TYPE == "cuda":
            # CUDA-specific options (added to base options)
            cuda_options = """
            "triton.enable_persistent_tma_matmul": torch.cuda.get_device_capability()[0] >= 9,"""
            # cutlass options were added in PyTorch 2.8.0
            if torch_version >= Version("2.8.0"):
                cuda_options += """
            "cuda.cutlass_epilogue_fusion_enabled": torch.cuda.get_device_capability()[0] >= 9,
            "cuda.cutlass_tma_only": torch.cuda.get_device_capability()[0] >= 9,"""
            cuda_options += """
            "cuda.compile_opt_level"              : "-O2",
            "cuda.enable_cuda_lto"                : True,
        }"""
            new_options = base_options + cuda_options
        else:
            # XPU, HIP, and other device types use base options only
            new_options = (
                base_options
                + """
        }"""
            )

        pattern = r"torch_compile_options\s*=\s*\{[^}]*\}"

        RLTrainer_source = re.sub(pattern, new_options, RLTrainer_source, flags = re.DOTALL)

        if trl_version >= Version("1.4.0"):
            # The `elif is_peft_model(model) and args.beta != 0.0:` ref-adapter block
            # was introduced in TRL 1.4.0 and is used through 1.7.x. Remove only that
            # block, anchored on the final ref_param copy so we do NOT also swallow the
            # following gradient-checkpointing enable_input_require_grads() block.
            peft_pattern = (
                r"\s*elif is_peft_model\(model\) and args\.beta != 0\.0:"
                r".*?"
                r"ref_param\.data\.copy_\(param\.data\)"
            )

            replacement_comment = (
                "\n        # PEFT initialization logic removed via script for trl >= 1.4.0\n"
            )

            RLTrainer_source = re.sub(
                peft_pattern, replacement_comment, RLTrainer_source, flags = re.DOTALL
            )

            if trl_version >= Version("1.7.0"):
                # router_aux_loss_coef / aux_loss_enabled were added in TRL 1.7.0. Unsloth's
                # optimized GRPO forward cannot compute the MoE router aux loss, so reject
                # explicit opt-in (router_aux_loss_coef > 0) at init rather than silently ignoring it.
                RLTrainer_source = RLTrainer_source.replace(
                    "self.aux_loss_enabled = is_moe and args.router_aux_loss_coef != 0.0",
                    "self.aux_loss_enabled = is_moe and args.router_aux_loss_coef != 0.0\n"
                    '        if self.aux_loss_enabled: raise NotImplementedError("Unsloth GRPO does not compute the MoE router auxiliary loss; set router_aux_loss_coef = 0 (the Unsloth default).")',
                )

        elif trl_version >= Version("0.27.0"):
            peft_pattern = (
                r"\s*if is_peft_available\(\) and is_peft_model\(model\) and args\.beta != 0\.0:"
                r".*?"
                r"param\.data = param\.data\.to\(torch\.bfloat16\)"
            )

            replacement_comment = (
                "\n        # PEFT initialization logic removed via script for trl >= 0.27.0\n"
            )

            RLTrainer_source = re.sub(
                peft_pattern, replacement_comment, RLTrainer_source, flags = re.DOTALL
            )

        elif trl_version >= Version("0.26.0"):
            peft_block_pattern = (
                r"\s*if is_peft_available\(\) and isinstance\(model, PeftModel\) and peft_config is not None:"
                r".*?"
                r"param\.data = param\.data\.to\(torch\.bfloat16\)"
            )

            RLTrainer_source = re.sub(
                peft_block_pattern,
                "\n        # TRL PEFT 0.26.0 initialization logic removed on unsloth side.\n",
                RLTrainer_source,
                flags = re.DOTALL,
            )

    # Remove TRL 0.26.0's unconditional bfloat16 cast of trainable params. It
    # hardcodes bfloat16 for QLoRA, ignoring the user's dtype and breaking
    # GradScaler with fp16=True. Unsloth already handles adapter dtype via
    # patch_model_and_tokenizer, so the block is unnecessary (and already a
    # no-op for GRPO, whose peft init block is removed above).
    RLTrainer_source = RLTrainer_source.replace(
        'if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):',
        "if False:",
    )
    # TRL >= 1.7.0 spells the same QLoRA bf16 cast as `if _is_quantized_model:`.
    RLTrainer_source = RLTrainer_source.replace(
        "if _is_quantized_model:",
        "if False:",
    )

    if RLTrainer_name == "SFTTrainer":
        original_text = (
            'self._signature_columns = ["input_ids", "attention_mask", "completion_mask"]'
        )
        new_text = (
            'self._signature_columns = ["input_ids", "attention_mask", "completion_mask","labels"]'
        )
        RLTrainer_source = RLTrainer_source.replace(original_text, new_text)

        # Do NOT override _is_vlm -- let TRL detect VLM models naturally
        # (forcing _is_vlm=False errors on vision datasets in TRL 0.27.1+).
        # But some notebooks pass a bare tokenizer as processing_class, so TRL
        # sets _is_vlm=False even for VLMs; add an architecture-based override
        # before the validation check.
        _vlm_check_original = (
            '        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample\n'
            "        if self._is_vision_dataset and not self._is_vlm:"
        )
        _vlm_check_patched = (
            '        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample\n'
            "        # Unsloth: override _is_vlm for VLM models that pass a bare tokenizer\n"
            "        if not self._is_vlm and self._is_vision_dataset:\n"
            "            _m = model\n"
            '            if hasattr(_m, "model"): _m = _m.model\n'
            '            if hasattr(getattr(_m, "config", None), "vision_config") or \\\n'
            '               _m.__class__.__name__.endswith("ForConditionalGeneration"):\n'
            "                self._is_vlm = True\n"
            "        if self._is_vision_dataset and not self._is_vlm:"
        )
        if _vlm_check_original in RLTrainer_source:
            RLTrainer_source = RLTrainer_source.replace(_vlm_check_original, _vlm_check_patched)

        # TRL 0.22.x keys off _is_vlm, not _is_vision_dataset (0.24.0+), so the
        # vision-only signature columns never overlap the tokenized ones. Merge
        # both sets; _remove_unused_columns ignores extras.
        _sig_vlm_old = 'self._signature_columns = ["messages", "prompt", "completion", "images"]'
        _sig_vlm_new = (
            'self._signature_columns = ["messages", "prompt", "completion", "images",'
            ' "input_ids", "labels", "attention_mask", "seq_lengths", "completion_mask", "assistant_masks"]'
        )
        RLTrainer_source = RLTrainer_source.replace(_sig_vlm_old, _sig_vlm_new)

        RLTrainer_source = _backport_vision_dataset_gate(RLTrainer_source)

        # Inject model reference before _prepare_dataset for dynamic
        # token_type_ids detection in sft_prepare_dataset
        _prep_pattern = r"([ \t]*)train_dataset = self\._prepare_dataset\("
        _prep_replacement = (
            r"\1self._unsloth_model_ref = model\n\1train_dataset = self._prepare_dataset("
        )
        RLTrainer_source = re.sub(_prep_pattern, _prep_replacement, RLTrainer_source, count = 1)

    # Silence TRL's noisy batch_size=1 + padding-free warning (handles both
    # the original "anihilate" typo and the corrected "annihilate" spelling)
    for _typo in ("anihilate", "annihilate"):
        _idx = RLTrainer_source.find(_typo)
        if _idx == -1:
            continue
        # Walk backwards to find "if args.per_device_train_batch_size"
        _block_start = RLTrainer_source.rfind("if args.per_device_train_batch_size == 1", 0, _idx)
        if _block_start == -1:
            continue
        # Walk backwards to the newline before the if
        _line_start = RLTrainer_source.rfind("\n", 0, _block_start)
        # Walk forwards past the closing paren to the end of the block
        _close = RLTrainer_source.find(")", _idx)
        if _close == -1:
            continue
        _block_end = RLTrainer_source.find("\n", _close)
        if _block_end == -1:
            continue
        RLTrainer_source = RLTrainer_source[:_line_start] + RLTrainer_source[_block_end:]
        break

    # Remove multiple doc strings
    if __RLConfig_doc__ != "" and RLTrainer_source.count(__RLTrainer_doc__) == 2:
        RLTrainer_source = RLTrainer_source.replace(__RLTrainer_doc__, "", 1)

    # Remove multiple newlines
    RLTrainer_source = re.sub(r"[\n]{3,}", "\n", RLTrainer_source)

    # Create new function
    _resolved_module = _trainer_resolved_module or _config_resolved_module
    _model_location = (
        _resolved_module.__name__ if _resolved_module is not None else f"trl.trainer.{trainer_file}"
    )
    created_module = create_new_function(
        f"Unsloth{RLTrainer_name}",
        RLTrainer_source,
        _model_location,
        imports,
        overwrite = False,
    )
    patched_trainer = getattr(created_module, f"Unsloth{RLTrainer_name}")
    if trainer_file == "grpo_trainer":
        _patch_resume_from_checkpoint_memory(patched_trainer)

    # Patch Trainer
    exec(
        f"trl.{RLTrainer_name} = created_module.Unsloth{RLTrainer_name}",
        locals(),
        globals(),
    )
    exec(
        f"trl.trainer.{RLTrainer_name} = created_module.Unsloth{RLTrainer_name}",
        locals(),
        globals(),
    )
    exec(
        f"trl.trainer.{trainer_file}.{RLTrainer_name} = created_module.Unsloth{RLTrainer_name}",
        locals(),
        globals(),
    )

    # Patch Config
    exec(
        f"trl.{RLConfig_name} = created_module.Unsloth{RLConfig_name}",
        locals(),
        globals(),
    )
    exec(
        f"trl.trainer.{RLConfig_name} = created_module.Unsloth{RLConfig_name}",
        locals(),
        globals(),
    )
    exec(
        f"trl.trainer.{trainer_file}.{RLConfig_name} = created_module.Unsloth{RLConfig_name}",
        locals(),
        globals(),
    )
    try:
        config_module_name = trainer_file.replace("_trainer", "_config")
        config_module = importlib.import_module(f"trl.trainer.{config_module_name}")
        if hasattr(config_module, RLConfig_name):
            setattr(
                config_module,
                RLConfig_name,
                getattr(created_module, f"Unsloth{RLConfig_name}"),
            )
    except Exception:
        pass

    if trainer_file == "sft_trainer":
        try:
            _wrap_sft_evaluate_cap(getattr(created_module, f"Unsloth{RLTrainer_name}"))
        except Exception as e:
            logger.info(f"Unsloth: Could not wrap evaluate for {RLTrainer_name}: {e}")

    if trainer_file == "grpo_trainer":
        try:
            _wrap_grpo_generate_and_score(getattr(created_module, f"Unsloth{RLTrainer_name}"))
        except Exception as e:
            logger.info(
                f"Unsloth: Could not wrap _generate_and_score_completions for {RLTrainer_name}: {e}"
            )
        try:
            _wrap_grpo_hidden_states_fallback(getattr(created_module, f"Unsloth{RLTrainer_name}"))
        except Exception as e:
            logger.info(
                f"Unsloth: Could not wrap GRPO hidden-state fallback for {RLTrainer_name}: {e}"
            )


def patch_functions(RLTrainer, trainer_file, RLTrainer_name, all_imports, imports):
    init = inspect.getsource(RLTrainer.__init__)
    old_init = init

    # Remove brackets in comments since it interferes ie (...)
    comments = re.findall(r"\#[^\n]{1,}\n", init)
    bracketed_comments = [x for x in comments if "(" in x or ")" in x]
    # Replace with [...] instead
    for bracketed_comment in bracketed_comments:
        init = init.replace(
            bracketed_comment,
            bracketed_comment.replace("(", "[").replace(")", "]"),
        )

    # Remove peft_config
    init = init.replace("elif peft_config is None:", "elif False:")
    init = init.replace("elif peft_config is not None:", "elif False:")
    init = init.replace("if peft_config is None:", "if False:")
    init = init.replace("if peft_config is not None:", "if False:")
    init = init.replace("get_peft_model(model, peft_config)", "model")
    # New TRL 0.20.0
    init = init.replace(
        "if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):",
        "if False:",
    )
    # New TRL 0.20.0
    init = init.replace("model = self._prepare_peft_model(model, peft_config, args)\n", "pass\n")
    # TRL 0.22.0+ uses prepare_peft_model as a standalone function
    init = init.replace("model = prepare_peft_model(model, peft_config, args)", "pass")

    # Skip add_adapter("ref") for reference model computation
    # Unsloth: We comment out the "ref" adapter creation because:
    # 1. We want to use the original BASE MODEL as the reference model, not the SFT/LoRA model
    # 2. PEFT doesn't allow multiple adapters when target_parameters is used (MoE models)
    # When "ref" is not in peft_config, GRPO/RLOO fallback uses disable_adapter()
    # which gives the base model logits - exactly what we want
    add_adapter_block_pattern = (
        r"([ \t]*)"  # Capture leading indentation
        r"if\s+is_peft_available\(\)\s+and\s+is_peft_model\(model\)\s+and\s+args\.beta\s*!=\s*0\.0\s*:"
        r"(.*?)"  # Match the entire block until ref_param.data.copy_
        r"ref_param\.data\.copy_\(param\.data\)"
    )

    def comment_out_block(match):
        """Comment out each line in the matched block, preserving indentation."""
        full_match = match.group(0)
        indent = match.group(1)
        lines = full_match.split("\n")
        commented_lines = []
        # Add explanation comment first
        commented_lines.append(
            f"{indent}# Unsloth: Commented out - use base model as reference, not SFT/LoRA model"
        )
        # Comment out each line - insert # after leading whitespace to preserve indentation
        for line in lines:
            if line.strip():
                stripped = line.lstrip()
                leading_ws = line[: len(line) - len(stripped)]
                commented_lines.append(f"{leading_ws}# {stripped}")
            else:
                commented_lines.append(line)
        return "\n".join(commented_lines)

    init = re.sub(add_adapter_block_pattern, comment_out_block, init, flags = re.DOTALL)

    # Set use_vllm if not set
    if "args.use_vllm" in init and "model" in init and "args" in init:
        # .*? matches first match. .+? matches final match.
        replacer = re.findall(
            r"def __init__\(.*?\).*?\:\n",
            init,
            flags = re.MULTILINE | re.DOTALL,
        )
        if len(replacer) != 0:
            replacer = replacer[0]
            vllm_setter = (
                "\n"
                + " " * 8
                + "if hasattr(model, 'vllm_engine') and hasattr(args, 'use_vllm'):\n"
                + " " * 12
                + "if (getattr(args, 'use_vllm', False) == False):\n"
                + " " * 16
                + "args.use_vllm = True\n"
            )
            # " " * 16 + "args.vllm_importance_sampling_correction = True\n" + \
            # " " * 16 + "args.vllm_importance_sampling_cap = 2.0\n"

            if "grpo" in trainer_file and trl_version >= Version("0.18.0"):
                # If model has vllm_engine, then use vllm in colocate mode. Donot wait for server
                vllm_setter += " " * 12 + "args.vllm_mode='colocate'\n"
                if trl_version >= Version("0.23.0"):
                    # Align TRL sleep mode with the engine's actual enable_sleep_mode
                    # (the vision standby gate may have disabled it); fall back to the
                    # standby env var when the engine cannot be introspected.
                    vllm_setter += (
                        " " * 12
                        + "_unsloth_esm = getattr(getattr(getattr(getattr(model.vllm_engine, 'llm_engine', None), 'vllm_config', None), 'model_config', None), 'enable_sleep_mode', None)\n"
                        + " " * 12
                        + "if (_unsloth_esm if _unsloth_esm is not None else os.environ.get('UNSLOTH_VLLM_STANDBY', '0') != '0'):\n"
                        + " " * 16
                        + "args.vllm_enable_sleep_mode=True\n"
                    )

            init = init.replace(replacer, replacer + vllm_setter)

    # breakpoint()

    vllm_part = re.findall(
        r"(\n[\s]{8}" r"if (self|args)\.use_vllm\:.*?" r"\n[\s]{8}" "else:\n)",
        init,
        flags = re.MULTILINE | re.DOTALL,
    )

    if len(vllm_part) == 1:
        vllm_part, args = vllm_part[0][0], vllm_part[0][1]
        # Strip all comments
        new_vllm_part = re.sub(
            r"^\s*\#[^\n]*\n?", "", vllm_part, flags = re.MULTILINE
        )  # to also remove whole comment line instead of just starting at #
        new_vllm_part = re.sub(
            r"\s*\#.*$", "", new_vllm_part, flags = re.MULTILINE
        )  # remove comments that occur after code

        # Get SamplingParams
        sampling_params = re.findall(
            r"\n[\s]{4,}(self\.[^\s]{1,}[\s]{0,}\=[\s]{0,}SamplingParams\(.+?\))",
            new_vllm_part,
            flags = re.MULTILINE | re.DOTALL,
        )

        if len(sampling_params) == 1:
            sampling_params = sampling_params[0]
            # Fix guided_decoding
            sampling_params = sampling_params.replace(
                "guided_decoding=guided_decoding,",
                "guided_decoding="
                'GuidedDecodingParams(backend="outlines", regex=args.vllm_guided_decoding_regex) '
                'if getattr(args, "vllm_guided_decoding_regex", None) is not None else None,',
            )
            # Replace with our vLLM engine when sharing weights
            sampling_params = (
                " " * 12
                + "if getattr(getattr(model, 'vllm_engine', None), 'shared_weights', False): "
                + "self.llm = model.vllm_engine; self._last_loaded_step = 0\n"
                + " " * 12
                + sampling_params
            )

            # count the indentation of last line of sampling_params.
            splitted_sampling_params = sampling_params.split("\n")
            if len(splitted_sampling_params) >= 2:
                last_line = splitted_sampling_params[-1]
                last_prev_line = splitted_sampling_params[-2]
                last_prev_indentation = len(last_prev_line) - len(last_prev_line.lstrip())
                last_indentation = len(last_line) - len(last_line.lstrip())

                # Add extra arguments to SamplingParams
                extra = "**getattr(getattr(args, 'vllm_sampling_params', vLLMSamplingParams()), '_set_kwargs', {})"
                # Backwards replace
                to_replace = (
                    ",\n"
                    + " " * last_prev_indentation
                    + extra
                    + ",\n"
                    + " " * last_indentation
                    + ")"
                )
                sampling_params = to_replace.join(sampling_params.rsplit(")", 1))
                # Strip multiple commas
                sampling_params = re.sub(r"[\,][\s]{0,}\,", ",", sampling_params)

                new_vllm_part = (
                    f"\n{' ' * 8}if {args}.use_vllm:\n{sampling_params}\n{' ' * 8}else:\n"
                )

        if trl_version >= Version("0.18.0"):
            # Guard LLM init - use existing vLLM engine when sharing weights,
            # otherwise keep the original LLM() creation for sync/reload path
            vllm_llm_init_pattern = r"(?P<indent>[ \t]*)self\.llm\s*=\s*LLM\(.*?\)*\)\s*?\n(?!,)"

            def guard_llm_init(match):
                indent = match.group("indent")
                original = match.group(0)
                return (
                    f"{indent}if getattr(getattr(model, 'vllm_engine', None), 'shared_weights', False):\n"
                    f"{indent}    self.llm = model.vllm_engine\n"
                    f"{indent}else:\n"
                    f"{indent}    {original.lstrip()}"
                )

            new_vllm_part = re.sub(
                vllm_llm_init_pattern,
                guard_llm_init,
                new_vllm_part,
                flags = re.DOTALL,
            )

        init = init.replace(vllm_part, new_vllm_part)

    # Search for vLLM calling in all child functions
    functions = dir(RLTrainer)
    RLTrainer_source = inspect.getsource(RLTrainer)
    functions = [x for x in functions if f"def {x}" in RLTrainer_source]

    changed = {
        "__init__": (
            old_init,
            init,
        )
    }
    edit_functions = RL_FUNCTIONS.get(trainer_file, [])

    for function in functions:
        if not hasattr(RLTrainer, function):
            continue
        if function in changed:
            original_source, source = changed[function]
        else:
            fx = getattr(RLTrainer, function)
            try:
                source = inspect.getsource(fx)
            except:
                continue
            original_source = source

        # Check for function
        for edit_function in edit_functions:
            source = edit_function(function, source)

        """
        import torch
        X = torch.ones((2, 2048, 201088), dtype = torch.bfloat16, device = "cuda")
        X[torch.randperm(2, dtype = torch.int64, device = X.device)]

        will error out in torch 2.8 AcceleratorError: CUDA error: invalid configuration argument
        """
        source = re.sub(
            r"(\n[\s]{4,})generation_batch = shuffle_sequence_dict\(generation_batch\)\n",
            r"\n\1try: generation_batch = shuffle_sequence_dict(generation_batch)\n\1except: pass\n",
            source,
        )

        # llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        source = re.sub(
            r"(\n[\s]{4,}).+?model_executor\.driver_worker.+?\n",
            r"\n\1pass\n",
            source,
        )

        # llm_model.load_weights(model.state_dict().items())
        source = re.sub(
            r"(\n[\s]{4,}).+?load_weights\(.+?\n",
            r"\n\1pass\n",
            source,
        )

        # .state_dict()
        source = re.sub(
            r"\.state_dict\(\)",
            r"",
            source,
        )

        # Replace self.llm.generate and self.llm.chat with lora_request (only when sharing weights)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            lora_name = (
                trainer_file
                + "_lora_model_' + "
                + "(os.environ.get('CUDA_VISIBLE_DEVICES', '0').replace(',',''))"
            )
        else:
            lora_name = trainer_file + "_lora_model'"
        source = re.sub(
            r"(self\.llm\.(?:generate|chat)\([^\)]{1,})\)",
            r"\1, lora_request = self.model.load_lora('"
            + lora_name
            + r", load_tensors = True)"
            + r" if getattr(self.llm, 'shared_weights', False)"
            + r" else None)",
            source,
        )
        # All these are to fix multiple commas before lora_request (in case the original code ends with something like ",)")
        # https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L1388 for eg has such an ending
        source = re.sub(r"\,[\s]{1,}\,[\s]{0,}lora_request", ", lora_request", source)
        source = re.sub(r"[\s]{1,}\,[\s]{0,}lora_request", ", lora_request", source)
        source = re.sub(r"[\,]{1,}[\s]{0,}lora_request", ", lora_request", source)
        # Prefer using unsloth's sampling params and fallback to trl's if not found
        # We'll enable this later separately when combining both this and GRPOConfig params
        # source = re.sub(
        #     r"sampling_params\s*=\s*sampling_params",
        #     r"sampling_params = getattr(self.args, 'vllm_sampling_params', sampling_params)",
        #     source
        # )
        # Fix later versions of SamplingParams via grpo_update_SamplingParams
        source = source.replace(
            "sampling_params = SamplingParams(**generation_kwargs)",
            "sampling_params = SamplingParams("
            "**grpo_update_SamplingParams("
            "SamplingParams, generation_kwargs, "
            "getattr(self.args, 'vllm_sampling_params', None)"
            ")"
            ")",
        )

        # Skip if no changes done
        if source == original_source:
            continue

        # Find all imports
        imports += [x for x in all_imports if not x.startswith("_") and x in source]

        changed[function] = (
            original_source,
            source,
        )

    # Import all functions
    imports = list(set(imports))

    # Patch all functions
    for function in changed:
        old, new = changed[function]
        RLTrainer_source = RLTrainer_source.replace(old, new)

    RLTrainer_source = RLTrainer_source.replace(
        f"class {RLTrainer_name}", f"class _Unsloth{RLTrainer_name}", 1
    )
    return RLTrainer_source


def patch_trl_rl_trainers():
    # Patch all TRL modules if they have vLLM or PEFT
    import trl.trainer

    all_trainers = dir(trl.trainer)
    all_trainers = [
        x for x in all_trainers if x.islower() and x.endswith("_trainer") and x != "base_trainer"
    ]
    for trainer in all_trainers:
        try:
            _patch_trl_rl_trainers(trainer)
        except Exception as e:
            logger.warning_once(f"Unsloth: Could not patch trl.trainer.{trainer}: {e}")
    return


def patch_trl_disable_gradient_checkpointing():
    # TRL 1.0.0+ wraps generation in:
    #   with torch.no_grad(), disable_gradient_checkpointing(self.model, ...):
    # The toggle only suppresses a cosmetic PyTorch warning; under no_grad it
    # has no functional effect. But on exit it calls
    # gradient_checkpointing_enable(), overwriting Unsloth's custom
    # "unsloth" wrapper -- for Gemma-4 this corrupts forward numerics and
    # blows GRPO KL divergence up to ~10^12 at step 1.
    #
    # Replacing the context manager with a no-op preserves Unsloth's wrapper.
    # trl < 1.0.0 (no disable_gradient_checkpointing): early return.
    # trl >= 1.0.0: noop is correct; only loss is the cosmetic warning.
    try:
        import trl.models.utils as _tmu
    except ImportError:
        return
    if not hasattr(_tmu, "disable_gradient_checkpointing"):
        return
    if getattr(
        _tmu.disable_gradient_checkpointing,
        "_unsloth_noop_patched",
        False,
    ):
        return

    @contextmanager
    def _noop_disable_gradient_checkpointing(model, gradient_checkpointing_kwargs = None):
        yield

    _noop_disable_gradient_checkpointing._unsloth_noop_patched = True

    _tmu.disable_gradient_checkpointing = _noop_disable_gradient_checkpointing

    # Also rebind any trl.* module that already imported the symbol by
    # reference (cached at import time). Walk sys.modules dynamically so this
    # catches every trainer doing
    # `from ...models.utils import disable_gradient_checkpointing`.
    for _mod_name, _mod in list(sys.modules.items()):
        if _mod is None or not _mod_name.startswith("trl."):
            continue
        try:
            _bound = getattr(_mod, "disable_gradient_checkpointing", None)
        except (AttributeError, ImportError):
            continue
        if _bound is None:
            continue
        try:
            setattr(
                _mod,
                "disable_gradient_checkpointing",
                _noop_disable_gradient_checkpointing,
            )
        except (AttributeError, TypeError):
            pass

    if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
        logger.warning_once(
            "Unsloth: Patched trl.models.utils.disable_gradient_checkpointing with "
            "a no-op to preserve Unsloth gradient checkpointing across TRL "
            "generation passes."
        )
    return


def patch_trl_openenv():
    for function in RL_ADDITIONAL_FUNCTIONS["openenv"]:
        logger.info(f"Unsloth: Patching trl openenv with function: {function.__name__}")
        function()  # Call the function to apply the patch
    return


def patch_trl_vllm_generation():
    # trl moved vllm stuff to trl/generation/vllm_generation.py
    # We need to min_p patch it to not instantiate another vLLM instance if we already have one with fast_inference
    # Find the instance of self.llm = LLM(..) (multiline) and wrap it around an if clause
    for function in RL_ADDITIONAL_FUNCTIONS["vllm_generation"]:
        logger.info(f"Unsloth: Patching trl VLLMGeneration with function: {function.__name__}")
        function()
    return


def patch_trl_vllm_generation():
    # trl moved vllm stuff to trl/generation/vllm_generation.py
    # We need to min_p patch it to not instantiate another vLLM instance if we already have one with fast_inference
    # Find the instance of self.llm = LLM(..) (multiline) and wrap it around an if clause
    for function in RL_ADDITIONAL_FUNCTIONS["vllm_generation"]:
        logger.info(f"Unsloth: Patching trl VLLMGeneration with function: {function.__name__}")
        function()
    return


def PatchFastRL(algorithm = None, FastLanguageModel = None):
    if FastLanguageModel is not None:
        PatchRL(FastLanguageModel)
    # Under UNSLOTH_ALLOW_CPU=1 (CPU-only CI), skip TRL trainer rewriting so
    # downstream `inspect.getsource(trl.SFTTrainer)` drift detectors see the
    # pristine upstream class, not the compiled Unsloth* wrappers.
    if os.environ.get("UNSLOTH_ALLOW_CPU", "0") == "1":
        return
    # Install the disable_gradient_checkpointing noop BEFORE
    # patch_trl_rl_trainers, which imports extra trl.* submodules; any module
    # imported after the sys.modules walk would keep the original broken
    # binding. Installing first ensures the canonical symbol is replaced before
    # those submodules bind it.
    patch_trl_disable_gradient_checkpointing()
    patch_trl_rl_trainers()
    patch_trl_openenv()
    patch_trl_vllm_generation()
    if type(algorithm) is str and algorithm.islower():
        PatchRLStatistics(algorithm)
