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
import copyreg
import importlib
import collections
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
    "max_autotune": False,
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}

# vLLM compatibility shim: TRL expects GuidedDecodingParams even when vLLM does not provide it.
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

try:
    torch_version = Version(torch.__version__.split("+")[0].split("a")[0].split("b")[0])
except Exception:
    torch_version = Version("0.0.0")

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
            # Local fallback: TRL removed or moved this symbol.
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
        # Snapshot before TRL's unwrap CM, which calls gradient_checkpointing_disable() before
        # yielding. Keep the mode value (e.g. "unsloth"), not a bool, so the finally restore matches.
        use_gradient_checkpointing = next(
            (
                v
                for v in (getattr(m, "gradient_checkpointing", False) for m in model.modules())
                if v
            ),
            False,
        )
        with unwrap_model_for_generation(model, *args, **kwargs) as unwrapped_model:
            FastLanguageModel.for_inference(model)

            # .clone is required because inference_mode is forced here; no_grad would have been the better choice.
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
        # For CLIP-like models capable of returning loss values: if return_loss is unset in inputs, check
        # whether model.forward defaults it to True.
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

        # labels may be popped when computing the loss (label smoothing), so grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        # Force logits during eval, but restore the user's prior setting after so an explicit
        # UNSLOTH_RETURN_LOGITS="1" is not silently turned off.
        _old_return_logits = os.environ.get("UNSLOTH_RETURN_LOGITS", "0")
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
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]
        os.environ["UNSLOTH_RETURN_LOGITS"] = _old_return_logits
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
# Drops/renames config arguments the installed TRL no longer accepts, so a
# script pinned to an older TRL keeps working after an upgrade. Falls back to
# the historical raw passthrough so this can never break trainer construction.
try:
    from unsloth.models.rl_config_compat import filter_config_init_kwargs as _unsloth_filter_config_init_kwargs
    # A cache file generated here can be imported by an older Unsloth whose filter
    # predates `mirrored_from`, so drop the argument rather than raise TypeError.
    if "mirrored_from" not in inspect.signature(_unsloth_filter_config_init_kwargs).parameters:
        _unsloth_filter_config_init_kwargs_old = _unsloth_filter_config_init_kwargs
        def _unsloth_filter_config_init_kwargs(config_class, kwargs, **kw):
            return _unsloth_filter_config_init_kwargs_old(config_class, kwargs)
except Exception:
    def _unsloth_filter_config_init_kwargs(config_class, kwargs, **kw): return kwargs
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
        # One dict so the filter sees the mirrored parameters AND `**kwargs`:
        # filtering kwargs alone would double-bind any argument TRL renamed,
        # since the new name is itself a mirrored parameter.
        _unsloth_config_arguments = dict({RLConfig_call_args}{RLConfig_kwargs})
        super().__init__(**_unsloth_filter_config_init_kwargs({RLConfig_name}, _unsloth_config_arguments, mirrored_from = __class__))
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


# Marks an Unsloth-generated config class. It is renamed to the TRL name it stands in for,
# so the "already patched" checks cannot go by __name__ alone.
_UNSLOTH_PATCHED_CONFIG_FLAG = "_unsloth_patched_rl_config"
# Set on the PRISTINE config class, pointing at the Unsloth subclass that has taken over its module attribute.
_UNSLOTH_CONFIG_PICKLE_TARGET = "_unsloth_config_pickle_target"


def _is_unsloth_patched_config(config_class):
    return bool(
        getattr(config_class, _UNSLOTH_PATCHED_CONFIG_FLAG, False)
        or getattr(config_class, "__name__", "").startswith("Unsloth")
    )


def _reduce_pristine_rl_config(config):
    """copyreg reducer for instances of a TRL config class Unsloth has replaced.

    Anyone holding the pristine class - code that imported trl before unsloth, or
    TRL's own `TrainingArguments` -> `<X>Config` conversion - owns instances whose
    class no longer answers to its own module attribute, so pickle refuses them.
    Reduce through `copyreg._reconstructor`, which is stdlib, and the Unsloth
    subclass, which pickles under the pristine module and name, so the resulting
    file loads as a plain `<X>Config` on a machine without unsloth.
    """
    target = getattr(type(config), _UNSLOTH_CONFIG_PICKLE_TARGET, None)
    if target is None:
        target = type(config)
    getstate = getattr(config, "__getstate__", None)
    state = getstate() if getstate is not None else config.__dict__
    return (copyreg._reconstructor, (target, object, None), state)


def _patch_config_pickle_identity(pristine_config, patched_config):
    """Keep `torch.save(trainer.args, ...)` working once the config is patched.

    `Trainer._save_checkpoint` ends in `torch.save(self.args, ...)`, and pickle
    stores a class as `__module__` + `__qualname__`, then refuses unless the
    object living at that path *is* the class. Patching a trainer rebinds
    `<X>Config` at the module the pristine class calls home, which breaks that
    identity for the pristine class, so every checkpoint save raises
    `PicklingError`. Left alone the patched class is no better: it advertises
    itself as `Unsloth<X>Trainer.Unsloth<X>Config`, a top level module that only
    exists next to a compiled cache, so its `training_args.bin` cannot be read
    back by a plain TRL install either.

    Give the patched class the pristine module and name, so it pickles as
    `trl.trainer.<x>_config.<X>Config` and reloads anywhere, and reduce pristine
    instances through it.
    """
    if pristine_config is None or patched_config is None:
        return
    if pristine_config is patched_config:
        return
    home_module_name = getattr(pristine_config, "__module__", None)
    qualname = getattr(pristine_config, "__qualname__", None) or getattr(
        pristine_config, "__name__", None
    )
    # A nested or generated class has nothing stable to point pickle at.
    if not home_module_name or not qualname or "." in qualname:
        return
    home_module = sys.modules.get(home_module_name)
    if home_module is None:
        try:
            home_module = importlib.import_module(home_module_name)
        except Exception:
            return
    # The rebinding above follows the trl.trainer.<x>_trainer -> <x>_config convention, but
    # pickle consults the pristine class's own __module__, which is also right for
    # trl.experimental wrappers the convention misses.
    current = getattr(home_module, qualname, None)
    if current is not pristine_config and current is not patched_config:
        # Something else owns the name; renaming the patched class would only move the failure.
        return
    try:
        setattr(home_module, qualname, patched_config)
        patched_config.__module__ = home_module_name
        patched_config.__qualname__ = qualname
        patched_config.__name__ = qualname
        setattr(patched_config, _UNSLOTH_PATCHED_CONFIG_FLAG, True)
    except Exception as e:
        logger.info(f"Unsloth: Could not repoint {qualname} for pickling: {e}")
        return
    _register_config_pickle_fallback(pristine_config, patched_config)


def _config_reduction_is_safe(displaced_config, patched_config):
    """Can instances of `displaced_config` be rebuilt as `patched_config`?

    The straightforward yes is a subclass: the patched class carries every field
    the displaced one declares.

    A sibling also qualifies, and TRL produces one. The deprecation shims at
    `trl.trainer.<x>_config.<X>Config` subclass the real class in
    `trl.experimental.<x>`, and the wrapper resolution above generates the
    patched class from that same parent rather than from the shim -- so the two
    end up siblings, not subclass and base. The shim adds only a `__post_init__`
    that warns, and every base it has is already in the patched class's MRO, so
    the patched class still holds everything such an instance can carry.

    Anything else is left alone: rebuilding an unrelated class as this one would
    silently drop state, which is worse than the PicklingError it would avoid.
    """
    if issubclass(patched_config, displaced_config):
        return True
    # `object` alone is no relationship at all, so it does not count as shared.
    bases = set(displaced_config.__mro__[1:]) - {object}
    return bool(bases) and bases.issubset(set(patched_config.__mro__))


def _register_config_pickle_fallback(displaced_config, patched_config):
    """Route instances of a config class Unsloth displaced through the patched one.

    Covers the pristine class, and any thin wrapper a TRL release leaves at a
    module attribute the patching has taken over.
    """
    if displaced_config is None or displaced_config is patched_config:
        return
    if not isinstance(displaced_config, type):
        return
    if not _config_reduction_is_safe(displaced_config, patched_config):
        return
    if copyreg.dispatch_table.get(displaced_config) is _reduce_pristine_rl_config:
        return
    try:
        setattr(displaced_config, _UNSLOTH_CONFIG_PICKLE_TARGET, patched_config)
        copyreg.dispatch_table[displaced_config] = _reduce_pristine_rl_config
    except Exception as e:
        logger.info(
            f"Unsloth: Could not make {getattr(displaced_config, '__name__', '?')} "
            f"instances picklable: {e}"
        )


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
        # `iterator is dataset` misses an IterableDataset whose __iter__ returns one stored
        # generator. Two iter() calls giving the same object catches both; a datasets.IterableDataset
        # restarts, so it answers False and is rewound.
        single_pass = iterator is dataset or iterator is iter(dataset)
        row = next(iterator, None)
        if single_pass and row is not None:
            import itertools
            source = itertools.chain([row], iterator)
    except Exception:
        row = None
    if isinstance(row, dict):
        names.update(row.keys())
    # Keep the probed row: on a one-shot stream it is the only row anything may see. Without it
    # _sliceable_per_token had no widths and cut input_ids alone, leaving labels overlength.
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
        # Per value, per row, like the map path: _sliceable_per_token judges from ONE row, and an
        # optional column that is a list there can be None later, which raised in the dataloader.
        # Only input_ids is cut unconditionally.
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
        # Everything else (column_names, features, ...) is the wrapped split's answer. Never a dunder
        # nor our own state: a DataLoader worker pickles the split, and __setstate__ before __init__
        # would recurse on _inner forever.
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
        # With no supervision columns _keep is True for every row, so building the index would
        # transform every item -- a whole extra tokenization pass for a with_transform split.
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

# Believe a producer's own truncation claim: scanning a with_transform split tokenizes every
# row in __init__, the eager pass it avoids. Only a cap at or below the enforced one counts.
_TRUNCATION_ATTESTATION_ATTR = "_unsloth_truncated_to"


def _attested_within_cap(dataset, cap):
    """The split's own truncation-width claim, or None if it makes none.

    Read from `__dict__`, not `getattr`: `_CappedBase.__getattr__` forwards to the
    inner split, so a wrapper would inherit a guarantee it does not carry.
    """
    own = getattr(dataset, "__dict__", None)
    if not isinstance(own, dict):
        return None
    claimed = own.get(_TRUNCATION_ATTESTATION_ATTR)
    if not isinstance(claimed, int) or isinstance(claimed, bool):
        return None
    return claimed <= cap


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
    attested = _attested_within_cap(dataset, cap)
    if attested is not None:
        return attested
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
    # A stream whose iter hands back the same exhausting generator cannot spare a row, and nothing here chains it back.
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
    # input_ids first, then a fixed order: `names` is a set, and the map path reads the width off
    # input_ids as it walks this list, so labels first sliced them against nothing.
    known = [c for c in _PER_TOKEN if c in names]
    # A custom per-token field (loss_mask, token_weights) is not in the allow-list, so it stayed
    # full length while input_ids was cut and a custom collator got mismatched lengths. Judge by
    # alignment, but only a flat vector of scalars, keeping `messages` out of the slice.
    custom = sorted(c for c in names if c not in _PER_TOKEN)
    per_token = known + custom
    if len(known) < 2 and not custom:
        return known
    # `probed` is the row _column_names already read. Preferring it is what lets a one-shot stream align
    # every per-token column: reading another row would cost the caller that example.
    row = probed if isinstance(probed, dict) else _first_row_without_consuming(dataset)
    if not isinstance(row, dict):
        return ["input_ids"] if "input_ids" in names else []
    try:
        width = len(row.get("input_ids"))
    except Exception:
        # Nothing to measure against, so a custom column has no evidence behind it; fall back to the named ones only.
        return known
    kept = []
    for name in per_token:
        value = row.get(name)
        if name in custom and not _is_token_vector(value, width):
            continue
        try:
            # As long as input_ids, which makes the FIRST axis the token axis: [seq_len, channels] slices
            # right, and a channel-major position_ids ([3, seq_len] under mrope) fails and is left alone.
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


def _pin_pristine_sft_loss_type(config_cls):
    """Pin `loss_type` to `nll` on TRL's own `SFTConfig`, not just on ours.

    Patching rebinds `trl.SFTConfig` to the generated subclass, so a caller who
    ran `from trl import SFTConfig` before importing Unsloth keeps the pristine
    class and would still get TRL >= 1.7.0's `chunked_nll` (that ordering is
    supported and covered by the padding-free tests). TRL declares the field as
    `None` and resolves it in `__post_init__`, so seeding `nll` there is enough;
    an explicit `loss_type = ` still wins, and `use_liger_kernel = True` already
    resolved to `nll`. Only the unresolved `None` default is touched, which also
    makes this a no-op on TRL < 1.7.0 and on a second call.
    """
    field = getattr(config_cls, "__dataclass_fields__", {}).get("loss_type")
    if field is None or field.default is not None:
        return False
    init = config_cls.__dict__.get("__init__")
    if init is None:
        return False
    try:
        parameters = inspect.signature(init).parameters
    except (TypeError, ValueError):
        return False
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameter.default is not inspect.Parameter.empty
    ]
    defaults = init.__defaults__ or ()
    keyword_defaults = init.__kwdefaults__ or {}
    if "loss_type" in positional and len(defaults) == len(positional):
        index = positional.index("loss_type")
        if defaults[index] is not None:
            return False
        new_defaults = list(defaults)
        new_defaults[index] = "nll"
        init.__defaults__ = tuple(new_defaults)
    elif "loss_type" in keyword_defaults:
        if keyword_defaults["loss_type"] is not None:
            return False
        keyword_defaults["loss_type"] = "nll"
    else:
        return False
    field.default = "nll"
    # The class attribute is the other copy of the default: dataclasses seeds it at class creation and a
    # later subclass reads the field, so leave the two agreeing rather than half-patched.
    setattr(config_cls, "loss_type", "nll")
    return True


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
        # The TRAINER's resolved value first: the collator uses it, and it is set whenever TRL
        # resolved it. The split's own schema read False off a pre-tokenized eval split with only
        # input_ids + completion_mask, so cut-away rows survived as all -100, i.e. a NaN eval loss.
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
        # evaluate() caps the split and Transformers then calls get_eval_dataloader, which is also
        # wrapped, so both reach here in one call. Re-capping is destructive over a one-shot stream:
        # each probe reads a row and _CappedStream re-opens the SAME exhausted source.
        if _cap_still_holds(dataset, cap, drop_unsupervised):
            return dataset
        names, dataset, probed = _column_names(dataset)
        if "input_ids" not in names:
            # No tokens here yet, so there is nothing to cut. TRL does not tokenize a late split either
            # (_prepare_dataset runs only from __init__), but that is its own gap.
            return _mark_capped(dataset, cap, drop_unsupervised)
        # eval_packing is consulted only where the packer reaches the split, hence packs_late per
        # entry point. Up to TRL 1.6 nothing packs a late split; from 1.7.0 `evaluate` prepares it
        # itself and the strategy owns the overflow, so capping rows first throws that away.
        if packs_late and _eval_packing_on(args):
            # Left FOR the packer, so it is not capped and must not be marked as if it were. TRL only
            # prepares a split PASSED to evaluate, so a stored split or string key arrives here untouched;
            # where it really packs, it hands back a NEW object.
            return dataset
        # A packed split carries document lengths, not tokens: slicing input_ids under a seq_lengths that
        # still describes the longer row builds position ids for tokens the row no longer has.
        if "seq_lengths" in names:
            return _mark_capped(dataset, cap, drop_unsupervised)
        try:
            # TRL slices [-max_length:] for keep_end, and so does the construction-time cap; always keeping the
            # prefix evaluates the wrong half of every long row.
            mode = getattr(args, "truncation_mode", "keep_start")
            # keep_start and keep_end are the only two slices there are, and a third value silently became
            # keep_start here. Refusing means handing the split back untouched so the caller still has it.
            if mode not in ("keep_start", "keep_end"):
                print(
                    f"Unsloth: `truncation_mode = {mode}` is not one of "
                    "keep_start / keep_end, so this split is left uncapped."
                )
                return dataset
            cut = slice(-cap, None) if mode == "keep_end" else slice(None, cap)
            per_token = _sliceable_per_token(dataset, names, cap, probed)
            # Never on the predict path: dropping rows is right for a loss and wrong for predict, whose contract
            # is one prediction per row IN ORDER.
            supervision = _supervision_columns(args, names) if drop_unsupervised else []
            # A stream has no length, cannot be rewound, and on datasets 4.x dataset[0] reads 0 as a
            # COLUMN name rather than failing. Use map(), which is lazy over every row it will yield.
            overlength = True
            if not _is_stream(dataset):
                try:
                    overlength = max(len(r) for r in dataset["input_ids"]) > cap
                except Exception:
                    # The scan only exists to skip a pointless map; a split with no column access cannot answer it, and
                    # that is no reason to hand it back uncapped.
                    pass
            # A split already under the cap still goes through the supervision filter below: being short is not
            # the same as being supervised, and a row whose labels are all -100 is a NaN loss either way.
            if not overlength and not supervision:
                return _mark_capped(dataset, cap, drop_unsupervised)
            # A split we cannot rewrite is capped on read instead: map belongs to datasets, and a with_transform
            # split has it but recreates its rows on every read, so mapping writes a table nobody reads.
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
                # Per value, not per column: _sliceable_per_token judges by ONE row, so an optional column
                # that is None three rows later raised inside map and the catch returned the UNCAPPED split.
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
            # A truncated row can end all -100, or with an all-zero mask the collator makes all -100, and
            # such a batch reports a NaN loss. Intersect labels AND every active mask, not one filter
            # each: masks are applied ONTO the labels, so separate filters still pass an all -100 row.
            if supervision:
                kept = new.filter(
                    lambda e, c = tuple(supervision): any(
                        all((x != -100) if n == "labels" else x for n, x in zip(c, v))
                        for v in zip(*[e[n] for n in c])
                    )
                )
                # Hand back the caller's own split when the filter dropped nothing: a copy of an unchanged dataset
                # is a new object for the trainer to cache and reload for no reason.
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
        # Carried onto args because _cap only ever sees those. `is not None` rather than truthiness: False
        # is TRL's answer just as much as True.
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
        # truncation_mode shapes the SLICE, so it belongs in the key: without it, evaluating once with
        # keep_start and again with keep_end handed back the cached prefixes for both.
        key = (
            id(dataset),
            drop_unsupervised,
            getattr(trainer.args, "truncation_mode", "keep_start"),
            # evaluate and get_eval_dataloader share drop_unsupervised and see the same object in one call, but
            # only the first may skip the cut under eval_packing.
            packs_late,
        )
        seen = memo.get(key)
        if seen is not None and seen[0] is dataset and seen[1] == cap and seen[3] == token:
            memo[key] = memo.pop(key)  # most recently used goes last
            return seen[2]
        capped = _cap(dataset, cap, trainer.args, drop_unsupervised, packs_late)
        memo[key] = (dataset, cap, capped, token)
        # Bounded: every entry pins both the original split and the capped copy for the trainer's
        # life, so a later split cannot inherit a freed id(). A fresh subset per epoch otherwise
        # accumulated Arrow tables until the host ran out.
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
        # evaluate(eval_dataset = "validation") picks one split out of a stored dict, and capping the
        # KEY is a no-op, so the split it names reached the collator uncapped.
        if isinstance(given, str):
            stored = getattr(trainer, "eval_dataset", None)
            if isinstance(stored, dict) and given in stored:
                capped = _cap_cached(trainer, stored[given], cap, drop_unsupervised, packs_late)
                # Staged for the caller to swap in and OUT: overwriting stored[given] destroyed the uncapped
                # original, so a later truncation_mode = "keep_end" could only re-cap the saved prefix.
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
            # A retained max_length does not prove the cap is enforced: it is what the construction block
            # leaves when it turns padding-free OFF, TRL's collator never truncates rows carrying
            # input_ids, and _prepare_dataset runs only from __init__.
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
                # A named split: swap the capped copy in for this call only, so the caller keeps the uncapped
                # original for the next mode.
                container, key, replacement = swap
                self._unsloth_pending_split_swap = None
                previous = container[key]
                container[key] = replacement
                try:
                    return original(self, *args, **kwargs)
                finally:
                    container[key] = previous
            # evaluate() with no split falls back to the one stored on the trainer, which a caller can install
            # after construction, where the constructor's cap can no longer see it.
            stored = getattr(self, keyword, None) if keyword == "eval_dataset" else None
            if stored is None:
                return original(self, *args, **kwargs)
            capped = _cap_splits(self, stored, cap, drop_unsupervised, packs_late)
            if capped is stored:
                return original(self, *args, **kwargs)
            # Swapped onto the trainer rather than passed down: Trainer.evaluate recurses over a dict of
            # splits by NAME when nothing was passed, and passing the dict makes that an override.
            setattr(self, keyword, capped)
            try:
                return original(self, *args, **kwargs)
            finally:
                setattr(self, keyword, stored)

        wrapped._unsloth_eval_cap_wrapped = True
        return wrapped

    # predict keeps every row: one prediction per row, in order. The two dataloader builders are
    # public API and bypass evaluate/predict, so get_eval_dataloader(late) met the padding-free
    # collator with max_length already cleared and nothing capping the split.
    # Only evaluate can hand its split to TRL's own prep, and only from 1.7.0. Read once, before
    # anything is wrapped, so the probe sees TRL's method rather than ours.
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
# Whether the MOST RECENT forward handed back real logits instead of hidden states: the warning
# attribute above is warn-once bookkeeping and is never cleared, so it answers "ever degraded".
_UNSLOTH_GRPO_HIDDEN_STATES_DEGRADED_ATTR = "_unsloth_grpo_hidden_states_degraded"


def _module_returns_logits(module):
    # get_output_embeddings() is None on the decoder bodies and the head module on the *ForCausalLM
    # wrappers, so it finds the head owner by behaviour rather than by a model-name list.
    if module is None:
        return False
    get_output_embeddings = getattr(module, "get_output_embeddings", None)
    if not callable(get_output_embeddings):
        return False
    try:
        return get_output_embeddings() is not None
    except Exception:
        return False


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
        if child is None or child is model or not hasattr(child, "forward"):
            continue
        # Descend only into an adapter that still owns the head: a bare *ForCausalLM (TRL's GRPO
        # ref_model) also has .model, but that is its decoder body, so wrapping it returns
        # [B, T, vocab] and the fallback is a silent no-op.
        if not _module_returns_logits(child):
            continue
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
    # The degradation flag is per call: a forward that splats **kwargs into a sub-module raises
    # only for the batches that reach it, so a sticky flag would send real hidden states through
    # the raw-logits helper.
    setattr(model, _UNSLOTH_GRPO_HIDDEN_STATES_DEGRADED_ATTR, True)
    if getattr(model, _UNSLOTH_GRPO_HIDDEN_STATES_WARNING_ATTR, False):
        return
    setattr(model, _UNSLOTH_GRPO_HIDDEN_STATES_WARNING_ATTR, True)
    logger.warning(message)


def _note_grpo_hidden_states_success(model):
    """Record that the forward about to return really is handing back hidden states."""
    setattr(model, _UNSLOTH_GRPO_HIDDEN_STATES_DEGRADED_ATTR, False)


def _minimise_logits_kwarg(forward_signature, args, forward_kwargs):
    """Ask the model for as few logits as it will give us, and say which kwarg did it.

    We are about to overwrite `outputs.logits` with hidden states, so every
    logit the forward computes is thrown away. transformers spells the limit
    `logits_to_keep` -- measured on 4.57.6, 5.0.0 and 5.15.0, all three declare
    that name and none declares `num_logits_to_keep`, so the second name is not
    for them. It is for us: `unsloth/models/llama.py` and `mistral.py` patch in
    forwards declaring both, and `unsloth/models/vision.py` probes for the old
    name FIRST because some VLM stacks still carry only it. Whichever name a
    forward takes, it reads the value as
    `slice(-value, None)`, so the DEFAULT of 0 becomes `slice(0, None)` -- the
    whole sequence. The GRPO trainer does not pass a value at all, so the
    lm_head projects every prompt and completion position over the full
    vocabulary and, for the softcapped models, multiplies the result twice more.

    Muse Glimmer 30B on a Kaggle 2xT4 measured that at a 1002 MiB allocation
    per chunk, over a 202048-wide vocabulary. On one card it is invisible: the
    trainer's `del outputs` frees it a line later. On a layer-split model
    accelerate copies it to the other card first and the run dies there.

    1, not 0: 0 means "all of them", and a model that computes its own loss from
    `labels` needs real logits, so that case is left alone.
    """
    if forward_kwargs.get("labels") is not None:
        return None
    try:
        bound = forward_signature.bind_partial(*args, **forward_kwargs)
    except TypeError:
        return None
    # A forward given labels positionally lands it in bound.arguments and never in forward_kwargs, so
    # the lookup above cannot see it and the loss the model computes would be one position wide.
    if bound.arguments.get("labels") is not None:
        return None
    accepts_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in forward_signature.parameters.values()
    )
    for name in ("logits_to_keep", "num_logits_to_keep"):
        declared = name in forward_signature.parameters
        if not declared and not accepts_var_keyword:
            continue
        # Positionally and by keyword is a TypeError. Give up rather than try the OTHER spelling,
        # which would fight the caller's width or be swallowed by **kwargs and silently ignored.
        if name in bound.arguments and name not in forward_kwargs:
            return None
        forward_kwargs[name] = 1
        return name
    return None


def _drop_spare_hidden_states(outputs):
    """Detach every hidden-state layer from `outputs`; the caller keeps the last.

    `outputs.hidden_states = None` does NOT do this. `ModelOutput.__setattr__`
    is

        if name in field_names and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    so assigning None sets the attribute and leaves the mapping entry holding
    the full tuple, and `ModelOutput` blocks `__delitem__`, `pop`, `update` and
    `setdefault` outright. Every consumer that walks the object as a mapping --
    accelerate's `send_to_device`, which is the one that matters here -- still
    sees and copies all of it. Writing through `OrderedDict.__setitem__` is what
    actually clears it, and leaves the mapping and the attribute agreeing on
    None, which is the state a forward returns with `output_hidden_states=False`.
    """
    try:
        if isinstance(outputs, collections.OrderedDict) and "hidden_states" in outputs:
            collections.OrderedDict.__setitem__(outputs, "hidden_states", None)
            object.__setattr__(outputs, "hidden_states", None)
        elif isinstance(outputs, dict) and "hidden_states" in outputs:
            outputs["hidden_states"] = None
        elif hasattr(outputs, "hidden_states"):
            outputs.hidden_states = None
    except Exception:
        # A frozen or exotic output object is not worth failing the step over; the caller has already taken
        # the layer it needs.
        logger.debug(
            "Unsloth: could not drop spare GRPO hidden states.",
            exc_info = True,
        )


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
        # accelerate's extract_model_from_parallel(keep_fp32_wrapper = False), called every GRPO step,
        # rebinds the forward as MethodType, so the module arrives as a leading positional argument;
        # original_forward is already bound, so drop it.
        while len(args) != 0 and args[0] is target_model:
            args = args[1:]
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") != "1":
            # nobody asked for hidden states, so this returns real logits
            setattr(target_model, _UNSLOTH_GRPO_HIDDEN_STATES_DEGRADED_ATTR, True)
            return original_forward(*args, **kwargs)

        # Copy: _drop_forward_kwargs_consumed_positionally returns kwargs unchanged when there is
        # nothing to drop, so mutating it would poison the caller's dict for the fallback calls.
        forward_kwargs = dict(
            _drop_forward_kwargs_consumed_positionally(forward_signature, args, kwargs)
        )
        num_logits_to_keep = _get_num_logits_to_keep(forward_signature, args, forward_kwargs)
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True
        logits_kwarg = _minimise_logits_kwarg(
            forward_signature,
            args,
            forward_kwargs,
        )

        def rejected_hidden_states(message):
            return "output_hidden_states" in message or "return_dict" in message

        def forward_without_hidden_states():
            _warn_grpo_hidden_states_fallback_once(
                target_model,
                f"Unsloth: GRPO fallback could not request hidden states for unsupported model {model_name}; using logits directly.",
            )
            return original_forward(*args, **kwargs)

        # TRL 0.26+: Config may be in a separate *_config.py module
        # Thin wrapper fallback: walk the Trainer's MRO to find Config in the real implementation module (e.g.,
        # trl.experimental.bco)
        try:
            outputs = original_forward(*args, **forward_kwargs)
        except TypeError as error:
            message = str(error)
            if logits_kwarg is None or logits_kwarg not in message:
                if not rejected_hidden_states(message):
                    raise
                return forward_without_hidden_states()
            # The signature advertised the parameter but the forward refuses the value: retry without the
            # minimisation, through the same fallback, rather than lose hidden states over it.
            forward_kwargs.pop(logits_kwarg, None)
            logits_kwarg = None
            try:
                outputs = original_forward(*args, **forward_kwargs)
            except TypeError as retry_error:
                if not rejected_hidden_states(str(retry_error)):
                    raise
                return forward_without_hidden_states()

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None or len(hidden_states) == 0:
            _warn_grpo_hidden_states_fallback_once(
                target_model,
                f"Unsloth: GRPO fallback did not receive hidden states for unsupported model {model_name}; using logits directly.",
            )
            if logits_kwarg is None:
                return outputs
            # outputs.logits is the return value now, and one position was asked for only to throw the
            # logits away; GRPO drops the last and slices the completion window, so restore the caller's
            # own limit and re-run.
            if logits_kwarg in kwargs:
                forward_kwargs[logits_kwarg] = kwargs[logits_kwarg]
            else:
                forward_kwargs.pop(logits_kwarg, None)
            return original_forward(*args, **forward_kwargs)

        hidden_states = hidden_states[-1]
        if num_logits_to_keep != 0:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]
        # Only the last layer is read, and accelerate's AlignDevicesHook.post_forward copies every
        # tensor in the returned object to the input device, so keeping the rest costs a cross-device
        # copy per layer as well as the memory.
        _drop_spare_hidden_states(outputs)
        _note_grpo_hidden_states_success(target_model)
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
    # Defensive wrapper matching patch_trl_rl_trainers()'s try/except, so direct callers do not see
    # exceptions from the impl on TRL versions that rename or move classes.
    try:
        return _patch_trl_rl_trainers_impl(trainer_file)
    except Exception as e:
        # Warning, not info: the impl RETURNS for the benign case, so reaching here means generation
        # failed and the run silently falls back to trl's trainer, losing Unsloth's compute_loss,
        # bf16/fp16 fixup and dataset handling.
        logger.warning_once(
            f"Unsloth: Could not build the patched trl.trainer.{trainer_file}, "
            f"so training will use trl's own trainer instead: "
            f"{type(e).__name__}: {e}"
        )
        return


def _patch_trl_rl_trainers_impl(trainer_file = "grpo_trainer"):
    import trl
    import trl.trainer

    try:
        trainer = eval(f"trl.trainer.{trainer_file}")
    except Exception as error:
        logger.info(f"Unsloth: Could not import trl.trainer.{trainer_file}: {error}")
        return

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
        # TRL 0.26+: the Config may be in a separate *_config.py module, or reachable by walking the
        # Trainer's MRO to the real implementation module (trl.experimental.bco).
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
        # TRL 0.26+: the Config may be in a separate *_config.py module, or loadable from the parent trainer's module.
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

    if RLTrainer.__name__.startswith("Unsloth"):
        print(f"Unsloth: {RLTrainer.__name__} is already patched.")
        return
    if _is_unsloth_patched_config(RLConfig):
        print(f"Unsloth: {RLConfig.__name__} is already patched.")
        return

    # TRL 0.26+: resolve thin wrappers (trl.trainer shims forwarding to trl.experimental) to their
    # parent class, and only when that parent really lives in a trl.experimental module.
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
                if "trl.experimental" in _parent_mod.__name__:
                    RLConfig = _parent
                    break
    except Exception:
        pass

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

    arguments, call_args = processed[0]
    RLTrainer_post = ""

    if "tokenizer" not in parameters and "processing_class" in parameters:
        arguments += f",\n{' ' * 8}tokenizer = None"
        call_args = call_args.replace(
            "processing_class = processing_class",
            "processing_class = tokenizer if tokenizer is not None else processing_class",
        )

    # Edit bf16, fp16 by checking the model's dtype/torch_dtype directly.
    extra_args = ""
    if "args" in call_args and "model" in call_args:
        mixed_precision = (
            "use_bf16 = getattr(args, 'bf16', False)\n"
            "if type(use_bf16) is not bool: use_bf16 = False\n"
            "use_fp16 = getattr(args, 'fp16', False)\n"
            "if type(use_fp16) is not bool: use_fp16 = False\n"
            "force_float32 = False\n"
            # Device-aware bf16 check (CUDA/XPU/HIP), so V100/T4 never pick bf16 while AMD/Intel are unaffected;
            # fall back on older unsloth_zoo.
            "try:\n"
            "    from unsloth_zoo.device_type import device_is_bf16_supported as _bf16_supported\n"
            "except Exception:\n"
            "    _bf16_supported = torch.cuda.is_bf16_supported\n"
            # FORCE_FLOAT32 models (Gemma3, gpt_oss) cannot use float16: without bf16 keep float32, with
            # bf16 full finetuning may still autocast. Stamped by from_pretrained, since the env is
            # process wide and a later load would otherwise answer for this trainer.
            "full_finetuning = getattr(model, '_unsloth_full_finetuning', None)\n"
            "if full_finetuning is None: full_finetuning = os.environ.get('UNSLOTH_ENABLE_FULL_FINETUNING', '0') == '1'\n"
            # Stamped by from_pretrained: the env is process wide, so a forced family loaded earlier would
            # answer here for a model that is not forced at all.
            "model_forced_float32 = getattr(model, '_unsloth_forced_float32', None)\n"
            "if model_forced_float32 is None: model_forced_float32 = os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1'\n"
            "if model_forced_float32 and not (full_finetuning and _bf16_supported()):\n"
            "    print('Unsloth: Switching to float32 training since model cannot work with float16')\n"
            "    force_float32 = True\n"
            "mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')\n"
            "dtype = getattr(model.config, 'dtype', None) or getattr(model.config, 'torch_dtype', None)\n"
            "if dtype is None: dtype = model.get_input_embeddings().weight.dtype\n"
            "from unsloth_zoo.utils import _get_dtype\n"
            "dtype = _get_dtype(dtype)\n"
            "float16 = dtype == torch.float16\n"
            "bfloat16 = dtype == torch.bfloat16\n"
            "float32 = dtype == torch.float32\n"
            # Set only when the caller passed dtype = torch.float32 themselves: a request, not a side effect of
            # upcasting, and immune to a second load.
            "user_float32 = bool(getattr(model, '_unsloth_user_float32', False))\n"
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
            "elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32' and float32 and user_float32 and not _bf16_supported():\n"
            # Without bf16 the only autocast is float16, whose exponent range overflows float32 to inf
            # then NaN. Gated on the explicit request: fp16 autocast over fp32 master weights is the
            # normal V100/T4 recipe (#4082).
            "    print('Unsloth: Model is in float32 and this GPU has no bfloat16 support, so training stays in float32. Pass fp16 = True to force float16 mixed precision instead.')\n"
            "    args.fp16 = False\n"
            "    args.bf16 = False\n"
            "    os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'\n"
            "    if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'\n"
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

    # Check if per_device_eval_batch_size (default 8) is bigger than bsz, and use FP16 / BF16 evaluation.
    if "args" in call_args:
        if "eval_dataset" in call_args:
            check_eval_dataset = (
                "if getattr(args, 'eval_dataset', None) is not None and "
                "getattr(args, 'eval_strategy', 'no') == 'no':\n"
                "    args.eval_strategy = 'steps'\n"
                "    if getattr(args, 'eval_steps', None) is None: args.eval_steps = 0.1\n"
            )
            extra_args += check_eval_dataset

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

    # Force logits to be produced if preprocess_logits_for_metrics or compute_metrics is used.
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

        # max_seq_length may be set here, but trl is moving to max_length.
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
            # TRL >= 1.0.0 refuses padding-free without packing while max_length is set, and Unsloth
            # auto-enables padding-free, so nearly every SFT user tripped that guard. Move the resolved
            # length to where it is enforced: max_seq_length when prep tokenizes, else padding-free off.
            # Unconditional copy: no TRL from 0.22.2 to 1.9.2 declares max_seq_length on SFTConfig, so a
            # hasattr() gate would skip every pristine config and the clear below would drop the cap.
            # Must be None, not 0: TRL's guard reads `args.max_length is not None`.
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
                    # Metadata first, a row only as fallback: reading one off a one-shot stream consumes it, and
                    # `iter(x) is iter(x)` marks those. A with_transform split reports BACKING columns, so one
                    # yielding input_ids over stored `text` read "raw"; probing its rows is free.
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
                    # Already-tokenized rows are not a dead end: TRL's _prepare_dataset truncates them and its LM
                    # collator passes no max_length, so that truncation is the only thing enforcing the cap. The
                    # Zoo returns pre-tokenized rows untouched. skip_prepare_dataset is the exception.
                    # Only a MATERIALISED tokenized dataset, judged per DATASET since splits differ: with_transform
                    # yields input_ids while column_names still says ["text"], raw conversational rows would be
                    # sliced into corrupted turns, and a transform's rows are rebuilt on every read.
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
                    # A packed split is out: TRL skips truncation when packing, and cutting input_ids under a
                    # seq_lengths that still describes the old row is worse than not cutting.
                    "            if 'seq_lengths' in (_cols or ()): return False\n"
                    "            return bool(_cols) and 'input_ids' in _cols\n"
                    "        except Exception:\n"
                    "            return False\n"
                    # Read a row BACK: every predicate above is a guess about what the dataset will do, and this is the
                    # one check that observes it.
                    "    _unsloth_cap = args.max_length\n"
                    # TRL slices [-max_length:] for keep_end, which callers use when the completion sits at the tail of
                    # a long prompt; always keeping the prefix trained on the wrong half of every row.
                    "    _unsloth_truncation_mode = getattr(args, 'truncation_mode', 'keep_start') or 'keep_start'\n"
                    # keep_start and keep_end are the only two slices, and TRL's SFT path never reads this
                    # attribute, so a third value would go uncaught. Refuse the enforcement claim instead.
                    "    _unsloth_keep_end = _unsloth_truncation_mode == 'keep_end'\n"
                    "    _unsloth_known_mode = _unsloth_truncation_mode in ('keep_start', 'keep_end')\n"
                    "    _unsloth_slice = slice(-_unsloth_cap, None) if _unsloth_keep_end else slice(None, _unsloth_cap)\n"
                    # Resolved outside the truncation block, since the fallback reads it even under
                    # skip_prepare_dataset. eval_packing is separate from packing, so packing=False with
                    # eval_packing=True lands here and TRL packs the eval split instead of truncating it.
                    "    _unsloth_eval_packing = getattr(args, 'packing', False) if getattr(args, 'eval_packing', None) is None else getattr(args, 'eval_packing')\n"
                    "    _unsloth_completion_only = getattr(args, 'completion_only_loss', None)\n"
                    # Column names first, a row only if free: on a one-shot stream this probe ate the first
                    # TRAINING example. A with_transform split answers with its BACKING table, so one yielding
                    # prompt/completion read False here while TRL read True and applied completion_mask.
                    "    if _unsloth_completion_only is None:\n"
                    "        try:\n"
                    # A set_format(output_all_columns = False) split yields only the named columns while
                    # column_names still lists the whole backing table, so completion-only resolved True off a
                    # completion the rows never hand over. The format's own column list is what is yielded.
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
                    # Parked on args so the late evaluate()/predict() cap reads the SAME value: it resolves from
                    # the train schema, and disagreeing with the collator leaves an all -100 row in.
                    "    args._unsloth_completion_only_loss = _unsloth_completion_only\n"
                    # EVERY row, not the first: a short row 0 before a long row 5000 read as within the cap. A
                    # map-style split is read in full; a stream cannot be rewound, so a bounded prefix is all
                    # there is and the check says so.
                    "    _UNSLOTH_SCAN_ROWS = 1024\n"
                    "    def _unsloth_within_cap(_ds):\n"
                    "        if _ds is None: return True\n"
                    # Believe a producer's own truncation claim: scanning a with_transform split tokenizes every
                    # row in __init__. Read from __dict__ so a wrapper does not inherit the inner split's claim.
                    "        _unsloth_own = getattr(_ds, '__dict__', None)\n"
                    "        if isinstance(_unsloth_own, dict):\n"
                    "            _unsloth_claim = _unsloth_own.get('_unsloth_truncated_to')\n"
                    "            if isinstance(_unsloth_claim, int) and not isinstance(_unsloth_claim, bool):\n"
                    "                return _unsloth_claim <= _unsloth_cap\n"
                    "        try:\n"
                    "            try:    _n = len(_ds)\n"
                    "            except Exception: _n = None\n"
                    # A single-pass stream cannot be scanned: reading it here IS consuming it, and two iter()
                    # calls returning the same object say so (a datasets.IterableDataset restarts and does not).
                    # Unverifiable, so answer as the prefix case: not proven within the cap.
                    "            _unsloth_rows = iter(_ds)\n"
                    "            if _unsloth_rows is iter(_ds): return False\n"
                    "            _seen = 0\n"
                    "            for _row in _unsloth_rows:\n"
                    "                if 'input_ids' not in _row: return True\n"
                    "                if len(_row['input_ids']) > _unsloth_cap: return False\n"
                    # An unexhausted stream is UNVERIFIED, not verified: treating the first 1024 fitting rows as proof
                    # let a later overlength row through, and nothing truncates a pre-tokenized row here.
                    "                _seen += 1\n"
                    "                if _n is None and _seen >= _UNSLOTH_SCAN_ROWS: return False\n"
                    "        except Exception:\n"
                    "            return False\n"
                    "        return True\n"
                    # Each eval split counts: one the truncation cannot rewrite is left alone above, and prep
                    # never re-tokenizes rows carrying input_ids, so trusting the train split left eval uncapped.
                    "    def _unsloth_splits_within_cap(_ev):\n"
                    "        _splits = list(_ev.values()) if isinstance(_ev, dict) else [_ev]\n"
                    "        return all(_unsloth_within_cap(_s) for _s in _splits)\n"
                    # Not train-only: _unsloth_prep_truncates is decided from the train split, so a raw train beside a
                    # pre-tokenized eval set skipped this whole block and left evaluation uncapped.
                    "    if not _unsloth_skip_prepare:\n"
                    # Honour TRL's preparation map_kwargs so a large pre-tokenized dataset is not rewritten
                    # single-process, through the same helper as every map site: the config layer writes serial as
                    # dataset_num_proc = 1, and datasets >= 4.1 builds a Pool(1) for it.
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
                    # Same rule as TRL's truncate_dataset: slice every per-row list column so input_ids, labels,
                    # attention_mask and the masks stay aligned. Written out rather than imported, since
                    # trl.data_utils drags in the processor stack and an ImportError would drop the cap.
                    "        def _unsloth_is_sequence_column(_col):\n"
                    "            try:\n"
                    "                if len(_col) == 0: return False\n"
                    "                _first = _col[0]\n"
                    "            except Exception:\n"
                    "                return False\n"
                    "            if isinstance(_first, (str, bytes)): return False\n"
                    # len(), not hasattr('__len__'): under set_format('torch') a scalar column batches to a 1-D
                    # tensor whose element is 0-dim, so len(_v) threw TypeError and the catch restored the
                    # overlength dataset.
                    "            try:    len(_first)\n"
                    "            except Exception: return False\n"
                    "            return True\n"
                    # Per-token columns only, matched by row length against input_ids: a packed split's
                    # seq_lengths are document lengths, and slicing them left padding-free building position ids
                    # for tokens the row no longer has. Per VALUE, since a list row 0 can be None later.
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
                    # A stream has no length, and IterableDataset.map takes no num_proc: passing one raised TypeError
                    # and the run died on "cannot be enforced" instead of being truncated.
                    "        def _unsloth_is_stream(_ds):\n"
                    "            try:    return not hasattr(_ds, '__len__')\n"
                    "            except Exception: return True\n"
                    # One split, capped and then checked. A stream's map is lazy and applies to EVERY row it will ever
                    # yield, a stronger guarantee than the bounded prefix scan.
                    # Enforcement, not observation: a with_transform split under the cap is not enforced, since it
                    # rebuilds its rows on every read, so it keeps max_length and turns padding-free off.
                    # Schema first, a row only when free: next(iter(_ds)) on a single-pass stream is a row the run
                    # then trains without. An unprobeable stream answers True, holding max_length.
                    "        def _unsloth_pretokenized(_ds):\n"
                    "            try:\n"
                    # Columns first, EXCEPT for a transform: a with_transform split reports its backing table, so
                    # one storing `text` and yielding overlength input_ids read "raw" and cleared max_length. Its
                    # rows are rebuilt on every read, so probing one costs nothing.
                    "                _cols = None if _unsloth_is_transformed(_ds) else getattr(_ds, 'column_names', None)\n"
                    "                if isinstance(_cols, dict):\n"
                    "                    _cols = [_c for _v in _cols.values() for _c in (_v or [])]\n"
                    "                if _cols is not None: return 'input_ids' in _cols\n"
                    "                _probe = iter(_ds)\n"
                    "                if _probe is _ds or _probe is iter(_ds): return True\n"
                    "                _row = next(_probe, None)\n"
                    "            except Exception: return True\n"
                    "            return isinstance(_row, dict) and 'input_ids' in _row\n"
                    # Every rank reaches this before TRL's _prepare_dataset, which runs its own maps under
                    # main_process_first. Without the same window, eight ranks start num_proc workers each against
                    # one Arrow cache. A single process gets a no-op context manager.
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
                    # TRL filters these right after truncating: a row whose prompt fills the cap is all -100 and
                    # contributes no loss. labels is only one of three supervision signals; completion-only and
                    # assistant-only rows carry completion_mask / assistant_masks instead.
                    # A mask is supervision when TRL will apply it, and the two are not symmetric:
                    # DataCollatorForLanguageModeling gates completion_mask on completion_only_loss but applies
                    # assistant_masks on presence, so gating on the flag left an all-zero mask and an all -100 row.
                    # A None completion_only_loss is NOT "on": TRL resolves it from the dataset shape (prompt plus
                    # completion), and a pre-tokenized split has neither, so treating None as enabled deleted rows
                    # with valid full-sequence supervision.
                    "            _unsloth_cols = getattr(_new, 'column_names', None) or ()\n"
                    # One mode for every split, from the TRAIN sample, because that is what the collator uses.
                    # Per-split resolution disagreed with it whenever the schemas differ, so rows whose mask
                    # truncated to all zeros survived and went all -100 at eval.
                    "            _unsloth_masks = []\n"
                    "            if _unsloth_completion_only and 'completion_mask' in _unsloth_cols:\n"
                    "                _unsloth_masks.append('completion_mask')\n"
                    "            if 'assistant_masks' in _unsloth_cols:\n"
                    "                _unsloth_masks.append('assistant_masks')\n"
                    "            try:\n"
                    # labels is unconditional: it IS the supervision. Intersect labels AND every active mask
                    # rather than filtering each, since masks are applied one after another onto the labels; zip
                    # stops at the shorter, which is what an intersection means for a ragged pair.
                    "                _unsloth_supervision = (['labels'] if 'labels' in _unsloth_cols else []) + _unsloth_masks\n"
                    # The masks are applied one after another onto the same labels, so what survives is their
                    # INTERSECTION. Filtering each on its own kept rows whose two masks light up in different
                    # positions, which TRL then labels all -100 -- the very rows this filter exists to drop. zip
                    # stops at the shorter, which is what an intersection means for a ragged pair.
                    "                if _unsloth_supervision:\n"
                    "                    _new = _new.filter(lambda _e, _c = tuple(_unsloth_supervision): any(all((_x != -100) if _n == 'labels' else _x for _n, _x in zip(_c, _v)) for _v in zip(*[_e[_n] for _n in _c])), **_kw)\n"
                    "            except Exception:\n"
                    "                pass\n"
                    # Recorded, not raised: the caller wraps these calls in a broad `except Exception` that would turn a
                    # raise into "could not truncate", so the raise happens past that handler.
                    "            try:\n"
                    "                if _unsloth_supervision and len(_new) == 0: _unsloth_emptied.append(1)\n"
                    "            except TypeError:\n"
                    "                pass\n"
                    "            return _new, (True if _unsloth_is_stream(_new) else _unsloth_within_cap(_new))\n"
                    # Resolved BEFORE the try, since the fallback needs it even when the try never ran.
                    # eval_packing is separate from packing, so TRL may PACK the eval split instead of truncating
                    # it: keep max_length and turn padding-free off, which packing requires anyway.
                    "        _unsloth_emptied = []\n"
                    "        _unsloth_orig_train = train_dataset\n"
                    "        _unsloth_orig_eval = eval_dataset if 'eval_dataset' in locals() else None\n"
                    "        try:\n"
                    "            _unsloth_capped = _unsloth_known_mode\n"
                    "            if not _unsloth_known_mode:\n"
                    "                print('Unsloth: `truncation_mode = ' + str(_unsloth_truncation_mode) + '` is not one of keep_start / keep_end, so `max_length` is not being enforced here.')\n"
                    # A raw train split is tokenized with the cap by prep, so leave it alone. `and`, not a plain
                    # assignment, which threw away the unknown-mode refusal seeded above and served an unhonoured
                    # truncation_mode as keep_start.
                    # _unsloth_known_mode too: clearing _unsloth_capped only drops the ENFORCEMENT claim while the
                    # slice still ran, so the fallback scanned an already-trimmed split. Leaving the split alone
                    # lets that scan see the real lengths.
                    "            if _unsloth_known_mode and not _unsloth_prep_truncates:\n"
                    "                train_dataset, _unsloth_split_ok = _unsloth_cap_split(train_dataset)\n"
                    "                _unsloth_capped = _unsloth_capped and _unsloth_split_ok\n"
                    # An eval split TRL will PACK must not be truncated first: the branch is gated on
                    # `not args.packing` while eval_packing resolves separately, and the wrapped strategy
                    # concatenates the stream before chunking. Drop the enforcement claim, not the split.
                    # Each eval split on its own: a raw one stays raw for the tokenizer pass that follows, and only a
                    # materialised tokenized one is cut.
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
                    # Splits that WERE rewritten keep their truncation: it is the cap the caller asked for, as
                    # TRL's truncate_dataset would apply it. Rolling them back for a sibling that cannot be
                    # rewritten put an overlength train set back; only the claim of enforcement is dropped.
                    "            if not _unsloth_capped:\n"
                    "                print('Unsloth: `max_length` cannot be enforced for every split here, so padding-free batching is being turned off instead.')\n"
                    "        except Exception as _unsloth_truncate_error:\n"
                    "            train_dataset = _unsloth_orig_train\n"
                    "            if 'eval_dataset' in locals(): eval_dataset = _unsloth_orig_eval\n"
                    # The flag is decided from the train split, so a failure while capping an eval split would otherwise
                    # leave it reading "cap enforced".
                    "            _unsloth_prep_truncates = False\n"
                    # Never silent: a swallowed failure here reads as the cap being enforced.
                    "            print('Unsloth: could not truncate the pre-tokenized dataset to `max_length` (' + str(_unsloth_truncate_error) + ').')\n"
                    # Outside the handler on purpose: if every row loses its supervised tokens the cap sits below
                    # where supervision starts, and every TRL 1.x reads next(iter(train_dataset)) in __init__, so
                    # an empty split surfaces as a bare StopIteration naming nothing.
                    "        if _unsloth_emptied:\n"
                    "            raise ValueError('Unsloth: truncating to `max_length = ' + str(args.max_length) + '` left every row with no supervised token, so there is nothing to train on. The supervised part of your rows starts past that length: raise `max_length`, or set `truncation_mode = \"keep_end\"` if the completion sits at the end of each row.')\n"
                    # A producer that truncates every row enforces the cap as truncate_dataset would, so keep
                    # padding-free rather than pay the fallback; Unsloth's online tokenization is this shape. Not
                    # under eval_packing: that split is overlength on purpose.
                    "    if not _unsloth_prep_truncates and not _unsloth_eval_packing:\n"
                    "        def _unsloth_attests(_ds):\n"
                    "            if _ds is None: return True\n"
                    "            _own = getattr(_ds, '__dict__', None)\n"
                    "            if not isinstance(_own, dict): return False\n"
                    "            _claim = _own.get('_unsloth_truncated_to')\n"
                    "            if not isinstance(_claim, int) or isinstance(_claim, bool): return False\n"
                    "            return _claim <= _unsloth_cap\n"
                    "        _unsloth_attest_eval = eval_dataset if 'eval_dataset' in locals() else None\n"
                    "        _unsloth_attest_splits = list(_unsloth_attest_eval.values()) if isinstance(_unsloth_attest_eval, dict) else [_unsloth_attest_eval]\n"
                    "        if _unsloth_attests(train_dataset) and all(_unsloth_attests(_s) for _s in _unsloth_attest_splits):\n"
                    "            _unsloth_prep_truncates = True\n"
                    "    if _unsloth_prep_truncates:\n"
                    "        args.max_seq_length = args.max_length\n"
                    "        args.max_length = None\n"
                    "        max_length = None\n"
                    "    else:\n"
                    # Turning padding-free off keeps max_length for TRL's collator, which does not truncate, so
                    # rows already carrying input_ids are unenforced. TRL's own guard used to make this a hard
                    # error, so an observed overlength row must stay one rather than run silently uncapped.
                    # skip_prepare_dataset used to exempt this, the one way to get a silently uncapped run: TRL
                    # then neither truncates nor gives its collator a truncation length.
                    # An eval split left for the packer is overlength ON PURPOSE, so scanning it here turned a working
                    # eval-packing run into a hard error. The train split is still scanned: nothing packs that one.
                    "        _unsloth_scan_eval = None if _unsloth_eval_packing else (eval_dataset if 'eval_dataset' in locals() else None)\n"
                    "        if not (_unsloth_within_cap(train_dataset) and _unsloth_splits_within_cap(_unsloth_scan_eval)):\n"
                    "            raise ValueError('Unsloth: `max_length = ' + str(args.max_length) + '` cannot be enforced. Your dataset already carries `input_ids` and holds rows longer than that, and nothing downstream truncates pre-tokenized rows. Truncate it yourself before passing it in, or drop `max_length`.')\n"
                    "        print('Unsloth: Turning padding-free batching off, since your dataset is already tokenized and cannot be truncated here. Padding-free batching cannot enforce a `max_length` of ' + str(args.max_length) + '.')\n"
                    "        args.padding_free = False\n"
                )
            extra_args += max_length_check

    # Sync chat_template from processing_class to vLLM's tokenizer This fixes base models that have custom chat
    # templates applied after loading
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

        # Also swap when .pad is missing on a VLM. LM/Seq2Seq collators only: preference collators
        # (DPODataCollatorWithPadding etc.) keep their own prompt/chosen/rejected handling.
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

    if "model" in call_args:
        training_check = (
            "if hasattr(self, 'train'):\n"
            "    self.train = MethodType(prepare_for_training_mode(self.__class__.train), self)\n"
            "pass\n"
        )
        RLTrainer_post += training_check

    # Sync chat_template from processing_class to vLLM's tokenizer, which fixes base models that have
    # custom chat templates applied after loading.
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

    other_metrics_processor = ""
    if trainer_file in RL_METRICS_CHANGES:
        process_extra_args = RL_METRICS_CHANGES[trainer_file]
        for process_extra_arg in process_extra_args:
            other_metrics_processor += process_extra_arg(old_RLTrainer_source, old_RLConfig_source)

    extra_args += (
        "other_metrics = []\n"
        f"{other_metrics_processor}\n"
        "from unsloth_zoo.logging_utils import PatchRLStatistics\n"
        f"PatchRLStatistics('{trainer_file}', other_metrics)\n"
    )

    if trainer_file in RL_EXTRA_ARGS:
        process_extra_args = RL_EXTRA_ARGS[trainer_file]
        for process_extra_arg in process_extra_args:
            extra_args += process_extra_arg(call_args, extra_args)

    extra_args = extra_args.split("\n")
    extra_args = "\n".join(" " * 8 + x for x in extra_args)
    RLTrainer_post = RLTrainer_post.split("\n")
    RLTrainer_post = "\n".join(" " * 8 + x for x in RLTrainer_post)
    RLTrainer_arguments = arguments
    RLTrainer_extra_args = extra_args
    RLTrainer_call_args = call_args

    arguments, call_args = processed[1]
    extra_args = ""

    replacements = {
        "output_dir": None,
        "logging_nan_inf_filter": False,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        # LoRA decays A and B toward 0, so W = W_init + (alpha/r) * B @ A is pulled to W_init, not 0
        # as in full FT; 0.001 keeps a small Frobenius prior without dragging the adapter to base.
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
        # steps_per_generation would otherwise default to ga_steps, which is wrong, and
        # generation_batch_size clashes with it.
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
        # Might fail, so persistent dataloader workers / prefetch are disabled for now.
    }
    # warmup_ratio is deprecated in transformers >= 5.0; warmup_steps accepts a float.
    if transformers_version >= Version("5.0.0"):
        replacements["warmup_steps"] = 0.1
    else:
        replacements["warmup_ratio"] = 0.1

    for k, v in replacements.items():
        x = f"{k}( = [^,\n]{{1,}})?,\n"
        y = f"'{v}'" if type(v) is str else f"{v}"
        y = f"{k} = {y},\n"
        arguments = re.sub(x, y, arguments)

    # GRPO beta default is 0.001: TRL used 0.04 and now 0.00. See huggingface/trl#3516 and the verl docs.
    if trainer_file == "grpo_trainer":
        replacements = {
            "loss_type": "bnpo",  # Default GRPO paper
            "beta": 0.001,  # Recommended as seen in verl
            "auto_find_batch_size": False,  # Cannot work on GRPO
            # See fengyao.notion.site/off-policy-rl and huggingface/trl#3867.
            "vllm_importance_sampling_correction": False,
            # TRL >= 1.7.0 enables the MoE router aux loss by default (0.001), but the optimized GRPO forward
            # does not compute it, so default it off; opt in via router_aux_loss_coef > 0.
            "router_aux_loss_coef": 0.0,
        }
        for k, v in replacements.items():
            x = f"{k}( = [^,\n]{{1,}})?,\n"
            y = f"'{v}'" if type(v) is str else f"{v}"
            y = f"{k} = {y},\n"
            arguments = re.sub(x, y, arguments)

    # TRL >= 1.7.0 defaults SFT to loss_type="chunked_nll" (trl#5846), which patches the lm_head and
    # calls the backbone directly, so unsloth_fused_ce_loss never runs (ours chunks too and peaks
    # 1.7-3.7GB lower on gemma-3-4b at 141-8192 tokens), and it divides by num_items_in_batch ignoring
    # model_accepts_loss_kwargs, so on models setting that flag False (gemma3, qwen-vl, paligemma,
    # glm4v) training_step divides by grad-accum again and loss and grads are scaled 1/GA. Explicit
    # loss_type= still wins. Kept scoped to sft_trainer, since loss_type is an unrelated field in
    # DPO/KTO/GRPO.
    if trainer_file == "sft_trainer":
        replacements = {"loss_type": "nll"}
        for k, v in replacements.items():
            x = f"{k}( = [^,\n]{{1,}})?,\n"
            y = f"'{v}'" if type(v) is str else f"{v}"
            y = f"{k} = {y},\n"
            arguments = re.sub(x, y, arguments)

    if "learning_rate" in call_args:
        learning_rate_check = (
            "if learning_rate < 1e-7: print(f'Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! "
            "Consider increasing it, otherwise gradient updates will be close to 0!')\n"
            "if learning_rate > 1: print(f'Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! "
            "Consider decreasing it to 1e-1, otherwise gradient updates will explode!')\n"
        )
        extra_args += learning_rate_check

    # Fix num_train_epochs = None causing a TypeError in Trainer.__init__, which does `args.num_train_epochs > 0`.
    if "num_train_epochs" in call_args:
        num_train_epochs_check = (
            "if num_train_epochs is None:\n"
            "    num_train_epochs = 3.0  # Default to 3 epochs if None, max_steps will override\n"
        )
        extra_args += num_train_epochs_check

    # Check whether max_seq_length is NOT defined; max_length is now the default.
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

    if "output_dir" in call_args:
        saving_check = (
            "if output_dir is None and save_strategy == 'steps' and save_steps == 500:\n"
            "    output_dir = 'unsloth_training_checkpoints'\n"
            "    save_strategy = 'no'\n"
        )
        extra_args += saving_check

    # The worker-count policy lives in unsloth_zoo.dataset_num_proc: it had drifted into four
    # inline copies, two wrong (stdlib multiprocessing start method, and `1` as the serial
    # sentinel when datasets >= 4.1 builds a Pool(1)). In the zoo so generated source never
    # imports back into its generator.
    # serial_as_none depends on the reader: unsloth_zoo.sft_prepare_dataset reads a config None as
    # "auto-size me", so SFT writes serial as 1 and the map() call site turns it back. DPO, KTO,
    # CPO, ORPO, Reward and PRM pass it straight to Dataset.map, where 1 is a Pool(1).
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

    # Check for loss_type = dr_grpo and scale_rewards for GRPO; DAPO uses per-token loss, so BNPO loss
    # is used. See huggingface/trl#3130 (comment 2746947835).
    if "loss_type" in call_args and "scale_rewards" in call_args:
        # See https://github.com/huggingface/trl/issues/3130#issuecomment-2746947835 DAPO uses per token loss so
        # BNPO loss used
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

    # Check the GRPO num_generations mismatch; if world size is not set by accelerate or torchrun at this point it is 1.
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

    # Temperature must not be <= 0, and stop if >= 10.
    if "temperature" in call_args:
        check_temperature = (
            "if temperature <= 0:\n"
            "    raise ValueError('Unsloth: Please set a positive non-zero temperature since your results will be wrong.')\n"
            "elif temperature >= 10:\n"
            "    raise ValueError('Unsloth: Please set a positive non-zero temperature less than 10, since sampling will be quite erratic.')\n"
            "\n"
        )
        extra_args += check_temperature

    if trainer_file in RL_CONFIG_CHANGES:
        process_extra_args = RL_CONFIG_CHANGES[trainer_file]
        for process_extra_arg in process_extra_args:
            extra_args += process_extra_arg(old_RLTrainer_source, old_RLConfig_source)

    extra_args = extra_args.split("\n")
    extra_args = "\n".join(" " * 8 + x for x in extra_args)
    RLConfig_arguments = arguments
    RLConfig_extra_args = extra_args
    RLConfig_call_args = call_args

    # TRL 0.27.0+ forces use_reentrant=False in gradient_checkpointing_kwargs, but Unsloth gradient
    # checkpointing requires True, so remove the setting after super().__init__() applies it.
    RLConfig_post = ""
    if trl_version >= Version("0.27.0"):
        RLConfig_post = (
            "        # Unsloth: Remove use_reentrant=False forced by TRL 0.27.0+\n"
            "        if getattr(self, 'gradient_checkpointing_kwargs', None) is not None:\n"
            "            if 'use_reentrant' in self.gradient_checkpointing_kwargs:\n"
            "                del self.gradient_checkpointing_kwargs['use_reentrant']\n"
        )

    RLTrainer_extras = patch_functions(
        RLTrainer, trainer_file, RLTrainer_name, all_imports, imports
    )
    if RLTrainer_extras is None:
        RLTrainer_extras = f"_Unsloth{RLTrainer_name} = {RLTrainer_name}"

    exec(f"from trl.trainer import ({RLTrainer_name}, {RLConfig_name},)")
    __RLTrainer_doc__ = eval(f"trl.trainer.{RLTrainer_name}").__doc__
    if __RLTrainer_doc__ is None:
        __RLTrainer_doc__ = ""
    __RLConfig_doc__ = eval(f"trl.trainer.{RLConfig_name}").__doc__
    if __RLConfig_doc__ is None:
        __RLConfig_doc__ = ""

    if trainer_file in RL_PRE_ITEMS:
        RL_pre = "\n".join(RL_PRE_ITEMS[trainer_file])
    else:
        RL_pre = ""

    if "SamplingParams" in old_RLTrainer_source:
        RL_pre = RL_pre + "\n" + inspect.getsource(vLLMSamplingParams)

    selective_log_softmax_code = inspect.getsource(selective_log_softmax)
    grpo_selective_log_softmax_code = inspect.getsource(grpo_selective_log_softmax)
    calculate_pad_tokens_in_prompt_code = inspect.getsource(calculate_pad_tokens_in_prompt)
    create_completion_attention_mask_code = inspect.getsource(create_completion_attention_mask)
    left_pack_padding_code = inspect.getsource(left_pack_padding)
    align_logprobs_with_mask_code = inspect.getsource(align_logprobs_with_mask)
    align_completion_tool_mask_code = inspect.getsource(align_completion_tool_mask)
    autotune_batch_and_chunks_code = inspect.getsource(autotune_batch_and_chunks)
    sanitize_logprob_code = inspect.getsource(sanitize_logprob)
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
        # Base torch_compile_options shared by all device types; CUDA adds its own, and XPU / HIP / others
        # use the base only.
        base_options = """torch_compile_options = {
            "epilogue_fusion"   : True,
            "max_autotune"      : False,
            "shape_padding"     : True,
            "trace.enabled"     : False,"""

        if DEVICE_TYPE == "cuda":
            # CUDA-specific options (added to base options)
            cuda_options = """
            "triton.enable_persistent_tma_matmul": torch.cuda.get_device_capability()[0] >= 9,"""
            # cutlass options were added in PyTorch 2.8.0.
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
            # The `elif is_peft_model(model) and args.beta != 0.0:` ref-adapter block exists from TRL
            # 1.4.0 through 1.7.x. Anchored on the final ref_param copy so the following
            # enable_input_require_grads() block is not swallowed.
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
                # router_aux_loss_coef / aux_loss_enabled arrived in TRL 1.7.0, and the optimized GRPO forward
                # cannot compute the MoE router aux loss, so reject an explicit opt-in at init.
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

    # Remove TRL 0.26.0's unconditional bfloat16 cast of trainable params: it ignores the user's
    # dtype and breaks GradScaler with fp16=True. patch_model_and_tokenizer already handles it.
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

        # Do NOT override _is_vlm: forcing it False errors on vision datasets in TRL 0.27.1+. A bare
        # tokenizer as processing_class makes TRL set it False even for VLMs, so add an
        # architecture-based override before the validation check.
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

        # TRL 0.22.x keys off _is_vlm, not _is_vision_dataset (0.24.0+), so the vision-only signature
        # columns never overlap the tokenized ones. Merge both sets; _remove_unused_columns ignores extras.
        _sig_vlm_old = 'self._signature_columns = ["messages", "prompt", "completion", "images"]'
        _sig_vlm_new = (
            'self._signature_columns = ["messages", "prompt", "completion", "images",'
            ' "input_ids", "labels", "attention_mask", "seq_lengths", "completion_mask", "assistant_masks"]'
        )
        RLTrainer_source = RLTrainer_source.replace(_sig_vlm_old, _sig_vlm_new)

        RLTrainer_source = _backport_vision_dataset_gate(RLTrainer_source)

        # Inject the model reference before _prepare_dataset for dynamic token_type_ids detection in
        # sft_prepare_dataset.
        _prep_pattern = r"([ \t]*)train_dataset = self\._prepare_dataset\("
        _prep_replacement = (
            r"\1self._unsloth_model_ref = model\n\1train_dataset = self._prepare_dataset("
        )
        RLTrainer_source = re.sub(_prep_pattern, _prep_replacement, RLTrainer_source, count = 1)

    # Silence TRL's noisy batch_size=1 + padding-free warning, handling both the original "anihilate"
    # typo and the corrected spelling.
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

    # TRL converts a plain TrainingArguments with `args = <X>Config(**dict_args)`, resolved
    # through the generated module's globals, so it hands back a PRISTINE config: no unsloth
    # fields (#3931), and a class the first checkpoint save cannot pickle. Only the construction
    # is rewritten; the isinstance guard still wants the pristine class.
    RLTrainer_source = RLTrainer_source.replace(
        f"args = {RLConfig_name}(**dict_args)",
        f"args = Unsloth{RLConfig_name}(**dict_args)",
    )

    if __RLConfig_doc__ != "" and RLTrainer_source.count(__RLTrainer_doc__) == 2:
        RLTrainer_source = RLTrainer_source.replace(__RLTrainer_doc__, "", 1)

    RLTrainer_source = re.sub(r"[\n]{3,}", "\n", RLTrainer_source)

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
    _displaced_config = None
    # TRL 1.0.0+ wraps generation in: with torch.no_grad(), disable_gradient_checkpointing(self.model, ...): The
    # toggle only suppresses a cosmetic PyTorch warning; under no_grad it has no functional effect. But on exit
    # it calls gradient_checkpointing_enable(), overwriting Unsloth's custom "unsloth" wrapper -- for Gemma-4
    # this corrupts forward numerics and blows GRPO KL divergence up to ~10^12 at step 1. Replacing the context
    # manager with a no-op preserves Unsloth's wrapper. trl < 1.0.0 (no disable_gradient_checkpointing): early
    # return. trl >= 1.0.0: noop is correct; only loss is the cosmetic warning.
    try:
        config_module_name = trainer_file.replace("_trainer", "_config")
        config_module = importlib.import_module(f"trl.trainer.{config_module_name}")
        if hasattr(config_module, RLConfig_name):
            # Remember what this attribute held: on the TRL releases that put a thin wrapper here it is a class
            # of its own, and its instances need the same pickling fallback the pristine class gets.
            _displaced_config = getattr(config_module, RLConfig_name)
            setattr(
                config_module,
                RLConfig_name,
                getattr(created_module, f"Unsloth{RLConfig_name}"),
            )
    except Exception:
        pass
    _patched_config = getattr(created_module, f"Unsloth{RLConfig_name}", None)
    _patch_config_pickle_identity(RLConfig, _patched_config)
    if _patched_config is not None:
        _register_config_pickle_fallback(_displaced_config, _patched_config)

    if trainer_file == "sft_trainer":
        try:
            for _config_base in getattr(created_module, f"Unsloth{RLConfig_name}").__mro__[1:]:
                if not _is_unsloth_patched_config(_config_base):
                    _pin_pristine_sft_loss_type(_config_base)
                    break
        except Exception as e:
            logger.info(f"Unsloth: Could not pin the {RLConfig_name} loss_type: {e}")
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

    # Remove brackets in comments, replacing (...) with [...], since they interfere.
    comments = re.findall(r"\#[^\n]{1,}\n", init)
    bracketed_comments = [x for x in comments if "(" in x or ")" in x]
    for bracketed_comment in bracketed_comments:
        init = init.replace(
            bracketed_comment,
            bracketed_comment.replace("(", "[").replace(")", "]"),
        )

    init = init.replace("elif peft_config is None:", "elif False:")
    init = init.replace("elif peft_config is not None:", "elif False:")
    init = init.replace("if peft_config is None:", "if False:")
    init = init.replace("if peft_config is not None:", "if False:")
    init = init.replace("get_peft_model(model, peft_config)", "model")
    init = init.replace(
        "if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):",
        "if False:",
    )
    init = init.replace("model = self._prepare_peft_model(model, peft_config, args)\n", "pass\n")
    # TRL 0.22.0+ uses prepare_peft_model as a standalone function.
    init = init.replace("model = prepare_peft_model(model, peft_config, args)", "pass")

    # Skip add_adapter("ref"): the BASE model is the wanted reference, and PEFT forbids multiple
    # adapters under target_parameters (MoE). Without "ref", GRPO/RLOO falls back to
    # disable_adapter(), which is exactly the base model logits.
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
        commented_lines.append(
            f"{indent}# Unsloth: Commented out - use base model as reference, not SFT/LoRA model"
        )
        # Comment out each line by inserting # after the leading whitespace, to preserve indentation.
        for line in lines:
            if line.strip():
                stripped = line.lstrip()
                leading_ws = line[: len(line) - len(stripped)]
                commented_lines.append(f"{leading_ws}# {stripped}")
            else:
                commented_lines.append(line)
        return "\n".join(commented_lines)

    init = re.sub(add_adapter_block_pattern, comment_out_block, init, flags = re.DOTALL)

    if "args.use_vllm" in init and "model" in init and "args" in init:
        # .*? matches the first match, .+? the final one.
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

            if "grpo" in trainer_file and trl_version >= Version("0.18.0"):
                # If the model has a vllm_engine, use vllm in colocate mode and do not wait for a server.
                vllm_setter += " " * 12 + "args.vllm_mode='colocate'\n"
                if trl_version >= Version("0.23.0"):
                    # Align TRL sleep mode with the engine's actual enable_sleep_mode (the vision standby gate may have
                    # disabled it); fall back to the standby env var when the engine cannot be introspected.
                    vllm_setter += (
                        " " * 12
                        + "_unsloth_esm = getattr(getattr(getattr(getattr(model.vllm_engine, 'llm_engine', None), 'vllm_config', None), 'model_config', None), 'enable_sleep_mode', None)\n"
                        + " " * 12
                        + "if (_unsloth_esm if _unsloth_esm is not None else os.environ.get('UNSLOTH_VLLM_STANDBY', '0') != '0'):\n"
                        + " " * 16
                        + "args.vllm_enable_sleep_mode=True\n"
                    )

            init = init.replace(replacer, replacer + vllm_setter)

    vllm_part = re.findall(
        r"(\n[\s]{8}" r"if (self|args)\.use_vllm\:.*?" r"\n[\s]{8}" "else:\n)",
        init,
        flags = re.MULTILINE | re.DOTALL,
    )

    if len(vllm_part) == 1:
        vllm_part, args = vllm_part[0][0], vllm_part[0][1]
        new_vllm_part = re.sub(
            r"^\s*\#[^\n]*\n?", "", vllm_part, flags = re.MULTILINE
        )  # to also remove whole comment line instead of just starting at #
        new_vllm_part = re.sub(r"\s*\#.*$", "", new_vllm_part, flags = re.MULTILINE)

        sampling_params = re.findall(
            r"\n[\s]{4,}(self\.[^\s]{1,}[\s]{0,}\=[\s]{0,}SamplingParams\(.+?\))",
            new_vllm_part,
            flags = re.MULTILINE | re.DOTALL,
        )

        if len(sampling_params) == 1:
            sampling_params = sampling_params[0]
            sampling_params = sampling_params.replace(
                "guided_decoding=guided_decoding,",
                "guided_decoding="
                'GuidedDecodingParams(backend="outlines", regex=args.vllm_guided_decoding_regex) '
                'if getattr(args, "vllm_guided_decoding_regex", None) is not None else None,',
            )
            # Replace with our vLLM engine when sharing weights.
            sampling_params = (
                " " * 12
                + "if getattr(getattr(model, 'vllm_engine', None), 'shared_weights', False): "
                + "self.llm = model.vllm_engine; self._last_loaded_step = 0\n"
                + " " * 12
                + sampling_params
            )

            splitted_sampling_params = sampling_params.split("\n")
            if len(splitted_sampling_params) >= 2:
                last_line = splitted_sampling_params[-1]
                last_prev_line = splitted_sampling_params[-2]
                last_prev_indentation = len(last_prev_line) - len(last_prev_line.lstrip())
                last_indentation = len(last_line) - len(last_line.lstrip())

                extra = "**getattr(getattr(args, 'vllm_sampling_params', vLLMSamplingParams()), '_set_kwargs', {})"
                # Backwards replace.
                to_replace = (
                    ",\n"
                    + " " * last_prev_indentation
                    + extra
                    + ",\n"
                    + " " * last_indentation
                    + ")"
                )
                sampling_params = to_replace.join(sampling_params.rsplit(")", 1))
                sampling_params = re.sub(r"[\,][\s]{0,}\,", ",", sampling_params)

                new_vllm_part = (
                    f"\n{' ' * 8}if {args}.use_vllm:\n{sampling_params}\n{' ' * 8}else:\n"
                )

        if trl_version >= Version("0.18.0"):
            # Guard LLM init: use the existing vLLM engine when sharing weights, otherwise keep the original
            # LLM() creation for the sync/reload path.
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

    # Search for vLLM calls in all child functions.
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

        source = re.sub(
            r"(\n[\s]{4,}).+?model_executor\.driver_worker.+?\n",
            r"\n\1pass\n",
            source,
        )

        source = re.sub(
            r"(\n[\s]{4,}).+?load_weights\(.+?\n",
            r"\n\1pass\n",
            source,
        )

        source = re.sub(
            r"\.state_dict\(\)",
            r"",
            source,
        )

        # Replace self.llm.generate and self.llm.chat with lora_request, only when sharing weights.
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
        # Fix multiple commas before lora_request, in case the original code ends with ",)" as trl's
        # grpo_trainer.py#L1388 does.
        source = re.sub(r"\,[\s]{1,}\,[\s]{0,}lora_request", ", lora_request", source)
        source = re.sub(r"[\s]{1,}\,[\s]{0,}lora_request", ", lora_request", source)
        source = re.sub(r"[\,]{1,}[\s]{0,}lora_request", ", lora_request", source)
        # Prefer Unsloth's sampling params and fall back to trl's; to be enabled once both these and
        # GRPOConfig params are combined.
        # Fix later versions of SamplingParams via grpo_update_SamplingParams.
        source = source.replace(
            "sampling_params = SamplingParams(**generation_kwargs)",
            "sampling_params = SamplingParams("
            "**grpo_update_SamplingParams("
            "SamplingParams, generation_kwargs, "
            "getattr(self.args, 'vllm_sampling_params', None)"
            ")"
            ")",
        )

        if source == original_source:
            continue

        imports += [x for x in all_imports if not x.startswith("_") and x in source]

        changed[function] = (
            original_source,
            source,
        )

    imports = list(set(imports))

    for function in changed:
        old, new = changed[function]
        RLTrainer_source = RLTrainer_source.replace(old, new)

    RLTrainer_source = RLTrainer_source.replace(
        f"class {RLTrainer_name}", f"class _Unsloth{RLTrainer_name}", 1
    )
    return RLTrainer_source


def patch_trl_rl_trainers():
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
    # TRL 1.0.0+ wraps generation in disable_gradient_checkpointing(), which on exit calls
    # gradient_checkpointing_enable() and overwrites Unsloth's "unsloth" wrapper: Gemma-4 forward
    # numerics corrupt and GRPO KL hits ~1e12 at step 1. A no-op CM keeps the wrapper.
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

    # Also rebind any trl.* module that imported the symbol by reference at import time, walking
    # sys.modules so every `from ...models.utils import disable_gradient_checkpointing` is caught.
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
        function()
    return


def patch_trl_vllm_generation():
    # trl moved vllm code to trl/generation/vllm_generation.py; patch it so it does not build a
    # second vLLM instance when fast_inference has one: wrap the multiline `self.llm = LLM(..)`.
    for function in RL_ADDITIONAL_FUNCTIONS["vllm_generation"]:
        logger.info(f"Unsloth: Patching trl VLLMGeneration with function: {function.__name__}")
        function()
    return


def PatchFastRL(algorithm = None, FastLanguageModel = None):
    if FastLanguageModel is not None:
        PatchRL(FastLanguageModel)
    # Under UNSLOTH_ALLOW_CPU=1 (CPU-only CI), skip TRL trainer rewriting so downstream
    # inspect.getsource(trl.SFTTrainer) drift detectors see the pristine upstream class.
    if os.environ.get("UNSLOTH_ALLOW_CPU", "0") == "1":
        return
    # Install the disable_gradient_checkpointing noop BEFORE patch_trl_rl_trainers, which imports
    # more trl.* submodules: anything imported after the sys.modules walk keeps the old binding.
    patch_trl_disable_gradient_checkpointing()
    patch_trl_rl_trainers()
    patch_trl_openenv()
    patch_trl_vllm_generation()
    if type(algorithm) is str and algorithm.islower():
        PatchRLStatistics(algorithm)
