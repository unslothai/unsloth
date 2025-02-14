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
]

METRICS_MOVE_TO_END = [
    "nll",
    "aux",
    "beta",
    "alpha",
]
import torch
try:
    from transformers.utils.notebook import (
        IntervalStrategy,
        NotebookTrainingTracker,
        NotebookProgressCallback,
    )
    HAS_NOTEBOOK = True
except:
    HAS_NOTEBOOK = False
pass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import inspect
import os
import re
import functools
from unsloth_zoo.compiler import create_new_function


def PatchRL(FastLanguageModel):

    from trl.models.utils import unwrap_model_for_generation
    from trl.trainer.utils import first_true_indices, get_reward
    from contextlib import contextmanager

    @contextmanager
    def unsloth_unwrap_model_for_generation(model, *args, **kwargs):
        with unwrap_model_for_generation(model, *args, **kwargs) as unwrapped_model:
            # Put the model in inference mode.
            FastLanguageModel.for_inference(unwrapped_model)

            # We must use .clone for Unsloth since we force inference_mode
            # Rather we should have used no_grad
            original_generate = unwrapped_model.generate
            def generate_with_clone(*args, **kwargs):
                out = original_generate(*args, **kwargs)
                if isinstance(out, torch.Tensor):
                    return out.clone()
                return out
            pass
            unwrapped_model.generate = generate_with_clone

            try:
                yield unwrapped_model
            finally:
                # Restore generate and return
                unwrapped_model.generate = original_generate
                FastLanguageModel.for_training(model)
            pass
        pass
    pass
    def unsloth_get_reward(
        model, query_responses, pad_token_id, context_length):
            attention_mask = query_responses != pad_token_id
            position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
            lm_backbone = getattr(model, model.base_model_prefix)
            input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
            FastLanguageModel.reset_functions()
            output = lm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,  # otherwise mistral-based RM would error out
            )
            reward_logits = model.score(output.hidden_states[-1])
            sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
            FastLanguageModel.set_functions()
            # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
            return (
                reward_logits,
                reward_logits[
                    torch.arange(reward_logits.size(0), device=reward_logits.device),
                    sequence_lengths,
                ].squeeze(-1),
                sequence_lengths,
            )

    from transformers import Trainer
    from transformers.utils import is_sagemaker_mp_enabled
    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp
        from smdistributed.modelparallel import __version__ as SMP_VERSION

        IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

        from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
    else:
        IS_SAGEMAKER_MP_POST_1_10 = False
    from transformers.trainer_pt_utils import nested_detach
    
    #breakpoint()
    
    def unsloth_prediction_step(self, model, inputs, prediction_loss_only,ignore_keys,):
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
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
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

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        #breakpoint()
                        tokenized_output = self.processing_class(inputs["prompt"], padding=True, truncation=True, return_tensors="pt")
                        outputs = model(**tokenized_output)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    import trl.trainer
    trainers = dir(trl.trainer)

    trainers = [x for x in trainers if x.endswith("_trainer")]
    trainers.append("callbacks")
    unwrap = "unwrap_model_for_generation"
    _get_reward = "get_reward"
    for trainer in trainers:
        try: current_trainer = eval(f"trl.trainer.{trainer}")
        except: continue
        if hasattr(current_trainer, unwrap):
            try: exec(f"trl.trainer.{trainer}.{unwrap} = unsloth_{unwrap}")
            except: continue
        if hasattr(current_trainer, _get_reward):
            try: exec(f"trl.trainer.{trainer}.{_get_reward} = unsloth_{_get_reward}")
            except: continue
    exec(f"Trainer.prediction_step=unsloth_prediction_step")
    pass
pass
#To do, also in trl/trainer/callbacks.py make that unwrap compatible

def NotebookProgressCallback_on_train_begin(Trainer_metrics):
    def _NotebookProgressCallback_on_train_begin(self, args, state, control, **kwargs):
        self.first_column = "Epoch" if args.eval_strategy == IntervalStrategy.EPOCH else "Step"
        self.training_loss = 0
        self.last_log = 0
        column_names = [self.first_column] + ["Training Loss"]
        if args.eval_strategy != IntervalStrategy.NO:
            column_names.append("Validation Loss")
        column_names += [x.replace("/", " / ") for x in Trainer_metrics]
        self.training_tracker = NotebookTrainingTracker(state.max_steps, column_names)
    pass
    return _NotebookProgressCallback_on_train_begin
pass


def NotebookProgressCallback_on_log(Trainer_metrics):
    def _NotebookProgressCallback_on_log(self, args, state, control, logs=None, **kwargs):
        # Only for when there is no evaluation
        if args.eval_strategy == IntervalStrategy.NO and "loss" in logs:
            values = {"Training Loss": logs["loss"]}
            for metric in Trainer_metrics:
                # Sometimes metric is not inside logs
                try: values[metric.replace("/", " / ")] = logs[metric]
                except: pass
            pass
            # First column is necessarily Step since we're not in epoch eval strategy
            values["Step"] = state.global_step
            self.training_tracker.write_line(values)
        pass
    pass
    return _NotebookProgressCallback_on_log
pass


def NotebookTrainingTracker_write_line(Trainer_metrics):
    set_Trainer_metrics = set(Trainer_metrics)
    def _NotebookTrainingTracker_write_line(self, values):
        """
        Write the values in the inner table.

        Args:
            values (`Dict[str, float]`): The values to display.
        """
        if self.inner_table is None:
            self.inner_table = [list(values.keys()), list(values.values())]
        else:
            columns = self.inner_table[0]
            new_values = {}
            for key, value in values.items():
                lowered = key.lower()
                if lowered in set_Trainer_metrics:
                    new_values[lowered.replace("/", " / ")] = value
                else:
                    new_values[key] = value
            pass
            values = new_values

            self.inner_table[0] = columns
            if len(self.inner_table) > 1:
                last_values = self.inner_table[-1]
                first_column = self.inner_table[0][0]
                if last_values[0] != values[first_column]:
                    # write new line
                    self.inner_table.append([values[c] if c in values else "No Log" for c in columns])
                else:
                    # update last line
                    new_values = values
                    for c in columns:
                        if c not in new_values.keys():
                            new_values[c] = last_values[columns.index(c)]
                    self.inner_table[-1] = [new_values[c] for c in columns]
            else:
                # Edit for evaluation purposes
                self.inner_table.append([values[c] if c in values else 0 for c in columns])
            pass
        pass
    pass
    return _NotebookTrainingTracker_write_line
pass


def _PatchRLStatistics(metrics, algorithm):
    if HAS_NOTEBOOK:
        if len(metrics) == 0:
            raise RuntimeError(f"Unsloth: RL statistics for {algorithm} failed with no metrics seen?")
        from transformers.trainer import is_in_notebook
        if is_in_notebook():
            # Patch DPO notebook printing
            NotebookTrainingTracker.write_line = NotebookTrainingTracker_write_line(metrics)
            from transformers.trainer import DEFAULT_PROGRESS_CALLBACK
            DEFAULT_PROGRESS_CALLBACK.on_train_begin = NotebookProgressCallback_on_train_begin(metrics)
            DEFAULT_PROGRESS_CALLBACK.on_log         = NotebookProgressCallback_on_log(metrics)
        pass
    pass
pass


@functools.cache
def get_trl_metrics():
    # Gets metrics so we can output them in notebooks

    import trl.trainer
    trainers = dir(trl.trainer)
    trainers = [x for x in trainers if x.endswith("_trainer")]
    filepath = inspect.getfile(trl.trainer)
    filepath = os.path.split(filepath)[0]

    all_metrics = dict()
    for trainer in trainers:
        filename = os.path.join(filepath, f"{trainer}.py")
        if not os.path.exists(filename): continue
        with open(filename, "r") as file: file = file.read()

        # Get metrics['kl'] or stats['kl']
        metrics = re.findall(r"metrics\[[\"\']([^\"\']{1,})[\"\']\]", file)
        stats = re.findall(r"stats\[[\"\']([^\"\']{1,})[\"\']\]", file)
        metrics = metrics + stats

        # Get optional f-strings
        metrics_f = re.findall(r"metrics\[f[\"\']\{[^\}]{1,}\}([^\"\']{1,})[\"\']\]", file)
        stats_f = re.findall(r"stats\[f[\"\']\{[^\}]{1,}\}([^\"\']{1,})[\"\']\]", file)
        metrics_f = metrics_f + stats_f
        # Filter out prefixes if seen
        # metrics[f"{prefix}rewards/chosen"]
        left_prefix = 'prefix = "eval_" if train_eval == "eval" else ""' in file
        if left_prefix: metrics += metrics_f

        # Move all eval_ things to the end and reward to the front
        beginning = []
        middle = []
        end = []
        for x in metrics:
            lowered = x.lower()
            if "reward" in lowered:
                beginning.append(x)
            elif x.lower().startswith("eval"):
                end.append(x)
            else:
                # Check if we want to move to the end
                moved = False
                for move_end in METRICS_MOVE_TO_END:
                    if move_end in lowered:
                        end.append(x)
                        moved = True
                        break
                if not moved:
                    middle.append(x)
            pass
        pass
        metrics = beginning + middle + end

        all_metrics[trainer[:trainer.find("_")].upper()] = metrics
    pass
    return all_metrics
pass


def PatchRLStatistics(algorithm = "GRPO"):
    # Get notebook statistics columns to show up
    algorithm = algorithm.upper()
    all_metrics = get_trl_metrics()
    if algorithm not in all_metrics:
        print(
            f"Unsloth for {algorithm.upper()} is not yet implemented! Just ignore this function.\n"\
            f"We support: `{list(all_metrics.keys())}`"
        )
    pass
    _PatchRLStatistics(all_metrics[algorithm], algorithm)
pass


def _patch_trl_rl_trainers(trainer_file = "grpo_trainer"):
    # Patch for vLLM and Unsloth PEFT
    import trl
    import trl.trainer

    trainer = eval(f"trl.trainer.{trainer_file}")
    name = [x for x in dir(trainer) if x.endswith("Trainer") and x != "Trainer" and trainer_file.split("_")[0] in x.lower()]
    assert(len(name) == 1)
    RLTrainer_name = name[0]
    RLTrainer = eval(f"trl.trainer.{trainer_file}.{RLTrainer_name}")

    try:
        __init__ = inspect.getsource(RLTrainer.__init__)
    except:
        # Already patched most likely!
        return
    old__init__ = __init__
    all_imports = dir(trainer)
    assert("Union" in all_imports)
    imports = [x for x in all_imports if not x.startswith("_")]
    imports += ["Trainer"]

    spaces = __init__.find("def")
    __init__ = __init__.split("\n")
    __init__ = "\n".join(x[spaces:] for x in __init__)

    # Replace vLLM sections since we already have it done!
    vllm_part = re.findall(
        r"(\n[\s]{4}"\
        r"if (self|args)\.use_vllm\:.+?"\
        r"\n[\s]{4,}"\
        "else:\n)",
        __init__,
        flags = re.MULTILINE | re.DOTALL,
    )
    if (len(vllm_part) != 1): return

    vllm_part, args = vllm_part[0][0], vllm_part[0][1]
    # Strip all comments
    new_vllm_part = re.sub(r"\#[^\n]{1,}\n", "", vllm_part)

    # Get SamplingParams
    sampling_params = re.findall(
        r"\n[\s]{4,}(self\.[^\s]{1,}[\s]{0,}\=[\s]{0,}"\
        r"SamplingParams\(.+?\))",
        new_vllm_part,
        flags = re.MULTILINE | re.DOTALL,
    )
    if len(sampling_params) != 1: return

    sampling_params = sampling_params[0]
    # Replace with our vLLM engine
    sampling_params = \
        " "*8 + "self.llm = model.vllm_engine; self._last_loaded_step = 0; " + \
        sampling_params # Add spaces
    new_vllm_part = f"\n    if {args}.use_vllm:\n{sampling_params}\n    else:\n"
    __init__ = __init__.replace(vllm_part, new_vllm_part)

    # Remove peft_config
    __init__ = __init__.replace("elif peft_config is None:", "elif False:")
    __init__ = __init__.replace("elif peft_config is not None:", "elif False:")
    __init__ = __init__.replace("if peft_config is None:", "if False:")
    __init__ = __init__.replace("if peft_config is not None:", "if False:")
    __init__ = __init__.replace("get_peft_model(model, peft_config)", "model")

    # Add spaces back into __init__
    __init__ = __init__.split("\n")
    __init__ = "\n".join(' '*spaces + x for x in __init__)

    # Search for vLLM calling in all child functions
    functions = dir(RLTrainer)
    RLTrainer_source = inspect.getsource(RLTrainer)
    functions = [x for x in functions if f"def {x}" in RLTrainer_source]

    changed = {"__init__" : (old__init__, __init__,)}
    for function in functions:
        if not hasattr(RLTrainer, function): continue
        fx = getattr(RLTrainer, function)
        try:
            source = inspect.getsource(fx)
        except:
            continue
        original_source = source

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
        
        # Replace self.llm.generate and self.llm.chat
        lora_name = trainer_file + "_lora_model"
        source = re.sub(
            r"(self\.llm\.(?:generate|chat)\([^\)]{1,})\)",
            r"\1, lora_request = self.model.load_lora('" + lora_name + r"', load_tensors = True))",
            source
        )

        # Skip if no changes done
        if source == original_source: continue

        # Find all imports
        imports += [x for x in all_imports if not x.startswith("_") and x in source]

        changed[function] = (original_source, source,)
    pass

    # Import all functions
    imports = list(set(imports))

    # Patch all functions
    for function in changed:
        old, new = changed[function]
        RLTrainer_source = RLTrainer_source.replace(old, new)
    pass
    RLTrainer_source = RLTrainer_source.replace(
        f"class {RLTrainer_name}", f"class Unsloth{RLTrainer_name}", 1
    )

    # Create new class in compiled cache and import it
    module = create_new_function(
        RLTrainer_name,
        RLTrainer_source,
        f"trl.trainer.{trainer_file}",
        imports,
    )

    # Patch over modules
    exec(f"trl.{RLTrainer_name} = module.Unsloth{RLTrainer_name}", locals(), globals())
    exec(f"trl.trainer.{RLTrainer_name} = module.Unsloth{RLTrainer_name}", locals(), globals())
    exec(f"trl.trainer.{trainer_file}.{RLTrainer_name} = module.Unsloth{RLTrainer_name}", locals(), globals())
    return module
pass


def patch_trl_rl_trainers():
    # Patch all TRL modules if they have vLLM or PEFT
    import trl.trainer
    all_trainers = dir(trl.trainer)
    all_trainers = [x for x in all_trainers if x.islower() and x.endswith("_trainer")]
    for trainer in all_trainers:
        _patch_trl_rl_trainers(trainer)
    return
pass


def PatchFastRL(algorithm = "GRPO", FastLanguageModel = None):
    if FastLanguageModel is not None: PatchRL(FastLanguageModel)
    patch_trl_rl_trainers()
    PatchRLStatistics(algorithm)
pass
