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

def PatchRL(FastLanguageModel):

    from trl.models.utils import unwrap_model_for_generation
    from contextlib import contextmanager

    @contextmanager
    def unsloth_unwrap_model_for_generation(model, accelerator):
        with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
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

    import trl.trainer
    trainers = dir(trl.trainer)
    trainers = [x for x in trainers if x.endswith("_trainer")]
    unwrap = "unwrap_model_for_generation"
    for trainer in trainers:
        if hasattr(eval(f"trl.trainer.{trainer}"), unwrap):
            exec(f"trl.trainer.{trainer}.{unwrap} = unsloth_{unwrap}")
    pass
pass


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
                values[metric.replace("/", " / ")] = logs[metric]
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

        # Remove optional items
        # if ...: metrics[...] = 
        metrics_optional = re.findall(
            r"if[^\n]{1,}\n[\s]{4,}"\
            r"(?:metrics|stats)"\
            r"\["\
            r"(?:f[\"\']\{[^\}]{1,}\})?"\
            r"([^\"\']{1,})[\"\']"\
            r"\]",
            file,
            flags = re.MULTILINE,
        )
        metrics_optional = set(metrics_optional)
        metrics = [x for x in metrics if x not in metrics_optional]

        # Remove all eval_ things
        metrics = [x for x in metrics if not x.startswith("eval_")]

        all_metrics[trainer[:trainer.find("_")].upper()] = metrics
    pass
    return all_metrics
pass


def PatchRLStatistics(algorithm = "GRPO"):
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


def PatchFastRL(algorithm = "GRPO", FastLanguageModel = None):
    if FastLanguageModel is not None: PatchRL(FastLanguageModel)
    PatchRLStatistics(algorithm)
pass
