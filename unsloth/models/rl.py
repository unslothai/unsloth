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
    "PatchRL",
    "PatchRLStatistics",
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


def PatchRL(FastLanguageModel):

    from trl.models.utils import unwrap_model_for_generation
    from contextlib import contextmanager

    @contextmanager
    def unsloth_unwrap_model_for_generation(model, *args, **kwargs):
        # Must use for_inference to allow inference in Unsloth
        print("$$$$$$$$$$$$$$$$$$$$$$$")
        with unwrap_model_for_generation(model, *args, **kwargs) as unwrapped_model:
            FastLanguageModel.for_inference(unwrapped_model)
            yield unwrapped_model
        # Return back to training mode
        FastLanguageModel.for_training (model)
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
            for metric in DPOTrainer_metrics:
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


def _PatchRLStatistics(metrics):
    if HAS_NOTEBOOK:
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


def PatchRLStatistics(algorithm = "GRPO"):
    algorithm = algorithm.upper()
    if algorithm == "GRPO":
        metrics = [
            "completion_length",
            "reward",
            "reward_std",
            "kl",
        ]
    elif algorithm == "DPO" or algorithm == "KTO":
        metrics = [
            "rewards/chosen",
            "rewards/rejected",
            "rewards/accuracies",
            "rewards/margins",
            "logps/rejected",
            "logps/chosen",
            "logits/rejected",
            "logits/chosen",
        ]
    else:
        print(f"Unsloth for {algorithm.upper()} is not yet implemented! Just ignore this function.")
    _PatchRLStatistics(metrics)
pass
