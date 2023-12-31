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

from transformers.utils.notebook import (
    IntervalStrategy,
    NotebookTrainingTracker,
    NotebookProgressCallback,
)
from transformers.trainer import DEFAULT_PROGRESS_CALLBACK
from trl import DPOTrainer
import types

DPOTrainer_metrics = [
    "rewards/chosen",
    "rewards/rejected",
    "rewards/accuracies",
    "rewards/margins",
    "logps/rejected",
    "logps/chosen",
    "logits/rejected",
    "logits/chosen",
]

def NotebookProgressCallback_on_train_begin(self, args, state, control, **kwargs):
    self.first_column = "Epoch" if args.evaluation_strategy == IntervalStrategy.EPOCH else "Step"
    self.training_loss = 0
    self.last_log = 0
    column_names = [self.first_column] + ["Training Loss"]
    if args.evaluation_strategy != IntervalStrategy.NO:
        column_names.append("Validation Loss")
    column_names += [x.replace("/", " / ") for x in DPOTrainer_metrics]
    self.training_tracker = NotebookTrainingTracker(state.max_steps, column_names)
pass


def NotebookProgressCallback_on_log(self, args, state, control, logs=None, **kwargs):
    # Only for when there is no evaluation
    if args.evaluation_strategy == IntervalStrategy.NO and "loss" in logs:
        values = {"Training Loss": logs["loss"]}
        for metric in DPOTrainer_metrics:
            if metric in logs:
                values[metric.replace("/", " / ")] = logs[metric]
            else:
                # Maybe not a DPO Trainer anymore? Redo the tracker
                column_names = [self.first_column] + ["Training Loss"]
                if args.evaluation_strategy != IntervalStrategy.NO:
                    column_names.append("Validation Loss")
                    self.training_tracker = NotebookTrainingTracker(state.max_steps, column_names)
                break
            pass
        pass
        # First column is necessarily Step since we're not in epoch eval strategy
        values["Step"] = state.global_step
        self.training_tracker.write_line(values)
    pass
pass


class FastDPOTrainer(DPOTrainer):
    # Patch DPO notebook printing
    if (DEFAULT_PROGRESS_CALLBACK is NotebookProgressCallback):

        DEFAULT_PROGRESS_CALLBACK.on_train_begin = types.MethodType(
            NotebookProgressCallback_on_train_begin,
            DEFAULT_PROGRESS_CALLBACK,
        )
        DEFAULT_PROGRESS_CALLBACK.on_log = types.MethodType(
            NotebookProgressCallback_on_log,
            DEFAULT_PROGRESS_CALLBACK,
        )
    pass
pass
