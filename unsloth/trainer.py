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

import warnings
from dataclasses import dataclass, field
from typing import Optional
from functools import wraps

import trl
from trl import SFTTrainer
try:
    from trl import SFTConfig as TrainingArguments
except:
    from transformers import TrainingArguments
pass
from . import is_bfloat16_supported
from unsloth_zoo.training_utils import unsloth_train as _unsloth_train
from packaging.version import Version

# Unsloth gradient accumulation fix:
from transformers import __version__ as transformers_version
if Version(transformers_version) > Version("4.45.2"):
    def unsloth_train(trainer):
        return trainer.train()
    pass
else:
    def unsloth_train(trainer):
        print(
            "Unsloth: Using our custom gradient accumulation fixed trainer, which is not feature complete.\n"\
            "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"\
            '`pip uninstall transformers -y && pip install --upgrade --no-cache-dir "git+https://github.com/huggingface/transformers.git"`'
        )
        return _unsloth_train(trainer)
    pass
pass

__all__ = [
    "UnslothTrainingArguments",
    "UnslothTrainer",
    "unsloth_train",
    "_patch_sft_trainer",
]


@dataclass
class UnslothTrainingArguments(TrainingArguments):
    embedding_learning_rate : Optional[float] = field(
        default = None,
        metadata = {"help" : "Different learning rates for embeddings and lm_head."}
    )
pass


def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr = 5e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = \
    {
        "non_embeddings" : {},
        "embeddings"     : {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[:-len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".")+1:]
            print(f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}.")
            param_groups["embeddings"]    [name] = param
        else:
            param_groups["non_embeddings"][name] = param
        pass
    pass

    optimizer_grouped_parameters = [
        {
            "params"       : list(param_groups["non_embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : lr,
        },
        {
            "params"       : list(param_groups["embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
pass


class UnslothTrainer(SFTTrainer):
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None: return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        pass
        return self.optimizer
    pass
pass

# From `trl>=0.13.0`, they changed how to pass several params to the trainer
# We need to patch to make the transition smooth

def _patch_sft_trainer():
    """
    Patches the SFTTrainer to maintain backward compatibility with the old syntax
    """
    import dataclasses

    original_init = SFTTrainer.__init__
    list_moved_kwargs = ['max_seq_length', 'dataset_num_proc', 'packing', 'dataset_text_field']

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        if Version(trl.__version__) >= Version("0.13.0.dev0"):
            if "args" in kwargs and not isinstance(kwargs["args"], trl.SFTConfig):
                warnings.warn(
                    "You are using TRL ≥0.13.0 with the old API style. While this will work for now, "
                    "consider updating your code to use SFTConfig in the future.",
                    DeprecationWarning,
                    stacklevel=2
                )

                training_args = kwargs.pop("args", None)

                # Need to manually add here by checking the fields
                # Since `TrainingArguments` is a subclass of `SFTConfig`
                # But has `post_init` argument which is not received
                # by the __init__ of `SFTConfig`

                # Get only the fields that should be passed to __init__
                sft_fields = {
                    field.name: field for field in dataclasses.fields(trl.SFTConfig) 
                    if field.init  # Only get fields where init=True
                }
                
                # Create config dict with only valid init fields
                config_dict = {
                    name: getattr(training_args, name)
                    for name in sft_fields
                    if hasattr(training_args, name)
                }
                
                # Add the parameters that were previously separate
                for param in list_moved_kwargs:
                    if param in kwargs:
                        config_dict[param] = kwargs.pop(param)

                
                sft_config = trl.SFTConfig(**config_dict)
                    
                kwargs["args"] = sft_config

            original_init(self, *args, **kwargs)
        else:
            original_init(self, *args, **kwargs)
                
    SFTTrainer.__init__ = new_init