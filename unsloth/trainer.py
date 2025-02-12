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
import inspect
from trl import SFTTrainer
from . import is_bfloat16_supported
from unsloth_zoo.training_utils import (
    unsloth_train as _unsloth_train,
)
from unsloth_zoo.vision_utils import (
    UnslothVisionDataCollator,
)
from packaging.version import Version
import dataclasses

__all__ = [
    "UnslothTrainingArguments",
    "UnslothTrainer",
    "unsloth_train",
    "_patch_trl_trainer",
    "UnslothVisionDataCollator",
]

# Unsloth gradient accumulation fix:
from transformers import __version__ as transformers_version
if Version(transformers_version) > Version("4.45.2"):
    def unsloth_train(trainer, *args, **kwargs):
        """Executes a custom training routine with gradient accumulation.

    This function serves as a wrapper for the custom gradient accumulation
    fixed trainer. It ensures that no additional arguments or keyword
    arguments are passed, as they are not supported. Users are advised to
    update their `transformers` package to the latest version if they require
    more features.

    Args:
        trainer: The training object or configuration to be used.

    Raises:
        RuntimeError: If any additional positional or keyword arguments
        are provided.

    Returns:
        The result of the `_unsloth_train` function, which runs the custom
        training routine."""
        return trainer.train(*args, **kwargs)
    pass
else:
    def unsloth_train(trainer, *args, **kwargs):
        """Executes a custom training routine with gradient accumulation.

    This function serves as a wrapper for the custom gradient accumulation
    fixed trainer. It ensures that no additional arguments or keyword
    arguments are passed, as they are not supported. Users are advised to
    update their `transformers` package to the latest version if they require
    more features.

    Args:
        trainer: The training object or configuration to be used.

    Raises:
        RuntimeError: If any additional positional or keyword arguments
        are provided.

    Returns:
        The result of the `_unsloth_train` function, which runs the custom
        training routine."""
        if len(args) != 0 or len(kwargs) != 0:
            raise RuntimeError(
                "Unsloth: Our custom gradient accumulation fixed trainer does not support other arguments.\n"\
                "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"\
                '`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`'
            )
        print(
            "Unsloth: Using our custom gradient accumulation fixed trainer, which is not feature complete.\n"\
            "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"\
            '`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`'
        )
        return _unsloth_train(trainer)
    pass
pass

try:
    from trl import SFTConfig as TrainingArguments
except:
    from transformers import TrainingArguments
pass
@dataclass
class UnslothTrainingArguments(TrainingArguments):
    """Arguments for configuring the training process with specific learning rates.

    This data class extends `TrainingArguments` to include an optional parameter
    for setting different learning rates specifically for embeddings and the
    language model head.

    Attributes:
        embedding_learning_rate (Optional[float]): Specifies a separate learning
            rate for embeddings and the language model head. If not set, the
            default learning rate from `TrainingArguments` is used for all
            components.

    Args:
        embedding_learning_rate: Optional; A float representing the learning rate
            for embeddings and lm_head. If `None`, the default learning rate is used.

    Returns:
        An instance of `UnslothTrainingArguments` configured with the specified
        learning rates."""
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
    """Creates an optimizer with separate learning rates for embeddings and non-embeddings.

    This function configures an optimizer for a model, applying a different learning
    rate to the model's embedding parameters than to the rest of the parameters.
    Parameters ending with 'modules_to_save.default.weight' are considered embedding
    parameters and are assigned a specified embedding learning rate.

    Args:
        model: The model containing parameters to be optimized.
        optimizer_cls: The class of the optimizer to be used.
        optimizer_kwargs: A dictionary of keyword arguments to be passed to the optimizer.
        embedding_lr: The learning rate to be applied to embedding parameters (default is 5e-5).

    Returns:
        An instance of the specified optimizer class with separate parameter groups for
        embeddings and non-embeddings, each with their respective learning rates."""
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
    """Trainer class that extends SFTTrainer to create a custom optimizer.

    The UnslothTrainer class overrides the `create_optimizer` method to allow 
    for a specific learning rate for embeddings if specified in the arguments.

    Methods:
        create_optimizer: Creates and returns an optimizer, using a custom 
        learning rate for embeddings if provided."""
    def create_optimizer(self):
        """Creates and returns an optimizer for model training.

    This method customizes the optimizer creation process by checking for a 
    specified `embedding_learning_rate` in the arguments. If such a rate is 
    defined, a specialized optimizer is created using this learning rate for 
    embeddings. Otherwise, the default optimizer creation process is invoked.

    Returns:
        Optimizer: The created optimizer instance for the model."""
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
def _backwards_compatible_trainer(trainer_class, config_class):
    """Modifies the initialization method of a trainer class for backward compatibility with newer versions.

    This function wraps the initialization method of the provided trainer class to ensure that
    parameters are correctly passed and processed, allowing for a smooth transition between
    different versions of a training library. Specifically, it adapts the parameter handling
    to accommodate changes introduced in version 0.13.0 of the `trl` library, such as the
    renaming of certain parameters and the introduction of new configuration handling.

    Args:
        trainer_class: The trainer class whose initialization method is to be modified.
        config_class: The configuration class used to create a configuration object with
            appropriate parameters extracted from the arguments.

    Returns:
        function: A new initialization function that replaces the original one in the trainer class,
        ensuring backward compatibility with the expected parameters."""
    original_init = trainer_class.__init__
    
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        """Initializes a Trainer-like object with updated arguments for compatibility.

    This function wraps the original initialization method of a Trainer-like
    object to ensure compatibility with newer versions of a library by adjusting
    the arguments passed to it. Specifically, it renames the 'tokenizer' argument
    to 'processing_class' if needed, and constructs a configuration object with
    parameters that have been moved from TrainingArguments to a separate Config
    class.

    Args:
        *args: Positional arguments passed to the original initialization method.
        **kwargs: Keyword arguments passed to the original initialization method.
            - 'processing_class': Replaces 'tokenizer' if present.
            - 'args': If present and the library version is 0.13.0.dev0 or newer,
              it is used to build a configuration object with relevant fields.

    Returns:
        None: This function does not return any value. It modifies the initialization
        process of the object it is applied to."""
        # All Trainer tokenizer are now called processing_class
        trainer_params = set(inspect.signature(original_init).parameters.keys())

        if "processing_class" in trainer_params and "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        pass

        if ("args" in kwargs) and (Version(trl.__version__) >= Version("0.13.0.dev0")):
            training_args = kwargs.pop("args", None)

            # Get parameters that Trainer.__init__ actually expects
            trainer_params.remove('self')
            trainer_params.remove('args')

            # Get fields that should be passed to Config init
            config_fields = {
                field.name: field for field in dataclasses.fields(config_class) 
                if field.init
            }
            
            # Create config dict with valid fields from training_args
            config_dict = {
                name: getattr(training_args, name)
                for name in config_fields
                if hasattr(training_args, name)
            }

            # Get parameters that exist in Config but not in TrainingArguments
            from transformers import TrainingArguments
            moved_params = \
                set(inspect.signature(config_class)     .parameters.keys()) - \
                set(inspect.signature(TrainingArguments).parameters.keys())
            
            # Separate kwargs into trainer kwargs and config kwargs
            trainer_kwargs = {}
            additional_config_kwargs = {}

            for key, value in kwargs.items():
                if key in trainer_params: trainer_kwargs[key] = value
                elif key in moved_params or key in config_fields:
                    additional_config_kwargs[key] = value
                else:
                    additional_config_kwargs[key] = value
                pass
            pass

            # Update config_dict with additional kwargs
            config_dict.update(additional_config_kwargs)

            # Create Config with all the collected parameters
            config = config_class(**config_dict)
            
            # Reconstruct kwargs for Trainer
            kwargs = trainer_kwargs
            kwargs["args"] = config
        pass
        original_init(self, *args, **kwargs)
    pass
    return new_init
pass


def _patch_trl_trainer():
    """Patches the training classes in the `trl` library for backwards compatibility.

    This function checks if the `trl` library has a specific backwards compatibility
    attribute set. If not, and if the library version is greater than 0.11.0, it identifies
    trainer and configuration classes within the `trl.trainer` module. It then modifies
    the `__init__` methods of these classes to ensure compatibility with older versions
    of the library. After patching, it sets an attribute on the `trl` module to indicate
    that the patch has been applied.

    This function does not take any arguments and does not return anything.

    Note:
        - This function relies on an external function `_backwards_compatible_trainer`
          which is assumed to adjust the `__init__` methods appropriately."""
    import trl
    if hasattr(trl, "__UNSLOTH_BACKWARDS_COMPATIBLE__"): return
    if Version(trl.__version__) <= Version("0.11.0"): return

    import trl.trainer
    trl_classes = dir(trl.trainer)
    trl_trainers = set(x[:-len("Trainer")] for x in trl_classes if x.endswith("Trainer"))
    trl_configs  = set(x[:-len("Config")]  for x in trl_classes if x.endswith("Config"))
    trl_classes = list(trl_trainers & trl_configs)

    for x in trl_classes:
        try:    exec(f"trl.{x}Trainer.__init__ = _backwards_compatible_trainer(trl.{x}Trainer, trl.{x}Config)", globals())
        except: continue
    pass

    trl.__UNSLOTH_BACKWARDS_COMPATIBLE__ = True
pass
