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
    "PatchDPOTrainer",
    "PatchKTOTrainer",
]


def PatchDPOTrainer():
    """Patch the TRL DPOTrainer to accept ``DPOConfig`` (trl >= 0.7) in addition
    to the legacy ``TrainingArguments``.

    When running DPO fine-tuning use ``trl.DPOConfig`` (or
    ``unsloth.UnslothDPOConfig``) instead of ``transformers.TrainingArguments``
    to receive DPO-specific hyper-parameters such as ``beta`` and
    ``loss_type``.  This helper ensures backward-compatibility so older code
    that passes ``TrainingArguments`` directly still works.

    Example::

        from trl import DPOConfig, DPOTrainer
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(...)
        model = FastLanguageModel.get_peft_model(model, ...)

        training_args = DPOConfig(
            beta=0.1,
            loss_type="sigmoid",
            output_dir="outputs",
            per_device_train_batch_size=2,
            num_train_epochs=3,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
    """
    try:
        from trl import DPOConfig, DPOTrainer
        import transformers

        _original_init = DPOTrainer.__init__

        def _patched_init(self, *args, **kwargs):
            # Accept either DPOConfig or legacy TrainingArguments
            training_args = kwargs.get("args")
            if training_args is None and len(args) > 2:
                training_args = args[2]
            if isinstance(training_args, transformers.TrainingArguments) and not isinstance(
                training_args, DPOConfig
            ):
                import warnings
                warnings.warn(
                    f"Unsloth: Passing '{type(training_args).__name__}' to DPOTrainer is deprecated. "
                    "Please use 'trl.DPOConfig' (or 'unsloth.UnslothDPOConfig') instead. "
                    "See: https://huggingface.co/docs/trl/dpo_trainer#trl.DPOConfig",
                    DeprecationWarning,
                    stacklevel = 2,
                )
            return _original_init(self, *args, **kwargs)

        DPOTrainer.__init__ = _patched_init
    except (ImportError, Exception):
        pass


def PatchKTOTrainer():
    return
