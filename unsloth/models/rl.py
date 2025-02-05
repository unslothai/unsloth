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
]


def PatchRL(FastLanguageModel):

    from trl.models.utils import unwrap_model_for_generation
    from contextlib import contextmanager

    @contextmanager
    def unsloth_unwrap_model_for_generation(model, *args, **kwargs):
        # Must use for_inference to allow inference in Unsloth
        FastLanguageModel.for_inference(model)
        with torch.inference_mode():
            with unwrap_model_for_generation(model, *args, **kwargs) as unwrapped_model:
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
