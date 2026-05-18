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

"""Training callbacks for monitoring activation statistics during fine-tuning."""

from __future__ import annotations

import math
from collections import deque
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

__all__ = ["ActivationNoveltyCallback"]


def _find_mlp_layers(model: nn.Module) -> List[nn.Module]:
    """Return MLP sub-modules in forward order by walking named_modules."""
    mlp_layers = []
    for name, mod in model.named_modules():
        basename = name.split(".")[-1]
        if basename in ("mlp", "feed_forward", "ffn"):
            mlp_layers.append(mod)
    return mlp_layers


class ActivationNoveltyCallback(TrainerCallback):
    """Tracks activation entropy of a chosen MLP layer during fine-tuning.

    Novelty is the normalised Shannon entropy of the mean absolute activation
    distribution across a batch.  Near 1.0 means capacity is used evenly;
    near 0.0 means a few neurons dominate (representation plateau).

    The metric is logged as ``log_key`` (default ``"activation_novelty"``) at
    every evaluation step and is compatible with WandB, TensorBoard, and any
    other logger attached to the trainer.

    When ``early_stop=True`` and novelty stays below ``novelty_threshold`` for
    ``window`` consecutive evaluations, training is halted automatically.  This
    is useful for catching runs that have plateaued well before ``num_epochs``
    and would otherwise waste compute.

    Args:
        layer_idx:          Which MLP block to monitor (0-indexed; -1 = last).
        novelty_threshold:  Novelty floor for early-stop logic (default 0.1).
        window:             Consecutive low-novelty evals required to stop.
        early_stop:         Halt training when plateau is detected.
        log_key:            Key written to trainer logs / WandB / TensorBoard.
        layer_getter:       Optional ``(model) -> nn.Module`` override.  When
                            provided ``layer_idx`` is ignored.  Useful for
                            non-standard architectures.

    Example::

        from unsloth import FastLanguageModel
        from unsloth.callbacks import ActivationNoveltyCallback

        model, tokenizer = FastLanguageModel.from_pretrained(...)
        model = FastLanguageModel.get_peft_model(model, ...)

        callback = ActivationNoveltyCallback(
            layer_idx = -1,
            early_stop = True,
            novelty_threshold = 0.08,
            window = 3,
        )

        trainer = UnslothTrainer(
            model = model,
            callbacks = [callback],
            ...
        )
        trainer.train()
    """

    def __init__(
        self,
        layer_idx: int = -1,
        novelty_threshold: float = 0.1,
        window: int = 3,
        early_stop: bool = False,
        log_key: str = "activation_novelty",
        layer_getter: Optional[Callable[[nn.Module], nn.Module]] = None,
    ) -> None:
        self.layer_idx = layer_idx
        self.novelty_threshold = novelty_threshold
        self.window = window
        self.early_stop = early_stop
        self.log_key = log_key
        self.layer_getter = layer_getter

        self._handle = None
        self._activations: List[torch.Tensor] = []
        self._history: deque = deque(maxlen = window)
        self._last_novelty: float = 1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_target_layer(self, model: nn.Module) -> Optional[nn.Module]:
        if self.layer_getter is not None:
            return self.layer_getter(model)
        mlp_layers = _find_mlp_layers(model)
        if not mlp_layers:
            return None
        return mlp_layers[self.layer_idx]

    def _register_hook(self, model: nn.Module) -> None:
        layer = self._get_target_layer(model)
        if layer is None:
            print(
                "Unsloth (ActivationNoveltyCallback): could not locate an MLP layer. "
                "Pass layer_getter= to specify the target module manually."
            )
            return

        def _hook(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.dim() > 2:
                out = out.reshape(out.size(0), -1)
            self._activations.append(out.detach().float())

        self._handle = layer.register_forward_hook(_hook)

    def _remove_hook(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _compute_novelty(self) -> float:
        if not self._activations:
            return self._last_novelty
        acts = torch.cat(self._activations, dim = 0)  # (N, D)
        mean_abs = acts.abs().mean(dim = 0)  # (D,)
        total = mean_abs.sum()
        if total < 1e-10:
            return self._last_novelty
        p = mean_abs / total
        d = p.numel()
        H = -(p * (p + 1e-10).log()).sum().item()
        return max(0.0, min(1.0, H / math.log(max(d, 2))))

    # ------------------------------------------------------------------
    # TrainerCallback interface
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        self._activations.clear()
        self._history.clear()
        self._last_novelty = 1.0
        if model is not None:
            self._register_hook(model)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[nn.Module] = None,
        **kwargs,
    ) -> TrainerControl:
        novelty = self._compute_novelty()
        self._activations.clear()
        self._last_novelty = novelty
        self._history.append(novelty)

        print(
            f"Unsloth (ActivationNoveltyCallback): "
            f"step {state.global_step} | {self.log_key} = {novelty:.4f}"
        )

        if (
            self.early_stop
            and len(self._history) == self.window
            and all(v < self.novelty_threshold for v in self._history)
        ):
            print(
                f"Unsloth (ActivationNoveltyCallback): novelty below "
                f"{self.novelty_threshold} for {self.window} consecutive evals — "
                f"stopping training."
            )
            control.should_training_stop = True

        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if logs is not None:
            logs[self.log_key] = self._last_novelty

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._remove_hook()
        self._activations.clear()
