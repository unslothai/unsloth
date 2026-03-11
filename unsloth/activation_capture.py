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

"""
Neuron activation capture for Unsloth Studio visualization.

Hooks into decoder layer outputs during finetuning to compile lightweight
per-channel statistics, which can be consumed by the Unsloth Studio JS UI
to animate a 3D block of neurons and show how finetuning reshapes the model.

Output layout (written to ``output_dir/``):
    metadata.json          — model config, capture config, channel indices
    activation_log.jsonl   — one JSON record per captured step (stream-friendly)

Each JSONL record has the shape::

    {
        "step": 100,
        "loss": 1.234,
        "layers": {
            "0": {"mean_abs": [float, ...N_CHANNELS], "mean": [float, ...N_CHANNELS]},
            "1": {...},
            ...
        }
    }

``mean_abs`` encodes brightness / activity level of each neuron channel.
``mean``     encodes polarity (positive vs negative activation direction).

Both are averaged across (batch × sequence_length) for the captured step.

Usage::

    from unsloth import FastLanguageModel
    from unsloth.activation_capture import (
        ActivationCaptureConfig,
        ActivationCapture,
        ActivationCaptureCallback,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(...)
    model = FastLanguageModel.get_peft_model(model, ...)

    capture_config = ActivationCaptureConfig(
        output_dir = "my_run/activations",
        capture_interval = 10,   # record every 10 steps
        max_channels = 64,       # channels per layer kept in the log
    )
    capture = ActivationCapture(model, capture_config)
    callback = ActivationCaptureCallback(capture)

    trainer = SFTTrainer(model=model, ..., callbacks=[callback])
    trainer.train()
    # -> my_run/activations/metadata.json
    # -> my_run/activations/activation_log.jsonl
"""

__all__ = [
    "ActivationCaptureConfig",
    "ActivationCapture",
    "ActivationCaptureCallback",
]

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ActivationCaptureConfig:
    """Configuration for the activation capture system.

    Attributes:
        output_dir:        Directory where ``metadata.json`` and
                           ``activation_log.jsonl`` are written.
        capture_interval:  Record activations every N optimizer steps.
                           Lower values give smoother animations but larger files.
                           Default: 10.
        max_channels:      Number of hidden-state channels (dimensions) to
                           track per layer. Channels are sampled once at init
                           and kept fixed for the whole run so the JS side can
                           display a stable grid.  Default: 64.
        capture_mlp_out:   If True, hook the MLP sub-module output in addition
                           to the full decoder-layer output.  Gives a more
                           "neuron-level" view but doubles storage.
                           Default: False.
        seed:              Random seed for reproducible channel sampling.
    """
    output_dir: str = "activation_logs"
    capture_interval: int = 10
    max_channels: int = 64
    capture_mlp_out: bool = False
    seed: int = 42


# ---------------------------------------------------------------------------
# Core capture logic
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Registers PyTorch forward hooks on transformer decoder layers and
    accumulates lightweight activation statistics during training.

    The hook fires on every forward pass but only stores data when
    ``mark_capture()`` has been called since the last ``flush()``.  This
    means there is zero overhead (a single bool check) on uncaptured steps,
    and gradient-checkpointing re-runs are deduplicated automatically.
    """

    def __init__(self, model, config: ActivationCaptureConfig):
        self.config = config
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        # { layer_idx -> {"mean_abs": [...], "mean": [...]} }
        self._buffer: Dict[int, Dict[str, List[float]]] = {}
        # Tracks which layer indices have already been captured this step
        # so gradient-checkpointing re-runs don't overwrite clean data.
        self._captured_layers: set = set()
        self._should_capture: bool = False
        self._step: int = 0
        self._loss: Optional[float] = None

        os.makedirs(config.output_dir, exist_ok=True)

        # ---- locate decoder layers (handles PEFT wrapping) ----------------
        # PeftModel:  model  →  .base_model  →  .model  →  .model  →  .layers
        # Plain HF:   model  →  .model  →  .layers
        candidate = model
        layers = None
        for _ in range(5):  # unwrap at most 5 levels of nesting
            layers = getattr(candidate, "layers", None)
            if layers is not None:
                break
            # try going one level deeper via common attribute names
            for attr in ("base_model", "model"):
                deeper = getattr(candidate, attr, None)
                if deeper is not None and deeper is not candidate:
                    candidate = deeper
                    break
            else:
                break  # nothing left to unwrap

        if layers is None:
            raise ValueError(
                "ActivationCapture: could not locate '.layers' on the model. "
                "Expected a standard HuggingFace causal LM (possibly wrapped by PEFT). "
                "Check that the model has a model.model.layers (or similar) structure."
            )
        self._layers = layers

        # ---- read config dimensions ---------------------------------------
        raw_cfg = None
        for obj in (model, getattr(model, "model", None), getattr(getattr(model, "model", None), "model", None)):
            if obj is not None and hasattr(obj, "config"):
                raw_cfg = obj.config
                break

        self._hidden_size      = getattr(raw_cfg, "hidden_size", None)
        self._intermediate_size= getattr(raw_cfg, "intermediate_size", None)
        model_name             = getattr(raw_cfg, "_name_or_path", "unknown")

        # ---- sample channels for display ----------------------------------
        n_total = self._hidden_size or config.max_channels
        rng = random.Random(config.seed)
        if n_total > config.max_channels:
            self._sampled_channels = sorted(
                rng.sample(range(n_total), config.max_channels)
            )
        else:
            self._sampled_channels = list(range(n_total))

        # ---- write metadata once ------------------------------------------
        metadata = {
            "model_name":         model_name,
            "num_layers":         len(self._layers),
            "hidden_size":        self._hidden_size,
            "intermediate_size":  self._intermediate_size,
            "captured_channels":  self._sampled_channels,
            "capture_interval":   config.capture_interval,
            "max_channels":       config.max_channels,
            "capture_mlp_out":    config.capture_mlp_out,
            "created_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(os.path.join(config.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # ---- prepare output file -----------------------------------------
        self._log_path = os.path.join(config.output_dir, "activation_log.jsonl")
        open(self._log_path, "w").close()  # truncate / create

        logger.info(
            "ActivationCapture: %d channels × %d layers → %s",
            len(self._sampled_channels), len(self._layers), config.output_dir,
        )

    # ------------------------------------------------------------------
    # Hook registration / removal
    # ------------------------------------------------------------------

    def attach(self):
        """Register forward hooks on all decoder layers (and optionally MLPs)."""
        for idx, layer in enumerate(self._layers):
            h = layer.register_forward_hook(self._make_layer_hook(idx, kind="layer"))
            self._hooks.append(h)
            if self.config.capture_mlp_out:
                mlp = getattr(layer, "mlp", None)
                if mlp is not None:
                    h2 = mlp.register_forward_hook(
                        self._make_layer_hook(idx, kind="mlp")
                    )
                    self._hooks.append(h2)
        logger.debug("ActivationCapture: %d hooks attached.", len(self._hooks))

    def detach(self):
        """Remove all registered hooks (call after training finishes)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.debug("ActivationCapture: all hooks removed.")

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    def _make_layer_hook(self, layer_idx: int, kind: str):
        """Return a forward hook closure for the given layer index and kind."""
        sampled = self._sampled_channels
        n_channels = len(sampled)
        key = (layer_idx, kind)

        def hook(module, inputs, output):
            if not self._should_capture:
                return
            # Deduplicate: gradient-checkpointing re-runs the forward pass
            # for checkpointed segments; ignore the recomputed activation.
            if key in self._captured_layers:
                return

            # The standard decoder-layer output is (hidden_states, ...) or
            # just hidden_states for inference.  MLP output is always a tensor.
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                return

            with torch.no_grad():
                # Cast to float32 to handle bfloat16 / float16 uniformly
                h = hidden.detach().float()          # [B, S, H]
                h_sampled = h[:, :, sampled]         # [B, S, C]
                flat = h_sampled.reshape(-1, n_channels)  # [B*S, C]
                mean_abs = flat.abs().mean(dim=0).tolist()
                mean     = flat.mean(dim=0).tolist()

            buf_key = layer_idx if kind == "layer" else f"{layer_idx}_mlp"
            self._buffer[buf_key] = {"mean_abs": mean_abs, "mean": mean}
            self._captured_layers.add(key)

        return hook

    # ------------------------------------------------------------------
    # Capture control
    # ------------------------------------------------------------------

    def mark_capture(self, step: int, loss: Optional[float] = None):
        """Arm the hooks to capture on the next forward pass.

        Called by :class:`ActivationCaptureCallback` before each scheduled step.
        """
        self._should_capture = True
        self._step = step
        self._loss = loss
        self._captured_layers.clear()

    def flush(self):
        """Write the buffered stats to JSONL and reset capture state.

        Called by :class:`ActivationCaptureCallback` after each step completes.
        """
        if not self._buffer:
            self._should_capture = False
            return

        record = {
            "step":   self._step,
            "loss":   self._loss,
            "layers": {str(k): v for k, v in self._buffer.items()},
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

        self._buffer.clear()
        self._should_capture = False


# ---------------------------------------------------------------------------
# HuggingFace Trainer callback
# ---------------------------------------------------------------------------

class ActivationCaptureCallback(TrainerCallback):
    """Drives an :class:`ActivationCapture` instance from a HuggingFace Trainer.

    Pass an instance of this class in the ``callbacks`` list when constructing
    any HF-compatible trainer (``SFTTrainer``, ``GRPOTrainer``, etc.).

    The callback:
    * attaches hooks at the start of ``on_train_begin``
    * arms capture every ``capture_interval`` steps (and always on the last step)
    * flushes statistics after each step
    * detaches hooks on ``on_train_end``

    Example::

        from unsloth.activation_capture import (
            ActivationCaptureConfig, ActivationCapture, ActivationCaptureCallback
        )

        capture_cfg = ActivationCaptureConfig(output_dir="run/activations", capture_interval=5)
        capture     = ActivationCapture(model, capture_cfg)
        callback    = ActivationCaptureCallback(capture)

        trainer = SFTTrainer(model=model, ..., callbacks=[callback])
        trainer.train()
    """

    def __init__(self, capture: ActivationCapture):
        self.capture = capture

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        self.capture.attach()
        # Arm capture for step 0 so the pre-finetune baseline is recorded.
        self.capture.mark_capture(step=0, loss=None)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                      control: TrainerControl, **kwargs):
        step = state.global_step
        interval = self.capture.config.capture_interval
        is_on_interval = (step % interval == 0)
        is_last_step   = (state.max_steps > 0 and step == state.max_steps - 1)

        if is_on_interval or is_last_step:
            loss = (
                state.log_history[-1].get("loss")
                if state.log_history else None
            )
            self.capture.mark_capture(step=step, loss=loss)

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        self.capture.flush()

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        # Final flush in case the very last step was on an interval boundary
        # and the callback had armed capture but training ended atomically.
        self.capture.flush()
        self.capture.detach()
        print(
            f"🦥 Unsloth: Activation log saved to "
            f"'{self.capture.config.output_dir}' "
            f"(metadata.json + activation_log.jsonl)"
        )
