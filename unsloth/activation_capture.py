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
Neuron activation capture for training visualization.

Hooks into decoder layer outputs during finetuning to compile lightweight
per-channel statistics to animate a 2D or 3D block of neurons and show how 
finetuning reshapes the model.

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
        capture_gradients: If True, capture gradient norms per layer during
                           backward pass. Shows which layers are actively learning.
                           Default: True.
        capture_lora_norms: If True, capture LoRA adapter norms ||B @ A||_F per
                            layer. Shows which adapters are claiming capacity.
                            Only meaningful for PEFT/LoRA models.
                            Default: True.
        seed:              Random seed for reproducible channel sampling.
    """
    output_dir: str = "activation_logs"
    capture_interval: int = 10
    max_channels: int = 64
    capture_mlp_out: bool = False
    capture_gradients: bool = True
    capture_lora_norms: bool = True
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
        self._model = model  # keep reference for LoRA norm computation
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._grad_hooks: List[torch.utils.hooks.RemovableHook] = []
        # { layer_idx -> {"mean_abs": [...], "mean": [...]} }
        self._buffer: Dict[int, Dict[str, List[float]]] = {}
        # { layer_idx -> gradient_norm }
        self._grad_buffer: Dict[int, float] = {}
        # Tracks which layer indices have already been captured this step
        # so gradient-checkpointing re-runs don't overwrite clean data.
        self._captured_layers: set = set()
        self._captured_grad_layers: set = set()
        self._should_capture: bool = False
        self._step: int = 0
        self._loss: Optional[float] = None
        # Cache LoRA modules per layer for fast lookup
        self._lora_modules: Dict[int, List[tuple]] = {}

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

        # ---- discover LoRA modules per layer ----------------------------
        if config.capture_lora_norms:
            self._discover_lora_modules()

        # ---- write metadata once ------------------------------------------
        lora_targets = []
        if self._lora_modules:
            # Get unique target names across all layers
            all_targets = set()
            for mods in self._lora_modules.values():
                for name, _, _ in mods:
                    all_targets.add(name)
            lora_targets = sorted(all_targets)

        metadata = {
            "model_name":         model_name,
            "num_layers":         len(self._layers),
            "hidden_size":        self._hidden_size,
            "intermediate_size":  self._intermediate_size,
            "captured_channels":  self._sampled_channels,
            "capture_interval":   config.capture_interval,
            "max_channels":       config.max_channels,
            "capture_mlp_out":    config.capture_mlp_out,
            "capture_gradients":  config.capture_gradients,
            "capture_lora_norms": config.capture_lora_norms,
            "lora_targets":       lora_targets,
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
    # LoRA module discovery
    # ------------------------------------------------------------------

    def _discover_lora_modules(self):
        """Find LoRA adapters (lora_A, lora_B pairs) for each decoder layer.

        Populates ``self._lora_modules`` as:
            { layer_idx: [(target_name, lora_A, lora_B), ...] }

        where target_name is e.g. "q_proj", "v_proj", etc.
        """
        for idx, layer in enumerate(self._layers):
            layer_loras = []
            for name, module in layer.named_modules():
                # PEFT LoRA modules have lora_A and lora_B as submodules
                lora_A = getattr(module, "lora_A", None)
                lora_B = getattr(module, "lora_B", None)
                if lora_A is not None and lora_B is not None:
                    # Extract the target name (last part of the path)
                    # e.g., "self_attn.q_proj" -> "q_proj"
                    parts = name.split(".")
                    target_name = parts[-1] if parts else name
                    # lora_A and lora_B might be ModuleDict (multi-adapter)
                    # or direct Linear layers (single adapter)
                    if hasattr(lora_A, "default"):
                        # Multi-adapter: lora_A.default, lora_B.default
                        lora_A = lora_A.default
                        lora_B = lora_B.default
                    layer_loras.append((target_name, lora_A, lora_B))
            if layer_loras:
                self._lora_modules[idx] = layer_loras

        if self._lora_modules:
            total_adapters = sum(len(v) for v in self._lora_modules.values())
            logger.info(
                "ActivationCapture: Found %d LoRA adapters across %d layers",
                total_adapters, len(self._lora_modules)
            )
        else:
            logger.info("ActivationCapture: No LoRA adapters found (non-PEFT model?)")

    def _compute_lora_norms(self) -> Dict[int, Dict[str, float]]:
        """Compute ||B @ A||_F for each LoRA adapter per layer.

        Returns:
            { layer_idx: { target_name: frobenius_norm, ... }, ... }
        """
        lora_norms = {}
        for layer_idx, adapters in self._lora_modules.items():
            layer_norms = {}
            for target_name, lora_A, lora_B in adapters:
                try:
                    with torch.no_grad():
                        # lora_A.weight: [r, in_features]
                        # lora_B.weight: [out_features, r]
                        # effective delta: B @ A  -> [out_features, in_features]
                        A = lora_A.weight.float()
                        B = lora_B.weight.float()
                        delta = B @ A
                        norm = torch.linalg.matrix_norm(delta, ord="fro").item()
                        layer_norms[target_name] = norm
                except Exception as e:
                    logger.debug(f"Could not compute LoRA norm for layer {layer_idx} {target_name}: {e}")
            if layer_norms:
                lora_norms[layer_idx] = layer_norms
        return lora_norms

    # ------------------------------------------------------------------
    # Hook registration / removal
    # ------------------------------------------------------------------

    def attach(self):
        """Register forward hooks on all decoder layers (and optionally MLPs).

        Also registers backward hooks for gradient capture if enabled.
        """
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

            # Register backward hook for gradient capture
            if self.config.capture_gradients:
                gh = layer.register_full_backward_hook(self._make_grad_hook(idx))
                self._grad_hooks.append(gh)

        logger.debug(
            "ActivationCapture: %d forward hooks, %d backward hooks attached.",
            len(self._hooks), len(self._grad_hooks)
        )

    def detach(self):
        """Remove all registered hooks (call after training finishes)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for hook in self._grad_hooks:
            hook.remove()
        self._grad_hooks.clear()
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

    def _make_grad_hook(self, layer_idx: int):
        """Return a backward hook closure to capture gradient norms.

        Captures the L2 norm of the gradient w.r.t. the layer output,
        which indicates how much this layer is contributing to the loss
        gradient — i.e., how much it's "learning" this step.
        """
        def hook(module, grad_input, grad_output):
            if not self._should_capture:
                return
            # Deduplicate for gradient checkpointing
            if layer_idx in self._captured_grad_layers:
                return

            # grad_output is a tuple; first element is the gradient w.r.t output
            grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
            if grad is None or not isinstance(grad, torch.Tensor):
                return

            with torch.no_grad():
                # Compute L2 norm of the gradient
                grad_norm = grad.detach().float().norm().item()
                self._grad_buffer[layer_idx] = grad_norm

            self._captured_grad_layers.add(layer_idx)

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
        self._captured_grad_layers.clear()

    def flush(self):
        """Write the buffered stats to JSONL and reset capture state.

        Called by :class:`ActivationCaptureCallback` after each step completes.
        """
        if not self._buffer and not self._grad_buffer:
            self._should_capture = False
            return

        record = {
            "step":   self._step,
            "loss":   self._loss,
            "layers": {str(k): v for k, v in self._buffer.items()},
        }

        # Add gradient norms if captured
        if self._grad_buffer:
            record["grad_norms"] = {str(k): v for k, v in self._grad_buffer.items()}

        # Add LoRA norms if configured
        if self.config.capture_lora_norms and self._lora_modules:
            lora_norms = self._compute_lora_norms()
            if lora_norms:
                # Flatten to { "layer_idx.target": norm, ... }
                record["lora_norms"] = {
                    f"{layer_idx}.{target}": norm
                    for layer_idx, targets in lora_norms.items()
                    for target, norm in targets.items()
                }

        with open(self._log_path, "a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

        self._buffer.clear()
        self._grad_buffer.clear()
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
