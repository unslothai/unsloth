# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in achieved TFLOPs and MFU logging with LoRA-aware FLOP counts.

Turn on with `UNSLOTH_LOG_MFU=1` before building the trainer, or call `add_mfu_callback(trainer)`.
Each training log then gains `train_tflops_per_gpu` and, when the GPU peak is known,
`train_mfu`. Set `UNSLOTH_PEAK_TFLOPS` to the per GPU dense peak to override the table
(for example a measured MAMF number).

Model FLOPs per non padding token, summed over every linear layer with n = in * out weights:
    forward                  2n   every linear, frozen or trainable
    backward input grads     2n   every linear (gradients flow through frozen weights to LoRA)
    backward weight grads    2n   only linears whose weight requires grad
    recompute                2n   every linear except the output head, when checkpointing
Attention adds 4 * layers * heads * head_dim FLOPs per causal (query, key) pair in the forward
(QK^T and AV), L * (L + 1) / 2 pairs for a sequence of length L, times 3 for forward plus
backward and times 4 with checkpointing. Experts held as `nn.Linear` in an `experts` ModuleList
count top_k / num_experts of their weights; experts stored as 3D tensors (transformers 5),
sliding windows, convolutions and embedding lookups are not counted, so those models report a
lower bound. Encoder-decoder models are not reported. A model split over several GPUs with
`device_map` reports the average per participating GPU.
"""

__all__ = [
    "count_linear_flops_per_token",
    "count_attention_flops_per_pair",
    "get_peak_tflops",
    "UnslothMFUCallback",
    "add_mfu_callback",
]

import os
import time

import torch
from transformers import TrainerCallback

# Dense (no sparsity) 16 bit tensor core TFLOPs per GPU, from the vendor datasheets.
# Order matters: the first substring found in the device name wins.
_PEAK_TFLOPS_16BIT = (
    ("H100 PCIE", 756.0),
    ("H100 NVL", 835.0),
    ("H100", 989.0),
    ("H200", 989.0),
    ("GH200", 989.0),
    ("B200", 2250.0),
    ("A100", 312.0),
    ("L4", 121.0),
    ("T4", 65.0),
    ("MI300A", 980.6),
    ("MI300X", 1307.4),
    ("MI325X", 1307.4),
)
_NO_BF16 = ("T4",)


def _linear_size(module):
    # PEFT LoRA wrappers expose in/out_features too but own no weight; their base_layer counts.
    if "weight" not in getattr(module, "_parameters", {}):
        return None
    for fan_in, fan_out in (("in_features", "out_features"), ("nx", "nf")):
        fan_in = getattr(module, fan_in, None)
        fan_out = getattr(module, fan_out, None)
        if isinstance(fan_in, int) and isinstance(fan_out, int):
            return fan_in * fan_out
    return None


def _is_linear_leaf(module):
    return _linear_size(module) is not None


def _expert_fractions(model):
    """id(linear) -> share of tokens routed to it, for experts kept as a ModuleList of linears."""
    config = _text_config(getattr(model, "config", None))
    default_top_k = getattr(config, "num_experts_per_tok", None)
    fractions = {}
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if not isinstance(experts, torch.nn.ModuleList) or len(experts) == 0:
            continue
        top_k = getattr(module, "top_k", None) or default_top_k
        fraction = 0
        if isinstance(top_k, int) and 0 < top_k <= len(experts):
            fraction = top_k / len(experts)
        for leaf in experts.modules():
            if _is_linear_leaf(leaf):
                fractions[id(leaf)] = fraction
    return fractions


def _output_head(model):
    try:
        head = model.get_output_embeddings()
    except Exception:
        head = None
    if head is not None:
        return head
    for name, module in model.named_modules():
        if name.endswith("lm_head"):
            return module
    return None


def count_linear_flops_per_token(model, gradient_checkpointing = False):
    head = _output_head(model)
    head_leaves = set()
    if head is not None:
        head_leaves = {id(m) for m in head.modules() if _is_linear_leaf(m)}
    experts = _expert_fractions(model)
    forward = input_grad = weight_grad = recompute = 0
    for module in model.modules():
        n = _linear_size(module)
        if n is None:
            continue
        if id(module) in experts:
            n = n * experts[id(module)]
        forward += 2 * n
        input_grad += 2 * n
        if module.weight.requires_grad:
            weight_grad += 2 * n
        if gradient_checkpointing and id(module) not in head_leaves:
            recompute += 2 * n
    breakdown = {
        "forward": forward,
        "input_grad": input_grad,
        "weight_grad": weight_grad,
        "recompute": recompute,
    }
    return forward + input_grad + weight_grad + recompute, breakdown


def _text_config(config):
    if config is None:
        return None
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            return get_text_config()
        except Exception:
            pass
    return config


def count_attention_flops_per_pair(config, gradient_checkpointing = False):
    config = _text_config(config)
    layers = getattr(config, "num_hidden_layers", None)
    heads = getattr(config, "num_attention_heads", None)
    if not layers or not heads:
        return 0
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // heads)
    forward = 4 * layers * heads * head_dim
    return forward * (4 if gradient_checkpointing else 3)


def get_peak_tflops(device = None, dtype = torch.float16):
    """Per GPU dense peak for 16 bit training, `UNSLOTH_PEAK_TFLOPS` first; None if unknown."""
    override = os.environ.get("UNSLOTH_PEAK_TFLOPS", "").strip()
    if override:
        try:
            value = float(override)
            if value > 0:
                return value
        except ValueError:
            pass
        return None
    if dtype not in (torch.float16, torch.bfloat16):
        return None
    try:
        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(device).upper()
    except Exception:
        return None
    for key, value in _PEAK_TFLOPS_16BIT:
        if key in name:
            if dtype == torch.bfloat16 and key in _NO_BF16:
                return None
            return value
    return None


def _gradient_checkpointing_on(model, args):
    if getattr(args, "gradient_checkpointing", False):
        return True
    if getattr(model, "is_gradient_checkpointing", False) is True:
        return True
    return any(getattr(m, "gradient_checkpointing", False) is True for m in model.modules())


def _training_dtype(model, args):
    if getattr(args, "bf16", False):
        return torch.bfloat16
    if getattr(args, "fp16", False):
        return torch.float16
    return getattr(model, "dtype", torch.float32)


def _sequence_lengths(kwargs, args):
    attention_mask = kwargs.get("attention_mask")
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        input_ids = args[0] if torch.is_tensor(args[0]) else None
    if input_ids is None:
        input_ids = kwargs.get("inputs_embeds")
        if input_ids is not None:
            input_ids = input_ids[..., 0]
    for key in ("packed_seq_lengths", "seq_lens"):
        lengths = kwargs.get(key)
        if torch.is_tensor(lengths) and lengths.numel() > 0:
            return lengths.reshape(-1)
    if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
        return attention_mask.sum(dim = -1)
    position_ids = kwargs.get("position_ids")
    if torch.is_tensor(position_ids) and position_ids.dim() == 2 and position_ids.shape[0] == 1:
        # Padding free packing: each document restarts its positions at 0.
        position_ids = position_ids[0]
        starts = torch.nonzero(position_ids == 0).reshape(-1)
        if starts.numel() > 0:
            ends = torch.cat([starts[1:], starts.new_tensor([position_ids.numel()])])
            return ends - starts
    if torch.is_tensor(input_ids) and input_ids.dim() >= 2:
        return torch.full(
            (input_ids.shape[0],), input_ids.shape[1], device = input_ids.device
        )
    return None


class UnslothMFUCallback(TrainerCallback):
    """Adds `train_tflops_per_gpu` and `train_mfu` to training logs. Prefer `add_mfu_callback`."""

    def __init__(self, peak_tflops = None, gradient_checkpointing = None):
        self.peak_tflops = peak_tflops
        self.gradient_checkpointing = gradient_checkpointing
        self._handle = None
        self._linear_per_token = 0
        self._attention_per_pair = 0
        self._num_devices = 1
        self._enabled = True
        self._reset()

    def _reset(self):
        self._tokens = 0
        self._pairs = 0
        self._window_start = None
        self._excluded = 0.0
        self._last_step_end = None
        self._paused = False

    def _count_batch(self, module, args, kwargs):
        # Generation, reference and evaluation forwards run without grad and are not training FLOPs.
        if not module.training or not torch.is_grad_enabled():
            return
        lengths = _sequence_lengths(kwargs, args)
        if lengths is None:
            return
        lengths = lengths.to(torch.int64)
        # Kept on device: one host sync per log window, not per micro batch.
        self._tokens = self._tokens + lengths.sum()
        self._pairs = self._pairs + (lengths * (lengths + 1) // 2).sum()

    def on_train_begin(self, args, state, control, model = None, **kwargs):
        if model is None:
            return
        config = getattr(model, "config", None)
        # One length per sequence cannot describe separate source and target lengths.
        self._enabled = not getattr(config, "is_encoder_decoder", False)
        if not self._enabled:
            return
        checkpointing = self.gradient_checkpointing
        if checkpointing is None:
            checkpointing = _gradient_checkpointing_on(model, args)
        self._linear_per_token, _ = count_linear_flops_per_token(model, checkpointing)
        self._attention_per_pair = (
            count_attention_flops_per_pair(config, checkpointing) if config is not None else 0
        )
        devices = sorted(
            {p.device.index for p in model.parameters() if p.device.type == "cuda"},
            key = lambda index: -1 if index is None else index,
        )
        self._num_devices = max(len(devices), 1)
        if self.peak_tflops is None:
            dtype = _training_dtype(model, args)
            peaks = [get_peak_tflops(index, dtype) for index in devices] or [
                get_peak_tflops(None, dtype)
            ]
            if all(peaks):
                self.peak_tflops = sum(peaks) / len(peaks)
        if self._handle is None:
            self._handle = model.register_forward_pre_hook(self._count_batch, with_kwargs = True)
        self._reset()

    def on_step_begin(self, args, state, control, **kwargs):
        now = time.perf_counter()
        if self._window_start is None:
            self._window_start = now
        elif self._paused and self._last_step_end is not None:
            self._excluded += now - self._last_step_end
        self._paused = False

    def on_step_end(self, args, state, control, **kwargs):
        self._last_step_end = time.perf_counter()

    def on_evaluate(self, args, state, control, **kwargs):
        self._paused = True

    def on_save(self, args, state, control, **kwargs):
        self._paused = True

    def on_log(self, args, state, control, logs = None, **kwargs):
        if not self._enabled or logs is None or "loss" not in logs or self._window_start is None:
            return
        tokens = int(self._tokens)
        pairs = int(self._pairs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._window_start - self._excluded
        self._reset()
        if tokens <= 0 or elapsed <= 0:
            return
        flops = tokens * self._linear_per_token + pairs * self._attention_per_pair
        flops = flops / self._num_devices
        tflops = flops / elapsed / 1e12
        extra = {"train_tflops_per_gpu": tflops}
        if self.peak_tflops:
            extra["train_mfu"] = tflops / self.peak_tflops
        logs.update(extra)
        # Trainer.log appended a copy of logs to log_history before calling the callbacks.
        if state.log_history and state.log_history[-1].get("step") == state.global_step:
            state.log_history[-1].update(extra)

    def on_train_end(self, args, state, control, **kwargs):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def add_mfu_callback(trainer, **kwargs):
    """Register an UnslothMFUCallback first, so reporting callbacks see the fields it adds."""
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, UnslothMFUCallback):
            return callback
    callback = UnslothMFUCallback(**kwargs)
    trainer.callback_handler.callbacks.insert(0, callback)
    return callback
