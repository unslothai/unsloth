# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in UNSLOTH_LOG_MFU=1: LoRA-aware TFLOPs and MFU logging; train_hfu counts checkpoint recompute."""

__all__ = [
    "count_linear_flops_per_token",
    "count_attention_flops_per_pair",
    "get_peak_tflops",
    "UnslothMFUCallback",
    "add_mfu_callback",
]

import os
import re
import time

from collections.abc import Mapping

import torch
from transformers import TrainerCallback

# Dense (no sparsity) 16 bit tensor core TFLOPs per GPU, from the vendor datasheets.
# Order matters: the first substring found in the device name wins.
_PEAK_TFLOPS_16BIT = (
    ("H100 PCIE", 756.0),
    ("H100 NVL", 835.0),
    ("H100", 989.0),
    ("H200 NVL", 835.5),
    ("H200", 989.0),
    ("GH200", 989.0),
    # Dense = half NVIDIA's sparse figure (nvidia.com/en-us/data-center/hgx, gb200-nvl72).
    # GB* first: the B200 / B300 keys would match inside their names.
    ("GB300", 2500.0),
    ("GB200", 2500.0),
    ("B300", 2250.0),
    ("B200", 2250.0),
    # sm120. NVIDIA RTX PRO Blackwell GPU Architecture whitepaper v1.1, Table 4 (dense FP16 / BF16
    # tensor with FP32 accumulate): Workstation 503.8, Max-Q 438.9. Server Edition: nvidia.com
    # rtx-pro-6000-blackwell-server-edition lists 1 PFLOP FP16 / BF16 with sparsity -> 500 dense.
    # GeForce RTX 5090: NVIDIA RTX Blackwell GPU Architecture whitepaper v1.1, Table 3, 209.5 dense
    # with FP32 accumulate (GeForce runs FP32 accumulate at half the FP16 accumulate rate).
    # Specific editions first; every key needs "PRO", so "RTX 6000 Ada Generation" never matches.
    ("RTX PRO 6000 BLACKWELL MAX-Q", 438.9),
    ("RTX PRO 6000 BLACKWELL SERVER", 500.0),
    ("RTX PRO 6000 BLACKWELL", 503.8),
    # Unknown (None): the China variant and the laptop part (a smaller GB203) have no figure there.
    ("RTX 5090 D", None),
    ("RTX 5090 LAPTOP", None),
    ("RTX 5090", 209.5),
    ("A100", 312.0),
    ("L40S", 362.05),
    ("L40", 181.05),
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


def _model_config(model):
    """The HF config; a SentenceTransformer keeps it on its first module's auto_model."""
    config = getattr(model, "config", None)
    if config is None and isinstance(model, torch.nn.Sequential) and len(model) > 0:
        config = getattr(getattr(model[0], "auto_model", None), "config", None)
    return config


def _expert_fractions(model):
    """id(linear) -> share of tokens routed to it, for experts kept as a ModuleList of linears."""
    config = _text_config(_model_config(model))
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


def _inactive_leaves(model):
    """ids of modules a PEFT modules_to_save wrapper holds but does not run this forward."""
    skipped = set()
    for module in model.modules():
        copies = getattr(module, "modules_to_save", None)
        original = getattr(module, "original_module", None)
        if not isinstance(copies, torch.nn.ModuleDict) or not isinstance(original, torch.nn.Module):
            continue
        active = getattr(module, "active_adapters", None) or []
        if isinstance(active, str):
            active = [active]
        # Mirrors ModulesToSaveWrapper.forward: any missing or disabled adapter runs the original.
        if (
            getattr(module, "disable_adapters", False)
            or not active
            or any(a not in copies for a in active)
        ):
            used = original
        else:
            used = copies[active[0]]
        held = {id(m) for m in original.modules()} | {id(m) for m in copies.modules()}
        skipped |= held - {id(m) for m in used.modules()}
    for module in model.modules():
        branches = [getattr(module, k, None) for k in ("lora_A", "lora_B")]
        if not all(isinstance(b, torch.nn.ModuleDict) for b in branches):
            continue
        active = getattr(module, "active_adapters", None) or []
        if isinstance(active, str):
            active = [active]
        # Mirrors LoraLayer.forward: disabled or merged adapters run only the base layer.
        if getattr(module, "disable_adapters", False) or getattr(module, "merged", False):
            active = []
        for branch in branches:
            for name, adapter in branch.items():
                if name not in active:
                    skipped |= {id(m) for m in adapter.modules()}
    return skipped


def _non_text_leaves(model):
    """ids of modules outside the text decoder and head of a multimodal model."""
    config = _model_config(model)
    if config is None or _text_config(config) is config:
        return set()
    try:
        decoder = model.get_decoder()
    except Exception:
        decoder = None
    get_base_model = getattr(model, "get_base_model", None)
    root = get_base_model() if callable(get_base_model) else model
    if not isinstance(decoder, torch.nn.Module) or decoder is root or decoder is model:
        return set()
    keep = {id(m) for m in decoder.modules()}
    head = _output_head(model)
    if head is not None:
        keep |= {id(m) for m in head.modules()}
    return {id(m) for m in model.modules() if id(m) not in keep}


def _post_pooling_leaves(model):
    """ids of a SentenceTransformer's modules after the transformer (Pooling, Dense, Normalize):
    they run once per sequence, not per token, so they are left out (negligible per sequence)."""
    if not isinstance(model, torch.nn.Sequential) or len(model) < 2:
        return set()
    if getattr(model[0], "auto_model", None) is None:
        return set()
    return {id(m) for module in list(model)[1:] for m in module.modules()}


def _linear_flops(model, gradient_checkpointing = False):
    head = _output_head(model)
    head_leaves = set()
    if head is not None:
        head_leaves = {id(m) for m in head.modules() if _is_linear_leaf(m)}
    skipped = _inactive_leaves(model) | _non_text_leaves(model) | _post_pooling_leaves(model)
    experts = _expert_fractions(model)
    forward = input_grad = weight_grad = recompute = head_total = 0
    for module in model.modules():
        n = _linear_size(module)
        if n is None or id(module) in skipped:
            continue
        if id(module) in experts:
            n = n * experts[id(module)]
        forward += 2 * n
        input_grad += 2 * n
        if module.weight.requires_grad:
            weight_grad += 2 * n
            if id(module) in head_leaves:
                head_total += 2 * n
        if id(module) in head_leaves:
            head_total += 4 * n
        elif gradient_checkpointing:
            recompute += 2 * n
    breakdown = {
        "forward": forward,
        "input_grad": input_grad,
        "weight_grad": weight_grad,
        "recompute": recompute,
    }
    return forward + input_grad + weight_grad + recompute, breakdown, head_total


def count_linear_flops_per_token(model, gradient_checkpointing = False):
    total, breakdown, _ = _linear_flops(model, gradient_checkpointing)
    return total, breakdown


def _all_attention_bidirectional(model):
    flags = [
        m.is_causal for m in model.modules() if isinstance(getattr(m, "is_causal", None), bool)
    ]
    if not flags:
        # transformers 4.x encoders (BERT, XLM-R) expose only is_decoder on their attention.
        flags = [
            m.is_decoder
            for m in model.modules()
            if isinstance(getattr(m, "is_decoder", None), bool)
        ]
    return bool(flags) and not any(flags)


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
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, (list, tuple)) and len(layer_types) == layers:
        # Hybrids (Qwen3-Next, LFM2, ...) run linear attention, convolutions or recurrences elsewhere.
        layers = sum(
            1 for t in layer_types if isinstance(t, str) and "attention" in t and "linear" not in t
        )
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // heads)
    # MLA (DeepSeek V2 / V3) scores with nope + rope dims and mixes values with v_head_dim.
    qk_dim, v_dim = head_dim, head_dim
    nope, rope = (
        getattr(config, "qk_nope_head_dim", None),
        getattr(config, "qk_rope_head_dim", None),
    )
    if isinstance(nope, int) and isinstance(rope, int):
        qk_dim = nope + rope
    if isinstance(getattr(config, "v_head_dim", None), int):
        v_dim = config.v_head_dim
    forward = 2 * layers * heads * (qk_dim + v_dim)
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
        # A trailing digit is a different model (Quadro T400 is not a T4).
        if re.search(re.escape(key) + r"(?!\d)", name):
            if dtype == torch.bfloat16 and key in _NO_BF16:
                return None
            return value
    return None


def _runs_unsloth_head_window(model):
    get_base_model = getattr(model, "get_base_model", None)
    base = get_base_model() if callable(get_base_model) else model
    forward = getattr(type(base), "forward", None)
    return getattr(forward, "__module__", "").startswith("unsloth.")


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
        # Padding free packing: each document restarts its positions at 0. Fixed shape [N] with
        # zero padding (no nonzero / data dependent branch): no host sync, and fullgraph compiles.
        position_ids = position_ids[0]
        starts = position_ids == 0
        doc = starts.cumsum(0) - 1
        lengths = torch.zeros_like(doc).scatter_add_(0, doc.clamp(min = 0), (doc >= 0).to(doc.dtype))
        whole_row = torch.zeros_like(doc)
        whole_row[0] = position_ids.numel()
        return torch.where(starts.any(), lengths, whole_row)
    if torch.is_tensor(input_ids) and input_ids.dim() >= 2:
        return torch.full((input_ids.shape[0],), input_ids.shape[1], device = input_ids.device)
    return None


def _unsupported_setup(
    model,
    args,
    trainer_name = "",
):
    """True where the per-token count does not describe the step, so nothing is reported:
    diffusion objectives (extra no-grad passes, encoder + canvas, recomputed heads) and
    single-process DataParallel (the hook runs once per replica, concurrently)."""
    model_type = str(getattr(_model_config(model), "model_type", "") or "")
    if "diffusion" in model_type.lower() or "Diffusion" in trainer_name:
        return True
    # Trainer sets n_gpu > 1 only for DataParallel; distributed processes see 1.
    return (getattr(args, "n_gpu", 1) or 1) > 1


class UnslothMFUCallback(TrainerCallback):
    """Adds `train_tflops_per_gpu` / `train_mfu` (model FLOPs) and `train_hw_tflops_per_gpu` /
    `train_hfu` (recompute included) to training logs. Prefer `add_mfu_callback`."""

    def __init__(
        self,
        peak_tflops = None,
        gradient_checkpointing = None,
    ):
        self.peak_tflops = peak_tflops
        self.gradient_checkpointing = gradient_checkpointing
        self._handle = None
        self._linear_per_token = 0
        self._head_per_token = 0
        self._attention_per_pair = 0
        self._hw_linear_per_token = 0
        self._hw_attention_per_pair = 0
        self._bidirectional = False
        self._num_devices = 1
        self._devices = []
        self._unsloth_head_window = False
        self._enabled = True
        self._trainer_name = ""
        self._reset()

    def _reset(self):
        self._tokens = 0
        self._head_tokens = 0
        self._pairs = 0
        self._window_start = None
        self._excluded = 0.0
        self._last_step_end = None
        self._step_end_events = None
        self._paused = False

    def _count_batch(self, module, args, kwargs):
        # Generation, reference and evaluation forwards run without grad and are not training FLOPs.
        if not module.training or not torch.is_grad_enabled():
            return
        if args and isinstance(args[0], Mapping):
            # SentenceTransformer losses call model(features) with one feature dict.
            kwargs = {**args[0], **kwargs}
            args = args[1:]
        lengths = _sequence_lengths(kwargs, args)
        if lengths is None:
            return
        lengths = lengths.to(torch.int64)
        # Kept on device: one host sync per log window, not per micro batch.
        self._tokens = self._tokens + lengths.sum()
        keep = kwargs.get("logits_to_keep")
        if self._unsloth_head_window:
            # Unsloth's Llama / Mistral forwards take the larger int spelling and run the full
            # head when either spelling is a tensor (models/llama.py CausalLM_fast_forward).
            legacy = kwargs.get("num_logits_to_keep")
            if torch.is_tensor(keep) or torch.is_tensor(legacy):
                keep = 0
            elif isinstance(legacy, int) and not isinstance(legacy, bool):
                keep = max(keep if isinstance(keep, int) else 0, legacy)
        if torch.is_tensor(keep) and keep.dim() == 1 and keep.numel() > 0:
            keep = keep.numel()
        if isinstance(keep, int) and not isinstance(keep, bool) and keep > 0:
            self._head_tokens = self._head_tokens + lengths.clamp(max = keep).sum()
        else:
            self._head_tokens = self._head_tokens + lengths.sum()
        if self._bidirectional:
            self._pairs = self._pairs + (lengths * lengths).sum()
        else:
            self._pairs = self._pairs + (lengths * (lengths + 1) // 2).sum()

    def on_train_begin(
        self,
        args,
        state,
        control,
        model = None,
        **kwargs,
    ):
        if model is None:
            return
        config = _model_config(model)
        # One length per sequence cannot describe separate source and target lengths.
        self._enabled = not getattr(config, "is_encoder_decoder", False) and not _unsupported_setup(
            model, args, self._trainer_name
        )
        if not self._enabled:
            return
        checkpointing = self.gradient_checkpointing
        if checkpointing is None:
            checkpointing = _gradient_checkpointing_on(model, args)
        total, breakdown, self._head_per_token = _linear_flops(model, checkpointing)
        self._hw_linear_per_token = total - self._head_per_token
        self._linear_per_token = self._hw_linear_per_token - breakdown["recompute"]
        self._bidirectional = _all_attention_bidirectional(model)
        self._unsloth_head_window = _runs_unsloth_head_window(model)
        self._attention_per_pair = (
            count_attention_flops_per_pair(config, False) if config is not None else 0
        )
        self._hw_attention_per_pair = (
            count_attention_flops_per_pair(config, checkpointing) if config is not None else 0
        )
        devices = sorted(
            {p.device.index for p in model.parameters() if p.device.type == "cuda"},
            key = lambda index: -1 if index is None else index,
        )
        self._num_devices = max(len(devices), 1)
        self._devices = [index for index in devices if index is not None]
        if self.peak_tflops is None:
            dtype = _training_dtype(model, args)
            # A model off the GPU (use_cpu) has no GPU peak; only UNSLOTH_PEAK_TFLOPS describes it.
            peaks = [get_peak_tflops(index, dtype) for index in devices] or [
                get_peak_tflops(None, dtype)
                if os.environ.get("UNSLOTH_PEAK_TFLOPS", "").strip()
                else None
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
            paused = now - self._last_step_end
            if self._step_end_events:
                # The step's kernels may still be queued at on_step_end: exclude only the time
                # after every GPU finished them.
                gaps = []
                for index, start in self._step_end_events:
                    end = torch.cuda.Event(enable_timing = True)
                    end.record(torch.cuda.current_stream(index))
                    end.synchronize()
                    gaps.append(start.elapsed_time(end) / 1000)
                paused = min([paused] + gaps)
            self._excluded += paused
        self._paused = False

    def on_step_end(self, args, state, control, **kwargs):
        self._last_step_end = time.perf_counter()
        self._step_end_events = None
        if self._devices and torch.cuda.is_available():
            # Recording is asynchronous; only a later evaluation or save waits on them.
            self._step_end_events = []
            for index in self._devices:
                event = torch.cuda.Event(enable_timing = True)
                event.record(torch.cuda.current_stream(index))
                self._step_end_events.append((index, event))

    def on_evaluate(self, args, state, control, **kwargs):
        self._paused = True

    def on_save(self, args, state, control, **kwargs):
        self._paused = True

    def on_log(
        self,
        args,
        state,
        control,
        logs = None,
        **kwargs,
    ):
        if not self._enabled or logs is None or "loss" not in logs or self._window_start is None:
            return
        tokens = int(self._tokens)
        head_tokens = int(self._head_tokens)
        pairs = int(self._pairs)
        if torch.cuda.is_available():
            # A model split with device_map may still be running kernels on the other GPUs.
            for index in self._devices or [None]:
                torch.cuda.synchronize(index)
        elapsed = time.perf_counter() - self._window_start - self._excluded
        self._reset()
        if tokens <= 0 or elapsed <= 0:
            return
        head = head_tokens * self._head_per_token
        flops = tokens * self._linear_per_token + head + pairs * self._attention_per_pair
        hw_flops = tokens * self._hw_linear_per_token + head + pairs * self._hw_attention_per_pair
        tflops = flops / self._num_devices / elapsed / 1e12
        hw_tflops = hw_flops / self._num_devices / elapsed / 1e12
        extra = {"train_tflops_per_gpu": tflops, "train_hw_tflops_per_gpu": hw_tflops}
        if self.peak_tflops:
            extra["train_mfu"] = tflops / self.peak_tflops
            extra["train_hfu"] = hw_tflops / self.peak_tflops
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
    callbacks = trainer.callback_handler.callbacks
    for index, callback in enumerate(callbacks):
        if isinstance(callback, UnslothMFUCallback):
            callbacks.insert(0, callbacks.pop(index))
            return callback
    callback = UnslothMFUCallback(**kwargs)
    callback._trainer_name = type(trainer).__name__
    trainer.callback_handler.callbacks.insert(0, callback)
    return callback
