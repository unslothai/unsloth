# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Block-diffusion families (LLaDA2.x, SDAR).

Training follows BD3-LM (https://arxiv.org/abs/2503.09573): each row is fed as [x_t ; x_0] with position ids
restarting at the clean copy. A noisy token sees its own noisy block and the clean blocks before it; clean tokens
see clean blocks causally. The loss reads the noisy half at the same position (no shift).
"""

import contextlib
import functools
import importlib.machinery
import inspect
import sys
import types

import torch
import torch.nn.functional as F

from .diffusion_profiles import (
    DiffusionProfile,
    register_diffusion_profile,
    response_mask,
    unwrap_diffusion_model,
)

__all__ = [
    "BlockDiffusionProfile",
    "block_diffusion_attention_mask",
    "LLADA2_PROFILE",
    "SDAR_PROFILE",
]


def block_diffusion_attention_mask(
    length,
    block_size,
    device = None,
):
    """Bool [2L, 2L], True = attend. Rows/cols 0..L-1 are x_t, L..2L-1 are x_0."""
    idx = torch.arange(2 * length, device = device)
    q, kv = idx[:, None], idx[None, :]
    q_clean, kv_clean = q >= length, kv >= length
    q_block = torch.where(q_clean, q - length, q) // block_size
    kv_block = torch.where(kv_clean, kv - length, kv) // block_size
    same_noisy_block = (q_block == kv_block) & (q_clean == kv_clean)
    noisy_to_past_clean = (q_block > kv_block) & kv_clean & ~q_clean
    clean_causal = (q_block >= kv_block) & kv_clean & q_clean
    return same_noisy_block | noisy_to_past_clean | clean_causal


_unwrap = unwrap_diffusion_model


def _is_distributed_wrapper(model):
    return type(model).__name__ in (
        "DistributedDataParallel",
        "FullyShardedDataParallel",
        "DeepSpeedEngine",
    )


class BlockDiffusionProfile(DiffusionProfile):
    # defaults keys: diffusion_block_size, noise_low, noise_high (mask probability range), time_weighting
    # ("none" | "inverse_t"), normalize ("masked" | "supervised"), eos_fill, key_padding, ensure_masked,
    # mask_token_id, mask_token_names, mask_format ("additive" | "bool").

    def prepare_model(self, model, tokenizer):
        mask_id = self._mask_token_from_tokenizer(tokenizer)
        if mask_id is not None:
            model._unsloth_diffusion_mask_token_id = mask_id
        eos = None
        if tokenizer is not None:
            tok = getattr(tokenizer, "tokenizer", tokenizer)
            eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
        if eos is not None:
            model._unsloth_diffusion_eos_token_id = eos
        return model

    def _mask_token_from_tokenizer(self, tokenizer):
        if tokenizer is None:
            return None
        tok = getattr(tokenizer, "tokenizer", tokenizer)
        if getattr(tok, "mask_token_id", None) is not None:
            return tok.mask_token_id
        unk = getattr(tok, "unk_token_id", None)
        for name in self.defaults.get("mask_token_names", ()):
            token_id = tok.convert_tokens_to_ids(name)
            if isinstance(token_id, int) and token_id != unk:
                return token_id
        return None

    def mask_token_id(self, model, args):
        base = _unwrap(model)
        for value in (
            getattr(args, "diffusion_mask_token_id", None),
            getattr(base, "_unsloth_diffusion_mask_token_id", None),
            getattr(getattr(base, "config", None), "mask_token_id", None),
            self.defaults.get("mask_token_id"),
        ):
            if value is not None:
                return int(value)
        raise ValueError(
            f"Unsloth: no [MASK] token id for {self.name}; set DiffusionConfig.diffusion_mask_token_id."
        )

    def eos_token_id(self, model):
        base = _unwrap(model)
        config = getattr(base, "config", None)
        for value in (
            getattr(base, "_unsloth_diffusion_eos_token_id", None),
            getattr(config, "eos_token_id", None),
            getattr(config, "pad_token_id", None),
        ):
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if value is not None:
                return int(value)
        raise ValueError(f"Unsloth: no EOS token id for {self.name}.")

    def block_size(self, model, args):
        value = getattr(args, "diffusion_block_size", None)
        if value is None:
            value = getattr(getattr(_unwrap(model), "config", None), "block_size", None)
        if value is None:
            value = self.defaults["diffusion_block_size"]
        return int(value)

    def build_batch(self, model, inputs, args):
        """-> clean ids [B, L], maskable [B, L], valid [B, L] (attended positions)."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        valid = (
            torch.ones_like(input_ids, dtype = torch.bool)
            if attention_mask is None
            else attention_mask.bool()
        )
        maskable = response_mask(inputs)
        if not self.defaults.get("eos_fill", False):
            return input_ids, maskable, valid
        if (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Unsloth: block-diffusion training expects right padding.")
        # The response ends with the turn terminator, not EOS, so the model learns to stop only from an EOS tail
        # (the reference pads every row to max length with supervised EOS). Pad to the next block boundary past
        # the longest row so every supervised row gets at least one EOS.
        block = self.block_size(model, args)
        batch, length = input_ids.shape
        new_length = -(-(length + 1) // block) * block
        pad = new_length - length
        eos = self.eos_token_id(model)
        input_ids = F.pad(input_ids, (0, pad), value = eos)
        maskable = F.pad(maskable, (0, pad), value = False)
        valid = F.pad(valid, (0, pad), value = False)
        real_length = valid.sum(dim = 1, keepdim = True)
        positions = torch.arange(new_length, device = input_ids.device)
        # A row clipped at max_length has no real end, so it gets no EOS tail (as for DiffusionGemma).
        max_length = getattr(args, "max_length", None)
        truncated = torch.zeros_like(real_length, dtype = torch.bool)
        if max_length is not None:
            last = torch.where(maskable, positions[None, :], -1).amax(dim = 1, keepdim = True)
            truncated = (real_length >= max_length) & (last == real_length - 1)
        fill = (positions[None, :] >= real_length) & maskable.any(dim = 1, keepdim = True) & ~truncated
        input_ids = torch.where(fill, eos, input_ids)
        return input_ids, maskable | fill, valid | fill

    def noise_range(self, args):
        eps = getattr(args, "diffusion_eps", None)
        if eps is not None:
            return float(eps), 1.0 - float(eps)
        return float(self.defaults["noise_low"]), float(self.defaults["noise_high"])

    def sample_noise(self, clean, maskable, mask_token_id, args):
        """-> noisy ids, masked bool [B, L], p [B] mask probability per row."""
        batch, length = clean.shape
        low, high = self.noise_range(args)
        p = low + (high - low) * torch.rand(batch, device = clean.device)
        masked = (torch.rand(batch, length, device = clean.device) < p[:, None]) & maskable
        if self.defaults.get("ensure_masked", False):
            empty = maskable.any(dim = 1) & ~masked.any(dim = 1)
            if empty.any():
                scores = torch.rand(batch, length, device = clean.device).masked_fill(~maskable, -1.0)
                pick = F.one_hot(scores.argmax(dim = 1), length).bool()
                masked = masked | (pick & empty[:, None])
        return torch.where(masked, mask_token_id, clean), masked, p

    def attention_mask(self, length, block_size, valid, dtype, device):
        mask = block_diffusion_attention_mask(length, block_size, device)[None, None]
        batch = valid.shape[0]
        if self.defaults.get("key_padding", False) and not bool(valid.all()):
            keys = torch.cat([valid, valid], dim = 1)[:, None, None, :]
            # Padding is its own segment in the reference; a padding query keeps its diagonal so no row is empty.
            mask = (mask & keys) | torch.eye(2 * length, dtype = torch.bool, device = device)[
                None, None
            ]
        mask = mask.expand(batch, 1, 2 * length, 2 * length)
        if self.defaults.get("mask_format", "additive") == "bool":
            return mask
        additive = torch.zeros(mask.shape, dtype = dtype, device = device)
        return additive.masked_fill(~mask, float("-inf"))

    def noisy_hidden_and_head(self, model, noisy, clean, valid, block_size):
        """Hidden states of the noisy half [B, L, H] and the output head."""
        batch, length = clean.shape
        base = _unwrap(model)
        head = base.get_output_embeddings()
        concat = torch.cat([noisy, clean], dim = 1)
        positions = torch.arange(length, device = clean.device)
        position_ids = torch.cat([positions, positions])[None].expand(batch, -1)
        dtype = head.weight.dtype if head.weight.dtype.is_floating_point else torch.float32
        mask = self.attention_mask(length, block_size, valid, dtype, clean.device)
        if _is_distributed_wrapper(model):
            # DDP / FSDP / DeepSpeed must see their own forward; this path materialises logits for both halves.
            out = model(
                input_ids = concat, attention_mask = mask, position_ids = position_ids, use_cache = False
            )
            return out.logits[:, :length], None
        out = base.get_decoder()(
            input_ids = concat,
            attention_mask = mask,
            position_ids = position_ids,
            use_cache = False,
            return_dict = True,
        )
        return out[0][:, :length], head

    def loss_from_noise(self, model, clean, noisy, masked, maskable, p, valid, args):
        block_size = self.block_size(model, args)
        hidden, head = self.noisy_hidden_and_head(model, noisy, clean, valid, block_size)
        if head is None:
            logits = hidden[masked].float()
        else:
            # Project only the masked positions: the vocab-sized logits are the memory peak otherwise.
            logits = head(hidden[masked]).float()
        nll = F.cross_entropy(logits, clean[masked], reduction = "none")
        weighting = self.option(args, "diffusion_time_weighting", "none")
        if weighting not in ("inverse_t", "none"):
            raise ValueError(
                f"Unsloth: {self.name} does not support diffusion_time_weighting={weighting!r}."
            )
        if weighting == "inverse_t":
            nll = nll / p[:, None].expand_as(masked)[masked].float()
        if self.defaults.get("normalize", "masked") == "supervised":
            denominator = maskable.sum()
        else:
            denominator = masked.sum()
        loss = nll.sum() / denominator.clamp_min(1)
        if not masked.any():
            loss = loss + hidden.sum() * 0.0
        return loss

    def compute_loss(
        self,
        model,
        inputs,
        args,
        num_items_in_batch = None,
    ):
        clean, maskable, valid = self.build_batch(model, inputs, args)
        mask_token_id = self.mask_token_id(model, args)
        noisy, masked, p = self.sample_noise(clean, maskable, mask_token_id, args)
        loss = self.loss_from_noise(model, clean, noisy, masked, maskable, p, valid, args)
        metrics = {
            "mask_ratio": (masked.sum() / maskable.sum().clamp_min(1)).item(),
            "masked_tokens": masked.sum().item(),
        }
        return loss, None, metrics


class _LLaDA2Profile(BlockDiffusionProfile):
    pass


# Reference: inclusionAI/dFactory tasks/train_llada2_bd.py and configs/sft/llada2_mini_bd_sft.yaml.
LLADA2_PROFILE = register_diffusion_profile(
    _LLaDA2Profile(
        name = "llada2",
        model_types = ("llada2_moe",),
        architectures = ("LLaDA2MoeModelLM",),
        noise = "mask",
        requires_remote_code = True,
        # Fused qkv + dense attention, the dense first-layer MLP and the always-on shared expert; the routed
        # experts and the router stay frozen.
        lora_target_modules = (
            r".*\.layers\.\d+\.(attention\.(query_key_value|dense)"
            r"|mlp\.(shared_experts\.)?(gate_proj|up_proj|down_proj))"
        ),
        defaults = dict(
            diffusion_block_size = 32,
            noise_low = 0.3,
            noise_high = 0.8,
            diffusion_time_weighting = "none",
            normalize = "masked",
            eos_fill = True,
            key_padding = False,
            ensure_masked = False,
            mask_token_id = 156895,
            mask_token_names = ("<|mask|>",),
            mask_format = "additive",
            attn_implementation = "sdpa",
        ),
    )
)


def _rms_norm_fn(
    x,
    weight,
    bias = None,
    eps = 1e-6,
    **kwargs,
):
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + eps)
    out = weight * x.to(dtype) if weight is not None else x.to(dtype)
    return out + bias if bias is not None else out


def _flash_attn_func(
    q,
    k,
    v,
    dropout_p = 0.0,
    softmax_scale = None,
    causal = False,
    **kwargs,
):
    # [B, S, H, D] layout like flash-attn; SDPA handles GQA via enable_gqa.
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal = causal,
        scale = softmax_scale,
        enable_gqa = k.shape[2] != q.shape[2],
    )
    return out.transpose(1, 2)


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    module.__path__ = []
    module.__dict__.update(attributes)
    return module


@contextlib.contextmanager
def _sdar_remote_code_imports():
    """SDAR's modeling file imports flash-attn's RMSNorm and 4.x `LossKwargs` at module level. Provide
    torch stand-ins only while it imports when flash-attn is absent; the module keeps its references."""
    import importlib.util
    import transformers.utils as transformers_utils

    if not hasattr(transformers_utils, "LossKwargs"):
        from typing import Optional, TypedDict
        class LossKwargs(TypedDict, total = False):
            num_items_in_batch: Optional[torch.Tensor]

        transformers_utils.LossKwargs = LossKwargs
    try:
        has_flash = importlib.util.find_spec("flash_attn") is not None
    except (ImportError, ValueError):
        has_flash = False
    if has_flash:
        yield
        return
    names = (
        "flash_attn",
        "flash_attn.ops",
        "flash_attn.ops.triton",
        "flash_attn.ops.triton.layer_norm",
    )
    stubs = {
        "flash_attn": _module("flash_attn", flash_attn_func = _flash_attn_func),
        "flash_attn.ops": _module("flash_attn.ops"),
        "flash_attn.ops.triton": _module("flash_attn.ops.triton"),
        "flash_attn.ops.triton.layer_norm": _module(
            "flash_attn.ops.triton.layer_norm", rms_norm_fn = _rms_norm_fn
        ),
    }
    from transformers import dynamic_module_utils

    original_check_imports = dynamic_module_utils.check_imports

    def check_imports(filename):
        try:
            return original_check_imports(filename)
        except ImportError as error:
            if "flash_attn" not in str(error):
                raise
            return dynamic_module_utils.get_relative_imports(filename)

    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.update(stubs)
    dynamic_module_utils.check_imports = check_imports
    try:
        yield
    finally:
        dynamic_module_utils.check_imports = original_check_imports
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _accept_cache_objects(model):
    """SDAR's attention reads `past_key_value[layer]` (4.x tuple API); transformers 5 caches have no
    __getitem__. Hand it a read view while `update` / `len` still reach the real cache."""
    from .remote_code_shims import _LegacyCacheView, _needs_legacy_view

    wrapped = 0
    for module in model.modules():
        cls = type(module)
        forward = cls.__dict__.get("forward")
        if forward is None or getattr(forward, "_unsloth_cache_view", False):
            continue
        try:
            source = inspect.getsource(forward)
        except (OSError, TypeError):
            continue
        if (
            "past_key_value[" not in source
            or "past_key_value" not in inspect.signature(forward).parameters
        ):
            continue

        def make(original):
            @functools.wraps(original)
            def forward(self, *args, **kwargs):
                cache = kwargs.get("past_key_value")
                if _needs_legacy_view(cache):
                    kwargs["past_key_value"] = _LegacyCacheView(cache)
                return original(self, *args, **kwargs)

            forward._unsloth_cache_view = True
            return forward

        cls.forward = make(forward)
        wrapped += 1
    return wrapped


class _SDARProfile(BlockDiffusionProfile):
    def prepare_model(self, model, tokenizer):
        _accept_cache_objects(model)
        # The released config fuses the head into its training loss, so a train-mode forward returns no logits.
        config = getattr(model, "config", None)
        if getattr(config, "fuse_cross_entropy", False):
            config.fuse_cross_entropy = False
        return super().prepare_model(model, tokenizer)

    def model_class(self, config, trust_remote_code, **hub_kwargs):
        from transformers import AutoModelForCausalLM
        class _Loader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                with _sdar_remote_code_imports():
                    return AutoModelForCausalLM.from_pretrained(*args, **kwargs)

        return _Loader


# Reference: JetAstra/SDAR training/model/SDAR-8B-Chat/modeling_sdar.py (forward_add_noise_packed,
# block_attn_mask, FusedLinearDiffusionCrossEntropyLoss); block length 4 from the model card.
SDAR_PROFILE = register_diffusion_profile(
    _SDARProfile(
        name = "sdar",
        model_types = ("sdar",),
        architectures = ("SDARForCausalLM",),
        noise = "mask",
        requires_remote_code = True,
        defaults = dict(
            diffusion_block_size = 4,
            noise_low = 1e-3,
            noise_high = 1.0,
            diffusion_time_weighting = "inverse_t",
            normalize = "supervised",
            eos_fill = False,
            key_padding = True,
            ensure_masked = True,
            mask_token_id = 151669,
            mask_token_names = ("<MASK>", "<|MASK|>"),
            mask_format = "bool",
            attn_implementation = "sdpa",
        ),
    )
)
