# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""DiffusionGemma block-diffusion SFT objective, matching the reference fine-tune of the released checkpoint
(the recipe in TRL's examples/sft_diffusion_gemma/sft_diffusion_gemma.py)."""

import contextlib

import torch
import torch.nn.functional as F

from .diffusion_profiles import DiffusionProfile, register_diffusion_profile, unwrap_diffusion_model

__all__ = ["DiffusionGemmaProfile"]

try:
    from trl.trainer.utils import maybe_gather_lm_head_ctx as _gather_lm_head
except ImportError:  # trl < 1.x: no ZeRO-3 aware helper, plain access is fine off ZeRO-3

    def _gather_lm_head(*params):
        return contextlib.nullcontext()


def _find_attr(model, name):
    seen = set()
    stack = [model]
    while stack:
        m = stack.pop()
        if m is None or id(m) in seen:
            continue
        seen.add(id(m))
        value = m.__dict__.get(name) if hasattr(m, "__dict__") else None
        if value is not None:
            return value
        stack += [
            getattr(m, "base_model", None),
            getattr(m, "model", None),
            getattr(m, "module", None),
        ]
    return None


def _eos_token_id(model):
    eos = _find_attr(model, "_unsloth_diffusion_eos_token_id")
    if eos is None:
        config = model.config
        eos = getattr(getattr(config, "text_config", config), "eos_token_id", None)
        if eos is None:
            eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    if eos is None:
        raise ValueError(
            "Unsloth: DiffusionGemma training needs an eos_token_id to fill the canvas."
        )
    return int(eos)


class DiffusionGemmaProfile(DiffusionProfile):
    def prepare_model(self, model, tokenizer):
        tok = getattr(tokenizer, "tokenizer", tokenizer)
        eos = getattr(tok, "eos_token_id", None)
        if eos is not None:
            model._unsloth_diffusion_eos_token_id = int(eos)
        return model

    def compute_loss(
        self,
        model,
        inputs,
        args,
        num_items_in_batch = None,
    ):
        eps = self.option(args, "diffusion_eps")
        sc_p = self.option(args, "diffusion_self_conditioning_p")
        ar_weight = self.option(args, "diffusion_encoder_ar_weight")
        prediction_type = self.option(args, "diffusion_prediction_type")
        if prediction_type not in ("mean", "mean_loo"):
            raise ValueError(
                f"Unsloth: diffusion_prediction_type must be 'mean' or 'mean_loo', got {prediction_type!r}."
            )

        base = unwrap_diffusion_model(model)
        config = base.config
        text_config = config.text_config
        block_size = config.canvas_length
        vocab_size = text_config.vocab_size
        softcap = text_config.final_logit_softcapping
        eos_token_id = _eos_token_id(base)
        max_length = getattr(args, "max_length", None)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        labels = inputs.get("labels")
        if labels is None:
            labels = input_ids.masked_fill(~attention_mask.bool(), -100)
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # The canvas covers the last supervised span only: the denoising loss needs one contiguous response.
        supervised = labels != -100
        positions = torch.arange(seq_len, device = device)
        span_starts = supervised & ~F.pad(supervised, (1, 0))[:, :-1]
        span_end = torch.where(supervised, positions, -1).amax(dim = 1)
        prefix_len = torch.where(span_starts, positions, -1).amax(dim = 1).clamp(min = 0)
        response_len = (span_end - prefix_len + 1) * (span_end >= 0)

        # A clipped response has no real end, so its EOS fill must not teach the model to stop there.
        if max_length is None:
            truncated = torch.zeros(batch_size, dtype = torch.bool, device = device)
        else:
            real_len = attention_mask.sum(dim = 1)
            truncated = (real_len >= max_length) & (span_end == real_len - 1)

        # One response block per example; the decoder sees the prompt plus clean blocks before it, nothing later.
        num_blocks = (response_len - 1).clamp(min = 0) // block_size + 1
        block_idx = (torch.rand(batch_size, device = device) * num_blocks).long()
        encoder_len = prefix_len + block_idx * block_size

        offsets = torch.arange(block_size, device = device)
        abs_idx = (encoder_len[:, None] + offsets).clamp(max = seq_len - 1)
        in_response = offsets < (response_len - block_idx * block_size)[:, None]
        canvas_target = torch.where(in_response, input_ids.gather(1, abs_idx), eos_token_id)

        # Uniform random-token noise, no mask token.
        t = eps + (1 - 2 * eps) * torch.rand(batch_size, 1, device = device)
        corrupt = torch.rand(batch_size, block_size, device = device) < t
        random_tokens = torch.randint(vocab_size, (batch_size, block_size), device = device)
        canvas_ids = torch.where(corrupt, random_tokens, canvas_target)

        cache_mask = positions < encoder_len[:, None]
        canvas_mask = torch.ones(batch_size, block_size, dtype = torch.bool, device = device)
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": canvas_ids,
            "decoder_attention_mask": torch.cat([cache_mask, canvas_mask], dim = 1),
            "decoder_position_ids": encoder_len[:, None] + offsets,
        }
        # Two-pass self-conditioning: a no-grad first pass supplies the logits, kept per example with prob p.
        with torch.no_grad():
            model_kwargs["self_conditioning_logits"] = model(**model_kwargs).logits
        model_kwargs["self_conditioning_mask"] = torch.rand(batch_size, device = device) < sc_p
        outputs = model(**model_kwargs)

        # Flat CE over the whole canvas, corrupted and clean alike: the uniform kernel's ELBO has no 1/t weight.
        diffusion_target = canvas_target.masked_fill(truncated[:, None] & ~in_response, -100)
        # A row with no supervised token (completion truncated away) would otherwise train an all-EOS canvas.
        diffusion_target = diffusion_target.masked_fill((span_end < 0)[:, None], -100)
        diffusion_logits = outputs.logits
        if prediction_type == "mean_loo":
            # The softmax then parameterises the leave-one-out posterior (arXiv 2605.22765); convert it to the
            # denoiser in fp32, the correction reaches ~log(K) nats.
            alpha = (1 - t).float()
            bump = torch.log1p(diffusion_logits.shape[-1] * alpha / (1 - alpha))
            diffusion_logits = diffusion_logits.float().scatter_add(
                2, canvas_ids.unsqueeze(-1), bump[:, :, None].expand(-1, block_size, 1)
            )
        if (diffusion_target != -100).any():
            diffusion_loss = F.cross_entropy(
                diffusion_logits.flatten(0, 1), diffusion_target.flatten(), ignore_index = -100
            )
        else:
            diffusion_loss = outputs.logits.sum() * 0.0

        # Autoregressive co-loss on the causal encoder over every valid next-token pair.
        lm_head = base.get_output_embeddings()
        hidden_states = outputs.encoder_last_hidden_state.to(lm_head.weight.dtype)
        with _gather_lm_head(lm_head.weight, lm_head.bias):
            encoder_logits = hidden_states @ lm_head.weight.t()
            if lm_head.bias is not None:
                encoder_logits = encoder_logits + lm_head.bias
        encoder_logits = encoder_logits.float()
        encoder_logits = torch.tanh(encoder_logits / softcap) * softcap
        ar_mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
        ar_targets = torch.where(ar_mask, input_ids[:, 1:], -100)
        ar_loss = F.cross_entropy(
            encoder_logits[:, :-1].flatten(0, 1), ar_targets.flatten(), ignore_index = -100
        )

        loss = diffusion_loss + ar_weight * ar_loss
        metrics = {
            "diffusion_loss": diffusion_loss.detach().item(),
            "encoder_ar_loss": ar_loss.detach().item(),
            "mean_t": t.mean().item(),
        }
        return loss, outputs, metrics


DIFFUSION_GEMMA_PROFILE = register_diffusion_profile(
    DiffusionGemmaProfile(
        name = "diffusion_gemma",
        model_types = ("diffusion_gemma", "diffusion_gemma4"),
        architectures = (
            "DiffusionGemmaForBlockDiffusion",
            "DiffusionGemma4ModelForBlockDiffusion",
            "DiffusionGemma4ForBlockDiffusion",
        ),
        noise = "uniform",
        # Text attention + dense MLP of the encoder and decoder only: the suffix list would also wrap the
        # self-conditioning block, and the experts / router / vision tower stay frozen like the reference.
        lora_target_modules = (
            r".*model\.(encoder\.language_model|decoder)\.layers\.\d+\."
            r"(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
        ),
        defaults = {
            "diffusion_eps": 1e-4,
            "diffusion_self_conditioning_p": 0.5,
            "diffusion_encoder_ar_weight": 1.0,
            "diffusion_prediction_type": "mean",
            "attn_implementation": "eager",
        },
    )
)
