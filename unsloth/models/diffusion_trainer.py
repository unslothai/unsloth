# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""SFT for text-diffusion language models: SFTTrainer data handling, the family's own denoising loss."""

from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Optional

import torch
from trl import SFTConfig

from ..trainer import UnslothTrainer
from .diffusion_loss import compile_backbone_layers, diffusion_compile_mode
from .diffusion_profiles import resolve_diffusion_profile, unwrap_diffusion_model

__all__ = ["DiffusionConfig", "DiffusionTrainer"]


@dataclass
class DiffusionConfig(SFTConfig):
    """``SFTConfig`` plus the diffusion objective knobs. ``None`` = the model family's reference value."""

    diffusion_eps: Optional[float] = field(
        default = None, metadata = {"help": "Noise level t is drawn from U(eps, 1 - eps)."}
    )
    diffusion_time_weighting: Optional[str] = field(
        default = None,
        metadata = {
            "help": "'inverse_t' (masked-diffusion ELBO), 'none', or a family-specific scheme."
        },
    )
    diffusion_mask_token_id: Optional[int] = field(
        default = None, metadata = {"help": "Override the [MASK] token for absorbing-noise families."}
    )
    diffusion_block_size: Optional[int] = field(
        default = None, metadata = {"help": "Block length for block-diffusion families."}
    )
    diffusion_self_conditioning_p: Optional[float] = field(
        default = None, metadata = {"help": "Probability of the two-pass self-conditioned forward."}
    )
    diffusion_encoder_ar_weight: Optional[float] = field(
        default = None,
        metadata = {
            "help": "Weight of the encoder autoregressive co-loss (encoder/decoder families)."
        },
    )
    diffusion_prediction_type: Optional[str] = field(
        default = None,
        metadata = {"help": "Family-specific prediction parameterisation (e.g. 'mean', 'mean_loo')."},
    )
    embedding_learning_rate: Optional[float] = field(
        default = None,
        metadata = {
            "help": "Learning rate for embed_tokens / lm_head in modules_to_save (continued pretraining)."
        },
    )
    diffusion_compile: Optional[str] = field(
        default = "auto",
        metadata = {
            "help": "'auto' compiles the chunked loss on CUDA; 'loss' always; 'full' (or True) also compiles "
            "each backbone layer; False / 'eager' disables. UNSLOTH_COMPILE_DISABLE=1 forces eager."
        },
    )

    def __post_init__(self):
        # Packed rows share one bidirectional attention window, so documents would see each other.
        self.packing = False
        self.padding_free = False
        # TRL 1.x enables MoE router logits from this coef; no diffusion objective adds the aux term.
        if hasattr(self, "router_aux_loss_coef"):
            self.router_aux_loss_coef = 0.0
        super().__post_init__()


def _unwrap_config(model):
    for candidate in (model, getattr(model, "base_model", None), getattr(model, "module", None)):
        config = getattr(candidate, "config", None)
        if config is not None:
            return config
    return None


class DiffusionTrainer(UnslothTrainer):
    def __init__(
        self,
        model = None,
        args = None,
        *pos,
        diffusion_profile = None,
        **kwargs,
    ):
        if args is None:
            args = DiffusionConfig(output_dir = "outputs")
        elif not isinstance(args, DiffusionConfig):
            # Not to_dict(): it masks every *_token field (eos_token "<EOS_TOKEN>", which SFTTrainer
            # rejects) and serialises objects such as parallelism_config, which then get dropped.
            args = DiffusionConfig(
                **{
                    f.name: getattr(args, f.name)
                    for f in fields(args)
                    if f.init and f.name in DiffusionConfig.__dataclass_fields__
                }
            )
        args.packing = False
        args.padding_free = False
        profile = diffusion_profile or resolve_diffusion_profile(_unwrap_config(model))
        if profile is None:
            raise ValueError(
                "Unsloth: DiffusionTrainer needs a text-diffusion model. "
                f"Got model_type={getattr(_unwrap_config(model), 'model_type', None)!r}."
            )
        self.diffusion_profile = profile
        self._diffusion_metrics = defaultdict(list)
        diffusion_compile_mode(args)  # reject a bad value before any work
        super().__init__(model, args, *pos, **kwargs)
        # A globally normalized loss already includes gradient accumulation in its denominator.
        self.model_accepts_loss_kwargs = self._uses_global_masked_tokens()
        if diffusion_compile_mode(args) == "full" and profile.compile_backbone:
            compile_backbone_layers(unwrap_diffusion_model(self.model))

    def _uses_global_masked_tokens(self):
        return getattr(self.diffusion_profile, "loss_normalization", None) == "global_masked_tokens"

    def get_batch_samples(
        self,
        epoch_iterator,
        num_batches,
        device = None,
    ):
        if not self._uses_global_masked_tokens():
            return super().get_batch_samples(epoch_iterator, num_batches, device)

        # Labels count eligible targets, not the random subset used by the diffusion loss.
        # Keep only inputs and noise for the window; forwards/backwards still run one microbatch at a time.
        batch_samples = []
        counts = []
        for _ in range(num_batches):
            try:
                inputs = next(epoch_iterator)
            except StopIteration:
                break
            inputs = self._prepare_inputs(inputs)
            prepared = self.diffusion_profile.prepare_batch(self.model, inputs, self.args)
            batch_samples.append({**inputs, "_diffusion_masked_batch": prepared})
            counts.append(prepared[3].sum())
        if not counts:
            return batch_samples, None
        num_items_in_batch = torch.stack(counts).sum()
        if self.args.average_tokens_across_devices and self.args.world_size > 1:
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()
        return batch_samples, num_items_in_batch

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        loss, outputs, metrics = self.diffusion_profile.compute_loss(
            model, inputs, self.args, num_items_in_batch = num_items_in_batch
        )
        if (
            self._uses_global_masked_tokens()
            and num_items_in_batch is not None
            and self.args.average_tokens_across_devices
        ):
            # DDP averages gradients; a denominator gathered across ranks needs a summed numerator.
            loss = loss * self.accelerator.num_processes
        mode = "train" if model.training else "eval"
        # Kept on device; one host sync per log() instead of one per metric per micro-batch.
        for key, value in (metrics or {}).items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().reshape(())
            self._diffusion_metrics[f"{mode}/{key}"].append(value)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def log(self, logs, *args, **kwargs):
        mode = "eval" if any(k.startswith("eval_") for k in logs) else "train"
        prefix = "eval_" if mode == "eval" else ""
        for key in [k for k in self._diffusion_metrics if k.startswith(mode + "/")]:
            values = self._diffusion_metrics.pop(key)
            if values:
                tensors = [v for v in values if isinstance(v, torch.Tensor)]
                total = sum(float(v) for v in values if not isinstance(v, torch.Tensor))
                if tensors:
                    device = tensors[0].device
                    total += float(torch.stack([v.to(device) for v in tensors]).sum())
                logs[prefix + key.split("/", 1)[1]] = total / len(values)
        return super().log(logs, *args, **kwargs)
