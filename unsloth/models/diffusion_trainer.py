# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""SFT for text-diffusion language models: SFTTrainer data handling, the family's own denoising loss."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
from trl import SFTConfig, SFTTrainer

from .diffusion_profiles import resolve_diffusion_profile

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

    def __post_init__(self):
        # Packed rows share one bidirectional attention window, so documents would see each other.
        self.packing = False
        self.padding_free = False
        super().__post_init__()


def _unwrap_config(model):
    for candidate in (model, getattr(model, "base_model", None), getattr(model, "module", None)):
        config = getattr(candidate, "config", None)
        if config is not None:
            return config
    return None


class DiffusionTrainer(SFTTrainer):
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
            args = DiffusionConfig(
                **{
                    k: v
                    for k, v in args.to_dict().items()
                    if k in DiffusionConfig.__dataclass_fields__
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
        super().__init__(model, args, *pos, **kwargs)
        # The loss is ours, so the Trainer must scale it for gradient accumulation itself.
        self.model_accepts_loss_kwargs = False

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
        mode = "train" if model.training else "eval"
        for key, value in (metrics or {}).items():
            self._diffusion_metrics[f"{mode}/{key}"].append(float(value))
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def log(self, logs, *args, **kwargs):
        mode = "eval" if any(k.startswith("eval_") for k in logs) else "train"
        prefix = "eval_" if mode == "eval" else ""
        for key in [k for k in self._diffusion_metrics if k.startswith(mode + "/")]:
            values = self._diffusion_metrics.pop(key)
            if values:
                logs[prefix + key.split("/", 1)[1]] = sum(values) / len(values)
        return super().log(logs, *args, **kwargs)
