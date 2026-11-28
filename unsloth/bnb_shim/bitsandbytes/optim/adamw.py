# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""bitsandbytes AdamW (plain, 8-bit, 32-bit, paged) for Unsloth's bitsandbytes stand-in: Adam
with weight_decay defaulting to 1e-2 (bitsandbytes applies weight decay decoupled, after the Adam
step, for Adam and AdamW alike). transformers' optim="adamw_8bit" / "adamw_bnb_8bit" /
"paged_adamw_8bit" build AdamW with optim_bits=8 (is_paged for the paged one)."""

from .adam import _AdamBase

__all__ = ["AdamW", "AdamW8bit", "AdamW32bit", "PagedAdamW", "PagedAdamW8bit", "PagedAdamW32bit"]


class AdamW(_AdamBase):
    _WEIGHT_DECAY = 1e-2


class AdamW8bit(_AdamBase):
    _WEIGHT_DECAY, _BITS = 1e-2, 8


class AdamW32bit(_AdamBase):
    _WEIGHT_DECAY, _BITS = 1e-2, 32


class PagedAdamW(_AdamBase):
    _WEIGHT_DECAY, _PAGED = 1e-2, True


class PagedAdamW8bit(_AdamBase):
    _WEIGHT_DECAY, _BITS, _PAGED = 1e-2, 8, True


class PagedAdamW32bit(_AdamBase):
    _WEIGHT_DECAY, _BITS, _PAGED = 1e-2, 32, True
