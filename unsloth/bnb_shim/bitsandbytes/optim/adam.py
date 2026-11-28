# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""bitsandbytes Adam (plain, 8-bit, 32-bit, paged) for Unsloth's bitsandbytes stand-in, with
bitsandbytes' constructor signatures and defaults; AdamW lives in adamw.py. All are
Optimizer2State("adam"). Paged variants keep their state in ordinary device memory here (see
optimizer.py)."""

from .optimizer import Optimizer2State

__all__ = [
    "Adam",
    "Adam8bit",
    "Adam32bit",
    "PagedAdam",
    "PagedAdam8bit",
    "PagedAdam32bit",
]


def _no_amsgrad(name, amsgrad):
    if amsgrad:
        raise ValueError(f"{name} does not support amsgrad=True")


def _fixed_bits(name, optim_bits):
    if optim_bits != 32:
        # bitsandbytes: the default 32 is accepted for signature compatibility, nothing else.
        raise ValueError(f"{name} only supports optim_bits=32 (default value for compatibility)")


class _AdamBase(Optimizer2State):
    _WEIGHT_DECAY = 0.0
    _BITS = None  # None: from optim_bits
    _PAGED = None  # None: from is_paged

    def __init__(
        self,
        params,
        lr = 1e-3,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay = None,
        amsgrad = False,
        optim_bits = 32,
        args = None,
        min_8bit_size = 4096,
        is_paged = False,
        **kwargs,
    ):
        name = type(self).__name__
        if weight_decay is None:
            weight_decay = self._WEIGHT_DECAY
        bits = optim_bits
        if self._BITS is not None:
            if self._BITS == 8:
                _no_amsgrad(name, amsgrad)
                _fixed_bits(name, optim_bits)
            bits = self._BITS
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            bits,
            args,
            min_8bit_size,
            is_paged = is_paged if self._PAGED is None else self._PAGED,
            **kwargs,
        )


class Adam(_AdamBase):
    pass


class Adam8bit(_AdamBase):
    _BITS = 8


class Adam32bit(_AdamBase):
    _BITS = 32


class PagedAdam(_AdamBase):
    _PAGED = True


class PagedAdam8bit(_AdamBase):
    _BITS, _PAGED = 8, True


class PagedAdam32bit(_AdamBase):
    _BITS, _PAGED = 32, True
