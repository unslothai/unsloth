# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""accelerate's FSDP2 keeps optim = 'adamw_8bit': a bitsandbytes Adam / AdamW steps each rank's local shard after
Accelerator._prepare_fsdp2, and a bitsandbytes optimizer that cannot is refused before the first step."""

from __future__ import annotations


import pytest

pytest.importorskip("trl")


def test_accelerate_fsdp2_prepare_localizes_bitsandbytes_adamw():
    bnb = pytest.importorskip("bitsandbytes")
    import torch

    from unsloth.models.fsdp2 import _localize_or_refuse_bnb
    from unsloth.models.fsdp2_bnb_optim import is_local_shard_bnb_optimizer

    p = torch.nn.Parameter(torch.zeros(8))
    opt = bnb.optim.AdamW8bit([p], lr = 1e-3)
    _localize_or_refuse_bnb([p, opt])
    assert is_local_shard_bnb_optimizer(opt)

    torch_opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(8))])
    _localize_or_refuse_bnb([torch_opt])  # torch optimizers pass untouched
    assert not is_local_shard_bnb_optimizer(torch_opt)


def test_accelerate_fsdp2_refuses_bnb_that_cannot_step_local_shards(monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")
    import torch

    from unsloth.models.fsdp2 import _localize_or_refuse_bnb

    lion = bnb.optim.Lion8bit([torch.nn.Parameter(torch.zeros(8))], lr = 1e-4)
    with pytest.raises(ValueError, match = "Unsloth: FSDP2 .*Lion8bit.*adamw_torch"):
        _localize_or_refuse_bnb([lion])

    monkeypatch.setenv("UNSLOTH_FSDP2_BNB_LOCAL", "0")
    opt = bnb.optim.AdamW8bit([torch.nn.Parameter(torch.zeros(8))], lr = 1e-3)
    with pytest.raises(ValueError, match = "UNSLOTH_FSDP2_BNB_LOCAL=0"):
        _localize_or_refuse_bnb([opt])
