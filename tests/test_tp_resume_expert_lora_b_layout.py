# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A tensor / pipeline parallel Trainer checkpoint holds each rank's fused expert lora_B as trained. Both
UNSLOTH_MOE_LORA_B_LAYOUT settings give it the same shape, so a resume under the other one must be refused."""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import tpn  # noqa: E402
from test_tp_save_expert_factor_swap_gloo import _build  # noqa: E402

LAYOUT = "UNSLOTH_MOE_LORA_B_LAYOUT"


def _trainer(model):
    args = types.SimpleNamespace(should_save = False)
    return types.SimpleNamespace(model = model, optimizer = None, lr_scheduler = None, args = args)


def _experts(model, x):
    index = torch.arange(len(x)).view(-1, 1) % 4
    return model.base_model.model.experts(x, index, torch.ones(len(x), 1))


def _fresh(monkeypatch, layout):
    if layout == "rank_major":
        monkeypatch.delenv(LAYOUT, raising = False)
    else:
        monkeypatch.setenv(LAYOUT, layout)
    try:
        return _build("legacy")
    except Exception as e:
        pytest.skip(reason = f"unsloth_zoo without the fused Llama 4 expert LoRA: {e!r}")


@pytest.mark.parametrize("saved", ["grouped_by_expert", "rank_major"])
@pytest.mark.parametrize("resumed", ["grouped_by_expert", "rank_major"])
def test_resume_under_another_expert_lora_b_layout_is_refused(
    monkeypatch, tmp_path, saved, resumed
):
    model = _fresh(monkeypatch, saved)
    x = torch.randn(12, 16)
    trained = _experts(model, x)
    tpn._tp_save_optimizer_and_scheduler(_trainer(model), str(tmp_path))

    model = _fresh(monkeypatch, resumed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.zero_()
    if saved == resumed:
        tpn._tp_load_from_checkpoint(_trainer(model), str(tmp_path))
        torch.testing.assert_close(_experts(model, x), trained, rtol = 0, atol = 0)
    else:
        with pytest.raises(RuntimeError, match = LAYOUT):
            tpn._tp_load_from_checkpoint(_trainer(model), str(tmp_path))


def test_checkpoint_without_the_recorded_layout_still_resumes(monkeypatch, tmp_path):
    model = _fresh(monkeypatch, "grouped_by_expert")
    tpn._tp_save_optimizer_and_scheduler(_trainer(model), str(tmp_path))
    path = tpn._ckpt_path(str(tmp_path), 0)
    payload = torch.load(path)
    payload["layout"].pop("lora_b", None)
    torch.save(payload, path)
    tpn._tp_load_from_checkpoint(_trainer(model), str(tmp_path))
