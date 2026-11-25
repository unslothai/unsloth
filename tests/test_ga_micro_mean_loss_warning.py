# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Remote heads returning their own CrossEntropyLoss mean fall back to a mean of micro-batch means under
gradient accumulation; Unsloth must say so once instead of silently. Helpers loaded by AST, CPU only."""

import collections.abc
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_accepts_loss_kwargs_remote_mean_loss as base  # noqa: E402


class _Log:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)


def _ns():
    base._NAMES.update({"_warn_micro_mean_loss", "_MICRO_MEAN_LOSS_CHECKED"})
    ns = base._load()
    ns["collections"] = collections
    ns["logger"] = _Log()
    return ns


def _trainer(model, accepts):
    return types.SimpleNamespace(
        model = model, model_accepts_loss_kwargs = accepts, compute_loss_func = None
    )


def _batches(n):
    ids = torch.ones(2, 4, dtype = torch.long)
    return [{"input_ids": ids, "labels": ids} for _ in range(n)]


def test_own_mean_head_warns_once_under_accumulation(tmp_path):
    ns = _ns()
    mods = base._models(tmp_path)
    model = mods.NemotronHForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](model)
    trainer = _trainer(model, getattr(model, "accepts_loss_kwargs"))
    for _ in range(3):
        ns["_warn_micro_mean_loss"](trainer, _batches(4))
    assert len(ns["logger"].messages) == 1
    assert "NemotronHForCausalLM" in ns["logger"].messages[0]
    assert "gradient accumulation" in ns["logger"].messages[0]


def test_no_warning_without_accumulation_or_for_count_consumers(tmp_path):
    ns = _ns()
    mods = base._models(tmp_path)
    own_mean = mods.NemotronHForCausalLM()
    ns["_warn_micro_mean_loss"](_trainer(own_mean, False), _batches(1))
    ns["_warn_micro_mean_loss"](_trainer(own_mean, False), [[{"prompt": "x"}], [{"prompt": "y"}]])
    for cls in (mods.NativeForCausalLM, mods.HandRolledForCausalLM, mods.SumLossForCausalLM):
        model = cls()
        ns["apply_accepts_loss_kwargs_fix"](model)
        ns["_warn_micro_mean_loss"](
            _trainer(model, getattr(model, "accepts_loss_kwargs", True)), _batches(4)
        )
        ns["_warn_micro_mean_loss"](_trainer(model, False), _batches(4))
    assert ns["logger"].messages == []
