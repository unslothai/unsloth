# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What Studio says about multi-GPU diffusion output: only CFG-branch parallel (and component / layer splits) match
one card bit for bit; the batch split of an even batch may differ, and several picked cards are used together."""

import ast
from pathlib import Path

from models.inference import DiffusionLoadRequest, VideoLoadRequest

MULTIGPU = Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion_multigpu.py"


def test_module_docstring_scopes_bit_identity_to_cfg_branch_parallel():
    doc = ast.get_docstring(ast.parse(MULTIGPU.read_text(encoding = "utf-8")))
    assert "outputs bit-identical" not in doc
    assert "CFG-branch parallel" in doc and "bit for bit" in doc
    assert "batch split" in doc and "may differ" in doc
    assert "UNSLOTH_DIFFUSION_BATCH_SPLIT=0" in doc


def test_gpu_ids_descriptions_match_multigpu_placement():
    for model in (DiffusionLoadRequest, VideoLoadRequest):
        text = model.model_fields["gpu_ids"].description
        # several cards are a multi-GPU selection (split or CFG replica), not one card picked by free VRAM
        assert "resolves to" not in text and "Neither engine shards" not in text
        assert "CFG replica" in text or "CFG\nreplica" in text
        assert "batch split" in text and "may differ" in text
        assert "UNSLOTH_DIFFUSION_BATCH_SPLIT=0" in text
