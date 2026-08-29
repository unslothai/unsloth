# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from hub.utils import gguf


def test_checkpoint_family_matches_quantized_and_split_variants():
    assert gguf.gguf_checkpoint_family("model-a-Q4_K_M.gguf") == "model-a"
    assert gguf.gguf_checkpoint_family("model-a-Q8_0.gguf") == "model-a"
    assert gguf.gguf_checkpoint_family("model-Q4_K_M-00001-of-00002.gguf") == "model"
    assert gguf.gguf_checkpoint_family("model-Q4_K_M-00002-of-00002.gguf") == "model"


def test_checkpoint_family_keeps_distinct_models_separate():
    assert gguf.gguf_checkpoint_family("model-a-Q4_K_M.gguf") == "model-a"
    assert gguf.gguf_checkpoint_family("model-b-Q4_K_M.gguf") == "model-b"
    assert gguf.gguf_checkpoint_family("alpha.gguf") == "alpha"
    assert gguf.gguf_checkpoint_family("beta.gguf") == "beta"


def test_checkpoint_family_normalizes_quant_directories_and_windows_separators():
    assert gguf.gguf_checkpoint_family("Q4_K_M/model.gguf") == "model"
    assert gguf.gguf_checkpoint_family(r"Q8_0\model.gguf") == "model"
    assert gguf.gguf_checkpoint_family("Q4_K_M.gguf") is None
