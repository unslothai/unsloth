# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context parallelism refuses n-gram lookup table models: a shard would hash without the previous tokens."""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _frozen_table_parallel as ftp  # noqa: E402

import unsloth  # noqa: E402, F401
from unsloth import context_parallel as cp  # noqa: E402


@pytest.fixture(autouse = True)
def _restore_environ():
    # ftp.load sets UNSLOTH_FROZEN_TABLE / UNSLOTH_IS_PRESENT / UNSLOTH_ZOO_DISABLE_GPU_INIT in this process; restore
    # them so later tests (and the subprocesses they spawn) see the session's own environment.
    keys = ("UNSLOTH_FROZEN_TABLE", "UNSLOTH_IS_PRESENT", "UNSLOTH_ZOO_DISABLE_GPU_INIT")
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.mark.parametrize("key", ["ple_layer_ids", "engram_layer_ids"])
def test_cp_refuses_ngram_tables(key):
    from transformers import PretrainedConfig

    config = PretrainedConfig(num_attention_heads = 8, num_hidden_layers = 2)
    setattr(config, key, [1])
    model = types.SimpleNamespace(config = config)
    with pytest.raises(NotImplementedError, match = "n-gram lookup tables"):
        cp._check_cp_model(model, 2, "ulysses")
    # Flip: without the guard the refusal (if any) is about something else.
    saved = cp._ngram_table_blocker
    cp._ngram_table_blocker = lambda c: None
    try:
        try:
            cp._check_cp_model(model, 2, "ulysses")
        except Exception as e:
            assert "n-gram" not in str(e)
    finally:
        cp._ngram_table_blocker = saved


@pytest.mark.skipif(ftp.available() is not None, reason = ftp.available() or "")
def test_a_sequence_shard_hashes_different_ngram_ids(tmp_path):
    path = ftp.make_checkpoint(str(tmp_path / "q"))
    ids, _ = ftp.batch(1)
    calls = []
    with ftp.cpu_forwards():
        model, _ = ftp.load(path, "disk")
        hooks = ftp._capture(model, calls)
        with torch.no_grad():
            model(input_ids = ids, use_cache = False)
            model(input_ids = ids[:, ftp.SEQ // 2 :], use_cache = False)
    for h in hooks:
        h.remove()
    whole, half = calls[0][1], calls[1][1]
    tail = whole[:, ftp.SEQ // 2 :]
    assert tail.shape == half.shape
    assert not torch.equal(
        tail, half
    )  # the first n-gram windows of the shard miss the previous tokens
