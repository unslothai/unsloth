# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Inkling-2-Mini and its pre-quantized bitsandbytes 4-bit repo: 4-bit loads resolve to
unsloth/Inkling-2-Mini-unsloth-bnb-4bit, 16-bit loads and full finetuning to the unsloth/Inkling-2-Mini mirror of
the original checkpoint, and a merged export of a model loaded from the pre-quantized repo reads the original."""

from unittest.mock import patch

import pytest

from unsloth.models import loader_utils
from unsloth.models.loader_utils import get_model_name

PREQUANTIZED = "unsloth/Inkling-2-Mini-unsloth-bnb-4bit"
MIRROR = "unsloth/Inkling-2-Mini"


def _no_remote_mapper():
    # int_to_float, float_to_int, map_to_16bit, fp8_block, fp8_row
    return {}, {}, {}, {}, {}


@pytest.mark.parametrize(
    "name, load_in_4bit, expected",
    [
        ("unsloth/Inkling-2-Mini", True, PREQUANTIZED),
        ("thinkingmachines/Inkling-2-Mini", True, PREQUANTIZED),
        ("THINKINGMACHINES/inkling-2-mini", True, PREQUANTIZED),
        (PREQUANTIZED, True, PREQUANTIZED),
        (PREQUANTIZED, False, MIRROR),
        ("thinkingmachines/Inkling-2-Mini", False, MIRROR),
    ],
)
def test_inkling2_mini_resolution(name, load_in_4bit, expected):
    with patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper):
        mapped = get_model_name(name, load_in_4bit = load_in_4bit)
    # Exact case: case-sensitive mirrors and caches reject a lowercased repo id.
    assert mapped == expected


def test_inkling2_small_is_not_mapped_to_mini():
    with patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper):
        for load_in_4bit in (True, False):
            mapped = get_model_name("thinkingmachines/Inkling-2-Small", load_in_4bit = load_in_4bit)
            assert mapped in (None, "thinkingmachines/Inkling-2-Small")


def _config_dir(tmp_path, name, quant):
    import json

    path = tmp_path / name
    path.mkdir()
    config = {"model_type": "inkling2_loop", "text_config": {"model_type": "inkling2_loop_text"}}
    if quant is not None:
        config["quantization_config"] = quant
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


BNB = {"quant_method": "bitsandbytes", "load_in_4bit": True}
MODELOPT = {"quant_method": "modelopt", "quant_algo": "NVFP4"}


def test_merge_reads_the_original_checkpoint_of_the_prequantized_repo(tmp_path, monkeypatch):
    import huggingface_hub

    from unsloth.models.inkling2 import inkling2_merge_source

    configs = {
        PREQUANTIZED: _config_dir(tmp_path, "bnb", BNB),
        MIRROR: _config_dir(tmp_path, "orig", MODELOPT),
    }
    asked = []

    def fake_download(
        repo_id,
        filename,
        revision = None,
        token = None,
        **kwargs,
    ):
        asked.append((repo_id, filename, revision))
        return configs[repo_id] + "/" + filename

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    # The pre-quantized repo's packed 4-bit tensors cannot be merged: read the original, at its own revision.
    assert inkling2_merge_source(PREQUANTIZED, "abc123") == (MIRROR, None)
    # The original checkpoint (today's quantize-on-load path) is read as before, revision kept.
    assert inkling2_merge_source(MIRROR, "def456") == (MIRROR, "def456")
    assert asked == [(PREQUANTIZED, "config.json", "abc123"), (MIRROR, "config.json", "def456")]


def test_merge_refuses_a_prequantized_folder_with_no_known_original(tmp_path):
    from unsloth.models.inkling2 import inkling2_merge_source

    folder = _config_dir(tmp_path, "local_bnb", BNB)
    with pytest.raises(NotImplementedError, match = "pre-quantized bitsandbytes 4-bit Inkling 2"):
        inkling2_merge_source(folder)
    original = _config_dir(tmp_path, "local_orig", MODELOPT)
    assert inkling2_merge_source(original, None) == (original, None)
    plain = _config_dir(tmp_path, "local_plain", None)
    assert inkling2_merge_source(plain, "r") == (plain, "r")
