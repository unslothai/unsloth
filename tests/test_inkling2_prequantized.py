# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Inkling-2-Mini and its pre-quantized bitsandbytes 4-bit repo: 4-bit loads resolve to
unsloth/Inkling-2-Mini-unsloth-bnb-4bit, 16-bit loads and full finetuning to the unsloth/Inkling-2-Mini mirror of
the original checkpoint, and a merged export of a model loaded from the pre-quantized repo reads the original."""

import sys
from unittest.mock import patch

import pytest

from unsloth.models import loader_utils
from unsloth.models.loader_utils import get_model_name

PREQUANTIZED = "unsloth/Inkling-2-Mini-unsloth-bnb-4bit"
MIRROR = "unsloth/Inkling-2-Mini"


def _no_remote_mapper():
    # int_to_float, float_to_int, map_to_16bit, fp8_block, fp8_row
    return {}, {}, {}, {}, {}


def _fake_zoo(monkeypatch, has_fix):
    import types

    import unsloth_zoo

    # A zoo with the fix has patching_utils.BARE_PARAMS4BIT_COMPUTE_DTYPE; an older one lacks the attribute.
    fake = types.ModuleType("unsloth_zoo.patching_utils")
    if has_fix:
        fake.BARE_PARAMS4BIT_COMPUTE_DTYPE = True
    monkeypatch.setitem(sys.modules, "unsloth_zoo.patching_utils", fake)
    monkeypatch.setattr(unsloth_zoo, "patching_utils", fake, raising = False)


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
def test_inkling2_mini_resolution(name, load_in_4bit, expected, monkeypatch):
    _fake_zoo(monkeypatch, True)
    with patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper):
        mapped = get_model_name(name, load_in_4bit = load_in_4bit)
    # Exact case: case-sensitive mirrors and caches reject a lowercased repo id.
    assert mapped == expected


@pytest.mark.parametrize("zoo_has_fix", [True, False])
@pytest.mark.parametrize(
    "name",
    ["unsloth/Inkling-2-Mini", "thinkingmachines/Inkling-2-Mini", PREQUANTIZED],
)
def test_inkling2_mini_4bit_gated_on_zoo_compute_dtype_fix(name, zoo_has_fix, monkeypatch):
    _fake_zoo(monkeypatch, zoo_has_fix)
    fetched = []

    def remote_mapper():
        fetched.append(True)
        return _no_remote_mapper()

    with patch.object(loader_utils, "_get_new_mapper", remote_mapper):
        mapped = get_model_name(name, load_in_4bit = True)
        if name != PREQUANTIZED:
            # The gate never falls through to the "update Unsloth" probe of the fetched mapper.
            assert fetched == []
        mapped_16bit = get_model_name(name, load_in_4bit = False)
    if zoo_has_fix or name == PREQUANTIZED:
        # An explicit pre-quantized name is the user's choice either way.
        assert mapped == PREQUANTIZED
    else:
        # Older zoo: keep quantize-on-load from the requested checkpoint, exactly as named.
        assert mapped == name
    # 16-bit loads do not depend on the fix.
    assert mapped_16bit == MIRROR


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


def test_merge_refuses_a_local_folder_named_like_the_prequantized_repo(tmp_path, monkeypatch):
    # A relative folder spelled as the Hub id is a local checkpoint: its tensors may not be the Hub repo's, so the
    # public original is not substituted for it.
    from unsloth.models.inkling2 import inkling2_merge_source

    (tmp_path / "unsloth").mkdir()
    _config_dir(tmp_path / "unsloth", "Inkling-2-Mini-unsloth-bnb-4bit", BNB)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(NotImplementedError, match = "use_exact_model_name = True"):
        inkling2_merge_source(PREQUANTIZED, None)


def test_merged_export_refuses_to_overwrite_the_prequantized_snapshot(tmp_path, monkeypatch):
    # Loaded from the Hub pre-quantized repo, merged into the original: the export folder must still not be the
    # loaded snapshot (it would delete its packed weights and overwrite its config).
    import types

    import huggingface_hub

    pytest.importorskip("transformers.core_model_loading")
    from unsloth.models.inkling2 import save_inkling2_merged_16bit

    snapshot = _config_dir(tmp_path, "snapshot", BNB)
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo_id, filename, **kwargs: snapshot + "/" + filename,
    )
    model = types.SimpleNamespace(
        config = types.SimpleNamespace(_name_or_path = PREQUANTIZED, _commit_hash = "abc123")
    )
    with pytest.raises(RuntimeError, match = "checkpoint it was loaded from"):
        save_inkling2_merged_16bit(model, None, snapshot)
    assert (tmp_path / "snapshot" / "config.json").exists()


def test_sharded_fill_refuses_a_converted_buffer_before_reading_any_tensor(tmp_path, monkeypatch):
    # The pre-quantized fallback retries with transformers' own load: the refusal must come before the first read,
    # buffers included, or the partial model stays alive through the retry.
    import torch
    from safetensors.torch import save_file

    from unsloth.models import tp_native

    save_file({"weight": torch.ones(4, 4)}, str(tmp_path / "model.safetensors"))
    with torch.device("meta"):
        model = torch.nn.Linear(4, 4, bias = False)
    model.register_buffer("table", torch.empty(4, device = "meta"))
    monkeypatch.setattr(tp_native, "_checkpoint_aliases", lambda *a, **k: {"table": None})
    with pytest.raises(tp_native._TensorByTensorUnsupported, match = "table"):
        tp_native._fill_from_checkpoint(
            model, str(tmp_path), {}, torch.device("cpu"), lambda old, reader, key: None
        )
    assert model.weight.is_meta


def _hub_cache(root, repo_id, files):
    snapshot = root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / ("0" * 40)
    snapshot.mkdir(parents = True)
    (root / f"models--{repo_id.replace('/', '--')}" / "refs").mkdir()
    (root / f"models--{repo_id.replace('/', '--')}" / "refs" / "main").write_text("0" * 40)
    for name in files:
        (snapshot / name).write_text("{}")


@pytest.mark.parametrize("cached", [False, True])
def test_inkling2_mini_4bit_offline_keeps_the_cached_original(cached, tmp_path, monkeypatch):
    # Offline, the pre-quantized repo is used only once it is fully cached; otherwise the requested checkpoint
    # loads with quantize-on-load, as before the remap.
    _fake_zoo(monkeypatch, True)
    _hub_cache(tmp_path, MIRROR, ["config.json", "model.safetensors"])
    if cached:
        _hub_cache(tmp_path, PREQUANTIZED, ["config.json", "model.safetensors"])
    with patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper):
        mapped = get_model_name(
            MIRROR, load_in_4bit = True, local_files_only = True, cache_dir = str(tmp_path)
        )
    assert mapped == (PREQUANTIZED if cached else MIRROR)
