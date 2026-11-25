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

    cml = pytest.importorskip("transformers.core_model_loading")
    # The merged export imports the expert dequantizer transformers 5.24 added with Inkling 2 itself; the
    # pyproject pin (5.18) and the 5.23 latest leg have core_model_loading but not it.
    if not hasattr(cml, "Inkling2LoopDequantizeExperts"):
        pytest.skip("Inkling 2 needs transformers >= 5.24 (no Inkling2LoopDequantizeExperts here)")
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


def _snapshot_dir(
    root,
    repo_id,
    quant,
    revision = "a" * 40,
):
    import json

    path = root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
    path.mkdir(parents = True)
    config = {"model_type": "inkling2_loop", "text_config": {"model_type": "inkling2_loop_text"}}
    if quant is not None:
        config["quantization_config"] = quant
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def test_merge_maps_a_hub_cache_snapshot_of_the_prequantized_repo(tmp_path, monkeypatch):
    # A run started from the cached repo (Studio's pinned snapshot) records the snapshot folder, not the repo id. Its
    # files are the repo's at that revision, so it merges into the original exactly as the repo id does.
    from unsloth.models.inkling2 import inkling2_merge_source

    monkeypatch.setattr(loader_utils, "_get_new_mapper", _no_remote_mapper)
    snapshot = _snapshot_dir(tmp_path, PREQUANTIZED, BNB)
    assert inkling2_merge_source(snapshot, None) == (MIRROR, None)
    assert inkling2_merge_source(snapshot + "/", "abc") == (MIRROR, None)


def test_merge_still_refuses_snapshots_that_are_not_a_mapped_prequantized_repo(tmp_path):
    from unsloth.models.inkling2 import inkling2_merge_source

    # An unknown pre-quantized repo has no original to merge into.
    unknown = _snapshot_dir(tmp_path, "someone/Inkling-2-Mini-bnb-4bit", BNB)
    with patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper):
        with pytest.raises(NotImplementedError, match = "needs the original weights"):
            inkling2_merge_source(unknown)
    # A snapshot of the original checkpoint is read in place, at its revision.
    original = _snapshot_dir(tmp_path, MIRROR, MODELOPT)
    assert inkling2_merge_source(original, "r") == (original, "r")
    # A folder that only looks like a snapshot below its root (a subfolder of one) is a plain local folder.
    nested = _snapshot_dir(tmp_path / "x", PREQUANTIZED, BNB)
    inner = tmp_path / "x" / "copy"
    import shutil

    shutil.copytree(nested, inner)
    with pytest.raises(NotImplementedError, match = "needs the original weights"):
        inkling2_merge_source(str(inner))


@pytest.mark.parametrize(
    "path, expected",
    [
        (
            "/c/models--unsloth--Inkling-2-Mini-unsloth-bnb-4bit/snapshots/abc",
            ("unsloth/Inkling-2-Mini-unsloth-bnb-4bit", "abc"),
        ),
        ("/c/models--gpt2/snapshots/abc/", ("gpt2", "abc")),
        ("C:\\hf\\models--org--name\\snapshots\\abc", ("org/name", "abc")),
        ("/c/models--org--name/snapshots", None),
        ("/c/models--org--name/snapshots/abc/sub", None),
        ("/c/datasets--org--name/snapshots/abc", None),
        ("/c/models--a--b--c/snapshots/abc", None),
        ("/c/models----b/snapshots/abc", None),
        (None, None),
    ],
)
def test_hub_cache_snapshot_repo(path, expected):
    import os
    if path is not None and "\\" in path and os.sep != "\\":
        path = path.replace("C:", "").replace("\\", "/")
    assert loader_utils.hub_cache_snapshot_repo(path) == expected


def _merge_base_model_name():
    # save.py's resolver for the generic merge (zoo's merge_and_overwrite_lora), loaded without importing save.py.
    import ast
    import pathlib

    source = pathlib.Path(loader_utils.__file__).resolve().parents[1] / "save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_merge_base_model_name"
    ]
    assert nodes, "save.py has no _merge_base_model_name"
    namespace = {
        "get_model_name": get_model_name,
        "prequantized_snapshot_original": loader_utils.prequantized_snapshot_original,
    }
    exec(compile(ast.Module(nodes, []), str(source), "exec"), namespace)
    return namespace["_merge_base_model_name"]


def test_generic_merge_maps_a_cached_prequantized_snapshot_like_its_repo_id(tmp_path):
    # Not Inkling-specific: any mapped pre-quantized bitsandbytes repo cached in the hub layout resolves to the same
    # 16-bit original its repo id does, so the generic merge never reads packed 4-bit tensors.
    resolve = _merge_base_model_name()
    repo = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    with patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper):
        by_id = get_model_name(repo, load_in_4bit = False)
        assert by_id != repo
        snapshot = _snapshot_dir(tmp_path, repo, BNB)
        assert resolve(snapshot, load_in_4bit = False) == by_id
        # 16-bit snapshot: still the local folder (copied, never downloaded).
        plain = _snapshot_dir(tmp_path, "meta-llama/Llama-3.2-1B-Instruct", None)
        assert resolve(plain, load_in_4bit = False) == get_model_name(plain, load_in_4bit = False)
        # A plain local copy of the pre-quantized repo is not provably the repo: unchanged.
        local = _config_dir(tmp_path, "my_llama_bnb", BNB)
        assert resolve(local, load_in_4bit = False) == get_model_name(local, load_in_4bit = False)
        # 4-bit lookups are untouched.
        assert resolve(snapshot, load_in_4bit = True) == get_model_name(snapshot, load_in_4bit = True)


def test_in_place_4bit_merge_keeps_resolving_the_recorded_base(monkeypatch):
    # merged_4bit folds LoRA into the loaded weights: zoo still resolves the base (always with load_in_4bit = False),
    # so a cached snapshot must stay local there instead of being sent to the 16-bit original on the Hub.
    import ast
    import pathlib

    source = pathlib.Path(loader_utils.__file__).resolve().parents[1] / "save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    nodes = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_merge_base_resolver"
    ]
    assert nodes, "save.py has no _merge_base_resolver"
    sentinels = {
        "get_model_name": object(),
        "_merge_base_model_name": object(),
        "_modelscope_base_model_name": object(),
    }
    namespace = {"os": __import__("os"), **sentinels}
    exec(compile(ast.Module(nodes, []), str(source), "exec"), namespace)
    resolver = namespace["_merge_base_resolver"]
    monkeypatch.delenv("UNSLOTH_USE_MODELSCOPE", raising = False)
    assert resolver(True) is sentinels["get_model_name"]
    assert resolver(False) is sentinels["_merge_base_model_name"]
    monkeypatch.setenv("UNSLOTH_USE_MODELSCOPE", "1")
    assert resolver(True) is sentinels["get_model_name"]
    assert resolver(False) is sentinels["_modelscope_base_model_name"]
