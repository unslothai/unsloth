# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""gpt-oss MXFP4 LoRA without load_in_16bit must take unsloth_zoo's packed experts path.

Native matmul_ogs has no backward. Source-level (the real branch needs a checkpoint); no GPU.
"""

import ast
import inspect
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "unsloth" / "models" / "vision.py"
SRC = VISION.read_text(encoding = "utf-8")
TREE = ast.parse(SRC)
ZOO_MXFP4 = "unsloth_zoo.temporary_patches.mxfp4"


def _helper():
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_mxfp4_lora_keeps_experts_packed":
            ns = {}
            exec(ast.get_source_segment(SRC, node), ns)
            return ns["_mxfp4_lora_keeps_experts_packed"]
    raise AssertionError("_mxfp4_lora_keeps_experts_packed not found in vision.py")


def _dequantize_branch():
    """The `if` in FastBaseModel.from_pretrained that sets quantizer_kwargs["dequantize"]."""
    found = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.If):
            continue
        body = "\n".join(ast.get_source_segment(SRC, stmt) or "" for stmt in node.body)
        if 'quantizer_kwargs["dequantize"] = True' in body and len(node.body) == 1:
            found.append(node)
    assert len(found) == 1, "the dequantize branch moved; update this test"
    return found[0]


class _Mxfp4Config:
    def __init__(
        self,
        modules_to_not_convert = None,
        dequantize = False,
        **kwargs,
    ):
        pass


class _OtherConfig:
    def __init__(
        self,
        bits = 4,
        **kwargs,
    ):
        pass


SHARD = re.compile(
    r"(?P<dir>.*/)?(?P<stem>model|pytorch_model)(?P<variant>\.[^.-]+)?-\d+-of-\d+\.(?P<ext>safetensors|bin)"
)


def _index_name(shard):
    # The index a sharded save writes next to its shards (transformers' _add_variant naming).
    m = SHARD.fullmatch(shard)
    if m is None:
        return None
    variant = m["variant"] or ""
    return f"{m['dir'] or ''}{m['stem']}.{m['ext']}.index{variant}.json"


@pytest.fixture
def sizes(monkeypatch, tmp_path):
    """Hermetic checkpoint size (Hub file metadata) and free accelerator memory, in GiB.

    Every sharded group gets the index a real save writes, listing all its shards, unless
    state["index"] pins that index's shard list.
    """
    import huggingface_hub
    import torch

    state = {
        "checkpoint": 13,
        "free": [80],
        "hub_raises": False,
        "calls": [],
        "probes": [],
        "extra": [],
        "snapshot": None,
        "snapshot_calls": [],
        "tokens": [],
        "index": {},
    }
    GiB = 2**30

    class _Sibling:
        def __init__(self, name, size):
            self.rfilename, self.size = name, size

    class _Api:
        def model_info(
            self,
            repo_id,
            revision = None,
            files_metadata = False,
            token = None,
        ):
            state["calls"].append((repo_id, revision))
            state["tokens"].append(token)
            if state["hub_raises"]:
                raise OSError("offline")
            half = int(state["checkpoint"] * GiB / 2)
            files = [("config.json", 1000)]
            if state["checkpoint"]:
                files += [
                    ("model-00001-of-00002.safetensors", half),
                    ("model-00002-of-00002.safetensors", half),
                ]
            files += [(name, int(gib * GiB)) for name, gib in state["extra"]]
            indexes = {_index_name(name) for name, _ in files} - {None}
            files += [(name, 100) for name in sorted(indexes)]
            state["files"] = [name for name, _ in files]
            return types.SimpleNamespace(siblings = [_Sibling(name, size) for name, size in files])

    def _hf_hub_download(repo_id, filename, **kwargs):
        if filename in state["index"]:
            shards = state["index"][filename]
        else:
            shards = [
                name.rsplit("/", 1)[-1] for name in state["files"] if _index_name(name) == filename
            ]
        path = tmp_path / "_hub" / filename
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}))
        return str(path)

    def _try_to_load_from_cache(repo_id, filename, **kwargs):
        state["snapshot_calls"].append((repo_id, kwargs))
        if state["snapshot"] is None or not os.path.isfile(
            os.path.join(state["snapshot"], filename)
        ):
            return None
        return os.path.join(state["snapshot"], filename)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _try_to_load_from_cache)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _hf_hub_download)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(state["free"]))

    def _mem_get_info(i):
        state["probes"].append(i)
        return (int(state["free"][i] * GiB), 0)

    monkeypatch.setattr(torch.cuda, "mem_get_info", _mem_get_info)
    return state


@pytest.fixture
def zoo(monkeypatch, sizes):
    """Install a stub unsloth_zoo.temporary_patches.mxfp4 whose gate returns `state["keep"]`."""
    state = {"keep": True, "raise": False}
    module = types.ModuleType(ZOO_MXFP4)

    def keep_mxfp4_experts_packed():
        if state["raise"]:
            raise RuntimeError("probe failed")
        return state["keep"]

    module.keep_mxfp4_experts_packed = keep_mxfp4_experts_packed
    monkeypatch.setitem(sys.modules, ZOO_MXFP4, module)
    return state


def _run_branch(
    load_in_16bit,
    quant_method,
    full_finetuning,
    quantizer = _Mxfp4Config,
    device_map = "sequential",
):
    ns = {
        "inspect": inspect,
        "load_in_16bit": load_in_16bit,
        "quant_method": quant_method,
        "full_finetuning": full_finetuning,
        "device_map": device_map,
        "quantizer": quantizer,
        "quantizer_kwargs": {},
        "_mxfp4_lora_keeps_experts_packed": _helper(),
        "model_name": "openai/gpt-oss-20b",
        "kwargs": {},
        "_revision": None,
        "token": None,
    }
    node = _dequantize_branch()
    code = ast.Module(body = [node], type_ignores = [])
    ast.fix_missing_locations(code)
    exec(compile(code, str(VISION), "exec"), ns)
    return ns["quantizer_kwargs"].get("dequantize", False)


def test_helper_on_for_mxfp4_lora_when_zoo_keeps_packed(zoo):
    assert _helper()("mxfp4", False, "cuda:0") is True
    assert _helper()("MXFP4", False, {"": 0}) is True


def test_helper_off_for_full_finetuning_and_other_methods(zoo):
    assert _helper()("mxfp4", True) is False
    for method in ("fp8", "bitsandbytes", "compressed-tensors", None):
        assert _helper()(method) is False


def test_helper_off_when_zoo_declines_or_fails(zoo):
    zoo["keep"] = False
    assert _helper()("mxfp4") is False
    zoo["keep"] = True
    zoo["raise"] = True
    assert _helper()("mxfp4") is False


def test_helper_off_without_the_zoo_packed_path(monkeypatch):
    monkeypatch.setitem(sys.modules, ZOO_MXFP4, types.ModuleType(ZOO_MXFP4))
    assert _helper()("mxfp4") is False


def test_native_mxfp4_lora_load_dequantizes_through_packed_path(zoo):
    # load_in_4bit = False without load_in_16bit: previously native Mxfp4GptOssExperts.
    assert _run_branch(False, "mxfp4", False) is True


def test_native_load_unchanged_without_packed_path(zoo):
    zoo["keep"] = False
    assert _run_branch(False, "mxfp4", False) is False
    zoo["keep"] = True
    assert _run_branch(False, "mxfp4", True) is False


def test_load_in_16bit_still_dequantizes(zoo):
    zoo["keep"] = False
    assert _run_branch(True, "mxfp4", False) is True
    assert _run_branch(True, "fp8", False) is True


def test_offloading_device_map_keeps_native_load(zoo):
    # Offloaded experts would be dequantized to a full 16 bit copy.
    offload = {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": "cpu", "lm_head": 0}
    assert _helper()("mxfp4", False, offload) is False
    assert _helper()("mxfp4", False, {**offload, "model.layers.1": "disk"}) is False
    assert _run_branch(False, "mxfp4", False, device_map = offload) is False
    assert _helper()("mxfp4", False, {"": 0}) is True
    assert _helper()("mxfp4", False, {"model.layers.0": 0, "model.layers.1": 1}) is True
    for device_map in ("sequential", "auto"):
        assert _run_branch(False, "mxfp4", False, device_map = device_map) is True
    # load_in_16bit asked for the dequantize itself, offload or not.
    assert _run_branch(True, "mxfp4", False, device_map = offload) is True


def test_quantizer_without_dequantize_argument_untouched(zoo):
    assert _run_branch(False, "mxfp4", False, quantizer = _OtherConfig) is False
    assert _run_branch(True, "mxfp4", False, quantizer = _OtherConfig) is False


def test_placement_strategy_that_would_offload_keeps_native_load(zoo, sizes):
    sizes["checkpoint"], sizes["free"] = 65, [40]
    for device_map in ("sequential", "auto", "balanced", "balanced_low_0"):
        assert _helper()("mxfp4", False, device_map, "openai/gpt-oss-120b") is False
        assert _run_branch(False, "mxfp4", False, device_map = device_map) is False
    sizes["free"] = [40, 40]
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-120b") is True
    assert (
        _helper()("mxfp4", False, "auto", "openai/gpt-oss-120b", {0: "30GiB", 1: "30GiB"}) is False
    )
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-120b", {0: 40 * 2**30}) is False
    assert _helper()("mxfp4", False, {"": 0}, "openai/gpt-oss-120b") is True


def test_placement_strategy_keeps_packed_when_it_fits_or_size_is_unknown(zoo, sizes):
    sizes["checkpoint"], sizes["free"] = 13, [80]
    assert _helper()("mxfp4", False, "sequential", "openai/gpt-oss-20b") is True
    sizes["hub_raises"] = True
    sizes["free"] = [4]
    assert _helper()("mxfp4", False, "sequential", "openai/gpt-oss-20b") is True


def test_placement_strategy_sizes_a_local_checkpoint(zoo, sizes, tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"0" * 4096)
    sizes["free"] = [4096 / 2**30]
    assert _helper()("mxfp4", False, "auto", str(tmp_path)) is False
    sizes["free"] = [1.0]
    assert _helper()("mxfp4", False, "auto", str(tmp_path)) is True


def test_placement_strategy_sizes_the_pinned_revision(zoo, sizes):
    _helper()("mxfp4", False, "auto", "openai/gpt-oss-120b", None, "refs/pr/7")
    assert sizes["calls"][-1] == ("openai/gpt-oss-120b", "refs/pr/7")
    _run_branch(False, "mxfp4", False, device_map = "auto")
    assert sizes["calls"][-1] == ("openai/gpt-oss-20b", None)


def test_zero_accelerator_capacity_counts_as_offload(zoo, sizes):
    sizes["checkpoint"], sizes["free"] = 13, [80, 80]
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-20b", {0: 0, 1: 0}) is False
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-20b", {"cpu": "200GiB"}) is False


def test_no_sizing_without_the_packed_path(zoo, sizes, monkeypatch):
    zoo["keep"] = False
    assert _helper()("mxfp4", False, "sequential", "openai/gpt-oss-20b") is False
    monkeypatch.setitem(sys.modules, ZOO_MXFP4, types.ModuleType(ZOO_MXFP4))
    assert _helper()("mxfp4", False, "sequential", "openai/gpt-oss-20b") is False
    assert sizes["calls"] == [] and sizes["probes"] == []


def test_named_cpu_device_map_keeps_native_load(zoo, sizes):
    for device_map in ("cpu", "disk", "cpu:0"):
        assert _helper()("mxfp4", False, device_map, "openai/gpt-oss-20b") is False
    assert sizes["calls"] == [] and sizes["probes"] == []


def test_only_the_loaded_weight_files_are_counted(zoo, sizes):
    # gpt-oss-120b's original/ copy (61 GiB) is never loaded.
    sizes["checkpoint"], sizes["free"] = 61, [80]
    sizes["extra"] = [("original/model--00001-of-00007.safetensors", 61)]
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-120b") is True
    sizes["extra"] = [
        ("model.fp16-00001-of-00002.safetensors", 30),
        ("model.fp16-00002-of-00002.safetensors", 30),
    ]
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-120b") is True
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-120b", None, None, "fp16") is True
    sizes["extra"] = [
        ("model.fp16-00001-of-00002.safetensors", 45),
        ("model.fp16-00002-of-00002.safetensors", 45),
    ]
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-120b", None, None, "fp16") is False


def test_max_memory_probes_only_the_allowed_cards(zoo, sizes):
    sizes["checkpoint"], sizes["free"] = 13, [80, 80, 80]
    assert (
        _helper()("mxfp4", False, "auto", "openai/gpt-oss-20b", {1: "40GiB", "cpu": "100GiB"})
        is True
    )
    assert sizes["probes"] == [1]


def test_offline_load_sizes_the_cached_snapshot(zoo, sizes, tmp_path):
    sizes["hub_raises"], sizes["free"] = True, [4096 / 2**30]
    (tmp_path / "model.safetensors").write_bytes(b"0" * 8192)
    sizes["snapshot"] = str(tmp_path)
    assert (
        _helper()("mxfp4", False, "auto", "openai/gpt-oss-20b", None, "main", None, "/cache")
        is False
    )
    repo, kwargs = sizes["snapshot_calls"][-1]
    assert repo == "openai/gpt-oss-20b"
    assert kwargs["revision"] == "main" and kwargs["cache_dir"] == "/cache"
    sizes["snapshot"] = None
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-20b") is True


def test_quantization_method_enum_is_recognized(zoo):
    from transformers.utils.quantization_config import QuantizationMethod
    if not hasattr(QuantizationMethod, "MXFP4"):
        pytest.skip("this transformers predates MXFP4")
    assert _helper()(QuantizationMethod.MXFP4, False, "cuda:0") is True


def test_torch_device_cpu_map_keeps_native_load(zoo, sizes):
    import torch

    assert _helper()("mxfp4", False, torch.device("cpu"), "openai/gpt-oss-20b") is False
    assert _helper()("mxfp4", False, torch.device("cuda:0"), "openai/gpt-oss-20b") is True
    assert sizes["calls"] == [] and sizes["probes"] == []


def test_subfolder_weights_are_sized(zoo, sizes, tmp_path):
    sizes["checkpoint"], sizes["free"] = 0, [80]
    sizes["extra"] = [
        ("hf/model-00001-of-00002.safetensors", 45),
        ("hf/model-00002-of-00002.safetensors", 45),
    ]
    assert _helper()("mxfp4", False, "auto", "org/repo") is True
    assert _helper()("mxfp4", False, "auto", "org/repo", None, None, None, None, "hf") is False
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "model.safetensors").write_bytes(b"0" * 8192)
    sizes["free"] = [4096 / 2**30]
    assert _helper()("mxfp4", False, "auto", str(tmp_path), None, None, None, None, "hf") is False
    assert _helper()("mxfp4", False, "auto", str(tmp_path)) is True


def test_files_transformers_ignores_are_not_counted(zoo, sizes):
    sizes["checkpoint"], sizes["free"] = 13, [80]
    sizes["extra"] = [
        ("consolidated.safetensors", 70),
        ("adapter_model.safetensors", 70),
        ("model_old.safetensors", 70),
    ]
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-20b") is True


def test_xpu_capacity_is_probed(zoo, sizes, monkeypatch):
    import torch

    GiB = 2**30
    probes = []

    def _xpu_mem(i):
        probes.append(i)
        return (8 * GiB, 0)

    xpu = types.SimpleNamespace(
        is_available = lambda: True, device_count = lambda: 1, mem_get_info = _xpu_mem
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "xpu", xpu, raising = False)
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-20b") is False
    assert probes == [0]


def test_a_card_that_fails_the_probe_adds_no_capacity(zoo, sizes, monkeypatch):
    import torch

    GiB = 2**30
    sizes["checkpoint"], sizes["free"] = 60, [40, 40]

    def _mem(i):
        if i == 1:
            raise RuntimeError("exclusive-process card")
        return (40 * GiB, 0)

    monkeypatch.setattr(torch.cuda, "mem_get_info", _mem)
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-20b") is False
    sizes["checkpoint"] = 13
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-20b") is True


def test_unset_device_map_follows_the_default_device(zoo, sizes, monkeypatch):
    import torch

    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("cpu"))
    assert _helper()("mxfp4", False, None) is False
    assert _run_branch(False, "mxfp4", False, device_map = None) is False
    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("cuda", 0))
    assert _helper()("mxfp4", False, None) is True


def test_indexed_cpu_in_an_explicit_map_keeps_native_load(zoo, sizes):
    import torch
    for cpu in ("cpu:0", torch.device("cpu", 0)):
        assert _helper()("mxfp4", False, {"model.layers.0": 0, "model.layers.1": cpu}) is False
    assert sizes["calls"] == [] and sizes["probes"] == []


def test_pytorch_bin_checkpoints_are_sized(zoo, sizes, tmp_path):
    sizes["checkpoint"], sizes["free"] = 0, [80]
    sizes["extra"] = [
        ("pytorch_model-00001-of-00002.bin", 45),
        ("pytorch_model-00002-of-00002.bin", 45),
    ]
    assert _helper()("mxfp4", False, "auto", "org/repo") is False
    sizes["checkpoint"] = 40
    assert _helper()("mxfp4", False, "auto", "org/repo") is True
    (tmp_path / "pytorch_model.bin").write_bytes(b"0" * 8192)
    sizes["free"] = [4096 / 2**30]
    assert _helper()("mxfp4", False, "auto", str(tmp_path)) is False


def test_local_files_only_skips_the_hub_lookup(zoo, sizes, tmp_path):
    sizes["free"] = [4096 / 2**30]
    (tmp_path / "model.safetensors").write_bytes(b"0" * 8192)
    sizes["snapshot"] = str(tmp_path)
    helper = _helper()
    assert (
        helper("mxfp4", False, "auto", "openai/gpt-oss-20b", None, None, None, None, None, True)
        is False
    )
    assert sizes["calls"] == [] and sizes["snapshot_calls"]


def test_the_caller_token_reaches_the_size_lookup(zoo, sizes):
    helper = _helper()
    assert (
        helper("mxfp4", False, "auto", "org/gated", None, None, None, None, None, False, "hf_x")
        is True
    )
    assert sizes["tokens"][-1] == "hf_x"


def test_use_safetensors_false_sizes_the_bin_files(zoo, sizes):
    sizes["checkpoint"], sizes["free"] = 40, [80]
    sizes["extra"] = [
        ("pytorch_model-00001-of-00002.bin", 45),
        ("pytorch_model-00002-of-00002.bin", 45),
    ]
    helper = _helper()
    assert helper("mxfp4", False, "auto", "org/repo") is True
    assert (
        helper("mxfp4", False, "auto", "org/repo", None, None, None, None, None, False, None, False)
        is False
    )


def test_only_the_file_or_index_from_pretrained_selects_is_sized(zoo, sizes, tmp_path):
    sizes["checkpoint"], sizes["free"] = 13, [80]
    sizes["extra"] = [(f"model-0000{i}-of-00003.safetensors", 30) for i in (1, 2, 3)]
    sizes["index"] = {
        "model.safetensors.index.json": [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
    }
    assert _helper()("mxfp4", False, "auto", "org/repo") is True
    sizes["index"] = {}
    assert _helper()("mxfp4", False, "auto", "org/repo") is False
    sizes["extra"] = [("model.safetensors", 13)] + sizes["extra"]
    assert _helper()("mxfp4", False, "auto", "org/repo") is True
    local = tmp_path / "local"
    local.mkdir()
    (local / "model-00001-of-00001.safetensors").write_bytes(b"0" * 8192)
    (local / "model-00001-of-00002.safetensors").write_bytes(b"0" * 64)
    (local / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
    )
    sizes["free"] = [4096 / 2**30]
    assert _helper()("mxfp4", False, "auto", str(local)) is True
    (local / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}})
    )
    assert _helper()("mxfp4", False, "auto", str(local)) is False


def test_index_shards_in_nested_folders_are_sized(zoo, sizes, tmp_path):
    local = tmp_path / "nested"
    (local / "weights").mkdir(parents = True)
    (local / "weights" / "model-00001-of-00001.safetensors").write_bytes(b"0" * 8192)
    (local / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "weights/model-00001-of-00001.safetensors"}})
    )
    sizes["free"] = [4096 / 2**30]
    assert _helper()("mxfp4", False, "auto", str(local)) is False


def test_an_explicit_max_memory_budget_is_used_whole(zoo, sizes):
    # accelerate uses max_memory as given (no margin).
    sizes["checkpoint"], sizes["free"] = 38, [80]
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-20b", {0: "40GiB"}) is True
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-20b", {0: "36GiB"}) is False
    sizes["free"] = [40]
    assert _helper()("mxfp4", False, "auto", "openai/gpt-oss-20b") is False


def test_balanced_low_0_still_counts_the_first_card(zoo, sizes):
    # get_balanced_memory(low_zero = True) still spills overflow onto GPU 0 before CPU.
    sizes["checkpoint"], sizes["free"] = 60, [80, 40]
    assert _helper()("mxfp4", False, "balanced_low_0", "openai/gpt-oss-120b") is True
    sizes["free"] = [30, 30]
    assert _helper()("mxfp4", False, "balanced_low_0", "openai/gpt-oss-120b") is False


def test_offline_sizing_ignores_repo_files_a_load_never_fetches(zoo, tmp_path, monkeypatch):
    # A real cache after an online load: its tree listing names metal/ and original/ files that were
    # never downloaded, which made snapshot_download(local_files_only=True) raise.
    import huggingface_hub
    import torch

    sha = "a" * 40
    repo = tmp_path / "models--org--gpt-oss-x"
    snapshot = repo / "snapshots" / sha
    snapshot.mkdir(parents = True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(sha)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}})
    )
    for shard in shards:
        (snapshot / shard).write_bytes(b"0" * 4096)
    names = ["config.json", "model.safetensors.index.json", *shards, "metal/model.bin"]
    names.append("original/model.safetensors")
    (repo / "trees").mkdir()
    (repo / "trees" / f"{sha}.json").write_text(
        json.dumps(
            {"format_version": 1, "files": {n: {"size": 1, "blob_id": "b" * 40} for n in names}}
        )
    )

    class _Offline:
        def model_info(self, *args, **kwargs):
            raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "HfApi", _Offline)
    from huggingface_hub.file_download import try_to_load_from_cache

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", try_to_load_from_cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (4096, 8192))
    args = ("mxfp4", False, "auto", "org/gpt-oss-x", None, None, None, str(tmp_path))
    assert _helper()(*args) is False
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (10**9, 10**9))
    assert _helper()(*args) is True
