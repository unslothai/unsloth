# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""Behavioral regressions for EXL3 conversion, loading and adapter persistence."""

import argparse
import json
import multiprocessing
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import save_file

import unsloth.exllama as X
from unsloth.exllama import loader as EL, moe as M, patcher as P, quantize as Q
from unsloth.models import loader as ML


def _source(path):
    path.mkdir(parents = True, exist_ok = True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
                "hidden_size": 128,
                "intermediate_size": 256,
                "vision_config": {"patch_size": 16},
            }
        )
    )
    save_file({"weight": torch.ones(128, 128)}, str(path / "original.safetensors"))
    return path


def _complete(path):
    path = Path(path)
    path.mkdir(parents = True, exist_ok = True)
    (path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "exl3"}}))
    save_file({"layer.trellis": torch.ones(1, dtype = torch.int16)}, str(path / "model.safetensors"))


@pytest.fixture
def converter(monkeypatch):
    cm = types.ModuleType("exllamav3.conversion.convert_model")
    cm.parser = argparse.ArgumentParser()
    cm.prepare = lambda args: (args, {}, True, None)
    cm.main = lambda args, job: _complete(args.out_dir)
    pkg = types.ModuleType("exllamav3")
    conv = types.ModuleType("exllamav3.conversion")
    pkg.conversion = conv
    conv.convert_model = cm
    for name, module in {
        "exllamav3": pkg,
        "exllamav3.conversion": conv,
        "exllamav3.conversion.convert_model": cm,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(Q, "require_exllama", lambda: None)
    monkeypatch.setattr(P, "patch_transformers_exl3", lambda: True)
    monkeypatch.setattr(Q, "_maybe_enable_uncalibrated", lambda cfg: None)
    return cm


@pytest.mark.parametrize(
    "overlap",
    [
        "equal",
        "source_parent",
        "source_child",
        "work_source",
        "work_child",
        "work_output",
        "output_parent",
        "symlink",
    ],
)
def test_conversion_rejects_overlaps_before_any_write(tmp_path, converter, overlap):
    source = _source(tmp_path / "source")
    output, work = tmp_path / "output", tmp_path / "work"
    if overlap == "equal":
        output = source
    elif overlap == "source_parent":
        output = tmp_path
    elif overlap == "source_child":
        output = source / "output"
    elif overlap == "work_source":
        work = source
    elif overlap == "work_child":
        work = source / "work"
    elif overlap == "work_output":
        work = output
    elif overlap == "output_parent":
        work = output / "work"
    else:
        output.symlink_to(source, target_is_directory = True)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    converter.prepare = Mock(side_effect = AssertionError("converter must not start"))
    with pytest.raises(ValueError, match = "must not overlap"):
        Q.quantize_to_exl3(
            str(source),
            Q.Exl3Config(calibrate = False),
            out_dir = str(output),
            work_dir = str(work),
            overwrite = True,
        )
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    converter.prepare.assert_not_called()


def test_completion_validates_config_shards_and_tensor_manifest(tmp_path):
    output = tmp_path / "output"
    _complete(output)
    assert Q._conversion_complete(str(output))
    config = output / "config.json"
    saved = config.read_text()
    config.unlink()
    index = output / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.trellis": "model.safetensors",
                    "other.trellis": "missing.safetensors",
                }
            }
        )
    )
    assert not Q._conversion_complete(str(output))
    config.write_text(saved)
    assert not Q._conversion_complete(str(output))
    save_file(
        {"wrong.trellis": torch.ones(1, dtype = torch.int16)}, str(output / "missing.safetensors")
    )
    assert not Q._conversion_complete(str(output))
    save_file(
        {"other.trellis": torch.ones(1, dtype = torch.int16)}, str(output / "missing.safetensors")
    )
    assert Q._conversion_complete(str(output))
    (output / "missing.safetensors").write_bytes(b"partial")
    assert not Q._conversion_complete(str(output))
    config.write_text("{")
    assert not Q._conversion_complete(str(output))


def _convert_worker(source, output, work, result, entered, release, calls, ready):
    cm = types.ModuleType("exllamav3.conversion.convert_model")
    cm.parser = argparse.ArgumentParser()

    def prepare(args):
        with calls.get_lock():
            calls.value += 1
        entered.set()
        assert release.wait(30)
        return args, {}, True, None

    cm.prepare = prepare
    cm.main = lambda args, job: _complete(args.out_dir)
    pkg = types.ModuleType("exllamav3")
    conv = types.ModuleType("exllamav3.conversion")
    pkg.conversion, conv.convert_model = conv, cm
    with (
        patch.dict(
            sys.modules,
            {
                "exllamav3": pkg,
                "exllamav3.conversion": conv,
                "exllamav3.conversion.convert_model": cm,
            },
        ),
        patch.object(Q, "require_exllama", lambda: None),
        patch.object(P, "patch_transformers_exl3", lambda: True),
        patch.object(Q, "_maybe_enable_uncalibrated", lambda cfg: None),
    ):
        try:
            ready.set()
            result.put(
                Q.quantize_to_exl3(
                    source, Q.Exl3Config(calibrate = False), out_dir = output, work_dir = work
                )
            )
        except Exception as error:
            result.put(type(error).__name__ + ": " + str(error))


@pytest.mark.parametrize("shared_output", [True, False])
def test_concurrent_processes_serialize_output_and_work(tmp_path, shared_output):
    source = _source(tmp_path / "source")
    first_output, work = str(tmp_path / "output"), str(tmp_path / "work")
    second_output = first_output if shared_output else str(tmp_path / "other-output")
    ctx = multiprocessing.get_context("spawn")
    entered, release, result = ctx.Event(), ctx.Event(), ctx.Queue()
    calls = ctx.Value("i", 0)
    arguments = (result, entered, release, calls)
    first_ready, second_ready = ctx.Event(), ctx.Event()
    first = ctx.Process(
        target = _convert_worker, args = (str(source), first_output, work, *arguments, first_ready)
    )
    second = ctx.Process(
        target = _convert_worker, args = (str(source), second_output, work, *arguments, second_ready)
    )
    first.start()
    assert entered.wait(30)
    second.start()
    try:
        assert second_ready.wait(30)
        time.sleep(0.25)
        assert calls.value == 1
        release.set()
        first.join(30)
        second.join(30)
        assert first.exitcode == second.exitcode == 0
        assert sorted([result.get(timeout = 2), result.get(timeout = 2)]) == sorted(
            [first_output, second_output]
        )
        assert calls.value == (1 if shared_output else 2)
    finally:
        release.set()
        for proc in (first, second):
            if proc.is_alive():
                proc.terminate()
            proc.join(5)


def test_distinct_outputs_with_shared_work_are_serialized(tmp_path, converter):
    source = _source(tmp_path / "source")
    active, peak = 0, 0
    guard = threading.Lock()

    def prepare(args):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        return args, {}, True, None

    def main(args, job):
        nonlocal active
        _complete(args.out_dir)
        with guard:
            active -= 1

    converter.prepare, converter.main = prepare, main
    with ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(
                lambda i: Q.quantize_to_exl3(
                    str(source),
                    Q.Exl3Config(calibrate = False),
                    out_dir = str(tmp_path / f"out{i}"),
                    work_dir = str(tmp_path / "shared"),
                ),
                range(2),
            )
        )
    assert len(results) == 2
    assert peak == 1


class Boundary(Exception):
    pass


@pytest.fixture
def routing(monkeypatch):
    monkeypatch.delenv("UNSLOTH_QUANT_BACKEND", raising = False)
    monkeypatch.setattr(EL, "is_exllama_available", lambda: True)
    monkeypatch.setattr(EL, "_cuda_available", lambda: True)
    monkeypatch.setattr(X, "require_exllama", lambda: None)
    monkeypatch.setattr(ML, "hf_login", lambda token = None: token)
    prepare = Mock(side_effect = Boundary("prepare"))
    monkeypatch.setattr(X, "prepare_exl3_checkpoint", prepare)
    return prepare


@pytest.mark.parametrize(
    "visible,rank,expected", [("3", None, "0"), ("GPU-abcd", None, "0"), ("2,3", "1", "1")]
)
def test_converter_receives_logical_device(tmp_path, monkeypatch, routing, visible, rank, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.delenv("LOCAL_RANK", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if rank is not None:
        monkeypatch.setenv("LOCAL_RANK", rank)
    with pytest.raises(Boundary, match = "prepare"):
        ML.FastLanguageModel.from_pretrained("org/base")
    assert routing.call_args.kwargs["devices"] == expected


@pytest.mark.parametrize("loader", [ML.FastLanguageModel, ML.FastModel])
def test_string_dtype_reaches_exl3_and_later_validation(tmp_path, monkeypatch, routing, loader):
    checkpoint = tmp_path / "checkpoint"
    _complete(checkpoint)
    routing.side_effect = lambda name, **kw: EL.Exl3LoadPlan(
        str(checkpoint), EL.Exl3Config(), False, name
    )
    monkeypatch.setattr(ML, "is_bfloat16_supported", lambda: True)
    monkeypatch.setattr(ML, "patch_compiled_autograd", Mock(side_effect = Boundary("after dtype")))
    with pytest.raises(Boundary, match = "after dtype"):
        loader.from_pretrained("org/base", dtype = "bfloat16")
    assert routing.call_args.kwargs["compute_dtype"] is torch.bfloat16


@pytest.mark.parametrize("previous", [None, "1"])
def test_moe_env_survives_failure_before_construction(tmp_path, monkeypatch, routing, previous):
    if previous is None:
        monkeypatch.delenv("UNSLOTH_MOE_GROUPED", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_MOE_GROUPED", previous)
    checkpoint = tmp_path / "checkpoint"
    _complete(checkpoint)
    routing.side_effect = lambda name, **kw: EL.Exl3LoadPlan(
        str(checkpoint), EL.Exl3Config(), False, name
    )
    with pytest.raises(AssertionError):
        ML.FastModel.from_pretrained("org/base", whisper_language = 123)
    assert os.environ.get("UNSLOTH_MOE_GROUPED") == previous


def test_delegation_retains_callback_and_qat(monkeypatch, routing):
    callback = Mock()
    delegated = Mock(return_value = (object(), object()))
    monkeypatch.setattr(ML.FastModel, "from_pretrained", delegated)
    ML.FastLanguageModel.from_pretrained(
        "org/base", on_model_resolved = callback, qat_scheme = "int8-int4"
    )
    assert delegated.call_args.kwargs["on_model_resolved"] is callback
    assert delegated.call_args.kwargs["qat_scheme"] == "int8-int4"


def test_qat_validation_precedes_conversion(routing):
    with pytest.raises(ValueError, match = "full_finetuning"):
        ML.FastLanguageModel.from_pretrained("org/base", qat_scheme = "int8-int4")
    routing.assert_not_called()


@pytest.mark.parametrize(
    "kwargs,name",
    [
        ({"fast_inference": True}, "org/base"),
        ({"load_in_8bit": True}, "org/base"),
        ({"load_in_fp8": True}, "org/base"),
        ({}, "org/base-bf16"),
        ({"load_in_16bit": True}, "org/base"),
    ],
)
@pytest.mark.parametrize("loader", [ML.FastLanguageModel, ML.FastModel])
def test_native_options_do_not_enter_conversion(monkeypatch, routing, kwargs, name, loader):
    selected = []
    original = EL.should_use_exl3

    def select(*args, **options):
        result = original(*args, **options)
        selected.append(result)
        return result

    monkeypatch.setattr(X, "should_use_exl3", select)
    monkeypatch.setattr(ML, "hf_login", Mock(side_effect = Boundary("native")))
    with pytest.raises(Boundary, match = "native"):
        loader.from_pretrained(name, **kwargs)
    assert selected == [False]
    routing.assert_not_called()


@pytest.mark.parametrize(
    "kwargs", [{"load_in_exl3": True}, {"quantization_config": {"quant_method": "exl3", "bits": 3}}]
)
def test_explicit_exl3_overrides_bf16_suffix(routing, kwargs):
    with pytest.raises(Boundary, match = "prepare"):
        ML.FastLanguageModel.from_pretrained("org/base-bf16", **kwargs)
    assert routing.call_count == 1


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("loader", [ML.FastLanguageModel, ML.FastModel])
def test_vllm_rejects_explicit_or_local_exl3_before_conversion(tmp_path, routing, local, loader):
    if local:
        _complete(tmp_path)
        name, kwargs = str(tmp_path), {}
    else:
        name, kwargs = "org/base", {"load_in_exl3": True}
    with pytest.raises(ValueError, match = "vLLM"):
        loader.from_pretrained(name, fast_inference = True, **kwargs)
    routing.assert_not_called()


@pytest.mark.parametrize("option", ["load_in_8bit", "load_in_fp8"])
def test_conflicting_exl3_flags_fail_before_conversion(routing, option):
    with pytest.raises(ValueError, match = "cannot be combined"):
        ML.FastLanguageModel.from_pretrained("org/base", load_in_exl3 = True, **{option: True})
    routing.assert_not_called()


def test_real_hf_custom_cache_and_immutable_revision(tmp_path, monkeypatch):
    sha = "a" * 40
    cache = tmp_path / "cache"
    repo = cache / "models--org--base"
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text(sha)
    snapshot = repo / "snapshots" / sha
    _complete(snapshot)
    monkeypatch.setattr(EL, "require_exllama", lambda: None)
    monkeypatch.setattr(EL, "patch_transformers_exl3", lambda: True)
    plan = EL.prepare_exl3_checkpoint("org/base", cache_dir = str(cache), local_files_only = True)
    assert plan.checkpoint_dir == str(snapshot)
    assert plan.source_model == "org/base"
    assert plan.source_revision == sha


def test_outer_loader_forwards_cache_and_modelscope(monkeypatch, routing):
    monkeypatch.setattr(ML, "USE_MODELSCOPE", True)
    with pytest.raises(Boundary):
        ML.FastLanguageModel.from_pretrained("org/base", cache_dir = "custom", local_files_only = True)
    assert routing.call_args.kwargs["cache_dir"] == "custom"
    assert routing.call_args.kwargs["use_modelscope"] is True


def test_modelscope_preparation_keeps_provider_and_source(tmp_path, monkeypatch):
    _complete(tmp_path)
    provider = types.ModuleType("modelscope")
    provider.snapshot_download = Mock(return_value = str(tmp_path))
    monkeypatch.setitem(sys.modules, "modelscope", provider)
    monkeypatch.setattr(EL, "require_exllama", lambda: None)
    monkeypatch.setattr(EL, "patch_transformers_exl3", lambda: True)
    plan = EL.prepare_exl3_checkpoint(
        "org/base",
        use_modelscope = True,
        revision = "release",
        cache_dir = "custom",
        local_files_only = True,
    )
    assert plan.source_model == "org/base"
    assert plan.source_provider == "modelscope"
    assert plan.source_revision == "release"
    assert provider.snapshot_download.call_args.kwargs["cache_dir"] == "custom"
    assert provider.snapshot_download.call_args.kwargs["revision"] == "release"


@pytest.mark.parametrize("stamp_revision", [None, "b" * 40])
def test_adapter_base_revision_is_distinct_from_adapter_revision(tmp_path, routing, stamp_revision):
    base_revision = "a" * 40
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {"revision": base_revision, "base_model_name_or_path": "org/base", "peft_type": "LORA"}
        )
    )
    (tmp_path / EL.EXL3_ADAPTER_STAMP).write_text(
        json.dumps(
            {
                "source_model": "org/base",
                "source_revision": stamp_revision,
                "exl3_config": {"bits": 3},
            }
        )
    )
    with pytest.raises(Boundary):
        ML.FastLanguageModel.from_pretrained(str(tmp_path), revision = "adapter-branch")
    assert routing.call_args.kwargs["revision"] == (stamp_revision or base_revision)
    assert routing.call_args.args[0] == "org/base"


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 6)
        self.name_or_path = "temporary-conversion"
        self.config = {"model_type": "custom"}

    def forward(self, x):
        return self.proj(x)


def test_real_peft_stamps_only_saved_adapters_and_preserves_identity(tmp_path, monkeypatch):
    from peft import LoraConfig, get_peft_model

    revision = "a" * 40
    base = _Toy()
    EL.set_exl3_source_model_reference(
        base, "org/base", EL.Exl3Config(bits = 3), source_revision = revision
    )
    model = get_peft_model(base, LoraConfig(r = 2, target_modules = ["proj"]))
    model.add_adapter("domain", LoraConfig(r = 2, target_modules = ["proj"]))
    assert model.get_base_model().config["_commit_hash"] == revision
    model.save_pretrained(str(tmp_path / "all"))
    for name in ("default", "domain"):
        directory = tmp_path / "all" / (name if name != "default" else "")
        data = json.loads((directory / EL.EXL3_ADAPTER_STAMP).read_text())
        assert data["source_revision"] == revision
        config = json.loads((directory / "adapter_config.json").read_text())
        assert config["base_model_name_or_path"] == "org/base"
        assert config["revision"] == revision
    # PEFT allows selected_adapters and is_main_process positionally too.
    model.save_pretrained(str(tmp_path / "selected"), True, ["domain"])
    assert (tmp_path / "selected" / "domain" / EL.EXL3_ADAPTER_STAMP).is_file()
    assert not (tmp_path / "selected" / EL.EXL3_ADAPTER_STAMP).exists()
    model.save_pretrained(str(tmp_path / "worker"), True, ["domain"], "auto", False)
    assert not list((tmp_path / "worker").rglob(EL.EXL3_ADAPTER_STAMP))
    monkeypatch.setattr(EL, "is_exllama_available", lambda: True)
    route = EL.read_exl3_adapter_stamp(str(tmp_path / "all" / "domain"))
    source, cfg = route
    assert (source, cfg.bits, route.source_revision) == ("org/base", 3, revision)


def test_remote_adapter_stamp_uses_requested_cache(tmp_path, monkeypatch):
    (tmp_path / EL.EXL3_ADAPTER_STAMP).write_text(
        json.dumps({"source_model": "org/base", "source_revision": "a" * 40})
    )
    download = Mock(return_value = str(tmp_path / EL.EXL3_ADAPTER_STAMP))
    monkeypatch.setattr(EL, "is_exllama_available", lambda: True)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    assert EL.read_exl3_adapter_stamp("org/adapter", cache_dir = "custom") is not None
    assert download.call_args.kwargs["cache_dir"] == "custom"


def test_one_token_exl3_materialization_follows_activation_dtype():
    from unsloth.exllama.quant_linear import ExllamaV3Linear
    from unsloth.kernels.utils import fast_linear_forward

    inner = types.SimpleNamespace(
        in_features = 4, out_features = 6, trellis = torch.zeros(1, dtype = torch.int16)
    )
    weight = torch.arange(24, dtype = torch.float16).reshape(4, 6) / 100
    inner.get_weight_tensor = lambda: weight
    layer = ExllamaV3Linear(4, 6, inner, compute_dtype = torch.float16).bfloat16()
    x = torch.randn(1, 1, 4, dtype = torch.bfloat16)
    actual = fast_linear_forward(layer, x)
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, layer(x))


class _Experts(torch.nn.Module):
    def __init__(self, count = 64):
        super().__init__()
        self.num_experts, self.hidden_dim, self.intermediate_dim = count, 128, 128
        self.act_fn = torch.nn.SiLU()
        self.gate_up_proj = torch.nn.Parameter(torch.empty(count, 256, 128, device = "meta"))
        self.down_proj = torch.nn.Parameter(torch.empty(count, 128, 128, device = "meta"))


@pytest.fixture
def expert_storage(monkeypatch):
    storage = types.ModuleType("exllamav3.loader")
    storage.SafetensorsCollection = Mock()
    monkeypatch.setitem(sys.modules, "exllamav3.loader", storage)
    monkeypatch.setattr(M, "is_exllama_available", lambda: True)
    monkeypatch.setattr(M, "require_exllama", lambda: None)
    monkeypatch.delenv("UNSLOTH_EXL3_QUANTIZED_EXPERTS", raising = False)
    return storage.SafetensorsCollection


@pytest.mark.parametrize(
    "device_map", [None, "auto", "sequential", {"": "cpu"}, {"": "disk"}, {"other": "cuda:1"}]
)
def test_custom_experts_reject_unresolved_placement_before_storage(
    tmp_path, expert_storage, device_map
):
    model = torch.nn.Module()
    model.experts = _Experts()
    original = model.experts
    with pytest.raises(ValueError, match = "concrete CUDA device_map"):
        M.install_quantized_experts_before_load(model, str(tmp_path), device_map = device_map)
    expert_storage.assert_not_called()
    assert model.experts is original


def test_custom_experts_accept_concrete_cuda_placement(tmp_path, monkeypatch, expert_storage):
    model = torch.nn.Module()
    model.experts = _Experts()
    loads = Mock(return_value = object())
    monkeypatch.setattr(M, "_resolve_quant_linear", loads)
    count = M.install_quantized_experts_before_load(model, str(tmp_path), device_map = {"": "cuda:2"})
    assert count == 128
    assert all(call.args[-1] == torch.device("cuda:2") for call in loads.call_args_list)


def test_dense_and_native_experts_are_not_rejected(tmp_path, expert_storage):
    for model in (torch.nn.Linear(2, 2), torch.nn.Module()):
        if isinstance(model, torch.nn.Linear):
            model.experts = _Experts(count = 4)
        assert M.install_quantized_experts_before_load(model, str(tmp_path), device_map = "auto") == 0
    expert_storage.assert_not_called()


def test_quantizer_propagates_unsupported_expert_placement(tmp_path, expert_storage):
    class Quantizer:
        def _process_model_before_weight_loading(self, model, *args, **kwargs):
            pass

    P._patch_moe_expert_replacement(Quantizer)
    model = torch.nn.Module()
    model.experts = _Experts()
    model.name_or_path = str(tmp_path)
    with pytest.raises(ValueError, match = "concrete CUDA device_map"):
        Quantizer()._process_model_before_weight_loading(model, device_map = "auto")
    expert_storage.assert_not_called()
