# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import json
import math
import os
import random
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from unsloth.models import decision, decision_gguf
from unsloth.models.decision_gguf import (
    effective_temperatures,
    gguf_eligibility,
    read_decision_temperatures,
    write_decision_temperatures,
)

# gguf-py ships with llama.cpp, not on PyPI at the version with the decision keys.
if os.environ.get("UNSLOTH_TEST_GGUF_PY"):
    sys.path.insert(0, os.environ["UNSLOTH_TEST_GGUF_PY"])
try:
    import gguf
    import numpy as np
except ImportError:
    gguf = None
needs_gguf = pytest.mark.skipif(gguf is None, reason = "needs gguf-py")
contract = decision_gguf._contract()


def _llama_cpp_temperature(temperatures: dict, qtype: str, n: int) -> float:
    # server_decision_context::get_temperature for a non-Lev model (server-decision.cpp, b11443).
    bucket = "2" if n <= 2 else "3_5" if n <= 5 else "6_10" if n <= 10 else "11"
    for name in (f"{qtype}.{bucket}", qtype):
        if name in temperatures:
            return temperatures[name]
    return 1.0


def test_uncalibrated_models_serve_at_one():
    for layout in ("clef", "laya"):
        assert effective_temperatures({}, layout) == {"choice": 1.0, "score": 1.0, "noul": 1.0}


def test_folded_head_temperature_is_not_applied_twice():
    config = {"folded_temperature": 1.1728, "temperature": [1.229, 0.681, 0.768]}
    assert effective_temperatures(config, "clef") == {
        "choice": 1.229,
        "score": 0.681,
        "noul": 0.768,
    }


def test_unfolded_head_temperature_at_the_floor_multiplies_without_a_reclamp():
    config = {"head_temperature": 0.05, "temperature": [1.0, 0.5, 1.0]}
    got = effective_temperatures(config, "clef")
    assert got == pytest.approx({"choice": 0.05, "score": 0.025, "noul": 0.05})
    # Laya has no head temperature: one left in its config is not what PyTorch applies.
    assert effective_temperatures(config, "laya") == {"choice": 1.0, "score": 0.5, "noul": 1.0}


def test_per_type_and_bucket_values_clamp_as_pytorch_serving_clamps():
    config = {
        "head_temperature": 2.0,
        "temperature": [0.1, 9.0, float("nan")],
        "temperature_by_options": {
            "choice:3-5": 0.1006,
            "noul:2": 0.7,
            "choice:11+": 2.0,
            "score:6-10": 50,
            "bogus:2": 3.0,
            "choice:7": 3.0,
        },
    }
    assert effective_temperatures(config, "clef") == pytest.approx(
        {
            "choice": 1.0,
            "score": 10.0,
            "noul": 2.0,
            "choice.3_5": 1.0,
            "noul.2": 1.4,
            "choice.11": 4.0,
            "score.6_10": 10.0,
        }
    )


@pytest.mark.parametrize(
    "config",
    [
        {"head_temperature": float("nan")},
        {"head_temperature": 0.0},
        {"head_temperature": -1.0},
        {"head_temperature": float("inf")},
        {"head_temperature": "warm"},
        {"temperature": [1.0, 1.0]},
        {"temperature": 1.0},
        {"temperature_by_options": [["choice:2", 1.0]]},
    ],
)
def test_invalid_temperatures_are_rejected(config):
    with pytest.raises(ValueError):
        effective_temperatures(config, "clef")


def test_llama_cpp_lookup_matches_pytorch_serving_on_random_configs():
    laya = decision._laya()
    rng = random.Random(0)
    sizes = list(range(2, 30))
    for trial in range(300):
        config = {"temperature": [rng.uniform(0.05, 8.0) for _ in range(3)]}
        if trial % 2:
            config["head_temperature"] = rng.choice([0.05, rng.uniform(0.05, 20.0)])
        config["temperature_by_options"] = {
            laya.common.temp_bucket(rng.randrange(3), rng.choice(sizes)): rng.uniform(0.01, 9.0)
            for _ in range(rng.randrange(4))
        }
        items = [{"qtype": rng.randrange(3)} for _ in range(20)]
        logits = [torch.zeros(rng.choice(sizes)) for _ in items]
        layout = "clef"
        served = decision._served_temperatures(config, logits, items)
        written = effective_temperatures(config, layout)
        for item, z, want in zip(items, logits, served):
            got = _llama_cpp_temperature(written, decision.QUESTION_TYPES[item["qtype"]], len(z))
            assert got == pytest.approx(want, rel = 1e-12)


def _write_gguf(
    path: Path,
    temperatures: dict,
    max_head_tokens = None,
) -> None:
    writer = gguf.GGUFWriter(str(path), arch = "clef")
    writer.add_string("general.name", "tiny")
    writer.add_uint32("clef.decision.block_count", 4)
    if max_head_tokens is not None:
        writer.add_uint32("clef.decision.max_head_tokens", max_head_tokens)
    writer.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
    writer.add_float32("clef.attention.layer_norm_epsilon", 1e-5)
    for name, value in temperatures.items():
        writer.add_float32(f"clef.decision.temperature.{name}", value)
    rng = np.random.default_rng(0)
    writer.add_tensor("token_embd.weight", rng.standard_normal((3, 64), dtype = np.float32))
    weights = rng.standard_normal((4, 64), dtype = np.float32)
    quantized = gguf.quants.quantize(weights, gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor(
        "blk.0.ffn_up.weight",
        quantized,
        raw_shape = quantized.shape,
        raw_dtype = gguf.GGMLQuantizationType.Q8_0,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _snapshot(path: Path) -> tuple:
    reader = gguf.GGUFReader(str(path), "r")
    fields = {
        name: (field.types, field.contents())
        for name, field in reader.fields.items()
        if ".decision.temperature." not in name and not name.startswith("GGUF.")
    }
    tensors = {
        t.name: (t.tensor_type, tuple(t.shape), bytes(np.asarray(t.data).tobytes()))
        for t in reader.tensors
    }
    return fields, tensors


@needs_gguf
def test_metadata_writer_replaces_only_the_temperatures_and_is_idempotent(tmp_path):
    path = tmp_path / "model-Q8_0.gguf"
    _write_gguf(path, {"choice": 3.0, "choice.2": 0.1})
    os.chmod(path, 0o644)
    before = _snapshot(path)
    wanted = {"choice": 1.229, "score": 0.025, "noul": 0.05, "noul.2": 0.7}
    assert write_decision_temperatures(path, wanted) is True
    got = read_decision_temperatures(path)
    assert set(got) == set(wanted)
    assert all(np.float32(got[k]) == np.float32(v) for k, v in wanted.items())
    assert _snapshot(path) == before
    assert os.name == "nt" or (path.stat().st_mode & 0o777) == 0o644
    data = path.read_bytes()
    assert write_decision_temperatures(path, dict(wanted)) is False
    assert path.read_bytes() == data
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


@needs_gguf
def test_metadata_writer_leaves_the_file_alone_on_a_bad_value(tmp_path):
    path = tmp_path / "model.gguf"
    _write_gguf(path, {"choice": 3.0})
    data = path.read_bytes()
    with pytest.raises(ValueError):
        write_decision_temperatures(path, {"choice": float("nan")})
    assert path.read_bytes() == data
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def _clef_folder(
    folder: Path,
    architectures = ("Qwen3_5ForConditionalGeneration",),
    config = None,
) -> Path:
    folder.mkdir(parents = True, exist_ok = True)
    model_config = {"architectures": list(architectures)}
    if architectures and architectures[0] == "Qwen3_5ForConditionalGeneration":
        model_config["vision_config"] = {"depth": 1}
    (folder / "config.json").write_text(json.dumps(model_config))
    (folder / "joint_head_config.json").write_text(json.dumps({"width": 8}))
    (folder / "joint_head.safetensors").write_bytes(b"head-weights")
    (folder / "unsloth_decision_config.json").write_text(
        json.dumps(
            config if config is not None else {"head_temperature": 0.05, "temperature": [1, 0.5, 1]}
        )
    )
    return folder


def _laya_folder(folder: Path, model_type = "modernbert") -> Path:
    (folder / "encoder").mkdir(parents = True)
    (folder / "tokenizer").mkdir()
    (folder / "encoder" / "config.json").write_text(json.dumps({"model_type": model_type}))
    (folder / "rl_agent_config.json").write_text(json.dumps({"temperature": [1.2, 0.3, 1.0]}))
    (folder / "model.safetensors").write_bytes(b"weights")
    return folder


def test_fingerprint_and_export_json(tmp_path):
    folder = _clef_folder(tmp_path / "run")
    first = contract.fingerprint(folder, "clef")
    assert first == contract.fingerprint(folder, "clef")
    (folder / "config.json").write_text("{}")  # not part of what the GGUF decision path reads
    assert contract.fingerprint(folder, "clef") == first
    (folder / "joint_head.safetensors").write_bytes(b"other")
    assert contract.fingerprint(folder, "clef") != first
    # A stock Clef checkpoint has no Unsloth decision config.
    second = contract.fingerprint(folder, "clef")
    (folder / "unsloth_decision_config.json").unlink()
    assert contract.fingerprint(folder, "clef") not in (first, second)
    laya = _laya_folder(tmp_path / "laya")
    laya_print = contract.fingerprint(laya, "laya")
    (laya / "model.safetensors").write_bytes(b"trained")
    assert contract.fingerprint(laya, "laya") != laya_print

    out = tmp_path / "run" / contract.EXPORT_DIR
    out.mkdir()
    files = {"Q8_0": {"model": "model-Q8_0.gguf", "mmproj": "mmproj-Q8_0.gguf"}}
    data = decision_gguf._write_export(out, "clef", files, "abc")
    assert data == {
        "format": "unsloth-decision-gguf",
        "version": 1,
        "layout": "clef",
        "quantizations": ["Q8_0"],
        "files": files,
        "source_fingerprint": "abc",
    }
    assert contract.read_export(tmp_path / "run") == data
    umask = os.umask(0)
    os.umask(umask)
    assert os.name == "nt" or ((out / "export.json").stat().st_mode & 0o777) == 0o666 & ~umask
    assert sorted(p.name for p in out.iterdir()) == ["export.json"]


def test_eligibility(tmp_path):
    assert gguf_eligibility(_clef_folder(tmp_path / "q35"))["eligible"]
    assert gguf_eligibility(_clef_folder(tmp_path / "q35text", ("Qwen3_5ForCausalLM",)))["eligible"]
    for name, arch in (
        ("llama", "LlamaForCausalLM"),
        ("moe", "Qwen3_5MoeForConditionalGeneration"),
    ):
        verdict = gguf_eligibility(_clef_folder(tmp_path / name, (arch,)))
        assert verdict["eligible"] is False and verdict["layout"] == "clef"
        assert (
            "Qwen3.5" in verdict["reason"]
            and arch in verdict["reason"]
            and "PyTorch" in verdict["reason"]
        )
    # Adapters only: judged by the base they sit on.
    base = _clef_folder(tmp_path / "base", ("Qwen3ForCausalLM",))
    adapters = _clef_folder(tmp_path / "adapters", config = {"base_model": str(base)})
    (adapters / "config.json").unlink()
    (adapters / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base)})
    )
    assert gguf_eligibility(adapters)["eligible"] is False
    assert gguf_eligibility(_laya_folder(tmp_path / "laya")) == {
        "eligible": True,
        "layout": "laya",
        "reason": None,
    }
    assert gguf_eligibility(_laya_folder(tmp_path / "laya_bert", "bert"))["eligible"] is False
    assert gguf_eligibility(tmp_path)["eligible"] is False


def test_save_refuses_a_non_qwen3_5_clef_before_merging(tmp_path, monkeypatch):
    model = types.SimpleNamespace(
        is_clef = True,
        _backbone = lambda: types.SimpleNamespace(
            config = types.SimpleNamespace(architectures = ["LlamaForCausalLM"])
        ),
    )
    model.save_pretrained_merged = lambda *a, **k: pytest.fail("merged before refusing")
    monkeypatch.setattr(
        decision_gguf, "_converter_dir", lambda *a: pytest.fail("looked for llama.cpp")
    )
    with pytest.raises(ValueError, match = "Qwen3.5"):
        decision_gguf.save_pretrained_gguf(model, tmp_path, quantization_method = "q8_0")
    with pytest.raises(ValueError, match = "q3_k_m"):
        decision_gguf.save_pretrained_gguf(model, tmp_path, quantization_method = ["q8_0", "q3_k_m"])
    assert list(tmp_path.iterdir()) == []


def test_converter_without_decision_support_stages_the_pinned_tag(tmp_path, monkeypatch):
    old = tmp_path / "llama.cpp"
    old.mkdir()
    (old / "convert_hf_to_gguf.py").write_text("")
    good = tmp_path / "staged"
    (good / "conversion").mkdir(parents = True)
    (good / "gguf-py" / "gguf").mkdir(parents = True)
    (good / "convert_hf_to_gguf.py").write_text("")
    (good / "conversion" / "clef.py").write_text("")
    (good / "gguf-py" / "gguf" / "constants.py").write_text(
        'CLEF = "clef"\nTEMPERATURE = "{arch}.decision.temperature.{name}"\n'
    )
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", raising = False)
    monkeypatch.setattr(decision_gguf, "_llama_cpp_folder", lambda: old)
    import unsloth_zoo.llama_cpp as zoo

    asked = []
    monkeypatch.setattr(
        zoo, "_stage_converter_sources", lambda tag, *a, **k: asked.append(tag) or good
    )
    assert decision_gguf._converter_dir() == good and asked == ["b11443"]
    monkeypatch.setattr(zoo, "_stage_converter_sources", lambda tag, *a, **k: None)
    with pytest.raises(RuntimeError, match = "b11443"):
        decision_gguf._converter_dir()
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(good))
    assert decision_gguf._converter_dir() == good


@pytest.fixture
def fake_llama_cpp(tmp_path, monkeypatch):
    calls = []

    def convert(converter, folder, outtype, outfile, mmproj, print_output):
        calls.append(("convert", outtype, outfile.name, mmproj))
        if mmproj:
            outfile.write_bytes(b"mmproj " + outtype.encode())
        else:
            # The converter's own temperatures and head length (Laya writes raw, unclamped ones).
            _write_gguf(outfile, {"choice": 0.1006, "choice.11": 0.1006}, max_head_tokens = 96)

    def quantize(quantizer, source, target, method, print_output):
        calls.append(("quantize", source.name, target.name))
        target.write_bytes(source.read_bytes())

    monkeypatch.setattr(decision_gguf, "_converter_dir", lambda *a: tmp_path)
    monkeypatch.setattr(decision_gguf, "_quantizer", lambda *a: "llama-quantize")
    monkeypatch.setattr(
        decision_gguf, "_kquant_quantizer", lambda *a: "llama-quantize", raising = False
    )
    monkeypatch.setattr(decision_gguf, "_convert", convert)
    monkeypatch.setattr(decision_gguf, "_quantize", quantize)
    return calls


@needs_gguf
def test_export_writes_models_mmproj_temperatures_and_export_json_last(tmp_path, fake_llama_cpp):
    folder = _clef_folder(tmp_path / "run")
    data = decision_gguf.export_decision_gguf(folder, ["q8_0", "q4_k_m"])
    out = folder / "gguf"
    assert data == contract.read_export(folder)
    assert data["layout"] == "clef" and data["quantizations"] == ["Q8_0", "Q4_K_M"]
    assert data["files"] == {
        "Q8_0": {"model": "model-Q8_0.gguf", "mmproj": "mmproj-Q8_0.gguf"},
        "Q4_K_M": {"model": "model-Q4_K_M.gguf", "mmproj": "mmproj-Q8_0.gguf"},
    }
    assert data["source_fingerprint"] == contract.fingerprint(folder, "clef")
    # What native serving picks up from the run folder.
    assert contract.served_files(folder, "clef") == (
        "Q8_0",
        out / "model-Q8_0.gguf",
        out / "mmproj-Q8_0.gguf",
    )
    assert sorted(p.name for p in out.iterdir()) == [
        "export.json",
        "mmproj-Q8_0.gguf",
        "model-Q4_K_M.gguf",
        "model-Q8_0.gguf",
    ]
    for name in ("model-Q8_0.gguf", "model-Q4_K_M.gguf"):
        assert read_decision_temperatures(out / name) == pytest.approx(
            {"choice": 0.05, "score": 0.025, "noul": 0.05}
        )
    # k-quants come from a bf16 intermediate, which is not kept.
    assert ("quantize", "model-BF16.gguf", "model-Q4_K_M.gguf") in fake_llama_cpp
    assert not any(p.name.startswith(".unsloth-gguf-") for p in folder.iterdir())

    # Same weights: a later export adds to the earlier one.
    data = decision_gguf.export_decision_gguf(folder, "f16")
    assert data["quantizations"] == ["F16", "Q8_0", "Q4_K_M"]
    # New weights: the old files go.
    (folder / "joint_head.safetensors").write_bytes(b"retrained")
    assert contract.served_files(folder, "clef") is None  # stale until re-exported
    data = decision_gguf.export_decision_gguf(folder, "q8_0")
    assert data["quantizations"] == ["Q8_0"]
    assert sorted(p.name for p in out.iterdir()) == [
        "export.json",
        "mmproj-Q8_0.gguf",
        "model-Q8_0.gguf",
    ]


@needs_gguf
def test_laya_export_overwrites_the_converters_raw_temperatures(tmp_path, fake_llama_cpp):
    folder = _laya_folder(tmp_path / "laya")
    config = {"temperature": [1.2, 0.3, 1.0], "temperature_by_options": {"choice:11+": 0.1006}}
    (folder / "rl_agent_config.json").write_text(json.dumps(config))
    data = decision_gguf.export_decision_gguf(folder, "f16")
    assert data["files"] == {"F16": {"model": "model-F16.gguf", "mmproj": None}}
    assert read_decision_temperatures(folder / "gguf" / "model-F16.gguf") == pytest.approx(
        {"choice": 1.2, "score": 0.5, "noul": 1.0, "choice.11": 0.5}
    )


def test_export_refuses_ineligible_and_adapter_only_folders(tmp_path, fake_llama_cpp):
    with pytest.raises(ValueError, match = "Qwen3.5"):
        decision_gguf.export_decision_gguf(
            _clef_folder(tmp_path / "llama", ("LlamaForCausalLM",)), "q8_0"
        )
    base = _clef_folder(tmp_path / "base")
    adapters = _clef_folder(tmp_path / "adapters", config = {"base_model": str(base)})
    (adapters / "config.json").unlink()
    (adapters / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match = "save_pretrained_gguf"):
        decision_gguf.export_decision_gguf(adapters, "q8_0")
    assert fake_llama_cpp == []


def _staged(folder: Path) -> list:
    return sorted(
        p.name
        for p in folder.iterdir()
        if p.name.startswith((".unsloth-merged-", ".unsloth-gguf-"))
    )


def _head_tokens(path: Path) -> int:
    return int(gguf.GGUFReader(str(path), "r").fields["clef.decision.max_head_tokens"].contents())


@needs_gguf
@pytest.mark.parametrize(
    "positions, head_max_len, served",
    [(8192, 96, 256), (300, 96, 150), (None, 400, 400)],
)
def test_laya_export_writes_the_head_length_pytorch_serves(
    tmp_path, fake_llama_cpp, positions, head_max_len, served
):
    folder = _laya_folder(tmp_path / "laya")
    encoder = {"model_type": "modernbert"}
    if positions is not None:
        encoder["max_position_embeddings"] = positions
    (folder / "encoder" / "config.json").write_text(json.dumps(encoder))
    config = {"temperature": [1.0, 1.0, 1.0], "max_len": 256, "head_max_len": head_max_len}
    (folder / "rl_agent_config.json").write_text(json.dumps(config))
    decision_gguf.export_decision_gguf(folder, ["f16", "q4_k_m"])
    for name in ("model-F16.gguf", "model-Q4_K_M.gguf"):
        assert _head_tokens(folder / "gguf" / name) == served


@needs_gguf
def test_clef_export_leaves_the_head_length_alone(tmp_path, fake_llama_cpp):
    folder = _clef_folder(tmp_path / "run")
    decision_gguf.export_decision_gguf(folder, "q8_0")
    assert _head_tokens(folder / "gguf" / "model-Q8_0.gguf") == 96


@needs_gguf
def test_metadata_writer_sets_the_head_length_idempotently(tmp_path):
    path = tmp_path / "model.gguf"
    _write_gguf(path, {"choice": 1.0}, max_head_tokens = 96)
    assert write_decision_temperatures(path, {"choice": 1.0}, max_head_tokens = 256) is True
    assert decision_gguf.read_decision_max_head_tokens(path) == 256
    assert read_decision_temperatures(path) == {"choice": 1.0}
    assert write_decision_temperatures(path, {"choice": 1.0}, max_head_tokens = 256) is False
    with pytest.raises(ValueError):
        write_decision_temperatures(path, {"choice": 1.0}, max_head_tokens = 0)


class _FakeClef:
    is_clef = True

    def __init__(
        self,
        head: bytes,
        fail = None,
    ):
        self.head, self.fail = head, fail

    def _backbone(self):
        return types.SimpleNamespace(
            config = types.SimpleNamespace(architectures = ["Qwen3_5ForCausalLM"])
        )

    def save_pretrained_merged(
        self,
        folder,
        tokenizer = None,
        **kwargs,
    ):
        self.merge_kwargs = kwargs
        _clef_folder(Path(folder), ("Qwen3_5ForCausalLM",))
        (Path(folder) / "joint_head.safetensors").write_bytes(self.head)
        if self.fail is not None:
            raise self.fail


@needs_gguf
def test_save_adopts_the_folder_only_when_it_holds_the_saved_weights(tmp_path, fake_llama_cpp):
    folder = _clef_folder(tmp_path / "run", ("Qwen3_5ForCausalLM",))
    data = decision_gguf.save_pretrained_gguf(_FakeClef(b"head-weights"), folder)
    assert data["source_fingerprint"] == contract.fingerprint(folder, "clef")
    assert contract.served_files(folder, "clef") is not None

    # Calibrated or trained in memory since the folder was saved: the folder is stale.
    data = decision_gguf.save_pretrained_gguf(_FakeClef(b"recalibrated"), folder)
    assert data["source_fingerprint"] != contract.fingerprint(folder, "clef")
    assert contract.served_files(folder, "clef") is None
    assert _staged(folder) == []


def _old_llama_cpp(tmp_path: Path) -> Path:
    old = tmp_path / "old_llama.cpp"
    old.mkdir()
    (old / "convert_hf_to_gguf.py").write_text("")
    return old


def _new_llama_cpp(tmp_path: Path) -> Path:
    good = tmp_path / "new_llama.cpp"
    (good / "conversion").mkdir(parents = True)
    (good / "gguf-py" / "gguf").mkdir(parents = True)
    (good / "convert_hf_to_gguf.py").write_text("")
    (good / "conversion" / "clef.py").write_text("")
    (good / "gguf-py" / "gguf" / "constants.py").write_text(
        'CLEF = "clef"\nTEMPERATURE = "{arch}.decision.temperature.{name}"\n'
    )
    return good


def test_kquants_are_refused_before_the_merge_when_llama_cpp_predates_decision(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(decision_gguf, "_converter_dir", lambda *a: tmp_path)
    monkeypatch.setattr(decision_gguf, "_quantizer", lambda *a: "llama-quantize")
    old = _old_llama_cpp(tmp_path)
    monkeypatch.setattr(decision_gguf, "_llama_cpp_folder", lambda: old)
    model = _FakeClef(b"h")
    model.save_pretrained_merged = lambda *a, **k: pytest.fail("merged before refusing")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match = "predates llama.cpp b11443"):
        decision_gguf.save_pretrained_gguf(model, out, quantization_method = ["q8_0", "q4_k_m"])
    monkeypatch.setattr(
        decision_gguf, "_convert", lambda *a: pytest.fail("converted before refusing")
    )
    with pytest.raises(RuntimeError, match = "predates llama.cpp b11443"):
        decision_gguf.export_decision_gguf(_clef_folder(tmp_path / "run"), "q4_k_m")


def _fake_quantizer(tmp_path: Path, output: str) -> str:
    script = tmp_path / "llama-quantize"
    script.write_text(f"#!{sys.executable}\nprint({output!r}, flush = True)\nraise SystemExit(1)\n")
    script.chmod(0o755)
    return str(script)


@pytest.mark.skipif(os.name == "nt", reason = "shebang script")
@pytest.mark.parametrize("print_output", [False, True])
def test_quantize_explains_an_old_quantizer_with_or_without_print_output(
    tmp_path, monkeypatch, capsys, print_output
):
    new = _new_llama_cpp(tmp_path)
    monkeypatch.setattr(decision_gguf, "_llama_cpp_folder", lambda: new)
    old = _fake_quantizer(tmp_path, "llama_model_load: error: unknown model architecture: 'clef'")
    with pytest.raises(RuntimeError, match = "predates llama.cpp b11443"):
        decision_gguf._quantize(old, tmp_path / "a", tmp_path / "b", "q4_k_m", print_output)
    assert ("unknown model architecture" in capsys.readouterr().out) == print_output
    other = _fake_quantizer(tmp_path, "out of disk space")
    with pytest.raises(RuntimeError, match = "out of disk space") as error:
        decision_gguf._quantize(other, tmp_path / "a", tmp_path / "b", "q4_k_m", print_output)
    assert "predates" not in str(error.value)


@needs_gguf
def test_an_interrupted_export_leaves_no_temp_folders(tmp_path, fake_llama_cpp, monkeypatch):
    folder = _clef_folder(tmp_path / "run", ("Qwen3_5ForCausalLM",))

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(decision_gguf, "_convert", interrupted)
    with pytest.raises(KeyboardInterrupt):
        decision_gguf.export_decision_gguf(folder, "q8_0")
    assert _staged(folder) == []
    with pytest.raises(KeyboardInterrupt):
        decision_gguf.save_pretrained_gguf(_FakeClef(b"h"), folder)
    with pytest.raises(KeyboardInterrupt):
        decision_gguf.save_pretrained_gguf(_FakeClef(b"h", KeyboardInterrupt()), folder)
    assert _staged(folder) == [] and _staged(folder.parent) == []


@needs_gguf
def test_save_merges_with_the_callers_token(tmp_path, fake_llama_cpp):
    model = _FakeClef(b"h")
    decision_gguf.save_pretrained_gguf(model, tmp_path / "a", token = "hf_private")
    assert model.merge_kwargs == {"token": "hf_private"}
    decision_gguf.save_pretrained_gguf(model, tmp_path / "b")
    assert model.merge_kwargs == {}


@needs_gguf
def test_a_failed_publish_keeps_the_export_of_the_same_weights(
    tmp_path, fake_llama_cpp, monkeypatch
):
    folder = _clef_folder(tmp_path / "run")
    decision_gguf.export_decision_gguf(folder, "q8_0")
    served = contract.served_files(folder, "clef")
    replace = os.replace

    def mapped(source, target):
        # Windows: llama-server has model-Q8_0.gguf mapped, so it cannot be replaced.
        if Path(target) == folder / "gguf" / "model-Q8_0.gguf":
            raise PermissionError(13, "The process cannot access the file", str(target))
        return replace(source, target)

    monkeypatch.setattr(decision_gguf.os, "replace", mapped)
    with pytest.raises(PermissionError):
        decision_gguf.export_decision_gguf(folder, ["q8_0", "f16"])
    monkeypatch.setattr(decision_gguf.os, "replace", replace)
    assert contract.served_files(folder, "clef") == served
    assert contract.read_export(folder)["quantizations"] == ["Q8_0"]

    # Other weights: nothing of the old export is current, so nothing is restored.
    (folder / "joint_head.safetensors").write_bytes(b"retrained")
    monkeypatch.setattr(decision_gguf.os, "replace", mapped)
    with pytest.raises(PermissionError):
        decision_gguf.export_decision_gguf(folder, "q8_0")
    assert contract.read_export(folder) is None


@needs_gguf
@pytest.mark.skipif(
    not hasattr(signal, "SIGKILL") or signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL,
    reason = "POSIX signals with the default SIGTERM handler",
)
def test_sigterm_cleans_up_and_a_killed_export_is_swept_later(
    tmp_path, fake_llama_cpp, monkeypatch
):
    folder = _clef_folder(tmp_path / "run", ("Qwen3_5ForCausalLM",))
    convert = decision_gguf._convert

    def terminated(*args):
        decision_gguf._run(
            [
                sys.executable,
                "-c",
                f"import os, signal, time; os.kill({os.getpid()}, signal.SIGTERM); time.sleep(60)",
            ],
            "converting",
            False,
        )

    # Studio cancels an export by terminating its worker process.
    monkeypatch.setattr(decision_gguf, "_convert", terminated)
    with pytest.raises(SystemExit):
        decision_gguf.save_pretrained_gguf(_FakeClef(b"h"), folder)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    assert _staged(folder) == []

    # SIGKILL runs no cleanup: the next export removes what a dead process left.
    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()
    for prefix in decision_gguf._TEMP_PREFIXES:
        (folder / f"{prefix}{dead.pid}-x").mkdir()
        (folder / f"{prefix}{os.getppid()}-alive").mkdir()
        (folder / f"{prefix}legacy").mkdir()
    monkeypatch.setattr(decision_gguf, "_convert", convert)
    decision_gguf.save_pretrained_gguf(_FakeClef(b"h"), folder)
    assert _staged(folder) == sorted(
        f"{prefix}{name}"
        for prefix in decision_gguf._TEMP_PREFIXES
        for name in (f"{os.getppid()}-alive", "legacy")
    )
