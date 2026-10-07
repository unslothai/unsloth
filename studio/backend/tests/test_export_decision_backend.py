# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decision checkpoints (Clef, Laya) in the export worker: detected before the chat / vision
loaders, GGUF only, through unsloth.models.decision_gguf (stubbed here)."""

from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _path in (_BACKEND_DIR, _TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from test_export_absolute_paths import (  # noqa: E402
    _install_export_backend_stubs,
    _load_module,
)

_QWEN35 = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}


def _write(
    folder: Path,
    files: dict,
    dirs = (),
) -> Path:
    folder.mkdir(parents = True, exist_ok = True)
    for name, content in files.items():
        data = content if isinstance(content, (bytes, str)) else json.dumps(content)
        (folder / name).write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    for name in dirs:
        (folder / name).mkdir(parents = True, exist_ok = True)
    return folder


def _clef_merged(tmp_path, config = _QWEN35) -> Path:
    return _write(
        tmp_path / "clef_run",
        {
            "config.json": config,
            "model.safetensors": b"w",
            "joint_head.safetensors": b"h",
            "joint_head_config.json": {},
            "unsloth_decision_config.json": {},
        },
    )


def _clef_adapter(tmp_path) -> Path:
    return _write(
        tmp_path / "clef_adapter",
        {
            "adapter_config.json": {"base_model_name_or_path": "unsloth/Qwen3.5-0.8B"},
            "adapter_model.safetensors": b"a",
            "joint_head.safetensors": b"h",
            "joint_head_config.json": {},
        },
    )


def _laya(tmp_path) -> Path:
    folder = _write(
        tmp_path / "laya_run",
        {"rl_agent_config.json": {}, "model.safetensors": b"w"},
        dirs = ("encoder", "tokenizer"),
    )
    (folder / "encoder" / "config.json").write_text(
        json.dumps({"model_type": "modernbert"}), encoding = "utf-8"
    )
    return folder


class _Recorder:
    def __init__(self):
        self.calls = []
        self.tokens = []


class _ForbiddenLoader:
    calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        raise AssertionError("chat / vision loader reached for a decision checkpoint")


@pytest.fixture
def env(monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    lib = _Recorder()
    lib.eligibility = None

    def gguf_eligibility(folder):
        lib.calls.append(("eligibility", folder))
        if lib.eligibility is not None:
            return lib.eligibility
        layout = "clef" if (Path(folder) / "joint_head_config.json").is_file() else "laya"
        return {"eligible": True, "layout": layout, "reason": None}

    def export_decision_gguf(
        checkpoint_folder,
        quantization_method,
        output_dir = None,
        source_folder = None,
        print_output = False,
    ):
        lib.calls.append(
            ("export", checkpoint_folder, list(quantization_method), output_dir, source_folder)
        )
        return {"quantizations": [q.upper() for q in quantization_method]}

    decision_gguf = types.ModuleType("unsloth.models.decision_gguf")
    decision_gguf.gguf_eligibility = gguf_eligibility
    decision_gguf.export_decision_gguf = export_decision_gguf
    models_pkg = types.ModuleType("unsloth.models")
    models_pkg.__path__ = []
    models_pkg.decision_gguf = decision_gguf
    monkeypatch.setitem(sys.modules, "unsloth.models", models_pkg)
    monkeypatch.setitem(sys.modules, "unsloth.models.decision_gguf", decision_gguf)

    class _DecisionModel:
        def save_pretrained_gguf(
            self,
            save_directory,
            tokenizer,
            quantization_method,
            source_folder = None,
            print_output = False,
            token = None,
        ):
            lib.tokens.append(token)
            lib.calls.append(
                (
                    "save_pretrained_gguf",
                    save_directory,
                    tokenizer,
                    list(quantization_method),
                    source_folder,
                )
            )
            return {"quantizations": [q.upper() for q in quantization_method]}

    class FastDecisionModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            lib.calls.append(("FastDecisionModel", model_name, kwargs))
            return _DecisionModel(), "processor"

    sys.modules["unsloth"].FastDecisionModel = FastDecisionModel

    mod = _load_module("test_core_export_backend_decision", "core/export/export.py", monkeypatch)
    _ForbiddenLoader.calls = []
    monkeypatch.setattr(mod, "FastLanguageModel", _ForbiddenLoader)
    monkeypatch.setattr(mod, "FastVisionModel", _ForbiddenLoader)
    probes = []
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: probes.append("audio"))
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: probes.append("vision"))
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod.ExportBackend, "cleanup_memory", lambda self: True)
    monkeypatch.setattr(mod.ExportBackend, "__init__", _bare_init)
    lib.probes = probes
    return mod, lib


def _bare_init(self):
    self.current_checkpoint = None
    self.current_model = None
    self.current_tokenizer = None
    self.is_vision = False
    self.is_peft = False
    self._audio_type = None
    self.decision = None


def _exports(lib):
    return [c for c in lib.calls if c[0] in ("export", "save_pretrained_gguf", "FastDecisionModel")]


def test_merged_clef_skips_chat_and_vision_loaders_and_exports_in_run_folder(env, tmp_path):
    mod, lib = env
    folder = _clef_merged(tmp_path)
    backend = mod.ExportBackend()

    ok, message = backend.load_checkpoint(str(folder))
    assert ok, message
    assert _ForbiddenLoader.calls == [] and lib.probes == []
    assert backend.decision == {"layout": "clef", "adapter_only": False}
    assert backend.is_peft is False and backend.is_vision is False

    ok, message, output = backend.export_gguf("ignored_dir", ["Q8_0", "q4_k_m", "q8_0"])
    assert ok, message
    assert _exports(lib) == [("export", str(folder), ["q8_0", "q4_k_m"], None, str(folder))]
    assert output == str(folder.resolve() / "gguf")
    assert "Q8_0, Q4_K_M" in message
    assert not (Path.cwd() / "ignored_dir").exists()


@pytest.mark.parametrize("token", ["hf_caller", False])
def test_adapter_only_clef_loads_and_merges_its_base_with_the_load_credential(env, tmp_path, token):
    mod, lib = env
    folder = _clef_adapter(tmp_path)
    backend = mod.ExportBackend()
    assert backend.load_checkpoint(str(folder), hf_token = token)[0]
    assert backend.export_gguf("x", "q8_0")[0]
    assert _exports(lib)[0][2]["token"] == token
    assert lib.tokens == [token]


def test_adapter_only_clef_merges_through_fast_decision_model(env, tmp_path):
    mod, lib = env
    folder = _clef_adapter(tmp_path)
    backend = mod.ExportBackend()

    ok, message = backend.load_checkpoint(str(folder))
    assert ok, message
    assert backend.decision == {"layout": "clef", "adapter_only": True}
    assert backend.is_peft is True

    ok, message, output = backend.export_gguf("x", "f16")
    assert ok, message
    calls = _exports(lib)
    assert calls[0][0] == "FastDecisionModel" and calls[0][1] == str(folder)
    assert calls[0][2]["load_in_4bit"] is False
    assert calls[1] == ("save_pretrained_gguf", str(folder), "processor", ["f16"], str(folder))
    assert not any(c[0] == "export" for c in calls)
    assert _ForbiddenLoader.calls == []
    assert output == str(folder.resolve() / "gguf")


def test_laya_exports_through_the_library(env, tmp_path):
    mod, lib = env
    folder = _laya(tmp_path)
    backend = mod.ExportBackend()

    assert backend.load_checkpoint(str(folder))[0]
    assert backend.decision == {"layout": "laya", "adapter_only": False}
    ok, _, _ = backend.export_gguf("x", ["bf16"])
    assert ok
    assert _exports(lib) == [("export", str(folder), ["bf16"], None, str(folder))]
    assert _ForbiddenLoader.calls == [] and lib.probes == []


def test_a_home_relative_checkpoint_path_is_expanded(env, tmp_path, monkeypatch):
    mod, lib = env
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    folder = _laya(tmp_path)
    backend = mod.ExportBackend()

    assert backend.load_checkpoint("~/laya_run")[0]
    assert backend.export_gguf("x", "q8_0")[0]
    assert ("eligibility", "~/laya_run") not in lib.calls
    assert [c for c in lib.calls if c[0] == "eligibility"] == [("eligibility", str(folder))] * 2
    assert _exports(lib) == [("export", str(folder), ["q8_0"], None, str(folder))]


def test_ineligible_clef_fails_the_load_with_the_reason(env, tmp_path):
    mod, lib = env
    folder = _clef_merged(tmp_path, config = {"architectures": ["LlamaForCausalLM"]})
    reason = "llama.cpp's decision graph is built for Qwen3.5 backbones only"
    lib.eligibility = {"eligible": False, "layout": "clef", "reason": reason}
    backend = mod.ExportBackend()

    ok, message = backend.load_checkpoint(str(folder))
    assert ok is False
    assert message == reason
    assert backend.decision is None
    assert _exports(lib) == [] and _ForbiddenLoader.calls == []


def test_ineligible_at_export_time_is_an_error_not_a_crash(env, tmp_path):
    mod, lib = env
    folder = _clef_merged(tmp_path)
    backend = mod.ExportBackend()
    assert backend.load_checkpoint(str(folder))[0]
    lib.eligibility = {"eligible": False, "layout": "clef", "reason": "not Qwen3.5"}

    assert backend.export_gguf("x", "q8_0") == (False, "not Qwen3.5", None)


def test_unknown_quant_is_refused_before_the_library(env, tmp_path):
    mod, lib = env
    folder = _clef_merged(tmp_path)
    backend = mod.ExportBackend()
    assert backend.load_checkpoint(str(folder))[0]

    ok, message, output = backend.export_gguf("x", ["q2_k_l"])
    assert ok is False and output is None
    assert "Q2_K_L" in message
    assert _exports(lib) == []


@pytest.mark.parametrize(
    "kwargs",
    [{"push_to_hub": True, "repo_id": "u/r"}, {"imatrix_file": True}, {"npu_q4nx": True}],
    ids = ["hub", "imatrix", "q4nx"],
)
def test_unsupported_gguf_options_are_refused(env, tmp_path, kwargs):
    mod, lib = env
    backend = mod.ExportBackend()
    assert backend.load_checkpoint(str(_clef_merged(tmp_path)))[0]

    ok, _, _ = backend.export_gguf("x", "q8_0", **kwargs)
    assert ok is False
    assert _exports(lib) == []


def test_other_export_methods_are_gguf_only_for_decision_models(env, tmp_path):
    mod, _ = env
    backend = mod.ExportBackend()
    assert backend.load_checkpoint(str(_clef_adapter(tmp_path)))[0]

    expected = (False, "Decision models export to GGUF only.", None)
    assert backend.export_merged_model("x") == expected
    assert backend.export_base_model("x") == expected
    assert backend.export_lora_adapter("x") == expected


def test_non_decision_checkpoint_still_uses_the_text_loader(env, tmp_path, monkeypatch):
    mod, lib = env
    folder = _write(tmp_path / "llama_run", {"config.json": {"model_type": "llama"}})
    loaded = []

    class _TextLoader:
        @staticmethod
        def from_pretrained(**kwargs):
            loaded.append(kwargs["model_name"])
            return object(), object()

    monkeypatch.setattr(mod, "FastLanguageModel", _TextLoader)
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: {})
    monkeypatch.setattr(mod, "restore_hf_cache_repo_identity", lambda *a, **k: None)
    backend = mod.ExportBackend()

    ok, message = backend.load_checkpoint(str(folder))
    assert ok, message
    assert loaded == [str(folder)]
    assert backend.decision is None
    assert lib.probes == ["audio", "vision"]
    assert not any(c[0] == "eligibility" for c in lib.calls)


def test_studio_quant_list_matches_the_library():
    from core.export.decision import DECISION_GGUF_QUANTIZATIONS

    source = _BACKEND_DIR.parent.parent / "unsloth" / "models" / "decision_gguf.py"
    if not source.is_file():
        pytest.skip("unsloth/models/decision_gguf.py is not in this checkout")
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "DECISION_GGUF_QUANTIZATIONS" for t in node.targets)
    ]
    assert values == [DECISION_GGUF_QUANTIZATIONS]


def test_orchestrator_keeps_the_decision_block_from_the_worker(monkeypatch, tmp_path):
    from core.export.orchestrator import ExportOrchestrator

    backend = ExportOrchestrator()
    monkeypatch.setattr(backend, "_ensure_subprocess_alive", lambda: False)
    monkeypatch.setattr(backend, "_spawn_subprocess", lambda config: None)
    decision = {"layout": "clef", "adapter_only": False}
    monkeypatch.setattr(
        backend,
        "_wait_response",
        lambda *a, **k: {
            "success": True,
            "message": "ok",
            "checkpoint": str(tmp_path),
            "decision": decision,
        },
    )
    assert backend.load_checkpoint(str(tmp_path), load_in_4bit = False)[0]
    assert backend.decision == decision

    monkeypatch.setattr(
        backend, "_wait_response", lambda *a, **k: {"success": False, "message": "no"}
    )
    assert backend.load_checkpoint(str(tmp_path), load_in_4bit = False) == (False, "no")
    assert backend.decision is None
