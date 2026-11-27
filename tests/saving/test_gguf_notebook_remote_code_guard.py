# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Notebook GGUF export keeps convert_hf_to_gguf.py to the load's trust_remote_code decision.

The converter loads several architectures' tokenizers with trust_remote_code=True. Without
consent at load time, the converter child gets an HF_MODULES_CACHE no directory can be created
at, so a checkpoint's auto_map code cannot be imported there (Studio's export does the same).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types

import pytest

_KEY = "HF_MODULES_CACHE"


@pytest.fixture
def save_mod():
    import unsloth.save as save
    return save


class _FakeTokenizer:
    chat_template = None

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok = True)


class _FakeModel:
    def __init__(self, name_or_path):
        self.config = types.SimpleNamespace(
            _name_or_path = name_or_path,
            architectures = ["LlamaForCausalLM"],
            model_type = "llama",
        )

    def save_pretrained(self, path, *args, **kwargs):
        os.makedirs(path, exist_ok = True)


def _cache_seen_by_converter(monkeypatch, save_mod, tmp_path, model, tokenizer):
    """Drive `save_pretrained_gguf` with the converter stubbed; return HF_MODULES_CACHE as the converter sees it."""
    seen = {}
    monkeypatch.setattr(save_mod, "_is_vlm", lambda _m: False)
    monkeypatch.setattr(save_mod, "_is_gpt_oss", lambda _m: False)
    monkeypatch.setattr(save_mod, "fix_tokenizer_bos_token", lambda _t: (False, None))
    monkeypatch.setattr(save_mod, "_resolve_imatrix_file", lambda *_a, **_k: None)
    monkeypatch.setattr(save_mod, "dtype_from_config", lambda _c: save_mod.torch.float16)
    monkeypatch.setattr(save_mod, "create_ollama_modelfile", lambda *_a, **_k: None)

    def _save_to_gguf(**kw):
        seen["cache"] = os.environ.get(_KEY)
        out = tmp_path / "export" / "model_gguf" / "model.Q8_0.gguf"
        out.parent.mkdir(parents = True, exist_ok = True)
        out.write_bytes(b"GGUF")
        return [str(out)], True, False

    monkeypatch.setattr(save_mod, "save_to_gguf", _save_to_gguf)
    save_mod.unsloth_save_pretrained_gguf(
        model,
        str(tmp_path / "export" / "model"),
        tokenizer = tokenizer,
        quantization_method = "q8_0",
    )
    return seen["cache"]


def _is_uncreatable(path):
    try:
        os.makedirs(path, exist_ok = True)
    except OSError:
        return True
    return False


def test_untrusted_load_blocks_converter_remote_code(monkeypatch, tmp_path, save_mod):
    preset = str(tmp_path / "modules")
    monkeypatch.setenv(_KEY, preset)
    model = _FakeModel("some-org/some-model")
    cache = _cache_seen_by_converter(monkeypatch, save_mod, tmp_path, model, _FakeTokenizer())
    assert cache != preset
    assert _is_uncreatable(cache)
    assert os.environ[_KEY] == preset  # restored after the export


def test_untrusted_load_restores_unset_cache(monkeypatch, tmp_path, save_mod):
    monkeypatch.delenv(_KEY, raising = False)
    model = _FakeModel("some-org/some-model")
    cache = _cache_seen_by_converter(monkeypatch, save_mod, tmp_path, model, _FakeTokenizer())
    assert cache is not None and _is_uncreatable(cache)
    assert _KEY not in os.environ


def test_trusted_load_keeps_converter_unchanged(monkeypatch, tmp_path, save_mod):
    preset = str(tmp_path / "modules")
    monkeypatch.setenv(_KEY, preset)
    model = _FakeModel("some-org/some-model")
    model._unsloth_trust_remote_code = True
    cache = _cache_seen_by_converter(monkeypatch, save_mod, tmp_path, model, _FakeTokenizer())
    assert cache == preset


def test_remote_code_tokenizer_counts_as_consent(monkeypatch, tmp_path, save_mod):
    preset = str(tmp_path / "modules")
    monkeypatch.setenv(_KEY, preset)
    remote_tok = type(
        "CustomTokenizer", (_FakeTokenizer,), {"__module__": "transformers_modules.x.tok"}
    )
    model = _FakeModel("some-org/some-model")
    cache = _cache_seen_by_converter(monkeypatch, save_mod, tmp_path, model, remote_tok())
    assert cache == preset


def _write_remote_code_checkpoint(path, marker):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "custom_probe",
                "auto_map": {"AutoConfig": "configuration_probe.ProbeConfig"},
            }
        )
    )
    (path / "configuration_probe.py").write_text(
        textwrap.dedent(
            f"""
            open({str(marker)!r}, "w").write("ran")
            from transformers import PretrainedConfig

            class ProbeConfig(PretrainedConfig):
                model_type = "custom_probe"
            """
        )
    )


def _child_loads_with_trust(checkpoint, env):
    code = (
        "import sys\n"
        "from transformers import AutoConfig\n"
        "AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code, str(checkpoint)],
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )


def test_guarded_env_stops_a_child_from_running_auto_map_code(tmp_path, save_mod):
    """The mechanism itself: a converter-like child that asks for trust_remote_code=True runs the
    checkpoint's auto_map code with a normal cache, and cannot under the guard."""
    marker = tmp_path / "ran.txt"
    checkpoint = tmp_path / "ckpt"
    _write_remote_code_checkpoint(checkpoint, marker)
    base_env = dict(os.environ, HF_HUB_OFFLINE = "1", HF_MODULES_CACHE = str(tmp_path / "mods"))

    control = _child_loads_with_trust(checkpoint, base_env)
    assert control.returncode == 0, control.stderr
    assert marker.exists(), "control arm must run the remote code, else this test proves nothing"
    marker.unlink()

    with save_mod._gguf_converter_remote_code(False):
        guarded_env = dict(base_env, HF_MODULES_CACHE = os.environ[_KEY])
    guarded = _child_loads_with_trust(checkpoint, guarded_env)
    assert guarded.returncode != 0
    assert not marker.exists()
