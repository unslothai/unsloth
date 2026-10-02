# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF export can also write a FastFlowLM Q4NX folder for the AMD Ryzen AI NPU.

Reuses the harness in test_export_gguf_discovery.py; the converter itself is stubbed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_export_gguf_discovery import _backend, _gguf  # noqa: E402


class _Tokenizer:
    def save_pretrained(self, path):
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "special_tokens_map.json",
        ):
            (Path(path) / name).write_text(f"hf {name}")


class _Model:
    def __init__(self):
        self.calls = 0

    def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
        self.calls += 1
        out = Path(model_save_path)
        out.mkdir(parents = True)
        (out / "config.json").write_text('{"model_type": "qwen3"}')
        quants = (
            quantization_method if isinstance(quantization_method, list) else [quantization_method]
        )
        return {"gguf_files": [str(_gguf(out / f"Model.{q.upper()}.gguf")) for q in quants]}


def _fake_converter(
    export_mod,
    monkeypatch,
    calls,
    fail = False,
):
    def convert(gguf_path, out_dir):
        if fail:
            raise RuntimeError("unsupported architecture")
        calls.append(Path(gguf_path).name)
        out_dir.mkdir(parents = True, exist_ok = True)
        (out_dir / "model.q4nx").write_bytes(b"q4nx")
        (out_dir / "tokenizer.json").write_text("from gguf")

    monkeypatch.setattr(export_mod.q4nx, "convert_gguf_to_q4nx", convert)


def test_q4nx_folder_matches_the_flm_layout(monkeypatch, tmp_path):
    model = _Model()
    export_mod, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, model)
    backend.current_tokenizer = _Tokenizer()
    calls = []
    _fake_converter(export_mod, monkeypatch, calls)

    success, message, output_path = backend.export_gguf(
        str(save_dir), ["q8_0", "q4_1", "q4_k_m"], npu_q4nx = True
    )

    assert success is True, message
    assert calls == ["Model.Q4_1.gguf"]
    q4nx = save_dir / "npu-q4nx"
    assert sorted(p.name for p in q4nx.iterdir()) == [
        "chat_template.jinja",
        "config.json",
        "model.q4nx",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert (q4nx / "tokenizer.json").read_text() == "hf tokenizer.json"
    assert (q4nx / "config.json").read_text() == '{"model_type": "qwen3"}'
    assert output_path == str(save_dir.resolve())


def test_no_q4nx_source_quant_fails_before_exporting(monkeypatch, tmp_path):
    model = _Model()
    _m, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, model)

    success, message, output_path = backend.export_gguf(str(save_dir), ["q8_0"], npu_q4nx = True)

    assert success is False
    assert "Q4_0, Q4_1 or Q4_K_M" in message
    assert model.calls == 0
    assert output_path is None


def test_a_failed_conversion_keeps_and_reports_the_ggufs(monkeypatch, tmp_path):
    export_mod, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())
    backend.current_tokenizer = _Tokenizer()
    _fake_converter(export_mod, monkeypatch, [], fail = True)

    success, message, output_path = backend.export_gguf(str(save_dir), "q4_0", npu_q4nx = True)

    assert success is False
    assert "GGUF files were saved" in message and "unsupported architecture" in message
    assert (save_dir / "Model.Q4_0.gguf").is_file()
    assert output_path == str(save_dir.resolve())


def test_without_the_flag_nothing_is_converted(monkeypatch, tmp_path):
    export_mod, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())
    calls = []
    _fake_converter(export_mod, monkeypatch, calls)

    success, message, _out = backend.export_gguf(str(save_dir), "q4_1")

    assert success is True, message
    assert calls == []
    assert not (save_dir / "npu-q4nx").exists()


def test_source_follows_the_selection_order(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    ggufs = ["/x/M.Q4_0.gguf", "/x/M.Q4_K_M.gguf", "/x/M.Q8_0.gguf"]
    assert export_mod.q4nx.source_gguf(ggufs, ["q8_0", "q4_k_m", "q4_0"]) == "/x/M.Q4_K_M.gguf"
    assert export_mod.q4nx.source_gguf(ggufs, ["q8_0"]) is None


def test_converter_runs_in_this_interpreter(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    script = tmp_path / "conv" / "convert.py"
    script.parent.mkdir()
    script.write_text("")

    class _Installer:
        @staticmethod
        def install(root):
            return script

    ran = {}

    def run(cmd, cwd, check):
        ran.update(cmd = cmd, cwd = cwd, check = check)
        (Path(cmd[-1]) / "model.q4nx").write_bytes(b"q4nx")

    monkeypatch.setattr(export_mod.q4nx, "_installer", lambda: _Installer)
    monkeypatch.setattr(export_mod.q4nx.subprocess, "run", run)
    out = tmp_path / "out"
    export_mod.q4nx.convert_gguf_to_q4nx("/x/M.Q4_1.gguf", out)

    assert ran["cmd"] == [sys.executable, str(script), "-i", "/x/M.Q4_1.gguf", "-o", str(out)]
    assert ran["cwd"] == str(script.parent) and ran["check"] is True


def _base_folder(tmp_path, files):
    base = tmp_path / "base"
    base.mkdir()
    for name, text in files.items():
        (base / name).write_text(text)
    return base


def test_existing_gguf_takes_companions_from_the_base_model(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    _fake_converter(export_mod, monkeypatch, calls := [])
    monkeypatch.setattr(export_mod.q4nx, "_gguf_chat_template", lambda _p: "{{ gguf }}")
    base = _base_folder(
        tmp_path,
        {
            "config.json": "{}",
            "tokenizer.json": "hf",
            "tokenizer_config.json": '{"eos_token": "x"}',
        },
    )
    gguf = _gguf(tmp_path / "hub" / "Qwen3-0.6B-Q4_1.gguf")

    out = export_mod.q4nx.convert_existing_gguf(gguf, str(base), tmp_path / "exports")

    assert out == tmp_path / "exports" / "Qwen3-0.6B-Q4_1-q4nx"
    assert calls == ["Qwen3-0.6B-Q4_1.gguf"]
    assert (out / "tokenizer.json").read_text() == "hf"
    # Neither the base folder nor its tokenizer_config carries a template, so the GGUF's is used.
    assert (out / "chat_template.jinja").read_text() == "{{ gguf }}"


def test_existing_gguf_keeps_a_template_already_in_tokenizer_config(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    _fake_converter(export_mod, monkeypatch, [])
    base = _base_folder(
        tmp_path, {"config.json": "{}", "tokenizer_config.json": '{"chat_template": "{{ hf }}"}'}
    )

    out = export_mod.q4nx.convert_existing_gguf(_gguf(tmp_path / "m.gguf"), str(base), tmp_path)

    assert not (out / "chat_template.jinja").exists()


def test_existing_gguf_without_tokenizer_config_fails_before_converting(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    _fake_converter(export_mod, monkeypatch, calls := [])
    base = _base_folder(tmp_path, {"config.json": "{}"})

    try:
        export_mod.q4nx.convert_existing_gguf(_gguf(tmp_path / "m.gguf"), str(base), tmp_path)
    except RuntimeError as e:
        assert "tokenizer_config.json" in str(e)
    else:
        raise AssertionError("expected a RuntimeError")
    assert calls == []
