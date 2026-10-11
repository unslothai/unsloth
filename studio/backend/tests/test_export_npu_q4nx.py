# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF export can also write a FastFlowLM Q4NX folder for the AMD Ryzen AI NPU.

Reuses the harness in test_export_gguf_discovery.py; the converter itself is stubbed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_export_gguf_discovery import _backend, _gguf  # noqa: E402


_HF_TOKENIZER = {"added_tokens": [{"id": 7, "content": "<|im_end|>"}], "model": {"vocab": {}}}


class _Tokenizer:
    def save_pretrained(self, path):
        path = Path(path)
        (path / "tokenizer.json").write_text(json.dumps(_HF_TOKENIZER))
        (path / "tokenizer_config.json").write_text(
            '{"eos_token": "<|im_end|>", "bos_token": null}'
        )
        for name in ("chat_template.jinja", "special_tokens_map.json"):
            (path / name).write_text(f"hf {name}")


class _Model:
    # Like Phi-4-mini: the chat-turn stop id is only in generation_config.
    generation_config = type("G", (), {"eos_token_id": [7, 9]})()

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
        (out_dir / "tokenizer.json").write_text('{"added_tokens": []}')

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
    # No config.json: copied over a catalog model it would drop FLM's flm_version.
    assert sorted(p.name for p in q4nx.iterdir()) == [
        "chat_template.jinja",
        "model.q4nx",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert json.loads((q4nx / "tokenizer.json").read_text()) == _HF_TOKENIZER
    tokenizer_config = json.loads((q4nx / "tokenizer_config.json").read_text())
    assert tokenizer_config["eos_token_id"] == [7, 9]
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


def _stub_installer(
    export_mod,
    monkeypatch,
    tmp_path,
    architecture = "qwen3",
):
    """Installer stub recording which pinned converter each architecture asked for."""
    real = export_mod.q4nx._installer()
    installed = []

    class _Installer:
        converter_for_architecture = staticmethod(real.converter_for_architecture)

        @staticmethod
        def install(root, name):
            installed.append(name)
            script = tmp_path / name / "convert.py"
            script.parent.mkdir(exist_ok = True)
            return script

    monkeypatch.setattr(export_mod.q4nx, "_installer", lambda: _Installer)
    monkeypatch.setattr(export_mod.q4nx, "_gguf_architecture", lambda _p: architecture)
    monkeypatch.setattr(export_mod.q4nx, "require_converter_deps", lambda: None)
    return installed


def _fake_popen(
    export_mod,
    monkeypatch,
    returncode = 0,
    stderr = "",
):
    ran = {}

    class _Popen:
        def __init__(self, cmd, cwd, **kwargs):
            ran.update(cmd = cmd, cwd = cwd, preexec_fn = kwargs.get("preexec_fn"), env = kwargs.get("env"))
            self.returncode = returncode
            if returncode == 0:
                (Path(cmd[-1]) / "model.q4nx").write_bytes(b"q4nx")

        def communicate(self):
            return None, stderr

    monkeypatch.setattr(export_mod.q4nx.subprocess, "Popen", _Popen)
    return ran


@pytest.mark.parametrize(
    "architecture, converter", [("qwen3", "q4nx"), ("llama", "q4nx"), ("qwen35", "q4k")]
)
def test_converter_runs_in_this_interpreter(monkeypatch, tmp_path, architecture, converter):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    installed = _stub_installer(export_mod, monkeypatch, tmp_path, architecture)
    ran = _fake_popen(export_mod, monkeypatch)
    out = tmp_path / "out"
    export_mod.q4nx.convert_gguf_to_q4nx("/x/M.Q4_1.gguf", out)

    assert installed == [converter]
    assert ran["cmd"][:2] == [sys.executable, "-c"]
    assert ran["cmd"][3:] == ["/x/M.Q4_1.gguf", str(out)]
    assert ran["cwd"] == str(tmp_path / converter)
    assert ran["env"]["PYTHONIOENCODING"] == "utf-8"
    if sys.platform.startswith("linux"):
        # Dies with its parent: cancelling an export must not orphan a converter.
        assert callable(ran["preexec_fn"])


def test_a_symlinked_output_folder_is_refused(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    link = tmp_path / "exports" / "M.Q4_1-q4nx"
    link.parent.mkdir()
    link.symlink_to(elsewhere, target_is_directory = True)

    with pytest.raises(RuntimeError, match = "symlink"):
        with export_mod.q4nx.staged_output(link) as staging:
            (staging / "model.q4nx").write_bytes(b"q4nx")
    assert list(elsewhere.iterdir()) == []


def test_a_symlink_planted_during_conversion_is_refused_at_publish(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "model.q4nx").write_bytes(b"someone else's")
    out = tmp_path / "exports" / "M.Q4_1-q4nx"

    with pytest.raises(RuntimeError, match = "symlink"):
        with export_mod.q4nx.staged_output(out) as staging:
            (staging / "model.q4nx").write_bytes(b"q4nx")
            out.symlink_to(elsewhere, target_is_directory = True)
    assert (elsewhere / "model.q4nx").read_bytes() == b"someone else's"


def test_a_reconversion_replaces_the_previous_model_files(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    out = tmp_path / "out"
    out.mkdir()
    for stale in ("chat_template.jinja", "config.json", "model.q4nx", "notes.txt"):
        (out / stale).write_text("previous model")

    with export_mod.q4nx.staged_output(out) as staging:
        (staging / "model.q4nx").write_bytes(b"new")

    assert sorted(p.name for p in out.iterdir()) == ["model.q4nx", "notes.txt"]
    assert (out / "model.q4nx").read_bytes() == b"new"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".out-")] == []


def test_a_failed_reconversion_keeps_the_previous_export(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    out = tmp_path / "out"
    out.mkdir()
    (out / "model.q4nx").write_bytes(b"previous")
    (out / "tokenizer_config.json").write_text("previous")

    with pytest.raises(RuntimeError, match = "converter died"):
        with export_mod.q4nx.staged_output(out) as staging:
            (staging / "model.q4nx").write_bytes(b"partial")
            raise RuntimeError("converter died")

    assert (out / "model.q4nx").read_bytes() == b"previous"
    assert (out / "tokenizer_config.json").read_text() == "previous"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".out-")] == []


def test_converter_failure_names_its_last_stderr_line(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    _stub_installer(export_mod, monkeypatch, tmp_path)

    _fake_popen(
        export_mod,
        monkeypatch,
        returncode = 1,
        stderr = "Traceback ...\nValueError: Unsupported model architecture: mistral3\n",
    )

    with pytest.raises(RuntimeError, match = "Unsupported model architecture: mistral3"):
        export_mod.q4nx.convert_gguf_to_q4nx("/x/M.Q4_1.gguf", tmp_path / "out")


def test_missing_converter_deps_fail_before_installing(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    q4nx = export_mod.q4nx
    real_find_spec = q4nx.importlib.util.find_spec
    monkeypatch.setattr(
        q4nx.importlib.util,
        "find_spec",
        lambda name, *a: None if name in ("torch", "einops") else real_find_spec(name, *a),
    )
    monkeypatch.setattr(q4nx, "_installer", lambda: pytest.fail("installer must not run"))

    with pytest.raises(RuntimeError, match = "torch, einops"):
        q4nx.convert_gguf_to_q4nx("/x/M.Q4_1.gguf", tmp_path / "out")


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
    tokenizer = json.dumps({"added_tokens": [{"id": 3, "content": "x"}]})
    base = _base_folder(
        tmp_path,
        {
            "config.json": '{"eos_token_id": 5}',
            "generation_config.json": '{"eos_token_id": [5, 6]}',
            "tokenizer.json": tokenizer,
            "tokenizer_config.json": '{"eos_token": "x"}',
        },
    )
    gguf = _gguf(tmp_path / "hub" / "Qwen3-0.6B-Q4_1.gguf")

    out = export_mod.q4nx.convert_existing_gguf(gguf, str(base), tmp_path / "exports")

    assert out == tmp_path / "exports" / "Qwen3-0.6B-Q4_1-q4nx"
    assert calls == ["Qwen3-0.6B-Q4_1.gguf"]
    assert (out / "tokenizer.json").read_text() == tokenizer
    assert not (out / "config.json").exists()
    # FLM exits without an eos_token_id array; the tokenizer's eos comes first, then the configs'.
    assert json.loads((out / "tokenizer_config.json").read_text())["eos_token_id"] == [3, 5, 6]
    # Neither the base folder nor its tokenizer_config carries a template, so the GGUF's is used.
    assert (out / "chat_template.jinja").read_text() == "{{ gguf }}"


def test_existing_gguf_keeps_a_template_already_in_tokenizer_config(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    _fake_converter(export_mod, monkeypatch, [])
    base = _base_folder(
        tmp_path,
        {"config.json": '{"eos_token_id": 2}', "tokenizer_config.json": '{"chat_template": "t"}'},
    )

    out = export_mod.q4nx.convert_existing_gguf(_gguf(tmp_path / "m.gguf"), str(base), tmp_path)

    assert not (out / "chat_template.jinja").exists()


def test_a_bos_token_gets_the_id_flm_requires(monkeypatch, tmp_path):
    q4nx = _backend(monkeypatch, tmp_path, object())[0].q4nx
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"added_tokens": [{"id": 128000, "content": "<|begin_of_text|>"}]})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        '{"bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>"}'
    )

    q4nx.write_flm_tokenizer_config(tmp_path, {"eos_token_id": [128001, 128008, 128009]})

    config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert config["bos_token_id"] == 128000
    assert config["eos_token_id"] == [128001, 128008, 128009]


def test_no_eos_id_anywhere_fails(monkeypatch, tmp_path):
    q4nx = _backend(monkeypatch, tmp_path, object())[0].q4nx
    (tmp_path / "tokenizer_config.json").write_text('{"eos_token": "<unknown>"}')
    with pytest.raises(RuntimeError, match = "end-of-sequence"):
        q4nx.write_flm_tokenizer_config(tmp_path, None, {})


def test_standalone_conversions_do_not_overlap(monkeypatch, tmp_path):
    import threading
    import time

    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    active, overlaps = [0], []

    def convert(gguf_path, out_dir):
        active[0] += 1
        overlaps.append(active[0])
        time.sleep(0.2)
        out_dir.mkdir(parents = True, exist_ok = True)
        (out_dir / "model.q4nx").write_bytes(b"q4nx")
        active[0] -= 1

    monkeypatch.setattr(export_mod.q4nx, "convert_gguf_to_q4nx", convert)
    monkeypatch.setattr(export_mod.q4nx, "_gguf_chat_template", lambda _p: None)
    base = _base_folder(
        tmp_path, {"config.json": '{"eos_token_id": 2}', "tokenizer_config.json": "{}"}
    )
    gguf = _gguf(tmp_path / "m.gguf")
    threads = [
        threading.Thread(
            target = export_mod.q4nx.convert_existing_gguf,
            args = (gguf, str(base), tmp_path / f"out{i % 2}"),
        )
        for i in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert overlaps == [1, 1, 1]


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
