# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Local picker selections stay local and GGUF-only agents reject non-GGUF paths."""

import json

import pytest
import typer
from typer.testing import CliRunner

import unsloth_cli.commands.start as start
from unsloth_cli import _model_catalog as catalog
from unsloth_cli import _model_picker as picker_module
from unsloth_cli._inference import ensure_studio_backend_path
from unsloth_cli.tests.test_start import fake_studio, GEMINI_CONNECTION
from unsloth_cli.tests.test_start_model_picker import picker, _empty_server, _loads


LOCAL_ID = "unsloth/gemma-3-270m-it"


@pytest.mark.parametrize("agent", ["opencode", "pi", "hermes"])
def test_local_selection_wins_over_connection_with_same_model_id(
    fake_studio, picker, monkeypatch, agent
):
    _empty_server(fake_studio, monkeypatch)
    fake_studio.state["providers"] = [{**GEMINI_CONNECTION, "models": [LOCAL_ID]}]
    assert (
        catalog._cached_model_load_id({"repo_id": LOCAL_ID, "model_format": "safetensors"})
        == LOCAL_ID
    )
    monkeypatch.setattr(
        catalog,
        "list_chat_models",
        lambda: [catalog.ModelEntry("Downloaded", "gemma-3-270m-it", "", LOCAL_ID)],
    )
    picker["answers"] = ["1"]
    result = CliRunner().invoke(start.start_app, [agent, "--no-launch"])
    assert result.exit_code == 0, result.output
    assert _loads(fake_studio) == [{"require_idle": True, "model_path": LOCAL_ID}], result.output
    assert "external::" not in result.output


@pytest.mark.parametrize("model", [LOCAL_ID, "external::c1::unsloth%2Fgemma-3-270m-it"])
def test_explicit_connection_still_works(fake_studio, picker, model):
    fake_studio.state["providers"] = [{**GEMINI_CONNECTION, "models": [LOCAL_ID]}]
    result = CliRunner().invoke(start.start_app, ["opencode", "--no-launch", "--model", model])
    assert result.exit_code == 0, result.output
    assert _loads(fake_studio) == []
    assert picker["listed"] == 0
    assert "external::c1::" in result.output


@pytest.fixture(params = ["my-run-GGUF", "my-run.gguf"])
def non_gguf_training_entry(tmp_path, monkeypatch, request):
    ensure_studio_backend_path(seed_cache_env = False)
    import utils.models as models
    from utils.models.model_config import _detect_training_output_type

    folder = tmp_path / "outputs" / request.param
    folder.mkdir(parents = True)
    (folder / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": LOCAL_ID, "peft_type": "LORA"})
    )
    (folder / "adapter_model.safetensors").write_bytes(b"nonempty-placeholder-not-loaded")
    actual_type = _detect_training_output_type(folder)
    assert actual_type == "lora"
    monkeypatch.setattr(
        models, "scan_trained_models", lambda: [(folder.name, str(folder), actual_type)]
    )
    monkeypatch.setattr(catalog, "_runs_by_output_dir", lambda: {})
    entries = catalog.trained_entries()
    assert len(entries) == 1 and entries[0].model == str(folder)
    return entries[0]


def test_non_gguf_directory_suffix_cannot_pass_gguf_filter(non_gguf_training_entry):
    assert picker_module._is_gguf_entry(non_gguf_training_entry) is False


def test_gguf_autostart_enter_skips_non_gguf_training_directory(
    non_gguf_training_entry, picker, monkeypatch, tmp_path
):
    real_gguf = tmp_path / "actual-Q4_K_M.gguf"
    real_gguf.write_bytes(b"GGUF-placeholder-not-loaded")
    monkeypatch.setattr(
        catalog,
        "list_chat_models",
        lambda: [
            non_gguf_training_entry,
            catalog.ModelEntry("Downloaded", "actual-GGUF", "Q4_K_M", str(real_gguf)),
        ],
    )
    monkeypatch.setattr(start, "find_studio_server", lambda: None)
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    seen = []

    def require(model, load, **kwargs):
        seen.append(model)
        raise typer.Exit(code = 7)

    monkeypatch.setattr(start, "_require_studio", require)
    picker["answers"] = [""]
    with pytest.raises(typer.Exit):
        start._connect(None, None, serve = True, launch = True, gguf_only = True)
    assert seen == [str(real_gguf)]


@pytest.mark.parametrize(
    "model",
    ["/outputs/my-run-GGUF", r"C:\outputs\my-run-GGUF", "C:/outputs/my-run-GGUF", "./my-run-GGUF"],
)
def test_non_gguf_path_spellings_do_not_become_repository_fallbacks(model):
    entry = catalog.ModelEntry("Fine-tunes", "my-run-GGUF", "lora", model)
    assert picker_module._is_gguf_entry(entry) is False


def test_repository_gguf_fallback_remains_supported():
    assert picker_module._is_gguf_entry(
        catalog.ModelEntry("Downloaded", "repo", "Q4_K_M", "unsloth/Tiny-GGUF")
    )


@pytest.mark.parametrize("state", ["idle", "foreign", "busy", "denied", "unverified", "unknown"])
def test_local_selection_revalidates_server_appearing_during_prompt(
    fake_studio, picker, monkeypatch, state
):
    status = {
        "foreign": {"yours": False, "loaded": [], "loading": []},
        "busy": {"loaded": ["resident"], "loading": []},
        "unknown": {},
    }.get(state)
    _empty_server(fake_studio, monkeypatch, status = status)
    original = start._http_json
    checked = []

    def http(
        method,
        url,
        key,
        payload = None,
        **kwargs,
    ):
        if url.endswith("/api/auth/model-access"):
            checked.extend(payload)
            return {"allowed": [] if state == "denied" else payload}
        return original(method, url, key, payload = payload, **kwargs)

    monkeypatch.setattr(start, "_http_json", http)
    if state == "unverified":
        monkeypatch.setattr(start, "verify_studio_identity", lambda base: False)
    fake_studio.state["providers"] = [{**GEMINI_CONNECTION, "models": [LOCAL_ID]}]
    monkeypatch.setattr(
        catalog,
        "list_chat_models",
        lambda: [catalog.ModelEntry("Downloaded", "local", "", LOCAL_ID)],
    )
    discoveries = []

    def discover():
        discoveries.append(True)
        return None if len(discoveries) == 1 else "http://127.0.0.1:8888"

    monkeypatch.setattr(start, "find_studio_server", discover)
    monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)
    picker["answers"] = ["1"]
    if state == "idle":
        result = start._connect(None, None, serve = True, launch = True, connections = True)
        assert result[2]["id"] == LOCAL_ID
        assert _loads(fake_studio) == [{"require_idle": True, "model_path": LOCAL_ID}]
        assert checked == [LOCAL_ID]
    else:
        with pytest.raises(typer.Exit):
            start._connect(None, None, serve = True, launch = True, connections = True)
        assert _loads(fake_studio) == []
        if state == "unverified":
            assert checked == []
