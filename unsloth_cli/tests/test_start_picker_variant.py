# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A picked GGUF file must not silently override an explicit quantization choice."""

import pytest
import typer

import unsloth_cli.commands.start as start
from unsloth_cli import _model_catalog as catalog
from unsloth_cli._inference import ensure_studio_backend_path
from unsloth_cli.tests.test_start import fake_studio
from unsloth_cli.tests.test_start_model_picker import picker, _empty_server, _loads


@pytest.fixture()
def cached_gguf_pick(tmp_path, monkeypatch, picker):
    ensure_studio_backend_path(seed_cache_env = False)
    repo = tmp_path / "models--Org--Tiny-GGUF"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    for quant in ("Q4_K_M", "Q8_0"):
        (snapshot / f"tiny-{quant}.gguf").write_bytes(b"GGUF" + b"\0" * 4092)
    target = catalog._cached_gguf_load_id({"repo_id": "Org/Tiny-GGUF", "cache_path": str(repo)})
    assert target == str(snapshot / "tiny-Q4_K_M.gguf")
    monkeypatch.setattr(
        catalog,
        "list_chat_models",
        lambda: [catalog.ModelEntry("Downloaded", "Tiny-GGUF", "Q4_K_M, Q8_0", target)],
    )
    return target


def _connect_picked(mode, load, picker, fake_studio, monkeypatch, attempts):
    _empty_server(fake_studio, monkeypatch)
    picker["answers"] = [""]
    if mode == "auto-start":
        monkeypatch.setattr(start, "find_studio_server", lambda: None)
        monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)

        def require(model, load, **kwargs):
            attempts.append((model, load))
            raise typer.Exit(code = 7)

        monkeypatch.setattr(start, "_require_studio", require)
    return start._connect(
        None,
        None,
        load,
        serve = mode == "auto-start",
        launch = mode == "auto-start",
        gguf_only = True,
    )


@pytest.mark.parametrize("mode", ["idle-server", "auto-start"])
def test_picked_file_rejects_conflicting_quant_before_load(
    cached_gguf_pick, picker, fake_studio, monkeypatch, capsys, mode
):
    from utils.models.model_config import _direct_gguf_for_variant, detect_gguf_model

    assert detect_gguf_model(cached_gguf_pick) == cached_gguf_pick
    assert _direct_gguf_for_variant(cached_gguf_pick, "Q8_0") is None
    attempts = []
    with pytest.raises(typer.Exit) as error:
        _connect_picked(
            mode,
            start.LoadOptions(gguf_variant = "Q8_0"),
            picker,
            fake_studio,
            monkeypatch,
            attempts,
        )
    assert error.value.exit_code == 1
    assert not attempts
    assert not _loads(fake_studio)
    output = capsys.readouterr()
    assert "--gguf-variant Q8_0" in output.out + output.err
    assert "--model" in output.out + output.err


@pytest.mark.parametrize("mode", ["idle-server", "auto-start"])
@pytest.mark.parametrize("variant", ["Q4_K_M", "q4_k_m", "tiny-Q4_K_M", "tiny-Q4_K_M.gguf"])
def test_picked_file_accepts_matching_quant_aliases(
    cached_gguf_pick, picker, fake_studio, monkeypatch, mode, variant
):
    from utils.models.model_config import _direct_gguf_for_variant

    assert _direct_gguf_for_variant(cached_gguf_pick, variant) == cached_gguf_pick
    attempts = []
    load = start.LoadOptions(gguf_variant = variant)
    if mode == "auto-start":
        with pytest.raises(typer.Exit) as error:
            _connect_picked(mode, load, picker, fake_studio, monkeypatch, attempts)
        assert error.value.exit_code == 7
        assert attempts == [(cached_gguf_pick, load)]
    else:
        _connect_picked(mode, load, picker, fake_studio, monkeypatch, attempts)
        assert _loads(fake_studio) == [{"model_path": cached_gguf_pick, "gguf_variant": variant}]


@pytest.mark.parametrize("mode", ["idle-server", "auto-start"])
def test_picked_directory_keeps_explicit_quant(
    cached_gguf_pick, picker, fake_studio, monkeypatch, mode
):
    from pathlib import Path

    directory = str(Path(cached_gguf_pick).parent)
    monkeypatch.setattr(
        catalog,
        "list_chat_models",
        lambda: [catalog.ModelEntry("Fine-tunes", "export", "gguf", directory)],
    )
    attempts = []
    load = start.LoadOptions(gguf_variant = "Q8_0")
    if mode == "auto-start":
        with pytest.raises(typer.Exit) as error:
            _connect_picked(mode, load, picker, fake_studio, monkeypatch, attempts)
        assert error.value.exit_code == 7
        assert attempts == [(directory, load)]
    else:
        _connect_picked(mode, load, picker, fake_studio, monkeypatch, attempts)
        assert _loads(fake_studio) == [{"model_path": directory, "gguf_variant": "Q8_0"}]
