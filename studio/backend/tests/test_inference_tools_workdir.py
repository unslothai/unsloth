# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path

from core.inference import tools


def test_tool_workdir_exposes_wiki_shortcuts(tmp_path: Path, monkeypatch):
    vault_root = tmp_path / "vault"
    (vault_root / "wiki" / "sources").mkdir(parents=True)
    (vault_root / "wiki" / "entities").mkdir(parents=True)
    (vault_root / "wiki" / "concepts").mkdir(parents=True)
    (vault_root / "wiki" / "analysis").mkdir(parents=True)
    (vault_root / "raw").mkdir(parents=True)

    home = tmp_path / "home"
    home.mkdir()

    monkeypatch.setenv("UNSLOTH_WIKI_VAULT", str(vault_root))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(tools, "_workdirs", {})

    workdir = Path(tools._get_workdir("session-abc"))

    assert workdir.joinpath("wiki").resolve() == (vault_root / "wiki").resolve()
    assert workdir.joinpath("sources").resolve() == (vault_root / "wiki" / "sources").resolve()
    assert workdir.joinpath("entities").resolve() == (vault_root / "wiki" / "entities").resolve()
    assert workdir.joinpath("concepts").resolve() == (vault_root / "wiki" / "concepts").resolve()
    assert workdir.joinpath("analysis").resolve() == (vault_root / "wiki" / "analysis").resolve()
    assert workdir.joinpath("raw").resolve() == (vault_root / "raw").resolve()


def test_tool_workdir_keeps_existing_real_directory(tmp_path: Path, monkeypatch):
    vault_root = tmp_path / "vault"
    (vault_root / "wiki" / "sources").mkdir(parents=True)

    home = tmp_path / "home"
    home.mkdir()

    monkeypatch.setenv("UNSLOTH_WIKI_VAULT", str(vault_root))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(tools, "_workdirs", {})

    workdir = Path(tools._get_workdir("session-xyz"))
    local_sources = workdir / "sources"

    # Replace the symlink with a real directory; the helper should not clobber it.
    if local_sources.is_symlink():
        local_sources.unlink()
    local_sources.mkdir(exist_ok=True)
    marker = local_sources / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    tools._ensure_wiki_shortcuts(str(workdir))

    assert marker.read_text(encoding="utf-8") == "keep"
