# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import typer


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _missing_structlog():
    raise ModuleNotFoundError("No module named 'structlog'", name = "structlog")


def test_desktop_runtime_check_reports_missing_dependency_as_json(monkeypatch, capsys):
    studio = importlib.import_module("unsloth_cli.commands.studio")
    monkeypatch.setattr(studio, "_load_run_module", _missing_structlog)

    with pytest.raises(typer.Exit) as exited:
        studio.desktop_runtime_check(_json_output = True)

    assert exited.value.exit_code == 1
    assert json.loads(capsys.readouterr().out) == {
        "runtime_ready": False,
        "reason": "missing_dependency",
        "module": "structlog",
    }


def test_desktop_runtime_check_reports_success(monkeypatch, capsys):
    studio = importlib.import_module("unsloth_cli.commands.studio")
    monkeypatch.setattr(studio, "_load_run_module", lambda: object())
    monkeypatch.setattr(studio, "_missing_studio_requirement", lambda _run_mod: None)

    studio.desktop_runtime_check(_json_output = True)

    assert json.loads(capsys.readouterr().out) == {"runtime_ready": True}


def test_desktop_runtime_check_catches_later_startup_dependency(monkeypatch, capsys, tmp_path):
    studio = importlib.import_module("unsloth_cli.commands.studio")
    backend = tmp_path / "backend"
    requirements = backend / "requirements"
    requirements.mkdir(parents = True)
    (requirements / "studio.txt").write_text(
        "definitely-missing-studio-package\n",
        encoding = "utf-8",
    )
    run_mod = SimpleNamespace(__file__ = str(backend / "run.py"))
    monkeypatch.setattr(studio, "_load_run_module", lambda: run_mod)

    with pytest.raises(typer.Exit):
        studio.desktop_runtime_check(_json_output = True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["module"] == "definitely-missing-studio-package"


def test_desktop_runtime_check_ignores_pip_flag_lines(monkeypatch, capsys, tmp_path):
    """Repair reinstalls the same file, so an unparseable line breaks it forever."""
    studio = importlib.import_module("unsloth_cli.commands.studio")
    backend = tmp_path / "backend"
    requirements = backend / "requirements"
    requirements.mkdir(parents = True)
    (requirements / "studio.txt").write_text(
        "--extra-index-url https://example.invalid/simple\n-r base.txt\n",
        encoding = "utf-8",
    )
    run_mod = SimpleNamespace(__file__ = str(backend / "run.py"))
    monkeypatch.setattr(studio, "_load_run_module", lambda: run_mod)

    studio.desktop_runtime_check(_json_output = True)

    assert json.loads(capsys.readouterr().out) == {"runtime_ready": True}


def test_desktop_runtime_check_accepts_a_prerelease_over_a_floor(monkeypatch, capsys, tmp_path):
    studio = importlib.import_module("unsloth_cli.commands.studio")
    backend = tmp_path / "backend"
    requirements = backend / "requirements"
    requirements.mkdir(parents = True)
    (requirements / "studio.txt").write_text("example-package>=1.0\n", encoding = "utf-8")
    run_mod = SimpleNamespace(__file__ = str(backend / "run.py"))
    monkeypatch.setattr(studio, "_load_run_module", lambda: run_mod)
    monkeypatch.setattr(
        importlib.import_module("importlib.metadata"),
        "distribution",
        lambda _name: SimpleNamespace(
            version = "2.0.0b1",
            files = [],
            requires = None,
            read_text = lambda _n: None,
        ),
    )

    studio.desktop_runtime_check(_json_output = True)

    assert json.loads(capsys.readouterr().out) == {"runtime_ready": True}


def test_desktop_runtime_check_rejects_version_mismatch(monkeypatch, capsys, tmp_path):
    studio = importlib.import_module("unsloth_cli.commands.studio")
    backend = tmp_path / "backend"
    requirements = backend / "requirements"
    requirements.mkdir(parents = True)
    (requirements / "studio.txt").write_text("example-package==2.0\n", encoding = "utf-8")
    run_mod = SimpleNamespace(__file__ = str(backend / "run.py"))
    monkeypatch.setattr(studio, "_load_run_module", lambda: run_mod)
    monkeypatch.setattr(
        importlib.import_module("importlib.metadata"),
        "distribution",
        lambda _name: SimpleNamespace(
            version = "1.0",
            files = [],
            requires = None,
            read_text = lambda _n: None,
        ),
    )

    with pytest.raises(typer.Exit):
        studio.desktop_runtime_check(_json_output = True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["module"] == "example-package"


def test_desktop_runtime_check_rejects_metadata_without_an_unpacked_package(
    monkeypatch, capsys, tmp_path
):
    """fastapi's wheel stores METADATA first, so an interrupted unpack leaves a
    readable version behind. RECORD is last, so its absence marks the unpack."""
    studio = importlib.import_module("unsloth_cli.commands.studio")
    backend = tmp_path / "backend"
    requirements = backend / "requirements"
    requirements.mkdir(parents = True)
    (requirements / "studio.txt").write_text("fastapi\n", encoding = "utf-8")
    run_mod = SimpleNamespace(__file__ = str(backend / "run.py"))
    monkeypatch.setattr(studio, "_load_run_module", lambda: run_mod)
    monkeypatch.setattr(
        importlib.import_module("importlib.metadata"),
        "distribution",
        lambda _name: SimpleNamespace(
            version = "0.140.5",
            files = None,
            requires = None,
            read_text = lambda _n: None,
        ),
    )

    with pytest.raises(typer.Exit):
        studio.desktop_runtime_check(_json_output = True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["reason"] == "missing_dependency"
    assert payload["module"] == "fastapi"


def _fake_distributions(monkeypatch, installed):
    metadata = importlib.import_module("importlib.metadata")

    def _distribution(name):
        try:
            version, requires = installed[name]
        except KeyError:
            raise metadata.PackageNotFoundError(name) from None
        return SimpleNamespace(
            version = version,
            files = [],
            requires = requires,
            read_text = lambda _n: None,
        )

    monkeypatch.setattr(metadata, "distribution", _distribution)


def test_desktop_runtime_check_rejects_a_missing_transitive_dependency(
    monkeypatch, capsys, tmp_path
):
    """starlette reaches the venv only as a FastAPI dependency, so a direct-only
    check calls the install ready and the server dies on start."""
    studio = importlib.import_module("unsloth_cli.commands.studio")
    backend = tmp_path / "backend"
    requirements = backend / "requirements"
    requirements.mkdir(parents = True)
    (requirements / "studio.txt").write_text("fastapi>=0.115\n", encoding = "utf-8")
    run_mod = SimpleNamespace(__file__ = str(backend / "run.py"))
    monkeypatch.setattr(studio, "_load_run_module", lambda: run_mod)
    _fake_distributions(monkeypatch, {"fastapi": ("0.140.5", ["starlette>=0.40"])})

    with pytest.raises(typer.Exit):
        studio.desktop_runtime_check(_json_output = True)

    assert json.loads(capsys.readouterr().out)["module"] == "starlette"


def test_desktop_runtime_check_ignores_optional_and_circular_dependencies(
    monkeypatch, capsys, tmp_path
):
    """Extras-only dependencies are not missing, and a cycle must terminate."""
    studio = importlib.import_module("unsloth_cli.commands.studio")
    backend = tmp_path / "backend"
    requirements = backend / "requirements"
    requirements.mkdir(parents = True)
    (requirements / "studio.txt").write_text("fastapi\n", encoding = "utf-8")
    run_mod = SimpleNamespace(__file__ = str(backend / "run.py"))
    monkeypatch.setattr(studio, "_load_run_module", lambda: run_mod)
    _fake_distributions(
        monkeypatch,
        {
            "fastapi": ("0.140.5", ['uvicorn; extra == "standard"', "starlette"]),
            # Transitive bounds are not enforced: --no-deps installs leave unmet
            # ones on venvs that work.
            "starlette": ("0.1", ["fastapi>=99"]),
        },
    )

    studio.desktop_runtime_check(_json_output = True)

    assert json.loads(capsys.readouterr().out) == {"runtime_ready": True}


def test_desktop_runtime_check_reports_a_rejected_setting_instead_of_exiting(monkeypatch, capsys):
    """run.py raises SystemExit for values like UNSLOTH_CPU_THREADS=invalid, and
    with no payload the app reinstalls over a value no install can change."""
    studio = importlib.import_module("unsloth_cli.commands.studio")

    def _rejected_setting():
        raise SystemExit("Error: Invalid UNSLOTH_CPU_THREADS value 'invalid'")

    monkeypatch.setattr(studio, "_load_run_module", _rejected_setting)

    with pytest.raises(typer.Exit) as exited:
        studio.desktop_runtime_check(_json_output = True)

    assert exited.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime_ready"] is False
    assert payload["reason"] == "backend_startup_failed"
    assert "UNSLOTH_CPU_THREADS" in payload["error"]


def test_a_root_pin_is_checked_even_when_a_dependency_names_it_first(monkeypatch, capsys, tmp_path):
    """datasets wants huggingface-hub<2, studio.txt pins ==0.36.2. Reached as a
    dependency first, the pin would never get to decide."""
    studio = importlib.import_module("unsloth_cli.commands.studio")
    backend = tmp_path / "backend"
    requirements = backend / "requirements"
    requirements.mkdir(parents = True)
    (requirements / "studio.txt").write_text(
        "datasets==4.3.0\nhuggingface-hub==0.36.2\n",
        encoding = "utf-8",
    )
    run_mod = SimpleNamespace(__file__ = str(backend / "run.py"))
    monkeypatch.setattr(studio, "_load_run_module", lambda: run_mod)
    _fake_distributions(
        monkeypatch,
        {
            "datasets": ("4.3.0", ["huggingface-hub>=0.25,<2"]),
            "huggingface-hub": ("1.25.1", None),
        },
    )

    with pytest.raises(typer.Exit):
        studio.desktop_runtime_check(_json_output = True)

    assert json.loads(capsys.readouterr().out)["module"] == "huggingface-hub"


def test_a_record_without_its_files_is_reported_missing(monkeypatch, capsys, tmp_path):
    """An interrupted replace can recreate the package directory and stop
    part-way through filling it, so the directory existing proves nothing.

    Built as a real dist-info rather than a stub: Distribution.files drops
    entries that no longer exist, which is the whole set this looks for."""
    studio = importlib.import_module("unsloth_cli.commands.studio")
    metadata = importlib.import_module("importlib.metadata")
    backend = tmp_path / "backend"
    requirements = backend / "requirements"
    requirements.mkdir(parents = True)
    (requirements / "studio.txt").write_text("structlog\n", encoding = "utf-8")
    run_mod = SimpleNamespace(__file__ = str(backend / "run.py"))
    monkeypatch.setattr(studio, "_load_run_module", lambda: run_mod)

    site_packages = tmp_path / "site-packages"
    dist_info = site_packages / "structlog-25.1.0.dist-info"
    dist_info.mkdir(parents = True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: structlog\nVersion: 25.1.0\n",
        encoding = "utf-8",
    )
    (dist_info / "RECORD").write_text(
        "structlog/__init__.py,,\n"
        "structlog/processors.py,,\n"
        "structlog/__pycache__/__init__.cpython-313.pyc,,\n"
        "structlog-25.1.0.dist-info/RECORD,,\n",
        encoding = "utf-8",
    )
    installed = next(iter(metadata.distributions(path = [str(site_packages)])))
    monkeypatch.setattr(metadata, "distribution", lambda _name: installed)

    with pytest.raises(typer.Exit):
        studio.desktop_runtime_check(_json_output = True)
    assert json.loads(capsys.readouterr().out)["module"] == "structlog"

    # The directory back but still a file short is the interrupted replace.
    (site_packages / "structlog").mkdir()
    (site_packages / "structlog" / "__init__.py").touch()
    with pytest.raises(typer.Exit):
        studio.desktop_runtime_check(_json_output = True)
    assert json.loads(capsys.readouterr().out)["module"] == "structlog"

    # Complete, and the never-written .pyc must not count as damage.
    (site_packages / "structlog" / "processors.py").touch()
    studio.desktop_runtime_check(_json_output = True)
    assert json.loads(capsys.readouterr().out) == {"runtime_ready": True}
