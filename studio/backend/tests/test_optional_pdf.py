# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First-use PDF installation is isolated, bounded, and never downloads in unit tests."""

from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from utils import optional_pdf as pdf


def _write_wheel(stage: Path) -> None:
    """Small wheel-shaped fixture; real-wheel integration is checked separately."""
    info = stage / "pymupdf-1.27.2.3.dist-info"
    (info / "licenses").mkdir(parents = True)
    files = {
        "pymupdf.py": "answer = 42\n",
        f"{info.name}/METADATA": "Metadata-Version: 2.1\nName: PyMuPDF\nVersion: 1.27.2.3\n",
        f"{info.name}/licenses/COPYING": "upstream license\n",
        "should-not-execute.pth": "import os; os.environ['PDF_PTH_EXECUTED'] = '1'\n",
    }
    for name, content in files.items():
        (stage / name).write_text(content, encoding = "utf-8")
    (info / "RECORD").write_text(
        "".join(f"{name},,\n" for name in [*files, f"{info.name}/RECORD"]), encoding = "utf-8"
    )


@pytest.fixture
def missing_pdf(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_PDF_AUTO_INSTALL", "1")
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "UV_OFFLINE",
        "UV_NO_INDEX",
        "PIP_NO_INDEX",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(pdf, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(pdf, "_missing_requirements", lambda: ["pymupdf==1.27.2.3"])
    monkeypatch.setattr(pdf, "_find_uv", lambda: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in pdf._MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    real_import = importlib.import_module

    def import_module(name):
        if name not in pdf._MODULES:
            return real_import(name)
        target = pdf._target()
        if str(target) not in sys.path or not (target / f"{name}.py").is_file():
            raise ModuleNotFoundError("PDF dependency absent", name = name)
        if sys.modules.get(name) is None:
            spec = importlib.util.spec_from_file_location(name, target / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            monkeypatch.setitem(sys.modules, name, module)
        return sys.modules[name]

    monkeypatch.setattr(pdf.importlib, "import_module", import_module)
    calls = []

    def install(command, **kwargs):
        calls.append((command, kwargs))
        stage = Path(command[command.index("--target") + 1])
        _write_wheel(stage)

    monkeypatch.setattr(pdf.subprocess, "run", install)
    return calls


@pytest.mark.parametrize("name", sorted(pdf._MODULES))
def test_installed_parser_wins_even_when_downloads_disabled(monkeypatch, name):
    installed = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, installed)
    monkeypatch.setenv("UNSLOTH_PDF_AUTO_INSTALL", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(pdf, "_target", lambda: pytest.fail("installed module needs no cache"))
    assert pdf.load_pdf_module(name) is installed


@pytest.mark.parametrize("name", ["os", "tabulate", "pymupdf;pip", ""])
def test_only_pdf_modules_are_allowed(monkeypatch, name):
    monkeypatch.setattr(pdf, "_import", lambda name: pytest.fail("invalid name imported"))
    with pytest.raises(ValueError, match = "Not an optional PDF module"):
        pdf.load_pdf_module(name)


@pytest.mark.parametrize("installer", ["uv", "pip"])
def test_parallel_requests_publish_one_complete_install(missing_pdf, monkeypatch, installer):
    monkeypatch.setattr(pdf, "_find_uv", lambda: "/tools/uv" if installer == "uv" else None)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-for-installer")
    monkeypatch.setenv("HF_TOKEN", "not-for-installer")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://unwanted.invalid")
    monkeypatch.setenv("UV_TARGET", "/unwanted-target")
    monkeypatch.setenv("SSL_CERT_FILE", "/transport/cert.pem")
    monkeypatch.delenv("PDF_PTH_EXECUTED", raising = False)
    original_path = list(sys.path)
    with ThreadPoolExecutor(max_workers = 4) as pool:
        modules = list(pool.map(pdf.load_pdf_module, ["pymupdf"] * 4))
    assert all(module.answer == 42 for module in modules)
    assert len(missing_pdf) == 1
    command, options = missing_pdf[0]
    assert command[0] == ("/tools/uv" if installer == "uv" else sys.executable)
    assert "--no-deps" in command and command[command.index("--only-binary") + 1] == ":all:"
    assert options["timeout"] <= pdf._INSTALL_TIMEOUT
    assert (
        not {"AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "PIP_EXTRA_INDEX_URL", "UV_TARGET"}
        & options["env"].keys()
    )
    assert options["env"]["SSL_CERT_FILE"] == "/transport/cert.pem"
    assert options["env"]["PIP_CONFIG_FILE"] == os.devnull
    target = pdf._target()
    assert (target / ".complete").is_file()
    assert (
        target / "pymupdf-1.27.2.3.dist-info/licenses/COPYING"
    ).read_text() == "upstream license\n"
    assert sys.path[:-1] == original_path and sys.path[-1] == str(target)
    assert "PDF_PTH_EXECUTED" not in os.environ
    assert not list(target.parent.glob(".pdf-*"))


def test_cache_reuse_after_clearing_import_state_offline_and_disabled(missing_pdf, monkeypatch):
    pdf.load_pdf_module("pymupdf")
    target = pdf._target()
    # Simulate a fresh interpreter's import state, retaining the actual wheel cache.
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    sys.path.remove(str(target))
    monkeypatch.setenv("UNSLOTH_PDF_AUTO_INSTALL", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert pdf.load_pdf_module("pymupdf").answer == 42
    assert len(missing_pdf) == 1


def test_failed_uv_attempt_falls_back_to_pip_in_a_fresh_stage(missing_pdf, monkeypatch):
    monkeypatch.setattr(pdf, "_find_uv", lambda: "/tools/uv")
    stages = []

    def install(command, **kwargs):
        stage = Path(command[command.index("--target") + 1])
        stages.append(stage)
        if command[0] == "/tools/uv":
            (stage / "partial-file").write_text("incomplete")
            raise subprocess.CalledProcessError(1, command)
        assert command[:3] == [sys.executable, "-m", "pip"]
        assert not (stage / "partial-file").exists()
        _write_wheel(stage)

    monkeypatch.setattr(pdf.subprocess, "run", install)
    assert pdf.load_pdf_module("pymupdf").answer == 42
    assert len(stages) == 2 and stages[0] != stages[1]
    assert not stages[0].exists()
    assert not (pdf._target() / "partial-file").exists()


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_disabled_downloads_are_actionable(missing_pdf, monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_PDF_AUTO_INSTALL", value)
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "downloads are disabled") as error:
        pdf.load_pdf_module("pymupdf")
    assert pdf.INSTALL_COMMAND in str(error.value)
    assert not missing_pdf and not pdf._target().exists()


@pytest.mark.parametrize("setting", ["HF_HUB_OFFLINE", "UV_OFFLINE", "PIP_NO_INDEX"])
def test_offline_requests_do_not_attempt_downloads(missing_pdf, monkeypatch, setting):
    monkeypatch.setenv(setting, "1")
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "offline mode"):
        pdf.load_pdf_module("pymupdf")
    assert not missing_pdf


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, ["pip"], stderr = "private proxy credential"),
        subprocess.TimeoutExpired(["pip"], 120),
        OSError("installer unavailable"),
    ],
)
def test_failed_installs_do_not_publish_partial_caches(missing_pdf, monkeypatch, failure):
    def install(command, **kwargs):
        stage = Path(command[command.index("--target") + 1])
        (stage / "partial").write_text("unfinished")
        raise failure

    monkeypatch.setattr(pdf.subprocess, "run", install)
    with pytest.raises(pdf.PDFDependencyUnavailable) as error:
        pdf.load_pdf_module("pymupdf")
    assert pdf.INSTALL_COMMAND in str(error.value)
    assert "private proxy credential" not in str(error.value)
    assert not pdf._target().exists()
    assert not list(pdf._target().parent.glob(".pdf-*"))


def test_success_without_complete_wheel_files_is_not_published(missing_pdf, monkeypatch):
    monkeypatch.setattr(pdf.subprocess, "run", lambda *a, **kw: None)
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "incomplete or damaged"):
        pdf.load_pdf_module("pymupdf")
    assert not pdf._target().exists()


@pytest.mark.parametrize("damage", ["marker", "file", "unfinished"])
def test_damaged_cache_is_actionable_and_not_overwritten(missing_pdf, damage):
    target = pdf._target()
    target.mkdir(parents = True)
    _write_wheel(target)
    marker = target / ".complete"
    marker.write_text(
        json.dumps({"version": pdf._CACHE_VERSION, "packages": ["pymupdf==1.27.2.3"]})
    )
    if damage == "marker":
        marker.write_text("not JSON")
    elif damage == "file":
        (target / "pymupdf.py").unlink()
    else:
        marker.unlink()
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "cache is incomplete") as error:
        pdf.load_pdf_module("pymupdf")
    assert pdf.INSTALL_COMMAND in str(error.value)
    assert not missing_pdf


@pytest.mark.parametrize("installed", ["0.8.0", "0.9.0", "0.10.0"])
def test_compatible_installed_tabulate_is_not_reinstalled(monkeypatch, installed):
    def installed_version(name):
        if name == "tabulate":
            return installed
        raise pdf.PackageNotFoundError(name)

    monkeypatch.setattr(pdf, "version", installed_version)
    requirements = pdf._missing_requirements()
    assert "pymupdf4llm==0.3.4" in requirements
    assert not any(entry.startswith("tabulate") for entry in requirements)


def test_an_incompatible_native_engine_is_not_replaced(monkeypatch):
    monkeypatch.setattr(pdf, "version", lambda name: "1.26.0")
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "will not be replaced") as error:
        pdf._missing_requirements()
    assert pdf.INSTALL_COMMAND in str(error.value)


@pytest.mark.parametrize(
    "failure", [ImportError("broken native library"), OSError("dlopen failed")]
)
def test_broken_installed_native_packages_get_repair_guidance(monkeypatch, failure):
    def fail(name):
        raise failure

    monkeypatch.setattr(pdf.importlib, "import_module", fail)
    monkeypatch.setattr(
        pdf, "_install", lambda *a: pytest.fail("must not replace native libraries")
    )
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "repair the installation"):
        pdf.load_pdf_module("pymupdf")


@pytest.mark.parametrize(
    "component", ["requirements", "prefix", "executable", "platform", "abi", "gil"]
)
def test_incompatible_interpreters_and_requirements_do_not_share_a_cache(
    monkeypatch, tmp_path, component
):
    before = pdf._target()
    if component == "requirements":
        path = tmp_path / "pdf.txt"
        path.write_text(pdf._REQUIREMENTS.read_text() + "\n# changed pins\n")
        monkeypatch.setattr(pdf, "_REQUIREMENTS", path)
    elif component in {"prefix", "executable"}:
        monkeypatch.setattr(pdf.sys, component, str(tmp_path / component))
    elif component == "platform":
        monkeypatch.setattr(pdf.sysconfig, "get_platform", lambda: "another-platform")
    else:
        original = pdf.sysconfig.get_config_var
        key = "SOABI" if component == "abi" else "Py_GIL_DISABLED"
        monkeypatch.setattr(
            pdf.sysconfig,
            "get_config_var",
            lambda name: "changed" if name == key else original(name),
        )
    assert pdf._target() != before


def test_thread_lock_wait_is_bounded(missing_pdf, monkeypatch):
    class Busy:
        def acquire(self, timeout):
            assert 0 < timeout <= pdf._INSTALL_TIMEOUT
            return False

    monkeypatch.setattr(pdf, "_LOCK", Busy())
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "timed out"):
        pdf.load_pdf_module("pymupdf")
    assert not missing_pdf


def test_process_lock_wait_is_bounded(missing_pdf, monkeypatch):
    class Busy:
        def __init__(self, path, timeout):
            assert 0 < timeout <= pdf._INSTALL_TIMEOUT

        def __enter__(self):
            raise TimeoutError("other process is installing")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(pdf, "FileLock", Busy)
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "timed out"):
        pdf.load_pdf_module("pymupdf")
    assert not missing_pdf
