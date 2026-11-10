# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First-use PDF installation targets this environment, is bounded, and never downloads in unit tests."""

from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
from types import ModuleType

import pytest

from utils import optional_pdf as pdf


class _Environment:
    """PDF modules become importable once the fake installer has run."""

    def __init__(self):
        self.installed = False
        self.calls = []
        self.modules = {}


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
    for name in pdf._MODULES:
        monkeypatch.delitem(sys.modules, name, raising = False)
    monkeypatch.setattr(pdf, "_lock_path", lambda: tmp_path / "install.lock")
    monkeypatch.setattr(pdf, "_missing_requirements", lambda: ["pymupdf==1.27.2.3"])
    monkeypatch.setattr(pdf, "_find_uv", lambda: None)
    environment = _Environment()
    real_import = pdf.importlib.import_module
    real_find_spec = pdf.importlib.util.find_spec

    def import_module(name):
        if name not in pdf._MODULES:
            return real_import(name)
        if not environment.installed:
            raise ModuleNotFoundError("PDF dependency absent", name = name)
        return environment.modules.setdefault(name, ModuleType(name))

    def find_spec(name, *args):
        if name not in pdf._MODULES | {"tabulate"}:
            return real_find_spec(name, *args)
        return object() if environment.installed else None

    def install(command, **kwargs):
        environment.calls.append((command, kwargs))
        environment.installed = True

    monkeypatch.setattr(pdf.importlib, "import_module", import_module)
    monkeypatch.setattr(pdf.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(pdf.subprocess, "run", install)
    return environment


@pytest.mark.parametrize("name", sorted(pdf._MODULES))
def test_installed_parser_wins_even_when_downloads_disabled(monkeypatch, name):
    installed = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, installed)
    monkeypatch.setenv("UNSLOTH_PDF_AUTO_INSTALL", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        pdf, "_install", lambda *a: pytest.fail("installed module needs no install")
    )
    assert pdf.load_pdf_module(name) is installed


@pytest.mark.parametrize("name", ["os", "tabulate", "pymupdf;pip", ""])
def test_only_pdf_modules_are_allowed(monkeypatch, name):
    monkeypatch.setattr(pdf, "_import", lambda name: pytest.fail("invalid name imported"))
    with pytest.raises(ValueError, match = "Not an optional PDF module"):
        pdf.load_pdf_module(name)


@pytest.mark.parametrize("installer", ["uv", "pip"])
def test_parallel_requests_install_once_into_this_environment(missing_pdf, monkeypatch, installer):
    monkeypatch.setattr(pdf, "_find_uv", lambda: "/tools/uv" if installer == "uv" else None)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-for-installer")
    monkeypatch.setenv("HF_TOKEN", "not-for-installer")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://unwanted.invalid")
    monkeypatch.setenv("UV_TARGET", "/unwanted-target")
    monkeypatch.setenv("SSL_CERT_FILE", "/transport/cert.pem")
    original_path = list(sys.path)
    with ThreadPoolExecutor(max_workers = 4) as pool:
        modules = list(pool.map(pdf.load_pdf_module, ["pymupdf"] * 4))
    assert len({id(module) for module in modules}) == 1
    assert len(missing_pdf.calls) == 1
    command, options = missing_pdf.calls[0]
    if installer == "uv":
        assert command[:6] == [
            "/tools/uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-config",
        ]
    else:
        assert command[:3] == [sys.executable, "-m", "pip"]
    # Installs into this interpreter's environment, so child interpreters see it too.
    assert "--target" not in command and "--prefix" not in command
    assert "--no-deps" in command and command[command.index("--only-binary") + 1] == ":all:"
    assert command[-1] == "pymupdf==1.27.2.3"
    assert options["timeout"] <= pdf._INSTALL_TIMEOUT
    assert (
        not {"AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "PIP_EXTRA_INDEX_URL", "UV_TARGET"}
        & options["env"].keys()
    )
    assert options["env"]["SSL_CERT_FILE"] == "/transport/cert.pem"
    assert options["env"]["PIP_CONFIG_FILE"] == os.devnull
    assert sys.path == original_path


def test_failed_uv_attempt_falls_back_to_pip(missing_pdf, monkeypatch):
    monkeypatch.setattr(pdf, "_find_uv", lambda: "/tools/uv")
    attempts = []

    def install(command, **kwargs):
        attempts.append(command[0])
        if command[0] == "/tools/uv":
            raise subprocess.CalledProcessError(1, command)
        missing_pdf.installed = True

    monkeypatch.setattr(pdf.subprocess, "run", install)
    assert pdf.load_pdf_module("pymupdf").__name__ == "pymupdf"
    assert attempts == ["/tools/uv", sys.executable]


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_disabled_downloads_are_actionable(missing_pdf, monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_PDF_AUTO_INSTALL", value)
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "downloads are disabled") as error:
        pdf.load_pdf_module("pymupdf")
    assert pdf.INSTALL_COMMAND in str(error.value)
    assert not missing_pdf.calls


@pytest.mark.parametrize("setting", ["HF_HUB_OFFLINE", "UV_OFFLINE", "PIP_NO_INDEX"])
def test_offline_requests_do_not_attempt_downloads(missing_pdf, monkeypatch, setting):
    monkeypatch.setenv(setting, "1")
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "offline mode"):
        pdf.load_pdf_module("pymupdf")
    assert not missing_pdf.calls


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, ["pip"], stderr = "private proxy credential"),
        subprocess.TimeoutExpired(["pip"], 120),
        OSError("installer unavailable"),
    ],
)
def test_failed_installs_give_guidance_without_installer_output(missing_pdf, monkeypatch, failure):
    def install(command, **kwargs):
        raise failure

    monkeypatch.setattr(pdf.subprocess, "run", install)
    with pytest.raises(pdf.PDFDependencyUnavailable) as error:
        pdf.load_pdf_module("pymupdf")
    assert pdf.INSTALL_COMMAND in str(error.value)
    assert "private proxy credential" not in str(error.value)


def test_success_that_leaves_the_module_missing_is_actionable(missing_pdf, monkeypatch):
    monkeypatch.setattr(pdf.subprocess, "run", lambda *a, **kw: None)
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "could not be imported"):
        pdf.load_pdf_module("pymupdf")


def test_a_load_during_an_install_waits_for_the_complete_package(missing_pdf, monkeypatch):
    import threading

    partial, complete = ModuleType("pymupdf"), ModuleType("pymupdf")
    complete.open = object()
    state = {"files": "absent"}
    readers, threads = [], []

    def import_module(name):
        if state["files"] == "absent":
            raise ModuleNotFoundError("PDF dependency absent", name = name)
        # Like sys.modules, the first import is kept, even of a half-written package.
        return missing_pdf.modules.setdefault(
            name, partial if state["files"] == "partial" else complete
        )

    def install(command, **kwargs):
        missing_pdf.calls.append(command)
        state["files"] = "partial"
        reader = threading.Thread(target = lambda: readers.append(pdf.load_pdf_module("pymupdf")))
        threads.append(reader)
        reader.start()
        reader.join(timeout = 0.5)
        state["files"] = "installed"

    monkeypatch.setattr(pdf.importlib, "import_module", import_module)
    monkeypatch.setattr(pdf.subprocess, "run", install)
    assert pdf.load_pdf_module("pymupdf") is complete
    threads[0].join(timeout = 5)
    assert readers == [complete]
    assert len(missing_pdf.calls) == 1


def test_a_concurrent_install_by_another_process_is_reused(missing_pdf, monkeypatch):
    class OtherProcessInstalled:
        def __init__(self, path, timeout):
            pass

        def __enter__(self):
            missing_pdf.installed = True

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(pdf, "FileLock", OtherProcessInstalled)
    assert pdf.load_pdf_module("fitz").__name__ == "fitz"
    assert not missing_pdf.calls


def test_ensure_installs_for_children_without_importing(missing_pdf, monkeypatch):
    imported = []
    monkeypatch.setattr(pdf, "_import", lambda name: imported.append(name))
    pdf.ensure_pdf_support()
    pdf.ensure_pdf_support()
    assert len(missing_pdf.calls) == 1
    assert not imported


def test_ensure_is_free_when_already_installed(missing_pdf, monkeypatch):
    missing_pdf.installed = True
    monkeypatch.setenv("UNSLOTH_PDF_AUTO_INSTALL", "0")
    pdf.ensure_pdf_support()
    assert not missing_pdf.calls


def test_ensure_reports_disabled_downloads(missing_pdf, monkeypatch):
    monkeypatch.setenv("UNSLOTH_PDF_AUTO_INSTALL", "0")
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "downloads are disabled"):
        pdf.ensure_pdf_support()


def test_an_unwritable_environment_imports_without_waiting_on_a_lock(
    missing_pdf, monkeypatch, tmp_path
):
    missing_pdf.installed = True
    monkeypatch.setattr(pdf, "_lock_path", lambda: tmp_path / "missing-dir" / "install.lock")
    monkeypatch.setattr(
        pdf, "FileLock", lambda *a, **kw: pytest.fail("no lock file can exist here")
    )
    assert pdf.load_pdf_module("pymupdf").__name__ == "pymupdf"


def test_install_lock_is_scoped_to_this_environment():
    assert pdf._lock_path().parent == pdf.Path(sys.prefix)


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
def test_broken_installed_native_packages_get_repair_guidance(monkeypatch, tmp_path, failure):
    def fail(name):
        raise failure

    monkeypatch.delitem(sys.modules, "pymupdf", raising = False)
    monkeypatch.setattr(pdf, "_lock_path", lambda: tmp_path / "install.lock")
    monkeypatch.setattr(pdf.importlib, "import_module", fail)
    monkeypatch.setattr(
        pdf, "_install", lambda *a: pytest.fail("must not replace native libraries")
    )
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "repair the installation"):
        pdf.load_pdf_module("pymupdf")


def test_requirements_file_reads_as_utf8():
    assert "pymupdf4llm==0.3.4" in pdf._REQUIREMENTS.read_text(encoding = "utf-8")


def test_thread_lock_wait_is_bounded(missing_pdf, monkeypatch):
    class Busy:
        def acquire(self, timeout):
            assert 0 < timeout <= pdf._INSTALL_TIMEOUT
            return False

    monkeypatch.setattr(pdf, "_LOCK", Busy())
    with pytest.raises(pdf.PDFDependencyUnavailable, match = "timed out"):
        pdf.load_pdf_module("pymupdf")
    assert not missing_pdf.calls


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
    assert not missing_pdf.calls
