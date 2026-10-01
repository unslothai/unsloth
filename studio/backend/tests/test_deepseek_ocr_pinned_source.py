# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The DeepSeek OCR modelling code is fetched from the Hub, so it must be pinned.

Two properties, both of which the previous version of this path lacked:

- the fetch names a revision, so the code that runs is the code that was reviewed
  rather than whatever the branch points at when a user starts a training run,
- the import comes from that fetch. The old code decided "already available" with a
  bare `from deepseek_ocr.modeling_deepseekocr import ...`, which any directory named
  `deepseek_ocr` anywhere on `sys.path` satisfied. That directory was imported, which
  runs its code, and the real download was then skipped.

No network: the fetch is stubbed and the pinned source is a local fixture. What is
exercised for real is which directory the import resolves to.
"""

import re
import sys

import pytest

from utils import third_party_source
from utils.third_party_source import (
    _DEEPSEEK_OCR_MODULES,
    _DEEPSEEK_OCR_PACKAGE,
    _DEEPSEEK_OCR_REPOSITORY,
    _DEEPSEEK_OCR_REVISION,
    ensure_deepseek_ocr_source,
    import_deepseek_ocr_module,
)


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    """Point cache_root() at a temp dir so a test never touches a real install."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio-home"))
    yield
    for name in [m for m in sys.modules if m.split(".")[0] == _DEEPSEEK_OCR_PACKAGE]:
        del sys.modules[name]


def _write_package(root, body = "VALUE = 'pinned'\n"):
    """A complete pinned package, as a successful install would leave it."""
    package = root / _DEEPSEEK_OCR_PACKAGE
    package.mkdir(parents = True, exist_ok = True)
    (package / "__init__.py").write_text("", encoding = "utf-8")
    for name in _DEEPSEEK_OCR_MODULES:
        (package / name).write_text(body, encoding = "utf-8")
    return package


def test_the_pinned_revision_is_a_full_commit_sha():
    """ "main" here would silently restore the unpinned behaviour."""
    assert re.fullmatch(r"[0-9a-f]{40}", _DEEPSEEK_OCR_REVISION)


def test_the_fetch_names_the_revision_and_only_python(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(
        repo_id = None,
        *args,
        **kwargs,
    ):
        calls.append((repo_id, kwargs))
        _write_package(tmp_path / "unused")
        destination = kwargs["local_dir"]
        from pathlib import Path

        package = Path(destination)
        package.mkdir(parents = True, exist_ok = True)
        for name in _DEEPSEEK_OCR_MODULES:
            (package / name).write_text("VALUE = 'pinned'\n", encoding = "utf-8")
        return str(package)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download, raising = True)

    source = ensure_deepseek_ocr_source()

    assert len(calls) == 1
    repo_id, kwargs = calls[0]
    assert repo_id == _DEEPSEEK_OCR_REPOSITORY
    assert kwargs["revision"] == _DEEPSEEK_OCR_REVISION
    assert kwargs["allow_patterns"] == ["*.py"]
    # Not inside the backend source tree: installing this must not put the backend's own
    # directories on the import path as a side effect.
    assert str(source).startswith(str(tmp_path))
    assert _DEEPSEEK_OCR_REVISION in str(source)


def test_a_complete_install_is_reused_without_fetching(tmp_path, monkeypatch):
    """Idempotent, and the second run is free."""

    def refuse(*args, **kwargs):
        raise AssertionError("a complete install must not fetch again")

    first_source = None

    def fake_snapshot_download(
        repo_id = None,
        *args,
        **kwargs,
    ):
        from pathlib import Path

        package = Path(kwargs["local_dir"])
        package.mkdir(parents = True, exist_ok = True)
        for name in _DEEPSEEK_OCR_MODULES:
            (package / name).write_text("VALUE = 'pinned'\n", encoding = "utf-8")
        return str(package)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    first_source = ensure_deepseek_ocr_source()

    monkeypatch.setattr("huggingface_hub.snapshot_download", refuse)
    assert ensure_deepseek_ocr_source() == first_source


def test_a_partial_install_is_rebuilt(tmp_path, monkeypatch):
    """A half-written tree is not a valid install, so it is replaced, not imported."""
    fetched = []

    def fake_snapshot_download(
        repo_id = None,
        *args,
        **kwargs,
    ):
        from pathlib import Path

        fetched.append(repo_id)
        package = Path(kwargs["local_dir"])
        package.mkdir(parents = True, exist_ok = True)
        for name in _DEEPSEEK_OCR_MODULES:
            (package / name).write_text("VALUE = 'pinned'\n", encoding = "utf-8")
        return str(package)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    source = ensure_deepseek_ocr_source()
    (source / _DEEPSEEK_OCR_PACKAGE / _DEEPSEEK_OCR_MODULES[0]).unlink()

    ensure_deepseek_ocr_source()

    assert len(fetched) == 2


def test_a_foreign_package_on_sys_path_is_not_what_gets_imported(tmp_path, monkeypatch):
    """The defect: a `deepseek_ocr` directory on sys.path used to be imported instead.

    The witness file is how this observes execution. A bare
    `from deepseek_ocr.modeling_deepseekocr import ...` runs the foreign package's
    module body, so the witness exists before any check can be made about where the
    code came from.
    """
    witness = tmp_path / "witness.txt"
    foreign = tmp_path / "foreign" / _DEEPSEEK_OCR_PACKAGE
    foreign.mkdir(parents = True)
    (foreign / "__init__.py").write_text(
        f"open({str(witness)!r}, 'a').write('imported\\n')\n", encoding = "utf-8"
    )
    (foreign / "modeling_deepseekocr.py").write_text(
        f"open({str(witness)!r}, 'a').write('imported\\n')\n"
        "def format_messages(*args, **kwargs):\n    return None\n",
        encoding = "utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path / "foreign"))

    pinned = tmp_path / "pinned"
    _write_package(pinned)

    module = import_deepseek_ocr_module("deepseek_ocr.modeling_deepseekocr", pinned)

    assert not witness.exists(), "the foreign deepseek_ocr package was executed"
    assert module.VALUE == "pinned"
    assert str(pinned.resolve()) in module.__file__


def test_the_trainer_entry_point_reports_success_from_the_pinned_source(tmp_path, monkeypatch):
    """`_ensure_deepseek_ocr_installed` keeps its True/False contract.

    The training flow checks the return value and surfaces an error to the user on
    False, so the contract is what keeps the UX identical.
    """
    pinned = tmp_path / "pinned"
    _write_package(pinned)
    monkeypatch.setattr(third_party_source, "ensure_deepseek_ocr_source", lambda *a, **k: pinned)

    from core.training import trainer

    assert trainer._ensure_deepseek_ocr_installed() is True


def test_the_trainer_entry_point_returns_false_when_the_source_is_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(third_party_source, "ensure_deepseek_ocr_source", unavailable)

    from core.training import trainer

    assert trainer._ensure_deepseek_ocr_installed() is False
