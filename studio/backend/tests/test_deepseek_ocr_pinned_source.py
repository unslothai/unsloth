# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The DeepSeek OCR modelling code is fetched from the Hub, so it must be pinned.

Three properties, none of which the previous version of this path had:

- the fetch names a revision, so the code that runs is the code that was reviewed
  rather than whatever the branch points at when a user starts a training run,
- the bytes are checked against committed digests before they are imported, so the
  revision pin is not undone by a cached file edited in place,
- the import comes from that fetch. The old code decided "already available" with a
  bare `from deepseek_ocr.modeling_deepseekocr import ...`, which any directory named
  `deepseek_ocr` anywhere on `sys.path` satisfied. That directory was imported, which
  runs its code, and the real download was then skipped.

No network: the fetch is stubbed and the pinned source is a local fixture, with the
digest map pointed at the fixture's own bytes. What is exercised for real is which
directory the import resolves to and what the install predicate accepts.
"""

import hashlib
import re
import sys
from pathlib import Path

import pytest

from utils import third_party_source
from utils.third_party_source import (
    _DEEPSEEK_OCR_MODULES,
    _DEEPSEEK_OCR_PACKAGE,
    _DEEPSEEK_OCR_REPOSITORY,
    _DEEPSEEK_OCR_REVISION,
    _deepseek_ocr_installed,
    ensure_deepseek_ocr_source,
    import_deepseek_ocr_module,
)


BODY = "VALUE = 'pinned'\n"


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    """Point cache_root() at a temp dir so a test never touches a real install."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio-home"))
    yield
    for name in [m for m in sys.modules if m.split(".")[0] == _DEEPSEEK_OCR_PACKAGE]:
        del sys.modules[name]


@pytest.fixture
def pinned_digests(monkeypatch):
    """Point the committed digest map at the fixture bodies these tests write."""
    digest = hashlib.sha256(BODY.encode("utf-8")).hexdigest()
    monkeypatch.setattr(
        third_party_source,
        "_DEEPSEEK_OCR_DIGESTS",
        {name: digest for name in _DEEPSEEK_OCR_MODULES},
    )


def _write_package(root):
    """A complete pinned package, as a successful install would leave it."""
    package = Path(root) / _DEEPSEEK_OCR_PACKAGE
    package.mkdir(parents = True, exist_ok = True)
    (package / "__init__.py").write_text("", encoding = "utf-8")
    for name in _DEEPSEEK_OCR_MODULES:
        (package / name).write_text(BODY, encoding = "utf-8")
    return package


def _fake_download(
    repo_id = None,
    *args,
    **kwargs,
):
    package = Path(kwargs["local_dir"])
    package.mkdir(parents = True, exist_ok = True)
    for name in _DEEPSEEK_OCR_MODULES:
        (package / name).write_text(BODY, encoding = "utf-8")
    return str(package)


def test_the_pinned_revision_is_a_full_commit_sha():
    """ "main" here would silently restore the unpinned behaviour."""
    assert re.fullmatch(r"[0-9a-f]{40}", _DEEPSEEK_OCR_REVISION)


def test_every_pinned_module_has_a_digest():
    """A module without one would be fetched and imported unverified."""
    assert set(third_party_source._DEEPSEEK_OCR_DIGESTS) == set(_DEEPSEEK_OCR_MODULES)
    for digest in third_party_source._DEEPSEEK_OCR_DIGESTS.values():
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_the_fetch_names_the_revision_and_only_python(tmp_path, monkeypatch, pinned_digests):
    calls = []

    def recording_download(
        repo_id = None,
        *args,
        **kwargs,
    ):
        calls.append((repo_id, kwargs))
        return _fake_download(repo_id, *args, **kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", recording_download)

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


def test_a_complete_install_is_reused_without_fetching(monkeypatch, pinned_digests):
    """Idempotent, and the second run is free."""

    def refuse(*args, **kwargs):
        raise AssertionError("a complete install must not fetch again")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_download)
    first_source = ensure_deepseek_ocr_source()

    monkeypatch.setattr("huggingface_hub.snapshot_download", refuse)
    assert ensure_deepseek_ocr_source() == first_source


def test_a_partial_install_is_rebuilt(monkeypatch, pinned_digests):
    """A half-written tree is not a valid install, so it is replaced, not imported."""
    fetched = []

    def counting_download(
        repo_id = None,
        *args,
        **kwargs,
    ):
        fetched.append(repo_id)
        return _fake_download(repo_id, *args, **kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", counting_download)
    source = ensure_deepseek_ocr_source()
    (source / _DEEPSEEK_OCR_PACKAGE / _DEEPSEEK_OCR_MODULES[0]).unlink()

    ensure_deepseek_ocr_source()

    assert len(fetched) == 2


def test_a_cached_module_edited_in_place_is_rebuilt(monkeypatch, pinned_digests):
    """The case names and file types cannot see.

    An edited file keeps its name, so presence alone would accept it, skip the download
    and import the altered bytes, which is the revision pin defeated at the last step.
    """
    fetched = []

    def counting_download(
        repo_id = None,
        *args,
        **kwargs,
    ):
        fetched.append(repo_id)
        return _fake_download(repo_id, *args, **kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", counting_download)
    source = ensure_deepseek_ocr_source()

    tampered = source / _DEEPSEEK_OCR_PACKAGE / _DEEPSEEK_OCR_MODULES[0]
    tampered.write_text(BODY + "import os\nos.environ['X'] = '1'\n", encoding = "utf-8")

    assert _deepseek_ocr_installed(source) is False
    ensure_deepseek_ocr_source()
    assert len(fetched) == 2
    assert tampered.read_text(encoding = "utf-8") == BODY


def test_a_fetch_that_does_not_match_the_digests_is_not_installed(monkeypatch, pinned_digests):
    """A wrong fetch raises instead of being published for a later call to accept."""

    def wrong_download(
        repo_id = None,
        *args,
        **kwargs,
    ):
        package = Path(kwargs["local_dir"])
        package.mkdir(parents = True, exist_ok = True)
        for name in _DEEPSEEK_OCR_MODULES:
            (package / name).write_text("VALUE = 'not what was pinned'\n", encoding = "utf-8")
        return str(package)

    monkeypatch.setattr("huggingface_hub.snapshot_download", wrong_download)

    with pytest.raises(RuntimeError, match = "does not match the pinned digests"):
        ensure_deepseek_ocr_source()

    assert not _deepseek_ocr_installed(third_party_source._deepseek_ocr_runtime())


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


def test_an_added_shadowing_package_is_rebuilt(monkeypatch, pinned_digests):
    """Digests on the listed files cannot see an ADDED file that wins the import.

    `modeling_deepseekocr/__init__.py` is imported in preference to
    `modeling_deepseekocr.py`, and an origin check passes it because it does sit inside
    the pinned root, so the contents have to be exactly what was installed.
    """
    fetched = []

    def counting_download(
        repo_id = None,
        *args,
        **kwargs,
    ):
        fetched.append(repo_id)
        return _fake_download(repo_id, *args, **kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", counting_download)
    source = ensure_deepseek_ocr_source()

    shadow = source / _DEEPSEEK_OCR_PACKAGE / "modeling_deepseekocr"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("VALUE = 'shadow'\n", encoding = "utf-8")

    assert _deepseek_ocr_installed(source) is False
    ensure_deepseek_ocr_source()
    assert len(fetched) == 2
    assert not shadow.exists()


def test_a_bare_added_directory_is_also_rebuilt(monkeypatch, pinned_digests):
    """A directory needs no __init__.py to be importable, so presence is enough."""
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_download)
    source = ensure_deepseek_ocr_source()

    (source / _DEEPSEEK_OCR_PACKAGE / "conversation").mkdir()

    assert _deepseek_ocr_installed(source) is False


def test_the_install_leaves_only_the_pinned_files(monkeypatch, pinned_digests):
    """The hub's own metadata directory is removed, so the exact-contents rule holds."""

    def download_with_metadata(
        repo_id = None,
        *args,
        **kwargs,
    ):
        result = _fake_download(repo_id, *args, **kwargs)
        metadata = Path(kwargs["local_dir"]) / ".cache" / "huggingface"
        metadata.mkdir(parents = True, exist_ok = True)
        (metadata / "download").write_text("x", encoding = "utf-8")
        return result

    monkeypatch.setattr("huggingface_hub.snapshot_download", download_with_metadata)

    source = ensure_deepseek_ocr_source()

    names = {entry.name for entry in (source / _DEEPSEEK_OCR_PACKAGE).iterdir()}
    assert names == {"__init__.py", *_DEEPSEEK_OCR_MODULES}


def test_generated_bytecode_does_not_invalidate_the_install(monkeypatch, pinned_digests):
    """Python writes __pycache__ on the first import.

    Treating it as an unexpected entry made the predicate go False immediately after a
    successful import, so every later run re-downloaded and an offline run failed on a
    source it had already installed and used.
    """
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_download)
    source = ensure_deepseek_ocr_source()

    for directory in (source, source / _DEEPSEEK_OCR_PACKAGE):
        cache = directory / "__pycache__"
        cache.mkdir()
        (cache / "modeling_deepseekocr.cpython-313.pyc").write_bytes(b"\x00")

    assert _deepseek_ocr_installed(source) is True

    def refuse(*args, **kwargs):
        raise AssertionError("bytecode must not trigger a re-download")

    monkeypatch.setattr("huggingface_hub.snapshot_download", refuse)
    assert ensure_deepseek_ocr_source() == source


def test_a_sibling_planted_in_the_import_root_is_rebuilt(monkeypatch, pinned_digests):
    """The root is on sys.path, and the origin check only looks at deepseek_ocr.*.

    The modelling code imports addict, so a planted addict.py beside the package would
    be imported by it with nothing objecting.
    """
    fetched = []

    def counting_download(
        repo_id = None,
        *args,
        **kwargs,
    ):
        fetched.append(repo_id)
        return _fake_download(repo_id, *args, **kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", counting_download)
    source = ensure_deepseek_ocr_source()

    planted = source / "addict.py"
    planted.write_text("VALUE = 'planted'\n", encoding = "utf-8")

    assert _deepseek_ocr_installed(source) is False
    ensure_deepseek_ocr_source()
    assert len(fetched) == 2
    assert not planted.exists()
