# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import io
import importlib
import py_compile
import shutil
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

import utils.third_party_source as source
import utils.utils as utils


def _run_git(repository: Path, *arguments: str) -> str:
    if shutil.which("git") is None:
        pytest.skip("git is required for this test")
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check = True,
        capture_output = True,
        text = True,
        encoding = "utf-8",
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "upstream"
    (repository / "sparktts" / "models").mkdir(parents = True)
    (repository / "sparktts" / "utils").mkdir()
    (repository / "sparktts" / "models" / "__init__.py").write_text("", encoding = "utf-8")
    (repository / "sparktts" / "utils" / "__init__.py").write_text("", encoding = "utf-8")
    (repository / ".gitignore").write_text("*.evil.py\n", encoding = "utf-8")
    tokenizer = repository / "sparktts" / "models" / "audio_tokenizer.py"
    tokenizer.write_text("VALUE = 'pinned'\n", encoding = "utf-8")
    (repository / "sparktts" / "models" / "origin_injector.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from types import ModuleType\n"
        "injected = ModuleType('sparktts.injected')\n"
        "injected.__file__ = str(Path(__file__).resolve().parents[2] / 'outside.py')\n"
        "sys.modules['sparktts.injected'] = injected\n",
        encoding = "utf-8",
    )
    (repository / "sparktts" / "utils" / "audio.py").write_text(
        "VALUE = 'audio'\n",
        encoding = "utf-8",
    )
    _run_git(repository, "init", "--quiet")
    _run_git(repository, "config", "user.name", "Test")
    _run_git(repository, "config", "user.email", "test@example.com")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "--quiet", "-m", "pinned")
    pinned = _run_git(repository, "rev-parse", "HEAD")
    tokenizer.write_text("VALUE = 'newer'\n", encoding = "utf-8")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "--quiet", "-m", "newer")
    return repository, pinned


def _configure(monkeypatch, tmp_path: Path, repository: Path, revision: str) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setattr(
        source,
        "SPARK_TTS_SOURCE",
        source.PinnedSource(
            name = "Spark-TTS",
            package = "sparktts",
            repository = str(repository),
            revision = revision,
            required_files = (
                "sparktts/models/audio_tokenizer.py",
                "sparktts/utils/audio.py",
            ),
            generated_files = (("sparktts/__init__.py", ""),),
        ),
    )
    monkeypatch.setattr(source, "cache_root", lambda: cache)
    monkeypatch.setattr(utils, "hf_env_offline", lambda: False)
    return cache


def _sealed_spec(spec, checkout: Path, runtime: Path):
    return replace(
        spec,
        source_tree_digest = source._manifest_digest(
            source._filesystem_source_manifest(checkout, spec)
        ),
        runtime_tree_digest = source._manifest_digest(source._runtime_manifest(runtime, spec)),
    )


def test_manifest_digest_protocol_is_canonical():
    manifest = {
        "sparktts/z.py": "0" * 64,
        "sparktts/a.py": "f" * 64,
    }

    assert source._manifest_digest(manifest) == (
        "96ce9c498a9112e52fa1ef4f93e33003c2fdb26e0f1dfdc9408e701ecd66c173"
    )


def _write_source_archive(path: Path, root: str, files: dict[str, bytes]) -> None:
    with tarfile.open(path, mode = "w:gz") as bundle:
        for relative, content in files.items():
            member = tarfile.TarInfo(f"{root}/{relative}")
            member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))


def test_fresh_sealed_source_installs_from_archive_without_git(monkeypatch, tmp_path):
    revision = "1" * 40
    repository = "https://github.com/example/Fixture"
    root = f"Fixture-{revision}"
    files = {
        "README.md": b"not installed\n",
        "sparktts/models/audio_tokenizer.py": b"VALUE = 'archive'\n",
        "sparktts/utils/audio.py": b"VALUE = 'audio'\n",
    }
    archive = tmp_path / "source.tar.gz"
    _write_source_archive(archive, root, files)
    source_manifest = {
        relative: hashlib.sha256(content).hexdigest()
        for relative, content in files.items()
        if relative.startswith("sparktts/")
    }
    runtime_manifest = {
        **source_manifest,
        "sparktts/__init__.py": hashlib.sha256(b"").hexdigest(),
    }
    spec = source.PinnedSource(
        name = "Fixture",
        package = "sparktts",
        repository = repository,
        revision = revision,
        required_files = (
            "sparktts/models/audio_tokenizer.py",
            "sparktts/utils/audio.py",
        ),
        generated_files = (("sparktts/__init__.py", ""),),
        source_tree_digest = source._manifest_digest(source_manifest),
        runtime_tree_digest = source._manifest_digest(runtime_manifest),
        archive_url = archive.as_uri(),
    )
    cache = tmp_path / "cache"
    monkeypatch.setattr(source, "cache_root", lambda: cache)
    monkeypatch.setattr(utils, "hf_env_offline", lambda: False)
    monkeypatch.setattr(
        source,
        "_git",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fresh archive provisioning must not invoke Git")
        ),
    )

    runtime = source.ensure_pinned_source(spec)

    assert source._valid_runtime(runtime, spec)
    assert (runtime / "sparktts" / "models" / "audio_tokenizer.py").read_bytes() == (
        files["sparktts/models/audio_tokenizer.py"]
    )
    assert not (runtime / "README.md").exists()


def test_archive_download_enforces_deadline_with_read1(monkeypatch, tmp_path):
    now = [0.0]
    calls = {"read": 0, "read1": 0, "timeout": None}

    class SlowResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            calls["read"] += 1
            raise AssertionError("read1 must be used when available")

        def read1(self, size):
            calls["read1"] += 1
            now[0] += 0.6
            return b"x"

    def urlopen(request, *, timeout):
        calls["timeout"] = timeout
        return SlowResponse()

    spec = source.PinnedSource(
        name = "Fixture",
        package = "sparktts",
        repository = "https://github.com/example/Fixture",
        revision = "3" * 40,
        required_files = (),
    )
    monkeypatch.setattr(source.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(source, "_ARCHIVE_DOWNLOAD_DEADLINE_SECONDS", 1)
    monkeypatch.setattr(source.urllib.request, "urlopen", urlopen)

    with pytest.raises(RuntimeError, match = "Timed out downloading"):
        source._download_archive("https://example.invalid/archive.tar.gz", tmp_path / "out", spec)

    assert calls == {"read": 0, "read1": 2, "timeout": source._ARCHIVE_SOCKET_TIMEOUT_SECONDS}


@pytest.mark.parametrize(
    ("member_name", "member_type"),
    (
        ("Fixture-" + "2" * 40 + "/sparktts/../escape.py", tarfile.REGTYPE),
        ("Fixture-" + "2" * 40 + "\\sparktts\\escape.py", tarfile.REGTYPE),
        ("/Fixture-" + "2" * 40 + "/sparktts/escape.py", tarfile.REGTYPE),
        ("Fixture-" + "2" * 40 + "/sparktts/C:escape.py", tarfile.REGTYPE),
        ("Fixture-" + "2" * 40 + "/sparktts/link.py", tarfile.SYMTYPE),
    ),
)
def test_source_archive_rejects_unsafe_members(monkeypatch, tmp_path, member_name, member_type):
    revision = "2" * 40
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, mode = "w:gz") as bundle:
        member = tarfile.TarInfo(member_name)
        member.type = member_type
        if member_type == tarfile.REGTYPE:
            member.size = 1
            bundle.addfile(member, io.BytesIO(b"x"))
        else:
            member.linkname = "target"
            bundle.addfile(member)
    spec = source.PinnedSource(
        name = "Fixture",
        package = "sparktts",
        repository = "https://github.com/example/Fixture",
        revision = revision,
        required_files = (),
        source_tree_digest = "0" * 64,
        runtime_tree_digest = "0" * 64,
        archive_url = archive.as_uri(),
    )
    monkeypatch.setattr(source, "cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(utils, "hf_env_offline", lambda: False)

    with pytest.raises(RuntimeError, match = "archive"):
        source.ensure_pinned_source(spec)


def test_installs_the_exact_revision_instead_of_repository_head(monkeypatch, tmp_path):
    repository, pinned = _repository(tmp_path)
    cache = _configure(monkeypatch, tmp_path, repository, pinned)

    installed = source.ensure_spark_tts_source()

    assert (
        installed == (cache / "third-party-sources" / "Spark-TTS" / pinned / "runtime-v1").resolve()
    )
    assert (installed / "sparktts" / "models" / "audio_tokenizer.py").read_text(
        encoding = "utf-8"
    ) == "VALUE = 'pinned'\n"
    checkout = installed.parent / "source"
    assert not (checkout / "sparktts" / "__init__.py").exists()
    assert (installed / "sparktts" / "__init__.py").read_bytes() == b""
    assert _run_git(checkout, "rev-parse", "HEAD") == pinned
    assert _run_git(checkout, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"


def test_valid_cached_revision_stays_offline(monkeypatch, tmp_path):
    repository, pinned = _repository(tmp_path)
    _configure(monkeypatch, tmp_path, repository, pinned)
    expected = source.ensure_spark_tts_source()
    checkout = expected.parent / "source"
    monkeypatch.setattr(
        source,
        "SPARK_TTS_SOURCE",
        _sealed_spec(source.SPARK_TTS_SOURCE, checkout, expected),
    )

    def reject_git(*args, **kwargs):
        raise AssertionError("valid sealed runtime must not invoke Git")

    monkeypatch.setattr(source, "_git", reject_git)
    monkeypatch.setattr(utils, "hf_env_offline", lambda: True)

    assert source.ensure_spark_tts_source() == expected


def test_sealed_checkout_reconstructs_runtime_offline_without_git(monkeypatch, tmp_path):
    repository, pinned = _repository(tmp_path)
    _configure(monkeypatch, tmp_path, repository, pinned)
    installed = source.ensure_spark_tts_source()
    checkout = installed.parent / "source"
    monkeypatch.setattr(
        source,
        "SPARK_TTS_SOURCE",
        _sealed_spec(source.SPARK_TTS_SOURCE, checkout, installed),
    )
    shutil.rmtree(installed)
    monkeypatch.setattr(utils, "hf_env_offline", lambda: True)
    monkeypatch.setattr(
        source,
        "_git",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sealed checkout reconstruction must not invoke Git")
        ),
    )

    rebuilt = source.ensure_spark_tts_source()

    assert source._valid_runtime(rebuilt, source.SPARK_TTS_SOURCE)


def test_exact_legacy_source_migrates_offline_without_git(monkeypatch, tmp_path):
    repository, pinned = _repository(tmp_path)
    cache = _configure(monkeypatch, tmp_path, repository, pinned)
    installed = source.ensure_spark_tts_source()
    checkout = installed.parent / "source"
    spec = _sealed_spec(source.SPARK_TTS_SOURCE, checkout, installed)
    legacy = tmp_path / "Spark-TTS"
    shutil.copytree(checkout, legacy, ignore = shutil.ignore_patterns(".git"))
    shutil.rmtree(cache)
    monkeypatch.setattr(source, "SPARK_TTS_SOURCE", spec)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, "hf_env_offline", lambda: True)
    monkeypatch.setattr(
        source,
        "_git",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy migration must not invoke Git")
        ),
    )

    rebuilt = source.ensure_spark_tts_source()

    assert source._valid_runtime(rebuilt, spec)


def test_dirty_cached_revision_fails_closed_offline(monkeypatch, tmp_path):
    repository, pinned = _repository(tmp_path)
    _configure(monkeypatch, tmp_path, repository, pinned)
    installed = source.ensure_spark_tts_source()
    tokenizer = installed.parent / "source" / "sparktts" / "models" / "audio_tokenizer.py"
    tokenizer.write_text("VALUE = 'tampered'\n", encoding = "utf-8")
    monkeypatch.setattr(utils, "hf_env_offline", lambda: True)

    with pytest.raises(RuntimeError, match = "not cached.*offline"):
        source.ensure_spark_tts_source()

    assert tokenizer.read_text(encoding = "utf-8") == "VALUE = 'tampered'\n"


def test_failed_repair_keeps_existing_cache_untouched(monkeypatch, tmp_path):
    repository, pinned = _repository(tmp_path)
    _configure(monkeypatch, tmp_path, repository, pinned)
    installed = source.ensure_spark_tts_source()
    tokenizer = installed.parent / "source" / "sparktts" / "models" / "audio_tokenizer.py"
    tokenizer.write_text("VALUE = 'tampered'\n", encoding = "utf-8")
    real_git = source._git

    def fail_fetch(arguments, **kwargs):
        if "fetch" in arguments:
            raise RuntimeError("network failed")
        return real_git(arguments, **kwargs)

    monkeypatch.setattr(source, "_git", fail_fetch)

    with pytest.raises(RuntimeError, match = "network failed"):
        source.ensure_spark_tts_source()

    assert tokenizer.read_text(encoding = "utf-8") == "VALUE = 'tampered'\n"


def test_ignored_checkout_files_are_rejected_and_never_enter_runtime(monkeypatch, tmp_path):
    repository, pinned = _repository(tmp_path)
    _configure(monkeypatch, tmp_path, repository, pinned)
    installed = source.ensure_spark_tts_source()
    checkout = installed.parent / "source"
    ignored = checkout / "sparktts" / "payload.evil.py"
    ignored.write_text("VALUE = 'untrusted'\n", encoding = "utf-8")

    assert _run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert "sparktts/payload.evil.py" not in source._checkout_manifest(
        checkout,
        source.SPARK_TTS_SOURCE,
    )
    assert not source._valid_checkout(checkout, source.SPARK_TTS_SOURCE)

    repaired = source.ensure_spark_tts_source()

    assert repaired == installed
    assert not (repaired / "sparktts" / "payload.evil.py").exists()
    assert not (checkout / "sparktts" / "payload.evil.py").exists()


@pytest.mark.parametrize("index_flag", ("--assume-unchanged", "--skip-worktree"))
def test_index_flags_cannot_hide_modified_tracked_source(monkeypatch, tmp_path, index_flag):
    repository, pinned = _repository(tmp_path)
    _configure(monkeypatch, tmp_path, repository, pinned)
    installed = source.ensure_spark_tts_source()
    checkout = installed.parent / "source"
    relative = "sparktts/models/audio_tokenizer.py"
    tokenizer = checkout / relative
    _run_git(checkout, "update-index", index_flag, relative)
    tokenizer.write_text("VALUE = 'untrusted'\n", encoding = "utf-8")

    assert _run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all") == ""
    with pytest.raises(ValueError, match = "does not match.*pinned"):
        source._checkout_manifest(checkout, source.SPARK_TTS_SOURCE)
    assert not source._valid_checkout(checkout, source.SPARK_TTS_SOURCE)

    repaired = source.ensure_spark_tts_source()

    assert repaired == installed
    assert (checkout / relative).read_text(encoding = "utf-8") == "VALUE = 'pinned'\n"
    assert (repaired / relative).read_text(encoding = "utf-8") == "VALUE = 'pinned'\n"


def test_windows_drive_relative_components_are_rejected(monkeypatch, tmp_path):
    revision = "0" * 40
    base_spec = source.PinnedSource(
        name = "Drive-Relative",
        package = "sparktts",
        repository = str(tmp_path),
        revision = revision,
        required_files = (),
    )
    generated_spec = source.PinnedSource(
        name = base_spec.name,
        package = base_spec.package,
        repository = base_spec.repository,
        revision = base_spec.revision,
        required_files = (),
        generated_files = (("sparktts/C:payload.py", ""),),
    )

    with pytest.raises(ValueError, match = "Invalid generated path"):
        source._generated_file_contents(generated_spec)
    with pytest.raises(ValueError, match = "Invalid required path"):
        source._configured_package_paths(
            ("sparktts/C:payload.py",),
            base_spec,
            kind = "required",
        )
    with pytest.raises(ValueError, match = "Invalid omitted path"):
        source._configured_package_paths(
            ("sparktts/C:payload.py",),
            base_spec,
            kind = "omitted",
        )

    def drive_relative_tree(*args, **kwargs):
        return subprocess.CompletedProcess(
            ["git"],
            0,
            stdout = f"100644 blob {revision}\tsparktts/C:payload.py\0",
            stderr = "",
        )

    monkeypatch.setattr(source, "_git", drive_relative_tree)
    with pytest.raises(ValueError, match = "Invalid tracked path"):
        source._tracked_package_blobs(tmp_path, base_spec)


def test_import_replaces_a_module_from_outside_the_pinned_source(monkeypatch, tmp_path):
    repository, pinned = _repository(tmp_path)
    _configure(monkeypatch, tmp_path, repository, pinned)
    installed = source.ensure_spark_tts_source()
    untrusted = tmp_path / "untrusted"
    (untrusted / "sparktts" / "models").mkdir(parents = True)
    (untrusted / "sparktts" / "__init__.py").write_text("", encoding = "utf-8")
    (untrusted / "sparktts" / "models" / "__init__.py").write_text("", encoding = "utf-8")
    (untrusted / "sparktts" / "models" / "audio_tokenizer.py").write_text(
        "VALUE = 'untrusted'\n",
        encoding = "utf-8",
    )
    sys.path.insert(0, str(untrusted))
    importlib.invalidate_caches()
    try:
        untrusted_module = importlib.import_module("sparktts.models.audio_tokenizer")
        assert untrusted_module.VALUE == "untrusted"

        pinned_module = source.import_sparktts_module(
            "sparktts.models.audio_tokenizer",
            installed,
        )

        assert pinned_module.VALUE == "pinned"
        assert Path(pinned_module.__file__).resolve().is_relative_to(installed)
    finally:
        while str(untrusted) in sys.path:
            sys.path.remove(str(untrusted))
        while str(installed) in sys.path:
            sys.path.remove(str(installed))
        for name in list(sys.modules):
            if name == "sparktts" or name.startswith("sparktts."):
                sys.modules.pop(name, None)


def test_generated_init_prevents_a_later_regular_package_from_taking_over(monkeypatch, tmp_path):
    repository, pinned = _repository(tmp_path)
    _configure(monkeypatch, tmp_path, repository, pinned)
    installed = source.ensure_spark_tts_source()
    untrusted = tmp_path / "untrusted"
    marker = tmp_path / "outside-init-executed"
    (untrusted / "sparktts" / "models").mkdir(parents = True)
    (untrusted / "sparktts" / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding = "utf-8",
    )
    (untrusted / "sparktts" / "models" / "__init__.py").write_text("", encoding = "utf-8")
    (untrusted / "sparktts" / "models" / "audio_tokenizer.py").write_text(
        "VALUE = 'untrusted'\n",
        encoding = "utf-8",
    )
    sys.path.insert(0, str(untrusted))
    importlib.invalidate_caches()
    try:
        pinned_module = source.import_sparktts_module(
            "sparktts.models.audio_tokenizer",
            installed,
        )

        assert pinned_module.VALUE == "pinned"
        assert not marker.exists()
    finally:
        while str(untrusted) in sys.path:
            sys.path.remove(str(untrusted))
        source.deactivate_pinned_package("sparktts", installed)


def test_import_purges_unchecked_bytecode_before_loading(monkeypatch, tmp_path):
    repository, pinned = _repository(tmp_path)
    _configure(monkeypatch, tmp_path, repository, pinned)
    installed = source.ensure_spark_tts_source()
    tokenizer = installed / "sparktts" / "models" / "audio_tokenizer.py"
    marker = tmp_path / "unchecked-bytecode-executed"
    malicious_source = tmp_path / "malicious.py"
    malicious_source.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
        "VALUE = 'untrusted'\n",
        encoding = "utf-8",
    )
    bytecode = Path(importlib.util.cache_from_source(str(tokenizer)))
    bytecode.parent.mkdir(parents = True, exist_ok = True)
    py_compile.compile(
        str(malicious_source),
        cfile = str(bytecode),
        dfile = str(tokenizer),
        doraise = True,
        invalidation_mode = py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )

    assert source.ensure_spark_tts_source() == installed
    try:
        pinned_module = source.import_sparktts_module(
            "sparktts.models.audio_tokenizer",
            installed,
        )

        assert pinned_module.VALUE == "pinned"
        assert not marker.exists()
    finally:
        source.deactivate_pinned_package("sparktts", installed)


def test_import_rejects_any_loaded_package_module_from_outside(monkeypatch, tmp_path):
    repository, pinned = _repository(tmp_path)
    _configure(monkeypatch, tmp_path, repository, pinned)
    installed = source.ensure_spark_tts_source()

    with pytest.raises(RuntimeError, match = "sparktts.injected"):
        source.import_sparktts_module("sparktts.models.origin_injector", installed)

    assert not any(name == "sparktts" or name.startswith("sparktts.") for name in sys.modules)
    assert str(installed) not in sys.path


def test_runtime_overlay_omits_incompatible_outetts_modules(monkeypatch, tmp_path):
    repository = tmp_path / "outetts-upstream"
    required = (
        "outetts/models/config.py",
        "outetts/utils/preprocessing.py",
        "outetts/version/v3/audio_processor.py",
        "outetts/version/v3/prompt_processor.py",
    )
    omitted = (
        "outetts/interface.py",
        "outetts/models/gguf_model.py",
    )
    upstream_only = ("outetts/__init__.py", *omitted)
    for relative in (*required, *upstream_only):
        path = repository / relative
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(f"SOURCE = {relative!r}\n", encoding = "utf-8")
    _run_git(repository, "init", "--quiet")
    _run_git(repository, "config", "user.name", "Test")
    _run_git(repository, "config", "user.email", "test@example.com")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "--quiet", "-m", "pinned")
    revision = _run_git(repository, "rev-parse", "HEAD")
    cache = tmp_path / "outetts-cache"
    monkeypatch.setattr(source, "cache_root", lambda: cache)
    monkeypatch.setattr(utils, "hf_env_offline", lambda: False)
    spec = source.PinnedSource(
        name = "OuteTTS",
        package = "outetts",
        repository = str(repository),
        revision = revision,
        required_files = required,
        omitted_files = omitted,
        generated_files = (("outetts/__init__.py", ""),),
    )

    runtime = source.ensure_pinned_source(spec)

    checkout = runtime.parent / "source"
    assert all((runtime / relative).is_file() for relative in required)
    assert all(not (runtime / relative).exists() for relative in omitted)
    assert (runtime / "outetts" / "__init__.py").read_bytes() == b""
    assert all((checkout / relative).is_file() for relative in upstream_only)
    assert (checkout / "outetts" / "__init__.py").read_text(encoding = "utf-8") != ""
    assert _run_git(checkout, "status", "--porcelain") == ""


def _configure_dac_artifact(monkeypatch, tmp_path: Path, payload: bytes) -> Path:
    hub_cache = tmp_path / "hub"
    monkeypatch.setattr(source, "_DAC_SIZE", len(payload))
    monkeypatch.setattr(source, "_DAC_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        "utils.hf_cache_settings.active_hf_hub_cache",
        lambda: str(hub_cache),
    )
    return hub_cache


def test_dac_weights_use_immutable_revision_and_active_cache(monkeypatch, tmp_path):
    payload = b"pinned DAC weights"
    hub_cache = _configure_dac_artifact(monkeypatch, tmp_path, payload)
    downloaded = tmp_path / "downloaded.pth"
    downloaded.write_bytes(payload)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(utils, "hf_env_offline", lambda: False)

    result = source.ensure_dac_speech_weights()

    assert result == downloaded.resolve()
    assert calls == [
        {
            "repo_id": source._DAC_REPOSITORY,
            "filename": source._DAC_FILENAME,
            "revision": source._DAC_REVISION,
            "cache_dir": str(hub_cache),
            "local_files_only": False,
        }
    ]


def test_exact_legacy_dac_weights_migrate_to_active_cache_offline(monkeypatch, tmp_path):
    payload = b"legacy pinned DAC weights"
    hub_cache = _configure_dac_artifact(monkeypatch, tmp_path, payload)
    legacy = tmp_path / "legacy" / source._DAC_FILENAME
    legacy.parent.mkdir()
    legacy.write_bytes(payload)
    monkeypatch.setattr(utils, "hf_env_offline", lambda: True)
    calls = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: calls.append(kwargs))

    result = source.ensure_dac_speech_weights(legacy)

    assert result.is_relative_to(hub_cache)
    assert result.read_bytes() == payload
    assert calls == []
    legacy.write_bytes(b"tampered")
    assert source.ensure_dac_speech_weights(legacy) == result


def test_default_legacy_dac_path_matches_windows_appdata(monkeypatch, tmp_path):
    payload = b"Windows legacy DAC weights"
    hub_cache = _configure_dac_artifact(monkeypatch, tmp_path, payload)
    appdata = tmp_path / "AppData" / "Roaming"
    legacy = appdata / "outeai" / "dac" / source._DAC_FILENAME
    legacy.parent.mkdir(parents = True)
    legacy.write_bytes(payload)
    monkeypatch.setattr(source.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(appdata))
    calls = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: calls.append(kwargs))

    result = source.ensure_dac_speech_weights()

    assert result.is_relative_to(hub_cache)
    assert result.read_bytes() == payload
    assert calls == []


def test_dac_weight_hash_mismatch_fails_closed(monkeypatch, tmp_path):
    payload = b"expected"
    _configure_dac_artifact(monkeypatch, tmp_path, payload)
    downloaded = tmp_path / "downloaded.pth"
    downloaded.write_bytes(b"tampered")
    monkeypatch.setattr(utils, "hf_env_offline", lambda: False)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(downloaded))

    with pytest.raises(RuntimeError, match = "failed integrity validation"):
        source.ensure_dac_speech_weights(tmp_path / "missing.pth")
