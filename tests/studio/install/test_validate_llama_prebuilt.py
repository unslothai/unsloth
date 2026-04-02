import importlib.util
import sys
from pathlib import Path


import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "validate-llama-prebuilt.py"

if not MODULE_PATH.is_file():
    pytest.skip(
        f"validate-llama-prebuilt.py not present at {MODULE_PATH}",
        allow_module_level = True,
    )

SPEC = importlib.util.spec_from_file_location("validate_llama_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATE
SPEC.loader.exec_module(VALIDATE)


def test_build_local_approved_checksums_uses_staged_upstream_tag(
    tmp_path: Path, monkeypatch
):
    stage_dir = tmp_path / "release-1"
    stage_dir.mkdir()
    asset_path = stage_dir / "app-test-linux-x64-cuda12-newer.tar.gz"
    asset_path.write_bytes(b"bundle")
    sibling_checksums = stage_dir / VALIDATE.installer.DEFAULT_PUBLISHED_SHA256_ASSET
    sibling_checksums.write_text(
        """
{
  "schema_version": 1,
  "component": "llama.cpp",
  "release_tag": "release-1",
  "upstream_tag": "b9001",
  "source_commit": "deadbeef",
  "artifacts": {
    "llama.cpp-source-b9001.tar.gz": {
      "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "repo": "ggml-org/llama.cpp",
      "kind": "upstream-source"
    }
  }
}
        """.strip()
        + "\n",
        encoding = "utf-8",
    )
    asset = VALIDATE.LocalAsset(
        path = asset_path,
        tag = "test",
        name = asset_path.name,
        install_kind = "linux-cuda",
        source_kind = "app-bundle",
        native_runnable = True,
        bundle_profile = "cuda12-newer",
        runtime_line = "cuda12",
    )

    checksums = VALIDATE.build_local_approved_checksums(
        asset,
        allow_network_source_hash = False,
    )

    assert checksums.release_tag == "release-1"
    assert checksums.upstream_tag == "b9001"
    assert "llama.cpp-source-b9001.tar.gz" in checksums.artifacts
    assert "llama.cpp-source-test.tar.gz" not in checksums.artifacts


def test_validate_native_asset_passes_release_tag_and_upstream_tag(
    tmp_path: Path, monkeypatch
):
    stage_dir = tmp_path / "release-7"
    stage_dir.mkdir()
    asset_path = stage_dir / "app-test-linux-x64-cuda12-newer.tar.gz"
    asset_path.write_bytes(b"bundle")
    sibling_checksums = stage_dir / VALIDATE.installer.DEFAULT_PUBLISHED_SHA256_ASSET
    sibling_checksums.write_text(
        """
{
  "schema_version": 1,
  "component": "llama.cpp",
  "release_tag": "release-7",
  "upstream_tag": "b9007",
  "source_commit": "deadbeef",
  "artifacts": {
    "llama.cpp-source-b9007.tar.gz": {
      "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "repo": "ggml-org/llama.cpp",
      "kind": "upstream-source"
    }
  }
}
        """.strip()
        + "\n",
        encoding = "utf-8",
    )
    asset = VALIDATE.LocalAsset(
        path = asset_path,
        tag = "test",
        name = asset_path.name,
        install_kind = "linux-cuda",
        source_kind = "app-bundle",
        native_runnable = True,
        bundle_profile = "cuda12-newer",
        runtime_line = "cuda12",
    )

    host = VALIDATE.installer.HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    monkeypatch.setattr(VALIDATE.installer, "detect_host", lambda: host)
    monkeypatch.setattr(
        VALIDATE.installer,
        "download_validation_model",
        lambda probe_path, cache_path: probe_path.write_bytes(b"probe"),
    )

    captured = {}

    def fake_validate_prebuilt_attempts(
        attempts,
        host,
        install_dir,
        work_dir,
        probe_path,
        *,
        requested_tag,
        llama_tag,
        release_tag,
        approved_checksums,
        initial_fallback_used = False,
        existing_install_dir = None,
    ):
        captured["requested_tag"] = requested_tag
        captured["llama_tag"] = llama_tag
        captured["release_tag"] = release_tag
        staging_dir = VALIDATE.installer.create_install_staging_dir(install_dir)
        return attempts[0], staging_dir, False

    monkeypatch.setattr(
        VALIDATE.installer,
        "validate_prebuilt_attempts",
        fake_validate_prebuilt_attempts,
    )

    record = VALIDATE.validate_native_asset(
        asset,
        keep_temp = False,
        allow_network_source_hash = False,
    )

    assert record.status == "PASS"
    assert captured == {
        "requested_tag": "test",
        "llama_tag": "b9007",
        "release_tag": "release-7",
    }
