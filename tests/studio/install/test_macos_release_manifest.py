"""Consumer-contract tests for the self-built macOS llama.cpp mirror release.

The daily `unsloth-macos-prebuilt` workflow on the unslothai/llama.cpp fork
publishes a macOS-only release whose `llama-prebuilt-manifest.json` /
`llama-prebuilt-sha256.json` must be selectable by this installer. These tests
pin that contract: given a manifest + checksum asset in the shape the workflow
emits, the installer parses the bundle, the source-hash shortcut is satisfied,
and both macOS slices resolve as published choices with stamped digests.

No network, no torch, no Mach-O toolchain -- the manifest bytes are synthesized
and download I/O is monkeypatched.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location(
    "studio_install_llama_prebuilt_macos_manifest", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
ILP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ILP
SPEC.loader.exec_module(ILP)

REPO = "unslothai/llama.cpp"
UPSTREAM_TAG = "b9999"
COMMIT = "25b1bc9f0a1b2c3d4e5f60718293a4b5c6d7e8f9"
RELEASE_TAG = f"llama-prebuilt-macos-{UPSTREAM_TAG}"
ARM64_ASSET = f"llama-{UPSTREAM_TAG}-bin-macos-arm64.tar.gz"
X64_ASSET = f"llama-{UPSTREAM_TAG}-bin-macos-x64.tar.gz"
EXACT_SOURCE_ASSET = f"llama.cpp-source-commit-{COMMIT}.tar.gz"


def _hex(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _source_fields():
    return {
        "source_repo": "ggml-org/llama.cpp",
        "source_repo_url": "https://github.com/ggml-org/llama.cpp",
        "source_ref_kind": "tag",
        "requested_source_ref": UPSTREAM_TAG,
        "resolved_source_ref": UPSTREAM_TAG,
        "source_commit": COMMIT,
        "source_commit_short": COMMIT[:7],
    }


def _manifest_bytes():
    manifest = {
        "schema_version": 1,
        "component": "llama.cpp",
        "upstream_repo": "ggml-org/llama.cpp",
        "upstream_tag": UPSTREAM_TAG,
        **_source_fields(),
        "artifacts": [
            {
                "asset_name": ARM64_ASSET,
                "install_kind": "macos-arm64",
                "bundle_profile": "macos-metal-arm64",
                "runtime_line": None,
                "coverage_class": None,
                "rank": 50,
            },
            {
                "asset_name": X64_ASSET,
                "install_kind": "macos-x64",
                "bundle_profile": "macos-cpu-x64",
                "runtime_line": None,
                "coverage_class": None,
                "rank": 50,
            },
        ],
    }
    return (json.dumps(manifest, indent = 2) + "\n").encode("utf-8")


def _checksums_payload(manifest_bytes, *, include_exact_source = True):
    artifacts = {
        ARM64_ASSET: {
            "sha256": _hex(ARM64_ASSET),
            "repo": REPO,
            "kind": "macos-arm64-app",
        },
        X64_ASSET: {"sha256": _hex(X64_ASSET), "repo": REPO, "kind": "macos-x64-app"},
        "llama-prebuilt-manifest.json": {
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "repo": REPO,
            "kind": "published-manifest",
        },
    }
    if include_exact_source:
        artifacts[EXACT_SOURCE_ASSET] = {
            "sha256": _hex(EXACT_SOURCE_ASSET),
            "repo": "ggml-org/llama.cpp",
            "kind": "exact-source",
        }
    return {
        "schema_version": 1,
        "component": "llama.cpp",
        "release_tag": RELEASE_TAG,
        "upstream_tag": UPSTREAM_TAG,
        **_source_fields(),
        "artifacts": artifacts,
    }


def _release_dict():
    names = [
        "llama-prebuilt-manifest.json",
        "llama-prebuilt-sha256.json",
        ARM64_ASSET,
        X64_ASSET,
        EXACT_SOURCE_ASSET,
    ]
    return {
        "tag_name": RELEASE_TAG,
        "assets": [
            {"name": n, "browser_download_url": f"https://example/{n}"} for n in names
        ],
    }


@pytest.fixture
def manifest_bytes(monkeypatch):
    data = _manifest_bytes()
    # parse_published_release_bundle downloads only the manifest URL.
    monkeypatch.setattr(ILP, "download_bytes", lambda url, **kw: data)
    monkeypatch.setattr(ILP, "auth_headers", lambda url: {})
    return data


def test_bundle_parses_both_slices(manifest_bytes):
    bundle = ILP.parse_published_release_bundle(REPO, _release_dict())
    assert bundle is not None
    assert bundle.upstream_tag == UPSTREAM_TAG
    assert sorted(a.install_kind for a in bundle.artifacts) == [
        "macos-arm64",
        "macos-x64",
    ]
    assert bundle.source_commit == COMMIT


def test_checksums_parse_and_source_shortcut(manifest_bytes):
    checksums = ILP.parse_approved_release_checksums(
        REPO, RELEASE_TAG, _checksums_payload(manifest_bytes)
    )
    assert ARM64_ASSET in checksums.artifacts
    assert X64_ASSET in checksums.artifacts
    # The exact-commit source archive satisfies validated_checksums_for_bundle
    # without the legacy llama.cpp-source-<tag>.tar.gz entry.
    assert ILP.exact_source_archive_hash(checksums) is not None


def test_manifest_sha_cross_check_matches(manifest_bytes):
    bundle = ILP.parse_published_release_bundle(REPO, _release_dict())
    checksums = ILP.parse_approved_release_checksums(
        REPO, RELEASE_TAG, _checksums_payload(manifest_bytes)
    )
    manifest_hash = checksums.artifacts.get(bundle.manifest_asset_name)
    assert manifest_hash is not None
    assert manifest_hash.sha256 == bundle.manifest_sha256


@pytest.mark.parametrize("kind", ["macos-arm64", "macos-x64"])
def test_published_choice_selectable_and_stamped(manifest_bytes, kind):
    bundle = ILP.parse_published_release_bundle(REPO, _release_dict())
    checksums = ILP.parse_approved_release_checksums(
        REPO, RELEASE_TAG, _checksums_payload(manifest_bytes)
    )
    choice = ILP.published_asset_choice_for_kind(bundle, kind)
    assert choice is not None
    assert choice.source_label == "published"
    assert choice.install_kind == kind
    stamped = ILP.apply_approved_hashes([choice], checksums)
    assert stamped
    assert stamped[0].expected_sha256 == checksums.artifacts[choice.name].sha256


def test_missing_source_hash_is_rejected(manifest_bytes):
    # No exact-source and no legacy source entry -> require_approved_source_hash
    # must reject, so we never silently ship a bundle with no approved source.
    checksums = ILP.parse_approved_release_checksums(
        REPO,
        RELEASE_TAG,
        _checksums_payload(manifest_bytes, include_exact_source = False),
    )
    assert ILP.exact_source_archive_hash(checksums) is None
    with pytest.raises(ILP.PrebuiltFallback):
        ILP.require_approved_source_hash(checksums, UPSTREAM_TAG)
