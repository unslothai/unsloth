# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install health checks read each dist-info METADATA once per call, headers only, same answers."""

from __future__ import annotations

import email
import importlib.metadata
import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "studio" / "install_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "studio_install_manifest_single_read", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


im = _load_module()

README = "".join(f"Line {i}: a README line that looks like a header\n" for i in range(200))

# Shapes a parser could split differently once the body is gone.
HEADER_SHAPES = {
    "lf": "Metadata-Version: 2.1\nName: lf\nVersion: 1.0\nRequires-Dist: a>=1\n\n" + README,
    "crlf": "Name: crlf\r\nVersion: 1.0\r\nRequires-Dist: a\r\n\r\nName: body\r\n",
    "lf_after_crlf": "Name: m\r\nVersion: 1.0\r\n\nRequires-Dist: hidden\r\n\r\nbody",
    "crlf_after_lf": "Name: m\nVersion: 1.0\n\r\nRequires-Dist: hidden\n\nbody",
    "cr_only": "Name: cr\rVersion: 1.0\r\rRequires-Dist: hidden\n\nbody",
    "folded": "Name: f\nVersion: 1.0\nRequires-Dist: a;\n  python_version >= '3'\n\nbody\n\nmore",
    "no_separator": "Name: n\nVersion: 1.0\nnot a header\nVersion: 7\n\nbody",
    "leading_blank": "\nName: lb\nVersion: 1.0\n\nbody",
    "headers_only": "Name: h\nVersion: 1.0\n",
    "unterminated": "Name: u\nVersion: 1.0",
    "multipart": "Name: mp\nVersion: 1.0\nContent-Type: multipart/mixed; boundary=X\n\n--X\nName: in\n\nx\n--X--\n",
    "from_line": "From nobody\nName: fl\nVersion: 1.0\n\nbody",
    "duplicate_headers": "Name: d\nName: e\nVersion: 1.0\nVersion: 2.0\n\nbody",
    "empty": "",
}


def _write(
    site: pathlib.Path,
    dirname: str,
    metadata,
    raw: bool = False,
) -> pathlib.Path:
    info = site / dirname
    info.mkdir(parents = True)
    if metadata is not None:
        if raw:
            (info / "METADATA").write_bytes(metadata)
        else:
            # write_bytes, not write_text(newline=""): that argument is 3.10+.
            (info / "METADATA").write_bytes(metadata.encode("utf-8"))
    return info


def _md(
    name: str,
    version: str,
    extra: str = "",
) -> str:
    return f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{extra}\n{README}"


@pytest.fixture
def site(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    site.mkdir()
    _write(site, "unsloth-2026.10.1.dist-info", _md("unsloth", "2026.10.1"))
    _write(site, "unsloth_zoo-2026.10.1.dist-info", _md("unsloth_zoo", "2026.10.1"))
    _write(site, "demo-1.0.dist-info", _md("demo", "1.0", "Requires-Dist: dep>=1\n"))
    _write(site, "dep-2.0.dist-info", _md("dep", "2.0"))
    for index in range(20):
        _write(site, f"filler{index}-1.0.dist-info", _md(f"filler{index}", "1.0"))
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])
    return site


@pytest.fixture
def metadata_reads(monkeypatch):
    """Counts METADATA reads by directory, through the stdlib reader every path uses."""
    reads = []
    original = importlib.metadata.PathDistribution.read_text

    def counting(self, filename):
        if filename == "METADATA":
            reads.append(pathlib.Path(str(self._path)).name)
        return original(self, filename)

    monkeypatch.setattr(importlib.metadata.PathDistribution, "read_text", counting)
    return reads


@pytest.mark.parametrize("shape", sorted(HEADER_SHAPES))
def test_header_block_parses_to_the_same_headers(shape):
    text = HEADER_SHAPES[shape]
    full = email.message_from_string(text)
    cut = email.message_from_string(im._header_block(text))
    assert cut.items() == full.items()
    assert im._header_block(im._header_block(text)) == im._header_block(text)


@pytest.mark.parametrize("shape", sorted(HEADER_SHAPES))
def test_single_read_distribution_answers_like_the_stdlib(tmp_path, shape):
    site = tmp_path / "site"
    _write(site, "pkg-1.0.dist-info", HEADER_SHAPES[shape])
    (dist,) = list(importlib.metadata.distributions(path = [str(site)]))
    fresh = importlib.metadata.PathDistribution(dist._path)
    single = im._single_read(importlib.metadata.PathDistribution(dist._path))

    def answers(d):
        out = []
        for read in (
            lambda: d.metadata.get("Name"),
            lambda: d.metadata.get_all("Requires-Dist"),
            lambda: d.version,
            lambda: d.requires,
        ):
            try:
                out.append(("ok", read()))
            except Exception as exc:
                out.append(("raise", type(exc)))
        return out

    assert answers(single) == answers(fresh)
    assert answers(single) == answers(fresh)


def test_undecodable_body_still_reads_as_unreadable(tmp_path, monkeypatch):
    """The whole file is decoded as before: a bad byte in the README is not skipped."""
    site = tmp_path / "site"
    _write(site, "latin-1.0.dist-info", b"Name: latin\nVersion: 1.0\n\nZo\xeb\n", raw = True)
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])
    assert im.installed_versions("latin") == [""]
    assert im.installed_version_probe("latin") == ("", True)


def test_probe_reads_each_metadata_once(site, metadata_reads):
    assert im.installed_version_probe("unsloth", ("unsloth-zoo",)) == ("2026.10.1", False)
    assert metadata_reads, "the probe must read metadata"
    assert len(metadata_reads) == len(set(metadata_reads)), sorted(metadata_reads)


def test_violated_constraints_reads_each_metadata_once(site, tmp_path, metadata_reads):
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(
        "".join(f"filler{index}<2\n" for index in range(20)) + "demo>=2\nabsent>=1\n",
        encoding = "utf-8",
    )
    assert im.violated_constraints(constraints) == ["demo"]
    assert len(metadata_reads) == len(set(metadata_reads)), sorted(metadata_reads)


def test_dependency_index_reads_each_metadata_once(site, metadata_reads):
    index = im.installed_dependency_index()
    assert index["demo"] == ("1.0", ["dep>=1"])
    assert len(metadata_reads) == len(set(metadata_reads)), sorted(metadata_reads)


def test_pip_backup_lookup_reads_only_backup_directories(site, metadata_reads):
    _write(site, "~nsloth_zoo-2026.9.1.dist-info", _md("unsloth_zoo", "2026.9.1"))
    backups = im.pip_backup_metadata_paths("unsloth-zoo")
    assert [path.name for path in backups] == ["~nsloth_zoo-2026.9.1.dist-info"]
    assert set(metadata_reads) == {"~nsloth_zoo-2026.9.1.dist-info"}


def test_the_shared_scan_never_outlives_one_call(site):
    """The installer changes site-packages between calls: each call must see the tree afresh."""
    assert im.installed_version_probe("late") == ("", False)
    _write(site, "late-1.0.dist-info", _md("late", "1.0"))
    assert im.installed_version_probe("late") == ("1.0", False)
    _write(site, "~ate-0.9.dist-info", _md("late", "0.9"))
    assert im.installed_version_probe("late") == ("", True)
    assert im._METADATA_SCAN.get() is None


def test_monkeypatched_installed_versions_still_drives_violated_constraints(tmp_path, monkeypatch):
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("torch<3\n", encoding = "utf-8")
    monkeypatch.setattr(im, "installed_versions", lambda name: ["2.0", "2.1"])
    assert im.violated_constraints(constraints) == ["torch"]


def test_studio_deps_mirror_matches_install_manifest(tmp_path):
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from unsloth_cli import _studio_deps
    finally:
        sys.path.remove(str(REPO_ROOT))
    for text in HEADER_SHAPES.values():
        assert _studio_deps._header_block(text) == im._header_block(text)
    site = tmp_path / "site"
    _write(site, "pkg-1.0.dist-info", HEADER_SHAPES["crlf"])
    (dist,) = list(importlib.metadata.distributions(path = [str(site)]))
    mirrored = _studio_deps._single_read(dist)
    own = im._single_read(dist)
    assert type(mirrored).__name__ == type(own).__name__ == "_SingleReadDistribution"
    assert (mirrored.version, mirrored.requires) == (own.version, own.requires)


class _FakeDist:
    """A finder result whose Version cannot be read although its Name can."""

    def __init__(self, path: pathlib.Path, name: str):
        self._path = path
        self.metadata = {"Name": name}

    @property
    def version(self):
        raise KeyError("Version")


def test_an_unreadable_version_counts_only_under_its_matching_directory(tmp_path, monkeypatch):
    dists = [
        _FakeDist(tmp_path / "foo-1.0.dist-info", "foo"),
        _FakeDist(tmp_path / "bar-1.0.dist-info", "baz"),
    ]
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(tmp_path)])
    monkeypatch.setattr(importlib.metadata, "distributions", lambda **_kw: iter(dists))
    assert im.installed_versions("foo") == [""]
    assert im.installed_versions("bar") == []
    assert im.installed_versions("baz") == []
    assert im.installed_version_probe("foo") == ("", True)
    assert im.installed_version_probe("baz") == ("", False)


def test_a_lookup_reads_versions_only_for_the_name_it_asks_about(site, monkeypatch):
    """A Version-less record elsewhere stays silent: 3.12 / 3.13 warn when its version is read."""
    _write(site, "versionless-1.0.dist-info", "Metadata-Version: 2.1\nName: versionless\n\nbody")
    asked = []
    original = importlib.metadata.Distribution.version

    def recording(self):
        asked.append(pathlib.Path(str(self._path)).name)
        return original.fget(self)

    monkeypatch.setattr(importlib.metadata.Distribution, "version", property(recording))
    assert im.installed_version_probe("demo", ("unsloth-zoo",)) == ("1.0", False)
    assert sorted(set(asked)) == ["demo-1.0.dist-info", "unsloth_zoo-2026.10.1.dist-info"]
