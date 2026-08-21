# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import errno
import os
import stat
import struct
import zipfile
from pathlib import Path
from typing import Optional

import pytest

from core.inference import skills


SKILL_MD = """---
name: unsloth
description: Train and run models with Unsloth. Use for Unsloth workflows.
compatibility: Requires an Unsloth installation
metadata:
  author: unslothai
  version: "1.0"
allowed-tools: Read Bash(python:*)
---

Unsloth instructions

Read [the configuration reference](references/config-reference.md) when needed.
"""


@pytest.fixture(autouse = True)
def isolated_skills_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(skills, "studio_root", lambda: tmp_path)


def _bundle(
    path: Path,
    entries: dict[str, str],
    *,
    symlink: Optional[str] = None,
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "SKILL.md")
    return path


def _corrupt_compressed_member(
    path: Path,
    entries: dict[str, str],
    member: str,
    compression: int = zipfile.ZIP_DEFLATED,
    corruption_offset: int = 0,
) -> Path:
    with zipfile.ZipFile(path, "w", compression = compression) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    with zipfile.ZipFile(path) as archive:
        entry = archive.getinfo(member)
        with path.open("r+b") as handle:
            handle.seek(entry.header_offset)
            header = handle.read(30)
            filename_length, extra_length = struct.unpack_from("<HH", header, 26)
            handle.seek(
                entry.header_offset + 30 + filename_length + extra_length + corruption_offset
            )
            handle.write(b"\x07")
    return path


def _mock_windows_reparse(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    reparse_point = 0x400
    real_lstat = skills.os.lstat

    class ReparsePointStatus:
        st_mode = stat.S_IFDIR
        st_file_attributes = reparse_point

    monkeypatch.setattr(
        skills.stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        reparse_point,
        raising = False,
    )
    monkeypatch.setattr(
        skills.os,
        "lstat",
        lambda path: ReparsePointStatus() if Path(path) == target else real_lstat(path),
    )


def test_create_skill_installs_markdown_and_text_resources(tmp_path: Path):
    created = skills.create_skill(
        "unsloth",
        SKILL_MD,
        [
            {"path": "references/config-reference.md", "content": "Use bf16.\n"},
            {"path": "scripts/check.py", "content": "print('ok')\n"},
            {"path": "assets/template.yaml", "content": "model: llama\n"},
        ],
    )

    assert created["name"] == "unsloth"
    assert created["enabled"] is True
    assert (tmp_path / "skills/unsloth/references/config-reference.md").read_text() == "Use bf16.\n"
    assert (tmp_path / "skills/unsloth/scripts/check.py").read_text() == "print('ok')\n"
    assert (tmp_path / "skills/unsloth/assets/template.yaml").read_text() == "model: llama\n"


def test_create_skill_tool_schema_and_scoped_execution(tmp_path: Path):
    from core.inference.tools import CREATE_SKILL_TOOL, execute_tool

    parameters = CREATE_SKILL_TOOL["function"]["parameters"]
    assert parameters["required"] == ["name", "skill_markdown"]
    assert parameters["properties"]["files"]["type"] == "array"
    assert parameters["properties"]["files"]["items"]["required"] == ["path", "content"]
    assert set(parameters["properties"]) == {"name", "skill_markdown", "files", "replace"}

    result = execute_tool(
        "create_skill",
        {
            "name": "unsloth",
            "skill_markdown": SKILL_MD,
            "files": [{"path": "references/guide.md", "content": "Guide\n"}],
        },
    )
    escaped = execute_tool(
        "create_skill",
        {
            "name": "escape",
            "skill_markdown": SKILL_MD.replace("name: unsloth", "name: escape"),
            "files": [{"path": "../escape.txt", "content": "escape"}],
        },
    )

    assert result == "Installed skill 'unsloth'. It will be available on the next turn."
    assert (tmp_path / "skills/unsloth/references/guide.md").read_text() == "Guide\n"
    assert escaped.startswith("Error: Archive contains unsafe path")
    assert not (tmp_path / "escape.txt").exists()


def test_create_skill_rejects_a_linked_skills_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from core.inference.tools import execute_tool

    root = tmp_path / "skills"
    real_lstat = skills.os.lstat

    def linked_root_lstat(path):
        if Path(path) == root:
            return os.stat_result((stat.S_IFLNK, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        return real_lstat(path)

    monkeypatch.setattr(skills.os, "lstat", linked_root_lstat)

    result = execute_tool(
        "create_skill",
        {"name": "unsloth", "skill_markdown": SKILL_MD},
        disable_sandbox = True,
    )

    assert result == "Error: Skills directory cannot be a symbolic link or reparse point."
    assert not root.exists()
    with pytest.raises(skills.SkillError, match = "symbolic link"):
        skills.list_skills()


def test_link_detector_recognizes_a_windows_reparse_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "skills"
    _mock_windows_reparse(monkeypatch, root)

    assert skills._is_linked_path(root) is True


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("unsloth", "missing or unsafe"),
        ("unsloth/SKILL.md", "contain SKILL.md"),
    ],
)
def test_installed_skills_reject_windows_reparse_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str, message: str
):
    skills.create_skill("unsloth", SKILL_MD)
    target = tmp_path / "skills" / relative_path
    _mock_windows_reparse(monkeypatch, target)

    with pytest.raises(skills.SkillError, match = message):
        skills._validate_installed_skill(tmp_path / "skills/unsloth")


@pytest.mark.skipif(os.name == "nt", reason = "POSIX symbolic-link regression")
def test_create_skill_allows_a_symlinked_studio_root_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    alias = tmp_path / "studio-alias"
    alias.symlink_to(real_root, target_is_directory = True)
    monkeypatch.setattr(skills, "studio_root", lambda: alias)

    skills.create_skill("unsloth", SKILL_MD)

    assert (real_root / "skills/unsloth/SKILL.md").is_file()


@pytest.mark.parametrize(
    ("name", "manifest", "files", "message"),
    [
        ("Bad Name", SKILL_MD, [], "lowercase letters"),
        ("unsloth", SKILL_MD.replace("name: unsloth", "name: other"), [], "must match"),
        ("unsloth", SKILL_MD, [{"path": "/tmp/out", "content": "x"}], "unsafe path"),
        ("unsloth", SKILL_MD, [{"path": ".", "content": "x"}], "unsafe path"),
        ("unsloth", SKILL_MD, [{"path": "./", "content": "x"}], "unsafe path"),
        ("unsloth", SKILL_MD, [{"path": "SKILL.md", "content": "x"}], "one SKILL.md"),
        (
            "unsloth",
            SKILL_MD,
            [
                {"path": "references/Guide.md", "content": "first"},
                {"path": "references/guide.md", "content": "second"},
            ],
            "duplicate path",
        ),
        (
            "unsloth",
            SKILL_MD,
            [
                {"path": "references", "content": "file"},
                {"path": "references/guide.md", "content": "nested"},
            ],
            "conflicting file paths",
        ),
    ],
)
def test_create_skill_rejects_invalid_names_and_paths(
    name: str, manifest: str, files: list[dict], message: str
):
    with pytest.raises(skills.SkillError, match = message):
        skills.create_skill(name, manifest, files)


def test_create_skill_rejects_recursive_yaml_frontmatter():
    nested = "[" * 500 + "0" + "]" * 500
    manifest = SKILL_MD.replace("metadata:\n", f"metadata:\n  nested: {nested}\n")

    with pytest.raises(skills.SkillError, match = "invalid YAML frontmatter"):
        skills.create_skill("unsloth", manifest)


@pytest.mark.parametrize("tagged_value", ["!!bool nope", "!!timestamp nope"])
def test_create_skill_rejects_invalid_yaml_tags(tagged_value: str):
    manifest = f"---\nname: unsloth\ndescription: {tagged_value}\n---\n"

    with pytest.raises(skills.SkillError, match = "invalid YAML frontmatter"):
        skills.create_skill("unsloth", manifest)


def test_import_rejects_recursive_root_yaml_frontmatter(tmp_path: Path):
    nested = "[" * 500 + "0" + "]" * 500
    manifest = SKILL_MD.replace("metadata:\n", f"metadata:\n  nested: {nested}\n")
    archive = _bundle(tmp_path / "recursive.zip", {"SKILL.md": manifest})

    with pytest.raises(skills.SkillError, match = "invalid YAML frontmatter"):
        skills.import_skill_archive(archive)


def test_import_rejects_invalid_root_yaml_timestamps(tmp_path: Path):
    manifest = SKILL_MD.replace("metadata:\n", "created: 2026-13-01\nmetadata:\n")
    archive = _bundle(tmp_path / "invalid-date.zip", {"SKILL.md": manifest})

    with pytest.raises(skills.SkillError, match = "invalid YAML frontmatter"):
        skills.import_skill_archive(archive)


def test_import_rejects_unpaired_surrogates(tmp_path: Path):
    manifest = SKILL_MD.replace(
        "description: Train and run models with Unsloth. Use for Unsloth workflows.",
        'description: "\\uD800"',
    )
    archive = _bundle(tmp_path / "surrogate.zip", {"SKILL.md": manifest})

    with pytest.raises(skills.SkillError, match = "valid Unicode"):
        skills.import_skill_archive(archive)


def test_create_skill_requires_explicit_replace_and_preserves_enabled_state(tmp_path: Path):
    from core.inference.tools import execute_tool

    skills.create_skill(
        "unsloth",
        SKILL_MD,
        [{"path": "references/old.md", "content": "old"}],
    )
    skills.set_skill_enabled("unsloth", False)
    replacement = SKILL_MD.replace("Train and run", "Fine-tune and run")

    with pytest.raises(skills.SkillError, match = "already installed"):
        skills.create_skill("unsloth", replacement)
    assert "Train and run models" in (tmp_path / "skills/unsloth/SKILL.md").read_text()
    with pytest.raises(skills.SkillError, match = "flag must be a boolean"):
        skills.create_skill("unsloth", replacement, replace = "true")

    result = execute_tool(
        "create_skill",
        {"name": "unsloth", "skill_markdown": replacement, "replace": True},
    )
    replaced = skills.list_skills()[0]

    assert result == "Installed skill 'unsloth'. It remains disabled."
    assert replaced["enabled"] is False
    assert "Fine-tune and run models" in (tmp_path / "skills/unsloth/SKILL.md").read_text()
    assert not (tmp_path / "skills/unsloth/references/old.md").exists()


def test_imports_pr_style_nested_agent_skill_bundle(tmp_path: Path):
    archive = _bundle(
        tmp_path / "pr-4443.zip",
        {
            "unsloth-pr/skills/unsloth/SKILL.md": SKILL_MD,
            "unsloth-pr/skills/unsloth/references/config-reference.md": "Config reference\n",
            "unsloth-pr/skills/unsloth/assets/train.yaml": "model: llama\n",
            "unsloth-pr/README.md": "repository file outside the skill bundle",
        },
    )

    imported = skills.import_skill_archive(archive)

    assert imported == {
        "name": "unsloth",
        "description": "Train and run models with Unsloth. Use for Unsloth workflows.",
        "compatibility": "Requires an Unsloth installation",
        "metadata": {"author": "unslothai", "version": "1.0"},
        "allowed_tools": "Read Bash(python:*)",
        "enabled": True,
    }
    assert (tmp_path / "skills/unsloth/assets/train.yaml").is_file()
    assert not (tmp_path / "skills/unsloth/README.md").exists()


def test_imports_nfkc_normalized_unicode_skill_names(tmp_path: Path):
    decomposed_name = "cafe\u0301"
    composed_name = "café"
    archive = _bundle(
        tmp_path / "unicode.zip",
        {
            f"{composed_name}/SKILL.md": SKILL_MD.replace(
                "name: unsloth", f"name: {decomposed_name}"
            ),
            f"{decomposed_name}/references/guide.md": "Unicode guide\n",
        },
    )

    imported = skills.import_skill_archive(archive)

    assert imported["name"] == composed_name
    assert skills.read_skill_resource(composed_name, "references/guide.md").endswith(
        "Unicode guide\n"
    )
    assert skills.set_skill_enabled(decomposed_name, False)["enabled"] is False
    assert skills.list_skills()[0]["name"] == composed_name
    skills.delete_skill(decomposed_name)
    assert skills.list_skills() == []


def test_rejects_skill_names_over_filesystem_component_limit(tmp_path: Path):
    name = "\U00010428" * 64
    archive = _bundle(
        tmp_path / "long-name.zip",
        {"SKILL.md": SKILL_MD.replace("name: unsloth", f"name: {name}")},
    )

    with pytest.raises(skills.SkillError, match = "255 UTF-8 bytes"):
        skills.import_skill_archive(archive)


def test_long_skill_names_keep_internal_temp_names_portable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    name = "\U00010428" * 63
    manifest = SKILL_MD.replace("name: unsloth", f"name: {name}")
    skills.create_skill(name, manifest)
    real_mkdtemp = skills.tempfile.mkdtemp
    real_replace = skills.os.replace

    def component_limited_mkdtemp(*args, **kwargs):
        prefix = kwargs.get("prefix", "")
        if len(prefix.encode("utf-8")) + 8 > 255:
            raise OSError(errno.ENAMETOOLONG, "file name too long")
        return real_mkdtemp(*args, **kwargs)

    def component_limited_replace(source, destination):
        if len(Path(destination).name.encode("utf-8")) > 255:
            raise OSError(errno.ENAMETOOLONG, "file name too long")
        return real_replace(source, destination)

    monkeypatch.setattr(skills.tempfile, "mkdtemp", component_limited_mkdtemp)
    monkeypatch.setattr(skills.os, "replace", component_limited_replace)

    skills.create_skill(
        name,
        manifest.replace("Train and run", "Fine-tune and run"),
        replace = True,
    )
    skills.delete_skill(name)

    assert skills.list_skills() == []


def test_rejects_resource_paths_over_filesystem_component_limit(tmp_path: Path):
    archive = _bundle(
        tmp_path / "long-resource.zip",
        {
            "unsloth/SKILL.md": SKILL_MD,
            f"unsloth/references/{'x' * 256}": "resource",
        },
    )

    with pytest.raises(skills.SkillError, match = "255 UTF-8 bytes"):
        skills.import_skill_archive(archive)


def test_rejects_archive_paths_over_filesystem_path_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(skills.os, "pathconf", lambda *_args: 256)
    nested_resource = "/".join(["segment"] * 40)
    archive = _bundle(
        tmp_path / "long-path.zip",
        {
            "unsloth/SKILL.md": SKILL_MD,
            f"unsloth/references/{nested_resource}/resource.txt": "resource",
        },
    )

    with pytest.raises(skills.SkillError, match = "filesystem path limit"):
        skills.import_skill_archive(archive)


def test_translates_filesystem_path_length_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archive_path = _bundle(
        tmp_path / "filesystem-limit.zip",
        {
            "unsloth/SKILL.md": SKILL_MD,
            "unsloth/references/resource.txt": "resource",
        },
    )
    original_open = zipfile.ZipFile.open

    def open_member(archive, name, *args, **kwargs):
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
        if filename.endswith("resource.txt"):
            raise OSError(errno.ENAMETOOLONG, "path too long")
        return original_open(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", open_member)
    with pytest.raises(skills.SkillError, match = "filesystem path limit"):
        skills.import_skill_archive(archive_path)


def test_rejects_windows_reserved_frontmatter_only_skill_names(tmp_path: Path):
    manifest = SKILL_MD.replace("name: unsloth", "name: con")
    archive = _bundle(tmp_path / "reserved.zip", {"SKILL.md": manifest})

    with pytest.raises(skills.SkillError, match = "Windows reserved"):
        skills.import_skill_archive(archive)


def test_repository_files_do_not_count_toward_bundle_limits(tmp_path: Path):
    entries = {"repo/skills/unsloth/SKILL.md": SKILL_MD}
    entries.update({f"repo/source/file-{index}.txt": "source" for index in range(1_100)})
    archive = _bundle(tmp_path / "repository.zip", entries)

    imported = skills.import_skill_archive(archive)

    assert imported["name"] == "unsloth"
    assert not (tmp_path / "skills/unsloth/source").exists()


def test_rejects_oversized_manifest_before_inflating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    archive_path = _bundle(
        tmp_path / "oversized.zip",
        {"unsloth/SKILL.md": SKILL_MD + "x" * skills.MAX_SKILL_MD_BYTES},
    )
    with zipfile.ZipFile(archive_path) as archive:
        monkeypatch.setattr(
            archive,
            "read",
            lambda *_args, **_kwargs: pytest.fail("oversized manifest was inflated"),
        )
        with pytest.raises(skills.SkillError, match = "512 KB"):
            skills._archive_source(archive)


def test_rejects_an_oversized_archive_resource(tmp_path: Path):
    archive = _bundle(
        tmp_path / "oversized-resource.zip",
        {
            "unsloth/SKILL.md": SKILL_MD,
            "unsloth/references/oversized.txt": "x" * (skills.MAX_SKILL_FILE_BYTES + 1),
        },
    )

    with pytest.raises(skills.SkillError, match = "2048 KB"):
        skills.import_skill_archive(archive)
    assert not (tmp_path / "skills/unsloth").exists()


def test_rejects_invalid_utf8_archive_filenames(tmp_path: Path):
    archive = _bundle(tmp_path / "invalid-name.zip", {"SKILL.md": SKILL_MD})
    payload = bytearray(archive.read_bytes())
    central_directory = payload.index(b"PK\x01\x02")
    flags = struct.unpack_from("<H", payload, central_directory + 8)[0]
    struct.pack_into("<H", payload, central_directory + 8, flags | 0x800)
    payload[central_directory + 46] = 0xFF
    archive.write_bytes(payload)

    with pytest.raises(skills.SkillError, match = "valid ZIP"):
        skills.import_skill_archive(archive)


@pytest.mark.parametrize(
    "member",
    ["unsloth/SKILL.md", "unsloth/references/config-reference.md"],
)
@pytest.mark.parametrize(
    ("compression", "corruption_offset"),
    [
        pytest.param(zipfile.ZIP_DEFLATED, 0, id = "deflate"),
        pytest.param(zipfile.ZIP_BZIP2, 0, id = "bzip2"),
        pytest.param(zipfile.ZIP_LZMA, 9, id = "lzma"),
        pytest.param(
            getattr(zipfile, "ZIP_ZSTANDARD", zipfile.ZIP_STORED),
            0,
            id = "zstandard",
            marks = pytest.mark.skipif(
                not hasattr(zipfile, "ZIP_ZSTANDARD"),
                reason = "Zstandard requires Python 3.14",
            ),
        ),
    ],
)
def test_translates_corrupt_compressed_streams(
    tmp_path: Path, member: str, compression: int, corruption_offset: int
):
    archive = _corrupt_compressed_member(
        tmp_path / "corrupt.zip",
        {
            "unsloth/SKILL.md": SKILL_MD,
            "unsloth/references/config-reference.md": "Use bf16.\n",
        },
        member,
        compression,
        corruption_offset,
    )

    with pytest.raises(skills.SkillError, match = "SKILL.md|valid ZIP"):
        skills.import_skill_archive(archive)
    assert not (tmp_path / "skills/unsloth").exists()


def test_rejects_case_insensitive_archive_collisions(tmp_path: Path):
    archive = _bundle(
        tmp_path / "collision.zip",
        {
            "unsloth/SKILL.md": SKILL_MD,
            "unsloth/references/Guide.md": "first",
            "unsloth/references/guide.md": "second",
        },
    )

    with pytest.raises(skills.SkillError, match = "duplicate path"):
        skills.import_skill_archive(archive)


def test_rejects_a_file_that_collides_with_the_skill_root(tmp_path: Path):
    archive = _bundle(
        tmp_path / "root-conflict.zip",
        {
            "repo/skills/unsloth": "file conflicts with the implicit directory",
            "repo/skills/unsloth/SKILL.md": SKILL_MD,
        },
    )

    with pytest.raises(skills.SkillError, match = "conflicting file paths"):
        skills.import_skill_archive(archive)


def test_rejects_unsupported_zip_compression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archive_path = _bundle(
        tmp_path / "unsupported.zip",
        {
            "unsloth/SKILL.md": SKILL_MD,
            "unsloth/references/config-reference.md": "Config reference\n",
        },
    )
    original_open = zipfile.ZipFile.open

    def open_member(archive, name, *args, **kwargs):
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
        if filename.endswith("config-reference.md"):
            raise NotImplementedError("compression type 93")
        return original_open(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", open_member)
    with pytest.raises(skills.SkillError, match = "unsupported compression"):
        skills.import_skill_archive(archive_path)


def test_progressively_reads_enabled_skill_resources(tmp_path: Path):
    archive = _bundle(
        tmp_path / "skill.zip",
        {
            "unsloth/SKILL.md": SKILL_MD,
            "unsloth/references/config-reference.md": "Config\nUse bf16.\n",
            "unsloth/ references/leading.md": "Leading space\n",
        },
    )
    skills.import_skill_archive(archive)

    manifest = skills.read_skill_resource("unsloth")
    whitespace_manifest = skills.read_skill_resource("unsloth", " \t")
    reference = skills.read_skill_resource("unsloth", "references/config-reference.md")
    backslash_reference = skills.read_skill_resource("unsloth", "references\\config-reference.md")
    leading = skills.read_skill_resource("unsloth", " references/leading.md")

    assert "Resource: SKILL.md" in manifest
    assert whitespace_manifest == manifest
    assert "Read [the configuration reference]" in manifest
    assert reference.endswith("Config\nUse bf16.\n")
    assert backslash_reference == reference
    assert leading.endswith("Leading space\n")
    skills.set_skill_enabled("unsloth", False)
    assert not any(skill["enabled"] for skill in skills.list_skills())
    with pytest.raises(skills.SkillError, match = "disabled"):
        skills.read_skill_resource("unsloth")
    with pytest.raises(skills.SkillError, match = "not installed"):
        skills.read_skill_resource("missing")


def test_large_resources_are_read_in_bounded_pages(tmp_path: Path):
    from core.inference.tools import execute_tool

    content = "x" * (skills.MAX_SKILL_PAGE_CHARS + 25)
    archive = _bundle(
        tmp_path / "skill.zip",
        {
            "unsloth/SKILL.md": SKILL_MD,
            "unsloth/references/large.txt": content,
        },
    )
    skills.import_skill_archive(archive)

    first = skills.read_skill_resource("unsloth", "references/large.txt")
    second = execute_tool(
        "read_skill",
        {
            "name": "unsloth",
            "resource": "references/large.txt",
            "offset": skills.MAX_SKILL_PAGE_CHARS,
        },
    )

    assert f"Characters: 0-{skills.MAX_SKILL_PAGE_CHARS}" in first
    assert f"offset={skills.MAX_SKILL_PAGE_CHARS}" in first
    page = first.split("\n\n", 1)[1].split("\n\nResource continues", 1)[0]
    assert page == content[: skills.MAX_SKILL_PAGE_CHARS]
    assert second.endswith("x" * 25)
    assert "Resource continues" not in second


def test_tool_pages_dense_resources_to_the_remaining_context_budget(tmp_path: Path):
    from core.inference.tools import execute_tool

    content = "😀" * 3_000
    archive = _bundle(
        tmp_path / "dense.zip",
        {
            "unsloth/SKILL.md": SKILL_MD,
            "unsloth/references/dense.txt": content,
        },
    )
    skills.import_skill_archive(archive)

    result = execute_tool(
        "read_skill",
        {"name": "unsloth", "resource": "references/dense.txt"},
        conversation_budget_tokens = 512,
    )

    assert len(result.encode("utf-8")) + 8 <= 512
    assert "Resource continues" in result


def test_tool_falls_back_when_the_exact_counter_returns_zero(tmp_path: Path):
    from core.inference.tools import execute_tool

    archive = _bundle(
        tmp_path / "dense.zip",
        {
            "unsloth/SKILL.md": SKILL_MD,
            "unsloth/references/dense.txt": "x" * 3_000,
        },
    )
    skills.import_skill_archive(archive)

    result = execute_tool(
        "read_skill",
        {"name": "unsloth", "resource": "references/dense.txt"},
        conversation_budget_tokens = 512,
        conversation_token_counter = lambda _text: 0,
    )

    assert len(result.encode("utf-8")) + 8 <= 512
    assert "Resource continues" in result


def test_rejects_an_enabled_catalog_over_the_context_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = _bundle(tmp_path / "first.zip", {"unsloth/SKILL.md": SKILL_MD})
    second_markdown = SKILL_MD.replace("name: unsloth", "name: helper").replace(
        "description: Train and run models with Unsloth. Use for Unsloth workflows.",
        "description: Another helper skill.",
    )
    second = _bundle(tmp_path / "second.zip", {"helper/SKILL.md": second_markdown})
    skills.import_skill_archive(first)
    monkeypatch.setattr(skills, "MAX_SKILL_CATALOG_BYTES", 80)

    with pytest.raises(skills.SkillError, match = "catalog exceeds"):
        skills.import_skill_archive(second)

    assert [skill["name"] for skill in skills.list_skills()] == ["unsloth"]

    skills.set_skill_enabled("unsloth", False)
    skills.import_skill_archive(second)
    with pytest.raises(skills.SkillError, match = "catalog exceeds"):
        skills.set_skill_enabled("unsloth", True)
    assert not next(
        skill["enabled"] for skill in skills.list_skills() if skill["name"] == "unsloth"
    )


def test_tool_routes_always_offer_create_and_gate_read_on_enabled_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from core.inference.tools import CREATE_SKILL_TOOL, READ_SKILL_TOOL, is_always_safe_tool
    from routes import inference

    selected = asyncio.run(
        inference._filter_unavailable_skill_tool([CREATE_SKILL_TOOL, READ_SKILL_TOOL])
    )
    assert [tool["function"]["name"] for tool in selected] == ["create_skill"]

    skills.create_skill("unsloth", SKILL_MD)
    selected = asyncio.run(
        inference._filter_unavailable_skill_tool([CREATE_SKILL_TOOL, READ_SKILL_TOOL])
    )

    assert [tool["function"]["name"] for tool in selected] == ["create_skill", "read_skill"]
    assert selected[1]["function"]["parameters"]["properties"]["offset"]["minimum"] == 0
    assert "- unsloth: Train and run models with Unsloth." in selected[1]["function"]["description"]
    assert "read_skill" in inference._ANTHROPIC_UNPROMPTED_SAFE_TOOLS
    assert is_always_safe_tool("read_skill") is True
    assert is_always_safe_tool("create_skill") is False

    monkeypatch.setattr(
        skills,
        "list_skills",
        lambda: (_ for _ in ()).throw(OSError("storage unavailable")),
    )
    selected = asyncio.run(
        inference._filter_unavailable_skill_tool([READ_SKILL_TOOL, CREATE_SKILL_TOOL])
    )
    assert [tool["function"]["name"] for tool in selected] == ["create_skill"]


def test_token_count_selection_keeps_the_create_skill_schema():
    from models.inference import ChatCountTokensRequest
    from routes import inference

    payload = ChatCountTokensRequest(
        messages = [{"role": "user", "content": "Create a skill."}],
        enable_tools = True,
        enabled_tools = ["create_skill"],
    )

    selected = asyncio.run(
        inference._select_request_tools(payload, tools_on = True, mcp_allowed = False)
    )

    assert [tool["function"]["name"] for tool in selected] == ["create_skill"]


def test_corrupt_registry_fails_closed_without_overwriting_state(tmp_path: Path):
    from core.inference.tools import READ_SKILL_TOOL
    from routes import inference

    archive = _bundle(tmp_path / "skill.zip", {"unsloth/SKILL.md": SKILL_MD})
    skills.import_skill_archive(archive)
    skills.set_skill_enabled("unsloth", False)
    registry = tmp_path / "skills/.registry.json"
    registry.write_text("{bad", encoding = "utf-8")

    with pytest.raises(skills.SkillError, match = "registry"):
        skills.list_skills()
    assert asyncio.run(inference._filter_unavailable_skill_tool([READ_SKILL_TOOL])) == []
    with pytest.raises(skills.SkillError, match = "registry"):
        skills.set_skill_enabled("unsloth", True)
    assert registry.read_text(encoding = "utf-8") == "{bad"


def test_replace_keeps_the_existing_enabled_state(tmp_path: Path):
    first = _bundle(tmp_path / "first.zip", {"unsloth/SKILL.md": SKILL_MD})
    second = _bundle(
        tmp_path / "second.zip",
        {"unsloth/SKILL.md": SKILL_MD.replace("Train and run", "Fine-tune and run")},
    )
    skills.import_skill_archive(first)
    skills.set_skill_enabled("unsloth", False)

    replaced = skills.import_skill_archive(second, replace = True)

    assert replaced["enabled"] is False
    assert skills.list_skills()[0]["description"].startswith("Fine-tune")


def test_replace_rolls_back_when_registry_cannot_be_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = _bundle(tmp_path / "first.zip", {"unsloth/SKILL.md": SKILL_MD})
    replacement_markdown = SKILL_MD.replace("Train and run", "Fine-tune and run")
    second = _bundle(tmp_path / "second.zip", {"unsloth/SKILL.md": replacement_markdown})
    skills.import_skill_archive(first)

    def fail_save(_registry: dict[str, bool]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(skills, "_save_registry", fail_save)
    with pytest.raises(OSError, match = "disk full"):
        skills.import_skill_archive(second, replace = True)

    assert "Train and run models" in skills.read_skill_resource("unsloth")
    assert "Fine-tune and run models" not in skills.read_skill_resource("unsloth")


def test_replacement_reports_cleanup_failure_and_uses_a_fresh_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    skills.create_skill("unsloth", SKILL_MD)
    real_rmtree = skills.shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".backup-"):
            if kwargs.get("ignore_errors"):
                return
            raise PermissionError("file is in use")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(skills.shutil, "rmtree", fail_backup_cleanup)
    with pytest.raises(PermissionError, match = "file is in use"):
        skills.create_skill(
            "unsloth",
            SKILL_MD.replace("Train and run", "Fine-tune and run"),
            replace = True,
        )

    monkeypatch.setattr(skills.shutil, "rmtree", real_rmtree)
    skills.create_skill(
        "unsloth",
        SKILL_MD.replace("Train and run", "Serve and run"),
        replace = True,
    )

    assert "Serve and run models" in skills.read_skill_resource("unsloth")


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (
            {
                "unsloth/SKILL.md": SKILL_MD,
                "unsloth/../escape.txt": "escape",
            },
            "unsafe path",
        ),
        (
            {
                "unsloth/SKILL.md": SKILL_MD,
                "unsloth/C:../escape.txt": "escape",
            },
            "unsafe path",
        ),
        (
            {
                "unsloth/SKILL.md": SKILL_MD,
                "unsloth/references/guide.md.": "unsafe on Windows",
            },
            "unsafe path",
        ),
        (
            {
                "unsloth/SKILL.md": SKILL_MD,
                "unsloth/references/COM¹.txt": "reserved on Windows",
            },
            "unsafe path",
        ),
        (
            {
                "one/SKILL.md": SKILL_MD.replace("name: unsloth", "name: one"),
                "two/SKILL.md": SKILL_MD.replace("name: unsloth", "name: two"),
            },
            "exactly one SKILL.md",
        ),
        (
            {"different/SKILL.md": SKILL_MD},
            "must match its parent directory",
        ),
    ],
)
def test_rejects_invalid_archives(tmp_path: Path, entries: dict[str, str], message: str):
    archive = _bundle(tmp_path / "invalid.zip", entries)

    with pytest.raises(skills.SkillError, match = message):
        skills.import_skill_archive(archive)


def test_rejects_symbolic_links(tmp_path: Path):
    archive = _bundle(
        tmp_path / "symlink.zip",
        {"unsloth/SKILL.md": SKILL_MD},
        symlink = "unsloth/references/link.md",
    )

    with pytest.raises(skills.SkillError, match = "symbolic links"):
        skills.import_skill_archive(archive)


def test_resource_paths_cannot_escape_the_bundle(tmp_path: Path):
    archive = _bundle(tmp_path / "skill.zip", {"unsloth/SKILL.md": SKILL_MD})
    skills.import_skill_archive(archive)

    with pytest.raises(skills.SkillError, match = "stay inside"):
        skills.read_skill_resource("unsloth", "../secret.txt")


def test_delete_rolls_back_when_registry_cannot_be_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    archive = _bundle(tmp_path / "skill.zip", {"unsloth/SKILL.md": SKILL_MD})
    skills.import_skill_archive(archive)

    def fail_save(_registry: dict[str, bool]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(skills, "_save_registry", fail_save)

    with pytest.raises(OSError, match = "disk full"):
        skills.delete_skill("unsloth")

    assert (tmp_path / "skills/unsloth/SKILL.md").is_file()
    assert skills.list_skills()[0]["name"] == "unsloth"


def test_delete_reports_quarantine_cleanup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    skills.create_skill("unsloth", SKILL_MD)
    real_rmtree = skills.shutil.rmtree

    def fail_quarantine_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".delete-"):
            if kwargs.get("ignore_errors"):
                return
            raise PermissionError("file is in use")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(skills.shutil, "rmtree", fail_quarantine_cleanup)

    with pytest.raises(PermissionError, match = "file is in use"):
        skills.delete_skill("unsloth")
