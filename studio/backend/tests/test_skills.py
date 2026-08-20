# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import stat
import zipfile
from pathlib import Path
from typing import Optional

import pytest

from core.inference import skills


BUILTIN_SKILLS_ROOT = Path(skills.__file__).with_name("builtin_skills")

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
    monkeypatch.setattr(skills, "_builtin_skills_root", lambda: tmp_path / "no-builtins")


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


def test_bundled_skill_creator_is_discoverable_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(skills, "_builtin_skills_root", lambda: BUILTIN_SKILLS_ROOT)

    bundled = skills.list_skills()

    assert [skill["name"] for skill in bundled] == ["skill-creator"]
    assert bundled[0]["metadata"]["bundled"] == "true"
    assert "Code is enabled" in skills.read_skill_resource("skill-creator")
    assert skills.set_skill_enabled("skill-creator", False)["enabled"] is False
    with pytest.raises(skills.SkillError, match = "cannot be deleted"):
        skills.delete_skill("skill-creator")


def test_existing_skill_creator_keeps_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(skills, "_builtin_skills_root", lambda: BUILTIN_SKILLS_ROOT)
    custom = SKILL_MD.replace("name: unsloth", "name: skill-creator").replace(
        "Train and run models with Unsloth. Use for Unsloth workflows.",
        "Keep my existing custom skill.",
    )
    skills.import_skill_archive(
        _bundle(tmp_path / "custom.zip", {"skill-creator/SKILL.md": custom})
    )

    assert skills.list_skills()[0]["description"] == "Keep my existing custom skill."
    skills.delete_skill("skill-creator")
    assert skills.list_skills()[0]["metadata"]["bundled"] == "true"


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
        },
    )

    imported = skills.import_skill_archive(archive)

    assert imported["name"] == composed_name
    assert skills.set_skill_enabled(decomposed_name, False)["enabled"] is False
    assert skills.list_skills()[0]["name"] == composed_name
    skills.delete_skill(decomposed_name)
    assert skills.list_skills() == []


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
        },
    )
    skills.import_skill_archive(archive)

    manifest = skills.read_skill_resource("unsloth")
    reference = skills.read_skill_resource("unsloth", "references/config-reference.md")

    assert "Resource: SKILL.md" in manifest
    assert "Read [the configuration reference]" in manifest
    assert reference.endswith("Config\nUse bf16.\n")
    skills.set_skill_enabled("unsloth", False)
    assert not any(skill["enabled"] for skill in skills.list_skills())
    with pytest.raises(skills.SkillError, match = "disabled"):
        skills.read_skill_resource("unsloth")


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


def test_tool_routes_offer_the_loader_only_for_enabled_skills(tmp_path: Path):
    from core.inference.tools import READ_SKILL_TOOL, is_always_safe_tool
    from routes import inference

    selected = asyncio.run(inference._filter_unavailable_skill_tool([READ_SKILL_TOOL]))
    assert selected == []

    archive = _bundle(tmp_path / "skill.zip", {"unsloth/SKILL.md": SKILL_MD})
    skills.import_skill_archive(archive)
    selected = asyncio.run(inference._filter_unavailable_skill_tool([READ_SKILL_TOOL]))

    assert selected[0]["function"]["name"] == "read_skill"
    assert selected[0]["function"]["parameters"]["properties"]["offset"]["minimum"] == 0
    assert "- unsloth: Train and run models with Unsloth." in selected[0]["function"]["description"]
    assert "read_skill" in inference._ANTHROPIC_UNPROMPTED_SAFE_TOOLS
    assert is_always_safe_tool("read_skill") is True


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
