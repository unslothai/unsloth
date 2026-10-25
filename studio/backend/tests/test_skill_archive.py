# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Skill ZIP export/import; every refusal must leave nothing behind."""

import io
import json
import os
import stat
import tracemalloc
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import skills
from routes.skills import router
from .test_skills import _BOB_ID, _write_skill, isolated_skills, managed_accounts  # noqa: F401

_MANIFEST = (
    b"---\nname: packed\ndescription: Use this skill for archive tests.\n---\nInstructions\n"
)


def _zip(*members, compression = zipfile.ZIP_DEFLATED) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression) as archive:
        for name, content in members:
            archive.writestr(name, content)
    return output.getvalue()


def _symlink(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    return info


def _future_version(archive: bytes) -> bytes:
    at = archive.index(b"PK\x01\x02") + 6
    return archive[:at] + (99).to_bytes(2, "little") + archive[at + 2 :]


def _corrupt_member(archive: bytes) -> bytes:
    # Flip the stored CRC of the last member so reading it fails.
    at = archive.rindex(b"PK\x01\x02") + 16
    return archive[:at] + bytes(b ^ 0xFF for b in archive[at : at + 4]) + archive[at + 4 :]


def _files(skill_dir) -> dict[str, bytes]:
    return {
        path.relative_to(skill_dir).as_posix(): path.read_bytes()
        for path in skill_dir.rglob("*")
        if path.is_file()
    }


def test_an_exported_skill_imports_back_with_the_same_files(isolated_skills):
    home, _ = isolated_skills
    skill_dir = _write_skill(
        home, "agents", "packed", frontmatter = "license: Apache-2.0\nmetadata:\n  author: tester"
    )
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "guide.md").write_bytes(b"# Guide\n")
    (skill_dir / "run.sh").write_bytes(b"echo hi\n")
    (skill_dir / "run.sh").chmod(0o755)
    before = _files(skill_dir)

    name, archive = skills.export_skill("packed", home = home)
    infos = zipfile.ZipFile(io.BytesIO(archive)).infolist()
    assert name == "packed"
    assert [info.filename for info in infos] == [
        "packed/SKILL.md",
        "packed/run.sh",
        "packed/references/guide.md",
    ]
    assert {info.date_time for info in infos} == {(1980, 1, 1, 0, 0, 0)}

    skills.delete_skill("packed", home = home)
    [record] = skills.import_skill_archive(archive, home = home)["imported"]

    assert (record["name"], record["source"], record["enabled"]) == ("packed", "agents", True)
    assert (record["license"], record["metadata"]) == ("Apache-2.0", {"author": "tester"})
    assert _files(skill_dir) == before
    if os.name == "posix":
        assert os.access(skill_dir / "run.sh", os.X_OK)
        assert not os.access(skill_dir / "SKILL.md", os.X_OK)


@pytest.mark.parametrize("prefix", ["", "my-skill-main/"])
def test_import_takes_skill_md_at_the_top_or_inside_one_folder(isolated_skills, prefix):
    home, _ = isolated_skills
    archive = _zip(
        (f"{prefix}scripts/", b""),
        (f"{prefix}SKILL.md", _MANIFEST),
        (f"{prefix}scripts/run.py", b"print('hi')\n"),
        (f"{prefix}.DS_Store", b"junk"),
        ("__MACOSX/._SKILL.md", b"junk"),
    )

    imported = skills.import_skill_archive(archive, home = home)["imported"]
    assert [record["name"] for record in imported] == ["packed"]
    assert _files(home / ".agents" / "skills" / "packed") == {
        "SKILL.md": _MANIFEST,
        "scripts/run.py": b"print('hi')\n",
    }


@pytest.mark.parametrize(
    "archive,reason",
    [
        pytest.param(
            _zip(("SKILL.md", _MANIFEST), ("../escape.md", b"x")), "inside", id = "traversal"
        ),
        pytest.param(
            _zip(("SKILL.md", _MANIFEST), ("/tmp/escape.md", b"x")), "inside", id = "absolute"
        ),
        pytest.param(_zip(("SKILL.md", _MANIFEST), ("C:\\escape.md", b"x")), "inside", id = "drive"),
        pytest.param(
            _zip(("SKILL.md", _MANIFEST), (_symlink("link.md"), b"/etc")), "symbolic", id = "symlink"
        ),
        pytest.param(
            _zip(("one/SKILL.md", _MANIFEST), ("README.md", b"x")),
            "'README.md' has none",
            id = "stray-file",
        ),
        pytest.param(_zip(("README.md", b"x")), "'README.md' has none", id = "no-manifest"),
        pytest.param(_zip(), "a skill folder", id = "empty"),
        pytest.param(b"not a zip", "ZIP file", id = "not-a-zip"),
        pytest.param(
            _future_version(_zip(("SKILL.md", _MANIFEST))), "ZIP file", id = "future-version"
        ),
        pytest.param(
            _zip(("SKILL.md", _MANIFEST), compression = zipfile.ZIP_LZMA),
            "stored or deflated",
            id = "lzma",
        ),
        pytest.param(
            _corrupt_member(_zip(("SKILL.md", _MANIFEST), ("ref.md", b"x" * 100))),
            "damaged",
            id = "bad-crc",
        ),
        pytest.param(
            _zip(("SKILL.md", _MANIFEST), ("big.bin", b"x" * (skills.MAX_SKILL_FILE_BYTES + 1))),
            "big.bin exceeds",
            id = "oversized-member",
        ),
        pytest.param(
            _zip(("SKILL.md", _MANIFEST), ("Guide.md", b"a"), ("guide.md", b"b")),
            "more than once",
            id = "case-duplicates",
        ),
    ],
)
def test_import_refuses_an_unsafe_archive_and_leaves_nothing(isolated_skills, archive, reason):
    home, _ = isolated_skills
    with pytest.raises(skills.SkillError, match = reason):
        skills.import_skill_archive(archive, home = home)
    assert not (home / ".agents").exists()


def _manifest(name: str) -> bytes:
    return f"---\nname: {name}\ndescription: The {name} skill.\n---\nInstructions\n".encode()


@pytest.mark.parametrize("wrapper", ["", "skills/"])
def test_import_takes_a_folder_of_skills(isolated_skills, wrapper):
    home, _ = isolated_skills
    archive = _zip(
        (f"{wrapper}sandbox-scout/SKILL.md", _manifest("sandbox-scout")),
        (f"{wrapper}train-replay/SKILL.md", _manifest("train-replay")),
        (f"{wrapper}train-replay/scripts/replay.py", b"print('replay')\n"),
        ("__MACOSX/._skills", b"junk"),
    )

    result = skills.import_skill_archive(archive, home = home)

    assert [record["name"] for record in result["imported"]] == ["sandbox-scout", "train-replay"]
    assert result["failed"] == []
    root = home / ".agents" / "skills"
    assert _files(root / "train-replay") == {
        "SKILL.md": _manifest("train-replay"),
        "scripts/replay.py": b"print('replay')\n",
    }
    assert _files(root / "sandbox-scout") == {"SKILL.md": _manifest("sandbox-scout")}


def test_a_folder_of_skills_reports_each_skill_it_could_not_install(isolated_skills):
    home, _ = isolated_skills
    existing = _write_skill(home, "agents", "train-replay", body = "Original")
    before = _files(existing)
    archive = _zip(
        ("broken/SKILL.md", b"no frontmatter"),
        ("sandbox-scout/SKILL.md", _manifest("sandbox-scout")),
        ("train-replay/SKILL.md", _manifest("train-replay")),
    )

    result = skills.import_skill_archive(archive, home = home)

    assert [record["name"] for record in result["imported"]] == ["sandbox-scout"]
    assert result["failed"] == [
        {"name": "broken", "error": "SKILL.md must start with YAML frontmatter."},
        {"name": "train-replay", "error": "Skill 'train-replay' already exists."},
    ]
    assert not (home / ".agents" / "skills" / "broken").exists()
    assert _files(existing) == before


def _nested_merges() -> str:
    lines = ["l0: &l0 {" + ", ".join(f"k{i}: v" for i in range(10)) + "}"]
    for level in range(1, 7):
        lines.append(f"l{level}: &l{level}")
        lines.append("  <<: [" + ", ".join([f"*l{level - 1}"] * 10) + "]")
        lines += [f"  x{level}_{i}: v" for i in range(10)]
    return "\n".join(lines)


def _wide_merge() -> str:
    pairs = ", ".join(f"k{i}: v" for i in range(3000))
    return f"a: &a {{{pairs}}}\nb:\n  <<: [" + ", ".join(["*a"] * 3000) + "]"


def _many_merges() -> str:
    pairs = ", ".join(f"k{i}: v" for i in range(5000))
    return f"a: &a {{{pairs}}}\n" + "\n".join(f"c{i}: {{<<: *a}}" for i in range(5000))


@pytest.mark.parametrize("frontmatter", [_nested_merges, _wide_merge, _many_merges])
def test_import_refuses_frontmatter_whose_merge_keys_explode(isolated_skills, frontmatter):
    home, _ = isolated_skills
    manifest = f"---\nname: bomb\ndescription: Looks normal.\n{frontmatter()}\n---\nHi\n".encode()

    tracemalloc.start()
    try:
        result = skills.import_skill_archive(_zip(("bomb/SKILL.md", manifest)), home = home)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert result == {
        "imported": [],
        "failed": [{"name": "bomb", "error": "SKILL.md contains invalid YAML frontmatter."}],
    }
    assert peak < 48 * 1024 * 1024
    assert not (home / ".agents" / "skills" / "bomb").exists()


def test_small_yaml_merges_still_load(isolated_skills):
    home, _ = isolated_skills
    manifest = (
        b"---\nname: merged\ndescription: Uses a merge.\n"
        b"base: &base {owner: team}\nmetadata:\n  <<: *base\n  tier: gold\n---\nHi\n"
    )

    result = skills.import_skill_archive(_zip(("merged/SKILL.md", manifest)), home = home)

    assert result["failed"] == []
    assert result["imported"][0]["metadata"] == {"owner": "team", "tier": "gold"}


def test_import_reads_backslash_separators_like_slashes(isolated_skills):
    home, _ = isolated_skills
    archive = _zip(
        ("my-skill\\SKILL.md", _MANIFEST),
        ("my-skill\\ref.md", b"ref"),
        ("my-skill\\.git\\config", b"junk"),
    )

    skills.import_skill_archive(archive, home = home)

    assert _files(home / ".agents" / "skills" / "packed") == {
        "SKILL.md": _MANIFEST,
        "ref.md": b"ref",
    }


def test_import_refuses_too_many_files_or_too_many_bytes(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    monkeypatch.setattr(skills, "MAX_SKILL_ARCHIVE_FILES", 2)
    with pytest.raises(skills.SkillError, match = "2-file limit"):
        skills.import_skill_archive(
            _zip(("SKILL.md", _MANIFEST), ("a.md", b"a"), ("b.md", b"b")), home = home
        )

    archive = _zip(("SKILL.md", _MANIFEST), ("big.md", b"x" * 100_000))
    monkeypatch.setattr(skills, "MAX_SKILL_ARCHIVE_BYTES", len(archive))
    with pytest.raises(skills.SkillError, match = "20 MB limit"):
        skills.import_skill_archive(archive, home = home)
    monkeypatch.setattr(skills, "MAX_SKILL_ARCHIVE_BYTES", len(archive) - 1)
    with pytest.raises(skills.SkillError, match = "20 MB limit"):
        skills.import_skill_archive(archive, home = home)
    assert not (home / ".agents").exists()


def test_a_failed_resource_write_removes_the_half_imported_skill(isolated_skills, monkeypatch):
    home, studio = isolated_skills
    studio.mkdir()
    (studio / "skill-overrides.json").write_text(json.dumps({"packed": False}), encoding = "utf-8")
    real_open = os.open

    def refuse_guide(path, flags, *args, **kwargs):
        if str(path).endswith("guide.md"):
            raise PermissionError("disk refused")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(skills.os, "open", refuse_guide)
    archive = _zip(("SKILL.md", _MANIFEST), ("references/guide.md", b"x"))
    result = skills.import_skill_archive(archive, home = home)

    assert result["failed"] == [{"name": "packed", "error": "Could not import skill 'packed'."}]

    assert not (home / ".agents" / "skills" / "packed").exists()
    assert skills._load_overrides() == {"packed": False}


def test_a_managed_account_imports_into_its_own_workspace(managed_accounts):
    from utils.account_context import run_as

    home, studio, alice, bob = managed_accounts
    archive = _zip(("SKILL.md", _MANIFEST), ("notes.md", b"notes"))

    [record] = run_as(bob, skills.import_skill_archive, archive)["imported"]

    assert record["path"] == "skills/packed/SKILL.md"
    assert (studio / "accounts" / _BOB_ID / "skills" / "packed" / "notes.md").is_file()
    assert not (home / ".agents" / "skills" / "packed").exists()
    with pytest.raises(skills.SkillNotFoundError):
        run_as(alice, skills.export_skill, "packed")


def test_export_refuses_a_file_name_import_would_refuse(isolated_skills):
    home, _ = isolated_skills
    skill_dir = _write_skill(home, "agents", "packed")
    (skill_dir / "con.md").write_text("x", encoding = "utf-8")

    with pytest.raises(skills.SkillError, match = "reserved device name"):
        skills.export_skill("packed", home = home)


def test_export_refuses_names_that_differ_only_in_case(isolated_skills):
    home, _ = isolated_skills
    skill_dir = _write_skill(home, "agents", "packed")
    (skill_dir / "Guide.md").write_text("a", encoding = "utf-8")
    try:
        (skill_dir / "guide.md").write_text("b", encoding = "utf-8")
    except FileExistsError:
        pass
    if len(list(skill_dir.glob("[Gg]uide.md"))) < 2:
        pytest.skip(reason = "case-insensitive filesystem")

    with pytest.raises(skills.SkillError, match = "more than once"):
        skills.export_skill("packed", home = home)


@pytest.mark.parametrize("kind", ["file", "folder"])
def test_export_refuses_a_linked_resource(isolated_skills, kind):
    home, _ = isolated_skills
    skill_dir = _write_skill(home, "agents", "packed")
    outside = home.parent / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding = "utf-8")
    target = outside if kind == "folder" else outside / "secret.txt"
    try:
        (skill_dir / "leak").symlink_to(target, target_is_directory = kind == "folder")
    except (OSError, NotImplementedError):
        pytest.skip(reason = "symlinks are unavailable on this platform")

    with pytest.raises(skills.SkillError, match = "symbolic links"):
        skills.export_skill("packed", home = home)


def test_authenticated_import_and_export_routes(isolated_skills, monkeypatch):
    from routes import inference as inference_routes
    from routes import skills as skills_routes

    home, _ = isolated_skills
    monkeypatch.setattr(skills, "_owner_home", lambda: home)
    app = FastAPI()
    app.include_router(router, prefix = "/api/skills")
    upload = {"file": ("packed.zip", _zip(("SKILL.md", _MANIFEST)), "application/zip")}
    anonymous = TestClient(app)
    assert anonymous.post("/api/skills/import", files = upload).status_code in (401, 403)
    assert anonymous.get("/api/skills/packed/export").status_code in (401, 403)
    assert not (home / ".agents").exists()
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    client = TestClient(app)

    monkeypatch.setattr(inference_routes, "_AGENT_SKILLS_CACHE", {None: (float("inf"), [])})
    response = client.post("/api/skills/import", files = upload)
    assert response.status_code == 200, response.text
    imported = response.json()["imported"]
    assert [record["path"] for record in imported] == ["~/.agents/skills/packed/SKILL.md"]
    assert inference_routes._AGENT_SKILLS_CACHE == {}
    response = client.post("/api/skills/import", files = upload)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "imported": [],
        "failed": [{"name": "packed", "error": "Skill 'packed' already exists."}],
    }
    bad = {"file": ("bad.zip", b"not a zip", "application/zip")}
    assert client.post("/api/skills/import", files = bad).status_code == 400

    response = client.get("/api/skills/packed/export")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == 'attachment; filename="packed.zip"'
    assert response.headers["cache-control"] == "no-store"
    assert zipfile.ZipFile(io.BytesIO(response.content)).namelist() == ["packed/SKILL.md"]
    assert client.get("/api/skills/missing/export").status_code == 404

    monkeypatch.setattr(skills_routes, "MAX_SKILL_ARCHIVE_BYTES", 8)
    assert client.post("/api/skills/import", files = upload).status_code == 413
