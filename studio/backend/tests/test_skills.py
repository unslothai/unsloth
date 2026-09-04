# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import skills
from routes.skills import router


def _write_skill(
    home: Path,
    source: str,
    name: str,
    *,
    description: str = "Use this skill for testing.",
    frontmatter: str = "",
    body: str = "Instructions",
) -> Path:
    root = home / (".agents" if source == "agents" else ".claude") / "skills" / name
    root.mkdir(parents = True)
    extra = f"\n{frontmatter.rstrip()}" if frontmatter else ""
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}{extra}\n---\n{body}",
        encoding = "utf-8",
    )
    return root


@pytest.fixture
def isolated_skills(tmp_path, monkeypatch):
    home = tmp_path / "home"
    studio = tmp_path / "studio"
    home.mkdir()
    monkeypatch.setattr(skills, "studio_root", lambda: studio)
    return home, studio


def test_enabled_skill_cache_is_fresh_after_slow_discovery(monkeypatch):
    from routes import inference as inference_routes

    scans = 0

    def discover():
        nonlocal scans
        scans += 1
        return [{"name": "cached"}]

    clock = iter((10.0, 12.0, 12.5))
    monkeypatch.setattr(skills, "enabled_skills", discover)
    monkeypatch.setattr(inference_routes.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(inference_routes, "_AGENT_SKILLS_CACHE", (0.0, []))

    assert inference_routes._enabled_agent_skills() == [{"name": "cached"}]
    assert inference_routes._enabled_agent_skills() == [{"name": "cached"}]
    assert scans == 1


def test_discovers_both_roots_with_agents_precedence(isolated_skills):
    home, _ = isolated_skills
    _write_skill(home, "agents", "shared", description = "Agent copy")
    _write_skill(home, "claude", "claude-only")
    _write_skill(home, "claude", "shared", description = "Claude copy")

    records = skills.list_skills(home = home)

    assert [(item["name"], item["source"], item["shadowed"]) for item in records] == [
        ("shared", "agents", False),
        ("claude-only", "claude", False),
        ("shared", "claude", True),
    ]
    assert records[0]["description"] == "Agent copy"
    assert records[0]["enabled"] is True
    assert records[2]["shadowed_by"] == "agents"


@pytest.mark.parametrize(
    "directory,manifest",
    [
        ("wrong-dir", "---\nname: other\ndescription: valid\n---\n"),
        ("BadName", "---\nname: BadName\ndescription: valid\n---\n"),
        (
            "bad-metadata",
            "---\nname: bad-metadata\ndescription: valid\nmetadata:\n  version: 1\n---\n",
        ),
        ("no-description", "---\nname: no-description\n---\n"),
    ],
)
def test_invalid_skill_is_reported_without_hiding_valid_skills(
    isolated_skills, directory, manifest
):
    home, _ = isolated_skills
    invalid = home / ".agents" / "skills" / directory
    invalid.mkdir(parents = True)
    (invalid / "SKILL.md").write_text(manifest, encoding = "utf-8")
    _write_skill(home, "agents", "valid")

    records = skills.list_skills(home = home)

    invalid_record = next(item for item in records if item["name"] == directory)
    assert invalid_record["valid"] is False
    assert invalid_record["error"]
    assert next(item for item in records if item["name"] == "valid")["valid"] is True


@pytest.mark.skipif(os.name == "nt", reason = "Surrogate-escaped POSIX filenames are unavailable on Windows")
def test_non_utf8_directory_name_does_not_hide_valid_skills(isolated_skills):
    home, _ = isolated_skills
    root = home / ".agents" / "skills"
    root.mkdir(parents = True)
    os.mkdir(os.fsencode(root) + b"/bad-\xff")
    _write_skill(home, "agents", "valid")

    records = skills.list_skills(home = home)

    assert [record["name"] for record in records] == ["valid"]
    json.dumps(records, ensure_ascii = False).encode("utf-8")


def test_disable_override_persists_without_touching_or_falling_through(isolated_skills):
    home, studio = isolated_skills
    winner = _write_skill(home, "agents", "shared", body = "winner")
    _write_skill(home, "claude", "shared", body = "shadowed")
    before = (winner / "SKILL.md").read_bytes()

    updated = skills.set_skill_enabled("shared", False, home = home)

    assert updated["enabled"] is False
    assert skills.enabled_skills(home = home) == []
    assert json.loads((studio / "skill-overrides.json").read_text()) == {"shared": False}
    assert (winner / "SKILL.md").read_bytes() == before
    skills.set_skill_enabled("shared", True, home = home)
    assert json.loads((studio / "skill-overrides.json").read_text()) == {}


def test_read_resource_is_contained_utf8_and_paginated(isolated_skills):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "reader")
    resource = root / "references" / "guide.md"
    resource.parent.mkdir()
    resource.write_text("abcdef", encoding = "utf-8")

    page = skills.read_skill_resource("reader", "references/guide.md", 1, page_chars = 3, home = home)

    assert "Characters: 1-4 of 6" in page
    assert "\nbcd\n" in page
    assert "offset=4" in page
    skills.set_skill_enabled("reader", False, home = home)
    with pytest.raises(skills.SkillError, match = "disabled"):
        skills.read_skill_resource("reader", home = home)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason = "POSIX FIFOs are unavailable on this platform")
def test_read_resource_rejects_fifo_without_blocking(isolated_skills):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "reader")
    pipe = root / "pipe"
    os.mkfifo(pipe)
    finished = threading.Event()
    errors = []

    def read_pipe():
        try:
            skills.read_skill_resource("reader", "pipe", home = home)
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target = read_pipe, daemon = True)
    worker.start()
    completed_without_writer = finished.wait(1)
    if not completed_without_writer:
        with pipe.open("wb"):
            pass
    worker.join(1)

    assert completed_without_writer, "reading a FIFO waited for a writer"
    assert errors and isinstance(errors[0], skills.SkillError)
    assert "regular file" in str(errors[0])


def test_read_resource_rejects_escaping_symlink(isolated_skills):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "reader")
    outside = home / "secret.txt"
    outside.write_text("secret", encoding = "utf-8")
    try:
        (root / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        # Reason: Windows may deny symlink creation without Developer Mode.
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(skills.SkillError, match = "symbolic links"):
        skills.read_skill_resource("reader", "link.txt", home = home)
    with pytest.raises(skills.SkillError, match = "stay inside"):
        skills.read_skill_resource("reader", "../secret.txt", home = home)


def test_read_resource_rejects_link_swapped_during_open(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "reader")
    resource = root / "guide.md"
    resource.write_text("safe", encoding = "utf-8")
    outside = home / "secret.txt"
    outside.write_text("secret", encoding = "utf-8")
    original_open = os.open
    swapped = False

    def replacing_open(path, *args, **kwargs):
        nonlocal swapped
        if path == resource and not swapped:
            swapped = True
            path.unlink()
            try:
                path.symlink_to(outside)
            except (OSError, NotImplementedError):
                # Reason: Windows may deny symlink creation without Developer Mode.
                pytest.skip("symlinks are unavailable on this platform")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", replacing_open)
    with pytest.raises(skills.SkillError, match = "symbolic links"):
        skills.read_skill_resource("reader", "guide.md", home = home)


def test_read_resource_rejects_skill_root_swapped_after_selection(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "reader")
    (root / "guide.md").write_text("safe", encoding = "utf-8")
    outside = home / "outside"
    outside.mkdir()
    (outside / "guide.md").write_text("secret", encoding = "utf-8")
    original_root = home / "original-reader"
    original_selected_skill = skills._selected_skill

    def replacing_selected_skill(name, *, home = None):
        record, path = original_selected_skill(name, home = home)
        root.rename(original_root)
        try:
            root.symlink_to(outside, target_is_directory = True)
        except (OSError, NotImplementedError):
            # Reason: Windows may deny symlink creation without Developer Mode.
            pytest.skip("symlinks are unavailable on this platform")
        return record, path

    monkeypatch.setattr(skills, "_selected_skill", replacing_selected_skill)
    with pytest.raises(skills.SkillError, match = "symbolic links"):
        skills.read_skill_resource("reader", "guide.md", home = home)


def test_skill_directory_name_must_match_exactly(isolated_skills):
    home, _ = isolated_skills
    root = home / ".agents" / "skills" / "ｓｋｉｌｌ"
    root.mkdir(parents = True)
    (root / "SKILL.md").write_text(
        "---\nname: skill\ndescription: test\n---\n",
        encoding = "utf-8",
    )

    record = skills.list_skills(home = home)[0]

    assert record["valid"] is False
    assert "match its parent directory" in record["error"]


def test_catalog_is_bounded_at_complete_entries():
    candidates = [{"name": f"skill-{index}", "description": "x" * 300} for index in range(20)]

    catalog = skills.format_skill_catalog(candidates)

    assert len(catalog.encode("utf-8")) <= skills.MAX_SKILL_CATALOG_BYTES
    assert all(line.startswith("- skill-") for line in catalog.splitlines())


def test_catalog_skips_an_oversized_entry_without_hiding_later_skills():
    candidates = [
        {"name": "oversized", "description": "界" * 600},
        {"name": "usable", "description": "Use this skill."},
    ]

    catalog = skills.format_skill_catalog(candidates)

    assert "oversized" not in catalog
    assert catalog == "- usable: Use this skill."


def test_authenticated_list_and_toggle_routes(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    _write_skill(home, "agents", "api-skill")
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)

    app = FastAPI()
    app.include_router(router, prefix = "/api/skills")
    assert TestClient(app).get("/api/skills").status_code in (401, 403)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    client = TestClient(app)

    response = client.get("/api/skills")
    assert response.status_code == 200
    assert response.json()[0]["name"] == "api-skill"
    response = client.put("/api/skills/api-skill/enabled", json = {"enabled": False})
    assert response.status_code == 200
    assert response.json()["enabled"] is False

    from routes import inference as inference_routes

    monkeypatch.setattr(
        inference_routes, "_AGENT_SKILLS_CACHE", (float("inf"), [{"name": "stale"}])
    )
    response = client.put("/api/skills/api-skill/enabled", json = {"enabled": True})
    assert response.status_code == 200
    assert inference_routes._AGENT_SKILLS_CACHE == (0.0, [])
    assert client.put("/api/skills/api-skill/enabled", json = {"enabled": "false"}).status_code == 422


def test_read_skill_tool_registration_selection_and_prompt(isolated_skills, monkeypatch):
    import asyncio

    from core.inference import tools as tools_module
    from models.inference import ChatCompletionRequest
    from routes import inference as inference_routes

    home, _ = isolated_skills
    _write_skill(home, "agents", "guided", description = "Guide this task")
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)
    monkeypatch.setattr(inference_routes, "_enabled_agent_skills", skills.enabled_skills)
    payload = ChatCompletionRequest(
        model = "test",
        messages = [{"role": "user", "content": "hello"}],
        enabled_tools = [],
    )

    selected = asyncio.run(
        inference_routes._select_request_tools(payload, tools_on = True, mcp_allowed = False)
    )
    assert [tool["function"]["name"] for tool in selected] == ["read_skill"]
    assert tools_module.is_always_safe_tool("read_skill") is True
    result = tools_module.execute_tool("read_skill", {"name": "guided"})
    assert "Skill: guided" in result
    nudge = inference_routes._build_tool_action_nudge(tools = selected, model_name = "test")
    assert "- guided: Guide this task" in nudge
    assert "@skill-name" in nudge
    assert ":skill[...]" in nudge

    skills.set_skill_enabled("guided", False, home = home)
    selected = asyncio.run(
        inference_routes._select_request_tools(payload, tools_on = True, mcp_allowed = False)
    )
    assert selected == []


def test_read_skill_tool_keeps_pagination_consistent_with_tight_room(isolated_skills, monkeypatch):
    from core.inference import tools as tools_module

    home, _ = isolated_skills
    _write_skill(home, "agents", "paged", body = "x" * 12_000)
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)

    result = tools_module.execute_tool(
        "read_skill",
        {"name": "paged"},
        context_tokens = 4096,
        result_budget_tokens = 300,
    )

    assert "Resource continues. Call read_skill again" in result
    header = next(line for line in result.splitlines() if line.startswith("Characters:"))
    end = int(header.split("-")[1].split()[0])
    assert f"offset={end}." in result
    assert "truncated to" not in result
