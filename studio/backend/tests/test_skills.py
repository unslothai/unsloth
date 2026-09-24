# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import sys
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


@pytest.fixture(autouse = True)
def _reset_tool_request_budget():
    # execute_tool never resets these, so the pagination test's tight budget must not leak.
    from core.inference import tools

    context_token = tools._REQUEST_CONTEXT_TOKENS.set(tools._UNSET_CONTEXT_TOKENS)
    budget_token = tools._REQUEST_RESULT_BUDGET.set(None)
    yield
    tools._REQUEST_CONTEXT_TOKENS.reset(context_token)
    tools._REQUEST_RESULT_BUDGET.reset(budget_token)


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
    monkeypatch.setattr(inference_routes, "_AGENT_SKILLS_CACHE", {})

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
    assert records[0]["linked"] is False
    assert records[2]["shadowed_by"] == "agents"


def test_bundled_skill_creator_is_enabled_and_user_override_wins(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
        ("bundled", Path(skills.__file__).with_name("bundled_skills")),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)

    creator = next(record for record in skills.list_skills() if record["name"] == "skill-creator")
    assert creator["source"] == "bundled"
    # Bundled skills ship disabled: a fresh install must not carry skill tools into every chat.
    assert creator["enabled"] is False
    assert skills.enabled_skills() == []
    with pytest.raises(skills.SkillError, match = "disabled"):
        skills.read_skill_resource("skill-creator")
    # Enabling one is an explicit override that survives a re-list; disabling clears it again.
    assert skills.set_skill_enabled("skill-creator", True)["enabled"] is True
    assert skills._load_overrides() == {"skill-creator": True}
    assert "create_skill" in skills.read_skill_resource("skill-creator")
    assert skills.set_skill_enabled("skill-creator", False)["enabled"] is False
    assert skills._load_overrides() == {}

    _write_skill(home, "agents", "skill-creator", description = "User override")
    records = [record for record in skills.list_skills() if record["name"] == "skill-creator"]
    assert [(record["source"], record["shadowed"]) for record in records] == [
        ("agents", False),
        ("bundled", True),
    ]


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


@pytest.mark.skipif(
    os.name == "nt" or sys.platform == "darwin",
    reason = "Windows has no surrogate-escaped filenames and APFS rejects non-UTF-8 names",
)
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


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason = "POSIX FIFOs are unavailable on this platform"
)
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
        # The descriptor-relative walk opens the bare component name against a dir_fd.
        if path in (resource, resource.name) and not swapped:
            swapped = True
            resource.unlink()
            try:
                resource.symlink_to(outside)
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
        record, path, identity = original_selected_skill(name, home = home)
        root.rename(original_root)
        try:
            root.symlink_to(outside, target_is_directory = True)
        except (OSError, NotImplementedError):
            # Reason: Windows may deny symlink creation without Developer Mode.
            pytest.skip("symlinks are unavailable on this platform")
        return record, path, identity

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


def test_create_skill_writes_valid_manifest_without_overwriting(isolated_skills):
    home, _ = isolated_skills

    record = skills.create_skill(
        "release-notes",
        "Draft concise release notes.",
        "# Workflow\n\n1. Inspect the diff.\n2. Summarize user-visible changes.",
        home = home,
    )

    assert record["name"] == "release-notes"
    assert record["source"] == "agents"
    created = home / ".agents" / "skills" / "release-notes" / "SKILL.md"
    metadata, real_dir = skills._validate_skill_dir(created.parent)
    assert metadata["description"] == "Draft concise release notes."
    assert real_dir == created.parent
    with pytest.raises(skills.SkillError, match = "already exists"):
        skills.create_skill("release-notes", "Different", "Do something else.", home = home)
    assert "Different" not in created.read_text(encoding = "utf-8")


@pytest.mark.parametrize("name", ("../escape", "Bad Name", "con"))
def test_create_skill_rejects_unsafe_names(isolated_skills, name):
    home, _ = isolated_skills
    with pytest.raises(skills.SkillError):
        skills.create_skill(name, "Description", "Instructions", home = home)


def test_create_skill_rejects_a_linked_agents_ancestor(isolated_skills):
    home, _ = isolated_skills
    outside = home.parent / "outside"
    outside.mkdir()
    try:
        (home / ".agents").symlink_to(outside, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(skills.SkillError, match = "unsafe"):
        skills.create_skill("escaped", "Description", "Instructions", home = home)

    assert not (outside / "skills").exists()


def test_create_skill_tool_invalidates_the_inference_cache(isolated_skills, monkeypatch):
    from core.inference import tools as tools_module
    from routes import inference as inference_routes

    home, _ = isolated_skills
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        inference_routes, "_AGENT_SKILLS_CACHE", {None: (float("inf"), [{"name": "stale"}])}
    )

    result = tools_module.execute_tool(
        "create_skill",
        {"name": "fresh", "description": "Description", "instructions": "Instructions"},
    )

    assert "Created Agent Skill 'fresh'" in result
    assert inference_routes._AGENT_SKILLS_CACHE == {}


def test_create_skill_tool_does_not_commit_when_override_clear_fails(isolated_skills, monkeypatch):
    from core.inference import tools as tools_module

    home, studio = isolated_skills
    studio.mkdir()
    (studio / "skill-overrides.json").write_text('{"blocked":false}', encoding = "utf-8")
    monkeypatch.setenv("HOME", str(home))

    def deny_override_write(_overrides):
        raise PermissionError("read-only overrides")

    monkeypatch.setattr(skills, "_save_overrides", deny_override_write)
    result = tools_module.execute_tool(
        "create_skill",
        {"name": "blocked", "description": "Description", "instructions": "Instructions"},
    )

    assert result.startswith("Error:")
    assert not (home / ".agents" / "skills" / "blocked").exists()


def test_catalog_is_bounded_at_complete_entries():
    candidates = [{"name": f"skill-{index}", "description": "x" * 300} for index in range(20)]

    catalog = skills.format_skill_catalog(candidates)

    *listed, marker = catalog.splitlines()
    assert len("\n".join(listed).encode("utf-8")) <= skills.MAX_SKILL_CATALOG_BYTES
    assert all(line.startswith("- skill-") for line in listed)
    assert (
        marker
        == f"- {20 - len(listed)} more enabled skills not listed; mention one as @skill-name."
    )
    large = skills.format_skill_catalog(candidates, budget = skills.LARGE_SKILL_CATALOG_BYTES)
    assert len(large.splitlines()) > len(listed)
    assert "more enabled skills" not in skills.format_skill_catalog(candidates[:2])


def test_linked_skill_directory_is_followed_once_and_pinned(isolated_skills, tmp_path):
    home, _ = isolated_skills
    real = tmp_path / "dotfiles" / "skills" / "linked"
    real.mkdir(parents = True)
    (real / "SKILL.md").write_text(
        "---\nname: linked\ndescription: Linked in.\n---\nREAL", encoding = "utf-8"
    )
    root = home / ".agents" / "skills"
    root.mkdir(parents = True)
    try:
        (root / "linked").symlink_to(real, target_is_directory = True)
        (root / "dangling").symlink_to(tmp_path / "gone", target_is_directory = True)
        (root / "to-file").symlink_to(real / "SKILL.md")
        (real / "escape.md").symlink_to(tmp_path / "dotfiles")
    except (OSError, NotImplementedError):
        # Reason: Windows may deny symlink creation without Developer Mode.
        pytest.skip("symlinks are unavailable on this platform")

    records = {record["name"]: record for record in skills.list_skills(home = home)}

    assert records["linked"]["valid"] is True
    assert records["dangling"]["valid"] is False and records["to-file"]["valid"] is False
    assert skills.read_skill_resource("linked", home = home).endswith("REAL")
    with pytest.raises(skills.SkillError, match = "symbolic links"):
        skills.read_skill_resource("linked", "escape.md", home = home)


def test_catalog_skips_an_oversized_entry_without_hiding_later_skills():
    candidates = [
        {"name": "oversized", "description": "界" * 600},
        {"name": "usable", "description": "Use this skill."},
    ]

    catalog = skills.format_skill_catalog(candidates)

    assert "oversized" not in catalog
    assert catalog.splitlines() == [
        "- usable: Use this skill.",
        "- 1 more enabled skills not listed; mention one as @skill-name.",
    ]


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

    from routes import inference as inference_routes

    monkeypatch.setattr(inference_routes, "_AGENT_SKILLS_CACHE", {None: (float("inf"), [])})
    response = client.get("/api/skills")
    assert response.status_code == 200
    assert response.json()[0]["name"] == "api-skill"
    assert inference_routes._AGENT_SKILLS_CACHE == {}
    response = client.put("/api/skills/api-skill/enabled", json = {"enabled": False})
    assert response.status_code == 200
    assert response.json()["enabled"] is False

    monkeypatch.setattr(
        inference_routes, "_AGENT_SKILLS_CACHE", {None: (float("inf"), [{"name": "stale"}])}
    )
    response = client.put("/api/skills/api-skill/enabled", json = {"enabled": True})
    assert response.status_code == 200
    assert inference_routes._AGENT_SKILLS_CACHE == {}
    assert client.put("/api/skills/api-skill/enabled", json = {"enabled": "false"}).status_code == 422


def test_skill_tool_selection_honors_explicit_allowlist(isolated_skills, monkeypatch):
    import asyncio

    from models.inference import ChatCompletionRequest
    from routes import inference as inference_routes

    home, _ = isolated_skills
    _write_skill(home, "agents", "guided")
    _write_skill(home, "agents", "skill-creator")
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)
    monkeypatch.setattr(inference_routes, "_enabled_agent_skills", skills.enabled_skills)
    read_only = ChatCompletionRequest(
        model = "test",
        messages = [{"role": "user", "content": "hello"}],
        enable_tools = True,
        enabled_tools = ["read_skill"],
        permission_mode = "auto",
        stream = True,
    )

    selected = asyncio.run(
        inference_routes._select_request_tools(read_only, tools_on = True, mcp_allowed = False)
    )
    names = [tool["function"]["name"] for tool in selected]
    inference_routes._reject_confirm_gate_without_channel(
        read_only, ui_events = False, selected_names = set(names)
    )
    assert names == ["read_skill"]

    local_default = read_only.model_copy(update = {"enabled_tools": ["read_skill", "create_skill"]})
    selected = asyncio.run(
        inference_routes._select_request_tools(local_default, tools_on = True, mcp_allowed = False)
    )
    assert [tool["function"]["name"] for tool in selected] == ["read_skill", "create_skill"]


def test_skill_tools_registration_selection_and_prompt(isolated_skills, monkeypatch):
    import asyncio

    from core.inference import tools as tools_module
    from models.inference import ChatCompletionRequest
    from routes import inference as inference_routes

    home, _ = isolated_skills
    _write_skill(home, "agents", "guided", description = "Guide this task")
    _write_skill(home, "agents", "skill-creator")
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)
    monkeypatch.setattr(inference_routes, "_enabled_agent_skills", skills.enabled_skills)
    payload = ChatCompletionRequest(
        model = "test",
        messages = [{"role": "user", "content": "hello"}],
        enabled_tools = ["read_skill", "create_skill"],
    )

    selected = asyncio.run(
        inference_routes._select_request_tools(payload, tools_on = True, mcp_allowed = False)
    )
    assert [tool["function"]["name"] for tool in selected] == ["read_skill", "create_skill"]
    assert tools_module.is_always_safe_tool("read_skill") is True
    assert tools_module.is_high_risk_tool_call("create_skill", {}) is True
    result = tools_module.execute_tool("read_skill", {"name": "guided"})
    assert "Skill: guided" in result
    nudge = inference_routes._build_tool_action_nudge(tools = selected, model_name = "test")
    assert "- guided: Guide this task" in nudge
    assert "@skill-name" in nudge
    assert "create_skill" in nudge
    # Codex and external paths skip the general nudge but keep the catalog for @mentions.
    narrow = inference_routes._build_tool_action_nudge(
        tools = [*selected, tools_module.WEB_SEARCH_TOOL],
        model_name = "test",
        full_access_only = True,
    )
    assert "- guided: Guide this task" in narrow
    assert inference_routes._TOOL_BASE_NUDGE not in narrow
    assert "web_search" not in narrow

    skills.set_skill_enabled("skill-creator", False, home = home)
    selected = asyncio.run(
        inference_routes._select_request_tools(payload, tools_on = True, mcp_allowed = False)
    )
    assert [tool["function"]["name"] for tool in selected] == ["read_skill"]
    assert "create_skill" not in inference_routes._build_tool_action_nudge(
        tools = selected, model_name = "test"
    )
    with pytest.raises(skills.SkillError, match = "disabled"):
        skills.read_skill_resource("skill-creator", home = home)

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


# Account scoping: the owner keeps the home folders, a managed account gets its own workspace.

_ALICE_ID = "a" * 32
_BOB_ID = "b" * 32


@pytest.fixture
def managed_accounts(isolated_skills, monkeypatch):
    from utils.account_context import AccountContext

    home, studio = isolated_skills
    monkeypatch.setattr(skills, "_owner_home", lambda: home)
    monkeypatch.setattr(
        skills, "_BUNDLED_ROOT", ("bundled", Path(skills.__file__).with_name("bundled_skills"))
    )
    monkeypatch.setattr(skills, "workspace_root", lambda: studio / "accounts" / _account_id())
    return home, studio, AccountContext(_ALICE_ID, "alice"), AccountContext(_BOB_ID, "bob")


def _account_id() -> str:
    from utils.account_context import current_account_id
    return current_account_id()


def _account_skill(studio: Path, account_id: str, name: str, description: str) -> None:
    root = studio / "accounts" / account_id / "skills" / name
    root.mkdir(parents = True)
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nBody", encoding = "utf-8"
    )


def test_managed_account_never_sees_the_owners_home_skills(managed_accounts):
    from utils.account_context import run_as

    home, studio, alice, bob = managed_accounts
    _write_skill(home, "agents", "owner-only", description = "OWNER_PRIVATE")
    _write_skill(home, "claude", "owner-claude", description = "OWNER_PRIVATE")
    _account_skill(studio, _ALICE_ID, "alice-skill", "ALICE_PRIVATE")

    owner_names = [record["name"] for record in skills.list_skills()]
    assert owner_names == ["owner-only", "owner-claude", "skill-creator"]

    alice_records = run_as(alice, skills.list_skills)
    assert [(r["name"], r["source"]) for r in alice_records] == [
        ("alice-skill", "agents"),
        ("skill-creator", "bundled"),
    ]
    assert run_as(bob, skills.list_skills)[0]["name"] == "skill-creator"
    assert "ALICE_PRIVATE" not in str(run_as(bob, skills.list_skills))

    with pytest.raises(skills.SkillNotFoundError):
        run_as(alice, skills.read_skill_resource, "owner-only")
    with pytest.raises(skills.SkillNotFoundError):
        run_as(bob, skills.read_skill_resource, "alice-skill")
    assert "Body" in run_as(alice, skills.read_skill_resource, "alice-skill")


def test_managed_account_overrides_and_creation_stay_in_its_workspace(managed_accounts):
    from utils.account_context import run_as

    home, studio, alice, bob = managed_accounts
    _write_skill(home, "agents", "shared-name", description = "owner copy")
    _account_skill(studio, _ALICE_ID, "shared-name", "alice copy")

    # Alice disabling her copy leaves the owner's enabled and writes only her overrides file.
    assert run_as(alice, skills.set_skill_enabled, "shared-name", False)["enabled"] is False
    assert (studio / "accounts" / _ALICE_ID / "skill-overrides.json").is_file()
    assert not (studio / "skill-overrides.json").exists()
    assert next(r for r in skills.list_skills() if r["name"] == "shared-name")["enabled"] is True
    assert run_as(bob, skills.enabled_skills) == []

    # Bob toggling a skill he cannot see is a not-found, not a write into Alice's file.
    with pytest.raises(skills.SkillNotFoundError):
        run_as(bob, skills.set_skill_enabled, "shared-name", False)
    assert not (studio / "accounts" / _BOB_ID / "skill-overrides.json").exists()

    # create_skill lands in the caller's own workspace, never in the owner's home.
    record = run_as(bob, skills.create_skill, "bob-made", "Bob's skill", "Instructions")
    assert record["path"] == "skills/bob-made/SKILL.md"
    assert (studio / "accounts" / _BOB_ID / "skills" / "bob-made" / "SKILL.md").is_file()
    assert not (home / ".agents" / "skills" / "bob-made").exists()
    assert [r["name"] for r in run_as(bob, skills.enabled_skills)] == ["bob-made"]
    assert "bob-made" not in [r["name"] for r in skills.list_skills()]

    owner_record = skills.create_skill("owner-made", "Owner's skill", "Instructions")
    assert owner_record["path"] == "~/.agents/skills/owner-made/SKILL.md"
    assert (home / ".agents" / "skills" / "owner-made" / "SKILL.md").is_file()


def test_inference_catalog_cache_is_per_account(managed_accounts, monkeypatch):
    import asyncio

    from routes import inference as inference_routes
    from utils.account_context import run_as

    home, studio, alice, bob = managed_accounts
    _write_skill(home, "agents", "owner-only", description = "OWNER_PRIVATE")
    _account_skill(studio, _ALICE_ID, "alice-skill", "ALICE_PRIVATE")
    monkeypatch.setattr(inference_routes, "_AGENT_SKILLS_CACHE", {})

    assert [s["name"] for s in inference_routes._enabled_agent_skills()] == ["owner-only"]
    assert [s["name"] for s in run_as(alice, inference_routes._enabled_agent_skills)] == [
        "alice-skill"
    ]
    assert run_as(bob, inference_routes._enabled_agent_skills) == []
    assert set(inference_routes._AGENT_SKILLS_CACHE) == {None, _ALICE_ID, _BOB_ID}

    # The catalog a request is built from follows the acting account.
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest(
        model = "test", messages = [{"role": "user", "content": "hi"}], enable_tools = True
    )

    def select():
        return asyncio.run(
            inference_routes._select_request_tools(payload, tools_on = True, mcp_allowed = False)
        )

    assert "read_skill" in [t["function"]["name"] for t in run_as(alice, select)]
    assert [
        t["function"]["name"] for t in run_as(bob, select) if "skill" in t["function"]["name"]
    ] == []
    nudge = run_as(
        alice,
        lambda: inference_routes._build_tool_action_nudge(
            tools = run_as(alice, select), model_name = "test"
        ),
    )
    assert "ALICE_PRIVATE" in nudge and "OWNER_PRIVATE" not in nudge

    inference_routes._invalidate_agent_skills_cache()
    assert inference_routes._AGENT_SKILLS_CACHE == {}


def test_read_skill_page_floor_reports_no_room_instead_of_slivers(isolated_skills, monkeypatch):
    from core.inference import tools as tools_module

    home, _ = isolated_skills
    monkeypatch.setattr(skills, "_owner_home", lambda: home)
    _write_skill(home, "agents", "long", body = "x" * 12_000)
    # Whatever the room, a page smaller than the floor is not worth a round trip.
    monkeypatch.setattr(tools_module, "_fit_result_to_room", lambda result, name: result[:40])
    result = tools_module.execute_tool("read_skill", {"name": "long"})
    assert result.startswith("Error: Not enough context room")


def test_read_resource_rejects_ancestor_swapped_after_selection(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "reader")
    (root / "guide.md").write_text("safe", encoding = "utf-8")
    skills_root = root.parent
    outside = home / "outside"
    _write_skill(outside, "agents", "reader")
    (outside / ".agents" / "skills" / "reader" / "guide.md").write_text("secret", encoding = "utf-8")
    original_selected_skill = skills._selected_skill

    def replacing_selected_skill(name, *, home = None):
        selection = original_selected_skill(name, home = home)
        # The skill directory itself stays a real directory; only its parent is swapped.
        skills_root.rename(home / "original-skills")
        try:
            skills_root.symlink_to(outside / ".agents" / "skills", target_is_directory = True)
        except (OSError, NotImplementedError):
            # Reason: Windows may deny symlink creation without Developer Mode.
            pytest.skip("symlinks are unavailable on this platform")
        return selection

    monkeypatch.setattr(skills, "_selected_skill", replacing_selected_skill)
    with pytest.raises(skills.SkillError, match = "changed after it was selected"):
        skills.read_skill_resource("reader", "guide.md", home = home)


@pytest.mark.skipif(
    os.name == "nt",
    reason = "Windows sharing semantics refuse to replace a manifest that is still open for writing",
)
def test_failed_create_keeps_a_manifest_another_writer_replaced(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    manifest = home / ".agents" / "skills" / "racer" / "SKILL.md"
    original_fsync = os.fsync

    def replacing_fsync(descriptor):
        original_fsync(descriptor)
        replacement = manifest.with_name("SKILL.md.new")
        replacement.write_text(
            "---\nname: racer\ndescription: Theirs.\n---\nTHEIRS", encoding = "utf-8"
        )
        os.replace(replacement, manifest)

    monkeypatch.setattr(os, "fsync", replacing_fsync)
    with pytest.raises(skills.SkillError, match = "changed while the manifest was being written"):
        skills.create_skill("racer", "Mine.", "MINE", home = home)

    assert manifest.read_text(encoding = "utf-8").endswith("THEIRS")


def test_unreadable_root_reports_itself_without_hiding_other_roots(isolated_skills):
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("permission bits are not enforced for this user")
    home, _ = isolated_skills
    _write_skill(home, "claude", "visible")
    unreadable = home / ".agents" / "skills"
    unreadable.mkdir(parents = True)
    unreadable.chmod(0)
    try:
        records = skills.list_skills(home = home)
    finally:
        unreadable.chmod(0o700)

    assert next(item for item in records if item["name"] == "visible")["valid"] is True
    failed = next(item for item in records if item["source"] == "agents")
    assert failed["valid"] is False
    assert "scan" in failed["error"]


def test_root_entry_limit_ignores_hidden_and_regular_files(isolated_skills):
    home, _ = isolated_skills
    root = _write_skill(home, "agents", "counted").parent
    for index in range(skills.MAX_SKILLS_PER_ROOT):
        (root / f".hidden-{index}").write_text("", encoding = "utf-8")
    (root / "README.md").write_text("about these skills", encoding = "utf-8")

    records = skills.list_skills(home = home)

    assert [item["name"] for item in records] == ["counted"]


def test_corrupt_overrides_are_ignored_and_repaired_by_the_next_toggle(isolated_skills):
    home, studio = isolated_skills
    _write_skill(home, "agents", "sturdy")
    studio.mkdir()
    (studio / "skill-overrides.json").write_text("{not json", encoding = "utf-8")

    assert next(item for item in skills.list_skills(home = home) if item["name"] == "sturdy")[
        "enabled"
    ]
    skills.set_skill_enabled("sturdy", False, home = home)

    assert json.loads((studio / "skill-overrides.json").read_text(encoding = "utf-8")) == {
        "sturdy": False
    }
    (studio / "skill-overrides.json").write_text(
        '{"sturdy": "no", "Bad Name": false, "other": true}', encoding = "utf-8"
    )
    assert next(item for item in skills.list_skills(home = home) if item["name"] == "sturdy")[
        "enabled"
    ]


def test_indented_separator_inside_a_block_scalar_stays_in_the_frontmatter(isolated_skills):
    home, _ = isolated_skills
    _write_skill(
        home,
        "agents",
        "divided",
        description = "|\n  Use for reports.\n  ---\n  Also for summaries.",
    )

    record = next(item for item in skills.list_skills(home = home) if item["name"] == "divided")

    assert record["valid"] is True
    assert record["description"] == "Use for reports.\n---\nAlso for summaries."


def test_read_skill_tool_applies_defaults_for_null_arguments(isolated_skills, monkeypatch):
    from core.inference import tools as tools_module

    home, _ = isolated_skills
    _write_skill(home, "agents", "nullable", body = "Body text")
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)

    result = tools_module.execute_tool(
        "read_skill", {"name": "nullable", "resource": None, "offset": None}
    )

    assert "Body text" in result


def test_reserved_device_name_resource_gets_its_own_message(isolated_skills):
    home, _ = isolated_skills
    _write_skill(home, "agents", "reserved")

    with pytest.raises(skills.SkillError, match = "reserved device name"):
        skills.read_skill_resource("reserved", "con.md", home = home)


def test_aliased_metadata_cannot_expand_past_the_manifest_limit(isolated_skills):
    home, _ = isolated_skills
    big = "x" * (4 * 1024)
    aliases = "\n".join(f"  k{i}: *big" for i in range(8))
    _write_skill(
        home,
        "agents",
        "aliased",
        frontmatter = f"big: &big {big}\nmetadata:\n{aliases}",
    )

    record = next(r for r in skills.list_skills(home = home) if r["name"] == "aliased")

    assert record["valid"] is False and "16 KB" in record["error"]


@pytest.mark.parametrize("field", ["allowed-tools", "license"])
def test_oversized_scalar_fields_are_rejected(isolated_skills, field):
    home, _ = isolated_skills
    _write_skill(home, "agents", "wide", frontmatter = f"{field}: {'x' * 2000}")

    record = next(r for r in skills.list_skills(home = home) if r["name"] == "wide")

    assert record["valid"] is False and "1024" in record["error"]


def test_update_skill_rewrites_the_manifest_and_keeps_the_rest_of_its_frontmatter(isolated_skills):
    home, _ = isolated_skills
    folder = _write_skill(
        home,
        "agents",
        "notes",
        description = "Old description",
        frontmatter = "license: MIT\nmetadata:\n  author: leo",
        body = "Old body",
    )
    skills.set_skill_enabled("notes", False, home = home)

    manifest = skills.read_skill_manifest("notes", home = home)
    assert (manifest["description"], manifest["instructions"]) == ("Old description", "Old body")
    assert manifest["path"] == "~/.agents/skills/notes/SKILL.md"

    record = skills.update_skill(
        "notes", " New description ", "# New body\n\nStep one.\n", home = home
    )

    assert record["description"] == "New description"
    assert record["license"] == "MIT" and record["metadata"] == {"author": "leo"}
    # The dialog only edits two fields; a disable set earlier is not an edit.
    assert record["enabled"] is False
    text = (folder / "SKILL.md").read_text(encoding = "utf-8")
    assert text.startswith("---\nname: notes\ndescription: New description\nlicense: MIT\n")
    assert text.endswith("---\n\n# New body\n\nStep one.\n")
    assert (
        skills.read_skill_manifest("notes", home = home)["instructions"] == "# New body\n\nStep one."
    )
    # The write went through a temporary file; none is left behind as a stray resource.
    assert [path.name for path in folder.iterdir()] == ["SKILL.md"]
    with pytest.raises(skills.SkillError, match = "1-1024"):
        skills.update_skill("notes", "", "Body", home = home)
    with pytest.raises(skills.SkillError, match = "non-empty"):
        skills.update_skill("notes", "Description", " ", home = home)


def test_delete_skill_removes_the_folder_and_its_override(isolated_skills):
    home, _ = isolated_skills
    folder = _write_skill(home, "agents", "gone")
    (folder / "scripts").mkdir()
    (folder / "scripts" / "run.py").write_text("print(1)", encoding = "utf-8")
    _write_skill(home, "agents", "stays")
    skills.set_skill_enabled("gone", False, home = home)
    skills.set_skill_enabled("stays", False, home = home)
    assert skills._load_overrides() == {"gone": False, "stays": False}

    record = skills.delete_skill("gone", home = home)

    assert record["name"] == "gone"
    assert not folder.exists()
    assert skills._load_overrides() == {"stays": False}
    assert [item["name"] for item in skills.list_skills(home = home)] == ["stays"]
    with pytest.raises(skills.SkillNotFoundError):
        skills.delete_skill("gone", home = home)


@pytest.mark.parametrize("operation", ("update", "delete"))
def test_only_agents_skills_can_be_changed_from_the_dialog(isolated_skills, monkeypatch, operation):
    home, _ = isolated_skills
    _write_skill(home, "claude", "claude-owned", body = "Claude body")
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
        ("bundled", Path(skills.__file__).with_name("bundled_skills")),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)

    def change(name: str):
        if operation == "update":
            return skills.update_skill(name, "Description", "Instructions")
        return skills.delete_skill(name)

    for name in ("claude-owned", "skill-creator"):
        # Readable in the editor, as a read-only view, even while disabled.
        manifest = skills.read_skill_manifest(name)
        assert manifest["source"] != "agents" and manifest["instructions"]
        with pytest.raises(skills.SkillError, match = "cannot be changed here"):
            change(name)
    assert skills.read_skill_manifest("claude-owned")["instructions"] == "Claude body"
    with pytest.raises(skills.SkillNotFoundError):
        change("missing")
    with pytest.raises(skills.SkillNotFoundError):
        skills.read_skill_manifest("missing")
    assert (home / ".claude" / "skills" / "claude-owned" / "SKILL.md").is_file()
    assert (
        Path(skills.__file__).with_name("bundled_skills") / "skill-creator" / "SKILL.md"
    ).is_file()


def test_linked_agents_skill_stays_read_only_in_the_dialog(isolated_skills, tmp_path):
    home, _ = isolated_skills
    real = _write_skill(tmp_path / "elsewhere", "agents", "linked")
    root = home / ".agents" / "skills"
    root.mkdir(parents = True)
    try:
        (root / "linked").symlink_to(real, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    listed = skills.list_skills(home = home)[0]
    assert (listed["valid"], listed["linked"]) == (True, True)

    assert skills.read_skill_manifest("linked", home = home)["linked"] is True
    with pytest.raises(skills.SkillError, match = "is a link"):
        skills.update_skill("linked", "Description", "Instructions", home = home)
    with pytest.raises(skills.SkillError, match = "is a link"):
        skills.delete_skill("linked", home = home)
    assert (real / "SKILL.md").is_file()
    assert (root / "linked").is_symlink()


def test_authenticated_create_read_update_and_delete_routes(isolated_skills, monkeypatch):
    home, _ = isolated_skills
    roots = (
        ("agents", home / ".agents" / "skills"),
        ("claude", home / ".claude" / "skills"),
    )
    monkeypatch.setattr(skills, "_skill_roots", lambda home = None: roots)
    monkeypatch.setattr(skills, "_owner_home", lambda: home)
    from routes import inference as inference_routes

    app = FastAPI()
    app.include_router(router, prefix = "/api/skills")
    draft = {"name": "made", "description": "Made in the dialog.", "instructions": "Do the thing."}
    anonymous = TestClient(app)
    assert anonymous.post("/api/skills", json = draft).status_code in (401, 403)
    assert anonymous.delete("/api/skills/made").status_code in (401, 403)
    assert not (home / ".agents" / "skills" / "made").exists()
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    client = TestClient(app)

    monkeypatch.setattr(inference_routes, "_AGENT_SKILLS_CACHE", {None: (float("inf"), [])})
    response = client.post("/api/skills", json = draft)
    assert response.status_code == 201, response.text
    created = response.json()
    assert (created["name"], created["enabled"], created["source"]) == ("made", True, "agents")
    assert created["path"] == "~/.agents/skills/made/SKILL.md"
    assert (home / ".agents" / "skills" / "made" / "SKILL.md").is_file()
    assert inference_routes._AGENT_SKILLS_CACHE == {}
    assert client.post("/api/skills", json = draft).status_code == 409
    assert client.post("/api/skills", json = {**draft, "name": "Bad Name"}).status_code == 400
    assert client.post("/api/skills", json = {**draft, "name": 1}).status_code == 422
    assert client.post("/api/skills", json = {**draft, "extra": True}).status_code == 422

    response = client.get("/api/skills/made")
    assert response.status_code == 200, response.text
    assert response.json()["instructions"] == "Do the thing."
    assert client.get("/api/skills/missing").status_code == 404

    monkeypatch.setattr(inference_routes, "_AGENT_SKILLS_CACHE", {None: (float("inf"), [])})
    edit = {"description": "Edited.", "instructions": "Do it better."}
    response = client.put("/api/skills/made", json = edit)
    assert response.status_code == 200, response.text
    assert response.json()["description"] == "Edited."
    assert client.get("/api/skills/made").json()["instructions"] == "Do it better."
    assert inference_routes._AGENT_SKILLS_CACHE == {}
    assert client.put("/api/skills/made", json = {**edit, "description": ""}).status_code == 400
    assert client.put("/api/skills/missing", json = edit).status_code == 404

    monkeypatch.setattr(inference_routes, "_AGENT_SKILLS_CACHE", {None: (float("inf"), [])})
    assert client.delete("/api/skills/made").status_code == 204
    assert not (home / ".agents" / "skills" / "made").exists()
    assert inference_routes._AGENT_SKILLS_CACHE == {}
    assert client.delete("/api/skills/made").status_code == 404


def test_managed_account_edits_and_deletes_only_its_own_skills(managed_accounts):
    from utils.account_context import run_as

    home, studio, alice, bob = managed_accounts
    owner_manifest = (
        _write_skill(home, "agents", "owner-made", description = "owner copy") / "SKILL.md"
    )
    run_as(bob, skills.create_skill, "bob-made", "Bob's skill", "Instructions")

    record = run_as(bob, skills.update_skill, "bob-made", "Bob's edited skill", "Edited")
    assert record["description"] == "Bob's edited skill"
    assert run_as(bob, skills.read_skill_manifest, "bob-made")["instructions"] == "Edited"
    # The owner's home is not in a managed account's roots, so it is not-found rather than edited.
    with pytest.raises(skills.SkillNotFoundError):
        run_as(bob, skills.update_skill, "owner-made", "Hijacked", "Body")
    with pytest.raises(skills.SkillNotFoundError):
        run_as(alice, skills.delete_skill, "bob-made")
    assert "owner copy" in owner_manifest.read_text(encoding = "utf-8")

    run_as(bob, skills.delete_skill, "bob-made")
    assert not (studio / "accounts" / _BOB_ID / "skills" / "bob-made").exists()
    assert owner_manifest.is_file()
