# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The published tag list is meant to be short: :core, :latest, :studio, the release
tags, and one dated nightly pin per image. Per-commit sha tags used to land on every
push, and the mutable :nightly said nothing :latest did not. The digest handoff
between jobs now uses a per-run handle that a cleanup job removes, so these pin the
scheme and run the cleanup step with curl stubbed.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-publish.yml"
HUB_README = REPO_ROOT / "docker" / "DOCKERHUB.md"


@pytest.fixture(scope = "module")
def doc() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _tag_rules(doc: dict, job: str) -> list[str]:
    steps = [s for s in doc["jobs"][job]["steps"] if s.get("id") == "meta"]
    assert len(steps) == 1
    return [l.strip() for l in steps[0]["with"]["tags"].splitlines() if l.strip() and not l.strip().startswith("#")]


def test_no_per_commit_tags_are_published(doc: dict):
    text = WORKFLOW.read_text(encoding = "utf-8")
    assert "type=sha" not in text, "sha-<commit> tags are back on every push"
    for job in ("merge", "merge-studio"):
        assert not any("nightly" in r and "{{date" not in r for r in _tag_rules(doc, job)), (
            f"{job} still writes a mutable nightly tag"
        )


@pytest.mark.parametrize(
    ("job", "handle", "pin"),
    [
        ("merge", "type=raw,value=core-build-${{ github.run_id }}", "type=schedule,pattern=core-nightly-{{date 'YYYY.MM.DD'}}"),
        ("merge-studio", "type=raw,value=build-${{ github.run_id }}", "type=schedule,pattern=nightly-{{date 'YYYY.MM.DD'}}"),
    ],
)
def test_each_image_has_a_per_run_handle_and_a_dated_pin(doc: dict, job: str, handle: str, pin: str):
    rules = _tag_rules(doc, job)
    assert handle in rules, f"{job} lost the per-run handle the digest handoff resolves"
    assert pin in rules, f"{job} lost its dated nightly pin"


def test_the_digest_export_resolves_the_handle(doc: dict):
    for job, needle in (("merge", ':core-build-'), ("merge-studio", ':build-')):
        step = [s for s in doc["jobs"][job]["steps"] if s.get("name") == "Export manifest digest"][0]["run"]
        assert f'select(contains("{needle}"))' in step


def test_the_hub_readme_lists_the_pins_not_the_handles():
    text = HUB_README.read_text(encoding = "utf-8")
    assert "core-nightly-<YYYY.MM.DD>" in text
    for stale in ("sha-<commit>", "core-sha-", "build-<run_id>"):
        assert stale not in text


@pytest.fixture(scope = "module")
def cleanup_job(doc: dict) -> dict:
    assert "cleanup" in doc["jobs"], "the handle cleanup job is missing"
    return doc["jobs"]["cleanup"]


def test_cleanup_runs_after_everything_except_on_dispatch(cleanup_job: dict):
    assert set(cleanup_job["needs"]) == {"merge", "merge-studio", "hub-readme", "smoke-test"}
    cond = cleanup_job["if"]
    assert "always()" in cond and "github.event_name != 'workflow_dispatch'" in cond


def _run_cleanup(
    step: str,
    tmp_path: Path,
    *,
    event: str,
    delete_code: str = "204",
    tags: list[str] | None = None,
    today: str = "2026.09.06",
) -> tuple[subprocess.CompletedProcess, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "curl.log"
    listing = tmp_path / "tags.json"
    listing.write_text(
        '{"results": [' + ", ".join(f'{{"name": "{t}"}}' for t in (tags or [])) + "]}", encoding = "utf-8"
    )
    (bin_dir / "curl").write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {log}\n"
        'case "$*" in\n'
        '  *auth/token*) printf \'{"access_token": "tok"}\' ;;\n'
        f"  *-X\\ DELETE*) printf '{delete_code}' ;;\n"
        f"  *) cat {listing} ;;\n"
        "esac\n",
        encoding = "utf-8",
    )
    (bin_dir / "curl").chmod(0o755)
    # the cutoff comes from `date -u -d "-N days"`; pin today so the test is stable
    (bin_dir / "date").write_text(
        "#!/usr/bin/env bash\n"
        f'/bin/date -u -d "{today.replace(".", "-")} -${{NIGHTLY_KEEP_DAYS}} days" +%Y.%m.%d\n',
        encoding = "utf-8",
    )
    (bin_dir / "date").chmod(0o755)
    script = (
        step.replace("${{ secrets.DOCKER_API_KEY }}", "not-a-secret")
        .replace("${{ env.REGISTRY_USERNAME }}", "unsloth")
        .replace("${{ github.run_id }}", "777")
        .replace("${{ github.event_name }}", event)
    )
    assert "${{" not in script, "unexpanded expression in the cleanup step"
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env.update(IMAGE_NAME = "unsloth/unsloth", NIGHTLY_KEEP_DAYS = "60")
    res = subprocess.run(
        ["bash", "-e", "-c", script], capture_output = True, text = True, env = env, cwd = str(tmp_path), timeout = 60
    )
    return res, log.read_text(encoding = "utf-8") if log.exists() else ""


def test_a_push_removes_both_handles_on_the_namespace_route(cleanup_job: dict, tmp_path: Path):
    step = cleanup_job["steps"][-1]["run"]
    res, log = _run_cleanup(step, tmp_path, event = "push")
    assert res.returncode == 0, res.stdout + res.stderr
    deletes = [l for l in log.splitlines() if "-X DELETE" in l]
    assert [l.split("/tags/")[1].split(" ")[0] for l in deletes] == ["core-build-777", "build-777"]
    assert all("/v2/namespaces/unsloth/repositories/unsloth/tags/" in l for l in deletes)
    assert "/v2/repositories/" not in log, "the legacy route answers every organization token with 403"
    assert "?page_size" not in log, "a push must not prune pins"


def test_a_missing_handle_is_not_a_failure(cleanup_job: dict, tmp_path: Path):
    """A failed merge never created the handle; 404 on delete is the expected shape."""
    step = cleanup_job["steps"][-1]["run"]
    res, _ = _run_cleanup(step, tmp_path, event = "push", delete_code = "404")
    assert res.returncode == 0, res.stdout + res.stderr


def test_a_handle_that_survives_fails_the_job(cleanup_job: dict, tmp_path: Path):
    step = cleanup_job["steps"][-1]["run"]
    res, _ = _run_cleanup(step, tmp_path, event = "push", delete_code = "403")
    assert res.returncode != 0
    assert "not removed" in res.stdout + res.stderr


def test_the_daily_run_prunes_only_old_dated_pins(cleanup_job: dict, tmp_path: Path):
    step = cleanup_job["steps"][-1]["run"]
    tags = [
        "latest", "core", "studio", "core-v2026.9.1", "stable",
        "nightly-2026.09.05", "core-nightly-2026.09.05",      # fresh, kept
        "nightly-2026.07.01", "core-nightly-2026.07.01",      # older than 60 days, pruned
        "nightly", "core-nightly",                            # not dated pins, never touched
        "2026.5.9-pt2.10.0-vllm-0.16.0-cu12.8-studio-release-v0.1.43-beta-2026-MAY-31",
    ]
    res, log = _run_cleanup(step, tmp_path, event = "schedule", tags = tags)
    assert res.returncode == 0, res.stdout + res.stderr
    deleted = [l.split("/tags/")[1].split(" ")[0] for l in log.splitlines() if "-X DELETE" in l]
    assert deleted == ["core-build-777", "build-777", "nightly-2026.07.01", "core-nightly-2026.07.01"], deleted
