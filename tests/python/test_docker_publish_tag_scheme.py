# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The published tag list must stay short: :core, :latest, :studio, the release tags,
and one dated nightly pin per image. Per-run handles are removed by cleanup."""

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
    return [
        l.strip()
        for l in steps[0]["with"]["tags"].splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]


def test_no_per_commit_tags_are_published(doc: dict):
    text = WORKFLOW.read_text(encoding = "utf-8")
    assert "type=sha" not in text, "sha-<commit> tags are back on every push"
    for job in ("merge", "merge-studio"):
        assert not any(
            "nightly" in r and "pin_date" not in r for r in _tag_rules(doc, job)
        ), f"{job} still writes a mutable nightly tag"


@pytest.mark.parametrize(
    ("job", "handle", "pin"),
    [
        (
            "merge",
            "type=raw,value=core-build-${{ github.run_id }}",
            "type=schedule,pattern=core-nightly-${{ needs.prepare.outputs.pin_date }}",
        ),
        (
            "merge-studio",
            "type=raw,value=build-${{ github.run_id }}",
            "type=schedule,pattern=nightly-${{ needs.prepare.outputs.pin_date }}",
        ),
    ],
)
def test_each_image_has_a_per_run_handle_and_a_dated_pin(
    doc: dict, job: str, handle: str, pin: str
):
    rules = _tag_rules(doc, job)
    assert handle in rules, f"{job} lost the per-run handle the digest handoff resolves"
    assert pin in rules, f"{job} lost its dated nightly pin"
    assert "prepare" in doc["jobs"][job]["needs"]
    assert doc["jobs"]["prepare"]["outputs"]["pin_date"] == "${{ steps.pin.outputs.date }}"


def _pin_step(doc: dict) -> dict:
    return [s for s in doc["jobs"]["prepare"]["steps"] if s.get("id") == "pin"][0]


def _run_pin(step: dict, tmp_path: Path, *, gh: str) -> tuple[subprocess.CompletedProcess, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text("#!/usr/bin/env bash\n" + gh + "\n", encoding = "utf-8")
    (bin_dir / "gh").chmod(0o755)
    out = tmp_path / "output"
    out.write_text("", encoding = "utf-8")
    script = (
        step["run"]
        .replace("${{ github.repository }}", "unslothai/unsloth")
        .replace("${{ github.run_id }}", "424242")
    )
    assert "${{" not in script
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["GITHUB_OUTPUT"] = str(out)
    res = subprocess.run(
        ["bash", "-e", "-c", script], capture_output = True, text = True, env = env, timeout = 60
    )
    return res, out.read_text(encoding = "utf-8")


def test_the_pin_date_is_the_run_creation_date(doc: dict, tmp_path: Path):
    """created_at is fixed at the first attempt, unlike run_started_at and the clock, so a late rerun cannot claim today's pin with an older commit."""
    step = _pin_step(doc)
    assert doc["jobs"]["prepare"]["permissions"].get("actions") == "read"
    assert step["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert "actions/runs/${{ github.run_id }}" in step["run"]
    assert "date -u" not in step["run"]
    res, out = _run_pin(
        step,
        tmp_path,
        gh = "echo \"$*\" >&2; printf '2026-09-05T18:17:03Z\\n'",
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert out == "date=2026.09.05\n"
    assert "repos/unslothai/unsloth/actions/runs/424242 --jq .created_at" in res.stderr


@pytest.mark.parametrize("gh", ["exit 1", "printf ''", "printf 'null'"])
def test_an_unreadable_creation_time_fails_prepare(doc: dict, tmp_path: Path, gh: str):
    res, out = _run_pin(_pin_step(doc), tmp_path, gh = gh)
    assert res.returncode != 0
    assert "::error::" in res.stdout
    assert out == ""


def test_the_digest_export_resolves_the_handle(doc: dict):
    for job, needle in (("merge", ":core-build-"), ("merge-studio", ":build-")):
        step = [s for s in doc["jobs"][job]["steps"] if s.get("name") == "Export manifest digest"][
            0
        ]["run"]
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


def test_cleanup_runs_after_everything_except_an_overridden_dispatch(cleanup_job: dict, doc: dict):
    """A default-input dispatch on the default branch publishes the stable tags, so its handles are not its only names and must go like a push's."""
    assert set(cleanup_job["needs"]) == {"merge", "merge-studio", "hub-readme", "smoke-test"}
    cond = cleanup_job["if"]
    assert cond.startswith("${{ always() && (github.event_name != 'workflow_dispatch' || (")
    gate = doc["jobs"]["hub-readme"]["if"].strip().removeprefix("${{").removesuffix("}}").strip()
    assert gate in cond, "the dispatch exception must be the stable-tag gate itself"


def _manifest_step(doc: dict, job: str) -> str:
    return [s for s in doc["jobs"][job]["steps"] if s.get("name") == "Create multi-arch manifest"][
        0
    ]["run"]


def _run_manifest(
    step: str,
    tmp_path: Path,
    *,
    existing: list[str],
    tags: list[str],
    probe_code: str = "404",
) -> tuple[subprocess.CompletedProcess, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker.log"
    (bin_dir / "sleep").write_text("#!/usr/bin/env bash\n", encoding = "utf-8")
    (bin_dir / "sleep").chmod(0o755)
    (bin_dir / "docker").write_text(
        "#!/usr/bin/env bash\n" + f"printf '%s\\n' \"$*\" >> {log}\n", encoding = "utf-8"
    )
    (bin_dir / "docker").chmod(0o755)
    probes = tmp_path / "curl.log"
    (bin_dir / "curl").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> {probes}\n'
        'case "$*" in\n'
        + "".join(f"  */tags/{e}) printf '200' ;;\n" for e in existing)
        # curl prints 000 and exits 7 when the connection fails
        + (
            "  *) printf '000'; exit 7 ;;\n"
            if probe_code == "000"
            else f"  *) printf '{probe_code}' ;;\n"
        )
        + "esac\n",
        encoding = "utf-8",
    )
    (bin_dir / "curl").chmod(0o755)
    digests = tmp_path / "digests"
    digests.mkdir()
    (digests / ("a" * 64)).write_text("", encoding = "utf-8")
    script = step.replace("${{ env.REGISTRY }}", "docker.io").replace(
        "${{ env.IMAGE_NAME }}", "unsloth/unsloth"
    )
    assert "${{" not in script
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["DOCKER_METADATA_OUTPUT_JSON"] = (
        '{"tags": [' + ", ".join(f'"docker.io/unsloth/unsloth:{t}"' for t in tags) + "]}"
    )
    res = subprocess.run(
        ["bash", "-e", "-c", script],
        capture_output = True,
        text = True,
        env = env,
        cwd = str(digests),
        timeout = 60,
    )
    return res, log.read_text(encoding = "utf-8") if log.exists() else ""


def _probes(tmp_path: Path) -> list[str]:
    path = tmp_path / "curl.log"
    return path.read_text(encoding = "utf-8").splitlines() if path.exists() else []


@pytest.mark.parametrize("job", ["merge", "merge-studio"])
def test_an_existing_dated_pin_is_never_replaced(doc: dict, job: str, tmp_path: Path):
    step = _manifest_step(doc, job)
    res, log = _run_manifest(
        step,
        tmp_path,
        existing = ["core-nightly-2026.09.06"],
        tags = ["core", "core-nightly-2026.09.06", "core-build-777"],
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "-t docker.io/unsloth/unsloth:core " in log
    assert "-t docker.io/unsloth/unsloth:core-build-777 " in log
    assert "core-nightly-2026.09.06" not in log
    assert "already exists" in res.stdout


@pytest.mark.parametrize("job", ["merge", "merge-studio"])
def test_a_new_dated_pin_is_created(doc: dict, job: str, tmp_path: Path):
    step = _manifest_step(doc, job)
    res, log = _run_manifest(
        step,
        tmp_path,
        existing = [],
        tags = ["latest", "nightly-2026.09.06", "build-777"],
    )
    assert res.returncode == 0, res.stdout + res.stderr
    for t in ("latest", "nightly-2026.09.06", "build-777"):
        assert f"-t docker.io/unsloth/unsloth:{t} " in log
    assert len(_probes(tmp_path)) == 1, "only the dated pin is probed, once"


@pytest.mark.parametrize("job", ["merge", "merge-studio"])
@pytest.mark.parametrize("code", ["000", "429", "503"])
def test_an_unanswered_probe_stops_the_merge(doc: dict, job: str, code: str, tmp_path: Path):
    """Anything but 404 used to count as absent, and bash -e is off inside the probe's condition, so a transport failure or a 429 became an overwrite."""
    step = _manifest_step(doc, job)
    res, log = _run_manifest(
        step,
        tmp_path,
        existing = [],
        tags = ["latest", "nightly-2026.09.06", "build-777"],
        probe_code = code,
    )
    assert res.returncode != 0
    assert f"(HTTP {code})" in res.stdout
    assert "::error::" in res.stdout
    assert log == "", "no manifest may be written while the pin's state is unknown"
    assert len(_probes(tmp_path)) == 5, "the probe is retried before giving up"


def _run_cleanup(
    step: str,
    tmp_path: Path,
    *,
    event: str,
    delete_code: str = "204",
    tags: list[str] | None = None,
    second_page: list[str] | None = None,
    today: str = "2026.09.06",
) -> tuple[subprocess.CompletedProcess, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "curl.log"
    listing = tmp_path / "tags.json"
    page2 = tmp_path / "tags2.json"
    nxt = (
        ', "next": "https://hub.docker.com/v2/namespaces/unsloth/repositories/unsloth/tags?page=2&page_size=100"'
        if second_page
        else ', "next": null'
    )
    listing.write_text(
        '{"results": [' + ", ".join(f'{{"name": "{t}"}}' for t in (tags or [])) + "]" + nxt + "}",
        encoding = "utf-8",
    )
    page2.write_text(
        '{"results": ['
        + ", ".join(f'{{"name": "{t}"}}' for t in (second_page or []))
        + '], "next": null}',
        encoding = "utf-8",
    )
    (bin_dir / "curl").write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {log}\n"
        'case "$*" in\n'
        '  *auth/token*) printf \'{"access_token": "tok"}\' ;;\n'
        f"  *-X\\ DELETE*) printf '{delete_code}' ;;\n"
        f"  *page=2*) cat {page2} ;;\n"
        f"  *) cat {listing} ;;\n"
        "esac\n",
        encoding = "utf-8",
    )
    (bin_dir / "curl").chmod(0o755)
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
        ["bash", "-e", "-c", script],
        capture_output = True,
        text = True,
        env = env,
        cwd = str(tmp_path),
        timeout = 60,
    )
    return res, log.read_text(encoding = "utf-8") if log.exists() else ""


def test_a_push_removes_both_handles_on_the_namespace_route(cleanup_job: dict, tmp_path: Path):
    step = cleanup_job["steps"][-1]["run"]
    res, log = _run_cleanup(step, tmp_path, event = "push")
    assert res.returncode == 0, res.stdout + res.stderr
    deletes = [l for l in log.splitlines() if "-X DELETE" in l]
    assert [l.split("/tags/")[1].split(" ")[0] for l in deletes] == ["core-build-777", "build-777"]
    assert all("/v2/namespaces/unsloth/repositories/unsloth/tags/" in l for l in deletes)
    assert (
        "/v2/repositories/" not in log
    ), "the legacy route answers every organization token with 403"
    assert "?page_size" not in log, "a push must not prune pins"


def test_a_missing_handle_is_not_a_failure(cleanup_job: dict, tmp_path: Path):
    """A failed merge never created the handle, so 404 on delete is expected."""
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
        "latest",
        "core",
        "studio",
        "core-v2026.9.1",
        "stable",
        "nightly-2026.09.05",
        "core-nightly-2026.09.05",
        "nightly-2026.07.01",
        "core-nightly-2026.07.01",
        "nightly",
        "core-nightly",
        "2026.5.9-pt2.10.0-vllm-0.16.0-cu12.8-studio-release-v0.1.43-beta-2026-MAY-31",
    ]
    res, log = _run_cleanup(step, tmp_path, event = "schedule", tags = tags)
    assert res.returncode == 0, res.stdout + res.stderr
    deleted = [l.split("/tags/")[1].split(" ")[0] for l in log.splitlines() if "-X DELETE" in l]
    assert deleted == [
        "core-build-777",
        "build-777",
        "nightly-2026.07.01",
        "core-nightly-2026.07.01",
    ], deleted


def test_the_prune_follows_every_page(cleanup_job: dict, tmp_path: Path):
    """The listing is newest first and two pins a day pass 100 tags inside the retention window, so expired pins live on pages a single request never returns."""
    step = cleanup_job["steps"][-1]["run"]
    res, log = _run_cleanup(
        step,
        tmp_path,
        event = "schedule",
        tags = ["latest", "nightly-2026.09.05"],
        second_page = ["nightly-2026.06.01", "core-nightly-2026.06.01", "stable"],
    )
    assert res.returncode == 0, res.stdout + res.stderr
    deleted = [l.split("/tags/")[1].split(" ")[0] for l in log.splitlines() if "-X DELETE" in l]
    assert deleted == [
        "core-build-777",
        "build-777",
        "nightly-2026.06.01",
        "core-nightly-2026.06.01",
    ], deleted
    assert log.count("page_size=100") == 2, "the second page was not fetched"
