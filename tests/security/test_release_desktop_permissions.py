"""Permission-boundary checks for the desktop release workflow."""

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"
UPDATER_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-desktop-updater.yml"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def test_only_publish_job_can_write_repository_contents():
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}

    write_jobs = [
        name
        for name, job in workflow["jobs"].items()
        if job.get("permissions", {}).get("contents") == "write"
    ]
    assert write_jobs == ["publish-release"]


def _poll_loop_body(script):
    """Return the body of the first live `while ...; do ... done` loop.

    Assertions about a wait have to land inside the loop that waits, not
    anywhere in the step, and that loop has to be one the shell actually
    enters: `while false; do` keeps a textually perfect body while skipping
    every API read and status check, and the step falls straight through to a
    download that races the matrix. So the condition must be the unconditional
    `:` or `true` that a poll exiting via `break` uses. Nesting is tracked by
    depth; every opener in this workflow ends its line with `do`.
    """
    lines = script.split("\n")
    opener = re.compile(r"\s*while\s+(?P<condition>.*?)\s*;\s*do\s*$")
    starts = [
        index
        for index, line in enumerate(lines)
        if (match := opener.match(line)) and match.group("condition") in (":", "true")
    ]
    assert starts, f"no live (`while :` / `while true`) poll loop in the wait step:\n{script}"

    start = starts[0]
    depth = 0
    for index in range(start, len(lines)):
        line = lines[index]
        if re.search(r"(?:^|;)\s*do\s*$", line):
            depth += 1
        if re.match(r"\s*done\b", line):
            depth -= 1
            if depth == 0:
                return "\n".join(lines[start + 1 : index])
    raise AssertionError(f"unterminated `while` loop in the wait step:\n{script}")


def test_build_matrix_hands_off_assets_without_release_credentials():
    """The build matrix signs bundles; only publish-release may release them.

    The handoff is one-way and credential-free: the matrix uploads artifacts and
    holds no release token, and publish-release downloads them. Since #8193 the
    ordering is no longer expressed as `needs: build` (publish-release starts
    alongside the matrix to queue for its runner in parallel) but by the "Wait
    for the build matrix" step, which must be at least as strict. Both halves
    are asserted below, so removing the wait does not silently reintroduce
    publishing a partial release.
    """
    jobs = _workflow()["jobs"]
    build = jobs["build"]
    publish = jobs["publish-release"]

    assert "permissions" not in build
    tauri_steps = [
        step
        for step in build["steps"]
        if step.get("uses", "").startswith("tauri-apps/tauri-action@")
    ]
    assert len(tauri_steps) == 3
    for step in tauri_steps:
        assert "GITHUB_TOKEN" not in step.get("env", {})
        assert not {"releaseId", "tagName", "releaseName"} & step.get("with", {}).keys()

    assert any(
        step.get("uses", "").startswith("actions/upload-artifact@") for step in build["steps"]
    )
    assert any(
        step.get("uses", "").startswith("actions/download-artifact@") for step in publish["steps"]
    )
    # publish-release deliberately does not `needs: build`, so the wait step is
    # the whole of the gate. It must cover every matrix leg by name, refuse to
    # publish on a leg that did not succeed, and refuse to publish a leg whose
    # job record never appeared, rather than defaulting to "finished".
    assert publish["needs"] == ["prepare-version"]
    wait = next(
        step for step in publish["steps"] if step.get("name") == "Wait for the build matrix"
    )
    wait_run = wait["run"]

    matrix_legs = {f"Build {entry['label']}" for entry in build["strategy"]["matrix"]["include"]}
    assert len(matrix_legs) == len(tauri_steps)
    for leg in matrix_legs:
        assert f"'{leg}'" in wait_run, leg

    assert "refusing to publish, these build jobs did not succeed" in wait_run
    assert "refusing to publish without confirming they ran" in wait_run
    # Every one of those refusals has to be terminal.
    assert wait_run.count("exit 1") >= 3

    # Assert the mechanism, not just the error strings: those survive a step that no longer loops or
    # no longer reads a conclusion, and then the download races the matrix. Everything below is
    # checked inside the loop body, because a one-shot `gh api` read beside a dead `while` would
    # satisfy the same substrings while waiting for nothing.
    loop_body = _poll_loop_body(wait_run)
    assert "actions/runs/${GITHUB_RUN_ID}/jobs" in loop_body, wait_run
    assert ".status" in loop_body and ".conclusion" in loop_body, wait_run
    # Not finished yet is "keep waiting"; finished but not `success` is a refusal.
    assert re.search(r'!=\s*"completed"', loop_body), wait_run
    assert re.search(r'!=\s*"success"', loop_body), wait_run
    # A loop that never sleeps is a spin, and one that never breaks never ends.
    assert re.search(r"^\s*sleep\b", loop_body, re.MULTILINE), wait_run
    assert re.search(r"^\s*break\b", loop_body, re.MULTILINE), wait_run

    names = [step.get("name") for step in publish["steps"]]
    assert names.index("Wait for the build matrix") < names.index("Publish release assets")
    # And it has to clear before the assets are pulled, or the download races the legs and publish-release dies on
    # artifacts that do not exist yet.
    download = next(
        index
        for index, step in enumerate(publish["steps"])
        if step.get("uses", "").startswith("actions/download-artifact@")
    )
    assert names.index("Wait for the build matrix") < download, names

    # The guard refuses a release that already carries desktop assets, so a
    # version is never published twice.
    release_step = next(
        step for step in publish["steps"] if step.get("name") == "Validate versioned release state"
    )
    assert 'gh api "repos/${GH_REPO}/releases/tags/${DESKTOP_RELEASE_TAG}"' in release_step["run"]
    assert "already carries desktop assets" in release_step["run"]

    # The release is the maintainer's: assets are uploaded onto it, but the
    # release itself is never created and its notes are never rewritten.
    assert not any("gh release create" in step.get("run", "") for step in publish["steps"])
    assert not any("gh release edit" in step.get("run", "") for step in publish["steps"])


def test_post_publish_scan_job_holds_no_release_credentials():
    """#8194 added a job that handles release bundles; it must not be able to release.

    virustotal-scan downloads the published assets and uploads them to a third
    party. It declares no `permissions` block, so it inherits the workflow's
    `contents: read`, and it carries no repository token of any kind: the only
    secret it sees is the VirusTotal key.
    """
    scan = _workflow()["jobs"]["virustotal-scan"]

    assert "permissions" not in scan
    assert "GITHUB_TOKEN" not in scan.get("env", {})
    assert "GH_TOKEN" not in scan.get("env", {})

    for step in scan["steps"]:
        env = step.get("env", {})
        assert "GITHUB_TOKEN" not in env, step.get("name")
        assert "GH_TOKEN" not in env, step.get("name")
        if step.get("uses", "").startswith("actions/checkout@"):
            assert step["with"]["persist-credentials"] is False
        # No `gh` calls: the job has no token to make them with.
        assert "gh release" not in (step.get("run") or "")

    secrets = {
        value
        for step in scan["steps"]
        for value in step.get("env", {}).values()
        if isinstance(value, str) and "secrets." in value
    }
    assert secrets == {"${{ secrets.VIRUS_TOTAL_API_TOKEN }}"}


def test_versioned_release_hides_updater_signature_assets():
    steps = _workflow()["jobs"]["publish-release"]["steps"]
    publish = next(step for step in steps if step.get("name") == "Publish release assets")

    assert '[[ "$asset" == *.sig ]] || release_assets+=("$asset")' in publish["run"]
    assert '"${release_assets[@]}"' in publish["run"]
    assert "--clobber" not in publish["run"]


def test_publishing_draft_validates_normal_release_without_rebuilding():
    workflow = yaml.safe_load(UPDATER_WORKFLOW.read_text(encoding = "utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["queue"] == "max"

    job = workflow["jobs"]["publish-updater"]
    assert "build" not in workflow["jobs"]
    assert job["permissions"] == {"contents": "write"}
    assert "startsWith(inputs.release_tag, 'v')" in job["if"]
    # The job runs for a mistakenly flagged prerelease so validation fails visibly.
    assert "prerelease" not in job["if"]
    assert not any("actions/checkout" in step.get("uses", "") for step in job["steps"])
    assert any("gh release delete-asset" in step.get("run", "") for step in job["steps"])

    validate = next(
        step for step in job["steps"] if step.get("name") == "Validate updater metadata"
    )
    assert "source-release.json" in validate["run"]
    assert "bundle_name not in release_assets" in validate["run"]
    assert "source.get('prerelease')" in validate["run"]
    assert "'/releases/latest/'" in validate["run"]

    downgrade = next(
        step for step in job["steps"] if step.get("name") == "Prevent GitHub latest downgrade"
    )
    assert "Refusing to replace GitHub latest" in downgrade["run"]
    assert "releases/latest" in downgrade["run"]

    promote = next(
        step
        for step in job["steps"]
        if step.get("name") == "Mark published release as GitHub latest"
    )
    # The API documents make_latest as a string, so -f, not -F.
    assert "-f make_latest=true" in promote["run"]
    assert "releases/latest" in promote["run"]

    bridge = next(
        step
        for step in job["steps"]
        if step.get("name") == "Bridge legacy desktop-latest clients once"
    )
    assert "inputs.bridge_legacy_channel" in bridge["if"]
    # Without it the bridge reads a manifest that the gated download step never fetched.
    assert "steps.gate.outputs.proceed == 'true'" in bridge["if"]
    assert "gh release create desktop-latest" not in bridge["run"]
    assert "gh release upload desktop-latest" in bridge["run"]
    assert "--clobber" in bridge["run"]

    assert "Refusing to move desktop-latest" in bridge["run"]
    assert 'releases/latest" --jq .tag_name' in bridge["run"]

    ordinary_steps = [
        step
        for step in job["steps"]
        if step.get("name") != "Bridge legacy desktop-latest clients once"
    ]
    assert not any(
        "gh release upload desktop-latest" in step.get("run", "") for step in ordinary_steps
    )


def test_the_updater_workflow_skips_releases_without_desktop_bundles():
    job = yaml.safe_load(UPDATER_WORKFLOW.read_text(encoding = "utf-8"))["jobs"]["publish-updater"]
    steps = {step.get("name"): step for step in job["steps"]}
    gate = steps["Check for desktop bundles"]
    assert gate["id"] == "gate"
    assert gate["env"]["REPAIR_POINTER"] == "${{ inputs.repair_pointer }}"
    # A repair dispatch must never be turned away by the state it exists to repair.
    assert "[ \"$REPAIR_POINTER\" = 'true' ]" in gate["run"]
    assert "grep -q '^Unsloth-Desktop-'" in gate["run"]
    # An unreadable release must not look like one that simply has no bundles.
    assert "refusing to advance the channel" in gate["run"]
    # Completeness is judged over the four public downloads, in whichever naming scheme the release was built with.
    # tests/security/test_desktop_updater_pointer.py executes the classification; these only pin that all three steps
    # share it.
    for step_name in (
        "Check for desktop bundles",
        "Validate updater metadata",
        "Mark published release as GitHub latest",
    ):
        run = steps[step_name]["run"]
        for suffix in ("MacOS.dmg", "Linux.AppImage", "Ubuntu.deb", "Windows.exe"):
            assert suffix in run, (step_name, suffix)
        assert "Unsloth-Desktop" in run, step_name
        # Every release published before the rename carries the version in each filename; refusing those would make the
        # workflow unusable on all of them.
        assert "version" in run, step_name

    for name in (
        "Download updater metadata",
        "Validate updater metadata",
        "Remove standalone signature assets",
        "Prevent GitHub latest downgrade",
        "Mark published release as GitHub latest",
    ):
        assert "steps.gate.outputs.proceed == 'true'" in steps[name]["if"], name

    # The v... release is shared, so the sweep must not reach past desktop assets.
    assert 'startswith("Unsloth-Desktop-")' in steps["Remove standalone signature assets"]["run"]


def test_the_updater_workflow_validates_the_target_before_deleting_its_assets():
    """The tag is typed by hand, so a mistyped or mis-flagged one names a real
    older release. Deleting release assets cannot be undone, so every check that
    rejects the target has to run before the sweep, or the rejected release is
    already missing its signatures by the time the run fails."""
    job = yaml.safe_load(UPDATER_WORKFLOW.read_text(encoding = "utf-8"))["jobs"]["publish-updater"]
    order = [step.get("name") for step in job["steps"]]

    remove = order.index("Remove standalone signature assets")
    for name in ("Validate updater metadata", "Prevent GitHub latest downgrade"):
        assert order.index(name) < remove, name
    # Still ahead of the promotion, which points clients here and reads the JSON it refreshes.
    assert remove < order.index("Mark published release as GitHub latest")


def test_the_updater_workflow_is_manual_dispatch_only():
    """It shares a concurrency group with release-desktop.yml, so an auto-fired
    run queues ahead of the desktop build dispatched right after it and stalls
    the release. Nothing may start this workflow except a maintainer."""
    workflow = yaml.safe_load(UPDATER_WORKFLOW.read_text(encoding = "utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_dispatch"}, triggers

    job = workflow["jobs"]["publish-updater"]
    # A leftover github.event.release ref is null under dispatch: silently false, not an error.
    conditions = [job["if"]] + [step["if"] for step in job["steps"] if "if" in step]
    for condition in conditions:
        assert "github.event" not in condition, condition

    # Dropping the release trigger is only safe while the pointer repair stays reachable.
    restore = next(
        step
        for step in job["steps"]
        if step.get("name") == "Restore latest complete Desktop release"
    )
    assert "inputs.repair_pointer" in restore["if"]
    assert "gh release upload" not in restore["run"]
    assert "-f make_latest=true" in restore["run"]
    assert triggers["workflow_dispatch"]["inputs"]["repair_pointer"]["default"] is False
