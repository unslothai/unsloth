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
    wait = next(step for step in publish["steps"] if step.get("name") == "Wait for the build matrix")
    wait_run = wait["run"]

    matrix_legs = {
        f"Build {entry['label']}" for entry in build["strategy"]["matrix"]["include"]
    }
    assert len(matrix_legs) == len(tauri_steps)
    for leg in matrix_legs:
        assert f"'{leg}'" in wait_run, leg

    assert "refusing to publish, these build jobs did not succeed" in wait_run
    assert "refusing to publish without confirming they ran" in wait_run
    # Every one of those refusals has to be terminal.
    assert wait_run.count("exit 1") >= 3

    # The wait is a poll of this run's job list, not a mention of an env var.
    # Assert the mechanism, because the error strings above could survive a step
    # that no longer loops or no longer reads a conclusion: the jobs API call,
    # the loop that repeats it, and the two states it decides on. Without these,
    # deleting the polling loop and the status checks while leaving GITHUB_RUN_ID
    # and LEGS in place still reads as a wait, and the download races the matrix.
    assert "actions/runs/${GITHUB_RUN_ID}/jobs" in wait_run
    assert ".status" in wait_run and ".conclusion" in wait_run
    # Not finished yet is "keep waiting"; finished but not `success` is a refusal.
    assert re.search(r'!=\s*"completed"', wait_run), wait_run
    assert re.search(r'!=\s*"success"', wait_run), wait_run
    # A single API read is a snapshot, not a wait: it has to loop and sleep.
    assert re.search(r"^\s*while\b", wait_run, re.MULTILINE), wait_run
    assert re.search(r"^\s*sleep\b", wait_run, re.MULTILINE), wait_run

    names = [step.get("name") for step in publish["steps"]]
    assert names.index("Wait for the build matrix") < names.index("Create versioned release")
    # And it has to clear before the assets are pulled, or the download races the
    # legs and publish-release dies on artifacts that do not exist yet.
    download = next(
        index
        for index, step in enumerate(publish["steps"])
        if step.get("uses", "").startswith("actions/download-artifact@")
    )
    assert names.index("Wait for the build matrix") < download, names

    # Creating a missing release is a separate step gated on validation, so a
    # non-draft release is never reserved before its assets are ready to upload.
    release_step = next(
        step for step in publish["steps"] if step.get("name") == "Validate versioned release state"
    )
    assert "gh release list" in release_step["run"]
    assert "resource_exists" in release_step["run"]

    create_step = next(
        step for step in publish["steps"] if step.get("name") == "Create versioned release"
    )
    assert "gh release create" in create_step["run"]
    assert create_step["if"] == "steps.versioned_release_state.outputs.create == 'true'"


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
    publish = next(step for step in steps if step.get("name") == "Publish versioned release assets")

    assert '[[ "$asset" == *.sig ]] || release_assets+=("$asset")' in publish["run"]
    assert '"${release_assets[@]}"' in publish["run"]
    assert "--clobber" not in publish["run"]


def test_publishing_draft_advances_updater_without_rebuilding():
    workflow = yaml.safe_load(UPDATER_WORKFLOW.read_text(encoding = "utf-8"))
    assert workflow.get("on", workflow.get(True)) == {"release": {"types": ["published"]}}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["queue"] == "max"

    job = workflow["jobs"]["publish-updater"]
    assert "build" not in workflow["jobs"]
    assert job["permissions"] == {"contents": "write"}
    assert "startsWith(github.event.release.tag_name, 'desktop-v')" in job["if"]
    assert not any("actions/checkout" in step.get("uses", "") for step in job["steps"])
    assert any("desktop-latest" in step.get("run", "") for step in job["steps"])
    assert any("gh release delete-asset" in step.get("run", "") for step in job["steps"])

    download = next(
        step for step in job["steps"] if step.get("name") == "Download updater metadata"
    )
    assert "HTTP 404" in download["run"]
    assert "desktop-current" not in download["run"] or "|| true" not in download["run"]

    validate = next(
        step for step in job["steps"] if step.get("name") == "Validate updater metadata"
    )
    assert "source-release.json" in validate["run"]
    assert "bundle_name not in release_assets" in validate["run"]
