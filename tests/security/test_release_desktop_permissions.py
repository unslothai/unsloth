"""Permission-boundary checks for the desktop release workflow."""

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
    assert "build" in publish["needs"]

    # The guard moved into a validation step that runs ahead of the VirusTotal
    # scan; creating a missing release is deferred to a separate step so a
    # non-draft release is never published empty for the length of the scan.
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


def test_versioned_release_hides_updater_signature_assets():
    steps = _workflow()["jobs"]["publish-release"]["steps"]
    publish = next(step for step in steps if step.get("name") == "Publish versioned release assets")

    assert '[[ "$asset" == *.sig ]] || release_assets+=("$asset")' in publish["run"]
    assert '"${release_assets[@]}"' in publish["run"]
    assert "--clobber" not in publish["run"]


def test_publishing_draft_validates_normal_release_without_rebuilding():
    workflow = yaml.safe_load(UPDATER_WORKFLOW.read_text(encoding = "utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["release"] == {"types": ["published"]}
    assert "workflow_dispatch" in triggers
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["queue"] == "max"

    job = workflow["jobs"]["publish-updater"]
    assert "build" not in workflow["jobs"]
    assert job["permissions"] == {"contents": "write"}
    assert "startsWith(github.event.release.tag_name, 'v')" in job["if"]
    # The job runs for a mistakenly flagged prerelease so validation fails visibly.
    assert "github.event.release.prerelease" not in job["if"]
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
        step for step in job["steps"] if step.get("name") == "Mark published release as GitHub latest"
    )
    assert "make_latest=true" in promote["run"]
    assert "releases/latest" in promote["run"]

    bridge = next(
        step for step in job["steps"] if step.get("name") == "Bridge legacy desktop-latest clients once"
    )
    assert "workflow_dispatch" in bridge["if"]
    assert "inputs.bridge_legacy_channel" in bridge["if"]
    assert "gh release create desktop-latest" not in bridge["run"]
    assert "gh release upload desktop-latest" in bridge["run"]
    assert "--clobber" in bridge["run"]

    assert "Refusing to move desktop-latest" in bridge["run"]
    assert 'releases/latest" --jq .tag_name' in bridge["run"]

    ordinary_steps = [
        step for step in job["steps"] if step.get("name") != "Bridge legacy desktop-latest clients once"
    ]
    assert not any("gh release upload desktop-latest" in step.get("run", "") for step in ordinary_steps)