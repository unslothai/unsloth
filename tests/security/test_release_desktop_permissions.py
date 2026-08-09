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
    download = next(
        (
            index
            for index, step in enumerate(publish["steps"])
            if step.get("uses", "").startswith("actions/download-artifact@")
        ),
        None,
    )
    assert download is not None, [step.get("name") for step in publish["steps"]]

    # The publish job starts alongside the build matrix rather than through
    # `needs: build`, so the hand-off is a wait step that has to clear before
    # the assets are downloaded. Either shape keeps the ordering guarantee.
    if "build" not in (publish.get("needs") or []):
        wait = next(
            (
                index
                for index, step in enumerate(publish["steps"])
                if "GITHUB_RUN_ID" in step.get("run", "")
            ),
            None,
        )
        assert wait is not None, "publish-release neither needs build nor waits for it"
        assert wait < download, [step.get("name") for step in publish["steps"]]

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


def test_the_wait_step_polls_every_build_matrix_leg_by_its_real_name():
    # The wait step matches jobs by name. Renaming a matrix label, or adding a
    # leg, would otherwise leave it waiting on a job that never reports, or
    # publishing while a leg is still building.
    jobs = _workflow()["jobs"]
    build = jobs["build"]
    if "build" in (jobs["publish-release"].get("needs") or []):
        return

    wait = next(
        step
        for step in jobs["publish-release"]["steps"]
        if "GITHUB_RUN_ID" in step.get("run", "")
    )
    # Leg names carry their own parentheses, so take the LEGS line, not to `)`.
    legs = re.findall(r"'([^']+)'", wait["run"].split("LEGS=(", 1)[1].split("\n", 1)[0])
    template = build["name"]
    expected = [
        template.replace("${{ matrix.label }}", entry["label"])
        for entry in build["strategy"]["matrix"]["include"]
    ]
    assert sorted(legs) == sorted(expected), (legs, expected)


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
