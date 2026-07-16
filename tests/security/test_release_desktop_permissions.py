"""Permission-boundary checks for the desktop release workflow."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"


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

    release_step = next(
        step
        for step in publish["steps"]
        if step.get("name") == "Create or validate versioned release"
    )
    assert "gh release view" in release_step["run"]
    assert "--json tagName,isDraft,isPrerelease" in release_step["run"]
