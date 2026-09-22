"""Behavior checks for the final macOS disk image notarization step."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _step(workflow, name):
    steps = workflow["jobs"]["build"]["steps"]
    return next(step for step in steps if step.get("name") == name)


def _step_names(workflow):
    return [step.get("name") for step in workflow["jobs"]["build"]["steps"]]


def _write_fake_command(path: Path, body: str):
    path.write_text("#!/bin/sh\nset -eu\n" + body, encoding = "utf-8")
    path.chmod(0o755)


def _run_script(run: str, env: dict[str, str], cwd: Path):
    return subprocess.run(
        ["bash", "-c", run],
        cwd = cwd,
        env = env,
        text = True,
        capture_output = True,
        check = False,
    )


def _run_credential_check(
    workflow,
    tmp_path: Path,
    *,
    env_overrides: dict[str, str] | None = None,
):
    env = os.environ.copy()
    env.update(
        {
            "APPLE_ID": "masked-apple-id",
            "APPLE_PASSWORD": "masked-password",
            "APPLE_TEAM_ID": "masked-team",
        }
    )
    env.update(env_overrides or {})
    return _run_script(
        _step(workflow, "Check Apple notarization credentials")["run"], env, tmp_path
    )


def _run_notarization_step(
    workflow,
    tmp_path: Path,
    *,
    submit_output: str | None = None,
    fail_submission: bool = False,
    notary_status: str | None = None,
    artifact_paths: str | None = None,
    staple_failures: int = 0,
):
    """Run the step's shell body against stubbed Apple tooling and report the calls it made."""
    dmg = tmp_path / "Final Desktop.dmg"
    dmg.write_bytes(b"signed dmg")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok = True)
    log = tmp_path / "commands.log"
    log.write_text("", encoding = "utf-8")
    # Record the full argv so the assertions below read the flags the step actually passed, not the text of the YAML
    # that produced them.
    _write_fake_command(
        fake_bin / "xcrun",
        """
printf 'xcrun %s\\n' "$*" >> "$COMMAND_LOG"
if [ "$1 $2" = "notarytool submit" ]; then
  printf '%s\\n' "$SUBMIT_OUTPUT"
  if [ "$FAIL_SUBMISSION" = "true" ]; then
    exit 23
  fi
elif [ "$1 $2" = "notarytool log" ]; then
  printf 'issue: The signature does not include a secure timestamp.\\n'
elif [ "$1 $2" = "stapler staple" ]; then
  count=0
  if [ -f "$STAPLE_COUNT" ]; then
    count="$(cat "$STAPLE_COUNT")"
  fi
  count=$((count + 1))
  printf '%s\n' "$count" > "$STAPLE_COUNT"
  if [ "$count" -le "$STAPLE_FAILURES" ]; then
    exit 68
  fi
fi
exit 0
""",
    )
    for name in ("codesign", "spctl"):
        _write_fake_command(fake_bin / name, f"""printf '{name} %s\\n' "$*" >> "$COMMAND_LOG"\n""")
    _write_fake_command(fake_bin / "sleep", 'printf \'sleep %s\\n\' "$*" >> "$COMMAND_LOG"\n')

    status = notary_status or ("Invalid" if fail_submission else "Accepted")
    env = os.environ.copy()
    env.update(
        {
            "APPLE_ID": "masked-apple-id",
            "APPLE_PASSWORD": "masked-password",
            "APPLE_TEAM_ID": "masked-team",
            "ARTIFACT_PATHS": (
                json.dumps([str(dmg)]) if artifact_paths is None else artifact_paths
            ),
            "COMMAND_LOG": str(log),
            "FAIL_SUBMISSION": "true" if fail_submission else "false",
            "STAPLE_COUNT": str(tmp_path / "staple-count"),
            "STAPLE_FAILURES": str(staple_failures),
            "SUBMIT_OUTPUT": (
                json.dumps({"id": "sub-1234", "status": status})
                if submit_output is None
                else submit_output
            ),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    result = _run_script(_step(workflow, "Notarize final macOS disk image")["run"], env, tmp_path)
    commands = log.read_text(encoding = "utf-8").splitlines()
    return result, commands


def _command_names(commands):
    """Reduce recorded argv lines to `tool subcommand` for sequence assertions."""
    names = []
    for line in commands:
        fields = line.split()
        names.append(" ".join(fields[1:3]) if fields[0] == "xcrun" else fields[0])
    return names


def test_notarization_step_runs_after_the_macos_build_and_before_staging():
    workflow = _workflow()
    step = _step(workflow, "Notarize final macOS disk image")
    assert step["if"] == "matrix.platform == 'macos-latest'"
    assert step["env"]["ARTIFACT_PATHS"] == "${{ steps.build_macos.outputs.artifactPaths }}"
    # notarytool's own --timeout only caps the polling, so the step still needs a backstop or a stalled upload holds the
    # serial matrix until GitHub's 6h job limit.
    assert isinstance(step["timeout-minutes"], int)

    names = _step_names(workflow)
    assert names.index("Build macOS app") < names.index("Notarize final macOS disk image")
    assert names.index("Notarize final macOS disk image") < names.index("Stage release assets")


def test_credentials_are_checked_before_the_expensive_build():
    workflow = _workflow()
    check = _step(workflow, "Check Apple notarization credentials")
    assert check["if"] == "matrix.platform == 'macos-latest'"

    names = _step_names(workflow)
    assert names.index("Check Apple notarization credentials") < names.index("Build macOS app")


def test_missing_apple_credentials_fail_the_release_instead_of_skipping(tmp_path):
    workflow = _workflow()
    for missing in ("APPLE_ID", "APPLE_PASSWORD", "APPLE_TEAM_ID"):
        result = _run_credential_check(workflow, tmp_path, env_overrides = {missing: ""})
        assert result.returncode == 1, missing
        assert f"Missing {missing}" in result.stderr

    assert _run_credential_check(workflow, tmp_path).returncode == 0


def test_final_dmg_is_notarized_stapled_and_gatekeeper_checked(tmp_path):
    result, commands = _run_notarization_step(_workflow(), tmp_path)
    assert result.returncode == 0, result.stderr
    assert _command_names(commands) == [
        "codesign",
        "notarytool submit",
        "stapler staple",
        "stapler validate",
        "spctl",
    ]


def test_transient_stapler_failure_is_retried(tmp_path):
    result, commands = _run_notarization_step(_workflow(), tmp_path, staple_failures = 2)

    assert result.returncode == 0, result.stderr
    assert _command_names(commands) == [
        "codesign",
        "notarytool submit",
        "stapler staple",
        "sleep",
        "stapler staple",
        "sleep",
        "stapler staple",
        "stapler validate",
        "spctl",
    ]
    assert [line for line in commands if line.startswith("sleep ")] == ["sleep 15", "sleep 30"]


def test_persistent_stapler_failure_stops_before_validation(tmp_path):
    result, commands = _run_notarization_step(_workflow(), tmp_path, staple_failures = 5)

    assert result.returncode == 68
    assert _command_names(commands) == [
        "codesign",
        "notarytool submit",
        "stapler staple",
        "sleep",
        "stapler staple",
        "sleep",
        "stapler staple",
    ]


def test_submission_carries_every_credential_and_a_bounded_wait(tmp_path):
    _, commands = _run_notarization_step(_workflow(), tmp_path)
    submit = next(line for line in commands if line.startswith("xcrun notarytool submit"))
    fields = submit.split()
    # Without --wait the submission returns before Apple has a verdict and the staple would race it;
    # --timeout then bounds that wait.
    for flag in ("--apple-id", "--password", "--team-id", "--wait", "--timeout"):
        assert flag in fields, submit
    assert fields[fields.index("--timeout") + 1] != "--wait"
    assert str(tmp_path / "Final Desktop.dmg") in submit


def test_rejection_fetches_the_notarization_log_and_skips_stapling(tmp_path):
    result, commands = _run_notarization_step(_workflow(), tmp_path, fail_submission = True)
    assert result.returncode == 23
    assert _command_names(commands) == ["codesign", "notarytool submit", "notarytool log"]

    log_call = commands[-1].split()
    assert "sub-1234" in log_call
    for flag in ("--apple-id", "--password", "--team-id"):
        assert flag in log_call, commands[-1]
    assert "secure timestamp" in result.stderr
    assert "masked-password" not in result.stdout
    assert "masked-password" not in result.stderr


def test_nonaccepted_service_result_fails_even_when_notarytool_exits_zero(tmp_path):
    for status in ("Invalid", "Rejected"):
        case_dir = tmp_path / status
        case_dir.mkdir()
        result, commands = _run_notarization_step(
            _workflow(),
            case_dir,
            notary_status = status,
        )

        assert result.returncode == 1
        assert _command_names(commands) == ["codesign", "notarytool submit", "notarytool log"]
        assert f"status={status}" in result.stderr


def test_rejection_without_a_parseable_submission_id_still_fails(tmp_path):
    result, commands = _run_notarization_step(
        _workflow(),
        tmp_path,
        fail_submission = True,
        submit_output = "Conn close by peer",
    )
    assert result.returncode == 23
    assert _command_names(commands) == ["codesign", "notarytool submit"]


def test_ambiguous_or_missing_disk_image_fails_closed(tmp_path):
    workflow = _workflow()
    for artifact_paths in ("[]", "{}", "not json", json.dumps(["/a/one.dmg", "/a/two.dmg"])):
        result, commands = _run_notarization_step(workflow, tmp_path, artifact_paths = artifact_paths)
        assert result.returncode != 0, artifact_paths
        assert commands == []
