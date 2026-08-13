# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""What the kernel builder generates and what the launcher can read back.

These cover the transport, not the payloads: the generated driver and payload
cells are EXECUTED here with Kaggle replaced by a stub, so a control-flow
mistake in a cell that only ever runs on a Kaggle T4 is caught on a runner.

No network call, no credential and no GPU: `subprocess` is swapped out
wholesale for the driver cells, and the payload cells that are executed stop
before they reach torch.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = REPO_ROOT / "tests" / "kaggle" / "t4_smoke"
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"

sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(CI_DIR))

import build_kernel  # noqa: E402
import launch  # noqa: E402
from legs import LEGS  # noqa: E402


# ------------------------------------------------------------------ driver


class _Stub:
    """Stands in for `subprocess` while the generated driver cells run.

    Answers the four commands the driver issues -- the GPU probe, `which uv`,
    the venv build, and papermill -- and records what papermill was handed,
    which is where the per-payload isolation is either present or not.
    """

    def __init__(
        self,
        *,
        gpus: int,
        venv_ok: bool = True,
    ):
        self.gpus = gpus
        self.venv_ok = venv_ok
        self.papermill: list[dict] = []
        self.TimeoutExpired = subprocess.TimeoutExpired
        self.CalledProcessError = subprocess.CalledProcessError
        self.STDOUT = subprocess.STDOUT

    def run(self, cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[0] == "nvidia-smi":
            if self.gpus < 0:
                raise OSError("nvidia-smi is not on this box")
            out = "".join("Tesla T4, 15360 MiB\n" for _ in range(self.gpus))
            return types.SimpleNamespace(returncode = 0, stdout = out, stderr = "")
        if cmd[0] == "which":
            return types.SimpleNamespace(returncode = 0, stdout = "/usr/bin/uv\n", stderr = "")
        if "papermill" in cmd:
            env = kw.get("env") or {}
            self.papermill.append(
                {
                    "notebook": Path(cmd[cmd.index("papermill") + 1]).name,
                    "cuda": env.get("CUDA_VISIBLE_DEVICES"),
                    "kernel": cmd[cmd.index("-k") + 1],
                    "compile_location": env.get("UNSLOTH_COMPILE_LOCATION"),
                }
            )
            Path(cmd[cmd.index("papermill") + 2]).write_text("{}", encoding = "utf-8")
            return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
        if not self.venv_ok:
            raise subprocess.CalledProcessError(1, cmd)
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")


def _drive(
    tmp_path: Path,
    leg_names,
    *,
    gpus: int,
    venv_ok: bool = True,
) -> dict:
    """Run the generated driver's setup and runner cells against the stub.

    The only edit to the generated source is the `/kaggle/working` literal,
    rewritten to a temp directory so the cells can run off a Kaggle box.
    """
    driver = build_kernel.build_kernel(
        SMOKE_DIR,
        leg_names,
        unsloth_ref = "main",
        zoo_ref = "main",
        extra_args = (),
        per_run_timeout = 60,
        skip_reference = True,
    )
    stub = _Stub(gpus = gpus, venv_ok = venv_ok)
    saved = sys.modules["subprocess"]
    sys.modules["subprocess"] = stub
    namespace: dict = {}
    raised = None
    try:
        for cell in driver["cells"][:2]:
            source = "".join(cell["source"]).replace("/kaggle/working", str(tmp_path))
            try:
                exec(compile(source, "<driver-cell>", "exec"), namespace)
            except SystemExit as exc:
                raised = exc
                break
    finally:
        sys.modules["subprocess"] = saved
    return {
        "stood_down": raised,
        "n_gpu": namespace.get("N_GPU"),
        "papermill": stub.papermill,
        "results": namespace.get("results") or {},
    }


def test_a_gpu_shortfall_stands_the_kernel_down(tmp_path):
    """A missing GPU is infrastructure, not a result.

    `max(1, len(GPUS))` put both payloads on device 0, where each child still
    sees exactly one card and passes its own visibility assertion, so a
    contended OOM came back looking like a code failure.
    """
    driven = _drive(tmp_path, ["control", "canary"], gpus = -1)
    assert driven["stood_down"] is not None, "a 1-GPU allocation ran both payloads anyway"
    assert driven["papermill"] == []


def test_two_gpus_still_run_both_payloads_one_per_card(tmp_path):
    driven = _drive(tmp_path, ["control", "canary"], gpus = 2)
    assert driven["stood_down"] is None
    assert sorted(p["cuda"] for p in driven["papermill"]) == ["0", "1"]


def test_a_payload_whose_venv_failed_is_not_run_in_the_system_kernel(tmp_path):
    """Falling back to `python3` puts both legs' installs in one tree.

    The legs deliberately install different library sets, so a shared
    site-packages does not merely risk corruption, it destroys the
    comparison -- and the resulting import error reads as a code regression.
    """
    driven = _drive(tmp_path, ["control", "canary"], gpus = 2, venv_ok = False)
    assert [
        p["kernel"] for p in driven["papermill"]
    ] == [], "a payload ran in the shared system kernel after its venv failed"
    assert driven["results"], "the skipped payloads left no record"
    for entry in driven["results"].values():
        assert entry["error"]


def test_each_payload_compiles_into_its_own_cache(tmp_path):
    """Concurrent legs must not share `unsloth_compiled_cache`.

    It is a relative path resolved against the working directory, which both
    papermill children inherit, and the legs compile the same modules against
    deliberately different transformers/TRL versions.
    """
    driven = _drive(tmp_path, ["control", "canary"], gpus = 2)
    locations = [p["compile_location"] for p in driven["papermill"]]
    assert all(locations), "no per-payload UNSLOTH_COMPILE_LOCATION was set"
    assert len(set(locations)) == len(locations), f"shared compile cache: {locations}"


def test_the_prune_still_reaches_the_per_payload_directories():
    """Whatever the per-payload names are, the tail cell must still drop them.

    `kernels output` ships the whole of /kaggle/working back over the wire; a
    sweep that missed the venvs once shipped 371MB.
    """
    driver = build_kernel.build_driver({"t4_control.ipynb": {"cells": []}}, 60)
    tail = "".join(driver["cells"][2]["source"])
    assert '"unsloth_compiled_cache*"' in tail or "'unsloth_compiled_cache*'" in tail
    assert '"t4_smoke_src*"' in tail or "'t4_smoke_src*'" in tail


# ----------------------------------------------------------------- payload


def _payload_cells(leg, **kw) -> list[str]:
    notebook = build_kernel.build_payload_notebook(
        SMOKE_DIR, leg, unsloth_ref = "main", zoo_ref = "main", reference = "", **kw
    )
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def test_each_payload_materialises_into_its_own_directory():
    """Two payloads writing one directory can truncate a file the other reads.

    `write_bytes` truncates first, and the two legs carry byte-identical
    copies of the same sources, so the loser of that race imports a partial
    file and dies for a reason that has nothing to do with the commit.
    """
    roots = set()
    for name in ("control", "canary"):
        materialise = _payload_cells(LEGS[name])[0]
        roots.add(materialise.split("ROOT = pathlib.Path(")[1].split(")")[0])
    assert len(roots) == 2, f"both payloads materialise into {roots}"


def test_a_shared_argument_does_not_override_a_legs_own_option():
    """`--smoke-args` is shared so control and canary stay comparable.

    It must not reach a leg that already sets that option: the gpt-oss leg's
    3 steps are a measured fit for a 16GB card, and argparse takes the LAST
    value, so appending the SFT legs' 10 silently retrained the 20B leg.
    """
    run_cell = _payload_cells(LEGS["gptoss"], extra_args = ("--max-steps", "10"))[3]
    argv = run_cell.split("cmd += [")[1].split("]")[0]
    assert argv.count('"--max-steps"') == 1, argv
    assert '"3"' in argv and '"10"' not in argv

    # A leg that does NOT set it still receives the shared value.
    canary = _payload_cells(LEGS["canary"], extra_args = ("--max-steps", "10"))[3]
    assert '"--max-steps", "10"' in canary.split("cmd += [")[1]


def test_a_probe_failure_is_reported_as_a_failed_payload(tmp_path, monkeypatch):
    """A commit that breaks `import unsloth` must not exit green.

    The probe raises before the run cell can write a report, so without a
    report of its own the launcher extracts nothing, calls the run `infra`
    and the workflow passes on a deterministic import regression.
    """
    monkeypatch.setattr(build_kernel, "KERNEL_ROOT", str(tmp_path / "src"))
    leg = LEGS["control"]
    broken = type(leg)(
        **{
            **{field: getattr(leg, field) for field in leg.__dataclass_fields__},
            "imports": ("unsloth_module_the_broken_commit_cannot_import",),
        }
    )
    cells = _payload_cells(broken)

    outputs = []
    for index in (0, 2):
        script = tmp_path / f"cell{index}.py"
        script.write_text(cells[index], encoding = "utf-8")
        proc = subprocess.run(
            [sys.executable, str(script)], capture_output = True, text = True, timeout = 600
        )
        outputs.append(proc.stdout + proc.stderr)
    assert "KAGGLE_T4_CI_PAYLOAD MISSING" in outputs[1]

    # What Kaggle brings back, and what the launcher makes of it.
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "t4_control_output.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "outputs": [{"output_type": "stream", "text": text}]}
                    for text in outputs
                ]
            }
        ),
        encoding = "utf-8",
    )
    reports = launch.extract_reports(evidence)
    assert reports, "the import failure produced no report at all"
    assert reports[0]["passed"] is False
    assert reports[0]["label"] == "control"
    assert any(
        "unsloth_module_the_broken_commit_cannot_import" in f for f in reports[0]["failures"]
    )

    # The resolved versions were computed and printed one line above the
    # failure, and report.version_table reads them off the REPORT rather than
    # off the log -- so without them the summary cannot say which release the
    # red leg had that the green one did not.
    import report as report_module

    assert report_module.resolved_versions(reports[0]), reports[0].keys()
    table = report_module.version_table(
        [reports[0], {"label": "control", "versions_flat": {"transformers": "4.57.6"}}]
    )
    assert any("| package | control |" in line for line in table), table


def test_an_install_that_cannot_be_resolved_is_reported_as_a_failed_payload(tmp_path, monkeypatch):
    """Three exhausted pip attempts used to raise without reporting anything.

    The launcher then sees a leg that produced no report, calls the run
    `partial` or `infra` and exits GREEN, so the job added to catch a broken
    distribution could not fail on one.
    """
    monkeypatch.setattr(build_kernel, "KERNEL_ROOT", str(tmp_path / "src"))
    install = _payload_cells(LEGS["control"])[1]
    script = tmp_path / "install.py"
    script.write_text(
        "import subprocess, time, types\n"
        "subprocess.run = lambda cmd, **kw: types.SimpleNamespace(\n"
        "    returncode=1, stdout='', stderr='ERROR: ResolutionImpossible')\n"
        "time.sleep = lambda _s: None\n" + install,
        encoding = "utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output = True, text = True, timeout = 600
    )
    assert proc.returncode != 0
    assert "KAGGLE_T4_CI_PAYLOAD INSTALL FAILED" in proc.stdout

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "kernel.log").write_text(proc.stdout + proc.stderr, encoding = "utf-8")
    reports = launch.extract_reports(evidence)
    assert reports, "the exhausted install produced no report at all"
    assert reports[0]["label"] == "control"
    assert reports[0]["passed"] is False
    assert any("ResolutionImpossible" in f for f in reports[0]["failures"])


def test_the_install_backs_off_between_attempts():
    """Three immediate retries all land inside the same upstream blip.

    The third failure is what the failed-payload report above is built on, so
    it has to mean "this cannot be resolved" rather than "one bad minute".
    """
    install = _payload_cells(LEGS["control"])[1]
    assert "time.sleep(15 * attempt)" in install


# ------------------------------------------------------------------ launcher


@pytest.mark.parametrize("plain", [False, True])
def test_a_report_reaches_the_launcher_through_kaggles_structured_log(tmp_path, plain):
    """The log fallback exists for the run whose notebook never came back.

    Kaggle hands the log over as a JSON array of stream records, so scanning
    the file as text finds no line that starts with the report prefix and a
    real failure is filed as `infra`.
    """
    payload = {
        "label": "control",
        "model": "unsloth/Qwen2.5-0.5B",
        "passed": False,
        "failures": ["reference band: out of band at step 3"],
    }
    line = launch.RESULT_PREFIX + json.dumps(payload) + "\n"
    body = (
        line
        if plain
        else json.dumps(
            [
                {"stream_name": "stdout", "time": 12.0, "data": "install done\n"},
                {"stream_name": "stdout", "time": 13.0, "data": line},
            ]
        )
    )
    kernel_dir = tmp_path / "unsloth-t4-ci-deadbeef"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.log").write_text(body, encoding = "utf-8")

    reports = launch.extract_reports(tmp_path)
    assert [r["label"] for r in reports] == ["control"]
    assert reports[0]["passed"] is False


def test_a_log_record_that_splits_the_report_is_still_read(tmp_path):
    """Record boundaries are not line boundaries; join before scanning."""
    payload = {"label": "canary", "model": "unsloth/Qwen2.5-0.5B", "passed": True}
    line = launch.RESULT_PREFIX + json.dumps(payload) + "\n"
    half = len(line) // 2
    kernel_dir = tmp_path / "unsloth-t4-ci-cafe"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.log").write_text(
        json.dumps(
            [
                {"stream_name": "stdout", "data": line[:half]},
                {"stream_name": "stdout", "data": line[half:]},
            ]
        ),
        encoding = "utf-8",
    )
    assert [r["label"] for r in launch.extract_reports(tmp_path)] == ["canary"]


def test_every_push_attempt_gets_its_own_slug(monkeypatch):
    """Retrying onto one slug pushes a SECOND session and hides the first.

    A push to an id that already exists creates a new VERSION and starts
    another batch session; it does not supersede the running one. And
    `kernels/output` and `kernels status` never pass a version label, so they
    answer for the latest session only. A retry after a lost response
    therefore reads the wrong execution's evidence while the first keeps
    billing unseen.
    """
    attempts: list[list[str]] = []
    deleted: list[str] = []

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        attempts.append(cmd)
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
            return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
        metadata = json.loads((Path(cmd[cmd.index("-p") + 1]) / "kernel-metadata.json").read_text())
        attempts[-1] = ["push", metadata["id"]]
        if len(deleted) + 1 < 3:
            return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "Connection reset")
        return types.SimpleNamespace(returncode = 0, stdout = "Successfully pushed", stderr = "")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)
    pushed = launch.push(Path(__file__), "someuser", 3600)

    slugs = [a[1] for a in attempts if a[0] == "push"]
    assert len(slugs) == 3, slugs
    assert len(set(slugs)) == 3, f"every retry reused one slug: {slugs}"
    assert pushed["ok"] and pushed["slug"] == slugs[-1]
    # Each earlier attempt may have landed despite the error it reported, so
    # it is deleted before the next one adds a second concurrent session.
    assert deleted == [s for s in slugs[:-1]]
    assert pushed["attempts"] == slugs


def _drive_main(
    monkeypatch,
    tmp_path,
    *,
    push_seconds,
    pushes,
    extra_argv = (),
):
    """Run `launch.main()` end to end with Kaggle replaced by stubs.

    Returns the per-kernel wait budgets, the slugs deleted on the way out and
    the launch result. The clock is fake so a push can be made to burn an
    arbitrary amount of wall time without the test taking any.
    """
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(launch.time, "time", lambda: clock["t"])
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)
    monkeypatch.setattr(launch, "_api", lambda: object())

    outcomes = list(pushes)

    def fake_push(
        notebook,
        user,
        kernel_timeout_sec,
        accelerator = "NvidiaTeslaT4",
    ):
        clock["t"] += push_seconds
        return outcomes.pop(0)

    waits: list[int] = []

    def fake_wait(api, slug, poll_every, max_wait):
        waits.append(max_wait)
        return "COMPLETE"

    deleted: list[str] = []

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(launch, "push", fake_push)
    monkeypatch.setattr(launch, "wait", fake_wait)
    monkeypatch.setattr(
        launch, "fetch_evidence", lambda slug, outdir, timeout = 300: {"notebooks": [], "log": None}
    )
    monkeypatch.setattr(
        launch,
        "extract_reports",
        lambda outdir: [{"label": "control", "model": "m", "passed": True}],
    )
    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.delenv("GITHUB_OUTPUT", raising = False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch.py",
            *[a for i in range(len(pushes)) for a in ("--notebook", f"k{i}.ipynb")],
            "--user",
            "someuser",
            "--outdir",
            str(tmp_path),
            "--expect",
            "1",
            "--max-wait",
            "5400",
            *extra_argv,
        ],
    )
    assert launch.main() == 0
    result = json.loads((tmp_path / "launch_result.json").read_text(encoding = "utf-8"))
    return waits, deleted, result


def test_the_deletion_deadline_covers_the_time_spent_pushing(monkeypatch, tmp_path):
    """A kernel bills from the moment Kaggle accepts it, not from the last push.

    Started after the push loop, the deadline gave the first kernel --max-wait
    on top of however long the SECOND push took to get through its retries:
    45 minutes of throttling turned a 90 minute ceiling into 135 minutes of
    billing, on the far side of the budget the gate reserved for the run.
    """
    waits, _, _ = _drive_main(
        monkeypatch,
        tmp_path,
        push_seconds = 1800.0,
        pushes = [
            {
                "ok": True,
                "slug": "someuser/unsloth-t4-ci-aaaa",
                "attempts": ["someuser/unsloth-t4-ci-aaaa"],
            },
            {
                "ok": True,
                "slug": "someuser/unsloth-t4-ci-bbbb",
                "attempts": ["someuser/unsloth-t4-ci-bbbb"],
            },
        ],
    )
    # 5400s of invocation deadline, 3600s of it spent pushing.
    assert waits == [1800, 1800]


def test_every_slug_a_push_filed_is_deleted_on_the_way_out(monkeypatch, tmp_path):
    """An attempt whose response was lost may be running, and is untracked.

    `push()` records every slug it filed precisely because Kaggle answers an
    accepted push with a 5xx or a reset connection often enough to be a known
    issue. Cleanup read only the ACCEPTED slug, so a failed final attempt
    (which has none) and any earlier attempt whose best-effort discard was
    refused kept a session slot and billed quota unseen.
    """
    _, deleted, result = _drive_main(
        monkeypatch,
        tmp_path,
        push_seconds = 0.0,
        pushes = [
            # Accepted on the third attempt; the first two may still be up.
            {
                "ok": True,
                "slug": "someuser/unsloth-t4-ci-cccc",
                "attempts": [
                    "someuser/unsloth-t4-ci-aaaa",
                    "someuser/unsloth-t4-ci-bbbb",
                    "someuser/unsloth-t4-ci-cccc",
                ],
            },
            # Never accepted, so no `slug` at all -- and the last attempt is
            # exactly the ambiguous one.
            {
                "ok": False,
                "reason": "push_failed",
                "detail": "Connection reset by peer",
                "attempts": ["someuser/unsloth-t4-ci-dddd", "someuser/unsloth-t4-ci-eeee"],
            },
        ],
    )
    assert sorted(deleted) == [
        "someuser/unsloth-t4-ci-aaaa",
        "someuser/unsloth-t4-ci-bbbb",
        "someuser/unsloth-t4-ci-cccc",
        "someuser/unsloth-t4-ci-dddd",
        "someuser/unsloth-t4-ci-eeee",
    ]
    assert all(k["released"] for k in result["kernels"])


def test_the_temp_dir_is_left_alone_when_the_log_is_not_json(tmp_path):
    """A plain-text log, and a JSON object that is not a record array."""
    kernel_dir = tmp_path / "unsloth-t4-ci-beef"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.log").write_text(json.dumps({"log": "nothing here"}), encoding = "utf-8")
    assert launch.extract_reports(tmp_path) == []


def test_a_push_that_runs_out_of_wall_clock_is_a_recorded_failure(monkeypatch):
    """`subprocess.run(timeout=...)` RAISES; it does not return a bad result.

    That exception used to leave `push()` entirely, taking the slugs it had
    filed with it. Those are the most ambiguous slugs there are -- the client
    was killed mid-call, so whether Kaggle accepted the kernel is unknowable
    from here -- and losing them means nothing can ever delete the session
    one of them may have started.
    """
    deleted: list[str] = []

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
            return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
        raise subprocess.TimeoutExpired(cmd, launch.PUSH_SUBPROCESS_TIMEOUT_SEC)

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)

    pushed = launch.push(Path(__file__), "someuser", 3600)
    assert pushed["ok"] is False
    # A timeout is Kaggle under load, so it retries like any other throttle,
    # and every slug it filed comes back for the caller to reconcile.
    assert len(pushed["attempts"]) == launch.PUSH_ATTEMPTS
    assert len(set(pushed["attempts"])) == launch.PUSH_ATTEMPTS
    assert "timed out" in pushed["detail"]
    # Each attempt discards the previous one before adding another session.
    assert deleted == pushed["attempts"][:-1]


def test_a_push_that_times_out_does_not_abandon_the_kernel_already_accepted(monkeypatch, tmp_path):
    """The case that costs quota: kernel 1 is up when kernel 2's push hangs.

    Nothing caught that exception, so `main()` exited without `release()` and
    without writing launch_result.json, and the workflow has no cleanup step
    after this script. The accepted kernel then billed to its own ceiling
    with nobody reading its result -- and Kaggle's push-time timeout has been
    measured not to stop a wedged one.
    """
    deleted: list[str] = []
    pushes = {"n": 0}

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
            return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
        pushes["n"] += 1
        if pushes["n"] == 1:
            return types.SimpleNamespace(
                returncode = 0, stdout = "Kernel version 1 successfully pushed", stderr = ""
            )
        raise subprocess.TimeoutExpired(cmd, launch.PUSH_SUBPROCESS_TIMEOUT_SEC)

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)
    monkeypatch.setattr(launch, "_api", lambda: object())
    monkeypatch.setattr(launch, "wait", lambda api, slug, poll_every, max_wait: "COMPLETE")
    monkeypatch.setattr(
        launch, "fetch_evidence", lambda slug, outdir, timeout = 300: {"notebooks": [], "log": None}
    )
    monkeypatch.setattr(
        launch,
        "extract_reports",
        lambda outdir: [{"label": "control", "model": "m", "passed": True}],
    )
    monkeypatch.delenv("GITHUB_OUTPUT", raising = False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch.py",
            "--notebook",
            str(tmp_path / "k0.ipynb"),
            "--notebook",
            str(tmp_path / "k1.ipynb"),
            "--user",
            "someuser",
            "--outdir",
            str(tmp_path / "ev"),
            "--expect",
            "2",
        ],
    )
    (tmp_path / "k0.ipynb").write_text("{}", encoding = "utf-8")
    (tmp_path / "k1.ipynb").write_text("{}", encoding = "utf-8")

    assert launch.main() == 0
    result = json.loads((tmp_path / "ev" / "launch_result.json").read_text(encoding = "utf-8"))
    accepted = result["kernels"][0]["slug"]
    assert accepted and accepted in deleted
    # Including every slug the timed-out push filed, since any of them may be
    # the session Kaggle accepted and never told us about.
    for slug in result["kernels"][1]["attempted"]:
        assert slug in deleted, slug
    assert all(k["released"] for k in result["kernels"])


def test_an_abort_anywhere_in_the_launcher_still_deletes_what_it_pushed(monkeypatch, tmp_path):
    """The outer guard, not the timeout specifically.

    Every line after the first push can leave a kernel running if it raises,
    and the runner is the only thing that would ever delete it. An abort is
    also `infra` by this file's contract -- nothing was learned about the code
    under test -- so it exits 0 rather than colouring a pull request red.
    """

    def boom(outdir):
        raise MemoryError("the runner ran out")

    deleted: list[str] = []

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(launch, "_api", lambda: object())
    monkeypatch.setattr(
        launch,
        "push",
        lambda notebook, user, kernel_timeout_sec, accelerator = "NvidiaTeslaT4": {
            "ok": True,
            "slug": "someuser/unsloth-t4-ci-abcd",
            "attempts": ["someuser/unsloth-t4-ci-abcd"],
        },
    )
    monkeypatch.setattr(launch, "wait", lambda api, slug, poll_every, max_wait: "COMPLETE")
    monkeypatch.setattr(
        launch, "fetch_evidence", lambda slug, outdir, timeout = 300: {"notebooks": [], "log": None}
    )
    monkeypatch.setattr(launch, "extract_reports", boom)
    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.delenv("GITHUB_OUTPUT", raising = False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch.py",
            "--notebook",
            "k0.ipynb",
            "--user",
            "someuser",
            "--outdir",
            str(tmp_path),
            "--expect",
            "1",
        ],
    )

    assert launch.main() == 0
    assert deleted == ["someuser/unsloth-t4-ci-abcd"]
    result = json.loads((tmp_path / "launch_result.json").read_text(encoding = "utf-8"))
    assert result["verdict"] == "infra"
    assert "MemoryError" in result["reason"]
