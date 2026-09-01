# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU-only unit tests for the Kaggle Unsloth GPU harness.

The payload itself needs a GPU, a browser and an Unsloth install, and none of
that runs here. What does run is everything that decides whether a green tick
means anything: the GPU-offload verdict, the polling predicates that separate
"finished" from "has not started", the adapter check, the generated kernel
notebook, the evidence transport, and the budget numbers in the workflow
header matching the flags underneath them.

Modules under .github/scripts/kaggle_studio_ci are imported by explicit path
rather than by putting that directory on sys.path: it contains a report.py
and a build_kernel.py, so does .github/scripts/kaggle_t4_ci, and
test_t4_smoke_harness.py puts the latter on sys.path for the whole session.
Whichever ran first would win.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_DIR = REPO_ROOT / "tests" / "kaggle" / "studio_gpu"
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_studio_ci"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "kaggle-t4-studio-gpu-ci.yml"

sys.path.insert(0, str(PAYLOAD_DIR))

import gpu_assert  # noqa: E402
import studio_client  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_kernel = _load("studio_ci_build_kernel", CI_DIR / "build_kernel.py")
collect_evidence = _load("studio_ci_collect_evidence", CI_DIR / "collect_evidence.py")




# --------------------------------------------------------------- nvidia-smi
def test_compute_apps_parses_the_bare_csv_the_payload_asks_for():
    apps = gpu_assert.parse_compute_apps("1234, 2048\n5678, 96\n")
    assert apps == {1234: 2048, 5678: 96}


def test_compute_apps_parses_the_same_output_with_units_left_on():
    """Whether nounits is honoured varies by driver, and guessing wrong
    silently yields an empty dict, which reads as 'nothing on the GPU'."""
    assert gpu_assert.parse_compute_apps("1234, 2048 MiB\n") == {1234: 2048}


def test_compute_apps_skips_a_header_and_an_unreported_size():
    text = "pid, used_gpu_memory\n1234, [N/A]\n5678, 512\n"
    assert gpu_assert.parse_compute_apps(text) == {5678: 512}


def test_no_compute_apps_at_all_is_an_empty_dict_not_a_crash():
    assert gpu_assert.parse_compute_apps("") == {}




# ------------------------------------------------------------ llama.cpp log
def test_the_offload_line_is_read_from_the_most_recent_load():
    """A session loads more than once: the chat model, then the export."""
    log = (
        "load_tensors: offloaded 0/29 layers to GPU\n"
        "...\n"
        "load_tensors: offloaded 25/25 layers to GPU\n"
    )
    assert gpu_assert.offloaded_layers(log) == (25, 25)


def test_a_log_with_no_offload_line_reports_nothing_rather_than_zero():
    """None and (0, N) mean opposite things and must not be conflated."""
    assert gpu_assert.offloaded_layers("llama_model_loader: loaded meta data") is None


def test_the_device_model_buffer_is_read_in_mib():
    log = "load_tensors:        CUDA0 model buffer size =   1918.35 MiB\n"
    assert gpu_assert.cuda_buffer_mib(log) == pytest.approx(1918.35)


def test_a_cpu_only_log_reports_no_device_buffer():
    log = "load_tensors:   CPU model buffer size =   1918.35 MiB\n"
    assert gpu_assert.cuda_buffer_mib(log) is None




# ------------------------------------------------------------ install kinds
def test_a_cuda_bundle_is_recognised_from_its_marker(tmp_path):
    """Written against a marker the installer actually produces.

    This test used to write ``{"install_kind": "linux-cuda"}`` and assert it
    came back, which is how the bug survived: the marker on disk has no
    install_kind key, so the assertion agreed with the payload about a field
    neither of them was reading from reality. Six hardware runs reported
    install_kind=None with a working CUDA llama.cpp installed, and this test
    was green for all six.
    """
    marker = tmp_path / "UNSLOTH_PREBUILT_INFO.json"
    marker.write_text(
        json.dumps(
            {
                "tag": "b10360",
                "asset": "app-b10360-mix-87da1a2-linux-x64-cuda13-older.tar.gz",
                "runtime_line": "cuda13",
                "coverage_class": "older",
            }
        )
    )
    assert gpu_assert.install_kind(marker) == "cuda13"
    assert gpu_assert.is_cuda_install(gpu_assert.install_kind(marker))


@pytest.mark.parametrize("kind", ["linux-cpu", "linux-vulkan", "linux-rocm", None, ""])
def test_everything_that_is_not_a_cuda_bundle_is_rejected(kind):
    """A CPU bundle on a box with a working NVIDIA driver is the regression
    this leg exists to see, so it must not read as 'close enough'."""
    assert not gpu_assert.is_cuda_install(kind)


def test_a_missing_or_unreadable_marker_is_not_a_cuda_install(tmp_path):
    assert gpu_assert.install_kind(tmp_path / "absent.json") is None
    broken = tmp_path / "UNSLOTH_PREBUILT_INFO.json"
    broken.write_text("{not json")
    assert gpu_assert.install_kind(broken) is None




# -------------------------------------------------------------- GGUF magic
def test_a_real_gguf_header_passes_and_a_truncated_one_does_not(tmp_path):
    good = tmp_path / "a.gguf"
    good.write_bytes(b"GGUF\x03\x00\x00\x00")
    bad = tmp_path / "b.gguf"
    bad.write_bytes(b"\x00\x00")
    assert gpu_assert.gguf_magic_ok(good)
    assert not gpu_assert.gguf_magic_ok(bad)
    assert not gpu_assert.gguf_magic_ok(tmp_path / "absent.gguf")




# ----------------------------------------------------------- offload verdict
def _verdict(**kw):
    base = {
        "server_pid": None,
        "compute_apps": None,
        "log_text": "",
        "device_vram_delta_mib": None,
        "status": None,
    }
    base.update(kw)
    return gpu_assert.offload_verdict(**base)


def test_no_evidence_at_all_is_a_failure_and_not_a_pass():
    """The asymmetry this whole module exists for. A CPU fallback returns
    text just like the GPU path, so an unanswerable question has not been
    answered in the affirmative."""
    verdict = _verdict()
    assert not verdict["passed"]
    assert any("no probe could show" in f for f in verdict["failures"])


def test_a_matching_process_holding_vram_is_enough_on_its_own():
    verdict = _verdict(server_pid = 42, compute_apps = {42: 2048})
    assert verdict["passed"]
    assert verdict["positives"]


def test_a_process_holding_only_a_cuda_context_is_not_enough():
    """Tens of MiB is what a bare context costs; the weights are elsewhere."""
    verdict = _verdict(server_pid = 42, compute_apps = {42: 30})
    assert not verdict["passed"]


def test_the_llama_cpp_offload_line_alone_is_enough():
    verdict = _verdict(log_text = "load_tensors: offloaded 25/25 layers to GPU")
    assert verdict["passed"]


def test_device_vram_growth_alone_is_enough():
    """The fallback probe for a container where nvidia-smi cannot see the
    process and llama.cpp's log did not reach us."""
    assert _verdict(device_vram_delta_mib = 1500.0)["passed"]


def test_a_small_vram_wobble_is_not_evidence():
    assert not _verdict(device_vram_delta_mib = 20.0)["passed"]


def test_zero_offloaded_layers_fails_however_much_else_looks_right():
    verdict = _verdict(
        server_pid = 42,
        compute_apps = {42: 4096},
        log_text = "load_tensors: offloaded 0/25 layers to GPU",
        device_vram_delta_mib = 4096.0,
    )
    assert not verdict["passed"]
    assert any("0/25" in f for f in verdict["failures"])


def test_a_declared_cpu_fallback_fails_however_much_else_looks_right():
    verdict = _verdict(
        server_pid = 42,
        compute_apps = {42: 4096},
        status = {"cpu_fallback_reason": "vulkan_startup_crash"},
    )
    assert not verdict["passed"]


def test_studio_reporting_zero_gpu_layers_fails():
    verdict = _verdict(server_pid = 42, compute_apps = {42: 4096}, status = {"gpu_layers": 0})
    assert not verdict["passed"]


def test_a_pid_that_is_not_among_the_gpu_processes_is_not_a_pass():
    """Something else on the box holding VRAM must not launder a CPU load."""
    verdict = _verdict(server_pid = 42, compute_apps = {99: 8000})
    assert not verdict["passed"]


def test_auto_mode_gpu_layers_of_minus_one_is_not_treated_as_zero():
    verdict = _verdict(server_pid = 42, compute_apps = {42: 4096}, status = {"gpu_layers": -1})
    assert verdict["passed"]




# ------------------------------------------------------------------ health
def test_health_is_not_ready_until_hardware_detection_settles():
    """Unsloth answers healthy while still detecting, and refuses to start a
    training run or an export in that window."""
    assert not studio_client.health_is_ready({"status": "healthy", "hardware_detecting": True})
    assert studio_client.health_is_ready({"status": "healthy"})
    assert not studio_client.health_is_ready({"status": "starting"})
    assert not studio_client.health_is_ready("connection refused")




# -------------------------------------------------------------------- wait
def test_wait_returns_as_soon_as_the_predicate_holds():
    values = iter([1, 2, 3])
    ok, last, reason = wait_helper(values, lambda v: v == 2)
    assert (ok, last, reason) == (True, 2, "")


def wait_helper(values, accept, **kw):
    clock = {"t": 0.0}

    def _now():
        return clock["t"]

    def _sleep(seconds):
        clock["t"] += seconds

    return studio_client.wait_for(
        probe = lambda: next(values),
        accept = accept,
        deadline_s = kw.pop("deadline_s", 100.0),
        interval_s = 1.0,
        now = _now,
        sleep = _sleep,
        **kw,
    )


def test_wait_times_out_rather_than_looping_forever():
    ok, _, reason = wait_helper(iter(range(1000)), lambda v: False, deadline_s = 5.0)
    assert not ok
    assert "timed out" in reason


def test_wait_gives_up_immediately_when_the_process_it_waits_on_is_dead():
    """Otherwise every crash at startup costs the full deadline and reports
    itself as slowness rather than as a crash."""
    ok, _, reason = wait_helper(iter(range(10)), lambda v: False, alive = lambda: False)
    assert not ok
    assert "exited" in reason


def test_a_probe_that_raises_is_retried_rather_than_fatal():
    """Polling a server that is not listening yet is the normal first case."""
    attempts = {"n": 0}

    def _probe():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionRefusedError("not up yet")
        return "ready"

    clock = {"t": 0.0}
    ok, last, _ = studio_client.wait_for(
        probe = _probe,
        accept = lambda v: v == "ready",
        deadline_s = 60.0,
        interval_s = 1.0,
        now = lambda: clock["t"],
        sleep = lambda s: clock.__setitem__("t", clock["t"] + s),
    )
    assert ok and last == "ready" and attempts["n"] == 3




# ---------------------------------------------------------------- training
def test_a_running_job_is_not_terminal():
    for phase in (
        "idle",
        "loading_model",
        "loading_dataset",
        "configuring",
        "training",
        "finalizing",
    ):
        assert studio_client.training_verdict({"phase": phase}) == (False, "")


def test_a_completed_job_is_terminal_and_clean():
    assert studio_client.training_verdict({"phase": "completed"}) == (True, "")


@pytest.mark.parametrize("phase", ["error", "stopped"])
def test_a_failed_job_stops_the_poll_and_names_the_cause(phase):
    """Waiting out a twenty-minute deadline on a job that already died turns
    a legible failure into a timeout."""
    terminal, reason = studio_client.training_verdict({"phase": phase, "error": "boom"})
    assert terminal
    assert phase in reason and "boom" in reason


def test_steps_with_a_logged_loss_are_counted_and_nulls_are_not():
    status = {"metric_history": {"loss": [1.0, 0.9, None, 0.8]}}
    assert studio_client.trained_steps(status) == 3
    assert studio_client.trained_steps({}) == 0
    assert studio_client.trained_steps({"metric_history": {"loss": "nope"}}) == 0




# ------------------------------------------------------------------ export
def test_an_export_that_has_not_started_is_not_read_as_finished():
    """is_export_active is false before the job starts as well as after it
    ends, and there is no job id, so the sequence number is the only thing
    that distinguishes them."""
    status = {"last_op_seq": 7, "is_export_active": False, "last_op_status": "success"}
    assert studio_client.export_verdict(status, baseline_seq = 7) == (False, "")


def test_an_export_still_running_is_not_finished():
    status = {"last_op_seq": 8, "is_export_active": True, "last_op_status": "success"}
    assert studio_client.export_verdict(status, baseline_seq = 7) == (False, "")


def test_a_new_successful_export_is_finished_and_clean():
    status = {"last_op_seq": 8, "is_export_active": False, "last_op_status": "success"}
    assert studio_client.export_verdict(status, baseline_seq = 7) == (True, "")


@pytest.mark.parametrize("outcome", ["error", "cancelled"])
def test_a_new_failed_export_is_finished_and_names_the_cause(outcome):
    status = {
        "last_op_seq": 8,
        "is_export_active": False,
        "last_op_status": outcome,
        "last_op_error": "conversion wrote no .gguf",
    }
    done, reason = studio_client.export_verdict(status, baseline_seq = 7)
    assert done and outcome in reason and "no .gguf" in reason


def test_the_newest_gguf_is_found_recursively(tmp_path):
    import os
    import time

    old = tmp_path / "sub" / "old.gguf"
    old.parent.mkdir()
    old.write_bytes(b"GGUF")
    new = tmp_path / "new.gguf"
    new.write_bytes(b"GGUF")
    os.utime(old, (time.time() - 600, time.time() - 600))
    assert studio_client.newest_gguf(tmp_path) == new
    assert studio_client.newest_gguf(tmp_path / "absent") is None




# ----------------------------------------------------------------- adapter
def _adapter(
    tmp_path,
    *,
    config = True,
    weights = 100_000,
):
    root = tmp_path / "run"
    root.mkdir()
    if config:
        (root / "adapter_config.json").write_text("{}")
    if weights is not None:
        (root / "adapter_model.safetensors").write_bytes(b"x" * weights)
    return root


def test_a_real_adapter_directory_passes(tmp_path):
    ok, failures, detail = studio_client.adapter_verdict(_adapter(tmp_path))
    assert ok and not failures
    assert detail["adapter_weights"] == "adapter_model.safetensors"


def test_a_run_that_saved_nothing_fails_even_though_it_said_completed(tmp_path):
    ok, failures, _ = studio_client.adapter_verdict(_adapter(tmp_path, weights = None))
    assert not ok
    assert any("adapter weights" in f for f in failures)


def test_a_stub_sized_adapter_fails(tmp_path):
    ok, failures, _ = studio_client.adapter_verdict(_adapter(tmp_path, weights = 12))
    assert not ok
    assert any("byte floor" in f for f in failures)


def test_a_missing_config_fails(tmp_path):
    ok, failures, _ = studio_client.adapter_verdict(_adapter(tmp_path, config = False))
    assert not ok
    assert any("adapter_config.json" in f for f in failures)


def test_no_output_dir_at_all_fails_rather_than_passing_vacuously():
    ok, failures, _ = studio_client.adapter_verdict(None)
    assert not ok and failures


def test_an_output_dir_that_does_not_exist_fails(tmp_path):
    ok, failures, _ = studio_client.adapter_verdict(tmp_path / "never-created")
    assert not ok and failures




# ----------------------------------------------------------- kernel builder
def _build(tmp_path, **kw):
    out = tmp_path / "kernel.ipynb"
    args = [
        sys.executable,
        str(CI_DIR / "build_kernel.py"),
        "--payload-dir",
        str(PAYLOAD_DIR),
        "--out",
        str(out),
        "--unsloth-ref",
        kw.pop("ref", "deadbeef"),
    ]
    for key, value in kw.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    proc = subprocess.run(args, capture_output = True, text = True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def _payload_notebook(driver: dict) -> dict:
    import gzip
    import re

    source = "".join(driver["cells"][0]["source"])
    blob = re.search(r'PAYLOAD = "([^"]+)"', source).group(1)
    return json.loads(gzip.decompress(base64.b64decode(blob)).decode("utf-8"))


def test_the_built_kernel_is_valid_notebook_json_asking_for_a_gpu(tmp_path):
    driver = _build(tmp_path)
    assert driver["nbformat"] == 4
    assert driver["metadata"]["accelerator"] == "GPU"
    assert _payload_notebook(driver)["metadata"]["accelerator"] == "GPU"


def test_every_generated_cell_compiles(tmp_path):
    """The notebook leg lost a whole GPU session to a generated cell that had
    a syntax error, before a single assertion ran."""
    driver = _build(tmp_path)
    for index, cell in enumerate(driver["cells"]):
        compile("".join(cell["source"]), f"driver{index}", "exec")
    for index, cell in enumerate(_payload_notebook(driver)["cells"]):
        compile("".join(cell["source"]), f"payload{index}", "exec")


def test_no_generated_cell_reads_a_name_nothing_defines(tmp_path):
    """Cells share one namespace in execution order, so a name used in cell 4
    has to have been bound by cell 0 through 4 or by a builtin."""
    import builtins
    for cells in (_build(tmp_path)["cells"], _payload_notebook(_build(tmp_path))["cells"]):
        defined = set(dir(builtins))
        for index, cell in enumerate(cells):
            source = "".join(cell["source"])
            tree = compile(source, f"cell{index}", "exec", flags = 0x400, dont_inherit = True)
            del tree
            import ast

            parsed = ast.parse(source)
            loaded, stored = set(), set()
            for node in ast.walk(parsed):
                if isinstance(node, ast.Name):
                    (loaded if isinstance(node.ctx, ast.Load) else stored).add(node.id)
                elif isinstance(node, ast.alias):
                    stored.add((node.asname or node.name).split(".")[0])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    stored.add(node.name)
                elif isinstance(node, ast.arg):
                    stored.add(node.arg)
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    stored.add(node.name)
                elif isinstance(node, (ast.comprehension,)):
                    pass
            defined |= stored
            missing = sorted(loaded - defined)
            assert not missing, f"cell {index} reads undefined {missing}"


def test_the_ref_under_test_is_pinned_into_the_clone(tmp_path):
    source = "\n".join(
        "".join(c["source"]) for c in _payload_notebook(_build(tmp_path, ref = "c0ffee1"))["cells"]
    )
    assert "c0ffee1" in source
    assert "FETCH_HEAD" in source


def test_the_payload_never_puts_its_work_under_kaggle_working(tmp_path):
    """/kaggle/working is 19.5 GB and is also what `kernels output` ships
    home. A torch install landing there fills the disk mid-run."""
    cells = _payload_notebook(_build(tmp_path))["cells"]
    setup = "".join(cells[0]["source"])
    assert 'pathlib.Path.home() / "unsloth_studio_ci"' in setup
    assert "/tmp/unsloth_studio_ci" in setup
    # The only thing allowed under /kaggle/working is the evidence directory.
    for index, cell in enumerate(cells):
        source = "".join(cell["source"])
        for line in source.splitlines():
            if line.lstrip().startswith("#"):
                continue
            if "/kaggle/working" in line:
                assert "studio_gpu_out" in line, f"cell {index}: {line}"


def _payload_source(driver: dict) -> str:
    return "\n".join("".join(c["source"]) for c in _payload_notebook(driver)["cells"])


def test_payload_args_reach_the_payload(tmp_path):
    source = _payload_source(_build(tmp_path, payload_args = "--max-steps 3"))
    assert '["--max-steps", "3"]' in source


def test_the_builder_refuses_a_payload_directory_that_lost_a_file(tmp_path):
    """A rename that breaks the kernel should fail on the runner in seconds,
    not forty minutes into a GPU session."""
    stub = tmp_path / "payload"
    stub.mkdir()
    (stub / "run_studio_gpu.py").write_text("")
    proc = subprocess.run(
        [
            sys.executable,
            str(CI_DIR / "build_kernel.py"),
            "--payload-dir",
            str(stub),
            "--out",
            str(tmp_path / "k.ipynb"),
        ],
        capture_output = True,
        text = True,
    )
    assert proc.returncode != 0
    assert "missing" in proc.stderr


def test_every_file_the_builder_requires_is_actually_in_the_payload_dir():
    for name in build_kernel.PAYLOAD_FILES:
        assert (PAYLOAD_DIR / name).is_file(), name


def test_the_result_prefix_matches_the_shared_launcher():
    """The launcher is reused unchanged; it scrapes this exact prefix."""
    launcher = (REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci" / "launch.py").read_text(
        encoding = "utf-8"
    )
    assert f'RESULT_PREFIX = "{build_kernel.RESULT_PREFIX}"' in launcher
    payload = (PAYLOAD_DIR / "run_studio_gpu.py").read_text(encoding = "utf-8")
    assert f'RESULT_PREFIX = "{build_kernel.RESULT_PREFIX}"' in payload




# ---------------------------------------------------------------- evidence
def _bundle(names: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj = buf, mode = "w:gz") as tar:
        for name, data in names.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _chunk_lines(blob: bytes, size: int = 64) -> list[str]:
    encoded = base64.b64encode(blob).decode("ascii")
    chunks = [encoded[i : i + size] for i in range(0, len(encoded), size)]
    return [
        f"{collect_evidence.EVIDENCE_PREFIX}{i + 1}/{len(chunks)} {c}" for i, c in enumerate(chunks)
    ]


def test_a_bundle_survives_the_round_trip_through_stdout(tmp_path):
    blob = _bundle(
        {"studio_gpu_report.json": b'{"passed": true}', "playwright/01-chat.png": b"\x89PNG"}
    )
    log = tmp_path / "kernel.log"
    log.write_text("noise\n" + "\n".join(_chunk_lines(blob)) + "\nmore noise\n")
    outdir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(CI_DIR / "collect_evidence.py"),
            "--evidence",
            str(log),
            "--outdir",
            str(outdir),
        ],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (outdir / "studio_gpu_report.json").read_bytes() == b'{"passed": true}'
    assert (outdir / "playwright" / "01-chat.png").is_file()


def test_a_truncated_log_is_refused_rather_than_half_unpacked(tmp_path):
    """A bundle reassembled from a truncated log decodes to something, and
    that something is not the evidence."""
    blob = _bundle({"a.txt": b"x" * 4000})
    lines = _chunk_lines(blob)
    log = tmp_path / "kernel.log"
    log.write_text("\n".join(lines[:-1]))
    outdir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(CI_DIR / "collect_evidence.py"),
            "--evidence",
            str(log),
            "--outdir",
            str(outdir),
        ],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0
    assert "incomplete" in proc.stdout
    assert not outdir.exists() or not list(outdir.iterdir())


def test_a_run_that_emitted_no_bundle_is_not_an_error(tmp_path):
    log = tmp_path / "kernel.log"
    log.write_text("nothing to see")
    proc = subprocess.run(
        [
            sys.executable,
            str(CI_DIR / "collect_evidence.py"),
            "--evidence",
            str(log),
            "--outdir",
            str(tmp_path / "out"),
        ],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0
    assert "no evidence bundle" in proc.stdout


@pytest.mark.parametrize("name", ["/etc/passwd", "../../escape.txt"])
def test_a_bundle_cannot_write_outside_the_output_directory(name):
    assert not collect_evidence.is_safe_member(name)
    assert collect_evidence.is_safe_member("playwright/01.png")


def test_chunks_are_reassembled_in_order_regardless_of_the_order_seen():
    chunks, total = collect_evidence.collect_chunks(
        [
            f"{collect_evidence.EVIDENCE_PREFIX}2/2 QQ",
            f"{collect_evidence.EVIDENCE_PREFIX}1/2 PP",
        ]
    )
    assert total == 2
    assert "".join(chunks[i] for i in (1, 2)) == "PPQQ"


def test_a_truncated_duplicate_does_not_overwrite_the_complete_chunk(tmp_path):
    """The executed notebook and the kernel log are two copies of one stdout.
    When Kaggle cuts the log inside a chunk line, the survivor still parses as
    `i/n <payload>`, so overwriting on every sighting replaced the notebook's
    complete chunk with the log's partial one. Every index was then present,
    the missing-chunk guard passed, and the bundle died in base64 with the
    complete source sitting on disk."""
    blob = _bundle({"studio_gpu_report.json": b'{"passed": true}', "shot.png": b"\x89PNG"})
    lines = _chunk_lines(blob)
    assert len(lines) > 2, "need a chunk that is neither first nor last"

    evidence = tmp_path / "kaggle_evidence" / "unsloth-t4-ci-deadbeef"
    evidence.mkdir(parents = True)
    (evidence / "studio_gpu_output.ipynb").write_text(
        json.dumps({"cells": [{"outputs": [{"text": "\n".join(lines)}]}]}), encoding = "utf-8"
    )
    (evidence / "kernel.log").write_text("\n".join(lines[:-2] + [lines[-2][:-9]]), encoding = "utf-8")

    outdir = tmp_path / "studio_evidence"
    proc = subprocess.run(
        [
            sys.executable,
            str(CI_DIR / "collect_evidence.py"),
            "--evidence",
            str(tmp_path / "kaggle_evidence"),
            "--outdir",
            str(outdir),
        ],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (outdir / "studio_gpu_report.json").read_bytes() == b'{"passed": true}'
    assert (outdir / "shot.png").read_bytes() == b"\x89PNG"


def test_a_complete_later_copy_repairs_a_truncated_earlier_one():
    """The cut can land in either copy, so a later sighting that EXTENDS what
    is held is taken."""
    prefix = collect_evidence.EVIDENCE_PREFIX
    chunks, total = collect_evidence.collect_chunks(
        [f"{prefix}1/2 PP\n{prefix}2/2 QQ", f"{prefix}1/2 PPPP\n{prefix}2/2 QQ"]
    )
    assert total == 2
    assert "".join(chunks[i] for i in (1, 2)) == "PPPPQQ"




# ---------------------------------------------------------------- workflow
def _workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _triggers(wf: dict) -> dict:
    """The `on:` block. YAML 1.1 reads a bare `on` as the boolean True."""
    return wf.get("on", wf.get(True))


def test_the_workflow_parses_and_gates_the_expensive_job_on_the_cheap_one():
    jobs = _workflow()["jobs"]
    assert jobs["studio-gpu"]["needs"] == "gate"
    assert "needs.gate.outputs.should_run == 'true'" in jobs["studio-gpu"]["if"]
    assert "fork != true" in jobs["gate"]["if"]
    for job in jobs.values():
        assert job["timeout-minutes"] >= 1


def test_the_workflow_never_cancels_a_run_that_may_hold_a_kernel():
    wf = _workflow()
    assert wf["concurrency"]["cancel-in-progress"] is False
    assert wf["jobs"]["studio-gpu"]["concurrency"]["cancel-in-progress"] is False


def test_the_two_kaggle_legs_fit_the_account_side_by_side():
    """One kernel each, against a 2-kernel per-ACCOUNT cap, so they can overlap.

    They used to SHARE a concurrency group, because the notebook leg pushed two
    kernels and took both of Kaggle's slots: this leg would have raced the cap
    and lost its push, so it had to queue behind instead. The cost was that it
    queued behind the whole notebook JOB even though the account was free again
    the moment that job's kernels finished -- measured, run 32607617804 waited
    about 40 minutes on run 32607621452.

    The notebook leg now packs its four legs into one kernel. So the invariant
    worth holding is no longer "same group" but the arithmetic that made the
    same group necessary: what each leg PUSHES has to sum to within the cap.
    Separate groups plus one kernel each is exactly 2 of 2, with no headroom,
    which is why this asserts the sum rather than the group names.
    """
    yaml = pytest.importorskip("yaml")
    # Read the cap out of gate.py's SOURCE rather than importing it.
    # Both .github/scripts/kaggle_studio_ci and .github/scripts/kaggle_t4_ci ship a module called `report`, so putting
    # either on sys.path here decides which one `import report` resolves to for every test that runs afterwards in the
    gate_src = (REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci" / "gate.py").read_text(
        encoding = "utf-8"
    )
    caps = re.findall(r"^MAX_CONCURRENT_GPU_KERNELS = (\d+)$", gate_src, re.MULTILINE)
    assert len(caps) == 1, caps
    MAX_CONCURRENT_GPU_KERNELS = int(caps[0])

    notebook_text = (REPO_ROOT / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml").read_text(
        encoding = "utf-8"
    )
    notebook = yaml.safe_load(notebook_text)
    studio_group = _workflow()["jobs"]["studio-gpu"]["concurrency"]["group"]
    notebook_group = notebook["jobs"]["t4-smoke"]["concurrency"]["group"]

    assert studio_group != notebook_group, (
        "the two legs share a concurrency group again, so Unsloth waits out the "
        "whole notebook job for a Kaggle session that is free"
    )
    # Neither may be keyed on the ref: one account, so two branches of the SAME workflow still must not overlap even
    # though the two workflows may.
    for group in (studio_group, notebook_group):
        assert "github.ref" not in group, group

    pushes = {}
    for label, text in (
        ("studio", WORKFLOW.read_text(encoding = "utf-8")),
        ("notebook", notebook_text),
    ):
        found = {int(k) for k in re.findall(r"--kernels (\d+)", text)}
        assert len(found) == 1, (label, found)
        pushes[label] = found.pop()
    assert sum(pushes.values()) <= MAX_CONCURRENT_GPU_KERNELS, pushes


def test_the_workflow_is_never_preempted_by_the_capacity_sweeper():
    """Cancelling it orphans a Kaggle kernel that then bills to its ceiling."""
    preempt = json.loads((REPO_ROOT / ".github" / "ci-preempt.json").read_text(encoding = "utf-8"))
    assert WORKFLOW.name in preempt["never"]
    for machines in preempt["heavy"].values():
        assert WORKFLOW.name not in machines


def test_the_paths_filter_covers_everything_the_payload_depends_on():
    """A change to the payload, the builder or the shared launcher has to be
    able to trigger the job that exercises it."""
    triggers = _triggers(_workflow())
    for event in ("pull_request", "push"):
        paths = triggers[event]["paths"]
        assert "tests/kaggle/studio_gpu/**" in paths
        assert ".github/scripts/kaggle_studio_ci/**" in paths
        assert ".github/scripts/kaggle_t4_ci/**" in paths
        assert "tests/studio/playwright_chat_ui.py" in paths
        assert WORKFLOW.name in " ".join(paths)


def test_the_paths_filter_is_not_the_whole_of_studio():
    """studio/** would roughly double the eligible stream for coverage this
    job cannot see, and the budget arithmetic in the header depends on it."""
    triggers = _triggers(_workflow())
    for event in ("pull_request", "push"):
        assert "studio/**" not in triggers[event]["paths"]


def test_the_sampling_rate_matches_the_arithmetic_in_the_header():
    """The header states 5%, ~38 launches and ~28 GPU-h a week. If the flag
    and the prose disagree, one of them is a lie to whoever reads it next."""
    source = WORKFLOW.read_text(encoding = "utf-8")
    assert "--percent 5" in source
    assert "x sampling rate            0.05" in source
    assert "= launches                 ~38 / week" in source
    assert "= EXPECTED SPEND           ~28 GPU-h / week" in source


def test_studio_is_sampled_harder_than_the_notebook_leg():
    """Harder on every axis the budget is denominated in -- launches, hours
    and share of the allowance -- and LOWER on the one axis that looks like
    the answer, the percentage.

    The inversion is structural, so it is asserted rather than tolerated. A
    point of rate costs ~570 x 0.01 GPU-h here against ~58 x 0.01 there, and
    a future edit that "fixes" the percentages to agree would either starve
    the notebook leg or blow the account. If that edit is ever the right one,
    this test is where the reasoning has to be argued with.
    """
    studio = WORKFLOW.read_text(encoding = "utf-8")
    notebook = (REPO_ROOT / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml").read_text(
        encoding = "utf-8"
    )

    assert "--percent 5" in studio and "--percent 15" in notebook
    assert "= EXPECTED SPEND           ~28 GPU-h / week" in studio
    assert "busy week   231 x 0.15 x 0.25 =   8.7 GPU-h" in notebook
    assert (
        "0.75 GPU-h" in studio and "TOTAL, expected                             ~0.25 h" in notebook
    )

    # Share of the shared 50h CI allowance, and the stand-down floor that enforces the priority:
    assert "The split is Unsloth 35, this leg 15" in notebook
    assert "--reserve-hours 20" in notebook and "--reserve-hours 10" in studio

    # The Unsloth block states the notebook leg's reserve in PROSE, so the number lives in two files and one of them is
    assert "reserve-hours is 10 rather than the notebook leg's 20" in studio

    # Both budget blocks must name the other leg.
    assert "kaggle-t4-studio-gpu-ci.yml" in notebook
    assert "kaggle-t4-notebook-ci.yml" in studio


def test_the_two_legs_together_fit_inside_the_ci_allowance():
    """Re-derived from the two headers rather than trusting either total."""
    import re

    studio = WORKFLOW.read_text(encoding = "utf-8")
    notebook = (REPO_ROOT / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml").read_text(
        encoding = "utf-8"
    )

    def rate(text):
        # The invoked flag, not the prose that quotes it.
        found = re.findall(r"^\s+--percent (\d+) \\$", text, re.M)
        assert len(found) == 1, found
        return int(found[0]) / 100.0

    studio_spend = 760 * rate(studio) * 0.75
    notebook_spend = 231 * rate(notebook) * 0.25
    assert studio_spend > notebook_spend
    assert studio_spend + notebook_spend <= 50.0
    # And with margin, because the ceiling is enforced by a quota read that only sees the account AFTER the hours are
    assert studio_spend + notebook_spend <= 40.0


def test_the_reserve_leaves_ci_the_fifty_hours_it_is_allowed():
    source = WORKFLOW.read_text(encoding = "utf-8")
    assert "--reserve-hours 10" in source
    assert "--budget-hours 2" in source


def test_the_budget_hours_flag_covers_the_kernel_ceiling():
    """budget-hours is the worst case one invocation can cost, and the
    kernel ceiling is what enforces that worst case."""
    source = WORKFLOW.read_text(encoding = "utf-8")
    assert "--kernel-timeout-sec 4200" in source
    ceiling_hours = 4200 / 3600
    assert ceiling_hours <= 2.0


def test_the_opt_in_label_the_summary_names_is_the_one_the_gate_reads():
    source = WORKFLOW.read_text(encoding = "utf-8")
    assert "--label-name kaggle-studio-gpu-ci" in source
    assert "kaggle-studio-gpu-ci" in (CI_DIR / "report.py").read_text(encoding = "utf-8")


def test_skipping_the_ui_driver_is_explicit_and_warns():
    source = WORKFLOW.read_text(encoding = "utf-8")
    assert "::warning title=UI driver disabled" in source
    assert source.count("--skip-ui") == 1


def test_the_gate_and_launcher_are_the_shared_ones():
    """Reused, not forked. A second copy of the quota and concurrency policy
    is a second copy that can drift out of agreement with the account."""
    source = WORKFLOW.read_text(encoding = "utf-8")
    assert ".github/scripts/kaggle_t4_ci/gate.py" in source
    assert ".github/scripts/kaggle_t4_ci/launch.py" in source
    assert not (CI_DIR / "gate.py").exists()
    assert not (CI_DIR / "launch.py").exists()




_PASSING = [{"label": "studio-gpu", "passed": True, "assertions": []}]
_FAILING = [{"label": "studio-gpu", "passed": False, "failures": ["x"], "assertions": []}]
# A training leg from the merged kernel. Not this reporter's payload.
_LEG = [{"label": "control", "passed": False, "steps": []}]


# ------------------------------------------------------------------ report
@pytest.mark.parametrize(
    ("verdict", "reports", "expected_exit"),
    [
        ("pass", _PASSING, 0),
        ("partial", [], 0),
        ("infra", [], 0),
        ("fail", _FAILING, 1),
        # The kernel verdict is the WHOLE kernel's, and since --with-studio
        # that kernel also carries four training legs. A leg failing must not
        # print "Studio GPU smoke: FAIL" over a passing Studio payload and send
        # someone to read the wrong half: the T4 reporter is what turns that
        # red, in the same job, from the same evidence directory.
        ("fail", _PASSING + _LEG, 0),
    ],
)
def test_only_a_real_assertion_failure_turns_the_job_red(tmp_path, verdict, reports, expected_exit):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "launch_result.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "reason": "test",
                "slug": "u/s",
                "kernel_state": "COMPLETE",
                "reports": reports,
            }
        )
    )
    proc = subprocess.run(
        [sys.executable, str(CI_DIR / "report.py"), "--evidence", str(evidence)],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == expected_exit, proc.stdout


def test_a_run_that_reported_nothing_still_names_its_cause(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    proc = subprocess.run(
        [sys.executable, str(CI_DIR / "report.py"), "--evidence", str(evidence)],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0
    assert "nothing is known about the code" in proc.stdout


def test_the_summary_says_what_each_assertion_is_worth(tmp_path):
    """A tick nobody can interpret is a tick nobody should trust."""
    report = _load("studio_ci_report", CI_DIR / "report.py")
    lines = "\n".join(
        report.render(
            {
                "label": "studio-gpu",
                "seconds": 2000,
                "environment": {"gpu_name": "Tesla T4", "llama_cpp_install_kind": "linux-cuda"},
                "config": {"chat_model": "m", "max_steps": 8},
                "assertions": [
                    {
                        "name": "gpu_inference",
                        "passed": True,
                        "evidence": ["nvidia-smi compute-apps: pid 42 holds 2048 MiB"],
                    },
                    {
                        "name": "lora_training",
                        "passed": False,
                        "output_dir": "/x",
                        "phase": "completed",
                        "steps_with_loss": 0,
                    },
                ],
                "failures": ["lora_training: logged a loss for only 0 of 8 steps"],
            }
        )
    )
    assert "not on a CPU fallback" in lines
    assert "AND left an adapter on disk" in lines
    assert "2048 MiB" in lines
    assert "**FAIL**" in lines


def test_the_reporter_survives_the_shared_module_being_unavailable(monkeypatch, tmp_path):
    """kaggle_t4_ci/report.py is owned elsewhere and under active change; the
    verdict must not depend on being able to import it."""
    report = _load("studio_ci_report_2", CI_DIR / "report.py")
    monkeypatch.setattr(report, "_SHARED", tmp_path / "gone.py")
    assert report._load_shared() is None




# ------------------------------------------------- the forced password change
class _RecordingStudio(studio_client.Studio):
    """An Unsloth whose HTTP layer is a script, so login can be driven off-box."""

    def __init__(self, responses):
        super().__init__(base_url = "http://127.0.0.1:0")
        self._responses = list(responses)
        self.calls = []

    def request(
        self,
        method,
        path,
        body = None,
        **kw,
    ):
        self.calls.append((method, path, body, kw.get("auth", True)))
        return self._responses.pop(0)


def _login_ok(must_change):
    return (200, {"access_token": "bootstrap-token", "must_change_password": must_change})


def test_login_retires_a_bootstrap_password_studio_says_must_change():
    """The failure this exists for: the first hardware run authenticated fine and
    then got 403 "Password change required" from /api/inference/load and
    /api/train/start, so inference, tool calling, training and export were all
    unmeasured behind a login step that reported success."""
    studio = _RecordingStudio(
        [
            _login_ok(True),
            (200, {"access_token": "post-change-token"}),
        ]
    )
    studio.login("bootstrap-secret")

    assert [c[1] for c in studio.calls] == ["/api/auth/login", "/api/auth/change-password"]
    # The session carries the token the change minted, not the one that cannot act.
    assert studio.token == "post-change-token"

    change_body = studio.calls[1][2]
    assert change_body["current_password"] == "bootstrap-secret"
    assert (
        change_body["new_password"] != "bootstrap-secret"
    ), "the route rejects an unchanged password"
    assert not any(
        ch.isspace() for ch in change_body["new_password"]
    ), "the route rejects whitespace"
    # Authenticated, or change-password 401s rather than 403s.
    assert studio.calls[1][3] is True


def test_login_leaves_a_password_alone_when_studio_does_not_ask():
    """An account already past the gate must not have its password rotated as a
    side effect of logging in."""
    studio = _RecordingStudio([_login_ok(False)])
    studio.login("already-changed")
    assert [c[1] for c in studio.calls] == ["/api/auth/login"]
    assert studio.token == "bootstrap-token"


def test_a_refused_password_change_is_raised_rather_than_carried_on_with():
    """Continuing with a token that cannot act is how the first run produced four
    unmeasured assertions and no explanation."""
    studio = _RecordingStudio([_login_ok(True), (400, {"detail": "nope"})])
    with pytest.raises(studio_client.StudioError) as excinfo:
        studio.login("bootstrap-secret")
    assert "400" in str(excinfo.value)
    assert "bootstrap-secret" not in str(excinfo.value), "no credential in the error"


def test_a_password_change_without_a_token_is_refused():
    studio = _RecordingStudio([_login_ok(True), (200, {"detail": "ok"})])
    with pytest.raises(studio_client.StudioError) as excinfo:
        studio.login("bootstrap-secret")
    assert "no access_token" in str(excinfo.value)


def test_the_session_remembers_which_password_is_current():
    """The Playwright driver rotates the password itself and asserts the old one
    stops working, so it needs whatever the session is CURRENTLY authenticated
    by. Reading the seeded file instead is how kernel unsloth-t4-ci-9ddd8ae4
    failed the driver with "the bootstrap password is gone" in the same run
    where retiring it fixed inference, tool calling and training."""
    changed = _RecordingStudio([_login_ok(True), (200, {"access_token": "t2"})])
    changed.login("bootstrap-secret")
    assert changed.password not in (None, "", "bootstrap-secret")
    assert changed.password == changed.calls[1][2]["new_password"]

    untouched = _RecordingStudio([_login_ok(False)])
    untouched.login("already-changed")
    assert untouched.password == "already-changed"


def test_the_ui_driver_gets_a_freshly_seeded_account():
    """The API path and the UI driver want OPPOSITE auth states, so the payload
    has to re-seed between them.

    authenticate() retires the bootstrap password to get past Unsloth's forced
    change; the driver's first UI step waits for #new-password on that very
    form. Three hardware runs walked the whole cycle -- 412345d2 failed the API
    assertions on the gate, 9ddd8ae4 fixed those and failed the driver on a
    stale password, the next failed the driver on the form being gone. The fix
    is a restart, because start_server() removes $STUDIO_HOME/auth and that is
    what re-seeds the bootstrap password.

    Asserted on the source rather than by running it: the restart is the whole
    fix, and a refactor that drops it would put the payload straight back to a
    driver that cannot find the form."""
    source = (PAYLOAD_DIR / "run_studio_gpu.py").read_text(encoding = "utf-8")
    body = source[source.index("def assert_chat_ui") :]
    body = body[: body.index("\n    def ")] if "\n    def " in body else body
    assert (
        "self.stop_server()" in body and "self.start_server()" in body
    ), "the driver needs a re-seeded account, which only a restart provides"
    # And it must hand over the RE-SEEDED value, not the retired session's.
    assert "self.remember_bootstrap()" in body
    assert (
        "self.studio.password" not in body
    ), "the retired password is exactly what the driver cannot use"
    assert body.index("self.start_server()") < body.index(
        '"STUDIO_OLD_PW"'
    ), "the password must be read after the restart, or it is the old one"


# The llama.cpp install step.
# Four hardware runs reported install_kind=None and failed the export assertion for it, because nothing had ever
# installed a llama.cpp under STUDIO_HOME.
# install_llama_prebuilt.py resolves a real "linux-cuda" kind on an x64 CUDA host, so the bundle was available the whole
# time and simply never fetched.


# time and simply never fetched.
# The llama.cpp install step.
def _load_payload():
    """Import run_studio_gpu under a private name.

    The module runs a CLI at import time only under __main__, so importing it
    is safe, but it must not collide with the copy test_t4_smoke_harness puts
    on sys.path.
    """
    spec = importlib.util.spec_from_file_location(
        "_studio_gpu_payload_under_test", PAYLOAD_DIR / "run_studio_gpu.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _session(module, tmp_path, **overrides):
    # Built from the REAL parser rather than a hand-listed Namespace. A
    # hand-listed one carries exactly the attributes someone remembered, so
    # every new flag breaks these tests with an AttributeError that says
    # nothing about the flag -- which is what --studio-password did.
    args = module.parse_args(
        [
            "--outdir",
            str(tmp_path / "out"),
            "--repo-root",
            str(tmp_path / "repo"),
            "--studio-home",
            str(tmp_path / "home"),
        ]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    session = module.Payload.__new__(module.Payload)
    session.repo_root = Path(args.repo_root)
    session.studio_home = Path(args.studio_home)
    session.args = args
    session.assertions = []
    session.failures = []
    session.secrets = set()
    return session


def test_the_llama_cpp_install_actually_invokes_the_installer(tmp_path, monkeypatch):
    """Exercises the call, not just its source text.

    `run()` in this payload already applies capture_output and text, so passing
    either again is a TypeError -- and one that only appears on hardware, since
    every other test here reads the file rather than calling it. That is exactly
    how it got written wrong the first time.
    """
    module = _load_payload()
    session = _session(module, tmp_path)
    installer = session.repo_root / "studio" / "install_llama_prebuilt.py"
    installer.parent.mkdir(parents = True)
    installer.write_text("")
    install_dir = session.studio_home / "llama.cpp"
    install_dir.mkdir(parents = True)
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"asset": "app-b1-linux-x64-cuda13-older.tar.gz", "runtime_line": "cuda13"})
    )

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["kw"] = kw
        return subprocess.CompletedProcess(cmd, 0, "linux_cuda_selection: ok", "")

    monkeypatch.setattr(module, "run", fake_run)
    assert session.install_llama_cpp() is True

    assert str(installer) in seen["cmd"]
    assert "--install-dir" in seen["cmd"]
    assert str(install_dir) in seen["cmd"]
    # The TypeError guard: `run` supplies these itself.
    assert "capture_output" not in seen["kw"]
    assert "text" not in seen["kw"]
    assert seen["kw"]["timeout"] == module.LLAMA_CPP_INSTALL_TIMEOUT_S

    recorded = [entry for entry in session.assertions if entry["name"] == "llama_cpp_install"]
    assert len(recorded) == 1
    assert recorded[0]["llama_cpp_install_kind"] == "cuda13"
    # install.sh --local claims to put a llama.cpp on disk.
    assert "install_kind_before" in recorded[0]


def test_a_successful_installer_that_picks_a_cpu_bundle_is_still_a_failure(tmp_path, monkeypatch):
    """The selection regression this leg exists to catch.

    Worded separately from "the installer failed", because a CPU bundle chosen
    ON PURPOSE by a working installer on a CUDA box is a different bug from an
    installer that could not run.
    """
    module = _load_payload()
    session = _session(module, tmp_path)
    installer = session.repo_root / "studio" / "install_llama_prebuilt.py"
    installer.parent.mkdir(parents = True)
    installer.write_text("")
    install_dir = session.studio_home / "llama.cpp"
    install_dir.mkdir(parents = True)
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"asset": "app-b1-linux-x64-cpu.tar.gz", "runtime_line": "cpu"})
    )

    monkeypatch.setattr(
        module, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", "")
    )
    assert session.install_llama_cpp() is False
    recorded = [e for e in session.assertions if e["name"] == "llama_cpp_install"][0]
    assert any("succeeded but selected" in f for f in recorded["failures"])


def test_a_failed_llama_cpp_install_does_not_stop_the_run():
    """Every other assertion still has to execute and report.

    A box where the bundle will not install should produce the same honest
    export red it did before, not a run that stops at the install.
    """
    source = (PAYLOAD_DIR / "run_studio_gpu.py").read_text(encoding = "utf-8")
    body = source[source.index("def execute(self)") :]
    body = body[: body.index("\n    def ")]
    assert "self.install_llama_cpp()" in body
    assert "if not self.install_llama_cpp()" not in body, (
        "a llama.cpp install failure now aborts the run, so a box that cannot "
        "install the bundle reports nothing about inference, training or the UI"
    )
    # And it must happen before the server, so the export route never sees a
    assert body.index("self.install_llama_cpp()") < body.index("self.start_server()")


def test_the_llama_cpp_marker_falls_back_to_the_canonical_location(tmp_path, monkeypatch):
    """Five runs reported install_kind=None with a llama.cpp sitting on disk.

    install.sh --local puts one at ~/.unsloth/llama.cpp, which is what
    install_llama_prebuilt.py's own default resolves to, and this payload read
    STUDIO_HOME/llama.cpp -- a path nothing ever wrote to. The installer said
    as much when asked to install again: "existing llama.cpp install already
    matches selected release b10360-mix-87da1a2; skipping download and
    install", while the directory it was pointed at stayed empty.
    """
    module = _load_payload()
    fake_home = tmp_path / "home"
    canonical = fake_home / ".unsloth" / "llama.cpp"
    canonical.mkdir(parents = True)
    (canonical / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"asset": "app-b1-linux-x64-cuda13-older.tar.gz", "runtime_line": "cuda13"})
    )
    monkeypatch.setattr(module.Path, "home", staticmethod(lambda: fake_home))

    studio_home = tmp_path / "studio_home"
    studio_home.mkdir()
    found = module.llama_cpp_marker(studio_home)
    assert found is not None, "the canonical ~/.unsloth/llama.cpp was not consulted"
    assert module.install_kind(found) == "cuda13"


def test_an_explicit_studio_home_install_wins_over_the_canonical_one(tmp_path, monkeypatch):
    """Most specific first. A caller who installed into STUDIO_HOME on purpose
    must not be answered with whatever is in the shared location."""
    module = _load_payload()
    fake_home = tmp_path / "home"
    canonical = fake_home / ".unsloth" / "llama.cpp"
    canonical.mkdir(parents = True)
    (canonical / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"asset": "app-b1-linux-x64-cpu.tar.gz", "runtime_line": "cpu"})
    )
    monkeypatch.setattr(module.Path, "home", staticmethod(lambda: fake_home))

    studio_home = tmp_path / "studio_home"
    (studio_home / "llama.cpp").mkdir(parents = True)
    (studio_home / "llama.cpp" / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"asset": "app-b1-linux-x64-cuda13-older.tar.gz", "runtime_line": "cuda13"})
    )
    assert module.install_kind(module.llama_cpp_marker(studio_home)) == "cuda13"


def test_no_llama_cpp_anywhere_is_still_reported_as_absent(tmp_path, monkeypatch):
    """The fallback must not turn a real absence into a guess."""
    module = _load_payload()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(module.Path, "home", staticmethod(lambda: fake_home))
    studio_home = tmp_path / "studio_home"
    studio_home.mkdir()
    assert module.llama_cpp_marker(studio_home) is None
    assert module.install_kind(None) is None


def test_the_marker_never_carried_an_install_kind(tmp_path):
    """The bug behind six runs of install_kind=None, pinned.

    install_llama_prebuilt.py writes `install_kind` only into the JSON its
    resolver prints to stdout. The marker it writes to disk records asset, tag,
    runtime_line, coverage_class and bundle_profile. Reading `install_kind`
    from the marker therefore answered None for every bundle on every box,
    including a working CUDA one, and the export assertion failed on that.
    """
    module = _load_payload()
    marker = tmp_path / "UNSLOTH_PREBUILT_INFO.json"
    # Exactly the shape the installer writes:
    marker.write_text(
        json.dumps(
            {
                "requested_tag": "b10360",
                "tag": "b10360",
                "release_tag": "b10360-mix-87da1a2",
                "asset": "app-b10360-mix-87da1a2-linux-x64-cuda13-older.tar.gz",
                "runtime_line": "cuda13",
                "coverage_class": "older",
                "bundle_profile": "mix",
            }
        )
    )
    kind = module.install_kind(marker)
    assert kind is not None, (
        "a marker with no install_kind key must still say what was installed; "
        "answering None here is what failed six hardware runs"
    )
    assert module.is_cuda_install(kind)


def test_a_non_cuda_runtime_line_is_not_a_cuda_install():
    """The assertion still has to be able to fail."""
    module = _load_payload()
    for kind in ("cpu", "vulkan", "rocm", "metal", None, ""):
        assert not module.is_cuda_install(kind), f"{kind!r} classified as CUDA"


def test_a_future_cuda_major_is_recognised():
    """Matching a fixed set of names fails closed on the next CUDA major,
    reporting a working install as not-CUDA -- the same failure this fix is
    for. The match is on the runtime line's shape instead."""
    module = _load_payload()
    assert module.is_cuda_install("cuda14")
    assert module.is_cuda_install("app-bX-linux-x64-cuda99-portable.tar.gz")
    # And not on a word that merely contains "cuda".
    assert not module.is_cuda_install("cudart")


class _FakeStudio:
    """Records calls and answers /api/inference/status from a script."""

    def __init__(
        self,
        status_body,
        *,
        unload_raises = False,
    ):
        self.status_body = status_body
        self.unload_raises = unload_raises
        self.posts = []

    def get(self, path, **kw):
        if path == "/api/inference/status":
            return 200, self.status_body
        return 404, {}

    def post(self, path, body, **kw):
        self.posts.append((path, body))
        if self.unload_raises:
            raise studio_client.StudioError("unload refused")
        return 200, {}


def _baseline_session(
    module,
    tmp_path,
    status_body,
    readings,
    *,
    unload_raises = False,
):
    session = _session(module, tmp_path, load_timeout = 30.0)
    session.studio = _FakeStudio(status_body, unload_raises = unload_raises)
    seq = list(readings)
    session._readings = seq
    return session


def test_the_baseline_waits_for_the_old_model_to_actually_leave(tmp_path, monkeypatch):
    """The run-7 shape, in miniature.

    A 3004 MiB chat model is resident; the probe about to run loads a 531 MB
    GGUF. Sampling before the unload gave -1866.0 and scored the load as
    never reaching the GPU. The baseline must be taken after the fall.
    """
    module = _load_payload()
    session = _baseline_session(
        module, tmp_path, {"model_identifier": "chat.gguf"}, [4000.0, 2600.0, 1000.0, 996.0, 996.0]
    )
    monkeypatch.setattr(module, "VRAM_SETTLE_POLL_S", 0.0)
    monkeypatch.setattr(module, "nvidia_used_mib", lambda: session._readings.pop(0))
    assert session.settled_baseline() == 996.0
    assert session.studio.posts[0][0] == "/api/inference/unload"
    assert session.studio.posts[0][1]["model_path"] == "chat.gguf"
    assert session.studio.posts[0][1]["force_cancel_active"] is True


def test_nothing_loaded_means_nothing_to_unload(tmp_path, monkeypatch):
    module = _load_payload()
    session = _baseline_session(module, tmp_path, {}, [500.0, 500.0])
    monkeypatch.setattr(module, "VRAM_SETTLE_POLL_S", 0.0)
    monkeypatch.setattr(module, "nvidia_used_mib", lambda: session._readings.pop(0))
    assert session.settled_baseline() == 500.0
    assert session.studio.posts == []


def test_a_refused_unload_does_not_abort_the_probe(tmp_path, monkeypatch):
    """The delta is evidence, not the assertion. Losing it must not lose the
    load result that the probe is actually there to record."""
    module = _load_payload()
    session = _baseline_session(
        module, tmp_path, {"model_identifier": "chat.gguf"}, [900.0, 900.0], unload_raises = True
    )
    monkeypatch.setattr(module, "VRAM_SETTLE_POLL_S", 0.0)
    monkeypatch.setattr(module, "nvidia_used_mib", lambda: session._readings.pop(0))
    assert session.settled_baseline() == 900.0


def test_driver_jitter_is_not_mistaken_for_a_release(tmp_path, monkeypatch):
    """A few MiB of wobble must settle immediately, not burn the full budget
    and return whatever the last sample happened to be."""
    module = _load_payload()
    session = _baseline_session(module, tmp_path, {}, [1000.0, 995.0, 1000.0])
    monkeypatch.setattr(module, "VRAM_SETTLE_POLL_S", 0.0)
    monkeypatch.setattr(module, "nvidia_used_mib", lambda: session._readings.pop(0))
    assert session.settled_baseline() == 995.0


def test_a_card_that_never_settles_is_bounded(tmp_path, monkeypatch):
    """Something else holding the card must not hang the run forever."""
    module = _load_payload()
    session = _session(module, tmp_path, load_timeout = 30.0)
    session.studio = _FakeStudio({})
    monkeypatch.setattr(module, "VRAM_SETTLE_POLL_S", 0.0)
    calls = [0]

    def _falling():
        calls[0] += 1
        return 10000.0 - calls[0] * 100.0

    monkeypatch.setattr(module, "nvidia_used_mib", _falling)
    session.settled_baseline()
    assert calls[0] <= module.VRAM_SETTLE_SAMPLES + 1


def test_no_nvidia_smi_at_all_is_not_a_crash(tmp_path, monkeypatch):
    module = _load_payload()
    session = _session(module, tmp_path, load_timeout = 30.0)
    session.studio = _FakeStudio({})
    monkeypatch.setattr(module, "VRAM_SETTLE_POLL_S", 0.0)
    monkeypatch.setattr(module, "nvidia_used_mib", lambda: None)
    assert session.settled_baseline() is None


def test_the_status_fields_read_are_fields_the_response_really_has(tmp_path):
    """Run 8's fix ran and did nothing, because it read model_path / model /
    active_model_name off a response that carries none of them. Bind the names
    to InferenceStatusResponse itself, so a rename or another guess fails here
    instead of on hardware forty minutes later."""
    import re

    model_src = (REPO_ROOT / "studio" / "backend" / "models" / "inference.py").read_text(
        encoding = "utf-8"
    )
    block = model_src.split("class InferenceStatusResponse", 1)[1]
    block = block.split("\nclass ", 1)[0]
    declared = set(re.findall(r"^    (\w+):", block, re.MULTILINE))
    payload_src = (PAYLOAD_DIR / "run_studio_gpu.py").read_text(encoding = "utf-8")
    fn = payload_src.split("def settled_baseline", 1)[1].split("\n    def ", 1)[0]
    read = set(re.findall(r'body\.get\("(\w+)"\)', fn))
    assert read, "settled_baseline reads no status field at all"
    assert read <= declared, (
        f"settled_baseline reads {sorted(read - declared)}, which "
        f"InferenceStatusResponse does not declare"
    )


def test_the_display_name_is_only_a_fallback(tmp_path, monkeypatch):
    """model_identifier is the loadable one and is what /unload wants."""
    module = _load_payload()
    session = _baseline_session(
        module,
        tmp_path,
        {"model_identifier": "loadable.gguf", "active_model": "Pretty Name"},
        [3000.0, 500.0, 500.0],
    )
    monkeypatch.setattr(module, "VRAM_SETTLE_POLL_S", 0.0)
    monkeypatch.setattr(module, "nvidia_used_mib", lambda: session._readings.pop(0))
    session.settled_baseline()
    assert session.studio.posts[0][1]["model_path"] == "loadable.gguf"


def test_a_loaded_list_alone_is_enough_to_unload(tmp_path, monkeypatch):
    module = _load_payload()
    session = _baseline_session(
        module, tmp_path, {"loaded": ["only-here.gguf"]}, [3000.0, 500.0, 500.0]
    )
    monkeypatch.setattr(module, "VRAM_SETTLE_POLL_S", 0.0)
    monkeypatch.setattr(module, "nvidia_used_mib", lambda: session._readings.pop(0))
    session.settled_baseline()
    assert session.studio.posts[0][1]["model_path"] == "only-here.gguf"


# The launcher collects each kernel into its own subdirectory and the workflow hands collect_evidence.py the parent, so


# ------------------------------------------------- the evidence the launcher left
def test_the_bundle_is_found_in_the_per_kernel_directory_the_launcher_writes(tmp_path):
    """launch.py::fetch_evidence writes kaggle_evidence/<slug>/..., and the
    workflow passes kaggle_evidence. A top-level glob reported every real run
    as having emitted no evidence at all."""
    blob = _bundle({"studio_gpu_report.json": b'{"passed": false}'})
    evidence = tmp_path / "kaggle_evidence" / "unsloth-t4-ci-deadbeef"
    evidence.mkdir(parents = True)
    (evidence / "kernel.log").write_text("\n".join(_chunk_lines(blob)), encoding = "utf-8")
    outdir = tmp_path / "studio_evidence"
    proc = subprocess.run(
        [
            sys.executable,
            str(CI_DIR / "collect_evidence.py"),
            "--evidence",
            str(tmp_path / "kaggle_evidence"),
            "--outdir",
            str(outdir),
        ],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (outdir / "studio_gpu_report.json").read_bytes() == b'{"passed": false}'


def test_a_nested_executed_notebook_is_read_too(tmp_path):
    blob = _bundle({"studio.log": b"a log"})
    nb = {
        "cells": [{"outputs": [{"text": "\n".join(_chunk_lines(blob))}]}],
    }
    nested = tmp_path / "evidence" / "unsloth-t4-ci-1234"
    nested.mkdir(parents = True)
    (nested / "studio_gpu_output.ipynb").write_text(json.dumps(nb), encoding = "utf-8")
    chunks, total = collect_evidence.collect_chunks(
        collect_evidence.iter_text(tmp_path / "evidence")
    )
    assert total and len(chunks) == total




# ------------------------------------------------------- a diverged training run
def test_a_nan_loss_is_not_a_trained_step():
    """A T4 has no bf16, so this trains in fp16, and an fp16 run that diverges
    logs NaN for every step while still reaching `completed` and still saving
    an adapter. Counting mere list occupancy scored that as a full run."""
    status = {"metric_history": {"loss": [float("nan")] * 8}}
    assert studio_client.trained_steps(status) == 0
    assert len(studio_client.nonfinite_losses(status)) == 8


def test_an_infinite_loss_is_not_a_trained_step():
    status = {"metric_history": {"loss": [1.4, float("inf"), 0.9, None]}}
    assert studio_client.trained_steps(status) == 2
    assert studio_client.nonfinite_losses(status) == [float("inf")]


def test_real_losses_still_count():
    status = {"metric_history": {"loss": [2.0, 1.5, 1.1]}}
    assert studio_client.trained_steps(status) == 3
    assert studio_client.nonfinite_losses(status) == []




# ------------------------------------------------------ which Unsloth gets started
def test_studio_is_launched_from_the_interpreter_running_the_payload(tmp_path, monkeypatch):
    """The payload runs under the Unsloth venv, whose bin is NOT on PATH. A
    global `unsloth` anywhere on PATH would otherwise win shutil.which() and
    the run would measure some other install instead of the checkout."""
    module = _load_payload()
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents = True)
    own = venv_bin / "unsloth"
    own.write_text("#!/bin/sh\n", encoding = "utf-8")
    own.chmod(0o755)

    stray_bin = tmp_path / "stray"
    stray_bin.mkdir()
    stray = stray_bin / "unsloth"
    stray.write_text("#!/bin/sh\n", encoding = "utf-8")
    stray.chmod(0o755)
    monkeypatch.setenv("PATH", str(stray_bin))

    session = _session(module, tmp_path, port = 18902)
    monkeypatch.setattr(module.sys, "executable", str(venv_bin / "python"))
    assert session.studio_command()[0] == str(own)


def test_without_a_console_script_the_same_interpreter_runs_the_module(tmp_path, monkeypatch):
    module = _load_payload()
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents = True)
    stray_bin = tmp_path / "stray"
    stray_bin.mkdir()
    stray = stray_bin / "unsloth"
    stray.write_text("#!/bin/sh\n", encoding = "utf-8")
    stray.chmod(0o755)
    monkeypatch.setenv("PATH", str(stray_bin))

    session = _session(module, tmp_path, port = 18902)
    monkeypatch.setattr(module.sys, "executable", str(venv_bin / "python"))
    command = session.studio_command()
    assert command[0] == str(venv_bin / "python")
    assert command[1] == "-c"




# ------------------------------------------------------- the llama-server pid
def test_the_payload_never_reads_a_pid_the_status_response_does_not_declare():
    """InferenceStatusResponse declares no llama_server_pid and no pid, and
    FastAPI drops what the response model does not declare, so that lookup was
    always None and the process-level VRAM probe could never fire."""
    import re

    model_src = (REPO_ROOT / "studio" / "backend" / "models" / "inference.py").read_text(
        encoding = "utf-8"
    )
    declared = set()
    for cls in ("class InferenceStatusResponse", "class _InferenceRuntimeFields"):
        block = model_src.split(cls, 1)[1].split("\nclass ", 1)[0]
        declared |= set(re.findall(r"^    (\w+):", block, re.MULTILINE))
    payload_src = (PAYLOAD_DIR / "run_studio_gpu.py").read_text(encoding = "utf-8")
    fn = payload_src.split("def load_model", 1)[1].split("\n    def ", 1)[0]
    read = set(re.findall(r'status_body\.get\("(\w+)"\)', fn))
    assert read <= declared, (
        f"load_model reads {sorted(read - declared)}, which the inference status "
        f"response does not declare"
    )


def test_a_discovered_llama_server_pid_is_enough_evidence():
    verdict = gpu_assert.offload_verdict(
        server_pid = None,
        server_pids = [4242],
        compute_apps = {4242: 2048},
        log_text = "",
        device_vram_delta_mib = None,
        status = {},
    )
    assert verdict["passed"]
    assert any("4242" in p for p in verdict["positives"])


def test_a_discovered_pid_holding_nothing_is_still_not_evidence():
    verdict = gpu_assert.offload_verdict(
        server_pid = None,
        server_pids = [4242],
        compute_apps = {4242: 12},
        log_text = "",
        device_vram_delta_mib = None,
        status = {},
    )
    assert not verdict["passed"]


def test_the_payload_can_find_a_llama_server_in_the_process_table(tmp_path):
    module = _load_payload()
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        executable = sys.executable,
    )
    try:
        # Only the discovery mechanism is exercised here;
        assert isinstance(module.llama_server_pids(), list)
    finally:
        proc.kill()
        proc.wait()




# ------------------------------------------- evidence belongs to the load it follows
def test_an_earlier_loads_offload_line_is_not_evidence_for_the_next(tmp_path):
    """The exported-model reload used to inherit the chat model's
    `offloaded 29/29` when its own load logged nothing, so the GPU check
    passed on evidence from a different model."""
    module = _load_payload()
    server_log = tmp_path / "studio.log"
    home = tmp_path / "home"
    home.mkdir()
    server_log.write_text("load_tensors: offloaded 29/29 layers to GPU\n", encoding = "utf-8")

    marks = module.log_marks(server_log, home)
    # The second load says nothing at all.
    with open(server_log, "a", encoding = "utf-8") as fh:
        fh.write("llama_server: listening\n")

    scoped = module.studio_log_text(server_log, home, since = marks)
    assert "offloaded" not in scoped
    assert gpu_assert.offloaded_layers(scoped) is None
    assert "offloaded" in module.studio_log_text(server_log, home)


def test_a_log_that_was_rotated_under_us_is_read_whole(tmp_path):
    module = _load_payload()
    server_log = tmp_path / "studio.log"
    home = tmp_path / "home"
    home.mkdir()
    server_log.write_text("x" * 500, encoding = "utf-8")
    marks = module.log_marks(server_log, home)
    server_log.write_text("offloaded 7/7 layers to GPU\n", encoding = "utf-8")
    assert "offloaded" in module.studio_log_text(server_log, home, since = marks)




# ---------------------------------------------- the log across the UI restart
def test_the_restart_that_reseeds_the_account_keeps_the_earlier_log(tmp_path, monkeypatch):
    """assert_chat_ui() restarts Unsloth, and a truncating open threw away the
    backend log of every GPU assertion before the evidence was packaged."""
    module = _load_payload()
    session = _session(
        module,
        tmp_path,
        port = 18902,
        health_deadline = 0.0,
    )
    session.outdir = tmp_path / "out"
    session.outdir.mkdir()
    session.server_log = session.outdir / "studio.log"
    session.server_log.write_text("the first session's traceback\n", encoding = "utf-8")
    session.proc = None
    session.base_url = "http://127.0.0.1:18902"
    session.studio = _FakeStudio({})
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: _DeadProc())
    monkeypatch.setattr(module, "wait_for", lambda **kw: (True, {}, ""))
    session.start_server()
    assert "the first session's traceback" in session.server_log.read_text()


class _DeadProc:
    def poll(self):
        return None

    def terminate(self):
        return None

    def wait(self, timeout = None):
        return 0




# ------------------------------------------------- an export that outlives its request
def _export_session(module, tmp_path, studio):
    session = _session(
        module,
        tmp_path,
        export_timeout = 1.0,
        export_deadline = 5.0,
        quantization = "q8_0",
        gpu_layers = 99,
    )
    session.outdir = tmp_path / "out"
    session.outdir.mkdir()
    session.server_log = session.outdir / "studio.log"
    session.studio = studio
    session.proc = _DeadProc()
    return session


class _ExportStudio:
    """load-checkpoint fine, the gguf POST times out in transport, and the
    export finishes on the backend regardless."""

    def __init__(self, marker: Path):
        self.marker = marker
        self.polls = 0

    def expect(
        self,
        method,
        path,
        body = None,
        **kw,
    ):
        if path == "/api/export/export/gguf":
            raise TimeoutError("the read timed out")
        return {}

    def get(self, path, **kw):
        if path == "/api/export/status":
            self.polls += 1
            if self.polls < 2:
                return 200, {"last_op_seq": 1, "is_export_active": True}
            return 200, {
                "last_op_seq": 2,
                "is_export_active": False,
                "last_op_status": "success",
                "last_op_output_path": str(self.marker.parent),
            }
        return 200, {}


def test_an_export_that_outlives_its_http_request_is_still_polled(tmp_path, monkeypatch):
    """The route blocks for the whole export and the transport timeout is
    shorter than the export deadline, so a slow quantize raised an unhandled
    TimeoutError and crashed the payload instead of waiting for the result."""
    module = _load_payload()
    exports = tmp_path / "exports"
    exports.mkdir()
    gguf = exports / "model.Q8_0.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    session = _export_session(module, tmp_path, _ExportStudio(gguf))
    session.args.chat_timeout = 1.0
    session.args.load_timeout = 1.0
    monkeypatch.setattr(module, "llama_cpp_marker", lambda home: None)
    monkeypatch.setattr(module, "install_kind", lambda marker: "cuda12")
    monkeypatch.setattr(
        module.Payload,
        "load_model",
        lambda self, path, *, variant, label: {"failures": [], "evidence": []},
    )
    monkeypatch.setattr(
        module.Payload,
        "chat",
        lambda self, messages, **kw: (200, {"choices": [{"message": {"content": "hi"}}]}),
    )
    ok = session.assert_gguf_export(str(tmp_path / "adapter"))
    detail = session.assertions[-1]
    assert "export_request_timeout" in detail, detail
    assert ok, detail.get("failures")




# ---------------------------------------------------- the oversized bundle
def test_the_capped_bundle_still_carries_the_logs(tmp_path, monkeypatch):
    """The fallback used to rebuild with the report ALONE while its message
    said it was shipping logs, discarding studio.log and the driver log in
    exactly the failing runs that need them."""
    module = _load_payload()
    session = _session(module, tmp_path)
    session.outdir = tmp_path / "out"
    session.art_dir = session.outdir / "playwright"
    session.art_dir.mkdir(parents = True)
    session.server_log = session.outdir / "studio.log"
    (session.outdir / "studio_gpu_report.json").write_text('{"passed": false}', encoding = "utf-8")
    session.server_log.write_text("backend traceback\n", encoding = "utf-8")
    (session.outdir / "playwright_chat_ui.log").write_text("driver log\n", encoding = "utf-8")
    # A screenshot that will not compress, so the first pack blows the cap.
    (session.art_dir / "01.png").write_bytes(os.urandom(3_000_000))

    printed: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **kw: printed.append(" ".join(map(str, a))))
    session.emit_evidence(False)
    encoded = "".join(
        line.split(" ")[-1] for line in printed if line.startswith(module.EVIDENCE_PREFIX)
    )
    blob = base64.b64decode(encoded)
    assert len(blob) <= module.MAX_EVIDENCE_BYTES
    with tarfile.open(fileobj = io.BytesIO(blob), mode = "r:gz") as tar:
        names = tar.getnames()
        assert "studio.log" in names
        assert "playwright_chat_ui.log" in names
        assert "studio_gpu_report.json" in names




# ------------------------------------------------- one report, and it is the last
def test_a_crash_while_packaging_the_evidence_does_not_publish_a_pass(tmp_path, monkeypatch):
    """extract_reports keeps the FIRST report per label|model, so printing a
    pass and then crashing published the pass and left the correction
    unread."""
    module = _load_payload()
    session = _session(
        module,
        tmp_path,
        label = "studio-gpu",
        chat_model = "m",
        chat_variant = None,
        train_model = "t",
        max_steps = 8,
        quantization = "q8_0",
        gpu_layers = 99,
    )
    session.outdir = tmp_path / "out"
    session.outdir.mkdir()
    session.server_log = session.outdir / "studio.log"
    session.proc = None
    session.started = 0.0

    def _boom(passed):
        raise OSError("the artifact could not be read")

    monkeypatch.setattr(session, "emit_evidence", _boom)
    printed: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **kw: printed.append(" ".join(map(str, a))))
    code = session.finish()
    reports = [line for line in printed if line.startswith(module.RESULT_PREFIX)]
    assert len(reports) == 1
    published = json.loads(reports[0][len(module.RESULT_PREFIX) :])
    assert published["passed"] is False
    assert code == 1




# ------------------------------------- an installer failure is a failure, not infra
def _cell_source(driver: dict, needle: str) -> str:
    for cell in _payload_notebook(driver)["cells"]:
        source = "".join(cell["source"])
        if needle in source:
            return source
    raise AssertionError(f"no generated cell contains {needle!r}")


def test_a_failing_install_under_test_reports_a_failure_rather_than_infra(tmp_path):
    """`install.sh --local` raising SystemExit left papermill with no
    T4_SMOKE_REPORT, the launcher called that `infra` and the reporter exited
    0 -- so this workflow passed exactly the installer regressions its path
    filter selects for."""
    driver = _build(tmp_path)
    payload_source = _payload_source(driver)
    fail_report_src = (
        "def fail_report" + payload_source.split("def fail_report", 1)[1].split("\n\ndef ", 1)[0]
    )

    emitted: list[str] = []
    namespace = {"json": json, "print": lambda *a, **kw: emitted.append("".join(map(str, a)))}
    exec(compile(fail_report_src, "<fail_report>", "exec"), namespace)

    class _Result:
        returncode = 1

    namespace.update(
        {
            "sh": lambda *a, **kw: _Result(),
            "REPO": tmp_path,
            "STUDIO_HOME": tmp_path,
            "pathlib": __import__("pathlib"),
        }
    )
    with pytest.raises(SystemExit):
        exec(compile(_cell_source(driver, "_install = sh("), "<install>", "exec"), namespace)

    reports = [line for line in emitted if line.startswith(build_kernel.RESULT_PREFIX)]
    assert len(reports) == 1, emitted
    report = json.loads(reports[0][len(build_kernel.RESULT_PREFIX) :])
    assert report["passed"] is False
    assert "install.sh" in report["failures"][0]


def test_an_install_that_leaves_no_interpreter_is_also_a_failure(tmp_path):
    driver = _build(tmp_path)
    payload_source = _payload_source(driver)
    fail_report_src = (
        "def fail_report" + payload_source.split("def fail_report", 1)[1].split("\n\ndef ", 1)[0]
    )
    emitted: list[str] = []
    namespace = {"json": json, "print": lambda *a, **kw: emitted.append("".join(map(str, a)))}
    exec(compile(fail_report_src, "<fail_report>", "exec"), namespace)

    class _Result:
        returncode = 0

    namespace.update({"sh": lambda *a, **kw: _Result(), "REPO": tmp_path, "STUDIO_HOME": tmp_path})
    with pytest.raises(SystemExit):
        exec(compile(_cell_source(driver, "_install = sh("), "<install>", "exec"), namespace)
    reports = [line for line in emitted if line.startswith(build_kernel.RESULT_PREFIX)]
    assert len(reports) == 1
    assert json.loads(reports[0][len(build_kernel.RESULT_PREFIX) :])["passed"] is False


def _run_verify_cell(driver: dict, tmp_path, *, probe: dict, host_gpus: list[str]):
    """Execute the generated dependency-probe cell against a fake venv + nvidia-smi.

    Returns every ``T4_SMOKE_REPORT`` the cell published.
    """
    payload_source = _payload_source(driver)
    fail_report_src = (
        "def fail_report" + payload_source.split("def fail_report", 1)[1].split("\n\ndef ", 1)[0]
    )
    emitted: list[str] = []
    namespace = {"json": json, "print": lambda *a, **kw: emitted.append("".join(map(str, a)))}
    exec(compile(fail_report_src, "<fail_report>", "exec"), namespace)

    class _FakeSubprocess:
        @staticmethod
        def run(cmd, **kw):
            if "nvidia-smi" in cmd[0]:
                return subprocess.CompletedProcess(cmd, 0, "\n".join(host_gpus) + "\n", "")
            return subprocess.CompletedProcess(cmd, 0, json.dumps(probe) + "\n", "")

    namespace.update(
        {
            "subprocess": _FakeSubprocess,
            "os": os,
            "VENV_PY": tmp_path / "python",
            "STUDIO_HOME": tmp_path,
        }
    )
    with pytest.raises(SystemExit):
        exec(compile(_cell_source(driver, 'probe["missing"]'), "<verify>", "exec"), namespace)
    return [line for line in emitted if line.startswith(build_kernel.RESULT_PREFIX)]


def test_a_venv_that_cannot_use_cuda_on_a_gpu_box_is_a_failure_not_infra(tmp_path):
    """`install.sh --local` resolving a CPU-only torch is the CUDA install
    regression this leg exists to catch, and it left no T4_SMOKE_REPORT at all,
    so the launcher filed it as `infra` and the reporter exited 0."""
    reports = _run_verify_cell(
        _build(tmp_path),
        tmp_path,
        probe = {"versions": {"torch": "2.9.0+cpu"}, "missing": [], "cuda": {"available": False}},
        host_gpus = ["Tesla T4"],
    )
    assert len(reports) == 1, reports
    report = json.loads(reports[0][len(build_kernel.RESULT_PREFIX) :])
    assert report["passed"] is False
    assert "cannot use CUDA" in report["failures"][0]


def test_a_session_kaggle_gave_no_gpu_at_all_stays_infra(tmp_path):
    """Nothing was learned about the code, so this one keeps the no-report path
    and must NOT turn a pull request red."""
    reports = _run_verify_cell(
        _build(tmp_path),
        tmp_path,
        probe = {"versions": {"torch": "2.9.0"}, "missing": [], "cuda": {"available": False}},
        host_gpus = [],
    )
    assert reports == []


def test_a_payload_that_hangs_past_the_driver_deadline_is_a_failure(tmp_path):
    """papermill killed mid-assertion emitted no report at all, so a hang in
    the code under test was filed as unavailable infrastructure."""
    driver = _build(tmp_path)
    runner = None
    for cell in driver["cells"]:
        source = "".join(cell["source"])
        if "papermill" in source:
            runner = source
    assert runner is not None

    emitted: list[str] = []

    class _FakeSubprocess:
        TimeoutExpired = subprocess.TimeoutExpired
        STDOUT = subprocess.STDOUT

        @staticmethod
        def run(cmd, **kw):
            kw["stdout"].write(
                f"{build_kernel.PAYLOAD_SENTINEL} exec /usr/bin/python run_studio_gpu.py\n".encode()
            )
            kw["stdout"].flush()
            raise subprocess.TimeoutExpired(cmd, 3900)

    namespace = {
        "WORK": tmp_path,
        "os": os,
        "sys": sys,
        "time": __import__("time"),
        "json": json,
        "subprocess": _FakeSubprocess,
        "print": lambda *a, **kw: emitted.append("".join(map(str, a))),
    }
    exec(compile(runner, "<runner>", "exec"), namespace)
    reports = [line for line in emitted if line.startswith(build_kernel.RESULT_PREFIX)]
    assert len(reports) == 1, emitted
    assert json.loads(reports[0][len(build_kernel.RESULT_PREFIX) :])["passed"] is False


def test_a_timeout_before_the_payload_started_is_still_infra(tmp_path):
    """A slow clone or install is Kaggle being slow, and must stay a no-report
    infra outcome rather than turning a pull request red."""
    driver = _build(tmp_path)
    runner = ["".join(c["source"]) for c in driver["cells"] if "papermill" in "".join(c["source"])][
        0
    ]
    emitted: list[str] = []

    class _FakeSubprocess:
        TimeoutExpired = subprocess.TimeoutExpired
        STDOUT = subprocess.STDOUT

        @staticmethod
        def run(cmd, **kw):
            kw["stdout"].write(b"cloning...\n")
            raise subprocess.TimeoutExpired(cmd, 3900)

    namespace = {
        "WORK": tmp_path,
        "os": os,
        "sys": sys,
        "time": __import__("time"),
        "json": json,
        "subprocess": _FakeSubprocess,
        "print": lambda *a, **kw: emitted.append("".join(map(str, a))),
    }
    exec(compile(runner, "<runner>", "exec"), namespace)
    assert not [line for line in emitted if line.startswith(build_kernel.RESULT_PREFIX)]




# ---------------------------------------------------------------- the gate flags
def test_the_gate_is_told_how_many_kernels_this_leg_actually_pushes():
    """This leg pushes ONE kernel and leaves the second T4 idle. The gate
    defaults to two, and refuses unless that many slots are free."""
    import re

    text = WORKFLOW.read_text(encoding = "utf-8")
    # EVERY invocation, found by the command rather than by splitting on the
    # first literal "gate.py" in the file: prose above the jobs mentions the
    # script by path, and there are now two calls -- the gate and the recheck
    # that re-asks with the account slot in hand. Both push one kernel, so both
    # have to say so, and a recheck left at the default of two would stand the
    # job down for a slot it does not need.
    invocations = re.findall(r"kaggle_t4_ci/gate\.py \\\n.*?(?=\n\s*\n|\Z)", text, re.DOTALL)
    assert invocations, "the workflow never runs the gate"
    for invocation in invocations:
        assert re.search(r"^\s+--kernels 1 \\?\s*$", invocation, re.MULTILINE), invocation
    assert "--expect 1" in text


def test_the_opt_in_label_can_actually_start_the_workflow():
    """GitHub's default pull_request activity types are opened, synchronize
    and reopened, so adding the advertised label started nothing."""
    triggers = _triggers(_workflow())
    assert "labeled" in triggers["pull_request"]["types"]
    for default in ("opened", "synchronize", "reopened"):
        assert default in triggers["pull_request"]["types"]


def test_an_unrelated_label_cannot_start_a_seventy_minute_kaggle_run():
    """GitHub has no per-label event filter, so `labeled` fires for every label
    added -- and the gate reads the whole label list, so once the opt-in label
    sits on the pull request it would call every one of those events forced."""
    condition = " ".join(_workflow()["jobs"]["gate"]["if"].split())
    assert "github.event.action != 'labeled'" in condition
    assert "github.event.label.name == 'kaggle-studio-gpu-ci'" in condition
    # The fork guard still has to survive the added clause.
    assert "fork != true" in condition


class _LoadStudio:
    def __init__(self, status_body):
        self.status_body = status_body

    def expect(
        self,
        method,
        path,
        body = None,
        **kw,
    ):
        return {}

    def get(self, path, **kw):
        return 200, self.status_body

    def post(
        self,
        path,
        body = None,
        **kw,
    ):
        return 200, {}


def test_load_model_does_not_inherit_the_previous_loads_offload_line(tmp_path, monkeypatch):
    """The end-to-end shape of the scoping fix: a reload whose own load logged
    nothing must not pass on the chat model's `offloaded 29/29`."""
    module = _load_payload()
    session = _session(module, tmp_path, load_timeout = 5.0, gpu_layers = 99)
    session.outdir = tmp_path / "out"
    session.outdir.mkdir()
    session.server_log = session.outdir / "studio.log"
    session.studio_home.mkdir(parents = True, exist_ok = True)
    session.server_log.write_text("load_tensors: offloaded 29/29 layers to GPU\n", encoding = "utf-8")
    session.studio = _LoadStudio({})
    monkeypatch.setattr(module, "nvidia_used_mib", lambda: None)
    monkeypatch.setattr(module, "nvidia_compute_apps", lambda: {})
    monkeypatch.setattr(module, "llama_server_pids", lambda: [])
    monkeypatch.setattr(module.Payload, "settled_baseline", lambda self: None)

    detail = session.load_model("exported.gguf", variant = None, label = "exported")
    assert detail["positives"] == []
    assert detail["failures"], "silence must be a failure, not the previous load's evidence"


def test_a_box_without_nvidia_smi_reports_no_gpu_rather_than_crashing(tmp_path, monkeypatch):
    """environment() runs inside finish(), on every path including the
    preflight one, so a FileNotFoundError there ended the run with a traceback
    and no report at all."""
    module = _load_payload()
    monkeypatch.setenv("PATH", str(tmp_path))
    assert module.gpu_inventory() == []
    assert module.nvidia_used_mib() is None
    assert module.nvidia_compute_apps() is None


def test_the_kaggle_client_is_new_enough_to_read_the_only_credential_we_have():
    """A client that cannot read KAGGLE_API_TOKEN makes this workflow report green forever.

    The gate is handed exactly one credential, KAGGLE_API_TOKEN, and nothing in
    the workflow or under .github/scripts/kaggle_t4_ci writes a kaggle.json. On
    2.x, authenticate() tries _authenticate_with_access_token() first, which
    reaches kagglesdk.get_access_token_from_env() and reads that variable. On
    1.7.4.5 it does not: KAGGLE_API_TOKEN appears only in
    kagglesdk/kaggle_http_client.py, whose own header says that client "is not
    currently usable by the CLI", so authenticate() falls through to
    read_config_file() and raises IOError, which IS OSError on Python 3.

    That error is not loud. gate.py turns any error into a skip unless
    --no-soft-fail is passed, and a skip exits 0, so the run is green with the
    GPU job skipped: identical on the surface to the ~95% of invocations the
    5% sampler declines. Observed on run 32605377065, dispatched with
    force=true to bypass the sampler, which still reported
    "could not authenticate to Kaggle: OSError" and a green workflow.

    The notebook leg has carried this guard since it was written; this leg had
    no equivalent and sat on 1.7.4.5, so it had never authenticated once.
    """
    packaging_version = pytest.importorskip("packaging.version")
    pins = re.findall(
        r"pip install [^\n]*'kaggle==([0-9][^']*)'", WORKFLOW.read_text(encoding = "utf-8")
    )
    assert pins, "no pinned kaggle client in the workflow"
    assert len(set(pins)) == 1, f"jobs disagree on the kaggle client: {pins}"
    assert packaging_version.Version(pins[0]) >= packaging_version.Version("2.2.0"), pins[0]


def test_no_studio_assertion_is_wired_to_a_constant_branch():
    """A repo-wide version of the guard that caught five vacuous guards at once.

    Every rule in the Studio payload was, at some point, protected by a test
    that asserted "the failure message appears in the source". That is
    satisfied by `if False:` sitting above an untouched message, so disabling a
    rule outright left its test green. Five mutations survived that way in one
    sitting.

    A constant test in an `assert_*` method means a branch that can never be
    taken, or one always taken. Neither is something a check has any business
    doing, and this catches it for every assertion at once rather than one test
    at a time.
    """
    import ast

    src = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "kaggle"
        / "studio_gpu"
        / "run_studio_gpu.py"
    ).read_text(encoding = "utf-8")
    offenders = []
    for func in ast.walk(ast.parse(src)):
        if not (isinstance(func, ast.FunctionDef) and func.name.startswith("assert_")):
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
                offenders.append(f"{func.name}: if {ast.unparse(node.test)}")
    assert offenders == [], f"assertions wired to a constant: {offenders}"


def test_every_assertion_carries_its_own_wall_clock():
    """Studio is the longest payload in the kernel and had no breakdown.

    On unsloth-probe-full-concurrent-417238 `studio_test` ran 1487.5s -- the
    single largest item on the critical path -- and the report carried 19
    assertions with no timing on any of them, so "where does Studio's time go"
    could only be answered by buying another session.

    Driven rather than grepped: a rule that only checked for a `time.time()`
    call passes on a payload that computes the number and drops it.
    """
    import importlib.util
    import types

    payload = PAYLOAD_DIR / "run_studio_gpu.py"
    spec = importlib.util.spec_from_file_location("_studio_payload_timing", payload)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    runner = types.SimpleNamespace(
        assertions = [],
        failures = [],
        started = module.time.time() - 5.0,
        record = None,
    )
    record = module.Payload.record.__get__(runner, module.Payload)
    module.log = lambda *a, **k: None
    record("first", True, {})
    record("second", True, {"failures": []})

    names = [a["name"] for a in runner.assertions]
    assert names == ["first", "second"]
    # The first entry measures from process start, which is the setup before
    # any assertion -- 5s here -- and is exactly the slice that would otherwise
    # be invisible.
    assert runner.assertions[0]["seconds_since_previous"] >= 5.0
    # The second measures from the first, not from the start, or every entry
    # would read as the whole run so far and the breakdown would be useless.
    assert runner.assertions[1]["seconds_since_previous"] < 1.0
    # And an absolute position, so a reader can line the report up against the
    # driver's own interval for the payload.
    assert runner.assertions[1]["at_seconds"] >= runner.assertions[0]["at_seconds"]
