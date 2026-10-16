# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""cuDNN fused attention for the sd.cpp fork: who gets a CUDA 12 cuDNN, how it is installed, what the status says."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from core.inference import sd_cpp_cudnn as cd

RT = cd.CUDNN_RUNTIMES[12]

# Sibling imports (the H3 load harness) need the tests dir on the path: pytest inserts rootdir, not this package.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Lines the fork prints at INFO level (copied from a B200 sd-server log of the cuDNN build).
LOADED = "[INFO   ] ggml - load_locked: cuDNN 92700 loaded from /x/nvidia/cudnn/lib/libcudnn.so.9 for attention"
PLAN_DIT = (
    "[INFO   ] ggml - ggml_cuda_cudnn_sdpa_prepare: cuDNN SDPA plan b=1 hq=56 hk=56 sq=19315 skv=19315 "
    "d=128 f16, workspace 0 B, built in 1321 ms"
)
NO_PLAN = (
    "[INFO   ] ggml - ggml_cuda_cudnn_sdpa_prepare: no cuDNN SDPA plan for b=1 hq=56 hk=56 sq=19315 "
    "skv=19315 d=128, keeping the ggml kernel"
)


def _binary(
    tmp_path: Path,
    *,
    cudnn: bool,
    cudart: int | None = 12,
    name = "sd-cli",
) -> str:
    blob = b"\x7fELF" + b"\0" * 64
    if cudart is not None:
        blob += f"libcudart.so.{cudart}".encode() + b"\0"
    blob += b"x" * (9 << 20)  # past one scan chunk, so the marker straddles reads
    if cudnn:
        blob += b"GGML_CUDA_CUDNN_LIB\0"
    path = tmp_path / name
    path.write_bytes(blob)
    return str(path)


def _preinstall(root: Path, runtime = RT) -> str:
    d = root / runtime.dirname
    (d / "nvidia/cudnn/lib").mkdir(parents = True)
    (d / "nvidia/cuda_nvrtc/lib").mkdir(parents = True)
    (d / "nvidia/cudnn/lib/libcudnn.so.9").write_bytes(b"lib")
    (d / "nvidia/cuda_nvrtc/lib/libnvrtc.so.12").write_bytes(b"lib")
    (d / "unsloth-cudnn.json").write_text(json.dumps({"requirements": runtime.requirements()}))
    return str(d / "nvidia/cudnn/lib/libcudnn.so.9")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for var in (cd.CUDNN_INSTALL_ENV, cd.GGML_CUDNN_LIB_ENV, cd.GGML_CUDNN_ATTN_ENV):
        monkeypatch.delenv(var, raising = False)
    cd._FAILED.clear()
    cd._SCAN_MEMO.clear()
    yield
    cd._FAILED.clear()


def _no_install(*_a, **_k):
    raise AssertionError("must not install")


def test_binary_scan_reads_the_build_marker_and_cuda_major(tmp_path):
    assert cd.binary_cudnn_build(_binary(tmp_path, cudnn = True)) == (True, 12)
    assert cd.binary_cudnn_build(_binary(tmp_path, cudnn = False, name = "old")) == (False, 12)
    assert cd.binary_cudnn_build(_binary(tmp_path, cudnn = True, cudart = 13, name = "c13")) == (True, 13)
    assert cd.binary_cudnn_build(str(tmp_path / "missing")) == (False, None)


def test_binary_scan_reads_a_shared_ggml_beside_the_binary(tmp_path):
    exe = _binary(tmp_path, cudnn = False, cudart = None)
    (tmp_path / "libggml-cuda.so").write_bytes(b"GGML_CUDA_CUDNN_LIB\0libcudart.so.12\0")
    assert cd.binary_cudnn_build(exe) == (True, 12)


@pytest.mark.parametrize(
    "platform, cc",
    [
        ("win32", (10, 0)),
        ("darwin", (10, 0)),
        ("linux", (7, 5)),
        ("linux", (6, 1)),
        ("linux", None),
    ],
)
def test_inert_off_linux_below_sm80_and_unknown_card(monkeypatch, tmp_path, platform, cc):
    """Windows, macOS, a T4 (sm75) and a card whose capability the build did not report (AMD, Vulkan, CPU):
    no install, no env, no status."""
    monkeypatch.setattr(cd, "ensure_library", _no_install)
    plan = cd.plan_cudnn_attention(
        _binary(tmp_path, cudnn = True), cc, platform = platform, root = tmp_path
    )
    assert plan.env == () and plan.state is None
    assert plan.status_fields() == {"sd_cpp_cudnn_attention": None, "sd_cpp_cudnn_reason": None}


def test_inert_for_a_prebuilt_without_the_cudnn_build(monkeypatch, tmp_path):
    monkeypatch.setattr(cd, "ensure_library", _no_install)
    plan = cd.plan_cudnn_attention(
        _binary(tmp_path, cudnn = False), (10, 0), platform = "linux", root = tmp_path
    )
    assert plan.env == () and plan.state is None


@pytest.mark.parametrize("cc", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)])
def test_sm80_plus_gets_the_managed_library_for_the_child_only(monkeypatch, tmp_path, cc):
    lib = _preinstall(tmp_path)
    monkeypatch.setattr(cd, "_install_locked", _no_install)
    before = dict(os.environ)
    plan = cd.plan_cudnn_attention(
        _binary(tmp_path, cudnn = True), cc, platform = "linux", root = tmp_path
    )
    assert plan.env == ((cd.GGML_CUDNN_LIB_ENV, lib),)
    assert plan.state == cd.STATE_READY
    assert dict(os.environ) == before


def test_user_values_win(monkeypatch, tmp_path):
    monkeypatch.setattr(cd, "ensure_library", _no_install)
    exe = _binary(tmp_path, cudnn = True)
    monkeypatch.setenv(cd.GGML_CUDNN_LIB_ENV, "/opt/cudnn/libcudnn.so.9")
    plan = cd.plan_cudnn_attention(exe, (10, 0), platform = "linux", root = tmp_path)
    assert plan.env == () and plan.state == cd.STATE_READY
    monkeypatch.delenv(cd.GGML_CUDNN_LIB_ENV)
    monkeypatch.setenv(cd.GGML_CUDNN_ATTN_ENV, "0")
    plan = cd.plan_cudnn_attention(exe, (10, 0), platform = "linux", root = tmp_path)
    assert plan.env == () and plan.state == cd.STATE_OFF
    monkeypatch.delenv(cd.GGML_CUDNN_ATTN_ENV)
    monkeypatch.setenv(cd.CUDNN_INSTALL_ENV, "0")
    plan = cd.plan_cudnn_attention(exe, (10, 0), platform = "linux", root = tmp_path)
    assert plan.env == () and plan.state == cd.STATE_OFF


def test_cuda13_build_gets_no_cuda12_library(monkeypatch, tmp_path):
    monkeypatch.setattr(cd, "ensure_library", _no_install)
    exe = _binary(tmp_path, cudnn = True, cudart = 13)
    plan = cd.plan_cudnn_attention(exe, (10, 0), platform = "linux", root = tmp_path)
    assert plan.env == () and plan.state == cd.STATE_UNAVAILABLE


def test_offline_load_never_installs(monkeypatch, tmp_path):
    monkeypatch.setattr(cd, "_install_locked", _no_install)
    plan = cd.plan_cudnn_attention(
        _binary(tmp_path, cudnn = True), (10, 0), platform = "linux", root = tmp_path, allow_install = False
    )
    assert plan.env == () and plan.state == cd.STATE_UNAVAILABLE


@pytest.mark.parametrize("uv", ["/usr/bin/uv", None])
def test_install_command_targets_a_studio_dir_never_the_venv(tmp_path, uv):
    """The CUDA 12 and CUDA 13 cuDNN wheels both write nvidia/cudnn/lib/libcudnn.so.9: installed into the venv,
    nvidia-cudnn-cu12 would replace torch's own. --target a separate dir, --no-deps, exact pins, wheels only."""
    cmd = cd.install_command(RT, tmp_path / "stage", uv)
    assert cmd[cmd.index("--target") + 1] == str(tmp_path / "stage")
    assert "--no-deps" in cmd and cmd[cmd.index("--only-binary") + 1] == ":all:"
    assert cmd[-2:] == ["nvidia-cudnn-cu12==9.27.0.42", "nvidia-cuda-nvrtc-cu12==12.8.93"]
    for banned in (
        "--upgrade",
        "-U",
        "--force-reinstall",
        "--reinstall",
        "--system",
        "--break-system-packages",
    ):
        assert banned not in cmd


def test_managed_root_is_under_the_studio_bin_dir_not_site_packages(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    root = cd.managed_root()
    assert root == (tmp_path / "home").resolve() / "bin" / "sd-cpp-cudnn"
    import sysconfig

    for key in ("purelib", "platlib"):
        site = Path(sysconfig.get_paths()[key]).resolve()
        assert site != root.resolve() and site not in root.resolve().parents


def _fake_installer(
    monkeypatch,
    *,
    version = "92700",
    write_files = True,
    drift = False,
):
    calls = []
    snaps = iter([{"torch": "2.12.1"}, {"torch": "2.12.1" if not drift else "2.13.0"}])
    monkeypatch.setattr(cd, "_venv_snapshot", lambda: next(snaps))
    monkeypatch.setattr(cd, "_reachable", lambda _url: True)
    monkeypatch.setattr(cd, "_uv_executable", lambda: None)
    monkeypatch.setattr(cd, "_installer_config", lambda _uv: {"mirror": False, "unprobed": False})

    def run(
        cmd,
        timeout,
        cancel_event = None,
    ):
        calls.append(cmd)
        if "--target" in cmd:
            target = Path(cmd[cmd.index("--target") + 1])
            if write_files:
                (target / "nvidia/cudnn/lib").mkdir(parents = True)
                (target / "nvidia/cuda_nvrtc/lib").mkdir(parents = True)
                (target / "nvidia/cudnn/lib/libcudnn.so.9").write_bytes(b"lib")
                (target / "nvidia/cuda_nvrtc/lib/libnvrtc.so.12").write_bytes(b"lib")
            return True, "Installed 2 packages"
        return True, version + "\n"

    monkeypatch.setattr(cd, "_run", run)
    return calls


def test_install_stages_verifies_and_publishes_atomically(monkeypatch, tmp_path):
    calls = _fake_installer(monkeypatch)
    lib, reason = cd.ensure_library(RT, root = tmp_path)
    assert reason is None and lib == str(tmp_path / RT.dirname / "nvidia/cudnn/lib/libcudnn.so.9")
    assert calls[1][0] == sys.executable and ".staging-" in calls[1][-1]
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".staging-")]
    monkeypatch.setattr(cd, "_run", _no_install)
    assert cd.ensure_library(RT, root = tmp_path) == (lib, None)


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"version": "92000"}, "verification failed"),
        ({"write_files": False}, "without libcudnn"),
        ({"drift": True}, "changed this environment"),
    ],
)
def test_failed_install_publishes_nothing_and_is_not_retried(monkeypatch, tmp_path, kwargs, needle):
    calls = _fake_installer(monkeypatch, **kwargs)
    lib, reason = cd.ensure_library(RT, root = tmp_path)
    assert lib is None and needle in reason
    assert not (tmp_path / RT.dirname).exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".staging-")]
    n = len(calls)
    assert cd.ensure_library(RT, root = tmp_path) == (None, reason)
    assert len(calls) == n


def test_unreachable_index_refuses_before_downloading(monkeypatch, tmp_path):
    monkeypatch.setattr(cd, "_reachable", lambda _url: False)
    monkeypatch.setattr(cd, "_run", _no_install)
    monkeypatch.setattr(cd, "_installer_config", lambda _uv: {"mirror": False, "unprobed": False})
    lib, reason = cd.ensure_library(RT, root = tmp_path)
    assert lib is None and "not reachable" in reason


def test_a_configured_mirror_skips_the_pypi_probe(monkeypatch, tmp_path):
    """A mirror named only in uv.toml / pip.conf (read by the NVFP4 installer's config reader) is the installer's to
    reach, so an unreachable pypi.org does not refuse the install."""
    calls = _fake_installer(monkeypatch)
    monkeypatch.setattr(cd, "_reachable", _no_install)
    monkeypatch.setattr(cd, "_installer_config", lambda _uv: {"mirror": True, "unprobed": False})
    lib, reason = cd.ensure_library(RT, root = tmp_path)
    assert reason is None and lib and "--target" in calls[0]


def test_a_pip_only_mirror_reaches_uv(monkeypatch, tmp_path):
    """uv does not read PIP_INDEX_URL: the command carries it over as --index-url."""
    for var in ("UV_INDEX_URL", "UV_DEFAULT_INDEX"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.example/simple")
    cmd = cd.install_command(RT, tmp_path / "stage", "/usr/bin/uv")
    assert cmd[cmd.index("--index-url") + 1] == "https://mirror.example/simple"


def test_cancel_kills_the_install_and_is_not_remembered(monkeypatch, tmp_path):
    """A cancelled load stops the installer at once; the next load installs again."""
    import threading
    import time

    cancel = threading.Event()
    monkeypatch.setattr(cd, "_venv_snapshot", lambda: {})
    monkeypatch.setattr(cd, "_installer_config", lambda _uv: {"mirror": True, "unprobed": False})
    monkeypatch.setattr(cd, "_uv_executable", lambda: None)
    monkeypatch.setattr(
        cd,
        "install_command",
        lambda *_a, **_k: [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    threading.Timer(0.5, cancel.set).start()
    t = time.monotonic()
    lib, reason = cd.ensure_library(RT, root = tmp_path, cancel_event = cancel)
    assert lib is None and reason == "cancelled" and time.monotonic() - t < 10
    assert RT.cuda_major not in cd._FAILED
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".staging-")]


def test_low_disk_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(cd.shutil, "disk_usage", lambda _p: type("U", (), {"free": 1 << 30})())
    monkeypatch.setattr(cd, "_run", _no_install)
    lib, reason = cd.ensure_library(RT, root = tmp_path)
    assert lib is None and "GiB free" in reason


def test_status_engaged_from_the_fork_log_and_kept_on_a_reused_server():
    a = cd.CudnnAttention(cd.STATE_READY, env = ((cd.GGML_CUDNN_LIB_ENV, "/l"),))
    a.begin_render()
    for line in ("[INFO   ] sampling", LOADED, PLAN_DIT):
        a.feed(line)
    a.end_render()
    assert a.state == cd.STATE_ENGAGED and a.cudnn_version == 92700
    assert (19315, 19315, 128) in a.shapes
    a.begin_render()
    a.feed("[INFO   ] sampling completed, taking 5.69s")
    a.end_render()
    assert a.state == cd.STATE_ENGAGED


def test_status_fallback_when_no_plan_or_library_not_loaded():
    a = cd.CudnnAttention(cd.STATE_READY)
    a.begin_render()
    a.feed(LOADED)
    a.feed(NO_PLAN)
    a.end_render()
    assert a.state == cd.STATE_FALLBACK and "no attention plan" in a.reason
    b = cd.CudnnAttention(cd.STATE_READY)
    b.begin_render()
    b.feed("[INFO   ] sampling completed")
    b.end_render()
    assert b.state == cd.STATE_FALLBACK and "did not load" in b.reason
    c = cd.CudnnAttention(cd.STATE_READY)
    c.begin_render()
    c.end_render(ok = False)
    assert c.state == cd.STATE_READY


def _h3_load(
    monkeypatch,
    tmp_path,
    *,
    devices,
    carries = True,
):
    import test_video_backend as tvb

    monkeypatch.setattr(cd, "binary_cudnn_build", lambda _b: (carries, 12))
    root = tmp_path / "cudnn_root"
    lib = _preinstall(root)
    monkeypatch.setattr(cd, "managed_root", lambda: root)
    monkeypatch.setattr(cd, "_install_locked", _no_install)
    monkeypatch.delenv("GGML_CUDA_QUANT_CUBLAS_MIN_BATCH", raising = False)
    state, _offload = tvb._load_h3_native_offload(
        monkeypatch, tmp_path, help_text = tvb._SAGE_HELP, speed_mode = "max", devices = devices
    )
    return state, lib


def test_h3_load_on_sm100_names_the_library_and_reports_ready(monkeypatch, tmp_path):
    import test_video_backend as tvb
    from core.inference.video import _sd_cpp_cudnn_status

    state, lib = _h3_load(monkeypatch, tmp_path, devices = tvb._cuda_devices("10.0"))
    env = dict(state.pipe.env)
    assert env[cd.GGML_CUDNN_LIB_ENV] == lib
    assert env["GGML_CUDA_QUANT_CUBLAS_MIN_BATCH"] == "1024"
    assert _sd_cpp_cudnn_status(state) == {
        "sd_cpp_cudnn_attention": "ready",
        "sd_cpp_cudnn_reason": None,
    }


@pytest.mark.parametrize("cc, carries", [("7.5", True), ("10.0", False)])
def test_h3_load_on_t4_or_old_prebuilt_is_unchanged(monkeypatch, tmp_path, cc, carries):
    import test_video_backend as tvb
    from core.inference.video import _sd_cpp_cudnn_status

    state, _lib = _h3_load(monkeypatch, tmp_path, devices = tvb._cuda_devices(cc), carries = carries)
    assert cd.GGML_CUDNN_LIB_ENV not in dict(state.pipe.env)
    assert _sd_cpp_cudnn_status(state) == {
        "sd_cpp_cudnn_attention": None,
        "sd_cpp_cudnn_reason": None,
    }


def test_status_route_model_carries_the_fields():
    from models.inference import VideoStatusResponse
    fields = VideoStatusResponse.model_fields
    assert "sd_cpp_cudnn_attention" in fields and "sd_cpp_cudnn_reason" in fields


def test_preflight_refusals_are_rechecked_on_the_next_load(monkeypatch, tmp_path):
    """Low disk and an unreachable index are rechecked; only a real install failure is remembered."""
    calls = _fake_installer(monkeypatch)
    monkeypatch.setattr(cd, "_reachable", lambda _url: False)
    lib, reason = cd.ensure_library(RT, root = tmp_path)
    assert lib is None and "not reachable" in reason and RT.cuda_major not in cd._FAILED
    monkeypatch.setattr(cd, "_reachable", lambda _url: True)
    lib, reason = cd.ensure_library(RT, root = tmp_path)
    assert reason is None and lib and "--target" in calls[0]


def test_a_plan_that_fails_to_execute_reports_fallback():
    a = cd.CudnnAttention(cd.STATE_READY, env = ((cd.GGML_CUDNN_LIB_ENV, "/l"),))
    a.begin_render()
    for line in (
        LOADED,
        PLAN_DIT,
        "ggml_cuda_cudnn_sdpa: cuDNN SDPA execute failed, keeping the ggml kernel for this shape: x",
    ):
        a.feed(line)
    a.end_render()
    assert a.state == cd.STATE_FALLBACK and "failed to run" in a.reason


def test_the_installer_own_index_is_not_overridden(monkeypatch, tmp_path):
    """A uv config naming its own default index keeps it; PIP_INDEX_URL is not forced over it."""
    monkeypatch.delenv("UV_INDEX_URL", raising = False)
    monkeypatch.delenv("UV_DEFAULT_INDEX", raising = False)
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.example/simple")
    cmd = cd.install_command(RT, tmp_path / "stage", "/usr/bin/uv", own_index = True)
    assert "--index-url" not in cmd


def test_status_reason_is_redacted_for_remote_callers():
    from hub.utils.host_paths import HOST_PATH_TEXT_FIELDS
    assert "sd_cpp_cudnn_reason" in HOST_PATH_TEXT_FIELDS


def test_installer_output_is_redacted_and_copies_files(monkeypatch):
    """Index credentials in installer output never reach the status reason; uv copies, never links, the target."""

    def fake_child_env():
        return {"UV_LINK_MODE": "symlink", "PATH": os.environ.get("PATH", "")}

    monkeypatch.setattr(cd, "_child_env", fake_child_env)
    code = (
        "import os; print('https://user:s3cret@mirror.example/simple', os.environ['UV_LINK_MODE'])"
    )
    ok, out = cd._run([sys.executable, "-c", code], 60)
    assert ok and "s3cret" not in out and out.endswith("copy")


def test_an_unusable_lock_fails_at_once(monkeypatch, tmp_path):
    class BrokenLock:
        def acquire(self, timeout):
            raise PermissionError("read-only file system")

    monkeypatch.setattr(cd, "_file_lock", lambda _p: BrokenLock())
    monkeypatch.setattr(cd, "_run", _no_install)
    import time

    t = time.monotonic()
    lib, reason = cd.ensure_library(RT, root = tmp_path)
    assert lib is None and "cannot lock" in reason and time.monotonic() - t < 5


def test_an_unexpected_install_error_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(cd, "binary_cudnn_build", lambda _b: (True, 12))

    def boom(*_a, **_k):
        raise OSError("disk went away")

    monkeypatch.setattr(cd, "ensure_library", boom)
    plan = cd.plan_cudnn_attention(
        "/x/sd-cli", (10, 0), platform = "linux", environ = {}, root = tmp_path
    )
    assert plan.state == cd.STATE_UNAVAILABLE and "disk went away" in plan.reason and plan.env == ()
