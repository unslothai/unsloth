"""_ensure_cuda_torch reinstalls CUDA torch when an NVIDIA-host venv carries a ROCm
build (the pre-fix KFD gpu_id false positive), but leaves healthy CUDA / CPU / ROCm /
macOS / Windows untouched. Fully mocked -- no GPU required.

Also covers _ensure_expected_torch_flavor, the Windows counterpart: _ensure_cuda_torch
returns early on Windows because setup.ps1 owns torch there, which left the update path
with no flavor invariant at all. See the bottom of this file."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Load module under test (mirrors test_rocm_support.py).

PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_STACK_SPEC = importlib.util.spec_from_file_location("studio_install_python_stack", _STACK_PATH)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
stack_mod = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = stack_mod
_STACK_SPEC.loader.exec_module(stack_mod)

# The probe prints its answer behind this marker, so chatter on either side of it cannot
# be mistaken for the answer. Mocked stdout has to carry it too.
_MARK = stack_mod._TORCH_PROBE_MARKER

_ensure_cuda_torch = stack_mod._ensure_cuda_torch
_detect_cuda_torch_index_url = stack_mod._detect_cuda_torch_index_url


def _dotted_cuda(tag):
    """ "cu128" -> "12.8", the torch.version.cuda form. Minor is the last digit."""
    digits = tag[2:] if tag.startswith("cu") else tag
    if not digits.isdigit() or len(digits) < 2:
        return ""
    return f"{digits[:-1]}.{digits[-1]}"


def _torch_probe_line(torch_state, cuda_version):
    """Render the shared probe's "<version>|<hip>|<cuda>" line for a test's
    "<marker>|<installed_cu>|<release>|<runtime_cu>" shorthand.

    _ensure_cuda_torch now derives the marker from torch.__version__ /
    torch.version.hip / torch.version.cuda rather than reading a pre-computed
    marker, so the mock has to emit a self-consistent build: a "cuda" marker
    means torch.version.cuda is set, which always yields a runtime family.
    """
    marker, installed_cu, release, runtime_cu = (torch_state.split("|") + ["", "", ""])[:4]
    release = release or "2.9.1"
    if marker == "hip":
        return _MARK + f"{release}+rocm6.4|6.4|"
    if marker == "cuda":
        version = f"{release}+{installed_cu}" if installed_cu else release
        cuda = _dotted_cuda(runtime_cu) or _dotted_cuda(installed_cu) or cuda_version
        return _MARK + f"{version}||{cuda}"
    return _MARK + f"{release}||"


def _make_run(
    torch_state = "hip",
    cuda_version = "12.8",
    torch_rc = 0,
    smi_rc = 0,
    compute_caps = ("8.6",),
):
    """subprocess.run side_effect: the shared torch probe (sys.executable, text
    stdout) vs the nvidia-smi version / compute-capability probes (smi path,
    text=True), keyed on the executable."""

    def _run(cmd, *args, **kwargs):
        result = MagicMock()
        exe = str(cmd[0]) if cmd else ""
        if exe == sys.executable:
            result.returncode = torch_rc
            out = _torch_probe_line(torch_state, cuda_version) + "\n"
            result.stdout = out if kwargs.get("text") else out.encode()
            return result
        result.returncode = smi_rc
        if len(cmd) > 1 and str(cmd[1]) == "--query-gpu=compute_cap":
            out = "".join(f"{cap}\n" for cap in compute_caps)
        else:
            out = f"CUDA Version: {cuda_version}\n" if cuda_version else "No devices found\n"
        result.stdout = out if kwargs.get("text") else out.encode()
        return result

    return _run


def _run_cuda_repair(
    *,
    backend = "",
    nvidia = True,
    torch_state = "hip",
    cuda_version = "12.8",
    torch_rc = 0,
    smi_rc = 0,
    compute_caps = ("8.6",),
    machine = "x86_64",
    is_macos = False,
    is_windows = False,
    no_torch = False,
    rocm_marker = False,
    smi_path = "/usr/bin/nvidia-smi",
    cvd = None,
    index_family = None,
    index_url = None,
):
    """Invoke _ensure_cuda_torch under a fully mocked host; return the pip mock.

    cvd controls CUDA_VISIBLE_DEVICES: None removes it from the env, any string sets it.
    index_family sets UNSLOTH_TORCH_INDEX_FAMILY (the explicit wheel-index pin).
    index_url sets UNSLOTH_TORCH_INDEX_URL (the full-URL pin form).
    compute_caps is what nvidia-smi reports for --query-gpu=compute_cap; machine
    pins platform.machine() so the architecture policy behaves the same on any test
    host."""
    env = {}
    if rocm_marker:
        env["UNSLOTH_ROCM_TORCH_INSTALLED"] = "1"
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd
    if index_family is not None:
        env["UNSLOTH_TORCH_INDEX_FAMILY"] = index_family
    if index_url is not None:
        env["UNSLOTH_TORCH_INDEX_URL"] = index_url

    def _which(name, *a, **k):
        if name == "nvidia-smi":
            return smi_path
        return None

    # The torch classification is memoized for the life of an install run, so each
    # scenario has to start from a clean slate.
    stack_mod._invalidate_torch_runtime_probe()

    with (
        patch.object(stack_mod, "_TORCH_BACKEND", backend),
        patch.object(stack_mod, "IS_MACOS", is_macos),
        patch.object(stack_mod, "IS_WINDOWS", is_windows),
        patch.object(stack_mod, "NO_TORCH", no_torch),
        patch.object(stack_mod.platform, "machine", return_value = machine),
        patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = nvidia),
        patch.object(stack_mod.shutil, "which", side_effect = _which),
        patch.object(stack_mod.os.path, "isfile", return_value = bool(smi_path)),
        patch.object(stack_mod, "pip_install") as mock_pip,
        patch.object(
            stack_mod.subprocess,
            "run",
            side_effect = _make_run(torch_state, cuda_version, torch_rc, smi_rc, compute_caps),
        ),
        patch.dict(stack_mod.os.environ, env, clear = False),
    ):
        if not rocm_marker:
            stack_mod.os.environ.pop("UNSLOTH_ROCM_TORCH_INSTALLED", None)
        if cvd is None:
            stack_mod.os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        if index_family is None:
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_FAMILY", None)
        if index_url is None:
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_URL", None)
        _ensure_cuda_torch()
    return mock_pip


def _index_url(mock_pip) -> str:
    """Return the --index-url value from the recorded pip_install call."""
    args = [str(a) for a in mock_pip.call_args.args]
    return args[args.index("--index-url") + 1]


# Repair fires only on the poisoning signature.


class TestCudaRepairFires:
    def test_hip_build_on_nvidia_triggers_repair(self):
        mock_pip = _run_cuda_repair(torch_state = "hip", cuda_version = "12.8")
        assert mock_pip.call_count == 1
        call_args = [str(a) for a in mock_pip.call_args.args]
        assert "--force-reinstall" in call_args
        assert "--no-cache-dir" in call_args
        assert "cu128" in _index_url(mock_pip)
        assert mock_pip.call_args.kwargs["constrain"] is False

    def test_rocm_in_version_string_triggers_repair(self):
        # AMD SDK / Radeon wheels may encode rocm in __version__ without torch.version.hip;
        # the probe prints "hip" for both.
        mock_pip = _run_cuda_repair(torch_state = "hip")
        assert mock_pip.call_count == 1

    def test_no_gpu_but_explicit_cuda_pin_repairs(self):
        # Headless / CI cross-install: an explicit cu* pin commits to CUDA wheels with no
        # NVIDIA GPU visible, so a ROCm-poisoned venv is still repaired to the pinned family.
        mock_pip = _run_cuda_repair(
            nvidia = False,
            backend = "cuda",
            index_family = "cu128",
            torch_state = "hip",
        )
        assert mock_pip.call_count == 1
        assert "cu128" in _index_url(mock_pip)

    def test_cvd_hidden_but_explicit_cuda_pin_repairs(self):
        # CVD=-1/"" hides the GPU, but an explicit cu* pin skips ALL host-GPU probing, so the
        # CVD hide gate must not suppress the repair (GPU-less CI: CVD=-1, FAMILY=cu128).
        for _cvd in ("-1", ""):
            mock_pip = _run_cuda_repair(
                nvidia = False,
                backend = "cuda",
                cvd = _cvd,
                index_family = "cu128",
                torch_state = "hip",
            )
            assert mock_pip.call_count == 1
            assert "cu128" in _index_url(mock_pip)

    def test_tagged_cuda_mismatch_repairs(self):
        # A healthy CUDA torch whose +cuXXX differs from the pin is repaired.
        mock_pip = _run_cuda_repair(
            index_family = "cu128",
            torch_state = "cuda|cu126",
            cuda_version = "12.8",
        )
        assert mock_pip.call_count == 1
        assert "cu128" in _index_url(mock_pip)

    def test_untagged_cuda_build_under_pin_repairs(self):
        # An untagged CUDA build (no +cuXXX tag -> empty installed cu) can't be confirmed
        # to match the pin, so the pin is enforced with a reinstall.
        mock_pip = _run_cuda_repair(
            index_family = "cu128",
            torch_state = "cuda",  # marker cuda, empty installed cu
            cuda_version = "12.8",
        )
        assert mock_pip.call_count == 1
        assert "cu128" in _index_url(mock_pip)

    def test_broken_probe_with_cuda_pin_repairs(self):
        # torch present but unimportable under a CUDA pin: the base update won't repair a
        # broken already-installed torch, so reinstall from the pin instead of stranding it.
        mock_pip = _run_cuda_repair(torch_state = "hip", torch_rc = 1, index_family = "cu128")
        assert mock_pip.call_count == 1
        assert "cu128" in _index_url(mock_pip)

    def test_broken_probe_with_cuda_url_pin_repairs(self):
        mock_pip = _run_cuda_repair(
            torch_state = "cpu",
            torch_rc = 1,
            index_url = "https://mirror.local/cu128",
        )
        assert mock_pip.call_count == 1
        assert "https://mirror.local/cu128" in _index_url(mock_pip)


# No-op cases.


class TestCudaRepairSkips:
    def test_healthy_cuda_torch_no_repair(self):
        mock_pip = _run_cuda_repair(torch_state = "cuda")
        mock_pip.assert_not_called()

    def test_deliberate_cpu_wheel_no_repair(self):
        mock_pip = _run_cuda_repair(torch_state = "cpu")
        mock_pip.assert_not_called()

    def test_backend_rocm_skips(self):
        mock_pip = _run_cuda_repair(backend = "rocm", torch_state = "hip")
        mock_pip.assert_not_called()

    def test_backend_cpu_skips(self):
        mock_pip = _run_cuda_repair(backend = "cpu", torch_state = "hip")
        mock_pip.assert_not_called()

    def test_unknown_backend_skips(self):
        mock_pip = _run_cuda_repair(backend = "auto", torch_state = "hip")
        mock_pip.assert_not_called()

    def test_no_nvidia_gpu_skips(self):
        mock_pip = _run_cuda_repair(nvidia = False, torch_state = "hip")
        mock_pip.assert_not_called()

    def test_torch_missing_no_pin_skips(self):
        # Non-zero probe exit = torch missing/un-importable. With NO CUDA pin the base
        # install owns it, so leave it alone (a pinned build reinstalls).
        mock_pip = _run_cuda_repair(torch_state = "hip", torch_rc = 1)
        mock_pip.assert_not_called()

    def test_macos_skips(self):
        mock_pip = _run_cuda_repair(is_macos = True, torch_state = "hip")
        mock_pip.assert_not_called()

    def test_windows_skips(self):
        mock_pip = _run_cuda_repair(is_windows = True, torch_state = "hip")
        mock_pip.assert_not_called()

    def test_no_torch_mode_skips(self):
        mock_pip = _run_cuda_repair(no_torch = True, torch_state = "hip")
        mock_pip.assert_not_called()

    def test_rocm_install_marker_skips(self):
        mock_pip = _run_cuda_repair(rocm_marker = True, torch_state = "hip")
        mock_pip.assert_not_called()

    def test_cvd_minus_one_skips(self):
        # CUDA_VISIBLE_DEVICES=-1 hides the NVIDIA GPU (mixed AMD+NVIDIA host on the AMD card).
        mock_pip = _run_cuda_repair(cvd = "-1", torch_state = "hip")
        mock_pip.assert_not_called()

    def test_cvd_empty_skips(self):
        mock_pip = _run_cuda_repair(cvd = "", torch_state = "hip")
        mock_pip.assert_not_called()

    def test_cvd_explicit_device_still_repairs(self):
        mock_pip = _run_cuda_repair(cvd = "0", torch_state = "hip")
        assert mock_pip.call_count == 1

    def test_matching_tagged_cuda_pin_no_repair(self):
        # Healthy CUDA torch whose +cuXXX already matches the pin: no reinstall.
        mock_pip = _run_cuda_repair(
            index_family = "cu128",
            torch_state = "cuda|cu128",
            cuda_version = "12.8",
        )
        mock_pip.assert_not_called()

    def test_custom_mirror_leaf_not_treated_as_cuda_pin(self):
        # A mirror leaf starting with "cu" but not cuXXX (.../custom, .../current) must
        # NOT be treated as a CUDA pin, so it can't bypass the NVIDIA gate.
        for _leaf in ("custom", "current"):
            mock_pip = _run_cuda_repair(
                nvidia = False,
                backend = "cuda",
                index_url = f"https://mymirror.example/{_leaf}",
                torch_state = "hip",
            )
            mock_pip.assert_not_called()

    def test_explicit_cuda_family_leaf_helper(self):
        # _explicit_cuda_torch_index_url matches cuXXX narrowly, not any cu* leaf.
        import contextlib

        def _with(url):
            with patch.dict(stack_mod.os.environ, {"UNSLOTH_TORCH_INDEX_URL": url}, clear = False):
                stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_FAMILY", None)
                return stack_mod._explicit_cuda_torch_index_url()

        assert _with("https://download.pytorch.org/whl/cu128") is not None
        assert _with("https://download.pytorch.org/whl/cu126") is not None
        assert _with("https://mymirror.example/custom") is None
        assert _with("https://mymirror.example/current") is None
        assert _with("https://download.pytorch.org/whl/cpu") is None
        with contextlib.suppress(Exception):
            stack_mod.os.environ.pop("UNSLOTH_TORCH_INDEX_URL", None)


class TestTorchBackendDerivationFromPin:
    """The module-level _TORCH_BACKEND derivation (standalone `studio update`
    with no install.sh-set UNSLOTH_TORCH_BACKEND) must classify the pinned index
    leaf via _is_cuda_family_leaf (^cu[0-9]), NOT a bare startswith("cu"). A
    full-override URL ending in /current or /custom must fall through to backend
    "" (probe the GPU) so _ensure_rocm_torch() still repairs a wrong/CPU torch on
    AMD hosts, instead of being wrongly branded "cuda" and returning early."""

    @staticmethod
    def _derive(env):
        # Re-run the module's import-time derivation, using its own _is_cuda_family_leaf
        # so this stays in lockstep.
        idx_override = (
            env.get("UNSLOTH_TORCH_INDEX_URL", "").strip()
            or env.get("UNSLOTH_TORCH_INDEX_FAMILY", "").strip()
        )
        backend = env.get("UNSLOTH_TORCH_BACKEND", "").lower()
        if not backend:
            leaf = idx_override.rstrip("/").rsplit("/", 1)[-1].lower()
            if leaf.startswith(("rocm", "gfx")):
                backend = "rocm"
            elif leaf == "cpu":
                backend = "cpu"
            elif stack_mod._is_cuda_family_leaf(leaf):
                backend = "cuda"
        return backend

    def test_cu128_pin_is_cuda(self):
        assert (
            self._derive({"UNSLOTH_TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128"})
            == "cuda"
        )

    def test_cu128_family_is_cuda(self):
        assert self._derive({"UNSLOTH_TORCH_INDEX_FAMILY": "cu128"}) == "cuda"

    def test_current_leaf_not_cuda(self):
        # ^cu[0-9] rejects /current -> backend stays "" (probe GPU), so an AMD host still
        # repairs a CPU/wrong torch instead of short-circuiting.
        assert self._derive({"UNSLOTH_TORCH_INDEX_URL": "https://mymirror.example/current"}) == ""

    def test_custom_leaf_not_cuda(self):
        assert self._derive({"UNSLOTH_TORCH_INDEX_URL": "https://mymirror.example/custom"}) == ""

    def test_rocm_and_gfx_pins_are_rocm(self):
        assert self._derive({"UNSLOTH_TORCH_INDEX_FAMILY": "rocm7.2"}) == "rocm"
        assert (
            self._derive({"UNSLOTH_TORCH_INDEX_URL": "https://repo.amd.com/rocm/whl/gfx120X-all"})
            == "rocm"
        )

    def test_cpu_pin_is_cpu(self):
        assert self._derive({"UNSLOTH_TORCH_INDEX_FAMILY": "cpu"}) == "cpu"

    def test_source_uses_helper_not_bare_startswith(self):
        # Guard against a regression back to elif _idx_leaf.startswith("cu").
        src = _STACK_PATH.read_text(encoding = "utf-8")
        assert (
            "elif _is_cuda_family_leaf(_idx_leaf):" in src
        ), "_TORCH_BACKEND derivation must classify CUDA via _is_cuda_family_leaf"
        assert (
            'elif _idx_leaf.startswith("cu"):' not in src
        ), "_TORCH_BACKEND derivation must not use a bare startswith('cu')"


# CUDA index ladder.


class TestCudaIndexResolution:
    def test_cuda_128_selects_cu128(self):
        assert "cu128" in _index_url(_run_cuda_repair(cuda_version = "12.8"))

    def test_cuda_130_selects_cu130(self):
        assert "cu130" in _index_url(_run_cuda_repair(cuda_version = "13.0"))

    def test_cuda_126_selects_cu126(self):
        assert "cu126" in _index_url(_run_cuda_repair(cuda_version = "12.6"))

    def test_cuda_124_selects_cu124(self):
        assert "cu124" in _index_url(_run_cuda_repair(cuda_version = "12.4"))

    def test_cuda_118_selects_cu118(self):
        assert "cu118" in _index_url(_run_cuda_repair(cuda_version = "11.8"))

    def test_unreadable_version_defaults_cu126(self):
        # nvidia-smi runs but prints no CUDA version line (or fails).
        mock_pip = _run_cuda_repair(cuda_version = "", smi_rc = 1)
        assert "cu126" in _index_url(mock_pip)

    def test_proc_fallback_no_smi_defaults_cu126(self):
        # NVIDIA usable via /proc fallback, nvidia-smi absent.
        mock_pip = _run_cuda_repair(smi_path = None)
        assert "cu126" in _index_url(mock_pip)

    def test_detect_index_url_uses_pytorch_base(self):
        with (
            patch.object(stack_mod.shutil, "which", return_value = None),
            patch.object(stack_mod.os.path, "isfile", return_value = False),
        ):
            url = _detect_cuda_torch_index_url()
        assert url == f"{stack_mod._PYTORCH_WHL_BASE}/cu126"


# PyTorch 2.11's cu128/cu130 start at sm_75, and their CUDA 13 runtime also costs a
# pre-Turing GPU its llama.cpp GGUF bundle, so such hosts get cu126 (#7765).


class TestPreTuringWheelFamily:
    def test_volta_host_selects_cu126_over_the_driver_family(self):
        assert "cu126" in _index_url(_run_cuda_repair(cuda_version = "13.0", compute_caps = ("7.0",)))

    def test_pascal_host_selects_cu126_over_the_driver_family(self):
        assert "cu126" in _index_url(_run_cuda_repair(cuda_version = "12.8", compute_caps = ("6.1",)))

    def test_turing_host_keeps_the_driver_family(self):
        assert "cu130" in _index_url(_run_cuda_repair(cuda_version = "13.0", compute_caps = ("7.5",)))

    def test_mixed_host_within_cu126_range_is_capped(self):
        # cu126 spans sm_50-90, so serving the older card costs the newer one nothing.
        for caps in (("7.0", "8.6"), ("6.1", "9.0"), ("5.0", "7.5")):
            assert "cu126" in _index_url(_run_cuda_repair(cuda_version = "13.0", compute_caps = caps))

    def test_mixed_host_outside_cu126_range_keeps_the_driver_family(self):
        # Blackwell is past cu126's ceiling and Kepler is under its floor, so no family
        # covers either mix whole. Capping would strand the newer card entirely.
        for caps in (("7.0", "12.0"), ("3.7", "8.6")):
            assert "cu130" in _index_url(_run_cuda_repair(cuda_version = "13.0", compute_caps = caps))

    def test_cu126_venv_is_repaired_after_a_blackwell_upgrade(self):
        # The span cuts both ways: a cu126 venv predating a GPU swap has nothing for
        # sm_120, and a fresh install on that host would pick cu130.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu126|2.11.0",
            cuda_version = "13.0",
            compute_caps = ("12.0",),
        )
        assert mock_pip.call_count == 1
        assert "cu130" in _index_url(mock_pip)

    def test_cu126_venv_is_kept_when_the_driver_allows_nothing_newer(self):
        # Same host, CUDA 12.6 driver: cu130 is not installable, so leave it rather
        # than reinstall cu126 over itself on every update.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu126|2.11.0",
            cuda_version = "12.6",
            compute_caps = ("12.0",),
        )
        mock_pip.assert_not_called()

    def test_partial_family_is_not_traded_for_another_partial_family(self):
        # A working V100 + cu126 box gains a Blackwell card. Neither family covers both,
        # so swapping to cu130 would kill the Volta to revive the Blackwell.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu126|2.11.0",
            cuda_version = "13.0",
            compute_caps = ("7.0", "12.0"),
        )
        mock_pip.assert_not_called()

    def test_cu118_kepler_build_is_kept(self):
        # torch 2.7's cu118 still built sm_37 and nothing newer does, so the replacement
        # would strand the GPU that works today.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu118|2.7.1",
            cuda_version = "13.0",
            compute_caps = ("3.7",),
        )
        mock_pip.assert_not_called()

    def test_uncovered_mix_is_not_repaired_in_a_loop(self):
        # The cap declines, so the replacement equals the installed family.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu130|2.11.0",
            cuda_version = "13.0",
            compute_caps = ("7.0", "12.0"),
        )
        mock_pip.assert_not_called()

    def test_mixed_host_within_cu126_range_is_repaired(self):
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu130|2.11.0",
            cuda_version = "13.0",
            compute_caps = ("7.0", "8.6"),
        )
        assert mock_pip.call_count == 1
        assert "cu126" in _index_url(mock_pip)

    def test_partial_inventory_keeps_the_driver_family(self):
        assert "cu130" in _index_url(
            _run_cuda_repair(cuda_version = "13.0", compute_caps = ("7.0", "N/A"))
        )

    def test_incompatible_family_is_repaired(self):
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu130|2.11.0",
            cuda_version = "13.0",
            compute_caps = ("7.0",),
        )
        assert mock_pip.call_count == 1
        assert "cu126" in _index_url(mock_pip)

    def test_pre_211_cu128_volta_build_is_kept(self):
        # torch 2.10's cu128 wheels still shipped sm_70; no reinstall is warranted.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu128|2.10.0",
            cuda_version = "13.0",
            compute_caps = ("7.0",),
        )
        mock_pip.assert_not_called()

    def test_pre_211_cu128_pascal_build_is_repaired(self):
        # ... but they never shipped sm_61, which only cu126 carries.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu128|2.10.0",
            cuda_version = "13.0",
            compute_caps = ("6.1",),
        )
        assert mock_pip.call_count == 1
        assert "cu126" in _index_url(mock_pip)

    def test_compatible_family_is_kept(self):
        for state, caps in (
            ("cuda|cu126|2.11.0", ("7.0",)),
            ("cuda|cu130|2.11.0", ("7.5",)),
            ("cuda|cu130|2.11.0", ("7.0", "12.0")),
            ("cuda|cu130|2.11.0", ("7.0", "N/A")),
        ):
            mock_pip = _run_cuda_repair(torch_state = state, cuda_version = "13.0", compute_caps = caps)
            mock_pip.assert_not_called()

    def test_explicit_pin_wins_over_the_architecture_policy(self):
        for pin in ("index_family", "index_url"):
            value = "cu130" if pin == "index_family" else "https://example.test/whl/cu130"
            mock_pip = _run_cuda_repair(
                torch_state = "cuda|cu130|2.11.0",
                cuda_version = "13.0",
                compute_caps = ("7.0",),
                **{pin: value},
            )
            mock_pip.assert_not_called()

    def test_untagged_cuda_build_uses_the_runtime_family(self):
        # An untagged build still reports torch.version.cuda, so _family falls back to
        # the runtime value and the architecture policy applies.
        # The "family unknown, leave it alone" branch needs BOTH the tag and
        # torch.version.cuda empty, which reads as a CPU build, so an untagged CUDA
        # build is always classifiable.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda||2.11.0",
            cuda_version = "13.0",
            compute_caps = ("7.0",),
        )
        assert mock_pip.call_count == 1
        assert "cu126" in _index_url(mock_pip)

    def test_non_x86_host_keeps_the_driver_family(self):
        # No aarch64 CUDA family ships sm_<80 kernels, so cu126 cannot help there.
        volta = MagicMock(returncode = 0, stdout = "7.0\n")
        with patch.object(stack_mod.subprocess, "run", return_value = volta):
            with patch.object(stack_mod.platform, "machine", return_value = "aarch64"):
                assert stack_mod._cap_cuda_family_for_pre_turing("cu130", "smi") == "cu130"
            with patch.object(stack_mod.platform, "machine", return_value = "x86_64"):
                assert stack_mod._cap_cuda_family_for_pre_turing("cu130", "smi") == "cu126"

    def test_permissive_family_is_never_probed(self):
        # cu126 has nothing older to fall back to, so it must not spawn nvidia-smi.
        with (
            patch.object(stack_mod.platform, "machine", return_value = "x86_64"),
            patch.object(stack_mod.subprocess, "run") as mock_run,
        ):
            assert stack_mod._cap_cuda_family_for_pre_turing("cu126", "smi") == "cu126"
            assert stack_mod._cap_cuda_family_for_pre_turing("cpu", "smi") == "cpu"
        mock_run.assert_not_called()

    def test_family_spans_track_the_pytorch_wheel_matrix(self):
        # Read off pytorch's .ci/manywheel/build_cuda.sh at each release tag. cu118 kept
        # Kepler: torch 2.7 still built sm_37 for it.
        assert stack_mod._cuda_family_sm_range("cu118") == (37, 90)
        assert stack_mod._cuda_family_sm_range("cu124") == (50, 90)
        assert stack_mod._cuda_family_sm_range("cu126") == (50, 90)
        assert stack_mod._cuda_family_sm_range("cu126", "2.10.0") == (50, 90)
        assert stack_mod._cuda_family_sm_range("cu128") == (75, 120)
        assert stack_mod._cuda_family_sm_range("cu128", "2.11.0") == (75, 120)
        assert stack_mod._cuda_family_sm_range("cu129", "2.9.0") == (70, 120)
        assert stack_mod._cuda_family_sm_range("cu130", "2.10.0") == (75, 120)
        assert stack_mod._cuda_family_sm_range("cpu") is None
        assert stack_mod._cuda_family_sm_range("") is None

    def test_cu128_volta_window_opens_at_torch_28(self):
        # 2.7's cu128 dropped sm_50-70 when CUDA 12.8 deprecated them; 2.8 put sm_70
        # back and 2.11 took it away again.
        assert stack_mod._cuda_family_sm_range("cu128", "2.7.1")[0] == 75
        assert stack_mod._cuda_family_sm_range("cu128", "2.8.0")[0] == 70
        assert stack_mod._cuda_family_sm_range("cu128", "2.10.0")[0] == 70
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu128|2.7.1",
            cuda_version = "13.0",
            compute_caps = ("7.0",),
        )
        assert mock_pip.call_count == 1
        assert "cu126" in _index_url(mock_pip)

    def test_untagged_pypi_wheel_is_classified_by_its_cuda_runtime(self):
        # PyPI forbids local versions, so a torch from PyPI has no +cuXXX tag;
        # torch.version.cuda is the only clue that it is a CUDA 13 build.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda||2.11.0|cu130",
            cuda_version = "13.0",
            compute_caps = ("7.0",),
        )
        assert mock_pip.call_count == 1
        assert "cu126" in _index_url(mock_pip)

        healthy = _run_cuda_repair(
            torch_state = "cuda||2.11.0|cu126",
            cuda_version = "13.0",
            compute_caps = ("7.0",),
        )
        healthy.assert_not_called()

    def test_repair_is_skipped_when_it_would_reinstall_the_same_family(self):
        # aarch64 has no CUDA family below sm_80, so the cap declines and the replacement
        # would be the condemned wheel itself, once per update forever.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu130|2.11.0",
            cuda_version = "13.0",
            compute_caps = ("7.0",),
            machine = "aarch64",
        )
        mock_pip.assert_not_called()

    def test_compute_sms_rejects_an_unreadable_inventory(self):
        def _sms(stdout, returncode = 0):
            with patch.object(
                stack_mod.subprocess,
                "run",
                return_value = MagicMock(returncode = returncode, stdout = stdout),
            ):
                return stack_mod._nvidia_compute_sms("nvidia-smi")

        assert _sms("7.0\n12.0\n") == [70, 120]
        assert _sms("  8.6  \n\n") == [86]
        assert _sms("7.0\nN/A\n") is None
        assert _sms("") is None
        assert _sms("7.0\n", returncode = 1) is None


# ── Windows torch-flavor invariant ───────────────────────────────────────────
#
# The in-app updater runs `unsloth studio update` -> setup.ps1 -> install_python_stack.py.
# install.ps1, which holds the only torch-flavor repair there has ever been, is never on
# that path, and the dependency steps install with deps and without an --index-url, so the
# resolver's default source is PyPI -- whose Windows torch is 2.11.0+cpu. Two users' cu124
# venvs came out of an update as 2.11.0+cpu with the update reporting success.

_ensure_expected_torch_flavor = stack_mod._ensure_expected_torch_flavor
_UNSET = object()


def _flavor_probe_stdout(version):
    """The shared probe's marked line for a literal torch.__version__.

    torch.version.cuda / .hip follow the local label, because a real wheel sets them
    together and _torch_build_is_gpu reads all three.
    """
    match = re.search(r"\+(cu\d+)", version)
    cuda = _dotted_cuda(match.group(1)) if match else ""
    hip = "6.4" if "+rocm" in version else ""
    return _MARK + f"{version}|{hip}|{cuda}\n"


def _run_flavor_invariant(
    *,
    installed = "2.11.0+cpu",
    repaired = None,
    expected_env = "cu124",
    recorded = None,
    install_index_url = None,
    backend = "",
    no_torch = False,
    nvidia = True,
    cuda_version = "12.4",
    torch_rc = 0,
    probe_timeout = False,
    disk_label = _UNSET,
    cvd = None,
    index_family = None,
    index_url = None,
):
    """Invoke _ensure_expected_torch_flavor against a fully mocked venv.

    `installed` is torch.__version__ before the pass. `repaired` is what the mocked
    pip_install leaves behind: None means the reinstall changed nothing, which is the
    state that must FAIL the update rather than report success.

    `expected_env` sets UNSLOTH_EXPECTED_TORCH_TAG (setup.ps1's handover), `recorded` the
    flavor read out of the previous manifest, and leaving both None forces the live probe.
    `disk_label` overrides the on-disk torch/version.py label the wedged-probe path reads.

    Returns (ok, pip_mock).
    """
    state = {"version": installed}

    env = {}
    if expected_env is not None:
        env["UNSLOTH_EXPECTED_TORCH_TAG"] = expected_env
    if install_index_url is not None:
        env["UNSLOTH_TORCH_INSTALL_INDEX_URL"] = install_index_url
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd
    if index_family is not None:
        env["UNSLOTH_TORCH_INDEX_FAMILY"] = index_family
    if index_url is not None:
        env["UNSLOTH_TORCH_INDEX_URL"] = index_url

    def _run(cmd, *args, **kwargs):
        result = MagicMock()
        exe = str(cmd[0]) if cmd else ""
        if exe == sys.executable:
            if probe_timeout:
                raise subprocess.TimeoutExpired(cmd, 90)
            result.returncode = torch_rc
            out = _flavor_probe_stdout(state["version"])
        else:
            result.returncode = 0
            if len(cmd) > 1 and str(cmd[1]) == "--query-gpu=compute_cap":
                out = "8.6\n"
            else:
                out = f"CUDA Version: {cuda_version}\n" if cuda_version else "No devices\n"
        result.stdout = out if kwargs.get("text") else out.encode()
        return result

    def _pip(*args, **kwargs):
        # The real pip_install invalidates the memoized classification. The mock has to
        # as well, or the re-probe after the repair answers with the pre-repair venv and
        # every successful repair would read as a failure.
        stack_mod._invalidate_torch_runtime_probe()
        if repaired is not None:
            state["version"] = repaired

    def _which(name, *a, **k):
        return "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None

    stack_mod._invalidate_torch_runtime_probe()

    with (
        patch.object(stack_mod, "_TORCH_BACKEND", backend),
        patch.object(stack_mod, "NO_TORCH", no_torch),
        patch.object(stack_mod, "_RECORDED_TORCH_TAG", recorded),
        patch.object(stack_mod.platform, "machine", return_value = "AMD64"),
        patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = nvidia),
        patch.object(stack_mod.shutil, "which", side_effect = _which),
        patch.object(stack_mod.os.path, "isfile", return_value = True),
        patch.object(
            stack_mod,
            "_installed_torch_label_on_disk",
            side_effect = (
                (lambda: state["version"].lower()) if disk_label is _UNSET else (lambda: disk_label)
            ),
        ),
        patch.object(stack_mod, "pip_install", side_effect = _pip) as mock_pip,
        patch.object(stack_mod.subprocess, "run", side_effect = _run),
        patch.dict(stack_mod.os.environ, env, clear = False),
    ):
        for name, value in (
            ("UNSLOTH_EXPECTED_TORCH_TAG", expected_env),
            ("UNSLOTH_TORCH_INSTALL_INDEX_URL", install_index_url),
            ("CUDA_VISIBLE_DEVICES", cvd),
            ("UNSLOTH_TORCH_INDEX_FAMILY", index_family),
            ("UNSLOTH_TORCH_INDEX_URL", index_url),
        ):
            if value is None:
                stack_mod.os.environ.pop(name, None)
        ok = _ensure_expected_torch_flavor()
    return ok, mock_pip


class TestExpectedTorchFlavorRepairs:
    def test_cpu_torch_under_a_cu124_expectation_is_repaired(self):
        ok, mock_pip = _run_flavor_invariant(repaired = "2.10.0+cu124")
        assert ok is True
        assert mock_pip.call_count == 1
        call_args = [str(a) for a in mock_pip.call_args.args]
        assert "--force-reinstall" in call_args
        assert "--no-cache-dir" in call_args
        assert _index_url(mock_pip).endswith("/cu124")
        assert mock_pip.call_args.kwargs["constrain"] is False

    def test_the_repair_uses_install_ps1s_bounded_trio(self):
        _ok, mock_pip = _run_flavor_invariant(repaired = "2.10.0+cu124")
        call_args = [str(a) for a in mock_pip.call_args.args]
        # Bounded, and NOT the <2.12 Linux range: an unbounded trio can resolve straight
        # back to the 2.11 wheel this repair exists to remove.
        for spec in ("torch>=2.4,<2.11.0", "torchvision>=0.19,<0.26.0", "torchaudio>=2.4,<2.11.0"):
            assert spec in call_args

    def test_untagged_pypi_wheel_is_repaired(self):
        # PyPI forbids the local +cuNNN label, so an untagged wheel is the PyPI build --
        # CPU-only on Windows. install.ps1's ConvertTo-TorchFlavorTag says "cpu" too.
        ok, mock_pip = _run_flavor_invariant(installed = "2.11.0", repaired = "2.10.0+cu124")
        assert ok is True
        assert mock_pip.call_count == 1

    def test_wrong_cuda_family_is_repaired_to_the_expected_one(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.9.1+cu118",
            expected_env = "cu128",
            repaired = "2.10.0+cu128",
        )
        assert ok is True
        assert _index_url(mock_pip).endswith("/cu128")

    def test_a_wedged_probe_classifies_from_disk_and_still_repairs(self):
        # A stalled GPU driver hangs `import torch`, which is the host these repairs exist
        # for. version.py on disk still names the wheel.
        ok, mock_pip = _run_flavor_invariant(
            probe_timeout = True,
            repaired = "2.10.0+cu124",
        )
        assert ok is True
        assert mock_pip.call_count == 1


class TestExpectedTorchFlavorFailsTheUpdate:
    def test_a_repair_that_leaves_cpu_torch_fails(self):
        # The missing invariant: today this state exits 0 and the app silently runs on CPU.
        ok, mock_pip = _run_flavor_invariant(repaired = None)
        assert ok is False
        assert mock_pip.call_count == 1

    def test_the_failure_verdict_reads_torch_version_cuda_not_just_the_tag(self):
        # An untagged wheel that does carry a CUDA runtime is a GPU build. Reinstalling it
        # is right (the family is unconfirmable), failing the update over it is not.
        ok, _mock_pip = _run_flavor_invariant(installed = "2.11.0", repaired = "2.11.0+cu124")
        assert ok is True

    def test_a_rocm_build_left_behind_is_not_called_cpu_only(self):
        ok, _mock_pip = _run_flavor_invariant(repaired = "2.9.1+rocm6.4")
        assert ok is True


class TestExpectedTorchFlavorSkips:
    def test_expected_cpu_is_a_no_op(self):
        ok, mock_pip = _run_flavor_invariant(expected_env = "cpu")
        assert ok is True
        mock_pip.assert_not_called()

    @pytest.mark.parametrize("backend", ["cpu", "rocm", "xpu", "auto"])
    def test_a_deliberate_backend_is_a_no_op(self, backend):
        ok, mock_pip = _run_flavor_invariant(backend = backend)
        assert ok is True
        mock_pip.assert_not_called()

    def test_backend_cuda_still_runs(self):
        ok, mock_pip = _run_flavor_invariant(backend = "cuda", repaired = "2.10.0+cu124")
        assert ok is True
        assert mock_pip.call_count == 1

    def test_no_torch_mode_is_a_no_op(self):
        ok, mock_pip = _run_flavor_invariant(no_torch = True)
        assert ok is True
        mock_pip.assert_not_called()

    def test_a_matching_flavor_is_a_no_op(self):
        ok, mock_pip = _run_flavor_invariant(installed = "2.9.1+cu124")
        assert ok is True
        mock_pip.assert_not_called()

    @pytest.mark.parametrize("cvd", ["", "-1", " ", "  -1 "])
    def test_hidden_cuda_devices_is_a_no_op(self, cvd):
        ok, mock_pip = _run_flavor_invariant(cvd = cvd)
        assert ok is True
        mock_pip.assert_not_called()

    def test_an_explicit_visible_device_still_repairs(self):
        ok, mock_pip = _run_flavor_invariant(cvd = "0", repaired = "2.10.0+cu124")
        assert ok is True
        assert mock_pip.call_count == 1

    @pytest.mark.parametrize("tag", ["rocm", "xpu", "current", "custom", "simple", "cu"])
    def test_a_non_cuda_expectation_is_a_no_op(self, tag):
        # rocm/xpu belong to the paths that own them (setup.ps1 installs both trios
        # itself); an unknown mirror leaf names no flavor at all.
        ok, mock_pip = _run_flavor_invariant(expected_env = tag)
        assert ok is True
        mock_pip.assert_not_called()

    def test_an_empty_handover_tag_falls_through_rather_than_deciding(self):
        # PowerShell deletes an $env: entry assigned "", so setup.ps1 publishes an unknown
        # leaf as empty. That must read as "nobody said", not as "cpu".
        ok, mock_pip = _run_flavor_invariant(
            expected_env = "",
            recorded = "cu128",
            repaired = "2.10.0+cu128",
        )
        assert ok is True
        assert _index_url(mock_pip).endswith("/cu128")

    def test_missing_or_unimportable_torch_is_a_no_op(self):
        # The base install owns a torch that cannot import at all, and force-reinstalling
        # over it turns a broken driver into a wheel problem. install.ps1 declines too.
        ok, mock_pip = _run_flavor_invariant(torch_rc = 1)
        assert ok is True
        mock_pip.assert_not_called()

    def test_an_unreadable_venv_is_a_no_op(self):
        # Probe timed out AND no on-disk label: nothing was learned, so nothing is done --
        # ambiguity must never fail an update on its own.
        ok, mock_pip = _run_flavor_invariant(probe_timeout = True, disk_label = "")
        assert ok is True
        mock_pip.assert_not_called()


class TestExpectedTorchFlavorResolution:
    def test_the_manifest_answers_when_the_environment_is_silent(self):
        # A `python install_python_stack.py` with no setup script in front of it.
        ok, mock_pip = _run_flavor_invariant(
            expected_env = None,
            recorded = "cu128",
            repaired = "2.10.0+cu128",
        )
        assert ok is True
        assert _index_url(mock_pip).endswith("/cu128")

    def test_the_environment_wins_over_the_manifest(self):
        _ok, mock_pip = _run_flavor_invariant(
            expected_env = "cu124",
            recorded = "cu128",
            repaired = "2.10.0+cu124",
        )
        assert _index_url(mock_pip).endswith("/cu124")

    def test_the_live_probe_answers_when_nothing_recorded_it(self):
        _ok, mock_pip = _run_flavor_invariant(
            expected_env = None,
            recorded = None,
            cuda_version = "12.8",
            repaired = "2.10.0+cu128",
        )
        assert _index_url(mock_pip).endswith("/cu128")

    def test_the_live_probe_declines_without_an_nvidia_gpu(self):
        # No GPU and no pin means no CUDA expectation exists to enforce; inventing one
        # would reinstall CUDA torch onto a CPU-only box on every update.
        ok, mock_pip = _run_flavor_invariant(expected_env = None, recorded = None, nvidia = False)
        assert ok is True
        mock_pip.assert_not_called()

    def test_a_cpu_pin_beats_the_live_probe(self):
        ok, mock_pip = _run_flavor_invariant(
            expected_env = None, recorded = None, index_family = "cpu"
        )
        assert ok is True
        mock_pip.assert_not_called()

    def test_the_setup_scripts_index_url_is_used_when_its_leaf_matches(self):
        # The only way an authenticated mirror can be repaired: the credentials are not
        # reconstructible from a family leaf.
        _ok, mock_pip = _run_flavor_invariant(
            install_index_url = "https://mirror.local/whl/cu124?token=secret",
            repaired = "2.10.0+cu124",
        )
        assert _index_url(mock_pip) == "https://mirror.local/whl/cu124?token=secret"

    def test_an_index_url_naming_another_family_is_ignored(self):
        # setup.ps1 hands over the /cpu index alongside a "rocm" tag on the AMD Windows
        # path; repairing a cu* mismatch from it would install the CPU wheel being removed.
        _ok, mock_pip = _run_flavor_invariant(
            install_index_url = "https://download.pytorch.org/whl/cpu",
            repaired = "2.10.0+cu124",
        )
        assert _index_url(mock_pip) == f"{stack_mod._PYTORCH_WHL_BASE}/cu124"

    def test_a_matching_family_pin_supplies_the_index(self):
        _ok, mock_pip = _run_flavor_invariant(
            index_url = "https://mirror.local/whl/cu124",
            repaired = "2.10.0+cu124",
        )
        assert _index_url(mock_pip) == "https://mirror.local/whl/cu124"


class TestTorchFlavorTagVocabulary:
    """_torch_flavor_tag is compared against tags install.ps1 and setup.ps1 produced, so
    it has to classify identically."""

    @pytest.mark.parametrize(
        "version,tag",
        [
            ("2.9.1+cu124", "cu124"),
            ("2.11.0+cu130", "cu130"),
            ("2.9.1+rocm6.4", "rocm"),
            ("2.11.0+rocm7.2.1", "rocm"),
            ("2.9.1+xpu", "xpu"),
            ("2.11.0+cpu", "cpu"),
            ("2.11.0", "cpu"),
            ("2.9.1+CU124", "cu124"),
            ("", ""),
        ],
    )
    def test_tags(self, version, tag):
        assert stack_mod._torch_flavor_tag(version) == tag


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
