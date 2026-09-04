"""_ensure_cuda_torch reinstalls CUDA torch when an NVIDIA-host venv carries a ROCm
build (the pre-fix KFD gpu_id false positive), but leaves healthy CUDA / CPU / ROCm /
macOS / Windows untouched. Fully mocked -- no GPU required.

Also covers _ensure_expected_torch_flavor, the Windows counterpart: _ensure_cuda_torch
returns early on Windows because setup.ps1 owns torch there, which left the update path
with no flavor invariant at all. See the bottom of this file."""

import importlib.util
import inspect
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_STACK_SPEC = importlib.util.spec_from_file_location("studio_install_python_stack", _STACK_PATH)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
stack_mod = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = stack_mod
_STACK_SPEC.loader.exec_module(stack_mod)

_SETUP_SRC = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")

# The probe prints its answer behind this marker, so chatter on either side of it cannot be mistaken for the answer.
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

    # The torch classification is memoized for the life of an install run, so each scenario has to start from a clean
    # slate.
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
        # AMD SDK / Radeon wheels may encode rocm in __version__ without torch.version.hip; the probe prints "hip" for
        # both.
        mock_pip = _run_cuda_repair(torch_state = "hip")
        assert mock_pip.call_count == 1

    def test_no_gpu_but_explicit_cuda_pin_repairs(self):
        # Headless / CI cross-install: an explicit cu* pin commits to CUDA wheels with no NVIDIA GPU visible, so a
        # ROCm-poisoned venv is still repaired to the pinned family.
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
        # torch present but unimportable under a CUDA pin: the base update won't repair a broken already-installed
        # torch, so reinstall from the pin instead of stranding it.
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
        # Non-zero probe exit = torch missing/un-importable.
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
        # Re-run the module's import-time derivation, using its own _is_cuda_family_leaf so this stays in lockstep.
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
        mock_pip = _run_cuda_repair(smi_path = None)
        assert "cu126" in _index_url(mock_pip)

    def test_detect_index_url_uses_pytorch_base(self):
        with (
            patch.object(stack_mod.shutil, "which", return_value = None),
            patch.object(stack_mod.os.path, "isfile", return_value = False),
        ):
            url = _detect_cuda_torch_index_url()
        assert url == f"{stack_mod._PYTORCH_WHL_BASE}/cu126"


# PyTorch 2.11's cu128/cu130 start at sm_75, and their CUDA 13 runtime also costs a pre-Turing GPU its llama.cpp GGUF
# bundle, so such hosts get cu126 (#7765).


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
        # Blackwell is past cu126's ceiling and Kepler under its floor, so no family covers either mix whole.
        # Capping would strand the newer card entirely.
        for caps in (("7.0", "12.0"), ("3.7", "8.6")):
            assert "cu130" in _index_url(_run_cuda_repair(cuda_version = "13.0", compute_caps = caps))

    def test_cu126_venv_is_repaired_after_a_blackwell_upgrade(self):
        # The span cuts both ways: a cu126 venv predating a GPU swap has nothing for sm_120, and a fresh install on
        # that host would pick cu130.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu126|2.11.0",
            cuda_version = "13.0",
            compute_caps = ("12.0",),
        )
        assert mock_pip.call_count == 1
        assert "cu130" in _index_url(mock_pip)

    def test_cu126_venv_is_kept_when_the_driver_allows_nothing_newer(self):
        # Same host, CUDA 12.6 driver: cu130 is not installable, so leave it rather than reinstall cu126 over itself
        # on every update.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu126|2.11.0",
            cuda_version = "12.6",
            compute_caps = ("12.0",),
        )
        mock_pip.assert_not_called()

    def test_partial_family_is_not_traded_for_another_partial_family(self):
        # A working V100 + cu126 box gains a Blackwell card. Neither family covers both, so swapping to cu130 would
        # kill the Volta to revive the Blackwell.
        mock_pip = _run_cuda_repair(
            torch_state = "cuda|cu126|2.11.0",
            cuda_version = "13.0",
            compute_caps = ("7.0", "12.0"),
        )
        mock_pip.assert_not_called()

    def test_cu118_kepler_build_is_kept(self):
        # torch 2.7's cu118 still built sm_37 and nothing newer does, so the replacement would strand the GPU that
        # works today.
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
        # An untagged build still reports torch.version.cuda, so _family falls back to the runtime value and the
        # architecture policy applies. The "family unknown, leave it alone" branch needs BOTH the tag and
        # torch.version.cuda empty, which reads as a CPU build, so an untagged CUDA build is always classifiable.
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
        # Read off pytorch's .ci/manywheel/build_cuda.sh at each release tag.
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
        # 2.7's cu128 dropped sm_50-70 when CUDA 12.8 deprecated them; 2.8 put sm_70 back and 2.11 took it away again.
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
        # PyPI forbids local versions, so a torch from PyPI has no +cuXXX tag; torch.version.cuda is the only clue that
        # it is a CUDA 13 build.
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
        # aarch64 has no CUDA family below sm_80, so the cap declines and the replacement would be the condemned wheel
        # itself, once per update forever.
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


# The updater runs setup.ps1 -> install_python_stack.py, never install.ps1, which held the only flavor repair.
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
    win_arm64 = False,
    probe_cuda = None,
    probe_hip = None,
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
            if (probe_cuda is not None or probe_hip is not None) and state["version"] == installed:
                out = _MARK + (f"{state['version']}|{probe_hip or ''}|{probe_cuda or ''}\n")
        else:
            result.returncode = 0
            if len(cmd) > 1 and str(cmd[1]) == "--query-gpu=compute_cap":
                out = "8.6\n"
            else:
                out = f"CUDA Version: {cuda_version}\n" if cuda_version else "No devices\n"
        result.stdout = out if kwargs.get("text") else out.encode()
        return result

    def _pip(*args, **kwargs):
        # The real pip_install invalidates the memoized classification.
        stack_mod._invalidate_torch_runtime_probe()
        if repaired is not None:
            state["version"] = repaired

    def _rocm(*args, **kwargs):
        # The ROCm arm delegates, so the stand-in must move the venv too.
        _pip()

    def _which(name, *a, **k):
        return "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None

    stack_mod._invalidate_torch_runtime_probe()

    with (
        patch.object(stack_mod, "_TORCH_BACKEND", backend),
        patch.object(stack_mod, "NO_TORCH", no_torch),
        patch.object(stack_mod, "_RECORDED_TORCH_TAG", recorded),
        patch.object(stack_mod.platform, "machine", return_value = "AMD64"),
        patch.object(stack_mod, "_is_windows_arm64", return_value = win_arm64),
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
        patch.object(stack_mod, "_ensure_rocm_torch", side_effect = _rocm) as mock_rocm,
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
    mock_pip.rocm_repair = mock_rocm
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
        # The bounded trio install.ps1 repairs with; Windows now shares the Linux <2.12 window.
        for spec in ("torch>=2.4,<2.12.0", "torchvision>=0.19,<0.27.0", "torchaudio>=2.4,<2.12.0"):
            assert spec in call_args

    def test_untagged_pypi_wheel_is_repaired(self):
        # PyPI forbids +cuNNN, so untagged is its CPU-only Windows build.
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
        ok, mock_pip = _run_flavor_invariant(
            probe_timeout = True,
            repaired = "2.10.0+cu124",
        )
        assert ok is True
        assert mock_pip.call_count == 1


class TestExpectedTorchFlavorFailsTheUpdate:
    def test_a_repair_that_leaves_cpu_torch_fails(self):
        ok, mock_pip = _run_flavor_invariant(repaired = None)
        assert ok is False
        assert mock_pip.call_count == 1

    def test_the_failure_verdict_reads_torch_version_cuda_not_just_the_tag(self):
        # Untagged but carrying a CUDA runtime is a GPU build: reinstall, do not fail.
        ok, _mock_pip = _run_flavor_invariant(installed = "2.11.0", repaired = "2.11.0+cu124")
        assert ok is True

    def test_a_rocm_build_left_behind_is_not_called_cpu_only(self, capsys):
        ok, _mock_pip = _run_flavor_invariant(repaired = "2.9.1+rocm6.4")
        assert ok is False
        out = capsys.readouterr().out
        assert "rocm build but cu124 was expected" in out
        assert "CPU-only" not in out


class TestExpectedTorchFlavorSkips:
    def test_expected_cpu_is_a_no_op(self):
        ok, mock_pip = _run_flavor_invariant(expected_env = "cpu")
        assert ok is True
        mock_pip.assert_not_called()

    @pytest.mark.parametrize("backend", ["cpu", "auto"])
    def test_a_non_gpu_backend_is_a_no_op(self, backend):
        """Decided without resolving the expectation: cpu and unrecognised values are
        deliberate whatever the handover said, and this exit costs no GPU probe."""
        ok, mock_pip = _run_flavor_invariant(backend = backend)
        assert ok is True
        mock_pip.assert_not_called()

    @pytest.mark.parametrize("backend", ["rocm", "xpu"])
    def test_a_gpu_backend_that_disagrees_with_the_handover_is_a_no_op(self, backend):
        """An explicit ROCm/XPU pin against a cu124 handover. The pin is the newer and
        more specific instruction; the handover only describes what the install arm
        above it happened to do, so the pin wins and this pass stays out of it."""
        ok, mock_pip = _run_flavor_invariant(backend = backend, expected_env = "cu124")
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
    def test_hidden_cuda_devices_is_a_no_op_for_an_inferred_expectation(self, cvd):
        ok, mock_pip = _run_flavor_invariant(cvd = cvd, expected_env = None)
        assert ok is True
        mock_pip.assert_not_called()

    @pytest.mark.parametrize("cvd", ["", "-1", " ", "  -1 "])
    def test_a_mask_does_not_veto_an_explicit_expectation(self, cvd):
        # A reason not to CONCLUDE cu124 from a probe, not to ignore a stated one.
        for kwargs in (
            {"expected_env": "cu124"},
            {"expected_env": None, "index_family": "cu124"},
            {"expected_env": None, "recorded": "cu124"},
        ):
            ok, mock_pip = _run_flavor_invariant(cvd = cvd, repaired = "2.10.0+cu124", **kwargs)
            assert ok is True, kwargs
            assert mock_pip.call_count == 1, kwargs

    def test_an_explicit_visible_device_still_repairs(self):
        ok, mock_pip = _run_flavor_invariant(cvd = "0", repaired = "2.10.0+cu124")
        assert ok is True
        assert mock_pip.call_count == 1

    @pytest.mark.parametrize("tag", ["current", "custom", "simple", "cu"])
    def test_an_unenforceable_expectation_is_a_no_op(self, tag):
        # "xpu" and "rocm" are NOT here: both are published and both are enforced.
        ok, mock_pip = _run_flavor_invariant(expected_env = tag)
        assert ok is True
        mock_pip.assert_not_called()

    def test_an_empty_handover_tag_falls_through_rather_than_deciding(self):
        # PowerShell 7.5+ keeps an entry assigned "", which must read as "nobody said".
        ok, mock_pip = _run_flavor_invariant(
            expected_env = "",
            recorded = "cu128",
            repaired = "2.10.0+cu128",
        )
        assert ok is True
        assert _index_url(mock_pip).endswith("/cu128")

    def test_missing_or_unimportable_torch_is_a_no_op(self):
        # Reinstalling over an unimportable torch turns a driver fault into a wheel one.
        ok, mock_pip = _run_flavor_invariant(torch_rc = 1)
        assert ok is True
        mock_pip.assert_not_called()

    def test_an_unreadable_venv_is_a_no_op(self):
        ok, mock_pip = _run_flavor_invariant(probe_timeout = True, disk_label = "")
        assert ok is True
        mock_pip.assert_not_called()


class TestExpectedTorchFlavorResolution:
    def test_the_manifest_answers_when_the_environment_is_silent(self):
        # PowerShell 7.5+ keeps an entry assigned "", which must read as "nobody said".
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
        # Inventing an expectation here reinstalls CUDA torch onto a CPU-only box.
        ok, mock_pip = _run_flavor_invariant(expected_env = None, recorded = None, nvidia = False)
        assert ok is True
        mock_pip.assert_not_called()

    def test_a_cpu_pin_beats_the_live_probe(self):
        ok, mock_pip = _run_flavor_invariant(expected_env = None, recorded = None, index_family = "cpu")
        assert ok is True
        mock_pip.assert_not_called()

    def test_the_setup_scripts_index_url_is_used_when_its_leaf_matches(self):
        # Credentials are not reconstructible from a family leaf.
        _ok, mock_pip = _run_flavor_invariant(
            install_index_url = "https://mirror.local/whl/cu124?token=secret",
            repaired = "2.10.0+cu124",
        )
        assert _index_url(mock_pip) == "https://mirror.local/whl/cu124?token=secret"

    def test_an_index_url_naming_another_family_is_ignored(self):
        # setup.ps1 hands over /cpu alongside a "rocm" tag on the AMD Windows path.
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


class TestUnknownFamilyPinIsNotOverridden:
    """An explicit index pin whose leaf names no flavor is applied verbatim at install
    time, so this pass has no standing to second-guess it. The only expectation it could
    act on comes from the manifest, i.e. from whatever was installed BEFORE the pin was
    set, and repairing off that stale tag reinstalls from the public pytorch index --
    overriding a deliberate package source, and failing outright on an air-gapped host.
    _ensure_cuda_torch already declines on the same test."""

    def test_a_simple_mirror_pin_suppresses_the_manifest_fallback(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            expected_env = None,
            recorded = "cu124",
            index_url = "https://mirror.corp.example/simple",
        )
        assert ok is True
        mock_pip.assert_not_called()

    def test_a_readable_cuda_pin_still_repairs(self):
        """Narrowness: the escape must not disable a normal cu* mirror."""
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            repaired = "2.6.0+cu124",
            expected_env = None,
            recorded = "cu124",
            index_url = "https://mirror.corp.example/whl/cu124",
        )
        assert ok is True
        args = mock_pip.call_args[0]
        assert "https://mirror.corp.example/whl/cu124" in args


class TestExpectedXpuFlavorIsEnforced:
    """setup.ps1 publishes "xpu" for an Arc host and installs the XPU trio before handing
    over, so declining to act on that expectation would leave the invariant carrying an
    answer it refuses to use. The exposure is identical to the CUDA one: the dependency
    steps re-resolve torch from PyPI, and _ensure_xpu_torch cannot clean up afterwards
    because step 13's whole repair set is gated off Windows."""

    def test_an_xpu_venv_that_lost_torch_to_pypi_is_repaired(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu", repaired = "2.9.1+xpu", expected_env = "xpu"
        )
        assert ok is True
        args = mock_pip.call_args[0]
        # XPU floor is 2.6, not the CUDA trio's 2.4: unsloth raises at import below it.
        assert "torch>=2.6,<2.11.0" in args
        assert "torchvision>=0.21,<0.26.0" in args
        assert any("xpu" in str(a) for a in args)

    def test_a_healthy_xpu_venv_is_untouched(self):
        ok, mock_pip = _run_flavor_invariant(installed = "2.9.1+xpu", expected_env = "xpu")
        assert ok is True
        mock_pip.assert_not_called()

    def test_an_xpu_repair_that_does_not_take_fails_the_update(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu", repaired = None, expected_env = "xpu"
        )
        assert ok is False
        mock_pip.assert_called_once()

    def test_a_cuda_mask_does_not_cancel_an_xpu_repair(self):
        """CUDA_VISIBLE_DEVICES hides an NVIDIA GPU, not an Arc one."""
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            repaired = "2.9.1+xpu",
            expected_env = "xpu",
            cvd = "",
        )
        assert ok is True
        mock_pip.assert_called_once()

    def test_rocm_is_delegated_rather_than_rebuilt_here(self):
        """AMD's Windows wheels live on a per-architecture repo.amd.com index that a
        generic "rocm" tag cannot name, and setup.ps1 hands over an index URL that still
        points at /cpu on that path. _ensure_rocm_torch already detects the arch, maps
        it, and honours an explicit pin, so the repair is delegated to it rather than
        rebuilt from a guessed URL -- but it IS repaired, because it runs at step 2b,
        before the dependency steps that can put PyPI's CPU wheel here."""
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            repaired = "2.11.0+rocm7.2",
            expected_env = "rocm",
        )
        mock_pip.rocm_repair.assert_called_once()
        # Not called directly: this pass must not invent a repo.amd.com URL of its own.
        mock_pip.assert_not_called()
        assert ok is True

    def test_a_healthy_rocm_venv_does_not_call_the_repair(self):
        ok, mock_pip = _run_flavor_invariant(installed = "2.11.0+rocm7.2", expected_env = "rocm")
        mock_pip.rocm_repair.assert_not_called()
        mock_pip.assert_not_called()
        assert ok is True

    def test_a_rocm_repair_that_does_not_take_fails_the_update(self):
        """The state that used to be written as a successful CPU-only manifest."""
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu", repaired = None, expected_env = "rocm"
        )
        mock_pip.rocm_repair.assert_called_once()
        mock_pip.assert_not_called()
        assert ok is False


class TestTheDelegatedRocmRepairIsVerifiedByFamily:
    """_torch_build_is_gpu is family-blind, so it cannot judge a ROCm repair.

    A transient repo.amd.com failure is non-fatal inside _ensure_rocm_torch, and the
    cu124 wheel it leaves behind passes that check, so the update exited 0 and the
    manifest recorded "rocm" over an environment that never received it.
    """

    def test_a_repair_that_left_a_cuda_wheel_now_fails(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.6.0+cu124",
            repaired = "2.6.0+cu124",
            expected_env = "rocm",
            backend = "rocm",
        )
        assert ok is False
        mock_pip.rocm_repair.assert_called_once()

    def test_a_repair_that_took_still_passes(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.6.0+cu124",
            repaired = "2.11.0+rocm7.2",
            expected_env = "rocm",
            backend = "rocm",
        )
        assert ok is True
        mock_pip.rocm_repair.assert_called_once()

    def test_a_repair_that_left_cpu_torch_still_reads_as_cpu(self):
        ok, _mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            repaired = "2.11.0+cpu",
            expected_env = "rocm",
            backend = "rocm",
        )
        assert ok is False

    def test_an_unreadable_venv_does_not_fail_the_update_on_its_own(self):
        ok, _mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            repaired = "2.11.0+cpu",
            expected_env = "rocm",
            backend = "rocm",
            probe_timeout = True,
            disk_label = "",
        )
        assert ok is True


class TestAPinnedCpuIndexIsEnforcedToo:
    """UV_TORCH_BACKEND is honoured by the unpinned dependency steps.

    A venv deliberately built against /cpu can therefore come out of them holding a GPU
    wheel, and the update would record expected_torch_tag: cpu over it. Only a PIN
    counts: setup.ps1 also publishes "cpu" for a host whose nvidia-smi probe returned
    nothing, and acting on that would push a healthy cu124 venv down to CPU.
    """

    def test_a_gpu_wheel_under_a_pinned_cpu_index_is_repaired(self):
        pin = "https://download.pytorch.org/whl/cpu"
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.6.0+cu124",
            repaired = "2.11.0+cpu",
            expected_env = "cpu",
            index_url = pin,
            backend = "cpu",
        )
        assert ok is True
        assert _index_url(mock_pip) == pin
        assert "--force-reinstall" in [str(a) for a in mock_pip.call_args.args]

    def test_a_published_cpu_tag_with_no_pin_is_still_left_alone(self):
        # setup.ps1 publishes "cpu" when nvidia-smi answers nothing; the healthy cu124
        # venv underneath must not be downgraded.
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.6.0+cu124",
            expected_env = "cpu",
            nvidia = False,
        )
        assert ok is True
        mock_pip.assert_not_called()

    def test_a_cpu_venv_under_a_cpu_pin_is_a_no_op(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            expected_env = "cpu",
            index_url = "https://download.pytorch.org/whl/cpu",
            backend = "cpu",
        )
        assert ok is True
        mock_pip.assert_not_called()

    def test_a_cpu_repair_that_did_not_take_fails(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.6.0+cu124",
            repaired = "2.6.0+cu124",
            expected_env = "cpu",
            backend = "cpu",
            index_url = "https://download.pytorch.org/whl/cpu",
        )
        assert ok is False
        assert mock_pip.call_count == 1

    def test_a_cpu_backend_under_a_gpu_expectation_is_still_left_alone(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            expected_env = "cu124",
            backend = "cpu",
        )
        assert ok is True
        mock_pip.assert_not_called()


class TestWindowsOnArmKeepsTheNoTorchaudioException:
    """No win_arm64 torchaudio wheel is published on any index.

    setup.ps1 drops it from all four of its install trios ($WinArm64NoAudio), so asking
    for it here would turn a repairable venv into a failed install, and the venv the
    repair rebuilds could not have been installed in the first place.
    """

    def test_torchaudio_is_dropped_on_windows_arm64(self):
        ok, mock_pip = _run_flavor_invariant(repaired = "2.10.0+cu124", win_arm64 = True)
        assert ok is True
        args = [str(a) for a in mock_pip.call_args.args]
        assert not any(a.startswith("torchaudio") for a in args)
        assert any(a.startswith("torch>=") for a in args)
        assert any(a.startswith("torchvision") for a in args)
        assert _index_url(mock_pip).endswith("/cu124")

    def test_the_xpu_trio_drops_it_too(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            repaired = "2.9.1+xpu",
            expected_env = "xpu",
            win_arm64 = True,
        )
        assert ok is True
        args = [str(a) for a in mock_pip.call_args.args]
        assert not any(a.startswith("torchaudio") for a in args)
        assert "torch>=2.6,<2.11.0" in args

    def test_x64_windows_still_installs_all_three(self):
        _ok, mock_pip = _run_flavor_invariant(repaired = "2.10.0+cu124")
        args = [str(a) for a in mock_pip.call_args.args]
        assert any(a.startswith("torchaudio") for a in args)


class TestTheRequestedFamilyIsVerifiedAfterEveryRepair:
    """A GPU build is not the same answer as THE GPU build that was asked for.

    A misconfigured mirror can answer a /cu128 request with a cached cu124, rocm or xpu
    wheel. _torch_build_is_gpu is deliberately family-blind, so the update exited 0 and
    the manifest recorded the requested tag over a build that never arrived.
    """

    @pytest.mark.parametrize("landed", ["2.6.0+cu124", "2.11.0+rocm7.2", "2.9.1+xpu"])
    def test_a_wheel_from_the_wrong_family_fails(self, landed):
        ok, _mock_pip = _run_flavor_invariant(
            expected_env = "cu128",
            repaired = landed,
        )
        assert ok is False

    def test_the_requested_family_passes(self):
        ok, _mock_pip = _run_flavor_invariant(
            expected_env = "cu128",
            repaired = "2.6.0+cu128",
        )
        assert ok is True

    def test_an_xpu_repair_is_held_to_the_same_rule(self):
        ok, _mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            expected_env = "xpu",
            repaired = "2.6.0+cu124",
        )
        assert ok is False

    def test_an_unreadable_venv_still_passes(self):
        ok, _mock_pip = _run_flavor_invariant(
            expected_env = "cu128",
            repaired = "2.6.0+cu128",
            probe_timeout = True,
            disk_label = "",
        )
        assert ok is True

    def test_a_still_cpu_venv_keeps_its_own_warning(self, capsys):
        ok, _mock_pip = _run_flavor_invariant(expected_env = "cu124", repaired = None)
        assert ok is False
        assert "CPU-only" in capsys.readouterr().out


class TestAnUntaggedGpuWheelIsNotACpuMatch:
    """_torch_flavor_tag reads every untagged version as "cpu".

    Right for PyPI, wrong for a private index serving an untagged CUDA or ROCm build:
    under a /cpu pin that wheel compared equal to the expectation, skipped the repair,
    and was recorded as cpu. The runtime probe already carries the markers that tell
    them apart.
    """

    def test_an_untagged_cuda_wheel_under_a_cpu_pin_is_repaired(self):
        # A deliberate CPU backend must not be dragged up to cu124.
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.6.0",
            repaired = "2.11.0+cpu",
            expected_env = "cpu",
            backend = "cpu",
            index_url = "https://download.pytorch.org/whl/cpu",
            probe_cuda = "12.4",
        )
        assert ok is True
        assert mock_pip.call_count == 1

    def test_a_genuinely_untagged_cpu_wheel_is_still_a_match(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.6.0",
            expected_env = "cpu",
            backend = "cpu",
            index_url = "https://download.pytorch.org/whl/cpu",
        )
        assert ok is True
        mock_pip.assert_not_called()


class TestAnExplicitPinOutranksTheManifest:
    """Direct `python install_python_stack.py` on Windows, which the invariant supports.

    The manifest records what a PREVIOUS run installed; a pin is the instruction for
    THIS one. Resolving the manifest first let a freshly set cu128 pin lose to a stale
    cu124 record, and _expected_torch_index_url then rejected the cu128 pin as a family
    mismatch and repaired from the PUBLIC cu124 index -- undoing both the family and the
    source the user had just chosen.
    """

    def test_a_new_family_pin_beats_a_stale_manifest(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.10.0+cu124",
            repaired = "2.10.0+cu128",
            expected_env = None,
            recorded = "cu124",
            index_family = "cu128",
        )
        assert ok is True
        assert "cu128" in _index_url(mock_pip)
        assert "cu124" not in _index_url(mock_pip)
        assert "--force-reinstall" in [str(a) for a in mock_pip.call_args.args]

    def test_a_pinned_url_beats_a_stale_manifest_and_is_the_repair_source(self):
        pin = "https://mirror.corp.example/whl/cu128"
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.10.0+cu124",
            repaired = "2.10.0+cu128",
            expected_env = None,
            recorded = "cu124",
            index_url = pin,
        )
        assert ok is True
        assert _index_url(mock_pip) == pin

    def test_a_rocm_pin_collapses_to_the_flavor_vocabulary(self):
        # Every AMD leaf (rocm6.4, gfx1151) is "rocm" in the tag vocabulary.
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+rocm7.2",
            expected_env = None,
            recorded = "cu124",
            backend = "rocm",
            index_family = "gfx1151",
        )
        assert ok is True
        mock_pip.assert_not_called()

    def test_a_cpu_pin_beats_a_gpu_manifest(self):
        # A deliberate move to CPU must not be reverted by the previous install.
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            expected_env = None,
            recorded = "cu124",
            index_family = "cpu",
        )
        assert ok is True
        mock_pip.assert_not_called()

    def test_an_unrecognised_pin_still_falls_through_to_the_manifest(self):
        # A /simple mirror names no family; the caller's unknown-pin gate is the guard.
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            expected_env = None,
            recorded = "cu124",
            index_url = "https://mirror.corp.example/simple",
        )
        assert ok is True
        mock_pip.assert_not_called()

    def test_the_setup_handover_still_wins_over_a_pin(self):
        # The handover describes the run that just installed; the pin may predate it.
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.10.0+cu124",
            repaired = "2.10.0+cu126",
            expected_env = "cu126",
            index_family = "cu128",
        )
        assert ok is True
        assert "cu126" in _index_url(mock_pip)


class TestExplicitlyPinnedGpuFlavorsAreStillEnforced:
    """An explicit GPU pin sets _TORCH_BACKEND at import, and an XPU pin additionally
    reads as an "unknown family" to the shared helper, whose known set predates XPU.
    Between them those two gates skipped the invariant on exactly the hosts that asked
    for that GPU family on purpose -- so a later dependency install could put PyPI's CPU
    wheel there and the update would still report success."""

    def test_a_pinned_xpu_backend_is_enforced_when_it_agrees(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            repaired = "2.9.1+xpu",
            backend = "xpu",
            expected_env = "xpu",
        )
        assert ok is True
        mock_pip.assert_called_once()

    def test_a_pinned_rocm_backend_is_enforced_when_it_agrees(self):
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            repaired = "2.11.0+rocm7.2",
            backend = "rocm",
            expected_env = "rocm",
        )
        assert ok is True
        mock_pip.rocm_repair.assert_called_once()

    def test_an_xpu_index_pin_repairs_from_that_pin(self):
        """The pin's leaf IS the expected family, so it is not an unknown pin at all --
        and it is the right index to repair from."""
        pin = "https://mirror.corp.example/whl/xpu"
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            repaired = "2.9.1+xpu",
            backend = "xpu",
            expected_env = "xpu",
            index_url = pin,
        )
        assert ok is True
        assert pin in mock_pip.call_args[0]

    def test_an_unknown_pin_is_still_left_alone(self):
        """Regression guard: narrowing the veto must not reopen the case it closed."""
        ok, mock_pip = _run_flavor_invariant(
            installed = "2.11.0+cpu",
            expected_env = None,
            recorded = "cu124",
            index_url = "https://mirror.corp.example/simple",
        )
        assert ok is True
        mock_pip.assert_not_called()


class TestSetupPs1CudaOnDiskFallback:
    """A wedged NVIDIA driver hangs or faults `import torch` exactly as a faulted HIP
    runtime does. XPU and ROCm both rescue that venv from torch/version.py on disk; CUDA
    had no such arm, so the probe-failure chain fell through with a NULL tag, the no-wipe
    escape could not see a cu* wheel to preserve, and a direct update deleted a healthy
    CUDA environment before aborting."""

    def test_the_classifier_exists_and_keeps_the_family(self):
        assert "function Get-VenvTorchCudaTag" in _SETUP_SRC
        assert "function Test-VenvTorchIsCuda" in _SETUP_SRC
        block = _SETUP_SRC[_SETUP_SRC.index("function Get-VenvTorchCudaTag") :][:1400]
        # The family, not a flat "cuda": the stale comparison below it is cu126-vs-cu128.
        assert "cu[0-9]+" in block
        assert "site-packages\\torch\\version.py" in block

    def test_it_joins_the_probe_failure_chain(self):
        chain_start = _SETUP_SRC.index("elseif (Test-VenvTorchIsXpu -VenvPath $VenvDir)")
        chain = _SETUP_SRC[chain_start : _SETUP_SRC.index("$shouldRebuild = $true", chain_start)]
        assert "Test-VenvTorchIsCuda -VenvPath $VenvDir" in chain
        assert "$installedTorchTag = Get-VenvTorchCudaTag" in chain

    def test_it_does_not_disturb_the_callers_match_state(self):
        """A bare -match would clobber $Matches in the caller's scope; the stale-venv
        block reads $Matches[1] a few lines above."""
        block = _SETUP_SRC[_SETUP_SRC.index("function Get-VenvTorchCudaTag") :][:1400]
        assert "[regex]::Match(" in block


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


class TestThePackagesTiedToTheTorchReleaseAreResettled:
    """A repair that MOVES the torch release invalidates two compiled extensions.

    torchao's cpp extensions are torch-release-specific, and step 4 chose its pin from
    the torch this repair then replaced: 0.17.0 selected for a 2.11 that the <2.11.0
    repair spec takes down to 2.10, whose matched build is 0.16.0. Leaving the wrong one
    installed silently drops to the slow fallback.

    xFormers is stricter: its _C.pyd is linked against one exact (torch, CUDA) pair, and
    beside any other pair torch.ops.load_library raises, which xformers/_cpp_lib.py
    downgrades to a log line, so the import "succeeds" with memory-efficient attention,
    SwiGLU and the sparse ops silently gone. This script never installs xFormers, so
    removal is the only correct action, and it is what the backend's own resolver
    already concludes: torch SDPA beats an extension that cannot load.
    """

    def _resync(
        self,
        before,
        after,
        *,
        resident_xformers = None,
        installed_spec = False,
        torchao_install_fails = False,
    ):
        calls = {"torchao": [], "removed": []}

        def _try(*a, **k):
            # Returns a BOOL, as the real pip_install_try does. list.append() returns None, so a stub that only
            # appended reported failure on every call and every test here silently exercised the
            # could-not-install branch.
            calls["torchao"].append([str(x) for x in a])
            return not torchao_install_fails

        with (
            patch.object(stack_mod, "_probe_installed_torch_version", return_value = after),
            patch.object(
                stack_mod,
                "_exact_distribution_spec_is_installed",
                return_value = installed_spec,
            ),
            patch.object(
                stack_mod,
                "_resident_xformers_build_torch",
                return_value = resident_xformers,
            ),
            patch.object(
                stack_mod,
                "pip_install_try",
                side_effect = _try,
            ),
            patch.object(
                stack_mod,
                "_uninstall_distribution",
                side_effect = lambda name: calls["removed"].append(name) or True,
            ),
            patch.object(stack_mod, "_note", lambda *a, **k: None),
        ):
            calls["ok"] = stack_mod._resync_torch_coupled_packages(before)
        return calls

    def test_a_release_that_moved_re_pins_torchao(self):
        calls = self._resync("2.11.0+cu124", "2.10.0+cu124")
        assert calls["torchao"], "torchao must be re-selected for the new release"
        assert any("torchao==0.16.0" in " ".join(c) for c in calls["torchao"])

    def test_a_build_that_did_not_move_at_all_does_nothing(self):
        calls = self._resync("2.10.0+cu124", "2.10.0+cu124", resident_xformers = "2.11.0+cu128")
        assert calls["torchao"] == []
        assert calls["removed"] == []
        assert calls["ok"] is True

    def test_a_cuda_major_change_alone_re_pins_torchao(self):
        calls = self._resync("2.10.0+cu124", "2.10.0+cu130")
        assert calls["torchao"], "the CUDA major moved, so the pin has to be re-selected"
        assert any("torchao==0.17.0" in " ".join(c) for c in calls["torchao"])

    def test_a_flavor_change_within_one_cuda_major_leaves_torchao_alone(self):
        calls = self._resync("2.10.0+cu124", "2.10.0+cu128")
        assert calls["torchao"] == []

    def test_a_flavor_change_alone_still_rechecks_xformers(self):
        # xFormers links against the exact (torch, CUDA) pair.
        calls = self._resync("2.10.0+cu124", "2.10.0+cu128", resident_xformers = "2.10.0+cu124")
        assert calls["torchao"] == [], "the release did not move; torchao is fine"
        assert calls["removed"] == ["xformers"]
        assert calls["ok"] is True, "nothing here touched torch"

    def test_the_torchao_reinstall_cannot_drag_torch_back(self):
        calls = self._resync("2.11.0+cu124", "2.10.0+cu124")
        assert calls["torchao"], "the release moved, so torchao is re-pinned"
        assert all("--no-deps" in c for c in calls["torchao"])

    def test_a_pass_that_ran_a_torch_touching_install_asks_to_be_re_verified(self):
        # torchao depends on torch:
        calls = self._resync("2.11.0+cu124", "2.10.0+cu124")
        assert calls["ok"] is False

    def test_a_pass_that_installed_nothing_does_not(self):
        calls = self._resync("2.11.0+cu124", "2.10.0+cu124", installed_spec = True)
        assert calls["torchao"] == []
        assert calls["ok"] is True

    def test_a_torchao_already_matching_is_left_alone(self):
        calls = self._resync("2.11.0+cu124", "2.10.0+cu124", installed_spec = True)
        assert calls["torchao"] == []

    def test_a_mismatched_xformers_is_removed(self):
        calls = self._resync("2.11.0", "2.10.0+cu124", resident_xformers = "2.11.0+cu128")
        assert calls["removed"] == ["xformers"]

    def test_an_xformers_built_for_the_resident_torch_is_kept(self):
        calls = self._resync("2.11.0", "2.10.0+cu124", resident_xformers = "2.10.0+cu124")
        assert calls["removed"] == []

    def test_an_absent_xformers_is_not_touched(self):
        calls = self._resync("2.11.0+cu124", "2.10.0+cu124", resident_xformers = None)
        assert calls["removed"] == []

    def test_a_torchao_that_cannot_be_reinstalled_across_a_cuda_major_is_removed(self):
        """cu124 to cu130 with the torchao source unreachable.

        _select_torchao_spec exists because a torchao compiled for CUDA 12 cannot load its
        cpp extension under cu130, so leaving the resident one behind completes the update
        with a package that may fail on import. Same remedy the xFormers arm applies.
        """
        calls = self._resync("2.10.0+cu124", "2.10.0+cu130", torchao_install_fails = True)
        assert calls["torchao"], "the CUDA major moved, so the pin is re-selected"
        assert calls["removed"] == ["torchao"]

    def test_a_release_only_move_that_cannot_reinstall_keeps_torchao(self):
        """No CUDA-major change, so the resident build still loads: this one IS the slow path."""
        calls = self._resync("2.11.0+cu124", "2.10.0+cu124", torchao_install_fails = True)
        assert calls["torchao"], "the release moved, so the pin is re-selected"
        assert calls["removed"] == [], "a same-major torchao is slower, not broken"

    def test_a_torchao_that_reinstalls_cleanly_is_not_removed(self):
        # _select_torchao_spec branches on cuda>=13, so cu124 -> cu130 moves the build.
        calls = self._resync("2.10.0+cu124", "2.10.0+cu130")
        assert calls["torchao"], "the CUDA major moved"
        assert calls["removed"] == [], "the replacement landed; nothing to remove"

    def test_an_unreadable_torch_after_the_repair_does_nothing(self):
        calls = self._resync("2.11.0+cu124", None, resident_xformers = "2.11.0+cu128")
        assert calls["torchao"] == []
        assert calls["removed"] == []
        assert calls["ok"] is True

    def test_neither_half_can_fail_the_update(self, capsys):
        with (
            patch.object(stack_mod, "_probe_installed_torch_version", return_value = "2.10.0+cu124"),
            patch.object(
                stack_mod,
                "_select_torchao_spec",
                side_effect = RuntimeError("index down"),
            ),
            patch.object(
                stack_mod,
                "_resident_xformers_build_torch",
                side_effect = RuntimeError("unreadable"),
            ),
        ):
            assert stack_mod._resync_torch_coupled_packages("2.11.0+cu124") is True
        out = capsys.readouterr().out
        assert "could not re-match torchao" in out
        assert "could not re-check xFormers" in out


class TestTheResidentXformersBuildIsReadFromDisk:
    def test_the_recorded_torch_is_returned(self, tmp_path):
        pkg = tmp_path / "xformers"
        pkg.mkdir()
        (pkg / "cpp_lib.json").write_text(
            '{"version": {"torch": "2.10.0+cu128"}}', encoding = "utf-8"
        )
        with patch.object(
            stack_mod.importlib.util,
            "find_spec",
            return_value = SimpleNamespace(submodule_search_locations = [str(pkg)]),
        ):
            assert stack_mod._resident_xformers_build_torch() == "2.10.0+cu128"

    @pytest.mark.parametrize(
        "body", ['{"version": {}}', "{not json", '{"version": {"torch": "   "}}']
    )
    def test_an_unusable_record_reads_as_unknown(self, tmp_path, body):
        pkg = tmp_path / "xformers"
        pkg.mkdir()
        (pkg / "cpp_lib.json").write_text(body, encoding = "utf-8")
        with patch.object(
            stack_mod.importlib.util,
            "find_spec",
            return_value = SimpleNamespace(submodule_search_locations = [str(pkg)]),
        ):
            assert stack_mod._resident_xformers_build_torch() is None

    def test_an_absent_xformers_reads_as_unknown(self):
        with patch.object(stack_mod.importlib.util, "find_spec", return_value = None):
            assert stack_mod._resident_xformers_build_torch() is None

    def test_a_find_spec_that_raises_reads_as_unknown(self):
        with patch.object(stack_mod.importlib.util, "find_spec", side_effect = ValueError("boom")):
            assert stack_mod._resident_xformers_build_torch() is None


class TestTheResyncNoticesItsOwnFailures:
    """Both halves report failure by return value, not by raising.

    Ignoring that let the update write a completion manifest over an incompatible
    torchao, or over an xFormers whose removal was blocked, with nothing said.
    """

    def _resync_with(
        self,
        *,
        torchao_ok = True,
        uninstall_ok = True,
    ):
        with (
            patch.object(stack_mod, "_probe_installed_torch_version", return_value = "2.10.0+cu124"),
            patch.object(stack_mod, "_exact_distribution_spec_is_installed", return_value = False),
            patch.object(stack_mod, "_resident_xformers_build_torch", return_value = "2.11.0+cu128"),
            patch.object(stack_mod, "pip_install_try", return_value = torchao_ok),
            patch.object(stack_mod, "_uninstall_distribution", return_value = uninstall_ok),
            patch.object(stack_mod, "_note", lambda *a, **k: None),
        ):
            return stack_mod._resync_torch_coupled_packages("2.11.0+cu124")

    def test_a_torchao_install_that_did_not_take_is_reported(self, capsys):
        self._resync_with(torchao_ok = False)
        assert "could not install" in capsys.readouterr().out

    def test_an_xformers_removal_that_was_blocked_is_reported(self, capsys):
        self._resync_with(uninstall_ok = False)
        out = capsys.readouterr().out
        assert "could not remove the mismatched xFormers" in out

    def test_a_clean_pass_says_nothing(self, capsys):
        self._resync_with()
        out = capsys.readouterr().out
        assert "[WARN]" not in out

    def test_a_failure_still_does_not_fail_the_update(self):
        # The return value reports whether torch may have MOVED, not that all was well.
        assert self._resync_with(torchao_ok = False, uninstall_ok = False) is False


class TestThePostRepairCheckUsesTheSameRuleAsThePreRepairOne:
    def test_an_untagged_gpu_wheel_does_not_satisfy_a_cpu_expectation(self):
        # Untagged reads as cpu, so the post-repair check would accept the very wheel the pre-repair check rejected.
        with (
            patch.object(
                stack_mod,
                "_probe_torch_runtime",
                return_value = (True, True, "2.6.0", "", "12.4"),
            ),
        ):
            assert stack_mod._installed_flavor_tag_now("cpu") == "cuda"
            with patch.object(
                stack_mod,
                "_probe_torch_runtime",
                return_value = (True, True, "2.6.0", "6.4", ""),
            ):
                assert stack_mod._installed_flavor_tag_now("cpu") == "rocm"

    def test_a_genuine_untagged_cpu_wheel_still_reads_as_cpu(self):
        with patch.object(
            stack_mod,
            "_probe_torch_runtime",
            return_value = (True, True, "2.6.0", "", ""),
        ):
            assert stack_mod._installed_flavor_tag_now("cpu") == "cpu"

    def test_the_adjustment_is_scoped_to_a_cpu_expectation(self):
        with patch.object(
            stack_mod,
            "_probe_torch_runtime",
            return_value = (True, True, "2.6.0", "", "12.4"),
        ):
            assert stack_mod._installed_flavor_tag_now("cu124") == "cpu"
            assert stack_mod._installed_flavor_tag_now() == "cpu"

    def test_an_untagged_xpu_wheel_does_not_satisfy_a_cpu_expectation(self):
        """torch.version.xpu is where an untagged source, conda or private-index XPU
        build carries its runtime -- .hip and .cuda are both empty there. Reading only
        those two accepted the XPU wheel under a /cpu pin, returned success without
        replacing it, and then recorded a PINNED cpu flavor for a venv still holding it.
        """
        with (
            patch.object(
                stack_mod,
                "_probe_torch_runtime",
                return_value = (True, True, "2.9.0", "", ""),
            ),
            patch.object(stack_mod, "_TORCH_RUNTIME_XPU", "20250101"),
        ):
            assert stack_mod._installed_flavor_tag_now("cpu") == "xpu"

    def test_the_cpu_pin_repair_agrees_with_the_check_that_triggers_it(self):
        """_ensure_cpu_torch's own GPU-build predicate has to see the untagged XPU build
        too. Reading only the tag and .hip/.cuda made it return without reinstalling, so
        the post-repair check saw xpu again and failed the update instead of honouring
        the pin: the detection improved and the repair did not follow it."""
        source = inspect.getsource(stack_mod._ensure_cpu_torch)
        predicate = source[source.index("_is_gpu_build = ") :]
        predicate = predicate[: predicate.index("if not _is_gpu_build")]
        assert (
            "_TORCH_RUNTIME_XPU" in predicate
        ), "an untagged XPU wheel carries its runtime only in torch.version.xpu"
        for marker in ("_hip", "_cuda", "+xpu", "rocm"):
            assert marker in predicate, f"{marker} must still count"

    def test_the_gpu_family_reading_prefers_the_explicit_runtimes(self):
        # An XPU marker beside a CUDA or HIP one names the accelerator that wheel was
        # BUILT for; xpu is the answer only when it is the sole marker.
        assert stack_mod._gpu_family_from_runtime_markers("6.4", "12.4") == "rocm"
        assert stack_mod._gpu_family_from_runtime_markers("", "12.4") == "cuda"
        assert stack_mod._gpu_family_from_runtime_markers("", "") == "xpu"


class TestAFailedGpuPinIsNotADeliberateCpuChoice:
    """setup.ps1 falls back to the CPU index when a pinned ROCm or XPU install fails and
    publishes the resolved cpu tag, while the original GPU pin is still in the
    environment. Recording that as pinned makes _expected_cpu_flavor_was_chosen() read a
    failed install as an intentional one and suppress the repair guidance for good."""

    @staticmethod
    def _pinned(
        monkeypatch,
        flavor,
        *,
        url = "",
        family = "",
        backend = "",
        recorded = None,
    ):
        for var in (
            "UNSLOTH_TORCH_INDEX_URL",
            "UNSLOTH_TORCH_INDEX_FAMILY",
            "UNSLOTH_TORCH_INSTALL_INDEX_URL",
        ):
            monkeypatch.delenv(var, raising = False)
        if url:
            monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", url)
        if family:
            monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", family)
        monkeypatch.delenv("UNSLOTH_TORCH_BACKEND_SOURCE", raising = False)
        monkeypatch.setattr(stack_mod, "_TORCH_BACKEND", backend)
        monkeypatch.setattr(stack_mod, "_RECORDED_TORCH_TAG", (recorded or ("", False))[0])
        monkeypatch.setattr(stack_mod, "_RECORDED_TORCH_TAG_PINNED", (recorded or ("", False))[1])
        return stack_mod._expected_torch_flavor_was_pinned(flavor)

    def test_a_rocm_pin_that_settled_on_cpu_is_not_a_cpu_choice(self, monkeypatch):
        assert (
            self._pinned(monkeypatch, "cpu", url = "https://download.pytorch.org/whl/rocm6.4")
            is False
        )

    def test_an_xpu_family_that_settled_on_cpu_is_not_a_cpu_choice(self, monkeypatch):
        assert self._pinned(monkeypatch, "cpu", family = "xpu") is False

    def test_a_gpu_backend_that_settled_on_cpu_is_not_a_cpu_choice(self, monkeypatch):
        assert self._pinned(monkeypatch, "cpu", backend = "rocm") is False

    def test_a_carried_forward_gpu_record_does_not_pin_a_cpu_fallback(self, monkeypatch):
        assert self._pinned(monkeypatch, "cpu", recorded = ("cu124", True)) is False

    def test_a_cpu_pin_that_settled_on_cpu_still_counts(self, monkeypatch):
        assert self._pinned(monkeypatch, "cpu", url = "https://download.pytorch.org/whl/cpu") is True

    def test_a_rocm_pin_that_settled_on_rocm_still_counts(self, monkeypatch):
        assert (
            self._pinned(monkeypatch, "rocm", url = "https://download.pytorch.org/whl/rocm6.4")
            is True
        )

    def test_a_cuda_pin_counts_for_any_cuda_flavor(self, monkeypatch):
        # cu124 and cu128 are both the cuda family: the pin names the family, and the
        # recorded tag names the exact index within it.
        assert (
            self._pinned(monkeypatch, "cu128", url = "https://download.pytorch.org/whl/cu124") is True
        )

    def test_a_carried_forward_cpu_record_still_counts_for_cpu(self, monkeypatch):
        assert self._pinned(monkeypatch, "cpu", recorded = ("cpu", True)) is True

    def test_a_gpu_request_this_run_retires_the_old_cpu_record(self, monkeypatch):
        """The record can only speak for a run that said nothing to contradict it.

        A ROCm pin that settled on CPU is the failed-pin case the arms above refuse to call
        deliberate, and reviving the old CPU provenance underneath them re-records the venv
        as pinned CPU anyway: the mismatch is then suppressed for good once the requested
        GPU works.
        """
        assert (
            self._pinned(
                monkeypatch,
                "cpu",
                url = "https://download.pytorch.org/whl/rocm6.4",
                recorded = ("cpu", True),
            )
            is False
        )
        assert self._pinned(monkeypatch, "cpu", backend = "cuda", recorded = ("cpu", True)) is False

    def test_a_derived_backend_does_not_retire_the_old_cpu_record(self, monkeypatch):
        # install.sh marks the backend it resolved, and "cpu" on a GPU-less machine is not a
        # preference either way, so it contradicts nothing the previous run recorded.
        monkeypatch.setenv("UNSLOTH_TORCH_BACKEND_SOURCE", "resolved")
        monkeypatch.setattr(stack_mod, "_TORCH_BACKEND", "cuda")
        monkeypatch.setattr(stack_mod, "_RECORDED_TORCH_TAG", "cpu")
        monkeypatch.setattr(stack_mod, "_RECORDED_TORCH_TAG_PINNED", True)
        for var in ("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY"):
            monkeypatch.delenv(var, raising = False)
        assert stack_mod._expected_torch_flavor_was_pinned("cpu") is True

    def test_an_authoritative_url_silences_a_stale_family(self, monkeypatch):
        """install.sh returns on the URL and never reads the family, so a family that
        disagrees is dead. An unknown-family corporate /simple URL names no flavor, and
        falling through to a stale ..._FAMILY=cpu recorded the CPU wheel a GPU-less host
        legitimately got as DELIBERATE. A later eGPU there gets no mismatch and no repair.
        """
        assert (
            self._pinned(
                monkeypatch,
                "cpu",
                url = "https://mirror.corp.invalid/simple",
                family = "cpu",
            )
            is False
        )

    def test_the_family_still_answers_when_no_url_was_supplied(self, monkeypatch):
        assert self._pinned(monkeypatch, "cpu", family = "cpu") is True

    def test_a_url_whose_leaf_does_name_the_flavor_still_counts(self, monkeypatch):
        assert (
            self._pinned(
                monkeypatch,
                "cpu",
                url = "https://download.pytorch.org/whl/cpu",
                family = "rocm6.4",
            )
            is True
        )

    def test_the_provenance_check_reads_the_family_only_through_the_shared_resolver(self):
        """One read of the pair, so the precedence cannot be bypassed by a second one."""
        body = inspect.getsource(stack_mod._expected_torch_flavor_was_pinned)
        assert "UNSLOTH_TORCH_INDEX_FAMILY" not in body, (
            "the family has to come through _explicit_torch_index_url(), which applies "
            "install.sh's precedence; a direct read here reintroduces the bug"
        )
        assert "_explicit_torch_index_url()" in body

    def test_asking_without_a_flavor_answers_as_it_always_did(self, monkeypatch):
        assert self._pinned(monkeypatch, "", url = "https://download.pytorch.org/whl/rocm6.4") is True


class TestTheWindowsXpuTritonSwapReachesADirectRun:
    """setup.ps1 performs the swap after this script exits; a direct run has no such
    postlude, and the core install pulls triton-windows over torch's XPU triton."""

    def test_the_handover_variable_gates_it(self):
        source = inspect.getsource(stack_mod._ensure_xpu_triton)
        assert "if NO_TORCH or IS_MACOS:" in source, "Windows must no longer be excluded outright"
        assert (
            'IS_WINDOWS and os.environ.get("UNSLOTH_EXPECTED_TORCH_TAG"' in source
        ), "under setup.ps1 the swap is still that script's job"

    def test_the_windows_branch_runs_it_after_the_invariant(self):
        source = inspect.getsource(stack_mod.install_python_stack)
        marker = "if IS_WINDOWS and not NO_TORCH:"
        assert marker in source
        block = source[source.index(marker) :][:1200]
        assert "_ensure_expected_torch_flavor" in block
        assert "_ensure_xpu_triton()" in block
        # After, for the reason step 13 puts it last: the swap keys off the +xpu label.
        assert block.index("_ensure_expected_torch_flavor") < block.index("_ensure_xpu_triton()")


class TestTheDelegatedRocmRepairKeepsTheArm64Exception:
    """_ensure_rocm_torch's Windows branch asked for the full trio unconditionally.

    No win_arm64 torchaudio wheel exists, so the whole trio is unresolvable there. On
    the delegated path that failure is nonfatal, so the CPU build the repair was meant
    to replace stays put and the family verification then fails the update -- a worse
    outcome than the plain trio case, where the failure is at least immediate.
    """

    def test_the_windows_rocm_install_drops_torchaudio_on_arm64(self):
        source = inspect.getsource(stack_mod._ensure_rocm_torch)
        block = source[source.index("_WINDOWS_ROCM_TORCH_PKG_SPECS.get") :][:1200]
        assert (
            "_is_windows_arm64()" in block
        ), "the delegated ROCm repair needs the same exception as the flavor repair"
        assert "*_rocm_trio" in block, "the trio has to be built, not passed positionally"

    def test_x64_windows_still_asks_for_all_three(self):
        source = inspect.getsource(stack_mod._ensure_rocm_torch)
        block = source[source.index("_WINDOWS_ROCM_TORCH_PKG_SPECS.get") :][:1200]
        assert "_rocm_trio = [_torch_pkg, _vision_pkg, _audio_pkg]" in block


class TestADefinitiveImportFailureIsNotADriverHang:
    """setup.ps1's disk-label rescue treated both the same.

    A wedged driver and a truncated torch both leave a +cu* version.py behind. Keeping
    the venv is right in both cases -- deleting it does not fix a driver, which is the
    whole point of the rescue (#8335, #7275) -- but only the first means the
    installation is sound, and the family-matched install below runs with bare
    requirements and no reinstall flag, so the second could write a completion manifest
    over a torch that still cannot import.
    """

    _SOURCE = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")

    def test_the_probe_distinguishes_a_timeout_from_an_answer(self):
        assert "TimedOut = $false" in self._SOURCE
        assert "$result.TimedOut = $true" in self._SOURCE

    def test_the_cuda_rescue_forces_a_reinstall_on_a_definitive_failure(self):
        block = self._SOURCE[self._SOURCE.index("Test-VenvTorchIsCuda -VenvPath $VenvDir") :][:2000]
        assert "$_verProbe -and -not $_verProbe.TimedOut" in block
        assert "$script:TorchImportDefinitivelyFailed = $true" in block

    def test_the_venv_is_still_kept_either_way(self):
        # A faulted driver raises at DLL load rather than timing out.
        start = self._SOURCE.index("Test-VenvTorchIsCuda -VenvPath $VenvDir")
        # The CUDA arm only: the trailing "no family matched" else SHOULD rebuild.
        arm = self._SOURCE[start : self._SOURCE.index("} else {", start)]
        code = "\n".join(line for line in arm.splitlines() if not line.strip().startswith("#"))
        assert "$shouldRebuild" not in code
        assert "$script:TorchImportDefinitivelyFailed = $true" in arm

    def test_the_flag_reaches_the_cuda_install(self):
        assert (
            "if ($script:PinChangedForceReinstall -or $script:TorchImportDefinitivelyFailed) {"
            in self._SOURCE
        )
        assert "$script:TorchImportDefinitivelyFailed = $false" in self._SOURCE


class TestARepairedTorchThatCannotImport:
    """The verification after a repair, when the new wheel does not load.

    _probe_torch_runtime answers (ran, importable, ...). A torch that RAN and did not
    import is a definitive answer, not the ambiguity the on-disk fallback exists for:
    version.py still reports the requested +cu*/+xpu tag from a half-written or DLL-less
    wheel, so reading it would accept the repair and write a completion manifest over a
    torch nothing can import. Only a probe that could not run at all (a hung driver) may
    fall back to disk.
    """

    def _probe(self, monkeypatch, *, ran, importable, version):
        monkeypatch.setattr(
            stack_mod,
            "_probe_torch_runtime",
            lambda: (ran, importable, version, None, "12.4" if "+cu" in (version or "") else None),
        )
        monkeypatch.setattr(stack_mod, "_installed_torch_label_on_disk", lambda: version or "")

    def test_a_definitive_import_failure_is_not_read_off_disk(self, monkeypatch):
        self._probe(monkeypatch, ran = True, importable = False, version = "2.6.0+cu124")
        assert stack_mod._installed_flavor_tag_now("cu124") == stack_mod._TORCH_TAG_UNIMPORTABLE

    def test_a_probe_that_could_not_run_still_falls_back_to_disk(self, monkeypatch):
        # The wedged-driver host this fallback exists for.
        self._probe(monkeypatch, ran = False, importable = False, version = "2.6.0+cu124")
        assert stack_mod._installed_flavor_tag_now("cu124") == "cu124"

    def test_a_healthy_probe_is_unaffected(self, monkeypatch):
        self._probe(monkeypatch, ran = True, importable = True, version = "2.6.0+cu124")
        assert stack_mod._installed_flavor_tag_now("cu124") == "cu124"

    def test_the_sentinel_cannot_be_mistaken_for_a_flavor(self):
        sentinel = stack_mod._TORCH_TAG_UNIMPORTABLE
        for tag in ("cpu", "cu124", "cu128", "rocm", "xpu", ""):
            assert sentinel != tag
        assert sentinel, "falsy would be swallowed by the ambiguity branch"

    def test_the_update_fails_rather_than_reporting_success(self, monkeypatch):
        """End to end: a cu124 repair whose wheel will not import must not return True."""
        lines = []
        monkeypatch.setattr(stack_mod, "IS_WINDOWS", True)
        monkeypatch.setattr(stack_mod, "IS_MACOS", False)
        monkeypatch.setattr(stack_mod, "NO_TORCH", False)
        monkeypatch.setattr(stack_mod, "_TORCH_BACKEND", "")
        monkeypatch.setattr(stack_mod, "_RECORDED_TORCH_TAG", None)
        monkeypatch.setattr(
            stack_mod, "_safe_print", lambda *a, **k: lines.append(" ".join(map(str, a)))
        )
        monkeypatch.setattr(stack_mod, "_has_usable_nvidia_gpu", lambda: True)
        monkeypatch.setenv("UNSLOTH_EXPECTED_TORCH_TAG", "cu124")
        monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
        monkeypatch.delenv("UNSLOTH_TORCH_INSTALL_INDEX_URL", raising = False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)

        state = {"ran": True, "importable": True, "version": "2.11.0+cpu"}
        monkeypatch.setattr(
            stack_mod,
            "_probe_torch_runtime",
            lambda: (state["ran"], state["importable"], state["version"], None, None),
        )
        monkeypatch.setattr(stack_mod, "_installed_torch_label_on_disk", lambda: state["version"])

        def _pip(_label, *_a, **_k):
            state.update(ran = True, importable = False, version = "2.6.0+cu124")

        monkeypatch.setattr(stack_mod, "pip_install", _pip)
        assert stack_mod._ensure_expected_torch_flavor() is False
        assert any("cannot be imported" in ln for ln in lines), lines


# What a localized nvidia-smi writes, which -X utf8 decodes as UTF-8 (#10173).
_LOCALIZED_NVIDIA_SMI = (
    "import sys\n"
    "if sys.argv[1:] == ['--query-gpu=compute_cap', '--format=csv,noheader,nounits']:\n"
    "    sys.stdout.buffer.write(b'8.6\\n')\n"
    "else:\n"
    "    sys.stdout.buffer.write(b'| NVIDIA-SMI 591.86    CUDA Version: 13.1 |\\n')\n"
    "    sys.stdout.buffer.write('\\u4e02\\u4fdd\\u7559\\u6240\\u6709\\u6743\\u5229\\u3002\\n'.encode('gbk'))\n"
)


def test_detect_index_url_reads_a_localized_nvidia_smi_banner(monkeypatch, tmp_path):
    fake = tmp_path / "nvidia-smi.py"
    fake.write_text(_LOCALIZED_NVIDIA_SMI, encoding = "utf-8")
    real_run = subprocess.run

    def run_fake_nvidia_smi(command, *args, **kwargs):
        if command and command[0] == "nvidia-smi":
            command = [sys.executable, str(fake), *command[1:]]
        kwargs.setdefault("encoding", "utf-8")  # what the launcher's -X utf8 does
        return real_run(command, *args, **kwargs)

    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    monkeypatch.setattr(stack_mod.subprocess, "run", run_fake_nvidia_smi)
    monkeypatch.setattr(
        stack_mod.shutil,
        "which",
        lambda name, *a, **k: "nvidia-smi" if name == "nvidia-smi" else None,
    )
    assert _detect_cuda_torch_index_url() == f"{stack_mod._PYTORCH_WHL_BASE}/cu130"
