"""_ensure_cuda_torch reinstalls CUDA torch when an NVIDIA-host venv carries a ROCm
build (the pre-fix KFD gpu_id false positive), but leaves healthy CUDA / CPU / ROCm /
macOS / Windows untouched. Fully mocked -- no GPU required."""

import importlib.util
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

_ensure_cuda_torch = stack_mod._ensure_cuda_torch
_detect_cuda_torch_index_url = stack_mod._detect_cuda_torch_index_url


def _make_run(
    torch_state = "hip",
    cuda_version = "12.8",
    torch_rc = 0,
    smi_rc = 0,
):
    """subprocess.run side_effect: torch-classify probe (sys.executable, bytes
    stdout) vs nvidia-smi version probe (smi path, text=True), keyed on the executable."""

    def _run(cmd, *args, **kwargs):
        result = MagicMock()
        exe = str(cmd[0]) if cmd else ""
        if exe == sys.executable:
            result.returncode = torch_rc
            result.stdout = (torch_state + "\n").encode()
            return result
        # nvidia-smi version probe (text = True)
        result.returncode = smi_rc
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
    index_url sets UNSLOTH_TORCH_INDEX_URL (the full-URL pin form)."""
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

    with (
        patch.object(stack_mod, "_TORCH_BACKEND", backend),
        patch.object(stack_mod, "IS_MACOS", is_macos),
        patch.object(stack_mod, "IS_WINDOWS", is_windows),
        patch.object(stack_mod, "NO_TORCH", no_torch),
        patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = nvidia),
        patch.object(stack_mod.shutil, "which", side_effect = _which),
        patch.object(stack_mod.os.path, "isfile", return_value = bool(smi_path)),
        patch.object(stack_mod, "pip_install") as mock_pip,
        patch.object(
            stack_mod.subprocess,
            "run",
            side_effect = _make_run(torch_state, cuda_version, torch_rc, smi_rc),
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
