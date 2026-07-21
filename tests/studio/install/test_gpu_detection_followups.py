"""GPU-detection follow-ups to PR 6174: NVIDIA precedence + /proc/driver/nvidia/gpus fallback ported to install_llama_prebuilt.py and setup.sh. Mocks/source-level only, no GPU."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]

# Load studio/install_llama_prebuilt.py the same way the sibling suite does.
_MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
_SPEC = importlib.util.spec_from_file_location(
    "studio_install_llama_prebuilt_followups", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
prebuilt_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = prebuilt_mod
_SPEC.loader.exec_module(prebuilt_mod)

detect_host = prebuilt_mod.detect_host
_apply_host_overrides = prebuilt_mod._apply_host_overrides

SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"


def _make_run_capture(rocminfo_stdout: str = ""):
    """Fake run_capture: rocminfo returns rocminfo_stdout, everything else empty."""

    def _run_capture(cmd, *args, **kwargs):
        exe = str(cmd[0]) if cmd else ""
        result = MagicMock()
        if exe.endswith("rocminfo"):
            result.returncode = 0
            result.stdout = rocminfo_stdout
        else:
            result.returncode = 1
            result.stdout = ""
        result.stderr = ""
        return result

    return _run_capture


def _run_detect_host(
    *,
    machine: str = "x86_64",
    system: str = "Linux",
    which_map: dict | None = None,
    proc_dir_entries: list | None = None,
    rocminfo_stdout: str = "",
    env: dict | None = None,
):
    """Drive detect_host() against a fully synthetic host."""
    which_map = which_map or {}
    proc_dir_entries = proc_dir_entries if proc_dir_entries is not None else []

    real_isdir = prebuilt_mod.os.path.isdir
    real_listdir = prebuilt_mod.os.listdir
    proc_path = "/proc/driver/nvidia/gpus"

    def fake_isdir(p):
        if str(p) == proc_path:
            return bool(proc_dir_entries)
        return real_isdir(p)

    def fake_listdir(p):
        if str(p) == proc_path:
            if not proc_dir_entries:
                raise OSError("no such dir")
            return list(proc_dir_entries)
        return real_listdir(p)

    patches = [
        patch.object(prebuilt_mod.platform, "system", return_value = system),
        patch.object(prebuilt_mod.platform, "machine", return_value = machine),
        patch.object(
            prebuilt_mod.platform, "mac_ver", return_value = ("", ("", "", ""), "")
        ),
        patch.object(
            prebuilt_mod.shutil, "which", side_effect = lambda n: which_map.get(n)
        ),
        patch.object(
            prebuilt_mod, "run_capture", side_effect = _make_run_capture(rocminfo_stdout)
        ),
        patch.object(prebuilt_mod.os.path, "isdir", side_effect = fake_isdir),
        patch.object(prebuilt_mod.os, "listdir", side_effect = fake_listdir),
        patch.object(prebuilt_mod.os, "access", return_value = False),
        patch.dict(prebuilt_mod.os.environ, env or {}, clear = False),
    ]
    for p in patches:
        p.start()
    try:
        # Don't let the host's CUDA_VISIBLE_DEVICES leak in unless the scenario sets it.
        if env is None or "CUDA_VISIBLE_DEVICES" not in env:
            prebuilt_mod.os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return detect_host()
    finally:
        for p in patches:
            p.stop()


# ── install_llama_prebuilt.detect_host(): /proc NVIDIA fallback ──────────────


class TestDetectHostProcFallback:
    def test_proc_fallback_marks_physical_nvidia_when_smi_absent(self):
        """No nvidia-smi, but /proc/driver/nvidia/gpus is populated -> NVIDIA."""
        host = _run_detect_host(
            which_map = {},  # nvidia-smi resolves to None
            proc_dir_entries = ["0000:01:00.0"],
        )
        assert host.has_physical_nvidia is True

    def test_proc_fallback_has_usable_nvidia_when_devices_visible(self):
        """Default CUDA_VISIBLE_DEVICES (unset) -> visible tokens non-empty -> usable."""
        host = _run_detect_host(
            which_map = {},
            proc_dir_entries = ["0000:01:00.0"],
        )
        assert host.has_usable_nvidia is True

    def test_proc_fallback_not_usable_when_devices_hidden(self):
        """CUDA_VISIBLE_DEVICES='' hides all GPUs -> physical yes, usable no."""
        host = _run_detect_host(
            which_map = {},
            proc_dir_entries = ["0000:01:00.0"],
            env = {"CUDA_VISIBLE_DEVICES": ""},
        )
        assert host.has_physical_nvidia is True
        assert host.has_usable_nvidia is False

    def test_empty_proc_dir_does_not_mark_nvidia(self):
        """A driver dir that exists but is empty must not assert a GPU."""
        host = _run_detect_host(which_map = {}, proc_dir_entries = [])
        assert host.has_physical_nvidia is False

    def test_proc_fallback_is_linux_only(self):
        """The /proc fallback must not run on Windows (path is Linux-only)."""
        host = _run_detect_host(
            system = "Windows",
            machine = "amd64",
            which_map = {},
            proc_dir_entries = ["0000:01:00.0"],
        )
        assert host.has_physical_nvidia is False


# ── install_llama_prebuilt.detect_host(): NVIDIA precedence over ROCm ────────


class TestDetectHostNvidiaPrecedence:
    def test_rocm_probe_skipped_when_proc_nvidia_present(self):
        """rocminfo reports gfx1100, but a proc-detected NVIDIA GPU wins."""
        host = _run_detect_host(
            which_map = {"rocminfo": "/usr/bin/rocminfo"},
            proc_dir_entries = ["0000:01:00.0"],
            rocminfo_stdout = "  Name:                    gfx1100\n",
        )
        assert host.has_usable_nvidia is True
        assert host.has_rocm is False

    def test_rocm_detected_when_no_nvidia(self):
        """With no NVIDIA signal at all, rocminfo gfx1100 -> has_rocm True."""
        host = _run_detect_host(
            which_map = {"rocminfo": "/usr/bin/rocminfo"},
            proc_dir_entries = [],
            rocminfo_stdout = "  Name:                    gfx1100\n",
        )
        assert host.has_usable_nvidia is False
        assert host.has_rocm is True


# ── _apply_host_overrides: forwarded --rocm-gfx / --has-rocm still win ───────


class TestOverridesStillWin:
    def test_forwarded_gfx_forces_rocm_on_non_nvidia_host(self):
        host = _run_detect_host(which_map = {}, proc_dir_entries = [])
        assert host.has_rocm is False
        overridden = _apply_host_overrides(host, override_rocm_gfx = "gfx1100")
        assert overridden.has_rocm is True
        assert overridden.rocm_gfx_target == "gfx1100"

    def test_override_has_rocm_forces_rocm(self):
        host = _run_detect_host(which_map = {}, proc_dir_entries = [])
        overridden = _apply_host_overrides(host, override_has_rocm = True)
        assert overridden.has_rocm is True

    def test_force_cpu_drops_nvidia_attributes(self):
        host = _run_detect_host(which_map = {}, proc_dir_entries = ["0000:01:00.0"])
        assert host.has_usable_nvidia is True
        overridden = _apply_host_overrides(host, force_cpu = True)
        assert overridden.has_usable_nvidia is False
        assert overridden.has_physical_nvidia is False
        assert overridden.has_rocm is False


# ── setup.sh source-level guarantees ────────────────────────────────────────


class TestSetupShHardening:
    @pytest.fixture(scope = "class")
    def setup_src(self) -> str:
        return SETUP_SH.read_text(encoding = "utf-8")

    def test_has_usable_nvidia_helper_exists(self, setup_src):
        assert "_setup_has_usable_nvidia_gpu()" in setup_src

    def test_helper_uses_proc_fallback(self, setup_src):
        start = setup_src.find("_setup_has_usable_nvidia_gpu()")
        end = setup_src.find("\n}", start)
        body = setup_src[start:end]
        assert (
            "/proc/driver/nvidia/gpus" in body
        ), "_setup_has_usable_nvidia_gpu must fall back to /proc/driver/nvidia/gpus"

    def test_gpu_summary_uses_helper(self, setup_src):
        assert "if _setup_has_usable_nvidia_gpu; then" in setup_src

    def test_timeout_wrapper_exists(self, setup_src):
        start = setup_src.find("_setup_run_smi()")
        assert start >= 0, "_setup_run_smi timeout wrapper must exist"
        end = setup_src.find("\n}", start)
        body = setup_src[start:end]
        assert "timeout 10" in body
        assert "command -v timeout" in body

    def test_cuda_source_build_gated_on_usable_nvidia(self, setup_src):
        """The nvcc source-build search must be gated on _setup_nvidia_usable."""
        anchor = setup_src.find('NVCC_PATH=""\n')
        assert anchor >= 0
        window = setup_src[anchor : anchor + 700]
        assert (
            'if [ "$_setup_nvidia_usable" = true ]' in window
        ), "CUDA toolkit search must require a usable NVIDIA GPU, not just nvcc"

    def test_nvidia_helper_honours_hidden_cvd(self, setup_src):
        """_setup_has_usable_nvidia_gpu must consult the hidden-CVD helper so CVD ""/-1 suppresses NVIDIA before AMD gating."""
        assert "_setup_cvd_hides_nvidia()" in setup_src
        start = setup_src.find("_setup_has_usable_nvidia_gpu() {")
        end = setup_src.find("\n}", start)
        body = setup_src[start:end]
        assert "_setup_cvd_hides_nvidia" in body

    def test_rocm_source_build_gated_on_amd_detected(self, setup_src):
        """The hipcc source-build search must be gated on _setup_amd_detected."""
        anchor = setup_src.find('ROCM_HIPCC=""')
        assert anchor >= 0
        window = setup_src[anchor : anchor + 400]
        assert (
            '[ "$_setup_amd_detected" = true ]' in window
        ), "ROCm toolkit search must require a detected AMD GPU, not just hipcc"

    def test_compute_cap_probe_timeout_wrapped(self, setup_src):
        # nvidia-smi is now a variable ($_smi_bin), so check the wrapper precedes
        # the probe rather than matching a literal. The string also appears in a
        # comment, so scan all occurrences and accept if any is wrapped.
        wrapped = False
        start = 0
        while True:
            idx = setup_src.find("--query-gpu=compute_cap", start)
            if idx < 0:
                break
            if "_setup_run_smi" in setup_src[max(0, idx - 80) : idx]:
                wrapped = True
                break
            start = idx + 1
        assert (
            wrapped
        ), "compute_cap probe must be wrapped in _setup_run_smi (timeout-bounded)"

    def test_driver_version_probe_timeout_wrapped(self, setup_src):
        start = setup_src.find("_cuda_driver_max_version()")
        end = setup_src.find("\n}", start)
        body = setup_src[start:end]
        assert "_setup_run_smi nvidia-smi" in body


# TEST: install.sh -- UNSLOTH_TORCH_BACKEND classified on the final path segment


class TestBackendExportLeafClassification:
    """A mirror base path containing "rocm"/"gfx" must not mislabel a cu*/cpu index; classification uses TORCH_INDEX_URL's leaf only."""

    @pytest.fixture(scope = "class")
    def install_src(self) -> str:
        return (PACKAGE_ROOT / "install.sh").read_text(encoding = "utf-8")

    def test_export_block_uses_leaf(self, install_src):
        anchor = install_src.find("_torch_index_leaf=")
        assert anchor >= 0, "backend export must classify on the final path segment"
        # Window spans the leaf-normalization prelude (query/frag drop + all-slash trim loop)
        # through the export case arms.
        window = install_src[anchor : anchor + 900]
        assert 'export UNSLOTH_TORCH_BACKEND="rocm"' in window
        assert 'export UNSLOTH_TORCH_BACKEND="cpu"' in window
        assert 'export UNSLOTH_TORCH_BACKEND="cuda"' in window

    def test_leaf_classification_behaviour(self, tmp_path):
        import subprocess as sp

        script = tmp_path / "leaf.sh"
        src = (PACKAGE_ROOT / "install.sh").read_text(encoding = "utf-8")
        anchor = src.find("_torch_index_leaf=")
        block = src[anchor : src.find("esac", anchor) + 4]
        # Drive the extracted block with adversarial mirror URLs.
        script.write_text(
            "#!/bin/sh\n"
            'TORCH_INDEX_URL="$1"\n' + block + "\n"
            'printf "%s" "$UNSLOTH_TORCH_BACKEND"\n'
        )
        cases = {
            "https://download.pytorch.org/whl/cu128": "cuda",
            "https://download.pytorch.org/whl/cpu": "cpu",
            "https://download.pytorch.org/whl/rocm6.4": "rocm",
            "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/": "rocm",
            "https://repo.amd.com/rocm/whl/gfx1151/": "rocm",
            "https://mirror.local/rocm-cache/cu128": "cuda",
            "https://mirror.local/gfx-cache/cpu": "cpu",
        }
        for url, expected in cases.items():
            out = sp.run(
                ["sh", str(script), url], capture_output = True, text = True, timeout = 30
            ).stdout.strip()
            assert (
                out == expected
            ), f"{url} classified as {out!r}, expected {expected!r}"


# TEST: CUDA_VISIBLE_DEVICES=""/-1 hides NVIDIA in every usable-GPU helper


_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_STACK_SPEC = importlib.util.spec_from_file_location(
    "studio_install_python_stack_followups", _STACK_PATH
)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
stack_mod = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = stack_mod
_STACK_SPEC.loader.exec_module(stack_mod)


def _stack_nvidia_usable(cvd):
    """Drive _has_usable_nvidia_gpu with a mocked nvidia-smi that always reports a GPU; cvd=None unsets the env var."""

    def fake_run(cmd, *args, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "GPU 0: NVIDIA Fake (UUID: GPU-x)\n"
        return result

    env = {} if cvd is None else {"CUDA_VISIBLE_DEVICES": cvd}
    with (
        patch.object(
            stack_mod.shutil,
            "which",
            side_effect = lambda n: "/usr/bin/nvidia-smi" if n == "nvidia-smi" else None,
        ),
        patch.object(stack_mod.subprocess, "run", side_effect = fake_run),
        patch.dict(stack_mod.os.environ, env, clear = False),
    ):
        if cvd is None:
            stack_mod.os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return stack_mod._has_usable_nvidia_gpu()


class TestHiddenCvdNotUsable:
    """CVD ""/-1 hides every NVIDIA device; all three _has_usable_nvidia_gpu impls must report not-usable so AMD/CPU routes run."""

    def test_python_unset_cvd_is_usable(self):
        assert _stack_nvidia_usable(None) is True

    def test_python_empty_cvd_not_usable(self):
        assert _stack_nvidia_usable("") is False

    def test_python_minus_one_not_usable(self):
        assert _stack_nvidia_usable("-1") is False

    def test_python_padded_minus_one_not_usable(self):
        assert _stack_nvidia_usable(" -1 ") is False

    def test_python_explicit_device_is_usable(self):
        assert _stack_nvidia_usable("0") is True

    def test_python_device_list_is_usable(self):
        assert _stack_nvidia_usable("0,1") is True

    def test_hidden_nvidia_restores_rocm_detection(self):
        """Mixed host, NVIDIA hidden via CVD=-1: _has_rocm_gpu must pass the NVIDIA guard and return True (pre-fix it ignored CVD)."""

        def fake_run(cmd, *args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            exe = str(cmd[0])
            if exe.endswith("rocminfo"):
                result.stdout = "  Name:                    gfx1100\n"
            else:
                result.stdout = "GPU 0: NVIDIA Fake (UUID: GPU-x)\n"
            return result

        which_map = {
            "rocminfo": "/usr/bin/rocminfo",
            "nvidia-smi": "/usr/bin/nvidia-smi",
        }
        with (
            patch.object(stack_mod.shutil, "which", side_effect = which_map.get),
            patch.object(stack_mod.subprocess, "run", side_effect = fake_run),
            patch.dict(
                stack_mod.os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}, clear = False
            ),
        ):
            assert stack_mod._has_rocm_gpu() is True

    @staticmethod
    def _run_sh_helper(tmp_path, src: str, fn_names: list, cvd):
        """Extract shell functions, run the usable-GPU one against a fake nvidia-smi; return "usable"/"not_usable"."""
        import os as _os
        import subprocess as sp

        blocks = []
        for name in fn_names:
            start = src.find(f"{name}() {{")
            assert start >= 0, f"{name} missing"
            end = src.find("\n}", start) + 2
            blocks.append(src[start:end])
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir(exist_ok = True)
        smi = fake_bin / "nvidia-smi"
        smi.write_text("#!/bin/sh\necho 'GPU 0: NVIDIA Fake (UUID: GPU-x)'\n")
        smi.chmod(0o755)
        script = tmp_path / "probe.sh"
        script.write_text(
            "#!/bin/sh\n" + "\n".join(blocks) + "\n"
            f"if {fn_names[-1]}; then echo usable; else echo not_usable; fi\n"
        )
        env = dict(_os.environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        if cvd is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = cvd
        return sp.run(
            ["sh", str(script)], capture_output = True, text = True, timeout = 30, env = env
        ).stdout.strip()

    @pytest.mark.parametrize(
        "cvd, expected",
        [(None, "usable"), ("", "not_usable"), ("-1", "not_usable"), ("0", "usable")],
    )
    def test_install_sh_helper_cvd(self, tmp_path, cvd, expected):
        src = (PACKAGE_ROOT / "install.sh").read_text(encoding = "utf-8")
        out = self._run_sh_helper(
            tmp_path,
            src,
            ["_run_bounded", "_cvd_hides_nvidia", "_has_usable_nvidia_gpu"],
            cvd,
        )
        assert out == expected

    @pytest.mark.parametrize(
        "cvd, expected",
        [(None, "usable"), ("", "not_usable"), ("-1", "not_usable"), ("0", "usable")],
    )
    def test_setup_sh_helper_cvd(self, tmp_path, cvd, expected):
        src = SETUP_SH.read_text(encoding = "utf-8")
        out = self._run_sh_helper(
            tmp_path,
            src,
            [
                "_setup_run_smi",
                "_setup_cvd_hides_nvidia",
                "_setup_has_usable_nvidia_gpu",
            ],
            cvd,
        )
        assert out == expected


class TestRedactInstallOutput:
    """_redact_install_output scrubs index-URL credentials from a captured install log
    before it is printed on failure (uv/pip embeds the failing --index-url verbatim)."""

    def test_userinfo_redacted(self):
        out = stack_mod._redact_install_output(
            "ERROR: failed https://alice:s3cr3t@download.pytorch.org/whl/cu128"
        )
        assert out == "ERROR: failed https://<redacted>@download.pytorch.org/whl/cu128"

    def test_bytes_input_decoded_and_redacted(self):
        out = stack_mod._redact_install_output(
            b"fetch https://ghp_deadbeef@host/whl/cu128 failed"
        )
        assert out == "fetch https://<redacted>@host/whl/cu128 failed"

    def test_query_values_redacted(self):
        out = stack_mod._redact_install_output(
            "url https://host/whl/cu128?token=abcd1234&channel=beta unreachable"
        )
        assert (
            out
            == "url https://host/whl/cu128?token=<redacted>&channel=<redacted> unreachable"
        )

    def test_fragment_redacted(self):
        out = stack_mod._redact_install_output(
            "ERROR: could not fetch https://mirror.local/whl/cu128#token=SECRET123 (403)"
        )
        assert (
            out
            == "ERROR: could not fetch https://mirror.local/whl/cu128#<redacted> (403)"
        )

    def test_query_and_fragment_both_redacted(self):
        out = stack_mod._redact_install_output(
            "https://host/whl/cu128?token=abc#sig=xyz done"
        )
        assert out == "https://host/whl/cu128?token=<redacted>#<redacted> done"

    def test_bare_hash_comment_untouched(self):
        # The fragment redaction is URL-anchored: a shell comment in tool output survives.
        assert (
            stack_mod._redact_install_output("# retrying with --no-cache-dir")
            == "# retrying with --no-cache-dir"
        )

    def test_plain_line_untouched(self):
        assert (
            stack_mod._redact_install_output("Resolved 42 packages in 1.2s")
            == "Resolved 42 packages in 1.2s"
        )

    def test_no_secret_substring_survives(self):
        out = stack_mod._redact_install_output(
            "https://alice:s3cr3t@host/whl/cu128?token=SUPERSECRET#frag=ALSOSECRET"
        )
        assert (
            "s3cr3t" not in out and "SUPERSECRET" not in out and "ALSOSECRET" not in out
        )


class TestTrimIndexPathSlashes:
    """_trim_index_path_slashes strips trailing PATH slashes only; a ?query/#fragment token
    ending in "/" must survive (a whole-URL rstrip would corrupt a base64 token)."""

    def test_double_path_slash_collapsed(self):
        assert (
            stack_mod._trim_index_path_slashes("https://h/whl/cu128//")
            == "https://h/whl/cu128"
        )

    def test_query_token_slash_preserved(self):
        assert (
            stack_mod._trim_index_path_slashes("https://h/whl/cu128?token=ab12cd/")
            == "https://h/whl/cu128?token=ab12cd/"
        )

    def test_path_slash_trimmed_query_kept(self):
        assert (
            stack_mod._trim_index_path_slashes("https://h/whl/cu128//?token=ab12cd/")
            == "https://h/whl/cu128?token=ab12cd/"
        )

    def test_fragment_slash_preserved(self):
        assert (
            stack_mod._trim_index_path_slashes("https://h/whl/cu128#anchor/")
            == "https://h/whl/cu128#anchor/"
        )


class TestRocmFamilyLeafParity:
    """_is_pip_rocm_family_leaf must match re.fullmatch(rocm\\d+(?:\\.\\d+)?): a trailing dot
    (rocm7.) is a CUSTOM pin, not a family (the historical bash/py validator asymmetry)."""

    @pytest.mark.parametrize(
        "leaf, expected",
        [
            ("rocm7", True),
            ("rocm7.2", True),
            ("gfx1151", True),
            ("rocm7.", False),
            ("rocm.7", False),
            ("rocm7..2", False),
            ("rocm7.2.1", False),
            ("rocm7.2-private", False),
            ("cpu", False),
            ("cu128", False),
        ],
    )
    def test_family_classification(self, leaf, expected):
        assert stack_mod._is_pip_rocm_family_leaf(leaf) is expected


class TestTorchIndexLeafAllSlashes:
    """_torch_index_leaf drops query/fragment then strips ALL trailing slashes, so a
    double-slash index still yields the real leaf (not an empty string)."""

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://m/whl/cu128//", "cu128"),
            ("https://m/whl/rocm7.2///", "rocm7.2"),
            ("https://m/whl/cu128//?token=x", "cu128"),
            ("https://m/whl/cu128/", "cu128"),
        ],
    )
    def test_leaf_never_empty_on_double_slash(self, url, expected):
        assert stack_mod._torch_index_leaf(url) == expected


class TestUvIndexEnvVarsScrub:
    """The pinned-install env scrub must drop PIP_NO_INDEX (which makes the pip fallback
    ignore ALL indexes, defeating the pin) and PIP_INDEX_URL (replaces the pinned index)."""

    def test_pip_no_index_scrubbed(self):
        assert "PIP_NO_INDEX" in stack_mod._UV_INDEX_ENV_VARS

    def test_pip_index_url_scrubbed(self):
        assert "PIP_INDEX_URL" in stack_mod._UV_INDEX_ENV_VARS
