"""Tests for install_python_stack._build_uv_cmd torch-backend handling."""

from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
import io
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


def _shared_setup_1(installs, monkeypatch):
    monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
    monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: True)
    monkeypatch.setattr(ips, "_stage_replacement", lambda _name: "/staged")
    monkeypatch.setattr(
        ips,
        "pip_install_try",
        lambda label, *args, **kwargs: installs.append((label, args, kwargs)) or True,
    )


def _shared_setup_2(monkeypatch, probes):
    monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda name: next(probes[name]))
    monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _name: [])
    monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
    monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)


def _shared_setup_3(tmp_path):
    backup = tmp_path / "~nsloth-2026.8.12.dist-info"
    backup.mkdir()
    (backup / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: unsloth\nVersion: 2026.8.12\n", encoding = "utf-8"
    )
    return backup


def _shared_setup_4(monkeypatch, probes):
    monkeypatch.setattr(
        ips.install_manifest,
        "installed_versions",
        lambda name: next(probes[name]),
    )


def _shared_setup_5(tmp_path):
    record = tmp_path / "unsloth-2026.8.12.dist-info"
    record.mkdir()
    (record / "METADATA").write_bytes(b"\xff\xfe")
    (record / "RECORD").write_text("unsloth/gone.py,,\n")
    return record


STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))

import install_python_stack as ips

STACK_SOURCE = (STUDIO_DIR / "install_python_stack.py").read_text(encoding = "utf-8")


# A CI image with its own /etc/pip.conf would leak into these assertions, and a test that
# mocks subprocess could poison the memoised read for whatever runs next under -p randomly.
@pytest.fixture(autouse = True)
def _hermetic_pinned_pip_config(request):
    ips._PINNED_PIP_CONFIG_LISTING = None
    ips._PINNED_PIP_CONFIG_ATTEMPTS = 2
    if "reads_real_pip_config" in request.keywords:
        yield
    else:
        with mock.patch.object(ips, "_pinned_pip_config_overrides", lambda *a, **k: {}):
            yield
    ips._PINNED_PIP_CONFIG_LISTING = None
    ips._PINNED_PIP_CONFIG_ATTEMPTS = 2


class TestUvOnlyBinaryOnPinnedCommands:
    """uv reads neither pip.conf nor PIP_ONLY_BINARY, and a pinned command runs with
    UV_NO_CONFIG=1, so restoring the policy in the environment alone leaves it unenforced
    on the leg that actually runs. Measured against uv 0.10.7: with PIP_ONLY_BINARY=:all:
    set, a pinned `uv pip install` builds the sdist anyway, and `--only-binary` refuses
    it."""

    PINNED = ("torch", "--index-url", "https://pin.example/whl")
    AMD = ("torch", "--index-url", "https://repo.amd.com/rocm/whl/gfx1151/")

    def _uv_cmd(self, args):
        return ips._pinned_cmd_and_env(ips._build_uv_cmd(args))[0]

    def _pip_cmd(self, args):
        return ips._pinned_cmd_and_env(ips._build_pip_cmd(args))[0]

    def test_a_pinned_uv_command_carries_only_binary_as_flags(self):
        with mock.patch.dict(os.environ, {"PIP_ONLY_BINARY": ":all:"}):
            cmd = self._uv_cmd(self.PINNED)
        assert cmd[-2:] == ["--only-binary", ":all:"]

    def test_each_entry_becomes_its_own_flag(self):
        """uv takes the option repeatably, not comma joined the way pip spells it."""
        with mock.patch.dict(os.environ, {"PIP_ONLY_BINARY": ":none:,numpy"}):
            cmd = self._uv_cmd(self.PINNED)
        assert cmd[-4:] == ["--only-binary", ":none:", "--only-binary", "numpy"]

    def test_an_amd_arch_pin_exempts_rocm_on_the_uv_leg(self):
        """Every torch on a gfx* index requires rocm[libraries], published as an sdist alone."""
        with mock.patch.dict(os.environ, {"PIP_ONLY_BINARY": ":all:"}):
            cmd = self._uv_cmd(self.AMD)
        assert cmd[-4:] == ["--only-binary", ":all:", "--no-binary", "rocm"]

    def test_an_amd_arch_pip_command_keeps_the_policy_in_env_and_the_exemption_in_argv(self):
        """pip reads PIP_ONLY_BINARY itself; the command line only has to exempt rocm."""
        with mock.patch.dict(os.environ, {"PIP_ONLY_BINARY": ":all:"}):
            cmd, env = ips._pinned_cmd_and_env(ips._build_pip_cmd(self.AMD))
        assert "--only-binary" not in cmd
        assert cmd[-2:] == ["--no-binary", "rocm"]
        assert env["PIP_ONLY_BINARY"] == ":all:"

    @pytest.mark.parametrize(
        "index",
        (
            "https://download.pytorch.org/whl/cu128",
            "https://download.pytorch.org/whl/cpu",
            "https://download.pytorch.org/whl/xpu",
            "https://download.pytorch.org/whl/rocm7.2",
            "https://mirror.corp/gfx1151/cu128",
            "https://mirror.corp/gfx-private",
        ),
    )
    @pytest.mark.parametrize("leg", ("uv", "pip"))
    def test_no_other_pin_is_exempted(self, index, leg):
        """Anywhere else the name would only let an index get a build past the policy."""
        with mock.patch.dict(os.environ, {"PIP_ONLY_BINARY": ":all:"}):
            args = ("torch", "--index-url", index)
            cmd = self._uv_cmd(args) if leg == "uv" else self._pip_cmd(args)
        assert "--no-binary" not in cmd

    @pytest.mark.parametrize(
        "index",
        (
            "https://mirror.corp/amd/gfx120X-all/",
            "https://mirror.corp/amd/gfx110X-all?token=x",
        ),
    )
    def test_a_mirrored_amd_arch_index_is_exempted_too(self, index):
        """UNSLOTH_AMD_ROCM_MIRROR / UNSLOTH_ROCM_WINDOWS_MIRROR keep the gfx leaf."""
        with mock.patch.dict(os.environ, {"PIP_ONLY_BINARY": ":all:"}):
            cmd = self._pip_cmd(("torch", "--index-url", index))
        assert cmd[-2:] == ["--no-binary", "rocm"]

    @pytest.mark.parametrize("policy", ("rocm", ":all:,rocm", ":all:,ROCm"))
    @pytest.mark.parametrize("leg", ("uv", "pip"))
    def test_a_package_the_operator_names_is_not_exempted(self, policy, leg):
        """A command-line --no-binary overrides the operator's own rule for that package."""
        with mock.patch.dict(os.environ, {"PIP_ONLY_BINARY": policy}):
            cmd = self._uv_cmd(self.AMD) if leg == "uv" else self._pip_cmd(self.AMD)
        assert "--no-binary" not in cmd

    @pytest.mark.reads_real_pip_config  # stubbed below with a read that fails once
    @pytest.mark.parametrize("installer", ("pip_install", "pip_install_try"))
    def test_the_flag_and_the_environment_can_never_disagree(self, monkeypatch, installer):
        """Both come from one read. A failed read is not memoised, so asking twice let a
        transient miss leave the flag off the argv while the retry put the policy in the
        environment, which uv never reads."""
        answers = iter(({}, {"PIP_ONLY_BINARY": ":all:"}))
        monkeypatch.setattr(ips, "_pinned_pip_config_overrides", lambda *a, **k: next(answers, {}))
        monkeypatch.delenv("PIP_ONLY_BINARY", raising = False)
        monkeypatch.setattr(ips, "USE_UV", True)
        runs = []
        monkeypatch.setattr(
            ips.subprocess,
            "run",
            lambda cmd, **kwargs: runs.append((cmd, kwargs.get("env")))
            or subprocess.CompletedProcess(cmd, 0, b""),
        )
        getattr(ips, installer)("torch", *self.PINNED, constrain = False)
        ((cmd, env),) = runs
        flagged = cmd[cmd.index("--only-binary") + 1] if "--only-binary" in cmd else None
        assert flagged == (env or {}).get("PIP_ONLY_BINARY")

    @pytest.mark.reads_real_pip_config  # stubbed below with reads that disagree
    def test_the_pip_fallback_runs_with_the_env_its_exemption_came_from(self, monkeypatch):
        """uv fails, so pip_install falls back. Had run() read the config again, a policy that
        appeared on that read would refuse rocm with no exemption on the argv."""
        answers = iter(({}, {}, {"PIP_ONLY_BINARY": ":all:"}))
        monkeypatch.setattr(ips, "_pinned_pip_config_overrides", lambda *a, **k: next(answers, {}))
        monkeypatch.delenv("PIP_ONLY_BINARY", raising = False)
        monkeypatch.setattr(ips, "USE_UV", True)
        runs = []

        def fake_run(cmd, **kwargs):
            runs.append((cmd, kwargs.get("env")))
            return subprocess.CompletedProcess(cmd, 1 if cmd[:1] == ["uv"] else 0, b"")

        monkeypatch.setattr(ips.subprocess, "run", fake_run)
        ips.pip_install("torch", *self.AMD, constrain = False)
        pip_cmd, pip_env = runs[-1]
        assert pip_cmd[:3] == [sys.executable, "-m", "pip"]
        assert ("--no-binary" in pip_cmd) == bool((pip_env or {}).get("PIP_ONLY_BINARY"))

    def test_a_non_pinned_command_is_left_alone(self):
        """It keeps its config file, so uv applies the operator's policy itself."""
        with mock.patch.dict(os.environ, {"PIP_ONLY_BINARY": ":all:"}):
            assert self._uv_cmd(("torch",)) == ips._build_uv_cmd(("torch",))
            assert self._pip_cmd(("torch",)) == ips._build_pip_cmd(("torch",))

    def test_no_policy_leaves_the_argv_untouched(self):
        """An unconfigured host must run exactly the command main ran, AMD indexes included."""
        env = {k: v for k, v in os.environ.items() if k != "PIP_ONLY_BINARY"}
        with mock.patch.dict(os.environ, env, clear = True):
            for args in (self.PINNED, self.AMD):
                assert self._uv_cmd(args) == ips._build_uv_cmd(args)
                assert self._pip_cmd(args) == ips._build_pip_cmd(args)


class TestBuildUvCmdTorchBackend:
    """Verify _build_uv_cmd only adds --torch-backend when UV_TORCH_BACKEND is set."""

    def _call(self, args: tuple[str, ...] = ()) -> list[str]:
        return ips._build_uv_cmd(args)

    def test_default_no_torch_backend(self):
        """Without UV_TORCH_BACKEND env var, no --torch-backend flag."""
        env = os.environ.copy()
        env.pop("UV_TORCH_BACKEND", None)
        with mock.patch.dict(os.environ, env, clear = True):
            cmd = self._call(("somepackage",))
        assert not any(
            a.startswith("--torch-backend") for a in cmd
        ), f"--torch-backend should not appear by default, got: {cmd}"

    @pytest.mark.parametrize(
        "backend, expected_flag",
        [
            pytest.param("auto", "--torch-backend=auto", id = "uv_torch_backend_auto"),
            pytest.param("cpu", "--torch-backend=cpu", id = "uv_torch_backend_cpu"),
            pytest.param("cpu", "--torch-backend=cpu", id = "uv_torch_backend_kept_for_unpinned"),
        ],
    )
    def test_build_uv_cmd_torch_backend_cases(self, backend, expected_flag):
        with mock.patch.dict(os.environ, {"UV_TORCH_BACKEND": backend}):
            cmd = self._call(("somepackage",))
        assert expected_flag in cmd

    def test_uv_torch_backend_empty(self):
        """UV_TORCH_BACKEND="" (empty string) should NOT add --torch-backend."""
        with mock.patch.dict(os.environ, {"UV_TORCH_BACKEND": ""}):
            cmd = self._call(("somepackage",))
        assert not any(
            a.startswith("--torch-backend") for a in cmd
        ), f"Empty UV_TORCH_BACKEND should not add flag, got: {cmd}"

    def test_uv_torch_backend_skipped_for_pinned_index(self):
        """A pinned-index command must NOT get --torch-backend: uv's torch backend
        redirects torch resolution to its own per-backend index even when
        --index-url is given (verified: cu128 pin + backend cpu installs
        torch+cpu), defeating the pin."""
        for pin_flag in ("--index-url", "--default-index"):
            with mock.patch.dict(os.environ, {"UV_TORCH_BACKEND": "cpu"}):
                cmd = self._call(("torch", pin_flag, "https://download.pytorch.org/whl/cu128"))
            assert not any(
                a.startswith("--torch-backend") for a in cmd
            ), f"{pin_flag} command must not carry --torch-backend, got: {cmd}"


class TestUvSafePath:
    """_uv_safe_path hands uv a space-free `-c`/`-r` path (issue #6503)."""

    def test_passthrough_when_no_space(self):
        """A path without a space is returned unchanged on every platform."""
        p = "/tmp/plain/constraints.txt"
        assert ips._uv_safe_path(p) == p

    @pytest.mark.skipif(ips.IS_WINDOWS, reason = "POSIX temp-copy fallback")
    def test_posix_space_path_preserves_relative_requirements(self, tmp_path):
        src = tmp_path / "Open Source" / "constraints.txt"
        src.parent.mkdir(parents = True)
        src.write_text("-r child.txt\n")
        (src.parent / "child.txt").write_text("torch>=2.6\n")

        out = ips._uv_safe_path(str(src))

        assert " " not in out, f"uv-safe path still has a space: {out!r}"
        assert out != str(src)
        assert Path(out).read_text() == "-r child.txt\n"
        assert (Path(out).parent / "child.txt").read_text() == "torch>=2.6\n"

    @pytest.mark.skipif(ips.IS_WINDOWS, reason = "POSIX temp-copy fallback")
    def test_posix_missing_file_falls_back_to_original(self):
        """No file to copy -> return the original path rather than raise."""
        p = "/nonexistent dir/constraints.txt"
        assert ips._uv_safe_path(p) == p


class TestUvSafePathHardening:
    """Edge cases for uv_safe_path + the UV_OVERRIDE channel (issue #6503)."""

    @pytest.mark.skipif(ips.IS_WINDOWS, reason = "POSIX temp-copy fallback")
    def test_tmpdir_with_space_falls_back(self, tmp_path, monkeypatch):
        """A space in the temp root itself -> fall back to the original path."""
        from backend.utils import uv_path_safety as uvps

        spaced = tmp_path / "tmp dir with space"
        spaced.mkdir()
        monkeypatch.setattr(uvps.tempfile, "mkdtemp", lambda *a, **k: str(spaced))
        src = tmp_path / "Open Source" / "constraints.txt"
        src.parent.mkdir(parents = True)
        src.write_text("idna\n")
        assert uvps.uv_safe_path(str(src)) == str(src)

    @pytest.mark.skipif(ips.IS_WINDOWS, reason = "POSIX temp-copy fallback")
    def test_alias_failure_falls_back_to_a_copy(self, tmp_path, monkeypatch):
        """A symlink failure must still hand uv a space-free path, and not orphan the dir."""
        from backend.utils import uv_path_safety as uvps

        src = tmp_path / "Open Source" / "constraints.txt"
        src.parent.mkdir(parents = True)
        src.write_text("idna\n")

        def boom(*a, **k):
            raise OSError("boom")

        monkeypatch.setattr(uvps.os, "symlink", boom)
        out = uvps.uv_safe_path(str(src))

        assert " " not in out
        assert Path(out).read_text() == "idna\n"
        assert str(Path(out).parent) in uvps._UV_SAFE_PATH_TMPDIRS

    @pytest.mark.skipif(ips.IS_WINDOWS, reason = "POSIX temp-copy fallback")
    def test_cleanup_removes_and_clears_registry(self, tmp_path):
        """The atexit-registered cleanup removes the copies and empties the list."""
        from backend.utils import uv_path_safety as uvps

        src = tmp_path / "Open Source" / "constraints.txt"
        src.parent.mkdir(parents = True)
        src.write_text("idna\n")
        out = uvps.uv_safe_path(str(src))
        tmp_dir = Path(out).parents[1]
        assert tmp_dir.is_dir() and str(tmp_dir) in uvps._UV_SAFE_PATH_TMPDIRS

        uvps._cleanup_uv_safe_path_tmpdirs()

        assert not tmp_dir.exists()
        assert uvps._UV_SAFE_PATH_TMPDIRS == []

    @pytest.mark.skipif(ips.IS_WINDOWS, reason = "POSIX temp-copy fallback")
    def test_uv_override_value_is_space_safe(self, tmp_path):
        """The value stored for UV_OVERRIDE must be space-free."""
        from backend.utils import uv_path_safety as uvps

        overrides = tmp_path / "Open Source" / "overrides-darwin-arm64.txt"
        overrides.parent.mkdir(parents = True)
        overrides.write_text("transformers>=4.57.6\n")

        value = uvps.uv_safe_path(overrides)

        assert " " not in value
        assert Path(value).read_text() == "transformers>=4.57.6\n"


class TestPinnedIndexClearsUvEnv:
    """A pinned torch install (--index-url / --default-index) must neutralise an
    inherited UV_INDEX / UV_EXTRA_INDEX_URL so the pinned wheel index wins.

    uv treats the default index (--index-url / --default-index) as LOWEST priority,
    so an inherited UV_INDEX / UV_EXTRA_INDEX_URL (a corporate/CPU mirror) would be
    searched first and, under uv's default first-index strategy, resolve torch from
    the wrong mirror -- after which the marker records a wheel index that was never
    used. install.sh (#6898), install.ps1 and setup.ps1 already clear these for
    pinned installs; install_python_stack must match (parity across all installers).
    """

    UV_VARS = ("UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_INDEX", "UV_EXTRA_INDEX_URL")

    def test_pinned_index_url_strips_uv_index_vars(self):
        cmd = [
            "uv",
            "pip",
            "install",
            "--force-reinstall",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ]
        with mock.patch.dict(
            os.environ,
            {
                "UV_INDEX": "https://mirror.corp/simple",
                "UV_EXTRA_INDEX_URL": "https://mirror.corp/extra",
                "UV_INDEX_URL": "https://mirror.corp/root",
                "UV_DEFAULT_INDEX": "https://mirror.corp/default",
            },
        ):
            env = ips._install_env_for_cmd(cmd)
        assert env is not None, "a --index-url install must run with a scrubbed env"
        for var in self.UV_VARS:
            assert var not in env, f"{var} must be cleared for a pinned-index install"

    def test_pinned_default_index_strips_uv_index_vars(self):
        # default-index must be gated too (matches install.sh / install.ps1).
        cmd = ["uv", "pip", "install", "torch", "--default-index", "https://x/cu126"]
        with mock.patch.dict(os.environ, {"UV_INDEX": "https://mirror.corp/simple"}):
            env = ips._install_env_for_cmd(cmd)
        assert env is not None
        assert "UV_INDEX" not in env

    def test_non_pinned_install_keeps_user_mirror(self):
        # A plain install (no --index-url) must NOT scrub the env, so a user's mirror still applies to base packages.
        cmd = ["uv", "pip", "install", "unsloth", "unsloth-zoo"]
        with mock.patch.dict(os.environ, {"UV_INDEX": "https://mirror.corp/simple"}):
            env = ips._install_env_for_cmd(cmd)
        assert env is None, "non-pinned installs must inherit the caller env unchanged"

    def test_scrubbed_env_preserves_other_vars(self):
        cmd = ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
        with mock.patch.dict(
            os.environ,
            {"UV_INDEX": "https://mirror.corp/simple", "PATH_SENTINEL_XYZ": "keepme"},
        ):
            env = ips._install_env_for_cmd(cmd)
        assert env is not None
        assert env.get("PATH_SENTINEL_XYZ") == "keepme", "only uv index vars are removed"

    def test_pinned_cmd_strips_pip_extra_index_url(self):
        """PIP_EXTRA_INDEX_URL is stripped for pinned commands so the pip
        fallback cannot satisfy torch from an inherited extra index."""
        with mock.patch.dict(os.environ, {"PIP_EXTRA_INDEX_URL": "https://mirror/simple"}):
            env = ips._install_env_for_cmd(
                ["pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env is not None and "PIP_EXTRA_INDEX_URL" not in env

    def test_pinned_cmd_strips_uv_torch_backend(self):
        """UV_TORCH_BACKEND is stripped for pinned commands so uv cannot read it
        from the environment and reroute torch off the pinned index."""
        with mock.patch.dict(os.environ, {"UV_TORCH_BACKEND": "cpu"}):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env is not None and "UV_TORCH_BACKEND" not in env

    def test_pinned_cmd_disables_uv_config_discovery(self):
        """A DISCOVERED uv.toml / pyproject [tool.uv] outranks the CLI pin too
        (verified with uv 0.10: [pip] torch-backend = "cpu" and a non-default
        [[index]] both resolve torch+cpu against an explicit --index-url /
        --default-index cu126 pin). Pinned commands must run with UV_NO_CONFIG=1
        and without an inherited UV_CONFIG_FILE."""
        with mock.patch.dict(os.environ, {"UV_CONFIG_FILE": "/etc/uv/uv.toml"}):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env is not None
        assert env.get("UV_NO_CONFIG") == "1"
        assert "UV_CONFIG_FILE" not in env

    def test_pinned_cmd_disables_pip_config_files(self):
        """devnull is the ONLY spelling that reaches a SITE or GLOBAL pip.conf. Measured
        on pip 26.2: naming a real file suppresses the per-user file alone, so a
        venv-level `no-index` still killed the pin."""
        env = ips._install_env_for_cmd(
            ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
        )
        assert env is not None
        assert env.get("PIP_CONFIG_FILE") == os.devnull

    def test_non_pinned_cmd_keeps_uv_config_discovery(self):
        """Non-pinned installs inherit the caller env unchanged, so a user's uv
        configuration still applies to base packages."""
        env = ips._install_env_for_cmd(["uv", "pip", "install", "unsloth"])
        assert env is None


class TestSdistOnlyBuildArgs:
    """A hardened user config must not be able to fail the extras step.

    #8530: `no-build = true` (uv.toml) or `only-binary = :all:` (pip.conf) makes every
    wheel-less requirement in extras.txt unresolvable, so the install died at "unsloth
    extras". A PACKAGE-SCOPED --no-binary overrides that for those names only -- verified
    against uv 0.10 and pip 26: drop one name and that name is refused again.
    """

    def test_emits_no_binary_for_every_sdist_only_package(self):
        args = ips._sdist_only_build_args(*ips.SDIST_ONLY_PACKAGES)
        for name in ips.SDIST_ONLY_PACKAGES:
            assert ["--no-binary", name] == args[
                args.index(name) - 1 : args.index(name) + 1
            ], f"{name} must be passed as a package-scoped --no-binary, got: {args}"
        assert len(args) == 2 * len(ips.SDIST_ONLY_PACKAGES)

    def test_openai_whisper_is_covered(self):
        """The package named in the issue, and the transitive one behind omegaconf."""
        assert "openai-whisper" in ips.SDIST_ONLY_PACKAGES
        # omegaconf==2.3.1 pins antlr4-python3-runtime below the 4.13.2 wheel, so it arrives as a transitive sdist and
        # fails no-build even though extras.txt never names it.
        assert "antlr4-python3-runtime" in ips.SDIST_ONLY_PACKAGES

    def test_flags_survive_translation_to_uv(self):
        """uv is the primary path, so the flags must reach _build_uv_cmd intact."""
        cmd = ips._build_uv_cmd(tuple(ips._sdist_only_build_args(*ips.SDIST_ONLY_PACKAGES)))
        for name in ips.SDIST_ONLY_PACKAGES:
            assert name in cmd
        assert cmd.count("--no-binary") == len(ips.SDIST_ONLY_PACKAGES)

    def test_flags_survive_translation_to_pip(self):
        """And the pip FALLBACK must carry them too, for the uv-less/uv-broken case."""
        cmd = ips._build_pip_cmd(tuple(ips._sdist_only_build_args(*ips.SDIST_ONLY_PACKAGES)))
        for name in ips.SDIST_ONLY_PACKAGES:
            assert name in cmd
        assert cmd.count("--no-binary") == len(ips.SDIST_ONLY_PACKAGES)

    @pytest.mark.parametrize(
        "is_macos, version, expected",
        [
            (True, (3, 14, 0), True),
            (True, (3, 13, 12), False),
            (False, (3, 14, 0), False),
            (False, (3, 13, 12), False),
        ],
    )
    def test_mecab_is_exempted_only_where_it_has_no_wheel(self, is_macos, version, expected):
        """extras.txt pins MeCab==0.996.5 on macOS cp314+, which ships only an sdist.

        MeCab is a C extension, so an unconditional exemption would force a
        compiler-dependent build on every other host -- a worse bug than the one being
        fixed. Verified against uv 0.10: 0.996.5 is refused under `no-build = true` for
        macOS cp314 and resolves with --no-binary MeCab; 0.996.13 stays a wheel elsewhere.
        """
        with (
            mock.patch.object(ips, "IS_MACOS", is_macos),
            mock.patch.object(sys, "version_info", version),
        ):
            names = ips._extras_sdist_only_packages()
        assert ("MeCab" in names) is expected
        assert set(ips.SDIST_ONLY_PACKAGES) <= set(names)

    def test_the_diffusers_release_is_not_forced_through_a_source_build(self):
        """The pinned release ships wheels, so forcing a source build defeats the pin."""
        pin_lines = (ips.REQ_ROOT / "diffusers-pin.txt").read_text(encoding = "utf-8").splitlines()
        pin_options = [line.split("#", 1)[0].strip() for line in pin_lines]
        assert not any(option.startswith("--no-binary") for option in pin_options)

        tree = ast.parse(Path(ips.__file__).read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "pip_install"):
                continue
            req = next((k for k in node.keywords if k.arg == "req"), None)
            if req is None or "diffusers-pin.txt" not in ast.unparse(req.value):
                continue
            try:
                literal_arguments = [ast.literal_eval(argument) for argument in node.args]
            except (ValueError, TypeError):
                pytest.fail("the diffusers pin options must remain literal and auditable")
            assert all(isinstance(argument, str) for argument in literal_arguments)
            assert not any(argument.startswith("--no-binary") for argument in literal_arguments)
            return
        pytest.fail("no pip_install(req=.../diffusers-pin.txt) call found")

    def test_the_extras_step_actually_passes_them(self):
        """The helper existing is not the fix; the extras call site using it is.

        extras.txt is the manifest that carries the wheel-less requirements, and its
        pip_install() is fatal, so this is the call that #8530 died on.
        """
        tree = ast.parse(Path(ips.__file__).read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "pip_install"):
                continue
            req = next((k for k in node.keywords if k.arg == "req"), None)
            if req is None or "extras.txt" not in ast.unparse(req.value):
                continue
            starred = [ast.unparse(a.value) for a in node.args if isinstance(a, ast.Starred)]
            assert any("_sdist_only_build_args(" in s for s in starred), (
                f"the extras.txt install at line {node.lineno} must splat "
                "_sdist_only_build_args() or a hardened uv.toml fails it again"
            )
            assert any("_extras_sdist_only_packages()" in s for s in starred), (
                "the extras install must use the platform-aware list so the macOS "
                "cp314 MeCab sdist is exempted too"
            )
            return
        pytest.fail("no pip_install(req=.../extras.txt) call found")

    def test_matches_the_ci_nobuild_allowlists(self):
        """CI already ratifies exactly these as audited pure-Python sdist builds.

        If the two lists drift, either CI fails a legitimate build or we exempt a
        package nobody audited, so pin them together.
        """
        repo = Path(ips.__file__).resolve().parents[1]
        shell = (repo / ".github/scripts/clean-machine-assert.sh").read_text(encoding = "utf-8")
        allow = shell[shell.index('_allow="$(printf') :]
        allow = allow[: allow.index("\n")]
        ps1 = (repo / ".github/scripts/assert-nobuild.ps1").read_text(encoding = "utf-8")
        for name in ips.SDIST_ONLY_PACKAGES:
            assert name in allow, f"{name} missing from clean-machine-assert.sh nobuild allowlist"
            assert f"'{name}'" in ps1, f"{name} missing from assert-nobuild.ps1 allowlist"
        # overlay:false releases still need the temporary Diffusers sdist exemption.
        assert "diffusers" in allow
        assert "'diffusers'" in ps1


class TestHardenedPipConfigRelaxation:
    """`require-hashes = true` in pip.conf killed the pip FALLBACK in #8530.

    Every requirements file we ship is pinned but unhashed, so hash-required mode can
    never be satisfied. pip applies env vars AFTER config files, so PIP_REQUIRE_HASHES=0
    in the child env overrides it while pip.conf's index-url, trusted-host, cert and
    proxy stay in force. It has no command-line equivalent, hence the env var.
    """

    HOSTILE = {
        "PIP_REQUIRE_HASHES": "1",
        "PIP_ONLY_BINARY": ":all:",
        "UV_NO_BUILD": "1",
        "UV_EXCLUDE_NEWER": "2024-01-01T00:00:00Z",
    }

    def test_uv_commands_are_left_alone(self):
        """uv reads none of the PIP_* vars, and its own no-build is handled by the
        package-scoped --no-binary, so a uv command must still inherit the env
        unchanged -- the mirror contract at test_non_pinned_install_keeps_user_mirror."""
        with mock.patch.dict(os.environ, self.HOSTILE):
            assert ips._install_env_for_cmd(["uv", "pip", "install", "-r", "extras.txt"]) is None

    def test_non_pinned_pip_install_relaxes_hash_mode(self):
        with mock.patch.dict(os.environ, self.HOSTILE):
            env = ips._install_env_for_cmd(["python", "-m", "pip", "install", "-r", "extras.txt"])
        assert env is not None, "the pip fallback must not inherit require-hashes"
        assert env["PIP_REQUIRE_HASHES"] == "0"

    def test_non_pinned_pip_keeps_the_user_mirror_and_binary_policy(self):
        """Only hash mode is relaxed. The mirror stays, and so does only-binary --
        the wheel-less packages are exempted per-package on the command line instead."""
        with mock.patch.dict(
            os.environ,
            dict(self.HOSTILE, PIP_INDEX_URL = "https://mirror.corp/simple"),
        ):
            env = ips._install_env_for_cmd(["python", "-m", "pip", "install", "x"])
        assert env["PIP_INDEX_URL"] == "https://mirror.corp/simple"
        assert env["PIP_ONLY_BINARY"] == ":all:"
        assert "PIP_CONFIG_FILE" not in env, "a non-pinned install must still read pip.conf"
        assert "UV_NO_CONFIG" not in env, "a non-pinned install must still read uv.toml"

    def test_non_install_commands_are_untouched(self):
        """run() routes EVERY command through this helper, not just installs."""
        with mock.patch.dict(os.environ, self.HOSTILE):
            assert ips._install_env_for_cmd(["python", "-m", "pip", "--version"]) is None
            assert ips._install_env_for_cmd(["python", "-m", "ensurepip", "--upgrade"]) is None

    def test_pinned_cmd_clears_hash_mode_only(self):
        """Hash enforcement, which our unhashed requirements cannot satisfy, and nothing
        else the operator hardened: dropping only-binary would let a compromised mirror
        run a source build they had forbidden."""
        with mock.patch.dict(os.environ, self.HOSTILE):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env is not None
        for name in ("PIP_REQUIRE_HASHES", "UV_REQUIRE_HASHES"):
            assert name not in env, f"{name} cannot be satisfied by an unhashed pin"
        assert env["PIP_ONLY_BINARY"] == ":all:"
        assert env["UV_NO_BUILD"] == "1"  # inert for uv, but not ours to drop either
        assert env["UV_NO_CONFIG"] == "1"

    def test_pinned_cmd_clears_an_upload_cutoff_uv_alone_would_honour(self):
        """Only uv reads it, and pip_install falls back to pip whenever uv fails, so
        honouring it on the uv leg alone lets the fallback install past the cutoff."""
        with mock.patch.dict(os.environ, self.HOSTILE):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert "UV_EXCLUDE_NEWER" not in env

    def test_pinned_cmd_clears_forced_source_builds(self):
        """Dropping it is hardening: it would force torch to be BUILT from an sdist the
        pinned index does not serve."""
        with mock.patch.dict(os.environ, {"PIP_NO_BINARY": ":all:", "UV_NO_BINARY": ":all:"}):
            env = ips._install_env_for_cmd(
                ["python", "-m", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env is not None
        assert "PIP_NO_BINARY" not in env and "UV_NO_BINARY" not in env

    LISTING = (
        "global.index-url='https://mirror/simple'\n"
        "global.extra-index-url='https://other/simple'\n"
        "global.no-index='true'\n"
        "global.no-binary=':all:'\n"
        "global.require-hashes='true'\n"
        "install.only-binary=':all:'\n"
        "global.cert='/etc/ssl/corp.pem'\n"
        "global.proxy='http://proxy.corp:3128'\n"
        "global.trusted-host='a.corp\\nb.corp'\n"
        "list.format='columns'\n"
        ":env:.no-binary=':all:'\n"
    )

    def _overrides(
        self,
        listing = None,
        subcommand = "install",
    ):
        return ips._parse_pinned_pip_config(
            (self.LISTING if listing is None else listing).encode(), subcommand
        )

    def test_devnull_gives_the_operators_policy_and_transport_back(self):
        """only-binary is the security half; cert / proxy / trusted-host are the half
        that makes a private index reachable, which devnull alone used to drop."""
        overrides = self._overrides()
        assert overrides["PIP_ONLY_BINARY"] == ":all:"
        assert overrides["PIP_CERT"] == "/etc/ssl/corp.pem"
        assert overrides["PIP_PROXY"] == "http://proxy.corp:3128"
        # Newline separated by `pip config list`, whitespace separated in the environment.
        assert overrides["PIP_TRUSTED_HOST"] == "a.corp b.corp"

    def test_the_pin_and_the_unsatisfiable_policy_never_come_back(self):
        """The pin replaces the source keys, no-binary would force a source build, and
        require-hashes cannot be met. None may come back."""
        overrides = self._overrides()
        for name in (
            "PIP_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
            "PIP_NO_INDEX",
            "PIP_NO_BINARY",
            "PIP_REQUIRE_HASHES",
        ):
            assert name not in overrides, f"{name} must not survive the pinned scrub"

    def test_options_from_unrelated_sections_are_not_translated(self):
        """`list.format` becoming PIP_FORMAT would apply it to install."""
        assert "PIP_FORMAT" not in self._overrides()

    def test_env_entries_are_skipped(self):
        """The child already inherits them, and re-asserting would undo the scrub."""
        assert "PIP_NO_BINARY" not in self._overrides()

    @pytest.mark.reads_real_pip_config
    @pytest.mark.parametrize(
        "outcome",
        [
            mock.Mock(returncode = 1, stdout = b""),  # no pip in the venv yet
            mock.Mock(returncode = 0, stdout = None),  # nothing captured
            OSError("no pip"),
            subprocess.TimeoutExpired("pip", 60),  # a wedged pip
        ],
    )
    def test_a_pip_that_cannot_answer_changes_nothing(self, outcome):
        """On the path to every pinned install, so anything but a clean listing has to
        degrade to no overrides."""
        kwargs = (
            {"side_effect": outcome}
            if isinstance(outcome, Exception)
            else {"return_value": outcome}
        )
        with mock.patch.object(ips.subprocess, "run", **kwargs):
            assert ips._pinned_pip_config_overrides() == {}

    @pytest.mark.reads_real_pip_config
    def test_only_a_successful_read_is_cached(self):
        """A transient miss must not cost the operator their cert and proxy for the rest
        of the run."""
        listing = b"global.cert='/etc/ssl/corp.pem'\n"
        with mock.patch.object(ips.subprocess, "run", side_effect = OSError("wedged")):
            assert ips._pinned_pip_config_overrides() == {}
        with mock.patch.object(ips.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode = 0, stdout = listing)
            assert ips._pinned_pip_config_overrides() == {"PIP_CERT": "/etc/ssl/corp.pem"}
            assert run.call_count == 1
        # ...and the success IS cached: N pinned commands, one subprocess.
        with mock.patch.object(ips.subprocess, "run") as run:
            assert ips._pinned_pip_config_overrides() == {"PIP_CERT": "/etc/ssl/corp.pem"}
            assert run.call_count == 0

    def test_garbage_in_the_listing_is_ignored_not_fatal(self):
        for listing in (
            b"",
            b"not a config listing\n",
            b"global.cert\n",
            b"=\n",
            b"global.cert=<unparseable>\n",
            b"\xff\xfe binary \x00\n",
        ):
            assert ips._parse_pinned_pip_config(listing) == {}

    def test_a_command_section_beats_global_for_a_scalar(self):
        """pip's own precedence for a single-valued option, resolved by position, not by
        the order the listing prints the two lines in."""
        for listing in (
            b"global.timeout='30'\ninstall.timeout='9'\n",
            b"install.timeout='9'\nglobal.timeout='30'\n",
        ):
            assert ips._parse_pinned_pip_config(listing)["PIP_TIMEOUT"] == "9"

    def test_the_section_read_is_the_one_pip_would_apply(self):
        """pip config is per subcommand: measured on pip 26.2, `[download] no-index` stops
        a `pip download` and leaves `pip install` alone. A PIP_ variable is command-wide,
        so reading `[install]` for a download both drops that command's own settings and
        imposes another command's."""
        listing = (
            b"global.timeout='30'\ninstall.only-binary=':all:'\n"
            b"install.timeout='9'\ndownload.timeout='5'\ndownload.cert='/etc/dl.pem'\n"
        )
        for_install = ips._parse_pinned_pip_config(listing, "install")
        assert for_install["PIP_TIMEOUT"] == "9" and for_install["PIP_ONLY_BINARY"] == ":all:"
        assert "PIP_CERT" not in for_install
        for_download = ips._parse_pinned_pip_config(listing, "download")
        assert for_download["PIP_TIMEOUT"] == "5" and for_download["PIP_CERT"] == "/etc/dl.pem"
        assert "PIP_ONLY_BINARY" not in for_download, "an install-only policy is not a download one"
        # A section belonging to neither is never read.
        assert "PIP_FORMAT" not in ips._parse_pinned_pip_config(
            b"list.format='columns'\n", "install"
        )

    @pytest.mark.parametrize(
        "cmd, expected",
        [
            (["python", "-m", "pip", "install", "x", "--index-url", "u"], "install"),
            (["python", "-m", "pip", "download", "x", "--index-url", "u"], "download"),
            (["python", "-m", "pip", "wheel", "x", "--index-url", "u"], "wheel"),
            # uv reads none of the PIP_ vars; they exist for its pip FALLBACK, an install.
            (["uv", "pip", "install", "x", "--index-url", "u"], "install"),
            (["python", "-m", "pip", "uninstall", "-y", "x"], "install"),
        ],
    )
    def test_the_subcommand_drives_which_section_is_read(self, cmd, expected):
        assert ips._pip_subcommand_of(cmd) == expected

    @pytest.mark.reads_real_pip_config
    def test_the_xpu_download_gets_the_download_section(self):
        """_ensure_xpu_triton's pinned fetch is a `pip download`, so a corporate
        `[download] cert` must reach it rather than an `[install]` one.

        PIP_CERT is cleared first: the caller's environment legitimately wins over the
        re-assertion, so a host that exports one would otherwise make this assert on the
        ambient value and pass or fail for reasons that have nothing to do with sections.
        """
        listing = b"download.cert='/etc/dl.pem'\ninstall.cert='/etc/inst.pem'\n"
        env_without_cert = {k: v for k, v in os.environ.items() if k != "PIP_CERT"}
        with (
            mock.patch.object(ips, "_PINNED_PIP_CONFIG_LISTING", listing),
            mock.patch.dict(os.environ, env_without_cert, clear = True),
        ):
            env = ips._install_env_for_cmd(
                ["python", "-m", "pip", "download", "triton", "--index-url", "https://x/xpu"]
            )
        assert env["PIP_CERT"] == "/etc/dl.pem"

    def test_an_empty_inherited_value_is_not_an_override(self):
        ambient = ""
        """pip ignores an EMPTY environment value and falls through to the config file
        (verified with `pip config debug`), which the pinned branch has just switched off
        with devnull. Treating it as set would lose the operator's cert entirely. Only a
        truly empty value: a whitespace one is a value pip would use, not ours to
        second-guess."""
        with (
            mock.patch.object(
                ips, "_pinned_pip_config_overrides", lambda *a, **k: {"PIP_CERT": "/etc/corp.pem"}
            ),
            mock.patch.dict(os.environ, {"PIP_CERT": ambient}),
        ):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env["PIP_CERT"] == "/etc/corp.pem"

    def test_a_repeatable_option_accumulates_across_sections(self):
        """Measured on pip 26.2: `[global] only-binary = :all:` plus
        `[install] only-binary = numpy` still refuses an unrelated sdist, so pip
        accumulates the two rather than letting the command section replace the global
        one. Keeping only `numpy` would drop the operator's :all: policy on every pinned
        install, which is the control this change exists to preserve."""
        listing = b"global.only-binary=':all:'\ninstall.only-binary='numpy'\n"
        assert ips._parse_pinned_pip_config(listing, "install")["PIP_ONLY_BINARY"] == ":all:,numpy"
        # A scalar still takes the command section alone: two certs cannot be concatenated.
        certs = b"global.cert='/etc/g.pem'\ninstall.cert='/etc/i.pem'\n"
        assert ips._parse_pinned_pip_config(certs, "install")["PIP_CERT"] == "/etc/i.pem"

    def test_a_non_utf8_listing_is_decoded_the_way_the_child_wrote_it(self, monkeypatch):
        """A piped child encodes stdout with ITS locale encoding, which on Windows is the
        ANSI code page, not UTF-8. Getting that wrong does not merely lose the setting: it
        yields a cert path that exists nowhere, so pip fails the pinned install outright on
        exactly the corporate host the allowlist exists to serve. The read dictates the
        child's encoding (below) rather than sniffing it, since cp1252 bytes can form valid
        UTF-8; this covers the fallback, for a listing produced some other way."""
        path = "C:\\Soci\u00e9t\u00e9\\ca.pem"
        # repr, the way `pip config list` itself prints a value.
        listing = f"global.cert={path!r}\n".encode("cp1252")
        monkeypatch.setattr(ips.locale, "getpreferredencoding", lambda *a: "cp1252")
        assert ips._parse_pinned_pip_config(listing) == {"PIP_CERT": path}
        # UTF-8 is still tried first and strictly, so the POSIX case is untouched.
        assert (
            ips._decode_pip_output("cert='/etc/caf\u00e9/ca.pem'".encode())
            == "cert='/etc/caf\u00e9/ca.pem'"
        )
        # Undecodable under either codec is skipped, never fatal.
        assert ips._parse_pinned_pip_config(b"global.cert=\xff\xfe\x00") == {}

    @pytest.mark.reads_real_pip_config
    def test_a_wedged_pip_costs_the_run_one_budget_not_one_per_command(self, monkeypatch):
        """Failures are deliberately not memoised, so a transient miss cannot cost the
        operator their cert for the whole run. Unbounded, a HANG paid the timeout once per
        pinned command instead of once."""
        attempts = []

        def always_fails(cmd, **kwargs):
            attempts.append(kwargs.get("timeout"))
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

        monkeypatch.setattr(ips.subprocess, "run", always_fails)
        ips._PINNED_PIP_CONFIG_LISTING = None
        ips._PINNED_PIP_CONFIG_ATTEMPTS = 2
        for _ in range(25):
            assert ips._pinned_pip_config_overrides() == {}
        assert len(attempts) == 2, attempts
        assert set(attempts) == {ips._PINNED_PIP_CONFIG_TIMEOUT}
        # ...and small enough that the worst case is a wait, not a hang.
        assert ips._PINNED_PIP_CONFIG_TIMEOUT * 2 <= 60

    @pytest.mark.reads_real_pip_config
    def test_a_cheap_failure_does_not_spend_the_budget(self, monkeypatch):
        """A fresh venv has no pip for the first part of the run. That answer is instant,
        so budgeting it would mean the operator's cert is lost for the rest of the run the
        moment pip does appear."""
        calls = []

        def missing_then_present(cmd, **kwargs):
            calls.append(1)
            if len(calls) < 8:
                raise OSError("no pip yet")
            return subprocess.CompletedProcess(cmd, 0, b"global.cert='/etc/corp/ca.pem'\n")

        monkeypatch.setattr(ips.subprocess, "run", missing_then_present)
        ips._PINNED_PIP_CONFIG_LISTING = None
        ips._PINNED_PIP_CONFIG_ATTEMPTS = 2
        for _ in range(7):
            assert ips._pinned_pip_config_overrides() == {}
        assert ips._pinned_pip_config_overrides() == {"PIP_CERT": "/etc/corp/ca.pem"}

    @pytest.mark.reads_real_pip_config
    def test_the_read_dictates_the_child_encoding(self, monkeypatch):
        """Sniffing cannot recover an undictated encoding, so the child is told one."""
        seen = {}

        def fake_run(cmd, **kwargs):
            seen.update(kwargs.get("env") or {})
            return subprocess.CompletedProcess(cmd, 0, b"")

        monkeypatch.setattr(ips.subprocess, "run", fake_run)
        ips._PINNED_PIP_CONFIG_LISTING = None
        ips._PINNED_PIP_CONFIG_ATTEMPTS = 2
        ips._pinned_pip_config_overrides()
        assert seen.get("PYTHONIOENCODING") == "utf-8"

    def test_trusted_host_takes_section_precedence_instead(self):
        """Not every list key accumulates. Asked of pip 26.2's own parser with [global]
        and [install] both set, `trusted_hosts` comes back as the install value alone (an
        append option, assigned per section) while `format_control` holds both (a callback
        that mutates in place). Accumulating trusted-host would re-trust a host the
        install section had dropped, and that is a TLS decision."""
        hosts = b"global.trusted-host='global-a.corp'\ninstall.trusted-host='install-b.corp'\n"
        assert (
            ips._parse_pinned_pip_config(hosts, "install")["PIP_TRUSTED_HOST"] == "install-b.corp"
        )
        # Multiple hosts WITHIN the winning section are still space separated.
        many = rb"install.trusted-host='a.corp\nb.corp'"
        assert ips._parse_pinned_pip_config(many, "install")["PIP_TRUSTED_HOST"] == "a.corp b.corp"

    def test_a_reset_entry_keeps_its_order(self):
        """pip applies a repeatable option IN ORDER and `:none:` empties the set, so a
        re-add after a reset has to survive. Measured on pip 26.2: [global] a,b with
        [install] :none:,a still refuses a's sdist, and so does this concatenation, while
        deduplicating dropped the re-add and left `:none:` last, which empties the set and
        allowed the very build the operator forbade."""
        listing = (
            b"global.only-binary='probe-sdist,probe-wheel'\n"
            b"install.only-binary=':none:,probe-sdist'\n"
        )
        assert ips._parse_pinned_pip_config(listing, "install")["PIP_ONLY_BINARY"] == (
            "probe-sdist,probe-wheel,:none:,probe-sdist"
        )

    @pytest.mark.parametrize(
        "listing, expected",
        [
            # One value, passed through: collapsing whitespace breaks a real path.
            (
                b"global.cert='C:\\Program  Files\\ca.pem'",
                {"PIP_CERT": "C:\\Program  Files\\ca.pem"},
            ),
            (
                b"global.proxy='http://user:pw@proxy.corp:3128'",
                {"PIP_PROXY": "http://user:pw@proxy.corp:3128"},
            ),
            # Measured: rendered on ONE line with an escaped \n, which literal_eval undoes.
            (rb"global.trusted-host='a.corp\nb.corp'", {"PIP_TRUSTED_HOST": "a.corp b.corp"}),
            # COMMA separated, not whitespace. Verified: PIP_ONLY_BINARY="a,b" refuses both.
            (rb"global.only-binary='numpy\nscipy'", {"PIP_ONLY_BINARY": "numpy,scipy"}),
            (b"global.only-binary=':all:'", {"PIP_ONLY_BINARY": ":all:"}),
        ],
    )
    def test_each_key_is_joined_the_way_pip_reads_it(self, listing, expected):
        assert ips._parse_pinned_pip_config(listing) == expected

    def test_the_callers_own_environment_wins(self):
        """The re-assertion fills gaps; it never overwrites a variable the caller set."""
        with (
            mock.patch.object(
                ips,
                "_pinned_pip_config_overrides",
                lambda *a, **k: {"PIP_CERT": "/etc/ssl/corp.pem"},
            ),
            mock.patch.dict(os.environ, {"PIP_CERT": "/home/me/mine.pem"}),
        ):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env["PIP_CERT"] == "/home/me/mine.pem"

    @pytest.mark.parametrize(
        "configured, exported, expected",
        [
            ("numpy", "scipy", "numpy,scipy"),
            (":all:", ":none:", ":all:,:none:"),
            (":none:", ":all:", ":none:,:all:"),
        ],
    )
    def test_only_binary_from_the_environment_adds_to_the_file(
        self, configured, exported, expected
    ):
        """only-binary accumulates: measured on pip 26.2, the file's entries apply first and the
        environment's after them. Letting the variable replace the file lost the file's rule."""
        with (
            mock.patch.object(
                ips,
                "_pinned_pip_config_overrides",
                lambda *a, **k: {"PIP_ONLY_BINARY": configured},
            ),
            mock.patch.dict(os.environ, {"PIP_ONLY_BINARY": exported}),
        ):
            cmd, env = ips._pinned_cmd_and_env(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env["PIP_ONLY_BINARY"] == expected
        flags = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--only-binary"]
        assert flags == expected.split(",")

    def test_no_uv_env_var_is_invented_for_a_uv_toml_no_build(self):
        """Measured on uv 0.10.7: UV_NO_BUILD / UV_NO_BINARY / UV_ONLY_BINARY are not uv
        environment variables, so nothing here may pretend to carry a uv.toml no-build
        onto a pinned command."""
        src = STACK_SOURCE
        assert (
            "_uv_config_build_policy" not in src
        ), "re-asserting UV_NO_BUILD would promise a guarantee uv does not honour"
        # A NON-pinned uv command inherits everything, which is where source builds happen.
        with mock.patch.dict(os.environ, {"UV_EXCLUDE_NEWER": "2024-01-01T00:00:00Z"}):
            assert ips._install_env_for_cmd(["uv", "pip", "install", "-r", "extras.txt"]) is None

    @pytest.mark.parametrize(
        "cmd",
        [
            ["python", "-m", "pip", "uninstall", "-y", "install"],
            ["python", "-m", "pip", "config", "list"],
            [sys.executable, "/opt/tools/install.py", "--download"],
            ["uv", "pip", "install", "-r", "extras.txt"],
        ],
    )
    def test_only_a_real_pip_install_is_relaxed(self, cmd):
        """Keyed on the pip SUBCOMMAND, not on the word appearing anywhere in argv."""
        with mock.patch.dict(os.environ, self.HOSTILE):
            assert ips._install_env_for_cmd(cmd) is None

    def test_the_parent_environment_is_never_mutated(self):
        """The relaxation is a child-env override. Leaking it into os.environ would
        weaken the user's policy for their own later pip commands in this session."""
        with mock.patch.dict(os.environ, self.HOSTILE):
            ips._install_env_for_cmd(["python", "-m", "pip", "install", "x"])
            ips._install_env_for_cmd(["uv", "pip", "install", "x", "--index-url", "https://y"])
            assert os.environ["PIP_REQUIRE_HASHES"] == "1"
            assert os.environ["UV_NO_BUILD"] == "1"

    @pytest.mark.parametrize(
        "cmd, relaxed",
        [
            # os.path.basename keeps a backslash off-Windows: the naive-stem traps.
            ([r"C:\Python313\python.exe", "-m", "pip", "install", "x"], True),
            ([r"C:\venv\Scripts\pip.exe", "install", "x"], True),
            ([r"C:\venv\Scripts\pip3.13.exe", "download", "x"], True),
            ([r"C:\Program Files\venv\Scripts\python.exe", "-m", "pip", "wheel", "x"], True),
            (["/venv/bin/pip", "install", "x"], True),
            (["/venv/bin/pip3", "install", "x"], True),
            (["python", "-m", "pip", "-q", "install", "x"], True),
            (["python", "-m", "pip", "--isolated", "install", "x"], True),
            # ...and the ones that only LOOK like an install.
            (["python", "-m", "pip", "uninstall", "-y", "install"], False),
            (["python", "-m", "pip", "show", "wheel"], False),
            (["python", "-m", "pip", "check", "install.txt"], False),
            (["python", "/opt/tools/install.py"], False),
            ([r"C:\tools\installer.exe", "--download"], False),
        ],
    )
    def test_the_subcommand_test_reads_every_platform_spelling(self, cmd, relaxed):
        assert ips._is_pip_subcommand(cmd, ("install", "download", "wheel")) is relaxed

    @pytest.mark.parametrize(
        "cmd",
        [[], [""], ["uv"], ["python"], ["python", "-m"], ["python", "-m", "pip"]],
    )
    def test_a_degenerate_command_never_raises(self, cmd):
        """run() routes EVERY command through this, so a non-command must not IndexError."""
        assert ips._install_env_for_cmd(cmd) is None

    def test_the_pip_fallback_receives_the_relaxation(self):
        """End of the real path: uv fails, pip_install falls back through run(), and
        that pip command is the one #8530 died on."""
        seen: dict = {}

        def _fake_run(label, cmd, *a, **kw):
            seen["env"] = ips._install_env_for_cmd(cmd)
            seen["cmd"] = cmd
            # pip_install reads the result to decide whether to attempt recovery.
            return mock.Mock(returncode = 0, stdout = b"")

        with (
            mock.patch.object(ips, "USE_UV", True),
            mock.patch.object(ips, "subprocess") as sp,
            mock.patch.object(ips, "run", _fake_run),
            mock.patch.dict(os.environ, self.HOSTILE),
        ):
            sp.run.return_value = mock.Mock(returncode = 1, stdout = "")
            sp.PIPE, sp.STDOUT = -1, -2
            ips.pip_install("deps", *ips._sdist_only_build_args(*ips.SDIST_ONLY_PACKAGES))

        assert seen["env"]["PIP_REQUIRE_HASHES"] == "0"
        for name in ips.SDIST_ONLY_PACKAGES:
            assert name in seen["cmd"], "the fallback lost the source-build exemptions"


class TestPackageManagerPolicyOptOut:
    """UNSLOTH_RESPECT_PM_POLICY declines the #8530 relaxations, and only those.

    #8530 set parts of a hardened host's pip/uv policy aside for the installer's OWN
    dependency installs, because every requirements file we ship is unhashed and a few
    requirements have no wheel at any version. That is still the operator's control being
    overridden, and until now it could not be refused. An operator who would rather the
    install STOP than proceed unhashed sets this variable: each relaxation is withheld and
    the install then fails on the first step their policy forbids, which is the answer they
    asked for.

    What the opt-out must NOT reopen is #6898: a discovered uv.toml or a stale PIP_INDEX_URL
    outranking an explicit --index-url is a provenance hole, not a policy the operator chose
    on this command, so the pinned branch still strips the ADDITIVE index variables on both
    arms. The variables that CARRY policy rather than add a source stay, or every offline and
    every private-index install breaks for the operators this variable exists to serve.

    The tests below pair every opt-out assertion with the same call without the variable:
    a guard that is never asked answers "declined" for free, and this whole class would then
    pass over a predicate wired to nothing.
    """

    HOSTILE = {
        "PIP_REQUIRE_HASHES": "1",
        "UV_REQUIRE_HASHES": "1",
        "PIP_ONLY_BINARY": ":all:",
        "UV_NO_BUILD": "1",
        "UV_NO_BINARY": ":all:",
        "PIP_NO_BINARY": ":all:",
        "UV_EXCLUDE_NEWER": "2024-01-01T00:00:00Z",
    }

    # A pinned command: the branch #6898 hardened, and the one the opt-out treats specially.
    PINNED = ["uv", "pip", "install", "torch", "--index-url", "https://pin.example/whl"]
    # A plain pip install of our own unhashed requirements: the command #8530 died on.
    UNPINNED = ["python", "-m", "pip", "install", "-r", "extras.txt"]

    # Policy, not a source: uv's config file, a config-level no-index, and the two
    # find-links that are the ONLY permitted source once that no-index is in force.
    POLICY_CARRYING = {
        "UV_CONFIG_FILE": "/etc/uv/uv.toml",
        "PIP_NO_INDEX": "1",
        "PIP_FIND_LINKS": "/opt/wheels",
        "UV_FIND_LINKS": "/opt/wheels",
    }
    # Each of these ADDS a candidate source to, or redirects, the pinned command (#6898).
    ADDITIVE = {
        "UV_INDEX_URL": "https://mirror.internal/simple",
        "UV_EXTRA_INDEX_URL": "https://extra.internal/simple",
        "UV_DEFAULT_INDEX": "https://default.internal/simple",
        "PIP_INDEX_URL": "https://mirror.internal/simple",
        "PIP_EXTRA_INDEX_URL": "https://extra.internal/simple",
        "UV_TORCH_BACKEND": "cu128",
    }

    @staticmethod
    @contextlib.contextmanager
    def _environment(extra, opt_out = None):
        """``extra`` exported, with the opt-out set to ``opt_out`` or definitively absent.

        mock.patch.dict cannot express "not set", and a host or CI image that happens to
        export UNSLOTH_RESPECT_PM_POLICY would otherwise turn every control arm below into
        a second opt-out arm that agrees with anything. The pop is inside the patch, so the
        whole environment is restored on exit either way.
        """
        with mock.patch.dict(os.environ, extra):
            os.environ.pop(ips._POLICY_OPT_OUT_ENV, None)
            if opt_out is not None:
                os.environ[ips._POLICY_OPT_OUT_ENV] = opt_out
            yield

    def test_the_hash_relaxation_is_withheld(self):
        """PIP_REQUIRE_HASHES=0 is the whole of #8530's pip-side fix, so declining it is
        what makes the install stop where the operator's policy says it should."""
        with self._environment(self.HOSTILE, opt_out = "1"):
            assert ips._relaxed_pip_policy_env(self.UNPINNED) == {}
            # ...and with nothing to relax there is no child env at all: the command
            # inherits the operator's require-hashes and fails on it.
            assert ips._install_env_for_cmd(self.UNPINNED) is None
        with self._environment(self.HOSTILE):
            assert ips._relaxed_pip_policy_env(self.UNPINNED) == {"PIP_REQUIRE_HASHES": "0"}
            assert ips._install_env_for_cmd(self.UNPINNED)["PIP_REQUIRE_HASHES"] == "0"

    def test_the_source_build_exemptions_are_withheld(self):
        """--no-binary exists only to override a user-level no-build / only-binary, so
        declining to override it IS the opt-out. One gate covers both callers."""
        with self._environment({}, opt_out = "1"):
            assert ips._sdist_only_build_args("x", "y") == []
            assert ips._sdist_only_build_args(*ips.SDIST_ONLY_PACKAGES) == []
        with self._environment({}):
            assert ips._sdist_only_build_args("x", "y") == [
                "--no-binary",
                "x",
                "--no-binary",
                "y",
            ]

    def test_a_pinned_command_keeps_the_operators_policy_and_config(self):
        """The point of the opt-out on the branch #6898 owns.

        The policy variables survive, uv's config discovery stays ON and pip.conf stays
        readable, so a user uv.toml `[pip] require-hashes = true` still fails the pinned
        install -- UV_NO_CONFIG=1 would make it succeed, which would discard the control
        this variable promises to respect.
        """
        hostile = dict(self.HOSTILE, **self.POLICY_CARRYING)
        with self._environment(hostile, opt_out = "1"):
            ambient_config_file = os.environ.get("PIP_CONFIG_FILE")
            env = ips._install_env_for_cmd(self.PINNED)
        assert env is not None, "a pinned command still needs an explicit child env"
        for name, value in self.POLICY_CARRYING.items():
            assert env[name] == value, f"{name} carries policy; dropping it breaks the install"
        for name, value in self.HOSTILE.items():
            assert env[name] == value, f"{name} is the operator's control, not ours to drop"
        assert "UV_NO_CONFIG" not in env, "uv config discovery carries the policy being honoured"
        assert env.get("PIP_CONFIG_FILE") == ambient_config_file, (
            "devnull would hide the pip.conf whose security settings the opt-out promises "
            "to leave in force"
        )

    def test_a_pinned_command_still_strips_the_additive_index_variables(self):
        """#6898 is not the operator's to reopen by accident: the pin is itself a
        provenance control, and none of these were chosen for THIS command."""
        with self._environment(dict(self.HOSTILE, **self.ADDITIVE), opt_out = "1"):
            env = ips._install_env_for_cmd(self.PINNED)
        for name in self.ADDITIVE:
            assert name not in env, f"{name} can still outrank or widen the --index-url pin"

    def test_the_default_path_is_exactly_what_it_was_before_the_opt_out(self):
        """THE REGRESSION GUARD. With the variable unset, nothing about a pinned install
        may have moved.

        PIP_ONLY_BINARY stays in force because the pinned indexes serve wheels, so it costs
        the pin nothing and dropping it would let a compromised mirror run a source build
        the operator had forbidden. UV_NO_BUILD is untouched because it is not a uv
        environment variable at all (measured on uv 0.10.7) and so is not ours to drop.
        Adding either to the scrub list is silent: the install still succeeds, and the
        control is simply gone. This asserts on the values, so such an edit fails here.
        """
        with self._environment(self.HOSTILE):
            env = ips._install_env_for_cmd(self.PINNED)
        assert env is not None
        assert env["PIP_ONLY_BINARY"] == ":all:", (
            "PIP_ONLY_BINARY must stay in force on a pinned command: popping it lets a "
            "compromised mirror run a source build the operator forbade"
        )
        assert env["UV_NO_BUILD"] == "1", (
            "UV_NO_BUILD is inert for uv and not ours to drop; popping it changes the "
            "default path this opt-out was supposed to leave alone"
        )
        # ...and the rest of the default contract, unchanged.
        for name in ips._PM_HASH_ENV_VARS + ips._PM_FORCE_SOURCE_ENV_VARS:
            assert name not in env, f"{name} must still go on the default pinned path"
        assert env["UV_NO_CONFIG"] == "1"
        assert env["PIP_CONFIG_FILE"] == os.devnull

    def test_the_pip_config_is_never_read_back_on_the_opt_out_arm(self):
        """_pinned_pip_config_overrides exists to put back what PIP_CONFIG_FILE=devnull
        removed. The opt-out never sets devnull, so pip reads the real file itself and
        re-asserting would apply the same keys twice -- and only-binary ACCUMULATES, so
        `:all:` would come back as `:all:,:all:`."""
        calls = []

        def recorder(*args, **kwargs):
            calls.append((args, kwargs))
            return {"PIP_ONLY_BINARY": ":all:"}

        with mock.patch.object(ips, "_pinned_pip_config_overrides", recorder):
            with self._environment(self.HOSTILE, opt_out = "1"):
                ips._install_env_for_cmd(self.PINNED)
            assert calls == [], "the opt-out arm read a pip.conf it never switched off"
            # The recorder is wired to the name the code calls: the default arm reaches it.
            with self._environment(self.HOSTILE):
                ips._install_env_for_cmd(self.PINNED)
            assert calls, "the recorder is not wired up, so the assertion above proves nothing"

    def test_the_parent_environment_is_never_mutated(self):
        """As the relaxation itself: these are child-env decisions. Leaking either way
        would change the operator's own later pip commands in this session."""
        probe = dict(self.HOSTILE, **self.ADDITIVE, **self.POLICY_CARRYING)
        for opt_out in ("1", None):
            with self._environment(probe, opt_out = opt_out):
                before = dict(os.environ)
                ips._relaxed_pip_policy_env(self.UNPINNED)
                ips._sdist_only_build_args("x", "y")
                ips._install_env_for_cmd(self.UNPINNED)
                ips._install_env_for_cmd(self.PINNED)
                assert dict(os.environ) == before, f"os.environ moved with opt_out={opt_out!r}"

    @pytest.mark.parametrize(
        "value, enabled",
        [
            ("1", True),
            ("true", True),
            ("yes", True),
            ("on", True),
            ("TRUE", True),
            (" on ", True),
            ("0", False),
            ("no", False),
            ("off", False),
            ("false", False),
            ("", False),
            ("garbage", False),
            (None, False),  # unset
        ],
    )
    def test_the_boolish_set_is_uvs_and_an_unknown_value_is_off(self, value, enabled):
        """uv's own boolish set, borrowed deliberately so install.sh, install.ps1,
        setup.ps1 and this module give one answer for one variable.

        An ALLOWLIST: an unrecognised value reads as OFF on purpose, so a typo lands on the
        default path. Anything unrecognised meaning ON would fail closed for this one
        control while every other UNSLOTH_ variable in the tree fails open, and that
        inconsistency is worse than the typo.
        """
        with self._environment({}, opt_out = value):
            assert ips._respect_pm_policy() is enabled
            # Asserted through an observable too: a predicate nothing consults is inert.
            assert (ips._sdist_only_build_args("x") == []) is enabled

    def test_the_duplicate_metadata_repair_declines_before_touching_anything(
        self, monkeypatch, capsys
    ):
        """The repair rewrites METADATA and moves pip's leftover backups aside before it
        reinstalls, and that reinstall cannot satisfy require-hashes honestly: hashing the
        artifact we just fetched and handing the digest back approves it with itself.

        So it must decline with the detection done and NOTHING touched. A decline that has
        already rewritten a METADATA or moved a directory is not a decline: the `finally`
        would unwind a normal return, but a SIGKILL or a power loss cannot run a finally,
        and duplicate metadata is untidy where a half-repaired venv is not.
        """
        probes = {"unsloth": iter((["2026.8.12", "2026.8.15"],))}
        _shared_setup_2(monkeypatch, probes)

        backed_up: list = []
        taken: list = []
        rewritten: list = []
        monkeypatch.setattr(
            ips, "_rewrite_minimal_metadata", lambda *a, **k: rewritten.append(a) or True
        )
        monkeypatch.setattr(
            ips._QuarantinedMetadata, "back_up", lambda _self, p: backed_up.append(p) or True
        )
        monkeypatch.setattr(
            ips._QuarantinedMetadata, "take", lambda _self, paths: taken.append(paths) or True
        )

        def refuse(*_a, **_k):
            raise AssertionError("nothing may run once the repair has declined")

        monkeypatch.setattr(ips, "_stage_replacement", refuse)
        monkeypatch.setattr(ips, "_run_ok", refuse)
        monkeypatch.setattr(ips, "pip_install_try", refuse)

        with self._environment({}, opt_out = "1"):
            assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        assert (backed_up, taken, rewritten) == ([], [], []), "the decline touched the tree"
        err = capsys.readouterr().err
        # Naming the package proves a duplicate WAS detected, so the False above is the
        # decline and not the "nothing to repair" return that shares its value.
        assert ips._POLICY_OPT_OUT_ENV in err and "unsloth" in err
        assert "the install cannot continue" in err, (
            "both callers do `if not _repair_duplicate_core_metadata(...): return 1`, so this "
            "decline STOPS the install; a message implying it carried on would be a lie"
        )

    def test_the_repairs_false_return_is_the_documented_stop(self):
        """The decline's return value is the caller's abort signal, not an aside.

        Returning True instead would let write_manifest record a null version for a package
        whose metadata is still ambiguous: the installer reports success and every later
        check rejects the environment. Pinned here because the value is load-bearing and a
        future "be less disruptive" edit would look harmless.
        """
        import ast

        tree = ast.parse(STACK_SOURCE)
        callers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Call)
            and getattr(node.test.operand.func, "id", "") == "_repair_duplicate_core_metadata"
        ]
        assert callers, "no `if not _repair_duplicate_core_metadata(...)` caller found"
        for node in callers:
            assert any(
                isinstance(stmt, ast.Return)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value == 1
                for stmt in node.body
            ), "a caller stopped treating a False return as a failed install"

    @pytest.mark.parametrize(
        ("environment", "expected"),
        [
            # uv is standing in for pip, so a pip-expressed hash policy must reach it.
            ({"PIP_REQUIRE_HASHES": "1"}, {"UV_REQUIRE_HASHES": "1"}),
            ({"PIP_REQUIRE_HASHES": "true"}, {"UV_REQUIRE_HASHES": "1"}),
            # pip's own false spellings mean the control is off, so nothing is carried.
            ({"PIP_REQUIRE_HASHES": "0"}, {}),
            ({"PIP_REQUIRE_HASHES": "no"}, {}),
            ({}, {}),
            # An explicit uv value the operator set outranks a translation of their pip one.
            ({"PIP_REQUIRE_HASHES": "1", "UV_REQUIRE_HASHES": "0"}, {}),
        ],
    )
    def test_pip_expressed_policy_reaches_the_uv_run(self, environment, expected):
        """The installer choosing uv must not decide whether the operator's policy applies.

        A hardened host is far more likely to carry a pip hash requirement than a uv one, and
        uv reads no PIP_ variable, so without this the primary uv install proceeds unhashed on
        exactly the machine the opt-out was set for.
        """
        with self._environment(environment, opt_out = "1"):
            assert ips._pip_policy_as_uv_env() == expected
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "--index-url", "https://x", "torch"]
            )
        for name, value in expected.items():
            assert env is not None and env[name] == value

    def test_nothing_is_carried_in_the_other_direction(self):
        """There is no uv-to-pip translation, because under the opt-out there is no fallback.

        A partial carry reads as an absolute promise while covering less than it appears to:
        a uv.toml `[pip] require-hashes = true` is invisible to any translation this module
        could make, so the fallback is refused instead. Asserted so a later edit does not
        reintroduce the weaker design.
        """
        assert not hasattr(ips, "_uv_policy_as_pip_env")
        assert not hasattr(ips, "_UV_TO_PIP_POLICY")
        with self._environment({"UV_REQUIRE_HASHES": "1"}, opt_out = "1"):
            assert ips._relaxed_pip_policy_env([sys.executable, "-m", "pip", "install", "x"]) == {}

    def test_the_pip_fallback_is_refused_under_the_opt_out(self):
        """A uv failure must not be retried with a resolver that never heard the policy.

        Asserted structurally: the branch lives in pip_install()'s uv arm, which needs a real
        uv to execute. It must sit BEFORE the "falling back to pip" line and must exit.
        """
        import ast

        tree = ast.parse(STACK_SOURCE)
        fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "pip_install"
        )
        body = ast.get_source_segment(STACK_SOURCE, fn) or ""
        guard = body.find("if _respect_pm_policy():")
        fallback = body.find("falling back to pip")
        assert guard != -1, "pip_install no longer refuses the fallback under the opt-out"
        assert guard < fallback, "the refusal must precede the fallback"
        assert (
            "_report_failed_command" in body[guard:fallback]
        ), "the refusal must exit rather than fall through"

    def test_a_uv_command_is_not_given_pip_variables(self):
        """uv reads UV_ itself; restating them as PIP_ for a uv command would be noise."""
        with self._environment({"UV_REQUIRE_HASHES": "1"}, opt_out = "1"):
            assert ips._relaxed_pip_policy_env(["uv", "pip", "install", "torch"]) == {}


class TestProgressLineNotes:
    """_progress() leaves the cursor mid-line, so anything printed between two
    progress steps must close that line first. Before centralising this, a real
    install glued the torchao message onto the bar:
      deps  [=======-------------]  5/14  dependency overrides   torch 2.11...
    """

    def _render(
        self,
        emit,
        *,
        columns = "100",
        verbose = False,
        color = False,
    ) -> str:
        buf = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"COLUMNS": columns}),
            mock.patch.object(ips, "VERBOSE", verbose),
            # _HAS_COLOR is resolved once at import from the tty;
            # pinning it keeps the assertions valid under FORCE_COLOR=1 or `pytest -s` in a terminal.
            mock.patch.object(ips, "_HAS_COLOR", color),
            mock.patch.object(ips, "_TOTAL", 14),
            mock.patch.object(ips, "_STEP", 4),
            mock.patch.object(ips, "_PROGRESS_LINE_ACTIVE", False),
            contextlib.redirect_stdout(buf),
        ):
            emit()
        return buf.getvalue()

    def test_note_does_not_glue_onto_the_progress_bar(self):
        msg = "torch 2.11.0+cu130 detected -- installing torchao==0.17.0"
        out = self._render(lambda: (ips._progress("dependency overrides"), ips._note(msg)))
        bar_lines = [ln for ln in out.split("\n") if "5/14" in ln]
        assert len(bar_lines) == 1, f"expected one bar line, got {bar_lines!r}"
        assert msg not in bar_lines[0], f"note glued onto the bar: {bar_lines[0]!r}"
        assert any(ln.strip() == msg for ln in out.split("\n")), out

    def test_note_aligns_under_the_step_value_column(self):
        out = self._render(lambda: (ips._progress("dependency overrides"), ips._note("hello")))
        note_line = next(ln for ln in out.split("\n") if ln.strip() == "hello")
        assert len(note_line) - len(note_line.lstrip()) == ips._INDENT + ips._COL

    def test_note_without_an_active_bar_prints_on_its_own(self):
        out = self._render(lambda: ips._note("standalone"))
        assert out == f"{' ' * (ips._INDENT + ips._COL)}standalone\n"

    def test_step_still_closes_an_active_bar(self):
        """_step() handed its inline line-break logic to _end_progress_line(); it
        must keep breaking out of the bar."""
        out = self._render(
            lambda: (ips._progress("dependency overrides"), ips._step("deps", "installed"))
        )
        bar_lines = [ln for ln in out.split("\n") if "5/14" in ln]
        assert len(bar_lines) == 1 and "installed" not in bar_lines[0], out

    def test_progress_line_state_is_cleared(self):
        """A stale _PROGRESS_LINE_ACTIVE blank-lines the next note instead of
        closing a bar that is no longer open."""

        def emit():
            ips._progress("dependency overrides")
            ips._note("first")
            assert ips._PROGRESS_LINE_ACTIVE is False
            ips._note("second")

        out = self._render(emit)
        assert "\n\n" not in out, out

    def test_uv_fallback_warning_does_not_glue_onto_the_bar(self):
        """A real producer, not _note() directly: pip_install() warns on the uv
        fallback while the bar for its own step is still open. This path survived
        the first pass of the fix, which only converted '   message' call sites."""
        fake = mock.Mock(returncode = 1, stdout = "uv output")
        with (
            mock.patch.object(ips, "USE_UV", True),
            mock.patch.object(ips, "subprocess") as sp,
            mock.patch.object(ips, "run") as fallback,
        ):
            fallback.return_value = mock.Mock(returncode = 0, stdout = b"")
            sp.run.return_value = fake
            sp.PIPE, sp.STDOUT = -1, -2
            out = self._render(
                lambda: (ips._progress("studio deps"), ips.pip_install("Installing studio deps"))
            )
        assert fallback.called, "expected the pip fallback to run"
        bar_lines = [ln for ln in out.split("\n") if "5/14" in ln]
        assert len(bar_lines) == 1, f"expected one bar line, got {bar_lines!r}"
        assert "uv failed" not in bar_lines[0], f"warning glued onto the bar: {bar_lines[0]!r}"

    def test_no_bare_print_calls(self):
        """The line-close lives in _safe_print(), so a direct print() anywhere in
        the module silently reintroduces the glued-line bug."""
        src = Path(ips.__file__).read_text(encoding = "utf-8")
        tree = ast.parse(src)
        allowed = [
            (n.lineno, n.end_lineno)
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_safe_print"
        ]
        offenders = [
            n.func.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "print"
            and not any(lo <= n.func.lineno <= hi for lo, hi in allowed)
        ]
        assert not offenders, (
            f"install_python_stack.py:{offenders} call print() directly; "
            "use _safe_print() or _note() so an open progress bar line is closed first"
        )

    def test_no_direct_stdout_writes(self):
        """Copying _progress()'s sys.stdout.write idiom elsewhere would glue onto
        the bar again while still passing test_no_bare_print_calls."""
        tree = ast.parse(Path(ips.__file__).read_text(encoding = "utf-8"))
        allowed = [
            (n.lineno, n.end_lineno)
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in {"_progress", "_end_progress_line"}
        ]
        offenders = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"write", "flush"}
            and isinstance(n.func.value, ast.Attribute)
            and n.func.value.attr == "stdout"
            and not any(lo <= n.lineno <= hi for lo, hi in allowed)
        ]
        assert not offenders, (
            f"install_python_stack.py:{offenders} write to sys.stdout directly; "
            "only _progress()/_end_progress_line() may, everything else uses _safe_print()"
        )

    def test_no_message_starts_with_a_newline(self):
        """_safe_print() closes the bar itself now, so a message literal still
        opening with \\n emits a second newline and a blank line."""
        tree = ast.parse(Path(ips.__file__).read_text(encoding = "utf-8"))

        def leads_with_newline(node) -> bool:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value.startswith("\n")
            if isinstance(node, ast.JoinedStr) and node.values:
                return leads_with_newline(node.values[0])
            return False

        offenders = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in {"_safe_print", "_note"}
            and n.args
            and leads_with_newline(n.args[0])
        ]
        assert not offenders, (
            f"install_python_stack.py:{offenders} start a message with a newline; "
            "_safe_print() already closes the progress bar line, so this blank-lines the output"
        )

    def test_note_wraps_at_the_value_column_on_a_narrow_terminal(self):
        """Wrapping is all that separates _note() from _safe_print(), and no other
        test makes it wrap: every message fits one line at COLUMNS=100."""
        msg = (
            "AMD GPU detected but ROCm PyTorch could not be auto-installed. Manual install "
            "may be required. See: https://docs.unsloth.ai/get-started/install-and-update/amd"
        )
        out = self._render(lambda: ips._note(msg), columns = "60")
        lines = [ln for ln in out.split("\n") if ln.strip()]
        assert len(lines) > 1, f"expected the message to wrap, got {out!r}"
        for line in lines:
            assert len(line) - len(line.lstrip()) == ips._INDENT + ips._COL, repr(line)
        assert " ".join(ln.strip() for ln in lines) == msg
        # break_long_words = False keeps the URL clickable rather than splitting it.
        assert any("https://docs.unsloth.ai" in ln for ln in lines), out

    def test_note_falls_back_to_the_flat_indent_in_verbose_mode(self):
        """Verbose prints no bar and no step line, so the value column would indent
        under nothing while neighbouring messages sit at column 3."""
        out = self._render(lambda: ips._note("hello"), verbose = True)
        assert out == "   hello\n", repr(out)

    def test_note_still_aligns_when_colour_is_on(self):
        """The layout must survive a colour terminal, where every line carries ANSI
        codes that occupy no columns."""
        out = self._render(lambda: ips._note("hello"), color = True)
        assert "\033[" in out, "expected ANSI codes with _HAS_COLOR on"
        plain = re.sub(r"\033\[[0-9;]*m", "", out)
        assert plain == f"{' ' * (ips._INDENT + ips._COL)}hello\n", repr(plain)

    def test_safe_print_to_stderr_survives_a_closed_stdout(self):
        """_safe_print() touches stdout on every call now, so stderr-bound manifest
        errors must not die on an unrelated stdout failure."""
        closed = io.StringIO()
        closed.close()
        err = io.StringIO()
        with (
            mock.patch.object(ips, "VERBOSE", False),
            mock.patch.object(ips, "_PROGRESS_LINE_ACTIVE", True),
            mock.patch.object(ips.sys, "stdout", closed),
        ):
            ips._safe_print("error: boom", file = err)
        assert err.getvalue() == "error: boom\n"

    def test_install_entry_clears_a_stale_progress_line(self):
        """_PROGRESS_LINE_ACTIVE outlives an aborted run, and now every _safe_print()
        reads it, so a stale flag newlines the next run."""
        src = Path(ips.__file__).read_text(encoding = "utf-8")
        fn = next(
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef) and n.name == "install_python_stack"
        )
        assert any(
            isinstance(n, ast.Global) and "_PROGRESS_LINE_ACTIVE" in n.names for n in ast.walk(fn)
        ), "install_python_stack() must declare _PROGRESS_LINE_ACTIVE global"
        assert any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_PROGRESS_LINE_ACTIVE" for t in n.targets)
            for n in ast.walk(fn)
        ), "install_python_stack() must reset _PROGRESS_LINE_ACTIVE"


class TestBuildPipCmdUpgradeIntent:
    """pip has no --upgrade-package, so uv's flag must be translated, not dropped.

    Dropping it made the fallback a no-op on the update path: pip saw the named
    distributions as already satisfied, installed nothing, and the update
    reported success.
    """

    def test_update_path_keeps_the_upgrade_intent(self):
        cmd = ips._build_pip_cmd(
            ("--no-cache-dir", "--upgrade-package", "unsloth", "--upgrade-package", "unsloth-zoo")
        )
        assert "--upgrade" in cmd
        assert "unsloth" in cmd and "unsloth-zoo" in cmd
        assert "--upgrade-package" not in cmd, "pip does not understand the uv flag"

    def test_torch_is_not_dragged_along(self):
        # only-if-needed is pip's current default, but it is the load-bearing part: eager would
        # re-resolve the existing torch build.
        cmd = ips._build_pip_cmd(("--upgrade-package", "unsloth"))
        assert cmd[cmd.index("--upgrade-strategy") + 1] == "only-if-needed"

    def test_names_already_present_are_not_duplicated(self):
        cmd = ips._build_pip_cmd(("--no-deps", "--upgrade-package", "unsloth", "unsloth"))
        assert cmd.count("unsloth") == 1

    def test_commands_without_the_flag_are_untouched(self):
        cmd = ips._build_pip_cmd(("--no-cache-dir", "somepackage"))
        assert cmd == [sys.executable, "-m", "pip", "install", "--no-cache-dir", "somepackage"]


class TestDamagedCorePayloadRepair:
    """An upgrade of a distribution already at the wanted version installs
    nothing: uv audits it, pip calls it satisfied, and both read metadata a
    quarantine of the payload leaves intact."""

    def test_the_uv_command_targets_only_the_named_package(self):
        cmd = ips._build_uv_cmd(
            ("--no-cache-dir", "--no-deps", "--reinstall-package", "x", "--force-reinstall", "x")
        )
        assert "--reinstall-package" in cmd
        # Not --reinstall: that is the whole environment, torch included.
        assert "--reinstall" not in cmd
        assert "--no-deps" in cmd

    def test_a_plain_force_reinstall_still_becomes_uv_reinstall(self):
        assert "--reinstall" in ips._build_uv_cmd(("--force-reinstall", "x"))

    def test_the_pip_fallback_drops_the_uv_only_flag(self):
        cmd = ips._build_pip_cmd(
            ("--no-cache-dir", "--no-deps", "--reinstall-package", "x", "--force-reinstall", "x")
        )
        assert "--reinstall-package" not in cmd
        assert "--force-reinstall" in cmd and "--no-deps" in cmd
        assert cmd.count("x") == 1
        # It is not an upgrade request, so it must not acquire pip's upgrade flags.
        assert "--upgrade" not in cmd

    def test_a_damaged_package_is_reinstalled(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            ips.install_manifest,
            "damaged_payload_files",
            lambda name, **kwargs: ["gone"] if name == "unsloth-zoo" else [],
        )
        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda name: ["1.0"])
        monkeypatch.setattr(ips, "pip_install_try", lambda label, *args: calls.append(args))
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        ips._repair_damaged_core_payload(("unsloth", "unsloth-zoo"))
        assert len(calls) == 1
        assert "--reinstall-package" in calls[0] and calls[0][-1] == "unsloth-zoo"

    def test_a_healthy_tree_runs_nothing(self, monkeypatch):
        monkeypatch.setattr(ips.install_manifest, "damaged_payload_files", lambda *a, **k: [])
        monkeypatch.setattr(
            ips, "pip_install_try", lambda *a, **k: pytest.fail("nothing to repair")
        )
        ips._repair_damaged_core_payload(("unsloth", "unsloth-zoo"))

    def test_a_local_checkout_is_left_alone(self, monkeypatch):
        """Its core packages are an editable overlay, not an index install."""
        monkeypatch.setattr(
            ips.install_manifest,
            "damaged_payload_files",
            lambda *a, **k: pytest.fail("a local checkout must not even be scanned"),
        )
        ips._repair_damaged_core_payload(("unsloth",), local_repo = "/src/unsloth")

    def test_a_scan_that_raises_does_not_stop_the_install(self, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("unreadable")

        monkeypatch.setattr(ips.install_manifest, "damaged_payload_files", boom)
        monkeypatch.setattr(
            ips, "pip_install_try", lambda *a, **k: pytest.fail("nothing was proven damaged")
        )
        ips._repair_damaged_core_payload(("unsloth",))

    def test_a_repair_that_leaves_the_files_missing_fails(self, monkeypatch, capsys):
        """Otherwise the pass that follows audits the intact metadata as
        satisfied and write_manifest records a success nothing rechecks."""
        monkeypatch.setattr(
            ips.install_manifest, "damaged_payload_files", lambda name, **kwargs: ["gone"]
        )
        monkeypatch.setattr(ips, "pip_install_try", lambda label, *args: False)
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        assert ips._repair_damaged_core_payload(("unsloth",)) is False

    def test_a_reinstall_that_restored_the_files_is_a_success(self, monkeypatch):
        """Judged on the tree, not pip's exit code: uv and pip both exit non-zero
        for reasons that have nothing to do with the payload."""
        seen = {"n": 0}

        def scan(name, **kwargs):
            seen["n"] += 1
            return ["gone"] if seen["n"] == 1 else []

        monkeypatch.setattr(ips.install_manifest, "damaged_payload_files", scan)
        # Stubbed, or the presence check added beside the scan answers for the
        # host: a source checkout with no unsloth distribution installed would
        # fail this on the environment rather than on the code.
        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda name: ["1.0"])
        monkeypatch.setattr(ips, "pip_install_try", lambda label, *args: False)
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        assert ips._repair_damaged_core_payload(("unsloth",)) is True

    def test_an_absent_distribution_is_refused_after_the_core_phase(self, monkeypatch):
        """It has no RECORD to walk, so the scan alone reads it as undamaged.

        The install.sh handoff sets SKIP_STUDIO_BASE=1 and the core phase never
        runs, so this is the only place that would see unsloth-zoo gone before
        write_manifest records the environment as a finished install.
        """
        monkeypatch.setattr(ips.install_manifest, "damaged_payload_files", lambda *a, **k: [])
        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda name: [])
        monkeypatch.setattr(ips, "_safe_print", lambda *a, **k: None)
        assert ips._repair_damaged_core_payload(("unsloth",), require_present = True) is False
        # Off before the core phase: a fresh run has nothing installed yet.
        assert ips._repair_damaged_core_payload(("unsloth",)) is True

    def test_a_present_distribution_passes_the_presence_check(self, monkeypatch):
        monkeypatch.setattr(ips.install_manifest, "damaged_payload_files", lambda *a, **k: [])
        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda name: ["1.0"])
        assert ips._repair_damaged_core_payload(("unsloth",), require_present = True) is True

    def test_the_companion_is_only_for_the_default_install(self):
        assert ips._core_package_names("unsloth") == ("unsloth", "unsloth-zoo")
        assert ips._core_package_names("unsloth-nightly") == ("unsloth-nightly",)

    def test_the_default_is_recognised_however_it_is_spelled(self):
        """verify_install canonicalizes, and the two disagreeing meant the deep
        check scanned zoo, forced a pass, and no repair gate would touch it."""
        for spelling in ("Unsloth", "UNSLOTH", "UnSloth"):
            assert ips._core_package_names(spelling)[1:] == ("unsloth-zoo",), spelling
        # PEP 503 folds runs of -_.
        assert ips._core_package_names("uns_loth") == ("uns_loth",)

    def test_both_repair_sites_use_that_list(self):
        source = inspect.getsource(ips)
        assert (
            source.count("_repair_damaged_core_payload(\n        _core_package_names")
            + source.count("_repair_damaged_core_payload(_core_package_names")
            == 2
        )
        assert "_repair_damaged_core_payload((package_name" not in source

    def test_the_second_site_runs_before_the_manifest_is_written(self):
        source = inspect.getsource(ips)
        gate = source.rindex("_repair_damaged_core_payload(")
        assert gate < source.index("install_manifest.write_manifest(")

    def test_a_reinstall_that_removed_the_distribution_fails(self, monkeypatch):
        """--force-reinstall uninstalls before it installs, and a failure in
        between leaves nothing: an absent distribution has no RECORD, so the
        payload scan alone would call the repair a success."""
        scans = {"n": 0}

        def scan(name, **kwargs):
            scans["n"] += 1
            return ["gone"] if scans["n"] == 1 else []

        monkeypatch.setattr(ips.install_manifest, "damaged_payload_files", scan)
        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda name: [])
        monkeypatch.setattr(ips, "pip_install_try", lambda label, *args: True)
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips, "_safe_print", lambda *a, **k: None)
        assert ips._repair_damaged_core_payload(("unsloth",)) is False

    def test_the_caller_aborts_the_install(self):
        source = inspect.getsource(ips)
        assert "if not _repair_damaged_core_payload(" in source

    def test_the_repair_runs_before_the_core_phase(self):
        """After it, the upgrade would already have audited the damage as fine."""
        source = inspect.getsource(ips)
        repair = source.index("_repair_damaged_core_payload(_core_package_names")
        core = source.index("# 3. Core packages")
        assert repair < core


class TestDuplicateCoreMetadataRepair:
    def test_an_unrewritable_record_stops_the_repair_before_pip_runs(
        self, tmp_path, monkeypatch, capsys
    ):
        """The invariant that replaced quarantine-and-proceed.

        A record that cannot be made readable cannot be uninstalled by pip either.
        Moving it aside only hid it: pip removed the readable records, the
        quarantine was discarded once the reinstall succeeded, and whatever the
        quarantined release owned alone stayed importable while the repair
        reported success and deleted the directory that was the evidence.

        So nothing may run: no staging, no pip, and the tree is left as found.
        A non-UTF-8 METADATA also makes pip raise for the whole environment, so
        refusing before pip is what that used to need quarantining for.
        """
        malformed = tmp_path / "unsloth-2026.8.12.dist-info"
        malformed.mkdir()
        (malformed / "METADATA").write_bytes(b"\xff\xfe")
        (malformed / "RECORD").write_text("unsloth/gone.py,,\n")

        monkeypatch.setattr(
            ips.install_manifest, "installed_versions", lambda _n: ["", "2026.8.15"]
        )
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _n: [malformed])
        monkeypatch.setattr(ips.install_manifest, "pip_backup_metadata_paths", lambda _n: [])
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        # The rewrite is what normally saves this record; deny it to reach the branch.
        monkeypatch.setattr(ips, "_rewrite_minimal_metadata", lambda *a, **k: False)

        def refuse(*_a, **_k):
            raise AssertionError("nothing may run once a record is unusable")

        monkeypatch.setattr(ips, "_stage_replacement", refuse)
        monkeypatch.setattr(ips, "_run_ok", refuse)
        monkeypatch.setattr(ips, "pip_install_try", refuse)

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        assert "cannot be read or rewritten" in capsys.readouterr().err
        assert malformed.is_dir()
        assert (malformed / "METADATA").read_bytes() == b"\xff\xfe"

    def test_an_unrecorded_stale_record_fails_closed_even_beside_a_good_one(
        self, tmp_path, monkeypatch, capsys
    ):
        """No RECORD means nothing knows which files that release owned, which is
        why _rewrite_minimal_metadata fails closed. Waiting for record_count to
        reach zero missed the case where another record survives: pip uninstalls
        only that one, the quarantine is discarded on success, and whatever the
        older release owned alone stays importable while the directory that was
        the evidence is deleted for good.
        """
        unreadable = tmp_path / "unsloth-2026.8.12.dist-info"
        unreadable.mkdir()
        (unreadable / "METADATA").write_bytes(b"\xff\xfe")  # no RECORD beside it

        monkeypatch.setattr(
            ips.install_manifest, "installed_versions", lambda _n: ["", "2026.8.15"]
        )
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _n: [unreadable])
        monkeypatch.setattr(ips.install_manifest, "pip_backup_metadata_paths", lambda _n: [])
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)

        def fake_stage(*_args, **_kwargs):
            raise AssertionError("nothing may be staged once a record is unusable")

        monkeypatch.setattr(ips, "_stage_replacement", fake_stage)

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        assert "cannot be read or rewritten" in capsys.readouterr().err
        # The evidence stays on disk so a later run can still see the conflict.
        assert unreadable.is_dir()

    def test_pips_tilde_backup_is_moved_aside_so_the_loop_can_converge(self, tmp_path, monkeypatch):
        """The commonest real conflict: pip renamed the outgoing distribution to a
        `~` sibling and was killed. Its METADATA still says Name: unsloth so it
        counts as a duplicate, but `pip uninstall unsloth` logs "Ignoring invalid
        distribution" and skips it, so the loop never converges and the repair
        fails on every future run. Verified in a real venv before this fix.
        """
        backup = _shared_setup_3(tmp_path)
        # Two records; one once the backup is aside; none after the uninstall; then the reinstalled one for the final
        # convergence probe.
        probes = iter((["2026.8.12", "2026.8.15"], ["2026.8.15"], [], ["2026.8.15"]))
        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _name: next(probes))
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _name: [])
        monkeypatch.setattr(
            ips.install_manifest, "pip_backup_metadata_paths", lambda _name: [backup]
        )
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _name: "/staged")
        monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: True)

        def record_run(_label, _cmd):
            assert not backup.exists()
            return True

        monkeypatch.setattr(ips, "_run_ok", record_run)

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is True
        assert not backup.exists()

    def test_a_sole_tilde_backup_is_repaired_by_a_fresh_install(self, tmp_path, monkeypatch):
        """pip killed after the rename but before the replacement landed leaves the
        backup alone: one readable version, so a version count sees nothing wrong
        while the package is genuinely unimportable. Once the backup is aside there
        is no payload left to lay a replacement over, so installing fresh is right
        and refusing would abort the installer on a trivially fixable state.
        """
        backup = _shared_setup_3(tmp_path)
        probes = iter((["2026.8.12"], [], ["2026.8.15"]))
        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _name: next(probes))
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _name: [])
        monkeypatch.setattr(
            ips.install_manifest, "pip_backup_metadata_paths", lambda _name: [backup]
        )
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _name: "/staged")
        installs = []
        monkeypatch.setattr(
            ips, "pip_install_try", lambda label, *a, **k: installs.append(label) or True
        )
        monkeypatch.setattr(
            ips, "_run_ok", lambda *_a: pytest.fail("nothing is left for pip to uninstall")
        )

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is True
        assert len(installs) == 1
        assert not backup.exists()

    def test_every_duplicate_record_is_uninstalled_before_reinstall(self, monkeypatch):
        probes = {
            "unsloth": iter(
                (
                    ["2026.8.12", "2026.8.15"],
                    ["2026.8.15"],
                    [],
                    ["2026.8.15"],
                )
            ),
            "unsloth-zoo": iter((["2026.8.10"],)),
        }
        installs = []
        invalidations = []
        commands = []

        _shared_setup_4(monkeypatch, probes)
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: invalidations.append(True))
        monkeypatch.setattr(
            ips,
            "_run_ok",
            lambda label, cmd: commands.append((label, cmd)) or True,
        )
        monkeypatch.setattr(ips, "_stage_replacement", lambda _name: "/staged")
        monkeypatch.setattr(
            ips,
            "pip_install_try",
            lambda label, *args, **kwargs: installs.append((label, args, kwargs)) or True,
        )

        assert ips._repair_duplicate_core_metadata(("unsloth", "unsloth-zoo")) is True
        assert [command for _label, command in commands] == [
            [sys.executable, "-m", "pip", "uninstall", "-y", "unsloth"],
            [sys.executable, "-m", "pip", "uninstall", "-y", "unsloth"],
        ]
        assert len(installs) == 1
        assert installs[0][1] == (
            "--no-cache-dir",
            "--no-deps",
            "--force-reinstall",
            "--no-index",
            "--find-links",
            "/staged",
            "unsloth",
        )
        assert len(invalidations) == 3

    def test_repair_fails_when_uninstall_does_not_remove_a_record(self, monkeypatch, capsys):
        probes = iter(
            (
                ["2026.8.12", "2026.8.15"],
                ["2026.8.12", "2026.8.15"],
            )
        )
        monkeypatch.setattr(
            ips.install_manifest,
            "installed_versions",
            lambda _name: next(probes),
        )
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: True)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _name: "/staged")
        installs = []
        monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: installs.append((a, k)))

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        assert installs == []
        assert "could not remove every metadata record" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "duplicate, overlay_args",
        [
            ("unsloth", ("--no-cache-dir", "--no-deps", "-e", "/src/unsloth")),
            (
                "unsloth-zoo",
                (
                    "--no-cache-dir",
                    "--no-deps",
                    "--force-reinstall",
                    "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo",
                ),
            ),
        ],
    )
    def test_local_repair_restores_only_the_source_it_replaced(
        self, monkeypatch, duplicate, overlay_args
    ):
        probes = {
            name: iter((["old", "new"], ["new"], [], ["new"]) if name == duplicate else (["new"],))
            for name in ("unsloth", "unsloth-zoo")
        }
        installs = []

        _shared_setup_4(monkeypatch, probes)
        _shared_setup_1(installs, monkeypatch)

        assert ips._repair_duplicate_core_metadata(
            ("unsloth", "unsloth-zoo"), local_repo = "/src/unsloth"
        )
        assert len(installs) == 1
        assert installs[0][1] == overlay_args
        assert installs[0][2]["constrain"] is False

    def test_install_pass_hands_local_provenance_to_duplicate_repair(self):
        source = inspect.getsource(ips.install_python_stack)
        assert "local_repo=local_repo" in source.replace(" ", "")

    def test_local_repair_reinstalls_a_custom_package_from_its_normal_source(self, monkeypatch):
        probes = iter((["old", "new"], ["new"], [], ["new"]))
        installs = []

        monkeypatch.setattr(
            ips.install_manifest,
            "installed_versions",
            lambda _name: next(probes),
        )
        _shared_setup_1(installs, monkeypatch)

        assert ips._repair_duplicate_core_metadata(("custom-package",), local_repo = "/src/unsloth")
        assert len(installs) == 1
        assert installs[0][1] == (
            "--no-cache-dir",
            "--no-deps",
            "--force-reinstall",
            "--no-index",
            "--find-links",
            "/staged",
            "custom-package",
        )

    def test_ci_repair_restores_only_the_candidate_unsloth_checkout(self, monkeypatch):
        probes = {
            "unsloth": iter((["old", "new"], ["new"], [], ["new"])),
            "unsloth-zoo": iter((["new"],)),
        }
        installs = []

        _shared_setup_4(monkeypatch, probes)
        _shared_setup_1(installs, monkeypatch)

        assert ips._repair_duplicate_core_metadata(
            ("unsloth", "unsloth-zoo"), ci_source_overlay = "/src/candidate"
        )
        assert len(installs) == 1
        assert installs[0][1] == ("--no-cache-dir", "--no-deps", "-e", "/src/candidate")
        assert installs[0][2]["constrain"] is False

    def test_the_repair_runs_again_before_the_manifest_is_written(self):
        """The first pass runs before the core packages are installed, so an
        upgrade that leaves a superseded record behind would survive it and
        write_manifest would record a null version under a successful exit."""
        source = inspect.getsource(ips.install_python_stack)
        first = source.index("_repair_duplicate_core_metadata")
        second = source.index("_repair_duplicate_core_metadata", first + 1)
        manifest = source.index("write_manifest(")
        assert first < second < manifest, (
            "the duplicate-metadata repair must run again after the core-package "
            "install and before the manifest is written"
        )

    def test_staging_relaxes_pip_hash_enforcement(self):
        """require-hashes applies to pip wheel exactly as it does to pip install
        (measured on pip 26.2: an unpinned name is refused before anything is
        built), so a hardened machine could never stage a replacement and the
        repair would abort on the very conflict it exists to remove."""
        env = ips._install_env_for_cmd(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", "/staged", "unsloth"]
        )
        assert env is not None and env.get("PIP_REQUIRE_HASHES") == "0"

    def _uv_only(self, monkeypatch):
        monkeypatch.setattr(ips, "USE_UV", True)
        for var in (
            "PIP_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
            "PIP_FIND_LINKS",
            "UV_INDEX",
            "UV_EXTRA_INDEX_URL",
            "UV_INDEX_URL",
            "UV_DEFAULT_INDEX",
            "UV_FIND_LINKS",
            "UV_EXCLUDE_NEWER",
            "UV_INDEX_STRATEGY",
            "UV_NO_BINARY",
            "UV_ONLY_BINARY",
            "PIP_NO_BINARY",
            "PIP_ONLY_BINARY",
            "UV_KEYRING_PROVIDER",
            "PIP_KEYRING_PROVIDER",
        ):
            monkeypatch.delenv(var, raising = False)

    UV_COMPILE_OUT = (
        b"# This file was autogenerated by uv via the following command:\n"
        b"#    uv pip compile --no-deps --emit-index-url -\n"
        b"--index-url https://pypi.org/simple\n"
        b"--extra-index-url https://mirror.corp/simple\n"
        b"--find-links /opt/wheels\n"
        b"\n"
        b"unsloth-zoo==2026.8.15\n"
        b"    # from https://mirror.corp/simple\n"
    )

    def _uv_plan(
        self,
        monkeypatch,
        stdout = None,
        returncode = 0,
    ):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if cmd[:3] == ["uv", "pip", "compile"]:
                return types.SimpleNamespace(
                    returncode = returncode,
                    stdout = self.UV_COMPILE_OUT if stdout is None else stdout,
                    stderr = b"",
                )
            return types.SimpleNamespace(returncode = 1, stdout = b"")

        monkeypatch.setattr(ips.subprocess, "run", fake_run)
        return calls

    def test_the_replacement_is_the_release_and_index_uv_resolved(self, monkeypatch):
        """Staging must run pip, because uv has no wheel subcommand, and uv's index
        configuration cannot be reconstructed from the environment: uv also reads
        uv.toml, pyproject [tool.uv], a user config and UV_CONFIG_FILE, applies an
        implicit PyPI default, and resolves under an index-strategy pip has no
        equivalent for. So uv is asked, and its answer is reproduced verbatim."""
        self._uv_only(monkeypatch)
        monkeypatch.setattr(ips, "USE_UV", True)
        self._uv_plan(monkeypatch)
        requirement, overrides, _options = ips._uv_staging_plan("unsloth_zoo")
        assert requirement == "unsloth-zoo==2026.8.15"
        # The annotation names the index the package actually came from, which is the one to reproduce -- not
        # --index-url, which is only uv's default.
        assert overrides["PIP_INDEX_URL"] == "https://mirror.corp/simple"
        assert overrides["PIP_EXTRA_INDEX_URL"] == ""
        assert overrides["PIP_FIND_LINKS"] == "/opt/wheels"

    def test_the_plan_asks_uv_about_this_interpreter(self, monkeypatch):
        """Markers and ABI tags come from the interpreter being repaired, not from
        whichever one uv would discover on its own."""
        self._uv_only(monkeypatch)
        calls = self._uv_plan(monkeypatch)
        ips._uv_staging_plan("unsloth-zoo")
        cmd = calls[0][0]
        assert cmd[cmd.index("--python") + 1] == sys.executable
        assert calls[0][1]["input"] == b"unsloth-zoo"

    def test_a_falling_index_aborts_the_repair_rather_than_substituting(self, monkeypatch, capsys):
        """uv fails the compile outright when a higher-priority index is unreachable,
        which is the behaviour first-index exists to give: a public release must not
        stand in for a private one just because the private mirror was down."""
        self._uv_only(monkeypatch)
        monkeypatch.setattr(ips, "USE_UV", True)
        self._uv_plan(monkeypatch, returncode = 1)
        assert ips._stage_replacement("unsloth-zoo") is None
        assert "cannot be preserved" in capsys.readouterr().err

    def test_offline_uv_leaves_the_install_alone(self, monkeypatch, capsys):
        """UV_OFFLINE forbids network access and pip has no offline mode, so the
        repair would have to break the policy to proceed."""
        self._uv_only(monkeypatch)
        monkeypatch.setattr(ips, "USE_UV", True)
        monkeypatch.setenv("UV_OFFLINE", "1")

        def fake_run(*args, **kwargs):
            raise AssertionError("nothing may run while uv is offline")

        monkeypatch.setattr(ips.subprocess, "run", fake_run)
        assert ips._stage_replacement("unsloth-zoo") is None
        assert "UV_OFFLINE" in capsys.readouterr().err

    @pytest.mark.parametrize("value", ("", "0", "false"))
    def test_an_unset_or_disabled_offline_flag_is_not_offline(self, monkeypatch, value):
        self._uv_only(monkeypatch)
        monkeypatch.setenv("UV_OFFLINE", value)
        assert ips._uv_is_offline() is False

    def test_the_plan_is_skipped_when_uv_is_not_the_package_manager(self, monkeypatch):
        """Plain pip already reads its own configuration, so there is nothing to
        translate and no uv to ask."""
        self._uv_only(monkeypatch)
        monkeypatch.setattr(ips, "USE_UV", False)
        calls = self._uv_plan(monkeypatch)
        assert ips._stage_replacement("unsloth-zoo") is None
        assert all(cmd[:1] != ["uv"] for cmd, _ in calls)
        assert calls[0][0][-1] == "unsloth-zoo"

    @pytest.mark.parametrize(
        "requirement",
        (
            "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo",
            "https://example.invalid/unsloth_zoo-1.0-py3-none-any.whl",
        ),
    )
    def test_a_direct_reference_is_staged_as_written(self, monkeypatch, requirement):
        """The overlay paths hand staging a git URL or a checkout, not a bare name.
        Such a requirement is its own provenance -- no index chose it -- and uv
        appends the resolved commit to what it emits, so asking uv would compare a
        bare spec against a pinned one and never match, aborting every local
        duplicate-zoo repair before it started."""
        self._uv_only(monkeypatch)
        monkeypatch.setattr(ips, "USE_UV", True)
        calls = self._uv_plan(monkeypatch)
        assert ips._stage_replacement(requirement) is None
        assert all(cmd[:3] != ["uv", "pip", "compile"] for cmd, _ in calls)
        assert calls[0][0][-1] == requirement

    def test_a_local_checkout_is_a_direct_reference(self, tmp_path):
        assert ips._is_direct_reference(str(tmp_path)) is True
        assert ips._is_direct_reference("unsloth-zoo") is False

    def test_replaying_uv_replaces_pips_candidate_sources(self, monkeypatch):
        """Replaying uv's answer means replacing pip's sources, not adding to them.
        An inherited PIP_NO_INDEX blocks the index uv picked, and an inherited extra
        index or find-links directory can satisfy the same version from somewhere uv
        never looked, which is the provenance swap this path exists to stop."""
        self._uv_only(monkeypatch)
        monkeypatch.setattr(ips, "USE_UV", True)
        for var, value in (
            ("PIP_NO_INDEX", "1"),
            ("PIP_EXTRA_INDEX_URL", "https://elsewhere/simple"),
            ("PIP_FIND_LINKS", "/tmp/stale-wheels"),
        ):
            monkeypatch.setenv(var, value)
        calls = self._uv_plan(monkeypatch)
        assert ips._stage_replacement("unsloth-zoo") is None
        env = calls[-1][1]["env"]
        # Measured on pip 26.2: an empty value reads as unset.
        assert env["PIP_NO_INDEX"] == ""
        assert env["PIP_EXTRA_INDEX_URL"] == ""
        assert env["PIP_FIND_LINKS"] == "/opt/wheels"
        assert env["PIP_INDEX_URL"] == "https://mirror.corp/simple"
        # pip.conf carries the same three settings, so it is replaced by a copy of itself with only those removed.
        assert env["PIP_CONFIG_FILE"].endswith("pip.conf")
        assert env["PIP_CONFIG_FILE"] != os.devnull

    def test_no_find_links_survives_when_uv_emitted_none(self, monkeypatch):
        self._uv_only(monkeypatch)
        monkeypatch.setenv("PIP_FIND_LINKS", "/tmp/stale-wheels")
        self._uv_plan(monkeypatch, stdout = b"unsloth-zoo==1.0\n    # from https://m/s\n")
        _requirement, overrides, _options = ips._uv_staging_plan("unsloth-zoo")
        assert overrides["PIP_FIND_LINKS"] == ""

    def test_pip_transport_settings_survive_the_source_replacement(self, tmp_path, monkeypatch):
        """Dropping pip.conf wholesale would take proxy, cert, client-cert and
        trusted-host with it, and those are how a private index is reached at all, so
        uv would resolve and pip would then fail to fetch. The four source keys are
        removed and everything else is written back."""
        listing = (
            b"global.cert='/etc/ssl/corp.pem'\n"
            b"global.proxy='http://proxy.corp:8080'\n"
            b"global.trusted-host='\\na.corp\\nb.corp'\n"
            b"global.index-url='https://bogus/simple'\n"
            b"global.extra-index-url='https://elsewhere/simple'\n"
            b"global.find-links='/tmp/stale'\n"
            b"global.no-index='true'\n"
            b"install.no-binary='numpy'\n"
            b":env:.config-file='/etc/pip.conf'\n"
        )
        monkeypatch.setattr(
            ips.subprocess,
            "run",
            lambda *a, **k: types.SimpleNamespace(returncode = 0, stdout = listing),
        )
        written = Path(ips._pip_config_without_sources(str(tmp_path))).read_text()
        assert "proxy = http://proxy.corp:8080" in written
        assert "cert = /etc/ssl/corp.pem" in written
        # A multi-value setting is spelled back as an indented continuation.
        assert "trusted-host =\n    a.corp\n    b.corp" in written
        assert "[install]" in written and "no-binary = numpy" in written
        for dropped in ("index-url", "extra-index-url", "find-links", "no-index"):
            assert dropped not in written, f"{dropped} must not survive"
        # :env: entries come from the environment, which is overridden separately.
        assert "config-file" not in written

    def test_an_unreadable_pip_config_yields_an_empty_one(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ips.subprocess,
            "run",
            lambda *a, **k: types.SimpleNamespace(returncode = 1, stdout = b""),
        )
        assert Path(ips._pip_config_without_sources(str(tmp_path))).read_text() == ""

    def test_uv_artifact_policy_is_replayed_from_its_configuration(self, monkeypatch):
        """A no-binary or only-binary rule would otherwise change the artifact type
        during the repair: a wheel downloaded under a no-binary rule, or an sdist
        built under an only-binary one."""
        self._uv_only(monkeypatch)
        self._uv_plan(
            monkeypatch,
            stdout = b"--only-binary :all:\nunsloth-zoo==1.0\n    # from https://m/s\n",
        )
        _requirement, _overrides, options = ips._uv_staging_plan("unsloth-zoo")
        assert options == ["--only-binary", ":all:"]

    def test_the_env_spelling_of_the_artifact_policy_is_translated(self, monkeypatch):
        """Measured on uv 0.10.7: --emit-build-options surfaces the policy from
        uv.toml but not the environment-variable spelling, so that half is
        translated by hand."""
        self._uv_only(monkeypatch)
        monkeypatch.setenv("UV_ONLY_BINARY", ":all:")
        monkeypatch.delenv("PIP_ONLY_BINARY", raising = False)
        self._uv_plan(monkeypatch)
        _requirement, overrides, options = ips._uv_staging_plan("unsloth-zoo")
        assert overrides["PIP_ONLY_BINARY"] == ":all:"
        assert options == []

    def test_an_explicit_pip_artifact_policy_is_left_alone(self, monkeypatch):
        self._uv_only(monkeypatch)
        monkeypatch.setenv("UV_ONLY_BINARY", ":all:")
        monkeypatch.setenv("PIP_ONLY_BINARY", "numpy")
        self._uv_plan(monkeypatch)
        _requirement, overrides, _options = ips._uv_staging_plan("unsloth-zoo")
        assert overrides.get("PIP_ONLY_BINARY") is None

    def test_the_keyring_provider_is_translated(self, monkeypatch):
        """uv reaches an authenticated index through the keyring CLI (uv 0.10.7:
        `--keyring-provider subprocess` uses the `keyring` command). Carrying only
        the URL leaves pip unable to fetch what uv just resolved, so every repair
        on a private index aborts. pip accepts the same two values.
        """
        self._uv_only(monkeypatch)
        monkeypatch.setenv("UV_KEYRING_PROVIDER", "subprocess")
        monkeypatch.delenv("PIP_KEYRING_PROVIDER", raising = False)
        self._uv_plan(monkeypatch)
        _requirement, overrides, _options = ips._uv_staging_plan("unsloth-zoo")
        assert overrides["PIP_KEYRING_PROVIDER"] == "subprocess"

    def test_an_explicit_pip_keyring_provider_is_left_alone(self, monkeypatch):
        self._uv_only(monkeypatch)
        monkeypatch.setenv("UV_KEYRING_PROVIDER", "subprocess")
        monkeypatch.setenv("PIP_KEYRING_PROVIDER", "import")
        self._uv_plan(monkeypatch)
        _requirement, overrides, _options = ips._uv_staging_plan("unsloth-zoo")
        assert overrides.get("PIP_KEYRING_PROVIDER") is None

    def test_the_staging_command_carries_the_build_options(self, monkeypatch):
        self._uv_only(monkeypatch)
        monkeypatch.setattr(ips, "USE_UV", True)
        calls = self._uv_plan(
            monkeypatch,
            stdout = b"--only-binary :all:\nunsloth-zoo==1.0\n    # from https://m/s\n",
        )
        assert ips._stage_replacement("unsloth-zoo") is None
        cmd = calls[-1][0]
        assert cmd[cmd.index("--only-binary") + 1] == ":all:"

    @pytest.mark.parametrize(
        "stdout",
        [
            # uv 0.10.7 strips userinfo from the `# from` annotation; trusting it hands pip an
            # unauthenticated private index URL, which 401s and aborts the repair.
            pytest.param(
                b"--index-url https://user:secret@private.corp/simple\n"
                b"unsloth-zoo==1.0\n"
                b"    # from https://private.corp/simple\n",
                id = "the_annotated_index_is_recovered_with_its_credentials",
            ),
            # uv leaves --index-url as the public default, so only --index carries the credentials.
            pytest.param(
                b"--index-url https://pypi.org/simple\n"
                b"--extra-index-url https://user:secret@private.corp/simple\n"
                b"unsloth-zoo==1.0\n"
                b"    # from https://private.corp/simple\n",
                id = "an_authenticated_extra_index_is_recovered_too",
            ),
            pytest.param(
                b"--index-url https://private.corp/simple\n"
                b"--extra-index-url https://user:secret@private.corp/simple\n"
                b"unsloth-zoo==1.0\n"
                b"    # from https://private.corp/simple\n",
                id = "the_credentialed_form_wins_over_a_bare_duplicate",
            ),
        ],
    )
    def test_duplicate_core_metadata_repair_cases(self, monkeypatch, stdout):
        self._uv_only(monkeypatch)
        self._uv_plan(monkeypatch, stdout = stdout)
        _requirement, overrides, _options = ips._uv_staging_plan("unsloth-zoo")
        assert overrides["PIP_INDEX_URL"] == "https://user:secret@private.corp/simple"

    @pytest.mark.parametrize(
        "url, bare",
        (
            ("https://u:p@h/simple", "https://h/simple"),
            ("https://h/simple", "https://h/simple"),
            ("https://h/simple?a=b", "https://h/simple?a=b"),
            ("/local/dir", "/local/dir"),
        ),
    )
    def test_userinfo_is_stripped_without_disturbing_the_rest(self, url, bare):
        assert ips._strip_userinfo(url) == bare

    def test_offline_still_stages_a_local_checkout(self, tmp_path, monkeypatch):
        """--local hands the repair a checkout on disk. It needs no network, so
        UV_OFFLINE has nothing to say about it, and refusing left the conflict in
        place and failed the update for no reason."""
        self._uv_only(monkeypatch)
        monkeypatch.setattr(ips, "USE_UV", True)
        monkeypatch.setenv("UV_OFFLINE", "1")
        calls = self._uv_plan(monkeypatch)
        assert ips._stage_replacement(str(tmp_path)) is None
        assert calls and all(cmd[:3] != ["uv", "pip", "compile"] for cmd, _ in calls)
        assert calls[-1][0][-1] == str(tmp_path)

    def test_offline_local_staging_never_reaches_the_index(self, tmp_path, monkeypatch):
        """The checkout needs no network, but pip builds it in an isolated
        environment and fetches the build backend for that, which UV_OFFLINE does not
        reach. Measured: an isolated build of a local project with no index reachable
        fails at installing build dependencies, and this repository pins its build
        requirements exactly, so they would be fetched unless already cached."""
        self._uv_only(monkeypatch)
        monkeypatch.setattr(ips, "USE_UV", True)
        monkeypatch.setenv("UV_OFFLINE", "1")
        calls = self._uv_plan(monkeypatch)
        assert ips._stage_replacement(str(tmp_path)) is None
        cmd, kwargs = calls[-1]
        assert "--no-build-isolation" in cmd
        assert kwargs["env"]["PIP_NO_INDEX"] == "1"

    def test_online_local_staging_keeps_build_isolation(self, tmp_path, monkeypatch):
        """Isolation is how the pinned build requirements are honoured, so it is only
        given up when the alternative is breaking the no-network policy."""
        self._uv_only(monkeypatch)
        monkeypatch.setattr(ips, "USE_UV", True)
        calls = self._uv_plan(monkeypatch)
        assert ips._stage_replacement(str(tmp_path)) is None
        cmd, kwargs = calls[-1]
        assert "--no-build-isolation" not in cmd
        assert kwargs["env"] is None or kwargs["env"].get("PIP_NO_INDEX") != "1"

    def test_offline_still_refuses_a_git_reference(self, monkeypatch, capsys):
        """A git URL is a network fetch however direct the reference is."""
        self._uv_only(monkeypatch)
        monkeypatch.setattr(ips, "USE_UV", True)
        monkeypatch.setenv("UV_OFFLINE", "1")

        def fake_run(*args, **kwargs):
            raise AssertionError("nothing may run while uv is offline")

        monkeypatch.setattr(ips.subprocess, "run", fake_run)
        assert ips._stage_replacement("unsloth-zoo @ git+https://example/x") is None
        assert "UV_OFFLINE" in capsys.readouterr().err

    def test_a_find_links_origin_does_not_displace_the_real_index(self, monkeypatch):
        """uv annotates a flat source with a file:// URL. That belongs in
        PIP_FIND_LINKS, which is already set, and must not become PIP_INDEX_URL: an
        sdist picked out of a flat directory still needs the index for its build
        backend, so staging would abort."""
        self._uv_only(monkeypatch)
        self._uv_plan(
            monkeypatch,
            stdout = (
                b"--index-url https://pypi.org/simple\n"
                b"--find-links /opt/wheels\n"
                b"unsloth-zoo==1.0\n"
                b"    # from file:///opt/wheels\n"
            ),
        )
        _requirement, overrides, _options = ips._uv_staging_plan("unsloth-zoo")
        assert overrides["PIP_INDEX_URL"] == "https://pypi.org/simple"
        assert overrides["PIP_FIND_LINKS"] == "/opt/wheels"

    def test_a_sole_unreadable_record_is_made_uninstallable(self, tmp_path):
        """Quarantining the only record leaves pip nothing to uninstall, so the
        staged wheel is laid over the existing tree and any module the new release
        dropped stays on disk, importable, while the repair reports success.

        Verified against a real venv: with the METADATA corrupted pip show raises
        UnicodeDecodeError for the whole environment; after this rewrite pip
        uninstalls the package and removes its entire payload."""
        record = tmp_path / "realpkg-1.2.3.dist-info"
        record.mkdir()
        (record / "METADATA").write_bytes(b"\xff\xfe")
        (record / "RECORD").write_text("realpkg/__init__.py,,\n")

        assert ips._rewrite_minimal_metadata(str(record), "realpkg") is True

        written = (record / "METADATA").read_text()
        assert "Name: realpkg" in written
        # The version comes from the directory name, where importlib's own fallback reads it when METADATA cannot
        # be parsed.
        assert "Version: 1.2.3" in written

    def test_a_record_without_a_manifest_fails_closed(self, tmp_path):
        """No RECORD means neither pip nor this installer knows which files belong
        to the package, so a replacement laid over them would leave whatever the new
        release no longer ships behind."""
        record = tmp_path / "realpkg-1.2.3.dist-info"
        record.mkdir()
        (record / "METADATA").write_bytes(b"\xff\xfe")
        assert ips._rewrite_minimal_metadata(str(record), "realpkg") is False

    def test_an_unversioned_directory_fails_closed(self, tmp_path):
        record = tmp_path / "realpkg.dist-info"
        record.mkdir()
        (record / "RECORD").write_text("realpkg/__init__.py,,\n")
        assert ips._rewrite_minimal_metadata(str(record), "realpkg") is False

    def test_a_path_object_is_accepted(self, tmp_path):
        """install_manifest.invalid_metadata_paths() returns Path, and the repair
        forwards it verbatim. Passing str here (as the tests around this one do)
        hid an AttributeError on every real unreadable record.
        """
        record = tmp_path / "realpkg-1.2.3.dist-info"
        record.mkdir()
        (record / "METADATA").write_bytes(b"\xff\xfe")
        (record / "RECORD").write_text("realpkg/__init__.py,,\n")

        assert ips._rewrite_minimal_metadata(record, "realpkg") is True
        assert "Version: 1.2.3" in (record / "METADATA").read_text()

    def test_an_absent_metadata_is_synthesized_not_quarantined(self, tmp_path):
        """A record with an intact RECORD but no METADATA at all.

        There is nothing to back up, which is not a failure: quarantining it
        instead drops its RECORD, so the uninstall loop removes only what the
        readable record claims and a module shipped solely by the older release
        stays on disk and importable while the repair reports success.
        """
        record = tmp_path / "realpkg-1.2.3.dist-info"
        record.mkdir()
        (record / "RECORD").write_text("realpkg/__init__.py,,\n")

        quarantine = ips._QuarantinedMetadata()
        assert quarantine.back_up(record / "METADATA") is True
        assert ips._rewrite_minimal_metadata(record, "realpkg") is True
        assert "Name: realpkg" in (record / "METADATA").read_text()

        # A failure after the rewrite must not leave the synthetic file behind: a readable record would tell the next
        # run there is nothing left to repair.
        quarantine.restore()
        assert not (record / "METADATA").exists()

    def test_the_repair_refuses_when_the_only_record_cannot_be_made_usable(
        self, tmp_path, monkeypatch, capsys
    ):
        record = tmp_path / "unsloth-2026.8.12.dist-info"
        record.mkdir()
        (record / "METADATA").write_bytes(b"\xff\xfe")
        # Quarantining the only record leaves nothing behind for pip to uninstall.
        probes = iter(([""], []))
        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _n: next(probes))
        monkeypatch.setattr(
            ips.install_manifest, "invalid_metadata_paths", lambda _n: [str(record)]
        )
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)

        def fake_stage(*args, **kwargs):
            raise AssertionError("nothing may be staged before the records are usable")

        monkeypatch.setattr(ips, "_stage_replacement", fake_stage)
        assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        assert "Recreate the environment" in capsys.readouterr().err
        # The record is left where it was, not quarantined into a temporary directory.
        assert record.is_dir()

    HASHED_OUT = (
        b"--index-url https://mirror.corp/simple\n"
        b"unsloth-zoo==2026.8.15 \\\n"
        b"    --hash=sha256:aaaa \\\n"
        b"    --hash=sha256:bbbb\n"
        b"    # from https://mirror.corp/simple\n"
    )

    def test_the_resolved_artifact_is_pinned_by_hash(self, monkeypatch):
        """Neither PIP_CONFIG_FILE nor --isolated suppresses a site pip.conf --
        measured: with both, a venv pip.conf's extra-index-url was still contacted.
        So pip may consult a source uv never considered, and the hashes are what stop
        it accepting a different artifact of the same version from one. pip verifies
        them even with PIP_REQUIRE_HASHES=0, which is also measured."""
        self._uv_only(monkeypatch)
        self._uv_plan(monkeypatch, stdout = self.HASHED_OUT)
        requirement, _overrides, _options = ips._uv_staging_plan("unsloth-zoo")
        # The pin line is continued with a backslash, which is not part of the pin.
        assert requirement.startswith("unsloth-zoo==2026.8.15 \\\n")
        assert "--hash=sha256:aaaa" in requirement
        assert "--hash=sha256:bbbb" in requirement

    def test_a_hashed_requirement_reaches_pip_as_a_file(self, tmp_path):
        """pip only accepts --hash entries from a requirements file."""
        requirement = "unsloth-zoo==1.0 \\\n    --hash=sha256:aaaa"
        args = ips._requirement_args(requirement, str(tmp_path))
        assert args[0] == "-r"
        assert Path(args[1]).read_text().strip() == requirement
        # It lives in the staging directory, so it is removed with it.
        assert Path(args[1]).parent == tmp_path

    def test_an_unhashed_requirement_is_passed_directly(self, tmp_path):
        assert ips._requirement_args("unsloth-zoo", str(tmp_path)) == ["unsloth-zoo"]
        assert list(tmp_path.iterdir()) == []

    def test_a_flat_source_with_no_index_forbids_the_index(self, monkeypatch):
        """A configured no-index looks like this on the way out: uv emits the
        find-links entry and no index line at all. Leaving PIP_NO_INDEX cleared would
        hand pip back the default PyPI and let it stage the same name and version
        from a source uv was told to exclude."""
        self._uv_only(monkeypatch)
        self._uv_plan(
            monkeypatch,
            stdout = (
                b"--find-links /opt/wheels\n"
                b"unsloth-zoo==1.0\n"
                b"    # from file:///opt/wheels\n"
            ),
        )
        _requirement, overrides, _options = ips._uv_staging_plan("unsloth-zoo")
        assert overrides["PIP_NO_INDEX"] == "1"
        assert overrides["PIP_FIND_LINKS"] == "/opt/wheels"
        assert "PIP_INDEX_URL" not in overrides

    def test_an_emitted_index_still_clears_no_index(self, monkeypatch):
        self._uv_only(monkeypatch)
        monkeypatch.setenv("PIP_NO_INDEX", "1")
        self._uv_plan(monkeypatch)
        _requirement, overrides, _options = ips._uv_staging_plan("unsloth-zoo")
        assert overrides["PIP_NO_INDEX"] == ""

    def test_every_unreadable_record_with_a_manifest_is_made_uninstallable(
        self, tmp_path, monkeypatch
    ):
        """One unreadable record beside a readable one used to be quarantined and
        discarded, so its RECORD was never applied: the uninstall loop removed only
        what the readable record claimed, and a module existing solely in the older
        release stayed on disk and importable while the repair reported success."""
        stale = tmp_path / "unsloth-2026.8.12.dist-info"
        stale.mkdir()
        (stale / "METADATA").write_bytes(b"\xff\xfe")
        (stale / "RECORD").write_text("unsloth/gone.py,,\n")
        probes = iter(
            (["", "2026.8.15"], ["2026.8.12", "2026.8.15"], ["2026.8.15"], [], ["2026.8.15"])
        )
        taken = []

        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _n: next(probes))
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _n: [str(stale)])
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _n: "/staged")
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: True)
        monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: True)
        monkeypatch.setattr(
            ips._QuarantinedMetadata, "take", lambda _self, paths: taken.append(paths) or True
        )

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is True
        # It was rewritten for pip rather than moved aside, so its RECORD is applied and nothing is quarantined at all.
        assert taken == []
        assert "Name: unsloth" in (stale / "METADATA").read_text()

    def test_a_repaired_package_is_not_rolled_back_by_a_later_failure(self, monkeypatch):
        """A single quarantine shared across both packages would, when the second
        fails, restore the first package's stale record on top of the install that
        has already replaced it: the conflict returns, and its old RECORD then
        describes a payload that is gone."""
        probes = {
            "unsloth": iter((["", "2026.8.15"], ["2026.8.15"], [], ["2026.8.15"])),
            "unsloth-zoo": iter((["", "2026.8.15"],)),
        }
        events = []

        monkeypatch.setattr(
            ips.install_manifest, "installed_versions", lambda name: next(probes[name])
        )
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _n: [])
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: True)
        monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: True)
        # The second package cannot be staged, so the repair fails after the first one has already been reinstalled.
        staged = iter(("/staged", None))
        monkeypatch.setattr(ips, "_stage_replacement", lambda _n: next(staged))
        monkeypatch.setattr(
            ips._QuarantinedMetadata, "discard", lambda _self: events.append("discard")
        )
        monkeypatch.setattr(
            ips._QuarantinedMetadata, "restore", lambda _self: events.append("restore")
        )

        assert ips._repair_duplicate_core_metadata(("unsloth", "unsloth-zoo")) is False
        # The first package was committed before the second was attempted, so the rollback at the end can only touch the
        # one that failed.
        assert events[0] == "discard"
        assert events[-1] == "restore"

    def test_a_direct_reference_pin_is_recognised(self, monkeypatch):
        """An override can redirect a package to a path, repository or URL, and uv
        then emits `name @ reference` rather than `name==version`. Treating the whole
        line as the name left the requirement empty and aborted every repair under
        that policy."""
        self._uv_only(monkeypatch)
        self._uv_plan(
            monkeypatch,
            stdout = (
                b"--index-url https://pypi.org/simple\n"
                b"unsloth-zoo @ file:///src/zoo\n"
                b"    # from https://pypi.org/simple\n"
            ),
        )
        requirement, _overrides, _options = ips._uv_staging_plan("unsloth_zoo")
        # The reference is kept as written; only the name is parsed out of it.
        assert requirement == "unsloth-zoo @ file:///src/zoo"

    @pytest.mark.parametrize(
        "line, name",
        (
            ("six==1.17.0", "six"),
            ("unsloth-zoo @ git+https://example/x", "unsloth-zoo"),
            ("unsloth_zoo @ file:///src", "unsloth_zoo"),
        ),
    )
    def test_the_name_is_taken_from_either_spelling(self, line, name):
        assert ips._requirement_name(line) == name

    def test_the_original_metadata_comes_back_when_the_repair_fails(self, tmp_path):
        """The rewrite has to happen before staging, and staging can still fail.
        Without a backup the original is gone and what remains parses, so the next
        run would see one readable record, decide nothing is wrong, and never attempt
        the payload repair that is still owed."""
        record = _shared_setup_5(tmp_path)
        quarantine = ips._QuarantinedMetadata()

        assert quarantine.back_up(str(record / "METADATA")) is True
        assert ips._rewrite_minimal_metadata(str(record), "unsloth") is True
        assert "Name: unsloth" in (record / "METADATA").read_text()

        quarantine.restore()

        assert (record / "METADATA").read_bytes() == b"\xff\xfe"

    def test_a_committed_rewrite_is_not_undone(self, tmp_path):
        record = _shared_setup_5(tmp_path)
        quarantine = ips._QuarantinedMetadata()
        quarantine.back_up(str(record / "METADATA"))
        ips._rewrite_minimal_metadata(str(record), "unsloth")

        quarantine.discard()
        quarantine.restore()

        assert "Name: unsloth" in (record / "METADATA").read_text()

    def test_an_unbackable_metadata_stops_the_repair(self, tmp_path, monkeypatch, capsys):
        """The reachable route to an unrewritable record: a METADATA that exists but
        cannot be read, as an elevated install leaves root-owned. back_up fails, so
        the rewrite is skipped and the record can never be handed to pip.

        Reproduced in a real venv with the file made unreadable: before this refused,
        the repair returned True, the module only the stale release shipped stayed
        importable, and its dist-info was deleted, so nothing could report it again.
        """
        record = _shared_setup_5(tmp_path)

        monkeypatch.setattr(
            ips.install_manifest, "installed_versions", lambda _n: ["", "2026.8.15"]
        )
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _n: [record])
        monkeypatch.setattr(ips.install_manifest, "pip_backup_metadata_paths", lambda _n: [])
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips._QuarantinedMetadata, "back_up", lambda _self, _p: False)

        def refuse(*_a, **_k):
            raise AssertionError("nothing may run once a record is unusable")

        monkeypatch.setattr(ips, "_stage_replacement", refuse)
        monkeypatch.setattr(ips, "_run_ok", refuse)

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        assert "cannot be read or rewritten" in capsys.readouterr().err
        # Byte for byte what was there, so the conflict is still detected next time.
        assert (record / "METADATA").read_bytes() == b"\xff\xfe"

    def test_an_unresolvable_name_stages_nothing(self, monkeypatch):
        self._uv_only(monkeypatch)
        monkeypatch.setattr(ips, "USE_UV", True)
        self._uv_plan(monkeypatch, stdout = b"--index-url https://pypi.org/simple\n")
        assert ips._uv_staging_plan("unsloth-zoo") is None

    def test_a_pin_for_another_package_is_not_mistaken_for_this_one(self, monkeypatch):
        self._uv_only(monkeypatch)
        self._uv_plan(monkeypatch, stdout = b"unsloth==2026.8.15\n")
        assert ips._uv_staging_plan("unsloth-zoo") is None

    def test_the_uv_upload_cutoff_reaches_pip(self, monkeypatch):
        """UV_EXCLUDE_NEWER limits candidates by upload time and pip ignores it, so
        staging could install a release the user's policy excludes. pip's
        --uploaded-prior-to is the same filter and takes the same date spellings
        (checked on pip 26.2: a bare 2020-01-01 staged six 1.13.0, not 1.17.0)."""
        self._uv_only(monkeypatch)
        monkeypatch.setenv("UV_EXCLUDE_NEWER", "2026-01-01")
        monkeypatch.setattr(ips, "USE_UV", False)
        monkeypatch.setattr(ips, "_pip_supports_upload_cutoff", lambda: True)
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return types.SimpleNamespace(returncode = 1, stdout = b"")

        monkeypatch.setattr(ips.subprocess, "run", fake_run)
        assert ips._stage_replacement("unsloth") is None
        assert "--uploaded-prior-to" in captured["cmd"]
        assert captured["cmd"][captured["cmd"].index("--uploaded-prior-to") + 1] == "2026-01-01"

    def test_an_unhonourable_cutoff_leaves_the_install_alone(self, monkeypatch, capsys):
        """--uploaded-prior-to only exists from pip 25.3. Staging a newer wheel
        anyway would silently break the policy, so the repair aborts instead, with
        the duplicate still in place and the package still installed."""
        self._uv_only(monkeypatch)
        monkeypatch.setenv("UV_EXCLUDE_NEWER", "2026-01-01")
        monkeypatch.setattr(ips, "USE_UV", False)
        monkeypatch.setattr(ips, "_pip_supports_upload_cutoff", lambda: False)

        def fake_run(*args, **kwargs):
            raise AssertionError("pip must not run when the cutoff cannot be honoured")

        monkeypatch.setattr(ips.subprocess, "run", fake_run)
        assert ips._stage_replacement("unsloth") is None
        assert "UV_EXCLUDE_NEWER" in capsys.readouterr().err

    def test_no_cutoff_argument_without_the_variable(self, monkeypatch):
        self._uv_only(monkeypatch)
        assert ips._uv_upload_cutoff_args() == []

    @pytest.mark.parametrize("label", ("_restore_from_staged", "the repair fallback"))
    def test_the_staged_wheel_is_reinstalled_with_pip_not_uv(self, monkeypatch, label):
        """UV_REQUIRE_HASHES would reject the unpinned name after the uninstall
        loop has already removed every record, leaving the package uninstalled.
        The wheel is already built, so pip is both safe and sufficient."""
        installs = []
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(
            ips,
            "pip_install_try",
            lambda _label, *args, **kwargs: installs.append(kwargs) or True,
        )
        if label == "_restore_from_staged":
            ips._restore_from_staged("unsloth", "/staged", removed_any = True)
        else:
            probes = iter((["old", "new"], ["new"], [], ["new"]))
            monkeypatch.setattr(
                ips.install_manifest, "installed_versions", lambda _name: next(probes)
            )
            monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
            monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: True)
            monkeypatch.setattr(ips, "_stage_replacement", lambda _name: "/staged")
            assert ips._repair_duplicate_core_metadata(("custom-package",))
        assert len(installs) == 1
        assert installs[0].get("force_pip") is True

    def test_ci_overlay_is_wired_into_duplicate_repair(self):
        source = inspect.getsource(ips.install_python_stack).replace(" ", "")
        assert 'os.environ.get("UNSLOTH_CI_SOURCE_OVERLAY","")' in source
        assert "ci_source_overlay=ci_source_overlay" in source

    def test_the_replacement_is_staged_before_any_record_is_removed(self, monkeypatch):
        """The uninstall loop deletes every record, so the replacement has to be
        in hand first. Otherwise an index that is unreachable at that moment
        leaves the venv with no unsloth and nothing to reinstall from."""
        probes = iter((["2026.8.12", "2026.8.15"], ["2026.8.15"], [], ["2026.8.15"]))
        order = []

        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _name: next(probes))
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(
            ips, "_stage_replacement", lambda _name: order.append("stage") or "/staged"
        )
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: order.append("uninstall") or True)
        monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: order.append("install") or True)

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is True
        # Two records, so two uninstalls: pip removes one per invocation.
        assert order == ["stage", "uninstall", "uninstall", "install"]

    def test_repair_leaves_the_install_alone_when_the_replacement_cannot_be_fetched(
        self, monkeypatch, capsys
    ):
        probes = iter((["2026.8.12", "2026.8.15"],))
        removals = []

        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _name: next(probes))
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _name: None)
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: removals.append(a) or True)
        monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: True)

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        assert removals == []
        assert "could not fetch a replacement" in capsys.readouterr().err

    def test_a_failed_reinstall_reports_instead_of_exiting(self, monkeypatch, capsys):
        probes = iter((["2026.8.12", "2026.8.15"], ["2026.8.15"], []))

        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _name: next(probes))
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _name: "/staged")
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: True)
        monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: False)

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        assert "no longer installed" in capsys.readouterr().err

    def test_a_failed_uninstall_reports_instead_of_exiting(self, monkeypatch, capsys):
        probes = iter((["2026.8.12", "2026.8.15"],))

        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _name: next(probes))
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _name: "/staged")
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: False)
        installs = []
        monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: installs.append(a))

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        assert installs == []
        assert "could not uninstall" in capsys.readouterr().err

    def test_a_rollback_reinstall_keeps_its_own_metadata(self, monkeypatch, tmp_path, capsys):
        """The rewritten record is uninstalled, a later uninstall fails, and
        _restore_from_staged puts the package back from the staged wheel. The
        finally block then ran quarantine.restore() over the top, either deleting
        the wheel's valid METADATA (original absent) or overwriting it with the
        original corrupt bytes, leaving the core package malformed after a
        recovery that existed to make it whole.
        """
        record = tmp_path / "unsloth-2026.8.15.dist-info"
        record.mkdir()
        (record / "METADATA").write_bytes(b"\xff\xfe")
        (record / "RECORD").write_text("unsloth/__init__.py,,\n")

        probes = iter((["2026.8.15", "2026.8.15"], ["2026.8.15", "2026.8.15"], ["2026.8.15"]))
        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _n: next(probes))
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _n: [record])
        monkeypatch.setattr(ips.install_manifest, "pip_backup_metadata_paths", lambda _n: [])
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _n: str(tmp_path / "staged"))

        runs = {"n": 0}

        def uninstall(_label, _cmd):
            runs["n"] += 1
            if runs["n"] == 1:
                shutil.rmtree(record)
                return True
            return False

        monkeypatch.setattr(ips, "_run_ok", uninstall)

        def reinstall(*_a, **_k):
            # The staged wheel recreates the same path with its own valid metadata.
            record.mkdir(exist_ok = True)
            (record / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: unsloth\nVersion: 2026.8.15\n", encoding = "utf-8"
            )
            return True

        monkeypatch.setattr(ips, "pip_install_try", reinstall)

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        # Whatever the rollback put back must survive the quarantine unwinding.
        assert record.is_dir()
        assert (record / "METADATA").read_text().startswith("Metadata-Version")
        assert "Name: unsloth" in (record / "METADATA").read_text()

    def test_a_quarantined_backup_is_restored_when_staging_fails(self, monkeypatch, tmp_path):
        """Quarantine's remaining user is pip's ~ leftover, moved aside so the
        uninstall loop can converge. Moving it and then failing to fetch the
        replacement would leave the venv worse than it was found, so a failed
        staging has to put it back.
        """
        backup = _shared_setup_3(tmp_path)
        probes = iter((["2026.8.12", "2026.8.15"], ["2026.8.15"]))

        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _n: next(probes))
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _n: [])
        monkeypatch.setattr(ips.install_manifest, "pip_backup_metadata_paths", lambda _n: [backup])
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _n: None)

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        assert backup.is_dir()
        assert "Name: unsloth" in (backup / "METADATA").read_text()

    def test_staging_builds_a_wheel_so_the_offline_install_can_work(self):
        """pip download leaves an sdist for a source-only index, and the install
        that follows runs --no-index, so its isolated build cannot fetch
        setuptools and the package stays uninstalled."""
        source = inspect.getsource(ips._stage_replacement).replace(" ", "")
        assert '"wheel",\n"--no-deps",' in source
        assert '"--wheel-dir",\nstaging,' in source
        assert '"download"' not in source
        assert 'glob.glob(os.path.join(staging,"*.whl"))' in source

    def test_a_git_overlay_is_staged_before_the_uninstall_loop(self, monkeypatch):
        """--local pulls unsloth-zoo from git, so an overlay is a network fetch
        too. Skipping staging for it meant an unreachable GitHub left the
        package uninstalled."""
        probes = {"unsloth-zoo": iter((["old", "new"], ["new"], [], ["new"]))}
        order = []

        _shared_setup_2(monkeypatch, probes)
        monkeypatch.setattr(
            ips, "_stage_replacement", lambda spec: order.append(("stage", spec)) or "/staged"
        )
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: order.append(("uninstall",)) or True)
        monkeypatch.setattr(
            ips, "pip_install_try", lambda label, *a, **k: order.append(("install",)) or True
        )

        assert ips._repair_duplicate_core_metadata(("unsloth-zoo",), local_repo = "/src/unsloth")
        assert order[0] == ("stage", "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo")
        assert order[1] == ("uninstall",)

    def test_an_editable_overlay_stages_the_checkout(self, monkeypatch):
        probes = {"unsloth": iter((["old", "new"], ["new"], [], ["new"]))}
        staged_for = []

        _shared_setup_2(monkeypatch, probes)
        monkeypatch.setattr(
            ips, "_stage_replacement", lambda spec: staged_for.append(spec) or "/staged"
        )
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: True)
        monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: True)

        assert ips._repair_duplicate_core_metadata(("unsloth",), local_repo = "/src/unsloth")
        assert staged_for == ["/src/unsloth"]

    def test_a_failed_overlay_falls_back_to_the_staged_source(self, monkeypatch):
        probes = {"unsloth-zoo": iter((["old", "new"], ["new"], [], ["new"]))}
        installs = []

        _shared_setup_2(monkeypatch, probes)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _spec: "/staged")
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: True)
        monkeypatch.setattr(ips, "_overlay_local_core_package", lambda *a, **k: False)
        monkeypatch.setattr(
            ips, "pip_install_try", lambda label, *a, **k: installs.append(a) or True
        )

        assert ips._repair_duplicate_core_metadata(("unsloth-zoo",), local_repo = "/src/unsloth")
        assert installs and "--find-links" in installs[0]

    def test_a_partial_uninstall_restores_the_payload(self, monkeypatch, capsys):
        """The first uninstall deletes the package tree. Returning after a later
        one fails would leave a dist-info claiming an install whose files are
        gone."""
        probes = iter((["a", "b", "c"], ["b", "c"]))
        uninstalls = iter([True, False])
        installs = []

        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _name: next(probes))
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _name: [])
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _spec: "/staged")
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: next(uninstalls))
        monkeypatch.setattr(
            ips, "pip_install_try", lambda label, *a, **k: installs.append(a) or True
        )

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        assert installs and "--find-links" in installs[0]
        assert "restored unsloth from the staged replacement" in capsys.readouterr().err

    def test_nothing_is_restored_when_no_record_was_removed(self, monkeypatch):
        probes = iter((["a", "b"],))
        installs = []

        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _name: next(probes))
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _name: [])
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _spec: "/staged")
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: False)
        monkeypatch.setattr(ips, "pip_install_try", lambda label, *a, **k: installs.append(a))

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is False
        assert installs == []

    def test_staging_directories_are_cleaned_up(self, monkeypatch, tmp_path):
        staged = tmp_path / "staged"
        staged.mkdir()
        probes = iter((["2026.8.12", "2026.8.15"], ["2026.8.15"], [], ["2026.8.15"]))

        monkeypatch.setattr(ips.install_manifest, "installed_versions", lambda _name: next(probes))
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _name: str(staged))
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: True)
        monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: True)

        assert ips._repair_duplicate_core_metadata(("unsloth",)) is True
        assert not staged.exists()

    def test_normal_local_overlay_still_applies_both_sources(self, monkeypatch):
        installs = []
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(
            ips,
            "pip_install",
            lambda label, *args, **kwargs: installs.append((label, args, kwargs)),
        )

        ips._overlay_local_core_packages("/src/unsloth")

        assert [call[1] for call in installs] == [
            ("--no-cache-dir", "--no-deps", "-e", "/src/unsloth"),
            (
                "--no-cache-dir",
                "--no-deps",
                "--force-reinstall",
                "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo",
            ),
        ]
        assert all(call[2]["constrain"] is False for call in installs)


class TestDesktopBackendVersionConstraint:
    """Verify that UNSLOTH_DESKTOP_BACKEND_VERSION adds the floor pin when upgrading unsloth."""

    def test_spec_includes_floor_when_env_set(self):
        with mock.patch.dict(os.environ, {"UNSLOTH_DESKTOP_BACKEND_VERSION": "2026.8.15"}):
            desktop_min_ver = os.environ.get("UNSLOTH_DESKTOP_BACKEND_VERSION", "").strip()
            package_name = "unsloth"
            unsloth_spec = (
                f"{package_name}>={desktop_min_ver}"
                if (desktop_min_ver and package_name == "unsloth")
                else package_name
            )
            assert unsloth_spec == "unsloth>=2026.8.15"

    def test_spec_bare_when_env_unset(self):
        with mock.patch.dict(os.environ, {}, clear = True):
            desktop_min_ver = os.environ.get("UNSLOTH_DESKTOP_BACKEND_VERSION", "").strip()
            package_name = "unsloth"
            unsloth_spec = (
                f"{package_name}>={desktop_min_ver}"
                if (desktop_min_ver and package_name == "unsloth")
                else package_name
            )
            assert unsloth_spec == "unsloth"


class TestRecordlessDistributionRecovery:
    """pip cannot replace a .dist-info with no RECORD, and the venv is reused.

    Observed on the Windows startup job: uv wrote pydantic-core, the pip fallback
    tried to uninstall it, found no RECORD, and the dependency step died. Nothing
    clears that state, so the same venv failed the same way on every later run.
    """

    _PIP_OUTPUT = (
        b"Attempting uninstall: pydantic-core\n"
        b"  Found existing installation: pydantic-core None\n"
        b"error: uninstall-no-record-file\n"
        b"x Cannot uninstall pydantic-core None\n"
        b"|-> The package's contents are unknown: no RECORD file was found for pydantic-core.\n"
    )

    def _site_packages(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ips.sysconfig, "get_path", lambda name: str(tmp_path))
        return tmp_path

    def _write_dist_info(self, root, name, *, record: bool):
        dist_info = root / name
        dist_info.mkdir()
        (dist_info / "METADATA").write_text("Name: x\n", encoding = "utf-8")
        if record:
            (dist_info / "RECORD").write_text("", encoding = "utf-8")
        return dist_info

    def test_the_named_stub_is_cleared(self, tmp_path, monkeypatch):
        root = self._site_packages(tmp_path, monkeypatch)
        stub = self._write_dist_info(root, "pydantic_core-2.46.5.dist-info", record = False)

        cleared = ips._purge_recordless_distributions(self._PIP_OUTPUT)

        assert cleared == ["pydantic_core-2.46.5.dist-info"]
        assert not stub.exists()

    def test_a_complete_install_of_the_same_name_is_left_alone(self, tmp_path, monkeypatch):
        # A RECORD means pip knows what it owns; whatever failed, it was not this.
        root = self._site_packages(tmp_path, monkeypatch)
        intact = self._write_dist_info(root, "pydantic_core-2.46.5.dist-info", record = True)

        assert ips._purge_recordless_distributions(self._PIP_OUTPUT) == []
        assert intact.exists()

    def test_other_packages_are_never_touched(self, tmp_path, monkeypatch):
        # The blast radius is the names pip named, not every stub in site-packages.
        root = self._site_packages(tmp_path, monkeypatch)
        bystander = self._write_dist_info(root, "fastapi-0.121.0.dist-info", record = False)

        assert ips._purge_recordless_distributions(self._PIP_OUTPUT) == []
        assert bystander.exists()

    def test_an_unrelated_failure_clears_nothing(self, tmp_path, monkeypatch):
        # No RECORD marker: a different problem, which deleting metadata cannot fix.
        root = self._site_packages(tmp_path, monkeypatch)
        stub = self._write_dist_info(root, "pydantic_core-2.46.5.dist-info", record = False)

        assert ips._purge_recordless_distributions(b"ERROR: could not resolve pydantic-core") == []
        assert stub.exists()

    @pytest.mark.parametrize("output", [None, b"", ""])
    def test_no_output_is_not_a_reason_to_delete(self, output, tmp_path, monkeypatch):
        self._site_packages(tmp_path, monkeypatch)
        assert ips._purge_recordless_distributions(output) == []

    def test_the_name_matches_across_dash_and_underscore_spellings(self, tmp_path, monkeypatch):
        # pip reports "pydantic-core"; the directory is written "pydantic_core".
        root = self._site_packages(tmp_path, monkeypatch)
        stub = self._write_dist_info(root, "pydantic.core-2.46.5.dist-info", record = False)

        assert ips._purge_recordless_distributions(self._PIP_OUTPUT) == [stub.name]
        assert not stub.exists()

    def test_the_fallback_retries_once_after_clearing(self, tmp_path, monkeypatch):
        """The recovery is only worth anything if pip_install actually re-runs."""
        root = self._site_packages(tmp_path, monkeypatch)
        self._write_dist_info(root, "pydantic_core-2.46.5.dist-info", record = False)
        monkeypatch.setattr(ips, "USE_UV", False)
        monkeypatch.setattr(ips, "CONSTRAINTS", tmp_path / "absent.txt")

        attempts = []

        def fake_run(
            label,
            cmd,
            *,
            quiet = True,
            check = True,
            env = None,
        ):
            attempts.append(cmd)
            failed = len(attempts) == 1
            return types.SimpleNamespace(
                returncode = 1 if failed else 0,
                stdout = self._PIP_OUTPUT if failed else b"",
            )

        monkeypatch.setattr(ips, "run", fake_run)
        ips.pip_install("studio deps", "pydantic-core")

        assert len(attempts) == 2 and attempts[0] == attempts[1]

    def test_a_failure_it_cannot_recover_from_still_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ips.sysconfig, "get_path", lambda name: str(tmp_path))
        monkeypatch.setattr(ips, "USE_UV", False)
        monkeypatch.setattr(ips, "CONSTRAINTS", tmp_path / "absent.txt")

        attempts = []

        def fake_run(
            label,
            cmd,
            *,
            quiet = True,
            check = True,
            env = None,
        ):
            attempts.append(cmd)
            return types.SimpleNamespace(returncode = 1, stdout = b"ERROR: no matching distribution")

        monkeypatch.setattr(ips, "run", fake_run)
        with pytest.raises(SystemExit) as excinfo:
            ips.pip_install("studio deps", "pydantic-core")

        assert excinfo.value.code == 1
        assert len(attempts) == 1, "a failure with nothing to clear must not be retried"


class TestExpectedTorchFlavorResolution:
    """_expected_torch_flavor_tag / _expected_torch_index_url: the two pure inputs to the
    Windows flavor invariant. The invariant itself is covered in
    tests/studio/install/test_cuda_repair.py; these pin the resolution ORDER, which is what
    decides whether a repair fires against the right index or not at all."""

    _KEYS = (
        "UNSLOTH_EXPECTED_TORCH_TAG",
        "UNSLOTH_TORCH_INSTALL_INDEX_URL",
        "UNSLOTH_TORCH_INDEX_URL",
        "UNSLOTH_TORCH_INDEX_FAMILY",
    )

    @contextlib.contextmanager
    def _env(self, **values):
        """Set the named vars and REMOVE every other one this resolution reads, so an
        ambient pin on the developer's box cannot change the answer."""
        with mock.patch.dict(os.environ, {k: v for k, v in values.items() if v is not None}):
            for key in self._KEYS:
                if values.get(key) is None:
                    os.environ.pop(key, None)
            yield

    def test_the_handover_tag_wins(self):
        with self._env(UNSLOTH_EXPECTED_TORCH_TAG = "cu124"):
            with mock.patch.object(ips, "_RECORDED_TORCH_TAG", "cu128"):
                assert ips._expected_torch_flavor_tag() == "cu124"

    def test_the_handover_tag_is_normalised(self):
        with self._env(UNSLOTH_EXPECTED_TORCH_TAG = " CU128 "):
            assert ips._expected_torch_flavor_tag() == "cu128"

    def test_the_manifest_answers_next(self):
        with self._env():
            with mock.patch.object(ips, "_RECORDED_TORCH_TAG", "cu128"):
                assert ips._expected_torch_flavor_tag() == "cu128"

    def test_a_resolved_cuda_backend_outranks_a_stale_cpu_record(self):
        """A CPU pin that was REMOVED must not outlive the install that replaced it.

        install.sh resolves the backend to cuda and installs a CUDA wheel; reading the old
        manifest here recorded the healthy venv as deliberately CPU-only, and
        _expected_cpu_flavor_was_chosen() would then read a later CPU wheel as the install
        working as asked and suppress the mismatch and its repair. The wheel's own tag is
        what says which cu index this venv came from.
        """
        with self._env():
            with (
                mock.patch.object(ips, "_TORCH_BACKEND", "cuda"),
                mock.patch.object(ips, "_RECORDED_TORCH_TAG", "cpu"),
                mock.patch.object(
                    ips, "_installed_torch_version_label", return_value = "2.9.1+cu128"
                ),
            ):
                assert ips._expected_torch_flavor_tag() == "cu128"
                # And the stale provenance goes with it: pinned answers per family, so the
                # cpu record cannot speak for a cuda tag. install.sh marks the backend it
                # derived, which is why the derived cuda does not count as pinned either.
                with (
                    mock.patch.dict(os.environ, {"UNSLOTH_TORCH_BACKEND_SOURCE": "resolved"}),
                    mock.patch.object(ips, "_RECORDED_TORCH_TAG_PINNED", True),
                ):
                    assert ips._expected_torch_flavor_was_pinned("cu128") is False

    def test_a_resolved_cuda_backend_over_a_cpu_wheel_still_reads_the_manifest(self):
        # Only when the family agrees with the wheel actually installed, the same rule the
        # cpu/rocm/xpu arm applies: a resolved cuda beside a CPU wheel is the mismatch this
        # whole path exists to repair, not a reason to call the venv CUDA.
        with self._env():
            with (
                mock.patch.object(ips, "_TORCH_BACKEND", "cuda"),
                mock.patch.object(ips, "_RECORDED_TORCH_TAG", "cu124"),
                mock.patch.object(ips, "_installed_torch_version_label", return_value = "2.11.0+cpu"),
            ):
                assert ips._expected_torch_flavor_tag() == "cu124"

    def test_a_gpuless_host_with_nothing_recorded_says_nothing(self):
        # Inventing a CUDA expectation from an absent GPU would reinstall CUDA torch onto a CPU box on every update.
        with self._env():
            with (
                mock.patch.object(ips, "_RECORDED_TORCH_TAG", None),
                mock.patch.object(ips, "_has_usable_nvidia_gpu", return_value = False),
            ):
                assert ips._expected_torch_flavor_tag() == ""

    def test_a_pin_answers_without_probing_the_gpu(self):
        with self._env(UNSLOTH_TORCH_INDEX_FAMILY = "cu126"):
            with (
                mock.patch.object(ips, "_RECORDED_TORCH_TAG", None),
                mock.patch.object(ips, "_has_usable_nvidia_gpu") as probe,
            ):
                assert ips._expected_torch_flavor_tag() == "cu126"
            probe.assert_not_called()

    def test_a_cpu_pin_resolves_to_cpu_not_to_the_host_gpu(self):
        with self._env(UNSLOTH_TORCH_INDEX_FAMILY = "cpu"):
            with mock.patch.object(ips, "_RECORDED_TORCH_TAG", None):
                assert ips._expected_torch_flavor_tag() == "cpu"

    def test_the_index_url_is_reused_only_for_its_own_family(self):
        # setup.ps1 hands over the /cpu index alongside a "rocm" tag on AMD Windows, so repairing from it would install
        # the very CPU wheel the repair exists to remove.
        with self._env(UNSLOTH_TORCH_INSTALL_INDEX_URL = "https://mirror.local/whl/cu124/"):
            assert ips._expected_torch_index_url("cu124") == "https://mirror.local/whl/cu124"
        with self._env(UNSLOTH_TORCH_INSTALL_INDEX_URL = "https://download.pytorch.org/whl/cpu"):
            assert ips._expected_torch_index_url("cu124") == f"{ips._PYTORCH_WHL_BASE}/cu124"

    def test_a_credentialed_index_survives_intact(self):
        # The URL is forwarded rather than rebuilt: userinfo and a token query are not
        # reconstructible from a family leaf.
        url = "https://user:tok@mirror.local/whl/cu128?token=abc"
        with self._env(UNSLOTH_TORCH_INSTALL_INDEX_URL = url):
            assert ips._expected_torch_index_url("cu128") == url

    def test_the_pin_supplies_the_index_when_the_setup_script_did_not(self):
        with self._env(UNSLOTH_TORCH_INDEX_URL = "https://mirror.local/whl/cu126"):
            assert ips._expected_torch_index_url("cu126") == "https://mirror.local/whl/cu126"

    def test_the_default_index_is_the_pytorch_mirror(self):
        with self._env():
            assert ips._expected_torch_index_url("cu124") == f"{ips._PYTORCH_WHL_BASE}/cu124"

    def test_no_index_url_is_ever_persisted_by_the_manifest_write(self):
        # The manifest lives in the venv, so a token in a pinned URL must not reach it.
        source = inspect.getsource(ips.install_python_stack)
        assert "expected_torch_tag = _recordable_torch_flavor_tag(torch_flavor_tag)," in source
        assert "torch_index_url" not in source
        # And the helper that answers it records a FLAVOR, never a URL, for the same
        # reason: it is reached with the pin still in the environment.
        helper = inspect.getsource(ips._recordable_torch_flavor_tag)
        assert "return" in helper and "_explicit_torch_index_url()" not in helper
