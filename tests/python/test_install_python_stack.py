"""Tests for install_python_stack._build_uv_cmd torch-backend handling."""

from __future__ import annotations

import ast
import contextlib
import importlib
import io
import os
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))

import install_python_stack as ips


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

    def test_uv_torch_backend_auto(self):
        """UV_TORCH_BACKEND=auto adds --torch-backend=auto."""
        with mock.patch.dict(os.environ, {"UV_TORCH_BACKEND": "auto"}):
            cmd = self._call(("somepackage",))
        assert "--torch-backend=auto" in cmd

    def test_uv_torch_backend_cpu(self):
        """UV_TORCH_BACKEND=cpu adds --torch-backend=cpu."""
        with mock.patch.dict(os.environ, {"UV_TORCH_BACKEND": "cpu"}):
            cmd = self._call(("somepackage",))
        assert "--torch-backend=cpu" in cmd

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

    def test_uv_torch_backend_kept_for_unpinned(self):
        """Non-pinned commands still honour UV_TORCH_BACKEND."""
        with mock.patch.dict(os.environ, {"UV_TORCH_BACKEND": "cpu"}):
            cmd = self._call(("somepackage",))
        assert "--torch-backend=cpu" in cmd


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
        # --default-index must be gated too (matches install.sh / install.ps1).
        cmd = ["uv", "pip", "install", "torch", "--default-index", "https://x/cu126"]
        with mock.patch.dict(os.environ, {"UV_INDEX": "https://mirror.corp/simple"}):
            env = ips._install_env_for_cmd(cmd)
        assert env is not None
        assert "UV_INDEX" not in env

    def test_non_pinned_install_keeps_user_mirror(self):
        # A plain install (no --index-url) must NOT scrub the env, so a user's mirror
        # still applies to base packages.
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
        """The pip FALLBACK honours user/site pip config files (pip config set
        global.extra-index-url) even with the PIP_* env vars stripped; pip loads
        NO configuration files when PIP_CONFIG_FILE is os.devnull. Harmless for
        uv, decisive for the fallback."""
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

    unslothai/unsloth#8530: `no-build = true` in ~/.config/uv/uv.toml (and
    `only-binary = :all:` in ~/.config/pip/pip.conf) makes every wheel-less requirement
    in extras.txt unresolvable, so `unsloth studio update` died at "unsloth extras".
    A PACKAGE-SCOPED --no-binary overrides that policy for those names only, leaving the
    user's binary-only policy in force for every other requirement -- verified against
    uv 0.10 and pip 26: dropping one name from the flags makes that name refused again.
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
        # omegaconf==2.3.1 pins antlr4-python3-runtime below the 4.13.2 wheel, so it
        # arrives as a transitive sdist and fails no-build even though extras.txt
        # never names it.
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
        compiler-dependent source build on every other host, which is a worse bug than
        the one being fixed. Verified against uv 0.10: 0.996.5 is refused under
        `no-build = true` for macOS cp314 and resolves with --no-binary MeCab, while
        0.996.13 still comes from a wheel elsewhere.
        """
        with (
            mock.patch.object(ips, "IS_MACOS", is_macos),
            mock.patch.object(sys, "version_info", version),
        ):
            names = ips._extras_sdist_only_packages()
        assert ("MeCab" in names) is expected
        # The unconditional ones are always present.
        assert set(ips.SDIST_ONLY_PACKAGES) <= set(names)

    def test_the_diffusers_pin_is_exempted_on_the_archive_path(self):
        """The pin is a source ARCHIVE, and uv's no-build refuses to build one, so the
        install still died here after extras.txt was fixed.

        Guarded on python >= 3.10 because diffusers-pin.txt resolves a released wheel
        below that, which must not be forced through a source build.
        """
        tree = ast.parse(Path(ips.__file__).read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "pip_install"):
                continue
            req = next((k for k in node.keywords if k.arg == "req"), None)
            if req is None or "diffusers-pin.txt" not in ast.unparse(req.value):
                continue
            splat = " ".join(
                ast.unparse(a.value) for a in node.args if isinstance(a, ast.Starred)
            )
            assert "_sdist_only_build_args('diffusers')" in splat, (
                f"the diffusers pin at line {node.lineno} must exempt the source archive"
            )
            assert "version_info >= (3, 10)" in splat, (
                "the exemption must be guarded so the pre-3.10 wheel is not forced "
                f"through a source build (line {node.lineno})"
            )
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


class TestHardenedPipConfigRelaxation:
    """`require-hashes = true` in pip.conf killed the pip FALLBACK in #8530.

    Every requirements file the installer ships is pinned but unhashed, so hash-required
    mode can never be satisfied. pip applies env vars AFTER config files, so
    PIP_REQUIRE_HASHES=0 in the child env overrides it while pip.conf's index-url,
    trusted-host, cert and proxy settings all stay in force. There is no command-line
    equivalent, which is why this one knob is handled through the environment.
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

    def test_pinned_cmd_strips_restrictive_policy_env(self):
        """The pinned branch neutralises the config FILES, but an env var outranks a
        config file, so a hardened shell could still fail a torch repair the pin was
        supposed to make deterministic."""
        with mock.patch.dict(os.environ, self.HOSTILE):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env is not None
        for name in ("PIP_REQUIRE_HASHES", "PIP_ONLY_BINARY", "UV_NO_BUILD", "UV_EXCLUDE_NEWER"):
            assert name not in env, f"{name} must be cleared for a pinned install"
        # The pre-existing pinned contract is unchanged.
        assert env["UV_NO_CONFIG"] == "1" and env["PIP_CONFIG_FILE"] == os.devnull

    def test_the_parent_environment_is_never_mutated(self):
        """The relaxation is a child-env override. Leaking it into os.environ would
        weaken the user's policy for their own later pip commands in this session."""
        with mock.patch.dict(os.environ, self.HOSTILE):
            ips._install_env_for_cmd(["python", "-m", "pip", "install", "x"])
            ips._install_env_for_cmd(["uv", "pip", "install", "x", "--index-url", "https://y"])
            assert os.environ["PIP_REQUIRE_HASHES"] == "1"
            assert os.environ["UV_NO_BUILD"] == "1"

    def test_the_pip_fallback_receives_the_relaxation(self):
        """End of the real path: uv fails, pip_install falls back through run(), and
        that pip command is the one #8530 died on."""
        seen: dict = {}

        def _fake_run(label, cmd, *a, **kw):
            seen["env"] = ips._install_env_for_cmd(cmd)
            seen["cmd"] = cmd

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
            # _HAS_COLOR is resolved once at import from the tty; pinning it keeps the
            # assertions valid under FORCE_COLOR=1 or `pytest -s` in a terminal.
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
        # only-if-needed is pip's current default, but it is the load-bearing
        # part: eager would re-resolve the existing torch build.
        cmd = ips._build_pip_cmd(("--upgrade-package", "unsloth"))
        assert cmd[cmd.index("--upgrade-strategy") + 1] == "only-if-needed"

    def test_names_already_present_are_not_duplicated(self):
        cmd = ips._build_pip_cmd(("--no-deps", "--upgrade-package", "unsloth", "unsloth"))
        assert cmd.count("unsloth") == 1

    def test_commands_without_the_flag_are_untouched(self):
        cmd = ips._build_pip_cmd(("--no-cache-dir", "somepackage"))
        assert cmd == [sys.executable, "-m", "pip", "install", "--no-cache-dir", "somepackage"]
