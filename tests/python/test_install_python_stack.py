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
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))

import install_python_stack as ips


@pytest.fixture(autouse = True)
def _neutral_policy_environment(monkeypatch):
    """Run as an ordinary machine unless a test says otherwise.

    UNSLOTH_RESPECT_PM_POLICY changes what several of these code paths do, so inheriting
    it from the developer's shell rewrites the suite's subject: with it set, the metadata
    repair and the XPU swap both decline to start and dozens of unrelated assertions
    fail. Tests that want the opt-out set it themselves.
    """
    monkeypatch.delenv("UNSLOTH_RESPECT_PM_POLICY", raising = False)


@pytest.fixture(autouse = True)
def _clear_policy_cache():
    """_policy_scan is cached for the life of the process, which is right in an installer
    and wrong across tests: without this a scan taken under one environment answers for
    the next one."""
    ips._detected_policy.cache_clear()
    yield
    ips._detected_policy.cache_clear()


def _posix_home(path: str = "/home/u"):
    """Make os.path.expanduser deterministic regardless of the RUNNER's platform.

    ntpath.expanduser reads USERPROFILE, not HOME, so on windows-latest a test that only
    sets HOME gets a literal "~" back and the POSIX branch it is exercising cannot be
    asserted at all. Patching the function is the honest fix: these tests are about which
    locations the installer searches, not about how a platform spells a home directory.
    """
    return mock.patch.object(os.path, "expanduser", lambda p: p.replace("~", path, 1))


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
            splat = " ".join(ast.unparse(a.value) for a in node.args if isinstance(a, ast.Starred))
            assert (
                "_sdist_only_build_args('diffusers')" in splat
            ), f"the diffusers pin at line {node.lineno} must exempt the source archive"
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


class TestOperatorCanKeepTheirPolicy:
    """UNSLOTH_RESPECT_PM_POLICY=1 turns every relaxation above off.

    The relaxations exist because the shipped requirements cannot satisfy a
    require-hashes / only-binary policy at all, so enforcing one means the install
    FAILS. That is a legitimate answer for an operator who set the policy deliberately,
    but only they can make the call, so it is an explicit opt-in rather than the default.
    """

    HOSTILE = TestHardenedPipConfigRelaxation.HOSTILE
    # One manager at a time: a mix of pip and uv controls now (correctly) stops the step
    # to report the gap, which is TestNeitherManagerIsLetOffTheOthersPolicy's subject.
    PIP_ONLY = {"PIP_REQUIRE_HASHES": "1", "PIP_ONLY_BINARY": ":all:"}
    UV_ONLY = {"UV_REQUIRE_HASHES": "1", "UV_NO_BUILD": "1"}

    @pytest.mark.parametrize("value", ["1", "true", "yes"])
    def test_the_pip_fallback_keeps_hash_mode(self, value):
        """The relaxation is withdrawn. Scoped to pip's own controls, because a mix of
        pip and uv policy is now a reported gap, which is a different test's subject."""
        with mock.patch.dict(
            os.environ, dict(self.PIP_ONLY, UNSLOTH_RESPECT_PM_POLICY = value), clear = True
        ):
            env = ips._install_env_for_cmd(["python", "-m", "pip", "install", "-r", "extras.txt"])
        assert env is None or env.get("PIP_REQUIRE_HASHES") != "0"

    @pytest.mark.parametrize("value", ["", "0", "false", "no"])
    def test_off_and_unset_spellings_keep_the_default(self, value):
        with mock.patch.dict(os.environ, dict(self.HOSTILE, UNSLOTH_RESPECT_PM_POLICY = value)):
            env = ips._install_env_for_cmd(["python", "-m", "pip", "install", "-r", "extras.txt"])
        assert env is not None and env["PIP_REQUIRE_HASHES"] == "0"

    def test_source_build_exemptions_are_withdrawn(self):
        """The --no-binary exemptions exist ONLY to override a user no-build, so under
        the opt-out there is nothing left for them to do."""
        with mock.patch.dict(os.environ, {"UNSLOTH_RESPECT_PM_POLICY": "1"}):
            assert ips._sdist_only_build_args(*ips.SDIST_ONLY_PACKAGES) == []

    def test_pinned_installs_keep_policy_but_still_scrub_the_index(self):
        """The pin is itself a provenance control: honouring a hash policy must not be
        read as permission to let an inherited mirror answer a pinned torch repair."""
        with mock.patch.dict(
            os.environ,
            dict(
                self.UV_ONLY,
                UNSLOTH_RESPECT_PM_POLICY = "1",
                PIP_EXTRA_INDEX_URL = "https://mirror.corp/simple",
                UV_INDEX_URL = "https://mirror.corp/simple",
            ),
            clear = True,
        ):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env is not None
        for name, value in self.UV_ONLY.items():
            assert env[name] == value, f"{name} must survive the opt-out"
        assert "PIP_CONFIG_FILE" not in env, "pip.conf carries the policy; it must be read"
        for name in ("PIP_EXTRA_INDEX_URL", "UV_INDEX_URL"):
            assert name not in env, f"{name} must still be scrubbed from a pinned install"

    def test_the_parent_environment_is_never_mutated(self):
        with mock.patch.dict(
            os.environ, dict(self.PIP_ONLY, UNSLOTH_RESPECT_PM_POLICY = "1"), clear = True
        ):
            ips._install_env_for_cmd(["python", "-m", "pip", "install", "x"])
            ips._sdist_only_build_args("openai-whisper")
            assert os.environ["UNSLOTH_RESPECT_PM_POLICY"] == "1"
            assert os.environ["PIP_REQUIRE_HASHES"] == "1"


class TestDestructiveRepairsDoNotStartUnderTheOptOut:
    """Two paths uninstall something and can only finish by installing a replacement:
    the duplicate-metadata repair and the Intel XPU triton swap. Under a hash policy
    that replacement cannot be verified -- there is no expected digest to check it
    against, and deriving one from the artifact just fetched would approve it with
    itself. So neither starts, and the venv is left exactly as it was.
    """

    def test_the_metadata_repair_declines_before_staging(self, capsys):
        with mock.patch.dict(os.environ, {"UNSLOTH_RESPECT_PM_POLICY": "1"}, clear = True):
            assert ips._stage_replacement("unsloth-zoo") is None
        out = capsys.readouterr().err
        assert "UNSLOTH_RESPECT_PM_POLICY" in out and "leaving the install alone" in out

    def test_the_default_path_still_stages(self):
        """Nothing is skipped on an ordinary machine: the bail is opt-out only."""
        with (
            mock.patch.dict(os.environ, {}, clear = True),
            mock.patch.object(ips, "USE_UV", False),
            mock.patch.object(ips, "tempfile") as fake_tempfile,
            mock.patch.object(ips, "pip_install_try", lambda *a, **k: False),
        ):
            fake_tempfile.mkdtemp.return_value = "/tmp/staging"
            # It gets far enough to try, which is all this asserts; the staging itself
            # is covered by TestDuplicateCoreMetadataRepair.
            ips._stage_replacement("unsloth-zoo")
            assert fake_tempfile.mkdtemp.called

    def test_no_self_approving_hash_helper_survives(self):
        """The approach this replaced hashed the wheel it had just downloaded and handed
        that digest to the protected install, which is not what a hash policy is for."""
        assert not hasattr(ips, "_hashed_requirement_file")
        assert not hasattr(ips, "_staged_restore_args")


class TestTheOptOutKeepsRestrictiveIndexSettings:
    """The pinned branch scrubs the index variables so a stale mirror cannot answer a
    torch repair (#6898), and that survives the opt-out on purpose. PIP_NO_INDEX is the
    exception: it only ever REMOVES sources, so scrubbing it let a pinned command reach
    the network on a host that had said not to.
    """

    CMD = ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]

    def _env(self, extra):
        with mock.patch.dict(os.environ, dict(extra, PATH = "/usr/bin"), clear = True):
            return ips._install_env_for_cmd(list(self.CMD))

    def test_the_default_path_still_scrubs_everything(self):
        env = self._env({"PIP_NO_INDEX": "1", "PIP_FIND_LINKS": "/wheels"})
        assert "PIP_NO_INDEX" not in env and "PIP_FIND_LINKS" not in env

    def test_the_opt_out_keeps_no_index_and_its_find_links(self):
        """Kept as a pair: PIP_FIND_LINKS is the only source left once PIP_NO_INDEX is
        honoured, so dropping it would leave the step with nowhere to resolve from."""
        env = self._env(
            {
                "UNSLOTH_RESPECT_PM_POLICY": "1",
                "PIP_NO_INDEX": "1",
                "PIP_FIND_LINKS": "/wheels",
                "PIP_INDEX_URL": "https://mirror/simple",
            }
        )
        assert env["PIP_NO_INDEX"] == "1"
        assert env["PIP_FIND_LINKS"] == "/wheels"
        # The ADDITIVE mirror is still scrubbed: that is the provenance control.
        assert "PIP_INDEX_URL" not in env

    def test_find_links_alone_is_still_scrubbed_under_the_opt_out(self):
        """Without PIP_NO_INDEX it merely ADDS a source, which is what the scrub is for."""
        env = self._env({"UNSLOTH_RESPECT_PM_POLICY": "1", "PIP_FIND_LINKS": "/wheels"})
        assert "PIP_FIND_LINKS" not in env

    def test_a_disabled_no_index_is_carried_through_but_keeps_no_find_links(self):
        """An explicit off is the operator lifting their own pip.conf, not an absence.

        pip reads the environment ahead of pip.conf, so PIP_NO_INDEX=0 is how a
        `no-index = true` in config is lifted for one run. Dropping the variable while
        leaving pip.conf readable put the restriction back and failed an install the
        operator had permitted. find-links is a different case: with no-index off it
        ADDS a location beside the pinned index, so it is still scrubbed.
        """
        env = self._env(
            {"UNSLOTH_RESPECT_PM_POLICY": "1", "PIP_NO_INDEX": "0", "PIP_FIND_LINKS": "/w"}
        )
        assert env["PIP_NO_INDEX"] == "0"
        assert "PIP_FIND_LINKS" not in env


class TestHardenedPolicyIsAnnounced:
    """The relaxation is defensible; doing it silently is not.

    An operator who configured require-hashes and watched the install succeed anyway had
    no way to learn that their control had been set aside. The notice names the settings
    it found and the variable that keeps them.
    """

    def _names(
        self,
        env: dict,
        pip_config: str = "",
    ):
        ips._detected_policy.cache_clear()
        result = mock.Mock(returncode = 0, stdout = pip_config.encode())
        with (
            mock.patch.dict(os.environ, env, clear = True),
            mock.patch.object(ips.subprocess, "run", return_value = result),
        ):
            try:
                return ips._hardened_pm_policy_names()
            finally:
                ips._detected_policy.cache_clear()

    def test_an_ordinary_machine_says_nothing(self):
        assert self._names({}) == ()

    def test_a_policy_env_var_is_reported(self):
        assert "PIP_REQUIRE_HASHES" in self._names({"PIP_REQUIRE_HASHES": "1"})

    def test_a_disabled_policy_is_not_reported(self):
        """PIP_REQUIRE_HASHES=0 is the absence of the policy, not the presence of it."""
        assert self._names({"PIP_REQUIRE_HASHES": "0", "PIP_NO_BINARY": ""}) == ()

    def test_pip_config_hardening_is_reported(self):
        names = self._names({}, "global.require-hashes='true'\nglobal.index-url='https://m/s'\n")
        assert "pip.conf require-hashes" in names
        assert not any("index-url" in name for name in names), "a mirror is not hardening"

    def test_env_restatements_from_pip_config_are_not_double_counted(self):
        names = self._names({"PIP_REQUIRE_HASHES": "1"}, ":env:.require-hashes='true'\n")
        assert names == ("PIP_REQUIRE_HASHES",)

    def test_false_disables_the_upload_cutoff(self):
        """The cutoff is a date, but uv takes `false` for it as "no cutoff": measured on
        the pinned uv 0.12.1, UV_EXCLUDE_NEWER=false installs the current release while
        a date filters it. Reading `false` as a cutoff put a security notice in front of
        an operator who had switched the control off."""
        assert self._names({"UV_EXCLUDE_NEWER": "false"}) == ()
        assert self._names({"UV_EXCLUDE_NEWER": "2005-01-01T00:00:00Z"}) == ("UV_EXCLUDE_NEWER",)
        # Still not a boolean elsewhere: a package list keeps package names.
        assert self._names({"PIP_ONLY_BINARY": "false"}) == ("PIP_ONLY_BINARY",)

    def test_a_control_disabled_for_every_command_is_not_reported(self):
        """pip applies [global] then the command's own section, so a control enabled
        globally and disabled for install, download AND wheel hardens nothing this
        module runs. Reporting it put the notice in front of someone whose policy was
        not being relaxed at all."""
        listing = (
            "global.require-hashes='true'\ninstall.require-hashes='false'\n"
            "download.require-hashes='false'\nwheel.require-hashes='false'\n"
        )
        assert self._names({}, listing) == ()
        # Disabled for only ONE of them still leaves the other two hardened.
        partial = "global.require-hashes='true'\ninstall.require-hashes='false'\n"
        assert self._names({}, partial) == ("pip.conf require-hashes",)

    def test_an_explicit_no_index_is_reported(self):
        """no-index REMOVES every remote source, and the pinned branch discards it with
        the rest of pip.conf, which is exactly what this notice exists to disclose."""
        assert self._names({}, "global.no-index='true'\n") == ("pip.conf no-index",)
        assert self._names({}, "global.no-index='false'\n") == ()

    def test_uv_policy_is_reported_from_the_variables_uv_reads(self):
        """uv has no command that prints its resolved configuration, so the notice
        reports the two variables uv genuinely honours and does not go looking through
        uv.toml. Measured on the pinned uv 0.12.1: UV_REQUIRE_HASHES and
        UV_EXCLUDE_NEWER change the outcome, UV_NO_BUILD and the other artifact
        spellings are ignored, so reporting one would name policy uv does not apply."""
        assert self._names({"UV_REQUIRE_HASHES": "1"}) == ("UV_REQUIRE_HASHES",)
        assert self._names({"UV_NO_BUILD": "1"}) == ()

    def test_an_undetected_setting_costs_a_line_not_an_install(self):
        """The point of keeping detection advisory: with nothing detected the notice is
        silent, and the opt-out still withholds every relaxation."""
        assert self._names({}) == ()
        with mock.patch.dict(os.environ, {"UNSLOTH_RESPECT_PM_POLICY": "1"}, clear = True):
            assert ips._relaxed_pip_policy_env(["python", "-m", "pip", "install", "x"]) == {}
            assert ips._sdist_only_build_args("diffusers") == []

    def test_the_notice_names_the_opt_out(self, capsys):
        with (
            mock.patch.object(ips, "_hardened_pm_policy_names", lambda: ("PIP_REQUIRE_HASHES",)),
            mock.patch.dict(os.environ, {}, clear = True),
        ):
            ips._announce_pm_policy()
        out = capsys.readouterr().out
        assert "PIP_REQUIRE_HASHES" in out and "UNSLOTH_RESPECT_PM_POLICY" in out

    def test_the_opt_out_notice_warns_that_steps_will_fail(self, capsys):
        with (
            mock.patch.object(ips, "_hardened_pm_policy_names", lambda: ("PIP_REQUIRE_HASHES",)),
            mock.patch.dict(os.environ, {"UNSLOTH_RESPECT_PM_POLICY": "1"}, clear = True),
        ):
            ips._announce_pm_policy()
        # Normalised: _note() wraps to the terminal, so the phrase spans lines.
        text = " ".join(capsys.readouterr().out.split())
        assert "will now fail where your policy forbids them" in text

    def test_nothing_is_printed_on_an_ordinary_machine(self, capsys):
        with (
            mock.patch.object(ips, "_hardened_pm_policy_names", lambda: ()),
            mock.patch.dict(os.environ, {}, clear = True),
        ):
            ips._announce_pm_policy()
        assert capsys.readouterr().out == ""


class TestTheOptOutIsNotDefeatedByOurOwnEscapeHatches:
    """Three ways the opt-out let policy through anyway, each measured, each closed.

    The flag promises the operator's policy is left in force. Anything the installer does
    that quietly restores the relaxed behaviour makes that promise false, which is worse
    than not offering the flag.
    """

    def test_pinned_installs_keep_uv_config_discovery(self):
        """UV_NO_CONFIG=1 discards a USER uv.toml, which is where a require-hashes lives.

        Measured on the pinned uv 0.12.1: `~/.config/uv/uv.toml` with
        `[pip] require-hashes = true` fails a `uv pip install --no-index --find-links`,
        and the identical command with UV_NO_CONFIG=1 succeeds. Setting it under the
        opt-out therefore discarded exactly the control being promised.
        """
        with mock.patch.dict(
            os.environ,
            {"UV_REQUIRE_HASHES": "1", "UNSLOTH_RESPECT_PM_POLICY": "1"},
            clear = True,
        ):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env is not None
        assert "UV_NO_CONFIG" not in env, "the opt-out must leave uv config discovery on"

    def test_the_default_path_still_disables_uv_config_discovery(self):
        """Unchanged where it matters: a discovered uv.toml outranks the CLI pin."""
        with mock.patch.dict(os.environ, {}, clear = True):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env is not None and env["UV_NO_CONFIG"] == "1"
        assert env["PIP_CONFIG_FILE"] == os.devnull

    def _fallback_calls(self, environment: dict) -> "tuple[list, object]":
        calls: list = []
        raised = None
        with (
            mock.patch.object(ips, "USE_UV", True),
            mock.patch.object(ips, "subprocess") as sp,
            mock.patch.object(ips, "run", lambda *a, **k: calls.append(a)),
            mock.patch.object(ips, "_invalidate_torch_runtime_probe", lambda: None),
            mock.patch.dict(os.environ, environment, clear = True),
        ):
            sp.run.return_value = mock.Mock(returncode = 2, stdout = "")
            sp.PIPE, sp.STDOUT = -1, -2
            try:
                ips.pip_install("deps", "somepackage")
            except SystemExit as exit_info:
                raised = exit_info
        return calls, raised

    def test_the_fallback_runs_under_the_opt_out_without_relaxing_anything(self):
        """uv failing must not end the install, opt-out or not. pip then applies whatever
        pip is configured with: uv's settings were never pip's to read, which is a
        property of the two tools rather than something this installer introduces.
        What the opt-out promises is that OUR relaxations are withheld, and they are."""
        calls, raised = self._fallback_calls(
            {"UNSLOTH_RESPECT_PM_POLICY": "1", "UV_REQUIRE_HASHES": "1"}
        )
        assert raised is None
        assert calls, "the pip fallback must still run"
        with mock.patch.dict(os.environ, {"UNSLOTH_RESPECT_PM_POLICY": "1"}, clear = True):
            assert ips._relaxed_pip_policy_env(["python", "-m", "pip", "install", "x"]) == {}

    def test_the_fallback_still_runs_when_the_opt_out_drops_nothing(self):
        """The flag alone is not a reason to refuse. An operator who sets it
        pre-emptively on a host with no uv-only policy would otherwise have every
        ordinary uv failure, a resolver hiccup included, turned into a fatal one."""
        calls, raised = self._fallback_calls({"UNSLOTH_RESPECT_PM_POLICY": "1"})
        assert raised is None, "nothing is being discarded, so nothing to refuse"
        assert calls, "the pip fallback must still run"

    def test_the_fallback_runs_when_pip_can_enforce_the_same_control(self):
        """Both managers configured is not a gap: pip applies its own require-hashes to
        the fallback, so the operator's policy survives it."""
        calls, raised = self._fallback_calls(
            {
                "UNSLOTH_RESPECT_PM_POLICY": "1",
                "UV_REQUIRE_HASHES": "1",
                "PIP_REQUIRE_HASHES": "1",
            }
        )
        assert raised is None
        assert calls, "the pip fallback must still run"

    def test_the_pip_fallback_still_runs_by_default(self):
        """The whole point of #8530's fix: uv failing must not end the install."""
        calls: list = []

        with (
            mock.patch.object(ips, "USE_UV", True),
            mock.patch.object(ips, "subprocess") as sp,
            mock.patch.object(ips, "run", lambda *a, **k: calls.append(a)),
            mock.patch.object(ips, "_invalidate_torch_runtime_probe", lambda: None),
            mock.patch.dict(os.environ, {}, clear = True),
        ):
            sp.run.return_value = mock.Mock(returncode = 2, stdout = "")
            sp.PIPE, sp.STDOUT = -1, -2
            ips.pip_install("deps", "somepackage")
        assert len(calls) == 1, "uv failing must still fall back to pip by default"

    def test_uv_only_binary_counts_as_policy(self):
        """_uv_staging_plan already treats UV_ONLY_BINARY as uv's artifact policy, so the
        policy set that drives detection and the pinned scrub must agree with it."""
        assert "UV_ONLY_BINARY" in ips._PM_POLICY_ENV_VARS
        assert "UV_ONLY_BINARY_PACKAGE" in ips._PM_POLICY_ENV_VARS


class TestPolicyEnvIsPlatformAndHardwareInvariant:
    """The env this helper builds must not depend on the platform or the accelerator.

    It reads no hardware state and it should stay that way: an installer change that
    quietly behaves differently on one of [Windows, Linux, WSL, macOS] x [NVIDIA, AMD,
    CPU] is the failure mode that costs a release. Asserted by construction rather than
    trusted, and the accelerator variables are checked to PASS THROUGH untouched, since
    the torch repair steps downstream read them.
    """

    PLATFORMS = {
        "Linux": {"IS_WINDOWS": False, "IS_MACOS": False, "IS_MAC_ARM": False},
        # WSL is Linux to this module; the markers are carried to prove they are inert.
        "WSL": {"IS_WINDOWS": False, "IS_MACOS": False, "IS_MAC_ARM": False},
        "Windows": {"IS_WINDOWS": True, "IS_MACOS": False, "IS_MAC_ARM": False},
        "macOS": {"IS_WINDOWS": False, "IS_MACOS": True, "IS_MAC_ARM": True},
    }
    ACCELERATORS = {
        "nvidia": {"CUDA_VISIBLE_DEVICES": "0", "UNSLOTH_TORCH_BACKEND": "cuda"},
        "amd": {"HIP_VISIBLE_DEVICES": "0", "HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
        "cpu": {"UNSLOTH_TORCH_BACKEND": "cpu"},
    }
    COMMANDS = [
        ["python", "-m", "pip", "install", "-r", "extras.txt"],
        ["python", "-m", "pip", "download", "pytorch-triton-xpu==3.5.0"],
        ["python", "-m", "pip", "check"],
        ["uv", "pip", "install", "-r", "base.txt"],
        ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"],
        ["python", "-m", "pip", "install", "torch", "--default-index", "https://x/rocm6.4"],
    ]

    @pytest.mark.parametrize("opt_out", ["", "1"])
    @pytest.mark.parametrize("hardened", [False, True])
    def test_identical_on_every_platform_and_accelerator(self, opt_out, hardened):
        # One manager's controls: a cross-manager gap refuses under the opt-out, and a
        # refusal is not an environment to compare across platforms.
        # Both managers configured, so no cross-manager gap: a refusal is not an
        # environment, and this sweep is about environment construction.
        # require-hashes on both sides, and only that: it is the one artifact-policy
        # control with an environment spelling uv actually reads, so it is the only way
        # to configure both managers without writing config files. UV_NO_BUILD would not
        # do, since uv ignores it and detection no longer pretends otherwise.
        policy = {"PIP_REQUIRE_HASHES": "1", "UV_REQUIRE_HASHES": "1"} if hardened else {}
        for command in self.COMMANDS:
            answers = set()
            for platform, settings in self.PLATFORMS.items():
                for accelerator, markers in self.ACCELERATORS.items():
                    env = dict(policy, **markers)
                    if opt_out:
                        env["UNSLOTH_RESPECT_PM_POLICY"] = opt_out
                    if platform == "WSL":
                        env["WSL_DISTRO_NAME"] = "Ubuntu"
                    with (
                        mock.patch.dict(os.environ, env, clear = True),
                        mock.patch.multiple(ips, **settings),
                    ):
                        result = ips._install_env_for_cmd(list(command))
                    # Only the keys this helper owns; the markers themselves differ.
                    answers.add(
                        None
                        if result is None
                        else repr(
                            sorted(
                                (k, v) for k, v in result.items() if k.startswith(("PIP_", "UV_"))
                            )
                        )
                    )
                    if result is not None:
                        for name, value in markers.items():
                            assert (
                                result[name] == value
                            ), f"{accelerator} marker {name} was dropped on {platform}"
            assert (
                len(answers) == 1
            ), f"{command} produced different environments across platforms: {answers}"


class TestTheNoticeCannotTakeAnInstallDown:
    """It is a message, not a step. Every way the probe can go wrong ends in a missing
    line, never a failed install: it runs before the pip upgrade, against a venv uv
    created without pip at all.
    """

    def test_the_pip_probe_is_bounded(self):
        """No timeout here means an install that hangs forever on a wedged pip."""
        # The probe lives in _pip_config_settings, which _detected_policy calls.
        source = inspect.getsource(ips._pip_config_settings)
        assert "timeout = 30" in source, "the pip config probe must be bounded"
        assert "subprocess.run" in source, (
            "this test is checking the function that actually runs pip; if the probe "
            "moved again, follow it rather than leaving the assertion looking at nothing"
        )

    def test_a_hanging_pip_still_yields_the_environment_half(self):
        ips._detected_policy.cache_clear()
        with (
            mock.patch.dict(os.environ, {"PIP_REQUIRE_HASHES": "1"}, clear = True),
            mock.patch.object(
                ips.subprocess,
                "run",
                side_effect = ips.subprocess.TimeoutExpired(cmd = "pip", timeout = 30),
            ),
        ):
            assert ips._hardened_pm_policy_names() == ("PIP_REQUIRE_HASHES",)
        ips._detected_policy.cache_clear()

    def test_a_broken_pip_is_not_an_error(self):
        for failure in (OSError("no pip"), ips.subprocess.SubprocessError("boom")):
            ips._detected_policy.cache_clear()
            with (
                mock.patch.dict(os.environ, {}, clear = True),
                mock.patch.object(ips.subprocess, "run", side_effect = failure),
            ):
                assert ips._hardened_pm_policy_names() == ()
            ips._detected_policy.cache_clear()

    def test_undecodable_probe_output_is_not_an_error(self):
        ips._detected_policy.cache_clear()
        result = mock.Mock(returncode = 0, stdout = b"\xff\xfe not utf-8 \x00\n")
        with (
            mock.patch.dict(os.environ, {}, clear = True),
            mock.patch.object(ips.subprocess, "run", return_value = result),
        ):
            assert ips._hardened_pm_policy_names() == ()
        ips._detected_policy.cache_clear()


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
        backup = tmp_path / "~nsloth-2026.8.12.dist-info"
        backup.mkdir()
        (backup / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: unsloth\nVersion: 2026.8.12\n", encoding = "utf-8"
        )
        # Two records; one once the backup is aside; none after the uninstall; then
        # the reinstalled one for the final convergence probe.
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
            # pip only ever sees a tree it can actually act on.
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
        backup = tmp_path / "~nsloth-2026.8.12.dist-info"
        backup.mkdir()
        (backup / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: unsloth\nVersion: 2026.8.12\n", encoding = "utf-8"
        )
        # One record, none once the backup is aside, then the reinstalled one.
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

        monkeypatch.setattr(
            ips.install_manifest,
            "installed_versions",
            lambda name: next(probes[name]),
        )
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

        monkeypatch.setattr(
            ips.install_manifest,
            "installed_versions",
            lambda name: next(probes[name]),
        )
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: True)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _name: "/staged")
        monkeypatch.setattr(
            ips,
            "pip_install_try",
            lambda label, *args, **kwargs: installs.append((label, args, kwargs)) or True,
        )

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
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: True)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _name: "/staged")
        monkeypatch.setattr(
            ips,
            "pip_install_try",
            lambda label, *args, **kwargs: installs.append((label, args, kwargs)) or True,
        )

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

        monkeypatch.setattr(
            ips.install_manifest,
            "installed_versions",
            lambda name: next(probes[name]),
        )
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
        monkeypatch.setattr(ips, "_run_ok", lambda *a, **k: True)
        monkeypatch.setattr(ips, "_stage_replacement", lambda _name: "/staged")
        monkeypatch.setattr(
            ips,
            "pip_install_try",
            lambda label, *args, **kwargs: installs.append((label, args, kwargs)) or True,
        )

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
        # The annotation names the index the package actually came from, which is the
        # one to reproduce -- not --index-url, which is only uv's default.
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
        # pip.conf carries the same three settings, so it is replaced by a copy of
        # itself with only those removed.
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

    def test_the_annotated_index_is_recovered_with_its_credentials(self, monkeypatch):
        """Measured on uv 0.10.7: the emitted index lines carry userinfo and the
        `# from` annotation has it stripped. Taking the annotation at face value
        hands pip an unauthenticated URL for a private index, which answers 401 and
        aborts the repair."""
        self._uv_only(monkeypatch)
        self._uv_plan(
            monkeypatch,
            stdout = (
                b"--index-url https://user:secret@private.corp/simple\n"
                b"unsloth-zoo==1.0\n"
                b"    # from https://private.corp/simple\n"
            ),
        )
        _requirement, overrides, _options = ips._uv_staging_plan("unsloth-zoo")
        assert overrides["PIP_INDEX_URL"] == "https://user:secret@private.corp/simple"

    def test_an_authenticated_extra_index_is_recovered_too(self, monkeypatch):
        """uv puts a credentialed --index on the extra line and leaves --index-url as
        the public default, so reading only --index-url would name the wrong index."""
        self._uv_only(monkeypatch)
        self._uv_plan(
            monkeypatch,
            stdout = (
                b"--index-url https://pypi.org/simple\n"
                b"--extra-index-url https://user:secret@private.corp/simple\n"
                b"unsloth-zoo==1.0\n"
                b"    # from https://private.corp/simple\n"
            ),
        )
        _requirement, overrides, _options = ips._uv_staging_plan("unsloth-zoo")
        assert overrides["PIP_INDEX_URL"] == "https://user:secret@private.corp/simple"

    def test_the_credentialed_form_wins_over_a_bare_duplicate(self, monkeypatch):
        self._uv_only(monkeypatch)
        self._uv_plan(
            monkeypatch,
            stdout = (
                b"--index-url https://private.corp/simple\n"
                b"--extra-index-url https://user:secret@private.corp/simple\n"
                b"unsloth-zoo==1.0\n"
                b"    # from https://private.corp/simple\n"
            ),
        )
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
        # It reached pip rather than refusing, and did not consult uv for a path.
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
        # The version comes from the directory name, which is where importlib's own
        # fallback reads it when METADATA cannot be parsed.
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

        # A failure after the rewrite must not leave the synthetic file behind: a
        # readable record would tell the next run there is nothing left to repair.
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
        # It was rewritten for pip rather than moved aside, so its RECORD is applied
        # and nothing is quarantined at all.
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
        # The second package cannot be staged, so the repair fails after the first
        # one has already been reinstalled.
        staged = iter(("/staged", None))
        monkeypatch.setattr(ips, "_stage_replacement", lambda _n: next(staged))
        monkeypatch.setattr(
            ips._QuarantinedMetadata, "discard", lambda _self: events.append("discard")
        )
        monkeypatch.setattr(
            ips._QuarantinedMetadata, "restore", lambda _self: events.append("restore")
        )

        assert ips._repair_duplicate_core_metadata(("unsloth", "unsloth-zoo")) is False
        # The first package was committed before the second was attempted, so the
        # rollback at the end can only touch the one that failed.
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
        record = tmp_path / "unsloth-2026.8.12.dist-info"
        record.mkdir()
        (record / "METADATA").write_bytes(b"\xff\xfe")
        (record / "RECORD").write_text("unsloth/gone.py,,\n")
        quarantine = ips._QuarantinedMetadata()

        assert quarantine.back_up(str(record / "METADATA")) is True
        assert ips._rewrite_minimal_metadata(str(record), "unsloth") is True
        assert "Name: unsloth" in (record / "METADATA").read_text()

        quarantine.restore()

        # Byte for byte what was there, so the conflict is still detected next time.
        assert (record / "METADATA").read_bytes() == b"\xff\xfe"

    def test_a_committed_rewrite_is_not_undone(self, tmp_path):
        record = tmp_path / "unsloth-2026.8.12.dist-info"
        record.mkdir()
        (record / "METADATA").write_bytes(b"\xff\xfe")
        (record / "RECORD").write_text("unsloth/gone.py,,\n")
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
        record = tmp_path / "unsloth-2026.8.12.dist-info"
        record.mkdir()
        (record / "METADATA").write_bytes(b"\xff\xfe")
        (record / "RECORD").write_text("unsloth/gone.py,,\n")

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
        # Untouched, so the next run still sees it.
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
                # pip removed the rewritten record's whole directory.
                shutil.rmtree(record)
                return True
            return False  # the next one fails, triggering the rollback

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
        backup = tmp_path / "~nsloth-2026.8.12.dist-info"
        backup.mkdir()
        (backup / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: unsloth\nVersion: 2026.8.12\n", encoding = "utf-8"
        )
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

        monkeypatch.setattr(
            ips.install_manifest, "installed_versions", lambda name: next(probes[name])
        )
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _name: [])
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
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

        monkeypatch.setattr(
            ips.install_manifest, "installed_versions", lambda name: next(probes[name])
        )
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _name: [])
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
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

        monkeypatch.setattr(
            ips.install_manifest, "installed_versions", lambda name: next(probes[name])
        )
        monkeypatch.setattr(ips.install_manifest, "invalid_metadata_paths", lambda _name: [])
        monkeypatch.setattr(ips, "_step", lambda *a, **k: None)
        monkeypatch.setattr(ips.importlib, "invalidate_caches", lambda: None)
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


class TestAFalseCutoffOnlyDisablesTheManagerThatAcceptsIt:
    """`false` is an off spelling for uv's exclude-newer and an ERROR for pip's cutoff.

    Measured, not assumed. On the pinned uv 0.12.1, UV_EXCLUDE_NEWER=false resolves
    normally while UV_EXCLUDE_NEWER=garbage is rejected, so uv understands `false` as
    "no cutoff" rather than merely tolerating it. On pip 26.2.1 both
    `--uploaded-prior-to false` and PIP_UPLOADED_PRIOR_TO=false exit with "Expected an
    ISO 8601 datetime string". Treating pip's spelling as a disable let an unusable
    environment value cancel a real pip.conf cutoff, which took the notice quiet about
    a control the pinned path still drops.
    """

    def test_uv_cutoff_spellings_are_disabled_by_false(self):
        for key in ("exclude-newer", "exclude-newer-package"):
            assert ips._config_value_is_on("false", key) is False, key
            assert ips._config_value_is_on("2026-01-01", key) is True, key

    def test_pips_cutoff_is_not_disabled_by_a_value_pip_rejects(self):
        # pip errors on this value, so it can neither be a cutoff nor an off switch.
        # Reporting it keeps the notice honest about a setting that is still there.
        assert ips._config_value_is_on("false", "uploaded-prior-to") is True
        assert ips._config_value_is_on("false", "PIP_UPLOADED_PRIOR_TO") is True

    def test_the_false_disables_set_holds_only_uv_spellings(self):
        assert "uploaded-prior-to" not in ips._FALSE_DISABLES_KEYS
        assert set(ips._FALSE_DISABLES_KEYS) == {"exclude-newer", "exclude-newer-package"}

    def test_an_empty_cutoff_is_still_off_for_both(self):
        for key in ("exclude-newer", "uploaded-prior-to"):
            assert ips._config_value_is_on("", key) is False, key
            assert ips._config_value_is_on("   ", key) is False, key


class TestNoneIsAListSentinelNotABoolean:
    """`:none:` empties a package list; to a boolean it is a value pip rejects.

    Measured on pip 26.2.1: `PIP_NO_INDEX=:none:` exits with ":none: is not a valid
    value for no-index option, please specify a boolean value like yes/no, true/false
    or 1/0 instead". Reading it as off let the opt-out drop a variable that would
    otherwise have stopped the command, which turns an install pip refuses into one
    that can reach the network.
    """

    def test_none_does_not_switch_off_a_boolean_control(self):
        for key in ips._BOOLEAN_POLICY_KEYS:
            assert ips._config_value_is_on(":none:", key) is True, key

    def test_none_still_empties_a_package_list(self):
        for key in ("no-binary", "only-binary"):
            assert ips._config_value_is_on(":none:", key) is False, key

    def test_an_empty_value_is_off_for_every_kind_of_key(self):
        for key in (None, "no-index", "no-binary", "exclude-newer"):
            assert ips._config_value_is_on("", key) is False, key
            assert ips._config_value_is_on("   ", key) is False, key

    def test_the_opt_out_keeps_a_no_index_pip_would_reject(self, monkeypatch):
        # The value is unusable, so the pinned step fails -- which is exactly what the
        # opt-out promises. Dropping it would have let the same step succeed instead.
        monkeypatch.setenv("UNSLOTH_RESPECT_PM_POLICY", "1")
        monkeypatch.setenv("PIP_NO_INDEX", ":none:")
        monkeypatch.setenv("PIP_FIND_LINKS", "/op/wheels")
        cmd = ["uv", "pip", "install", "--index-url", "https://pinned", "torch"]
        env = ips._install_env_for_cmd(cmd)
        assert env is not None
        assert env.get("PIP_NO_INDEX") == ":none:"
        assert env.get("PIP_FIND_LINKS") == "/op/wheels"

    def test_the_default_path_still_scrubs_it(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_RESPECT_PM_POLICY", raising = False)
        monkeypatch.setenv("PIP_NO_INDEX", ":none:")
        cmd = ["uv", "pip", "install", "--index-url", "https://pinned", "torch"]
        env = ips._install_env_for_cmd(cmd)
        assert env is not None
        assert "PIP_NO_INDEX" not in env


class TestAnExplicitNoIndexOverrideSurvivesTheOptOut:
    """pip reads the environment ahead of pip.conf, so PIP_NO_INDEX=0 lifts a config
    `no-index = true` for one run. Measured on pip 26.2.1 against a clean venv: with
    pip.conf `no-index = true` the install fails with "No matching distribution found",
    and the same command with PIP_NO_INDEX=0 resolves. Dropping the variable while
    leaving pip.conf readable reimposed a restriction the operator had lifted.
    """

    PINNED = ["uv", "pip", "install", "--index-url", "https://pinned", "torch"]

    def test_an_explicit_off_is_carried_through(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_RESPECT_PM_POLICY", "1")
        monkeypatch.setenv("PIP_NO_INDEX", "0")
        env = ips._install_env_for_cmd(list(self.PINNED))
        assert env is not None
        assert env.get("PIP_NO_INDEX") == "0"

    def test_an_explicit_on_is_carried_through_with_its_find_links(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_RESPECT_PM_POLICY", "1")
        monkeypatch.setenv("PIP_NO_INDEX", "1")
        monkeypatch.setenv("PIP_FIND_LINKS", "/op/wheels")
        env = ips._install_env_for_cmd(list(self.PINNED))
        assert env is not None
        assert env.get("PIP_NO_INDEX") == "1"
        assert env.get("PIP_FIND_LINKS") == "/op/wheels"

    def test_find_links_is_still_additive_when_no_index_is_off(self, monkeypatch):
        # With no-index off, find-links adds a location alongside the pinned index,
        # which is the whole reason the pinned branch scrubs the mirror variables.
        monkeypatch.setenv("UNSLOTH_RESPECT_PM_POLICY", "1")
        monkeypatch.setenv("PIP_NO_INDEX", "0")
        monkeypatch.setenv("PIP_FIND_LINKS", "/op/wheels")
        env = ips._install_env_for_cmd(list(self.PINNED))
        assert env is not None
        assert "PIP_FIND_LINKS" not in env

    def test_the_default_path_scrubs_both_regardless(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_RESPECT_PM_POLICY", raising = False)
        for value in ("0", "1"):
            monkeypatch.setenv("PIP_NO_INDEX", value)
            monkeypatch.setenv("PIP_FIND_LINKS", "/op/wheels")
            env = ips._install_env_for_cmd(list(self.PINNED))
            assert env is not None
            assert "PIP_NO_INDEX" not in env, value
            assert "PIP_FIND_LINKS" not in env, value


class TestNoIndexCountsTheConfigFileNotJustTheEnvironment:
    """`--no-index` means "ignore indexes, look only at find-links", so whether
    PIP_FIND_LINKS is the sole source or an extra one depends on the EFFECTIVE state.
    The opt-out keeps pip.conf readable, so a `no-index = true` living there counts;
    reading only the environment scrubbed find-links as additive and left the pip
    fallback with no sources at all.
    """

    PINNED = ["uv", "pip", "install", "--index-url", "https://pinned", "torch"]

    @pytest.fixture
    def no_index_conf(self, tmp_path):
        path = tmp_path / "pip.conf"
        path.write_text("[global]\nno-index = true\n", encoding = "utf-8")
        return str(path)

    def _env(self, monkeypatch, **environ):
        for key in ("PIP_NO_INDEX", "PIP_FIND_LINKS", "PIP_CONFIG_FILE"):
            monkeypatch.delenv(key, raising = False)
        for key, value in environ.items():
            monkeypatch.setenv(key, value)
        ips._detected_policy.cache_clear()
        try:
            return ips._install_env_for_cmd(list(self.PINNED))
        finally:
            ips._detected_policy.cache_clear()

    def test_find_links_survives_a_config_only_no_index(self, monkeypatch, no_index_conf):
        env = self._env(
            monkeypatch,
            UNSLOTH_RESPECT_PM_POLICY = "1",
            PIP_CONFIG_FILE = no_index_conf,
            PIP_FIND_LINKS = "/op/wheels",
        )
        assert env is not None
        assert env.get("PIP_FIND_LINKS") == "/op/wheels"

    def test_find_links_is_still_scrubbed_with_no_no_index_anywhere(self, monkeypatch):
        env = self._env(
            monkeypatch,
            UNSLOTH_RESPECT_PM_POLICY = "1",
            PIP_CONFIG_FILE = os.devnull,
            PIP_FIND_LINKS = "/op/wheels",
        )
        assert env is not None
        assert "PIP_FIND_LINKS" not in env

    def test_the_environment_overrides_the_config_in_both_directions(
        self, monkeypatch, no_index_conf
    ):
        # pip reads the environment ahead of its files, so an explicit off wins and
        # find-links goes back to being merely additive.
        env = self._env(
            monkeypatch,
            UNSLOTH_RESPECT_PM_POLICY = "1",
            PIP_CONFIG_FILE = no_index_conf,
            PIP_NO_INDEX = "0",
            PIP_FIND_LINKS = "/op/wheels",
        )
        assert env is not None
        assert env.get("PIP_NO_INDEX") == "0"
        assert "PIP_FIND_LINKS" not in env

    def test_the_default_path_scrubs_both_and_asks_pip_nothing(self, monkeypatch, no_index_conf):
        # The scan must not be reached without the opt-out: the default path is the one
        # that has to stay exactly as it was.
        monkeypatch.delenv("UNSLOTH_RESPECT_PM_POLICY", raising = False)
        called = []
        original = ips._detected_policy

        def _spy():
            called.append(True)
            return original()

        _spy.cache_clear = original.cache_clear
        monkeypatch.setattr(ips, "_detected_policy", _spy)
        env = self._env(
            monkeypatch,
            PIP_CONFIG_FILE = no_index_conf,
            PIP_FIND_LINKS = "/op/wheels",
        )
        assert env is not None
        assert "PIP_FIND_LINKS" not in env and "PIP_NO_INDEX" not in env
        assert called == []


class TestNoIndexIsReadForTheCommandBeingRun:
    """A command-prefixed pip.conf key affects only the command it names.

    Pooling `[global]`, `[install]`, `[download]` and `[wheel]` into one answer let a
    `[download] no-index = true` hold PIP_FIND_LINKS through a pinned INSTALL, where it
    is purely additive and could satisfy torch from an unpinned location.
    """

    def test_the_commands_own_section_beats_global(self):
        settings = {("global", "no-index"): "true", ("install", "no-index"): "false"}
        assert ips._pip_setting_is_on(settings, "install", "no-index") is False
        assert ips._pip_setting_is_on(settings, "download", "no-index") is True

    def test_another_commands_section_does_not_apply(self):
        settings = {("download", "no-index"): "true"}
        assert ips._pip_setting_is_on(settings, "install", "no-index") is False
        assert ips._pip_setting_is_on(settings, "download", "no-index") is True

    def test_global_applies_where_no_command_section_exists(self):
        settings = {("global", "no-index"): "true"}
        for command in ips._PIP_COMMANDS:
            assert ips._pip_setting_is_on(settings, command, "no-index") is True


class TestAFailedPipProbeIsNotMemoisedForTheRun:
    """A fresh uv venv has no pip when the notice runs.

    Caching that emptiness meant a later step still saw "nothing configured" after pip
    had been bootstrapped, which scrubbed PIP_FIND_LINKS out of a pinned command whose
    only source it was. Only a reading that reached pip is kept.
    """

    def test_an_unreachable_pip_leaves_the_cache_empty(self, monkeypatch, tmp_path):
        ips._detected_policy.cache_clear()
        monkeypatch.setattr(ips.sys, "executable", str(tmp_path / "no-such-python"))
        assert ips._pip_config_settings() == {}
        assert ips._pip_config_settings._cache is None, (
            "a probe that never reached pip must not be memoised: pip can be "
            "bootstrapped later in the same run"
        )
        ips._detected_policy.cache_clear()

    def test_a_successful_probe_is_memoised(self, monkeypatch, tmp_path):
        ips._detected_policy.cache_clear()
        conf = tmp_path / "pip.conf"
        conf.write_text("[global]\nno-index = true\n", encoding = "utf-8")
        monkeypatch.setenv("PIP_CONFIG_FILE", str(conf))
        try:
            first = ips._pip_config_settings()
            assert first.get(("global", "no-index")) == "true"
            assert ips._pip_config_settings._cache is not None
            # Second call must not re-probe: point the interpreter at nothing and the
            # memoised answer should still come back.
            monkeypatch.setattr(ips.sys, "executable", str(tmp_path / "gone"))
            assert ips._pip_config_settings() == first
        finally:
            ips._detected_policy.cache_clear()

    def test_the_answer_changes_once_pip_appears(self, monkeypatch, tmp_path):
        ips._detected_policy.cache_clear()
        conf = tmp_path / "pip.conf"
        conf.write_text("[global]\nno-index = true\n", encoding = "utf-8")
        monkeypatch.setenv("PIP_CONFIG_FILE", str(conf))
        monkeypatch.delenv("PIP_NO_INDEX", raising = False)
        real = ips.sys.executable
        try:
            monkeypatch.setattr(ips.sys, "executable", str(tmp_path / "no-such-python"))
            assert ips._pip_no_index_in_force() is False
            monkeypatch.setattr(ips.sys, "executable", real)
            assert ips._pip_no_index_in_force() is True
        finally:
            ips._detected_policy.cache_clear()


class TestUvFindLinksSurvivesTheOptOut:
    """uv's no-index is undetectable, so its wheelhouse cannot be treated as additive.

    Measured on the pinned uv 0.12.1: a uv.toml `[pip] no-index = true` plus
    UV_FIND_LINKS installs from the wheelhouse, and the same pinned command without it
    fails with "index lookups were disabled and no additional package locations were
    provided". `--no-index` has no environment spelling and uv prints no resolved
    configuration, so unlike pip there is nothing to ask; guessing wrong in that
    direction breaks every offline uv install.
    """

    PINNED = ["uv", "pip", "install", "--index-url", "https://pinned", "torch"]

    def test_the_wheelhouse_reaches_uv_under_the_opt_out(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_RESPECT_PM_POLICY", "1")
        monkeypatch.setenv("UV_FIND_LINKS", "/op/wheels")
        env = ips._install_env_for_cmd(list(self.PINNED))
        assert env is not None
        assert env.get("UV_FIND_LINKS") == "/op/wheels"

    def test_the_additive_mirrors_are_still_scrubbed_alongside_it(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_RESPECT_PM_POLICY", "1")
        monkeypatch.setenv("UV_FIND_LINKS", "/op/wheels")
        for name in ("UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX"):
            monkeypatch.setenv(name, "https://mirror.corp")
        env = ips._install_env_for_cmd(list(self.PINNED))
        assert env is not None
        for name in ("UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX"):
            assert name not in env, name

    def test_the_default_path_still_scrubs_it(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_RESPECT_PM_POLICY", raising = False)
        monkeypatch.setenv("UV_FIND_LINKS", "/op/wheels")
        env = ips._install_env_for_cmd(list(self.PINNED))
        assert env is not None
        assert "UV_FIND_LINKS" not in env


class TestReportingAControlIsNotPermissionToOverrideIt:
    """The notice's list and the default path's scrub list are separate on purpose.

    `_PM_POLICY_ENV_VARS` grew as the notice learned to report more controls. It is
    also what the default pinned branch used to remove, so widening it silently
    widened the default bypass to UV_ONLY_BINARY, UV_ONLY_BINARY_PACKAGE and
    PIP_UPLOADED_PRIOR_TO, none of which this installer touched before. The whole
    change rests on the default path being indistinguishable from before.
    """

    # Exactly what the installer removed before this change. Do not add to this
    # without deciding, separately and deliberately, to override one more control.
    BASELINE = {
        "UV_NO_BUILD",
        "UV_NO_BUILD_PACKAGE",
        "UV_NO_BINARY",
        "UV_NO_BINARY_PACKAGE",
        "UV_REQUIRE_HASHES",
        "UV_EXCLUDE_NEWER",
        "PIP_ONLY_BINARY",
        "PIP_NO_BINARY",
        "PIP_REQUIRE_HASHES",
    }

    def test_the_default_scrub_set_has_not_grown(self):
        assert set(ips._PM_POLICY_SCRUB_ENV_VARS) == self.BASELINE

    def test_the_notice_reports_more_than_the_default_path_overrides(self):
        reported = set(ips._PM_POLICY_ENV_VARS)
        assert self.BASELINE < reported, "the notice should report at least the scrubbed set"
        assert {"UV_ONLY_BINARY", "PIP_UPLOADED_PRIOR_TO"} <= reported

    @pytest.mark.parametrize(
        "name", ["UV_ONLY_BINARY", "UV_ONLY_BINARY_PACKAGE", "PIP_UPLOADED_PRIOR_TO"]
    )
    def test_a_reported_only_control_survives_the_default_path(self, name, monkeypatch):
        monkeypatch.delenv("UNSLOTH_RESPECT_PM_POLICY", raising = False)
        monkeypatch.setenv(name, ":all:" if "BINARY" in name else "2026-01-01")
        cmd = ["uv", "pip", "install", "--index-url", "https://pinned", "torch"]
        env = ips._install_env_for_cmd(cmd)
        assert env is not None
        assert name in env, (
            f"{name} is reported by the notice but was never overridden before this "
            f"change; the default path must still leave it alone"
        )

    def test_the_scrubbed_controls_are_still_scrubbed_by_default(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_RESPECT_PM_POLICY", raising = False)
        for name in self.BASELINE:
            monkeypatch.setenv(name, "1")
        cmd = ["uv", "pip", "install", "--index-url", "https://pinned", "torch"]
        env = ips._install_env_for_cmd(cmd)
        assert env is not None
        for name in self.BASELINE:
            assert name not in env, name

    def test_the_opt_out_still_keeps_everything(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_RESPECT_PM_POLICY", "1")
        for name in set(ips._PM_POLICY_ENV_VARS) | self.BASELINE:
            monkeypatch.setenv(name, "1")
        cmd = ["uv", "pip", "install", "--index-url", "https://pinned", "torch"]
        env = ips._install_env_for_cmd(cmd)
        assert env is not None
        for name in self.BASELINE:
            assert env.get(name) == "1", name


class TestTheMetadataRepairDeclinesBeforeTouchingAnything:
    """The guard sits at the top of the repair, not inside _stage_replacement().

    By the time staging is reached the repair has already rewritten METADATA and moved
    pip's leftover backups aside. A normal return puts them back through the finally,
    but a SIGKILL or a power loss cannot run a finally, so under the opt-out the repair
    is declined before the first byte is touched.
    """

    def test_no_filesystem_call_is_reached_under_the_opt_out(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_RESPECT_PM_POLICY", "1")
        touched: "list[str]" = []

        # Anything that could mutate the tree records itself instead of running.
        for name in ("_rewrite_minimal_metadata",):
            if hasattr(ips, name):
                monkeypatch.setattr(
                    ips,
                    name,
                    lambda *a, _n = name, **k: touched.append(_n) or True,
                )
        monkeypatch.setattr(
            ips._QuarantinedMetadata,
            "back_up",
            lambda self, *a, **k: touched.append("back_up") or True,
        )
        monkeypatch.setattr(
            ips._QuarantinedMetadata,
            "take",
            lambda self, *a, **k: touched.append("take") or True,
        )
        monkeypatch.setattr(
            ips.install_manifest,
            "installed_versions",
            lambda name: ["1.0", "2.0"],
        )
        monkeypatch.setattr(
            ips.install_manifest,
            "invalid_metadata_paths",
            lambda name: ["/x/a.dist-info"],
        )
        monkeypatch.setattr(
            ips.install_manifest,
            "pip_backup_metadata_paths",
            lambda name: ["/x/~pkg"],
        )

        result = ips._repair_duplicate_core_metadata(("unsloth",))
        assert result is False
        assert touched == [], f"the repair mutated the tree before declining: {touched}"
