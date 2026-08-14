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


@pytest.fixture(autouse = True)
def _default_pm_policy_mode(monkeypatch):
    """Most of this file asserts the DEFAULT contract, so strict mode has to be opted in.

    Inherited from the shell it would flip that contract underneath every case here, and
    a suite whose result depends on the operator's own environment tests nothing.
    """
    monkeypatch.delenv(ips._STRICT_PM_POLICY_ENV_VAR, raising = False)


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

    def test_pinned_cmd_strips_the_policy_it_cannot_satisfy(self):
        """The pinned branch neutralises the config FILES, but an env var outranks a
        config file, so a hardened shell could still fail a torch repair the pin was
        supposed to make deterministic. Only the unsatisfiable half goes: the pinned
        specs carry no hashes, and an exclude-newer cutoff can remove the pinned release
        from the resolution outright."""
        with mock.patch.dict(os.environ, self.HOSTILE):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env is not None
        for name in ("PIP_REQUIRE_HASHES", "UV_REQUIRE_HASHES", "UV_EXCLUDE_NEWER"):
            assert name not in env, f"{name} must be cleared for a pinned install"
        # The pre-existing pinned contract is unchanged.
        assert env["UV_NO_CONFIG"] == "1" and env["PIP_CONFIG_FILE"] == os.devnull

    def test_pinned_cmd_keeps_the_binary_only_policy(self):
        """Every pinned index we use serves wheels, so a no-build/only-binary policy IS
        satisfiable there and must survive -- it is the control that stops an sdist
        running setup.py during a torch repair."""
        with mock.patch.dict(os.environ, self.HOSTILE):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env["PIP_ONLY_BINARY"] == ":all:"
        assert env["UV_NO_BUILD"] == "1"

    def test_pinned_cmd_still_drops_the_force_source_build_vars(self):
        """The mirror image: NO_BINARY forces a source BUILD of whatever the pin
        fetches, i.e. compiling torch from source. More untrusted execution, not less,
        so it goes under strict policy too."""
        forced = {
            "PIP_NO_BINARY": ":all:",
            "UV_NO_BINARY": ":all:",
            "UV_NO_BINARY_PACKAGE": "torch",
        }
        for strict in ("0", "1"):
            with mock.patch.dict(os.environ, dict(forced, UNSLOTH_STRICT_PM_POLICY = strict)):
                env = ips._install_env_for_cmd(
                    ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
                )
            for name in forced:
                assert name not in env, f"{name} must never reach a pinned install"

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


class TestStrictPmPolicyOptOut:
    """UNSLOTH_STRICT_PM_POLICY=1 is the operator's answer to every relaxation above.

    require-hashes / no-build is a security control, so an operator who set it
    deliberately must be able to have it honoured -- and then the install is allowed to
    fail on a wheel-less or unhashed requirement, which is #8530 again, by choice.
    """

    @pytest.fixture(autouse = True)
    def _no_ambient_uv_config(self, tmp_path, monkeypatch):
        """The uv config scan is memoized and walks the cwd's parents, so these cases run
        off the checkout and from an unscanned state."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ips, "_UV_POLICY_CONFIG", None)
        monkeypatch.setattr(ips, "_UV_POLICY_PROJECTION", None)
        monkeypatch.setattr(ips, "_PIP_CONFIG_POLICY", {})

    HOSTILE = dict(
        TestHardenedPipConfigRelaxation.HOSTILE,
        UV_REQUIRE_HASHES = "1",
        UNSLOTH_STRICT_PM_POLICY = "1",
    )

    def test_non_pinned_pip_inherits_the_policy_verbatim(self):
        # Without the cutoff, which pip cannot honour at all: a pip command under THAT is
        # refused outright, and TestTheStrictFallbackRefusesWhatPipCannotHonour pins it.
        env = dict(self.HOSTILE)
        env.pop("UV_EXCLUDE_NEWER", None)
        with mock.patch.dict(os.environ, env):
            assert (
                ips._install_env_for_cmd(["python", "-m", "pip", "install", "-r", "extras.txt"])
                is None
            ), "strict policy must not switch hash mode off"

    def test_pinned_cmd_keeps_the_policy_it_can_carry(self):
        """Policy travels by environment on a pinned command, and the index scrub the pin
        needs is untouched by the switch."""
        with mock.patch.dict(os.environ, dict(self.HOSTILE, UV_INDEX = "https://mirror.corp/simple")):
            env = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env is not None
        assert "UV_INDEX" not in env, "the pin still overrides an inherited index"
        for name, value in self.HOSTILE.items():
            assert env[name] == value, f"strict policy must keep {name}"

    def test_a_pinned_cmd_never_reads_a_config_file_in_either_mode(self):
        """Config discovery carries the operator's INDEX as well as their policy, and a
        uv.toml index outranks the CLI pin, so honouring the file here would resolve a
        CUDA/ROCm/XPU repair from their mirror and install the wrong torch. A refused
        policy is recoverable; the wrong wheel is what the pin exists to prevent."""
        for strict in ("0", "1"):
            with mock.patch.dict(
                os.environ,
                dict(
                    self.HOSTILE,
                    UNSLOTH_STRICT_PM_POLICY = strict,
                    UV_CONFIG_FILE = "/etc/uv/uv.toml",
                ),
            ):
                env = ips._install_env_for_cmd(
                    ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
                )
            assert env["UV_NO_CONFIG"] == "1", f"strict={strict}"
            assert env["PIP_CONFIG_FILE"] == os.devnull, f"strict={strict}"
            # Strict mode may point it at the generated policy file, which carries no
            # index. Theirs is what must not survive.
            assert env.get("UV_CONFIG_FILE") != "/etc/uv/uv.toml", f"strict={strict}"

    def test_the_pip_fallback_inherits_the_uv_policy_translated(self):
        """pip_install() falls back to pip whenever uv fails, INCLUDING when uv failed
        because of the policy, and pip reads no UV_* variable. Without the translation the
        fallback performs exactly the install uv had just refused."""
        with mock.patch.dict(
            os.environ,
            {"UNSLOTH_STRICT_PM_POLICY": "1", "UV_NO_BUILD": "1", "UV_REQUIRE_HASHES": "1"},
            clear = True,
        ):
            env = ips._install_env_for_cmd(["python", "-m", "pip", "install", "-r", "extras.txt"])
        assert env is not None
        assert env["PIP_ONLY_BINARY"] == ":all:"
        assert env["PIP_REQUIRE_HASHES"] == "1"

    def test_the_translation_never_overwrites_an_explicit_pip_setting(self):
        with mock.patch.dict(
            os.environ,
            {"UNSLOTH_STRICT_PM_POLICY": "1", "UV_NO_BUILD": "1", "PIP_ONLY_BINARY": "numpy"},
            clear = True,
        ):
            env = ips._install_env_for_cmd(["python", "-m", "pip", "install", "x"])
        # Nothing to add, so nothing is added: the inherited environment already carries
        # their setting, and an env of None IS that environment.
        assert env is None

    def test_a_uv_config_policy_reaches_the_pip_fallback(self, tmp_path, monkeypatch):
        """#8530's no-build lived in uv.toml. uv refuses the wheel-less extras, pip_install
        falls back to pip, and pip reads neither that file nor any UV_* variable, so a
        translation that looked only at the environment let the fallback build them."""
        (tmp_path / "uv.toml").write_text("no-build = true\n", encoding = "utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ips, "_UV_POLICY_CONFIG", None)
        with mock.patch.dict(os.environ, {"UNSLOTH_STRICT_PM_POLICY": "1"}, clear = True):
            env = ips._install_env_for_cmd(["python", "-m", "pip", "install", "-r", "extras.txt"])
        assert env is not None and env["PIP_ONLY_BINARY"] == ":all:"

    def test_a_uv_config_no_binary_is_never_translated(self, tmp_path, monkeypatch):
        """no-binary FORCES a source build, so carrying it over would be the opposite of
        preserving the policy."""
        (tmp_path / "uv.toml").write_text("no-binary = true\n", encoding = "utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ips, "_UV_POLICY_CONFIG", None)
        with mock.patch.dict(os.environ, {"UNSLOTH_STRICT_PM_POLICY": "1"}, clear = True):
            assert ips._install_env_for_cmd(["python", "-m", "pip", "install", "x"]) is None

    def test_no_uv_policy_means_no_translation(self):
        with mock.patch.dict(os.environ, {"UNSLOTH_STRICT_PM_POLICY": "1"}, clear = True):
            assert ips._install_env_for_cmd(["python", "-m", "pip", "install", "x"]) is None

    def _pinned_env(self, tmp_path, monkeypatch, config: str, env: dict):
        """The child env for a pinned repair, with `config` as the cwd's uv.toml."""
        (tmp_path / "uv.toml").write_text(config, encoding = "utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ips, "_UV_POLICY_CONFIG", None)
        monkeypatch.setattr(ips, "_UV_POLICY_PROJECTION", None)
        with mock.patch.dict(os.environ, env, clear = True):
            return ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )

    def test_a_pinned_uv_command_is_served_the_file_policy_without_the_index(
        self, tmp_path, monkeypatch
    ):
        """UV_NO_CONFIG=1 is what keeps a uv.toml INDEX from outranking the CLI pin, and it
        takes that file's POLICY down with it. Translating only to PIP_* leaves the uv
        command that actually runs under no build restriction at all, so an sdist-only
        mirror builds from source in the mode that promised the policy verbatim.

        --config-file replaces discovery, so the generated file carries the policy across
        without carrying the mirror that made the pin necessary."""
        env = self._pinned_env(
            tmp_path,
            monkeypatch,
            "no-build = true\nrequire-hashes = true\nexclude-newer = '2026-01-01'\n",
            {"UNSLOTH_STRICT_PM_POLICY": "1"},
        )
        projected = Path(env["UV_CONFIG_FILE"]).read_text(encoding = "utf-8")
        assert "no-build = true" in projected
        assert "require-hashes = true" in projected
        assert 'exclude-newer = "2026-01-01"' in projected
        settings = [line for line in projected.splitlines() if not line.startswith("#")]
        assert not any(
            "index" in line or "find-links" in line for line in settings
        ), f"the operator's mirror must not travel with the policy: {settings}"
        assert env["UV_NO_CONFIG"] == "1", "discovery stays off whichever variable wins"

    def test_a_per_package_cutoff_reaches_the_pinned_command(self, tmp_path, monkeypatch):
        """uv's exclude-newer-package is the same control per package, and a pinned
        command reads no config file, so it travels in the projection like the rest."""
        env = self._pinned_env(
            tmp_path,
            monkeypatch,
            '[pip]\nexclude-newer-package = { docopt = "2020-01-01T00:00:00Z" }\n',
            {"UNSLOTH_STRICT_PM_POLICY": "1"},
        )
        projected = Path(env["UV_CONFIG_FILE"]).read_text(encoding = "utf-8")
        assert 'exclude-newer-package = { "docopt" = "2020-01-01T00:00:00Z" }' in projected

    def test_a_per_package_cutoff_also_stops_the_pip_fallback(self, tmp_path, monkeypatch):
        """pip has no equivalent for this one either, so falling back would install what
        the cutoff excluded."""
        (tmp_path / "uv.toml").write_text(
            '[pip]\nexclude-newer-package = { docopt = "2020-01-01T00:00:00Z" }\n',
            encoding = "utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ips, "_UV_POLICY_CONFIG", None)
        with mock.patch.dict(os.environ, {"UNSLOTH_STRICT_PM_POLICY": "1"}, clear = True):
            assert ips._untranslatable_strict_policy() == ["exclude-newer-package"]

    def test_an_unwritable_projection_fails_the_install(self, tmp_path, monkeypatch):
        """Carrying on under --no-config would run the pinned command with no policy at
        all, which is the bypass the switch was set to refuse."""
        (tmp_path / "uv.toml").write_text("no-build = true\n", encoding = "utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ips, "_UV_POLICY_CONFIG", None)
        monkeypatch.setattr(ips, "_UV_POLICY_PROJECTION", None)

        def no_temp_dir(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(ips.tempfile, "mkdtemp", no_temp_dir)
        with (
            mock.patch.dict(os.environ, {"UNSLOTH_STRICT_PM_POLICY": "1"}, clear = True),
            pytest.raises(SystemExit),
        ):
            ips._install_env_for_cmd(
                ["uv", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )

    def test_the_projection_is_written_once(self, tmp_path, monkeypatch):
        """Every torch repair builds an env, and a file per command would litter."""
        first = self._pinned_env(
            tmp_path, monkeypatch, "no-build = true\n", {"UNSLOTH_STRICT_PM_POLICY": "1"}
        )["UV_CONFIG_FILE"]
        with mock.patch.dict(os.environ, {"UNSLOTH_STRICT_PM_POLICY": "1"}, clear = True):
            second = ips._install_env_for_cmd(
                ["uv", "pip", "install", "torchvision", "--index-url", "https://x/cu128"]
            )["UV_CONFIG_FILE"]
        assert first == second

    def test_the_projection_does_not_outlive_the_process(self, tmp_path, monkeypatch):
        """It is a temp file handed to uv by path, so it goes where the other ones go:
        the exit hook uv_path_safety already registers."""
        from backend.utils import uv_path_safety as uvps

        path = Path(
            self._pinned_env(
                tmp_path, monkeypatch, "no-build = true\n", {"UNSLOTH_STRICT_PM_POLICY": "1"}
            )["UV_CONFIG_FILE"]
        )
        assert str(path.parent) in uvps._UV_SAFE_PATH_TMPDIRS
        uvps._cleanup_uv_safe_path_tmpdirs()
        assert not path.exists()

    def test_the_default_mode_carries_no_file_policy_into_a_pinned_command(
        self, tmp_path, monkeypatch
    ):
        """The translation is the switch's job. Without it a torch repair on a hardened
        host fails exactly as #8530 did."""
        env = self._pinned_env(tmp_path, monkeypatch, "no-build = true\n", {})
        assert "UV_CONFIG_FILE" not in env and "PIP_ONLY_BINARY" not in env
        assert env["UV_NO_CONFIG"] == "1"

    def test_an_unconfigured_host_gets_no_projection(self, tmp_path, monkeypatch):
        """Nothing to carry, nothing written: the pinned command runs as it always did."""
        env = self._pinned_env(tmp_path, monkeypatch, "", {"UNSLOTH_STRICT_PM_POLICY": "1"})
        assert "UV_CONFIG_FILE" not in env

    def test_a_package_scoped_policy_translates_at_its_own_scope(self, tmp_path, monkeypatch):
        """`no-build-package = ["some-other-package"]` is a policy about that package.
        Widening it to :all: fails Studio's four wheel-less requirements over a setting
        that never covered them."""
        env = self._pinned_env(
            tmp_path,
            monkeypatch,
            'no-build-package = ["some-other-package"]\n',
            {"UNSLOTH_STRICT_PM_POLICY": "1"},
        )
        # pip's format control is comma-separated; uv's pip table takes only-binary, which
        # is the same restriction spelled the way that interface reads it.
        assert env["PIP_ONLY_BINARY"] == "some-other-package"
        projected = Path(env["UV_CONFIG_FILE"]).read_text(encoding = "utf-8")
        assert 'only-binary = ["some-other-package"]' in projected
        assert "no-build = true" not in projected

    def test_the_uv_command_carries_the_build_policy_as_a_flag(self):
        """uv's own UV_NO_BUILD does not reach the `uv pip` interface (0.10.7 builds the
        sdist anyway), so without the flag the primary uv attempt builds exactly what
        strict mode has already told the pip fallback to refuse."""
        with mock.patch.dict(
            os.environ, {"UNSLOTH_STRICT_PM_POLICY": "1", "UV_NO_BUILD": "1"}, clear = True
        ):
            assert "--no-build" in ips._build_uv_cmd(("numpy",))

    def test_the_uv_command_keeps_the_flag_package_scoped(self):
        with mock.patch.dict(
            os.environ,
            {"UNSLOTH_STRICT_PM_POLICY": "1", "UV_NO_BUILD_PACKAGE": "some-other-package"},
            clear = True,
        ):
            cmd = ips._build_uv_cmd(("numpy",))
        assert "--no-build" not in cmd
        # --only-binary is what uv's pip interface takes: --no-build-package exits 2
        # there ("unexpected argument", uv 0.12.1), which would fail every install.
        assert cmd[-2:] == ["--only-binary", "some-other-package"]

    def test_the_default_uv_command_is_unchanged(self):
        """Everything here is opt-in: without the switch the command is byte-identical."""
        with mock.patch.dict(os.environ, {"UV_NO_BUILD": "1"}, clear = True):
            cmd = ips._build_uv_cmd(("numpy",))
        assert not any(arg.startswith("--no-build") for arg in cmd)

    def test_an_empty_package_list_translates_to_nothing(self, tmp_path, monkeypatch):
        (tmp_path / "uv.toml").write_text("no-build-package = []\n", encoding = "utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ips, "_UV_POLICY_CONFIG", None)
        with mock.patch.dict(os.environ, {"UNSLOTH_STRICT_PM_POLICY": "1"}, clear = True):
            assert ips._install_env_for_cmd(["python", "-m", "pip", "install", "x"]) is None

    def test_an_all_scoped_policy_still_translates_globally(self, tmp_path, monkeypatch):
        (tmp_path / "uv.toml").write_text("[pip]\nonly-binary = [':all:']\n", encoding = "utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ips, "_UV_POLICY_CONFIG", None)
        with mock.patch.dict(os.environ, {"UNSLOTH_STRICT_PM_POLICY": "1"}, clear = True):
            env = ips._install_env_for_cmd(["python", "-m", "pip", "install", "x"])
        assert env["PIP_ONLY_BINARY"] == ":all:"

    def test_the_source_build_exemptions_are_dropped(self):
        """--no-binary for the four wheel-less names IS the no-build override, so it
        cannot survive the switch that turns overriding off."""
        with mock.patch.dict(os.environ, self.HOSTILE):
            assert ips._sdist_only_build_args(*ips.SDIST_ONLY_PACKAGES) == []
            assert ips._sdist_only_build_args("diffusers") == []

    def test_default_behaviour_is_unchanged(self):
        """Everything above is opt-in: an unset (or 0) variable installs as before."""
        for value in ("", "0"):
            with mock.patch.dict(os.environ, {"UNSLOTH_STRICT_PM_POLICY": value}):
                assert ips._sdist_only_build_args("openai-whisper") == [
                    "--no-binary",
                    "openai-whisper",
                ]
                env = ips._install_env_for_cmd(["python", "-m", "pip", "install", "x"])
                assert env is not None and env["PIP_REQUIRE_HASHES"] == "0"


class TestPipConfigPolicySurvivesThePin:
    """A pinned command nulls PIP_CONFIG_FILE, and under the switch the policy in that
    file has to survive it.

    The operator's index lives in pip.conf beside their policy, and the pin has to win, so
    the file goes. pip applies environment variables AFTER config files, which is the one
    channel left for the half that is a security control rather than a mirror.
    """

    @pytest.fixture(autouse = True)
    def _no_ambient_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ips, "_UV_POLICY_CONFIG", None)
        monkeypatch.setattr(ips, "_UV_POLICY_PROJECTION", None)
        monkeypatch.setattr(ips, "_PIP_CONFIG_POLICY", None)

    def _pinned_env(self, pip_config: str, env: dict):
        result = mock.Mock(returncode = 0, stdout = pip_config)
        with (
            mock.patch.dict(os.environ, env, clear = True),
            mock.patch.object(ips.subprocess, "run", return_value = result),
        ):
            return ips._install_env_for_cmd(
                ["python", "-m", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )

    def test_the_pinned_pip_command_keeps_a_pip_conf_policy(self):
        env = self._pinned_env(
            "global.require-hashes='true'\nglobal.only-binary=':all:'\n",
            {"UNSLOTH_STRICT_PM_POLICY": "1"},
        )
        assert env["PIP_CONFIG_FILE"] == os.devnull, "the pin still overrides their index"
        assert env["PIP_REQUIRE_HASHES"] == "1"
        assert env["PIP_ONLY_BINARY"] == ":all:"

    def test_a_scoped_pip_conf_policy_keeps_its_scope(self):
        env = self._pinned_env(
            "global.only-binary='some-other-package'\n", {"UNSLOTH_STRICT_PM_POLICY": "1"}
        )
        assert env["PIP_ONLY_BINARY"] == "some-other-package"

    def test_a_command_scoped_section_stays_on_its_command(self):
        """`[download] require-hashes = true` binds `pip download` and nothing else, so
        carrying it into a torch install fails a repair the operator never restricted."""
        env = self._pinned_env(
            "download.require-hashes='true'\n", {"UNSLOTH_STRICT_PM_POLICY": "1"}
        )
        assert "PIP_REQUIRE_HASHES" not in env

    def test_a_matching_section_is_carried(self):
        result = mock.Mock(returncode = 0, stdout = "download.require-hashes='true'\n")
        with (
            mock.patch.dict(os.environ, {"UNSLOTH_STRICT_PM_POLICY": "1"}, clear = True),
            mock.patch.object(ips.subprocess, "run", return_value = result),
        ):
            env = ips._install_env_for_cmd(
                ["python", "-m", "pip", "download", "torch", "--index-url", "https://x/cu128"]
            )
        assert env["PIP_REQUIRE_HASHES"] == "1"

    def test_the_command_section_overrides_global(self):
        result = mock.Mock(
            returncode = 0,
            stdout = "global.only-binary=':all:'\ninstall.only-binary='numpy'\n",
        )
        with (
            mock.patch.dict(os.environ, {"UNSLOTH_STRICT_PM_POLICY": "1"}, clear = True),
            mock.patch.object(ips.subprocess, "run", return_value = result),
        ):
            env = ips._install_env_for_cmd(
                ["python", "-m", "pip", "install", "torch", "--index-url", "https://x/cu128"]
            )
        assert env["PIP_ONLY_BINARY"] == "numpy"

    def test_the_default_mode_still_drops_it(self):
        """Without the switch this is the #8530 relaxation, and it stays: the pinned
        specs carry no hashes, so honouring the file fails every torch repair."""
        env = self._pinned_env("global.require-hashes='true'\n", {})
        assert env["PIP_CONFIG_FILE"] == os.devnull
        assert "PIP_REQUIRE_HASHES" not in env

    def test_a_switched_off_pip_conf_policy_is_not_carried(self):
        env = self._pinned_env(
            "global.require-hashes='false'\n", {"UNSLOTH_STRICT_PM_POLICY": "1"}
        )
        assert "PIP_REQUIRE_HASHES" not in env

    def test_their_own_variable_is_never_overwritten(self):
        env = self._pinned_env(
            "global.only-binary=':all:'\n",
            {"UNSLOTH_STRICT_PM_POLICY": "1", "PIP_ONLY_BINARY": "numpy"},
        )
        assert env["PIP_ONLY_BINARY"] == "numpy"


class TestTheStrictFallbackRefusesWhatPipCannotHonour:
    """pip_install() falls back to pip whenever uv fails, and pip has no --exclude-newer.

    So the one case the fallback must not take is uv refusing a release published past the
    operator's cutoff: pip would install precisely that release. Strict mode promises the
    policy verbatim, which makes a failed install the honest answer.
    """

    @pytest.fixture(autouse = True)
    def _no_ambient_uv_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ips, "_UV_POLICY_CONFIG", None)

    def _uv_fails(self, monkeypatch, env: dict):
        """Drive pip_install with uv present and failing; return whether pip ever ran."""
        ran_pip: list[list[str]] = []
        monkeypatch.setattr(ips, "USE_UV", True)
        monkeypatch.setattr(ips, "VERBOSE", False)
        monkeypatch.setattr(ips, "run", lambda label, cmd, **kw: ran_pip.append(cmd))
        monkeypatch.setattr(
            ips.subprocess, "run", lambda *a, **kw: mock.Mock(returncode = 1, stdout = b"")
        )
        with mock.patch.dict(os.environ, env, clear = True):
            ips.pip_install("deps", "numpy", constrain = False)
        return ran_pip

    def test_exclude_newer_stops_the_fallback(self, monkeypatch):
        with pytest.raises(SystemExit):
            self._uv_fails(
                monkeypatch,
                {"UNSLOTH_STRICT_PM_POLICY": "1", "UV_EXCLUDE_NEWER": "2024-01-01T00:00:00Z"},
            )

    def test_a_translatable_policy_still_falls_back(self, monkeypatch):
        """no-build and require-hashes DO have pip equivalents, so the fallback keeps
        working under them: refusing it there would fail installs for no gain."""
        assert self._uv_fails(
            monkeypatch, {"UNSLOTH_STRICT_PM_POLICY": "1", "UV_NO_BUILD": "1"}
        ), "a policy pip can be told must not cost the fallback"

    def test_the_default_mode_falls_back_as_before(self, monkeypatch):
        assert self._uv_fails(monkeypatch, {"UV_EXCLUDE_NEWER": "2024-01-01T00:00:00Z"})

    def _pip_is_the_only_backend(self, monkeypatch, env: dict):
        """Drive pip_install with no uv at all; return whether pip ever ran."""
        ran_pip: list[list[str]] = []
        monkeypatch.setattr(ips, "USE_UV", False)
        monkeypatch.setattr(ips, "VERBOSE", False)
        monkeypatch.setattr(ips, "run", lambda label, cmd, **kw: ran_pip.append(cmd))
        with mock.patch.dict(os.environ, env, clear = True):
            ips.pip_install("deps", "numpy", constrain = False)
        return ran_pip

    def test_a_direct_pip_install_is_refused_too(self, monkeypatch):
        """No uv failure announces this one: a host without uv, or a caller forcing pip,
        reaches pip with the same policy pip cannot honour."""
        with pytest.raises(SystemExit):
            self._pip_is_the_only_backend(
                monkeypatch,
                {"UNSLOTH_STRICT_PM_POLICY": "1", "UV_EXCLUDE_NEWER": "2024-01-01T00:00:00Z"},
            )

    def test_a_direct_pip_install_runs_when_the_policy_translates(self, monkeypatch):
        assert self._pip_is_the_only_backend(
            monkeypatch, {"UNSLOTH_STRICT_PM_POLICY": "1", "UV_NO_BUILD": "1"}
        )

    @pytest.mark.parametrize(
        "cmd",
        [
            ["python", "-m", "pip", "install", "--upgrade", "pip"],
            ["python", "-m", "pip", "download", "pytorch-triton-xpu", "-d", "/tmp/x"],
        ],
    )
    def test_every_direct_pip_fetch_is_refused(self, cmd):
        """The bootstrap's own pip upgrade and the XPU download do not go through
        pip_install(), so the refusal sits where every pip command builds its
        environment. Otherwise the switch lets those two fetch past the cutoff."""
        with (
            mock.patch.dict(
                os.environ,
                {"UNSLOTH_STRICT_PM_POLICY": "1", "UV_EXCLUDE_NEWER": "2024-01-01T00:00:00Z"},
                clear = True,
            ),
            pytest.raises(SystemExit),
        ):
            ips._install_env_for_cmd(cmd)

    def test_a_uv_command_is_not_refused(self):
        """uv honours the cutoff itself, so it is only pip that cannot run under one."""
        with mock.patch.dict(
            os.environ,
            {"UNSLOTH_STRICT_PM_POLICY": "1", "UV_EXCLUDE_NEWER": "2024-01-01T00:00:00Z"},
            clear = True,
        ):
            ips._install_env_for_cmd(["uv", "pip", "install", "numpy"])

    def test_an_optional_install_declines_rather_than_exits(self, monkeypatch):
        """pip_install_try exists for installs with a follow-up, so it reports and
        returns False instead of taking the whole install down."""
        monkeypatch.setattr(ips, "USE_UV", False)
        monkeypatch.setattr(ips, "VERBOSE", False)
        monkeypatch.setattr(
            ips.subprocess, "run", lambda *a, **kw: mock.Mock(returncode = 0, stdout = b"")
        )
        with mock.patch.dict(
            os.environ,
            {"UNSLOTH_STRICT_PM_POLICY": "1", "UV_EXCLUDE_NEWER": "2024-01-01T00:00:00Z"},
            clear = True,
        ):
            assert ips.pip_install_try("optional", "numpy", constrain = False) is False


class TestPmPolicyRelaxationIsReported:
    """A bypassed security control that nobody is told about is the actual finding.

    The relaxations stay (nothing we ship is hash-locked), so the install still works on
    a hardened host -- but it says which policy it overrode and which switch enforces it.
    """

    def _sources(
        self,
        env: dict,
        pip_config: str = "",
    ) -> list[str]:
        result = mock.Mock(returncode = 0, stdout = pip_config)
        with (
            mock.patch.dict(os.environ, env, clear = True),
            mock.patch.object(ips.subprocess, "run", return_value = result),
        ):
            return ips._hardened_pm_policy_sources()

    @pytest.fixture(autouse = True)
    def _rescan_uv_config(self, monkeypatch):
        """The scans are memoized for the run, so each case starts from unscanned."""
        monkeypatch.setattr(ips, "_UV_POLICY_CONFIG", None)
        monkeypatch.setattr(ips, "_PIP_CONFIG_POLICY", None)

    @pytest.fixture(autouse = True)
    def _off_the_repo_cwd(self, tmp_path, monkeypatch):
        """The repo root has a pyproject.toml, and uv config discovery walks the cwd, so
        a test about that discovery must not read whatever checkout it runs from."""
        monkeypatch.chdir(tmp_path)

    def test_quiet_on_an_unconfigured_host(self):
        assert self._sources({}) == []

    def test_zero_and_empty_are_not_a_policy(self):
        assert self._sources({"PIP_REQUIRE_HASHES": "0", "UV_NO_BUILD": ""}) == []

    def test_names_the_policy_environment_variables(self):
        found = self._sources({"PIP_REQUIRE_HASHES": "1", "UV_NO_BUILD": "1"})
        assert set(found) == {"PIP_REQUIRE_HASHES", "UV_NO_BUILD"}

    def test_reads_the_pip_config_files_too(self):
        """#8530 was pip.conf and uv.toml, not env vars, so env alone would miss it.
        `pip config list` renders every file pip would load, PIP_CONFIG_FILE included."""
        found = self._sources(
            {},
            pip_config = (
                "global.require-hashes='true'\n"
                "global.only-binary=':all:'\n"
                "global.index-url='https://mirror.corp/simple'\n"
                # pip renders the PIP_* env vars as a section of its own; those are
                # reported from the environment, so listing them again reads like a
                # second, separate setting.
                ":env:.require-hashes='true'\n"
            ),
        )
        assert found == ["pip config global.require-hashes", "pip config global.only-binary"]

    def test_names_the_package_scoped_build_policy_too(self):
        """_sdist_only_build_args() overrides a package-scoped no-build on the command
        line, so an operator who set only that must still hear about it."""
        assert self._sources({"UV_NO_BUILD_PACKAGE": "openai-whisper"}) == ["UV_NO_BUILD_PACKAGE"]

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "none", ":none:", " "])
    def test_a_switched_off_policy_is_not_a_policy(self, value):
        """Reporting only-binary=:none: would send someone looking for a hardened
        setting they deliberately turned off."""
        assert self._sources({"PIP_ONLY_BINARY": value, "UV_NO_BUILD": value}) == []

    def test_a_pip_config_key_set_to_false_is_not_reported(self):
        found = self._sources(
            {}, pip_config = "global.require-hashes='false'\nglobal.only-binary=':none:'\n"
        )
        assert found == []

    def test_reads_uv_config_files(self, tmp_path, monkeypatch):
        """#8530's no-build lived in ~/.config/uv/uv.toml. pip knows nothing about that
        file, so an environment-and-pip-only report would miss the canonical case."""
        (tmp_path / "uv.toml").write_text("no-build = true\n", encoding = "utf-8")
        monkeypatch.chdir(tmp_path)
        assert self._sources({}) == ["uv config uv.toml: no-build"]

    def test_reads_the_uv_table_of_a_pyproject(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = 'x'\n\n[tool.uv.pip]\nrequire-hashes = true\n",
            encoding = "utf-8",
        )
        monkeypatch.chdir(tmp_path)
        assert self._sources({}) == ["uv config pyproject.toml: require-hashes"]

    def test_a_pyproject_without_a_uv_table_is_not_ours(self, tmp_path, monkeypatch):
        """A no-build under somebody else's tool is somebody else's setting."""
        (tmp_path / "pyproject.toml").write_text(
            "[tool.someoneelse]\nno-build = true\n", encoding = "utf-8"
        )
        monkeypatch.chdir(tmp_path)
        assert self._sources({}) == []

    def test_an_explicit_uv_config_file_wins(self, tmp_path, monkeypatch):
        cfg = tmp_path / "corp-uv.toml"
        cfg.write_text("[pip]\nno-build = true\n", encoding = "utf-8")
        monkeypatch.chdir(tmp_path)
        assert self._sources({"UV_CONFIG_FILE": str(cfg)}) == ["uv config corp-uv.toml: no-build"]

    def test_an_unreadable_uv_config_is_not_fatal(self, tmp_path, monkeypatch):
        (tmp_path / "uv.toml").write_text("this is not = = toml [[[\n", encoding = "utf-8")
        monkeypatch.chdir(tmp_path)
        assert self._sources({"PIP_REQUIRE_HASHES": "1"}) == ["PIP_REQUIRE_HASHES"]

    def test_the_pip_half_is_reported_once_a_uv_venv_has_pip(self, monkeypatch, capsys):
        """`uv venv` creates no pip, so the pre-bootstrap report cannot read pip.conf, and
        a policy living only there went unnamed while every pip install still relaxed it.
        The second pass names it once pip exists."""
        monkeypatch.setattr(ips, "_PIP_CONFIG_POLICY", {})
        monkeypatch.setattr(ips, "_PIP_CONFIG_REACHED_PIP", False)
        result = mock.Mock(returncode = 0, stdout = "global.require-hashes='true'\n")
        with (
            mock.patch.dict(os.environ, {}, clear = True),
            mock.patch.object(ips.subprocess, "run", return_value = result),
        ):
            ips._report_pm_policy_relaxation_once_pip_exists()
        assert "global.require-hashes" in capsys.readouterr().out

    def test_the_second_pass_is_silent_when_the_first_read_pip(self, monkeypatch, capsys):
        """Otherwise a normal install prints the same line twice."""
        monkeypatch.setattr(ips, "_PIP_CONFIG_POLICY", {})
        monkeypatch.setattr(ips, "_PIP_CONFIG_REACHED_PIP", True)
        with mock.patch.dict(os.environ, {}, clear = True):
            ips._report_pm_policy_relaxation_once_pip_exists()
        assert capsys.readouterr().out == ""

    def test_a_broken_pip_is_not_fatal(self):
        """Best effort: the report is one printed line, never a failed install."""
        with (
            mock.patch.dict(os.environ, {"PIP_REQUIRE_HASHES": "1"}, clear = True),
            mock.patch.object(ips.subprocess, "run", side_effect = OSError("no pip")),
        ):
            assert ips._hardened_pm_policy_sources() == ["PIP_REQUIRE_HASHES"]

    def test_a_force_source_key_is_not_offered_the_switch(self, tmp_path):
        """no-binary is the mirror image of the reported keys: it FORCES a source build,
        every pinned command drops it whatever the switch says, and an unpinned command
        inherits it untouched. Nothing overrides it, so "set the switch to enforce your
        policy" would promise an enforcement that cannot happen."""
        (tmp_path / "uv.toml").write_text("no-binary = true\n", encoding = "utf-8")
        assert self._sources({}, pip_config = "global.no-binary='openai-whisper'\n") == []


class TestUvConfigDiscoveryMatchesUv:
    """Where the scan looks, and what it reads once it gets there.

    This stopped being a question about a printed line when strict mode started
    translating the answer into policy for pip and for a pinned uv command: a file uv
    ignores, or a key belonging to another tool, now fails an install rather than adding
    a word to a note.
    """

    @pytest.fixture(autouse = True)
    def _rescan_uv_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ips, "_UV_POLICY_CONFIG", None)

    def _keys(self) -> list[str]:
        return [f"{name}: {key}" for name, key, _value in ips._scan_uv_policy_config()]

    def test_an_adjacent_uv_toml_beats_the_pyproject(self, tmp_path):
        """ "uv.toml files take precedence over pyproject.toml files, so if both are
        present in a directory, configuration will be read from uv.toml". Reading the
        losing file too turns a [tool.uv] table uv ignores into enforced policy."""
        (tmp_path / "uv.toml").write_text("require-hashes = true\n", encoding = "utf-8")
        (tmp_path / "pyproject.toml").write_text(
            "[tool.uv.pip]\nno-build = true\n", encoding = "utf-8"
        )
        assert self._keys() == ["uv.toml: require-hashes"]

    def test_a_pyproject_with_a_uv_table_ends_the_parent_walk(self, tmp_path, monkeypatch):
        """uv reads the nearest project configuration, so a [tool.uv] table takes an
        ancestor uv.toml out of force. Measured on uv 0.10.7 with `no-build = true` in
        the parent: adding an unrelated [tool.uv] key to the child makes the sdist
        install succeed that the parent policy had refused."""
        (tmp_path / "uv.toml").write_text("no-build = true\n", encoding = "utf-8")
        nested = tmp_path / "project"
        nested.mkdir()
        (nested / "pyproject.toml").write_text(
            "[project]\nname = 'x'\n\n[tool.uv]\nlink-mode = 'copy'\n", encoding = "utf-8"
        )
        monkeypatch.chdir(nested)
        assert self._keys() == []

    def test_a_pyproject_without_a_uv_table_does_not_end_the_walk(self, tmp_path, monkeypatch):
        """The other half of the same measurement: without [tool.uv] the child is not a
        uv configuration at all, the parent's policy stays in force (the same install is
        refused), and stopping here would drop a policy uv is enforcing."""
        (tmp_path / "uv.toml").write_text("no-build = true\n", encoding = "utf-8")
        nested = tmp_path / "project"
        nested.mkdir()
        (nested / "pyproject.toml").write_text(
            "[project]\nname = 'x'\n", encoding = "utf-8"
        )
        monkeypatch.chdir(nested)
        assert self._keys() == ["uv.toml: no-build"]

    @pytest.mark.skipif(ips.IS_WINDOWS, reason = "XDG is the Unix system-config path")
    def test_only_the_first_system_config_is_read(self, tmp_path, monkeypatch):
        """"only the first-discovered file will be used". Unioning them lets a
        lower-priority file uv ignores fail an install under the switch."""
        first, second = tmp_path / "a", tmp_path / "b"
        for directory, policy in ((first, "require-hashes = true"), (second, "no-build = true")):
            (directory / "uv").mkdir(parents = True)
            (directory / "uv" / "uv.toml").write_text(policy + "\n", encoding = "utf-8")
        monkeypatch.setenv("XDG_CONFIG_DIRS", f"{first}{os.pathsep}{second}")
        assert self._keys() == ["uv.toml: require-hashes"]

    def test_discovery_starts_at_uv_working_dir(self, tmp_path, monkeypatch):
        """uv changes to --directory / UV_WORKING_DIR before it runs and discovers from
        there. Measured on uv 0.12.1: a uv.toml under it refuses an sdist while the
        original cwd has no config at all."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "uv.toml").write_text("[pip]\nno-build = true\n", encoding = "utf-8")
        monkeypatch.setenv("UV_WORKING_DIR", str(elsewhere))
        assert self._keys() == ["uv.toml: no-build"]

    def test_a_commented_uv_table_does_not_end_the_walk(self, tmp_path, monkeypatch):
        """A `# [tool.uv]` in a comment is not a table, and uv keeps reading the parent
        (checked on 0.12.1), so stopping here would drop the parent's policy."""
        (tmp_path / "uv.toml").write_text("no-build = true\n", encoding = "utf-8")
        nested = tmp_path / "project"
        nested.mkdir()
        (nested / "pyproject.toml").write_text(
            "[project]\nname = 'x'\n# [tool.uv] is documented over at ...\n",
            encoding = "utf-8",
        )
        monkeypatch.chdir(nested)
        assert self._keys() == ["uv.toml: no-build"]

    def test_the_pip_table_beats_the_root_of_the_same_file(self, tmp_path):
        """`uv pip install` reads the [pip] table, so a root `no-build = false` under a
        `[pip] no-build = true` leaves source builds disabled (measured on 0.12.1)."""
        (tmp_path / "uv.toml").write_text(
            "no-build = false\n\n[pip]\nno-build = true\n", encoding = "utf-8"
        )
        assert self._keys() == ["uv.toml: no-build"]

    @pytest.mark.skipif(ips.IS_WINDOWS, reason = "XDG is the Unix system-config path")
    def test_the_xdg_default_system_directory_is_searched(self, monkeypatch):
        """/etc/xdg is XDG's own default for an unset XDG_CONFIG_DIRS, and uv reads
        /etc/xdg/uv/uv.toml there, so a fleet policy at the standard path counts."""
        monkeypatch.delenv("XDG_CONFIG_DIRS", raising = False)
        # Only the first system file that EXISTS is read, so the default has to be
        # present for the candidate to survive the filter.
        monkeypatch.setattr(ips, "_file_exists", lambda p: str(p) == "/etc/xdg/uv/uv.toml")
        assert Path("/etc/xdg/uv/uv.toml") in ips._uv_config_candidates()

    def test_a_none_reset_clears_an_earlier_all(self, tmp_path):
        """`:none:` clears what came before it: uv 0.12.1 installs an sdist-only package
        under `only-binary = [":all:", ":none:"]`, so reading any `:all:` as final turns
        the operator's own reset into a global build ban."""
        (tmp_path / "uv.toml").write_text(
            '[pip]\nonly-binary = [":all:", ":none:"]\n', encoding = "utf-8"
        )
        assert self._keys() == []

    def test_a_multiline_list_is_read_by_the_line_scanner(self, tmp_path):
        """The 3.9/3.10 fallback saw `only-binary = [` and read no packages, so a policy
        written the way TOML formatters write it became no policy under the switch."""
        path = tmp_path / "uv.toml"
        path.write_text(
            '[pip]\nonly-binary = [\n  "foo",\n  "bar",\n]\nrequire-hashes = true\n',
            encoding = "utf-8",
        )
        assert [key for _n, key, _v in ips._scan_uv_policy_config_by_line([path])] == [
            "only-binary",
            "require-hashes",
        ]

    def test_uv_no_config_means_there_is_no_config(self, tmp_path, monkeypatch):
        """"--no-config: Avoid discovering configuration files". A policy uv is not
        reading is not one to enforce, translate or report, and treating it as live fails
        installs uv itself would have run."""
        (tmp_path / "uv.toml").write_text("no-build = true\n", encoding = "utf-8")
        monkeypatch.setenv("UV_NO_CONFIG", "1")
        assert self._keys() == []

    def test_an_explicit_config_file_survives_uv_no_config(self, tmp_path, monkeypatch):
        """Measured on uv 0.10.7: with both set, the --config-file is still honoured (the
        sdist install is refused by its no-build), so the pointer wins here too."""
        cfg = tmp_path / "corp-uv.toml"
        cfg.write_text("[pip]\nno-build = true\n", encoding = "utf-8")
        monkeypatch.setenv("UV_NO_CONFIG", "1")
        monkeypatch.setenv("UV_CONFIG_FILE", str(cfg))
        assert self._keys() == ["corp-uv.toml: no-build"]

    def test_a_higher_precedence_false_wins_over_a_lower_true(self, tmp_path, monkeypatch):
        """uv reads the higher-priority value and ignores the rest, so a project
        `no-build = false` above a user `no-build = true` is no policy at all. Recording
        only truthy values and reading on turns that into a global binary-only install."""
        user = tmp_path / "home" / ".config" / "uv"
        user.mkdir(parents = True)
        (user / "uv.toml").write_text("no-build = true\n", encoding = "utf-8")
        project = tmp_path / "project"
        project.mkdir()
        (project / "uv.toml").write_text(
            "no-build = false\nrequire-hashes = true\n", encoding = "utf-8"
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
        monkeypatch.chdir(project)
        assert self._keys() == ["uv.toml: require-hashes"]

    @pytest.mark.parametrize(
        "header, ends_the_walk",
        [
            ("[tool.uv]", True),
            ("[tool.uv.pip]", True),
            ("[tool.uvicorn]", False),
            ("# [tool.uv]", False),
        ],
    )
    def test_the_fallback_matches_the_uv_table_exactly(self, header, ends_the_walk):
        """[tool.uvicorn] is somebody else's table. Ending the walk there drops the parent
        policy uv does apply, which under the switch is the source build it forbade."""
        assert ips._pyproject_uv_table_by_line(f"{header}\nx = 1\n") is ends_the_walk

    def test_the_line_scanner_honours_the_same_precedence(self, tmp_path):
        project, user = tmp_path / "a.toml", tmp_path / "b.toml"
        project.write_text("no-build = false\n", encoding = "utf-8")
        user.write_text("no-build = true\n", encoding = "utf-8")
        assert ips._scan_uv_policy_config_by_line([project, user]) == []

    def test_a_uv_toml_ends_the_parent_walk(self, tmp_path, monkeypatch):
        """uv reads the nearest configuration file, not the union of the tree."""
        (tmp_path / "uv.toml").write_text("require-hashes = true\n", encoding = "utf-8")
        nested = tmp_path / "project"
        nested.mkdir()
        (nested / "uv.toml").write_text("no-build = true\n", encoding = "utf-8")
        monkeypatch.chdir(nested)
        assert self._keys() == ["uv.toml: no-build"]

    @pytest.mark.skipif(ips.IS_WINDOWS, reason = "XDG is the Unix system-config path")
    def test_the_system_config_can_live_under_xdg_config_dirs(self, tmp_path, monkeypatch):
        """ "If multiple system-level configuration files are found, e.g. at both
        /etc/uv/uv.toml and $XDG_CONFIG_DIRS/uv/uv.toml, only the first-discovered file
        will be used, with XDG taking priority." A managed fleet is exactly where the
        policy lives outside /etc, and missing it means a silent note and an untranslated
        policy."""
        corp = tmp_path / "corp"
        (corp / "uv").mkdir(parents = True)
        (corp / "uv" / "uv.toml").write_text("no-build = true\n", encoding = "utf-8")
        monkeypatch.setenv("XDG_CONFIG_DIRS", str(corp))
        assert self._keys() == ["uv.toml: no-build"]

    def test_an_empty_package_list_is_not_a_policy(self, tmp_path):
        """`no-build-package = []` covers no package at all. Read as a policy in force it
        becomes a global build ban, which fails the wheel-less requirements over a setting
        that restricted nothing."""
        (tmp_path / "uv.toml").write_text("no-build-package = []\n", encoding = "utf-8")
        assert self._keys() == []

    def test_the_package_list_survives_the_scan(self, tmp_path):
        (tmp_path / "uv.toml").write_text(
            'no-build-package = ["some-other-package"]\n', encoding = "utf-8"
        )
        assert ips._scan_uv_policy_config() == [
            ("uv.toml", "no-build-package", ["some-other-package"])
        ]

    def test_the_line_scanner_stays_inside_the_uv_table(self, tmp_path):
        """The 3.9/3.10 fallback has no tomllib, so it tracks sections by hand. Without
        that, any [tool.*] table's no-build in a pyproject that happens to mention
        [tool.uv] anywhere reads as uv's own."""
        path = tmp_path / "pyproject.toml"
        path.write_text(
            "[tool.uv]\nlink-mode = 'copy'\n\n[tool.someoneelse]\nno-build = true\n",
            encoding = "utf-8",
        )
        assert ips._scan_uv_policy_config_by_line([path]) == []

    def test_the_line_scanner_reads_the_uv_table(self, tmp_path):
        path = tmp_path / "pyproject.toml"
        path.write_text(
            "[tool.uv.pip]\nno-build = true\n\n[tool.someoneelse]\nrequire-hashes = true\n",
            encoding = "utf-8",
        )
        assert ips._scan_uv_policy_config_by_line([path]) == [
            ("pyproject.toml", "no-build", "true")
        ]

    def test_the_line_scanner_drops_a_trailing_toml_comment(self, tmp_path):
        """`no-build = false # builds allowed` is a policy switched OFF. With the comment
        left attached the value is not one of the disabled spellings, so it reads as a
        policy in force and the switch turns it into a global binary-only install."""
        path = tmp_path / "uv.toml"
        path.write_text("no-build = false # builds allowed\n", encoding = "utf-8")
        assert ips._scan_uv_policy_config_by_line([path]) == []

    def test_the_line_scanner_keeps_a_hash_inside_a_string(self, tmp_path):
        path = tmp_path / "uv.toml"
        path.write_text('exclude-newer = "2026-01-01#T00:00:00Z"\n', encoding = "utf-8")
        assert ips._scan_uv_policy_config_by_line([path]) == [
            ("uv.toml", "exclude-newer", '"2026-01-01#T00:00:00Z"')
        ]

    def test_the_line_scanner_skips_a_uv_toml_subtable_that_is_not_pip(self, tmp_path):
        path = tmp_path / "uv.toml"
        path.write_text("[[index]]\nname = 'corp'\n\n[pip]\nno-build = true\n", encoding = "utf-8")
        assert ips._scan_uv_policy_config_by_line([path]) == [("uv.toml", "no-build", "true")]


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
