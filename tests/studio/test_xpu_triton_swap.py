# SPDX-License-Identifier: AGPL-3.0-only
"""Generic Triton must not shadow torch's XPU Triton.

Both distributions own the top-level ``triton`` package, and resolving unsloth against a
pinned ``+xpu`` torch pulls both (uv reports ``pytorch-triton-xpu 3.5.0`` alongside
``triton 3.7.1``), so the CUDA-oriented build can land last and ``torch.compile`` then loads
the wrong library on an Intel GPU.

The swap lives in ``install_python_stack.py`` rather than ``install.sh`` because install.sh
runs setup.sh, which runs this module: one copy covers the fresh install and
``unsloth studio update``, which never touches install.sh.

Three things here are easy to get wrong and are asserted by execution rather than by reading:

* the ORDER. Fetch, then uninstall, then install. Uninstalling last deletes the shared paths
  the XPU build just wrote, because those paths are in generic triton's own RECORD.
* the venv has no pip. ``uv venv`` is created without ``--seed``, so a fresh venv cannot run
  ``pip download`` at all, and without a bootstrap the swap silently never happens.
* the pin is ONE-SHOT. ``UNSLOTH_TORCH_INDEX_FAMILY=xpu ./install.sh`` leaves nothing behind
  in the environment, so a later plain ``unsloth studio update`` must recognise the installed
  ``+xpu`` wheel instead. See TestTheInstalledWheelIsThePin.
"""

import os
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


REPO = Path(__file__).resolve().parents[2]
STACK = REPO / "studio/install_python_stack.py"


def _load_real_index_env_scrub():
    """The module's OWN _install_env_for_cmd, so the scrub is executed, not re-implemented.

    It is defined below the slice the swap comes from, so it is pulled in separately rather
    than stubbed -- a hand-written copy here would agree with a broken original forever.
    """
    import ast as _ast
    import atexit as _atexit
    import functools as _functools
    import locale as _locale
    import os as _os
    import shutil as _shutil
    import subprocess as _subprocess
    import sys as _sys
    import tempfile as _tempfile

    src = STACK.read_text(encoding = "utf-8")
    ns: dict = {
        "os": _os,
        "ast": _ast,
        "atexit": _atexit,
        "functools": _functools,
        "shutil": _shutil,
        "subprocess": _subprocess,
        "locale": _locale,
        "sys": _sys,
        "tempfile": _tempfile,
        # The one dependency of the extracted code that is not a module.
        "_windows_hidden_subprocess_kwargs": dict,
    }
    for anchor, end, keep in (
        ("_UV_INDEX_ENV_VARS = (", "\n)\n", 2),
        # The opt-out, which _install_env_for_cmd now asks before it scrubs. Both halves:
        # the predicate reads the constant, so extracting the function alone is a
        # NameError at call time rather than at exec time.
        ("_POLICY_OPT_OUT_ENV = ", "\n", 1),
        ("def _respect_pm_policy(", "\n\ndef ", 0),
        # Resolved from this namespace at CALL time, so an omission is a NameError later.
        ("_PM_HASH_ENV_VARS = (", "\n)\n", 2),
        ("_PM_FORCE_SOURCE_ENV_VARS = (", "\n)\n", 2),
        # One line, so it ends at the first newline; "\n)\n" would swallow the file.
        ("_PINNED_PIP_CONFIG_KEEP_KEYS = (", "\n)\n", 2),
        ("_PINNED_PIP_CONFIG_GLOBAL_SECTION = ", "\n", 1),
        ("_PINNED_PIP_CONFIG_DEFAULT_SECTION = ", "\n", 1),
        ("_PINNED_PIP_CONFIG_SEPARATORS = ", "\n", 1),
        ("_PINNED_PIP_CONFIG_ACCUMULATING = ", "\n", 1),
        ("_PINNED_PIP_CONFIG_LISTING: ", "\n", 1),
        ("_PINNED_PIP_CONFIG_TIMEOUT = ", "\n", 1),
        ("_PINNED_PIP_CONFIG_ATTEMPTS = ", "\n", 1),
        ("def _pip_subcommand_of(", "\n\ndef ", 0),
        ("def _decode_pip_output(", "\n\ndef ", 0),
        ("def _pinned_pip_config_overrides(", "\n\ndef ", 0),
        # Omitting it made the exec'd scrub raise into a broad except and agree with
        # anything.
        ("def _parse_pinned_pip_config(", "\n\ndef ", 0),
        ("def _relaxed_pip_policy_env(", "\n\ndef ", 0),
        ("def _is_pip_subcommand(", "\n\ndef ", 0),
        ("def _is_pinned_index_cmd(", "\n\ndef ", 0),
        ("def _install_env_for_cmd(", "\n\ndef ", 0),
    ):
        start = src.index(anchor)
        exec(compile(src[start : src.index(end, start) + keep], str(STACK), "exec"), ns)
    assert "PIP_NO_INDEX" in ns["_UV_INDEX_ENV_VARS"], "extraction lost the pip vars"
    assert "PIP_REQUIRE_HASHES" in ns["_PM_HASH_ENV_VARS"], "extraction lost the hash vars"
    # Execute it once: a missing dependency here is otherwise an inert scrub that passes.
    parsed = ns["_parse_pinned_pip_config"](b"global.cert='/etc/corp/ca.pem'\n")
    assert parsed == {"PIP_CERT": "/etc/corp/ca.pem"}, f"extraction is inert: {parsed}"
    # Exercise the opt-out too, for the same reason: a predicate that always answers False
    # would let every decline test pass by never being asked.
    with mock.patch.dict(os.environ, {ns["_POLICY_OPT_OUT_ENV"]: "1"}):
        assert ns["_respect_pm_policy"](), "the opt-out predicate is inert"
    return ns["_install_env_for_cmd"], ns["_respect_pm_policy"], ns["_POLICY_OPT_OUT_ENV"]


(
    _real_install_env_for_cmd,
    _real_respect_pm_policy,
    _POLICY_OPT_OUT_ENV,
) = _load_real_index_env_scrub()


def _load(
    monkeypatch,
    tmp_path,
    *,
    spec,
    generic,
    has_pip = True,
    ensurepip_works = True,
    download_ok = True,
    drops_wheel = True,
    uninstall_ok = True,
    install_ok = True,
    pinned = True,
    torch_label = "2.9.1+xpu",
):
    """Import the module with the world stubbed, and return (module, action log)."""
    log: list[str] = []
    counted: list[int] = []

    mod = types.ModuleType("_stack_under_test")
    src = STACK.read_text(encoding = "utf-8")
    # Only these helpers are needed; importing the whole module would run the installer.
    start = src.index("def _installed_torch_version_label() -> str:")
    end = src.index("def _ensure_cpu_torch() -> None:")
    body = src[start:end]
    assert "_ensure_xpu_triton" in body, "extraction lost the swap"
    assert "_ensure_venv_pip" in body, "extraction lost the pip bootstrap"
    # The WARN assertions need the stub wired to the name the slice actually calls; a rename would leave them silently
    # dead.
    assert "_safe_print(" in body, "extraction lost the print helper the WARN stub hooks"

    import glob as _glob
    import importlib.util as _importlib_util
    import os as _os
    import re as _re
    import shutil as _shutil
    import tempfile as _tempfile

    # A real torch/version.py on disk, so the label read is executed rather than faked.
    # Only the LOCATION step is stubbed, since find_spec would resolve this process's own torch.
    _pkg = tmp_path / "torch"
    _pkg.mkdir()
    (_pkg / "__init__.py").write_text("raise AssertionError('torch must never be imported')\n")
    if torch_label is not None:
        (_pkg / "version.py").write_text(
            f"from typing import Optional\n__version__ = '{torch_label}'\ndebug = False\n"
        )
    monkeypatch.setattr(
        _importlib_util,
        "find_spec",
        lambda name: (
            types.SimpleNamespace(origin = str(_pkg / "__init__.py"))
            if name == "torch" and torch_label is not None
            else None
        ),
    )

    pip_state = {"present": has_pip}
    index_urls: list[str] = []
    download_envs: list = []

    def fake_run(cmd, **kw):
        joined = " ".join(str(c) for c in cmd)
        if "download" in cmd:
            download_envs.append(kw.get("env"))
        if "-m pip --version" in joined or ("pip" in cmd and "--version" in cmd):
            return subprocess.CompletedProcess(cmd, 0 if pip_state["present"] else 1)
        if "ensurepip" in joined:
            log.append("ENSUREPIP")
            if ensurepip_works:
                pip_state["present"] = True
            return subprocess.CompletedProcess(cmd, 0)
        if "importlib.metadata" in joined:
            out = f"SPEC={spec}\nGENERIC={generic}\n".encode()
            return subprocess.CompletedProcess(cmd, 0, stdout = out)
        if "download" in cmd:
            log.append("DOWNLOAD")
            index_urls.append(cmd[cmd.index("--index-url") + 1])
            if download_ok and drops_wheel:
                target = cmd[cmd.index("-d") + 1]
                Path(target, "pytorch_triton_xpu-3.5.0-py3-none-any.whl").write_bytes(b"")
            return subprocess.CompletedProcess(cmd, 0 if download_ok else 1, stdout = b"")
        if "uninstall" in cmd:
            log.append("UNINSTALL")
            return subprocess.CompletedProcess(cmd, 0 if uninstall_ok else 1)
        return subprocess.CompletedProcess(cmd, 0, stdout = b"")

    def fake_pip_install_try(label, *args, **kw):
        if label.startswith("pip"):
            log.append("BOOTSTRAP")
            if ensurepip_works:
                pip_state["present"] = True
            return pip_state["present"]
        log.append("INSTALL")
        return True

    def fake_pip_install(label, *args, **kw):
        # The real one exits the process via run(), which is what keeps the completion manifest unwritten, so the stub
        # raises SystemExit rather than returning.
        log.append("INSTALL")
        if not install_ok:
            raise SystemExit(1)

    ns = {
        "subprocess": types.SimpleNamespace(
            run = fake_run,
            CompletedProcess = subprocess.CompletedProcess,
            TimeoutExpired = subprocess.TimeoutExpired,
            DEVNULL = subprocess.DEVNULL,
            PIPE = subprocess.PIPE,
            STDOUT = subprocess.STDOUT,
        ),
        "sys": sys,
        "glob": _glob,
        "importlib": types.SimpleNamespace(util = _importlib_util, invalidate_caches = lambda: None),
        "os": _os,
        "re": _re,
        "shutil": _shutil,
        "tempfile": _tempfile,
        "Path": Path,
        "NO_TORCH": False,
        "IS_MACOS": False,
        "IS_WINDOWS": False,
        "_PYTORCH_WHL_BASE": "https://download.pytorch.org/whl",
        "_install_env_for_cmd": _real_install_env_for_cmd,
        # The real predicate, not a stub: a decline test is only worth anything if the
        # thing deciding is the code that ships.
        "_respect_pm_policy": _real_respect_pm_policy,
        "_POLICY_OPT_OUT_ENV": _POLICY_OPT_OUT_ENV,
        "_explicit_xpu_torch_index_url": (
            (lambda: "https://download.pytorch.org/whl/xpu") if pinned else (lambda: None)
        ),
        "pip_install_try": fake_pip_install_try,
        "pip_install": fake_pip_install,
        # The slice removes the generic triton itself, outside pip_install, and the final
        # pip check only runs when something said it changed the environment.
        "_count_install_action": lambda: counted.append(1),
        "_red": lambda s: s,
        # _ensure_xpu_triton reads the setup-script handover through this helper rather
        # than inline, so the slice needs it by name or the guard tests NameError at call
        # time. Same semantics as the real one: the env var, lowercased.
        "_handover_torch_flavor_tag": (
            lambda: os.environ.get("UNSLOTH_EXPECTED_TORCH_TAG", "").strip().lower()
        ),
        # _safe_print, not print: the slice calls it by name, so stubbing "print" would leave _safe_print undefined at
        # exec time.
        # The policy decline is recorded by NAME rather than by the phrase: it spells the outcome "leaving it in place",
        # so "left in place" alone read it as no warning at all, while widening that literal to "in place" would also
        # start recording the failed-uninstall warning, whose log the order assertions read exactly.
        "_safe_print": (
            lambda *a, **k: (
                log.append("WARN")
                if a and ("left in place" in str(a[0]) or _POLICY_OPT_OUT_ENV in str(a[0]))
                else None
            )
        ),
    }
    exec(compile(body, str(STACK), "exec"), ns)
    mod.__dict__.update(ns)
    mod.__dict__["_test_counted"] = counted
    mod.__dict__["_test_index_urls"] = index_urls
    mod.__dict__["_test_download_envs"] = download_envs
    return mod, log


def _run(monkeypatch, tmp_path, **kw):
    mod, log = _load(monkeypatch, tmp_path, **kw)
    mod.__dict__["_ensure_xpu_triton"]()
    return log


class TestXpuTritonSwap:
    def test_orders_fetch_uninstall_install(self, monkeypatch, tmp_path):
        # The whole point: the uninstall sits between the fetch and the install.
        log = _run(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        assert log == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    def test_the_uninstall_counts_as_an_environment_change(self, monkeypatch, tmp_path):
        """It removes a distribution outside pip_install, and the pass's final pip check only
        runs when something said the environment moved. Uncounted, dropping generic triton on
        an XPU host would leave torch's dependency unmet with nothing left to report it."""
        mod, _log = _load(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        mod.__dict__["_ensure_xpu_triton"]()
        assert mod.__dict__["_test_counted"], "the uninstall did not count as a change"

    def test_handles_the_triton_xpu_rename(self, monkeypatch, tmp_path):
        # torch 2.10 renamed the distribution; the spec is read from torch, never hardcoded.
        log = _run(monkeypatch, tmp_path, spec = "triton-xpu==3.6.0", generic = "3.7.1")
        assert log == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    def test_bootstraps_pip_when_the_venv_has_none(self, monkeypatch, tmp_path):
        log = _run(
            monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1", has_pip = False
        )
        assert log[0] == "ENSUREPIP"
        assert log[-3:] == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    def test_falls_back_to_installing_pip(self, monkeypatch, tmp_path):
        # uv venv has no --seed, so a fresh venv cannot run pip download at all.
        log = _run(
            monkeypatch,
            tmp_path,
            spec = "pytorch-triton-xpu==3.5.0",
            generic = "3.7.1",
            has_pip = False,
            ensurepip_works = False,
        )
        # ensurepip failed, so it tries a real pip install; that fails too, and the swap must warn rather than uninstall
        # with nothing to install from.
        assert "BOOTSTRAP" in log
        assert "UNINSTALL" not in log

    @pytest.mark.parametrize(
        "spec, generic",
        [
            ("pytorch-triton-xpu==3.5.0", ""),  # nothing shadowing it
            ("triton==3.7.1", "3.7.1"),  # torch is not the +xpu wheel
            ("", "3.7.1"),  # torch declares no triton at all
        ],
    )
    def test_leaves_a_healthy_venv_alone(self, monkeypatch, tmp_path, spec, generic):
        assert _run(monkeypatch, tmp_path, spec = spec, generic = generic) == []

    def test_a_dead_mirror_removes_nothing(self, monkeypatch, tmp_path):
        # Warn and leave the venv working; never uninstall with nothing to install from.
        log = _run(
            monkeypatch,
            tmp_path,
            spec = "pytorch-triton-xpu==3.5.0",
            generic = "3.7.1",
            download_ok = False,
        )
        assert "UNINSTALL" not in log and "INSTALL" not in log

    def test_a_successful_exit_with_no_wheel_removes_nothing(self, monkeypatch, tmp_path):
        # The exit code alone is not enough: no wheel on disk means nothing to install from.
        log = _run(
            monkeypatch,
            tmp_path,
            spec = "pytorch-triton-xpu==3.5.0",
            generic = "3.7.1",
            drops_wheel = False,
        )
        assert "UNINSTALL" not in log and "INSTALL" not in log


class TestFailedSwapIsNotSurvivable:
    def test_a_failed_uninstall_changes_nothing(self, monkeypatch, tmp_path):
        # A read-only or locked venv leaves generic triton registered; installing over it would let
        # a later upgrade delete the shared files again and repeat the swap every pass.
        log = _run(
            monkeypatch,
            tmp_path,
            spec = "pytorch-triton-xpu==3.5.0",
            generic = "3.7.1",
            uninstall_ok = False,
        )
        assert log == ["DOWNLOAD", "UNINSTALL"]
        assert "INSTALL" not in log

    def test_a_failed_install_propagates(self, monkeypatch, tmp_path):
        # The uninstall already took the shared files, so a warning would commit a venv with a broken torch.compile
        # that the next update fast-paths past (generic triton is gone, so nothing is left to trigger on).
        with pytest.raises(SystemExit):
            _run(
                monkeypatch,
                tmp_path,
                spec = "pytorch-triton-xpu==3.5.0",
                generic = "3.7.1",
                install_ok = False,
            )


class TestTheInstalledWheelIsThePin:
    """`UNSLOTH_TORCH_INDEX_FAMILY=xpu ./install.sh` is a ONE-SHOT pin.

    It is gone from the environment by the next plain `unsloth studio update`, yet that
    update's dependency pass can pull generic triton back in (unsloth declares triton as a
    core dep). Gating the swap on the pin alone therefore leaves every already-installed
    XPU venv shadowed forever. The +xpu wheel on disk is the durable signal, and setup.sh
    already raises the bitsandbytes floor off exactly that.
    """

    def test_swaps_with_no_pin_when_torch_is_the_xpu_wheel(self, monkeypatch, tmp_path):
        log = _run(
            monkeypatch,
            tmp_path,
            spec = "pytorch-triton-xpu==3.5.0",
            generic = "3.7.1",
            pinned = False,
        )
        assert log == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    def test_no_pin_falls_back_to_the_default_xpu_index(self, monkeypatch, tmp_path):
        mod, _ = _load(
            monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1", pinned = False
        )
        mod.__dict__["_ensure_xpu_triton"]()
        assert mod.__dict__["_test_index_urls"] == ["https://download.pytorch.org/whl/xpu"]

    @pytest.mark.parametrize("label", ["2.9.1+cu128", "2.9.1+rocm6.4", "2.9.1", "", None])
    def test_no_pin_and_no_xpu_wheel_does_nothing(self, monkeypatch, tmp_path, label):
        # No pin and no +xpu torch is an ordinary CUDA/ROCm/CPU venv, where generic triton is correct and removing it
        # would break torch.compile.
        assert (
            _run(
                monkeypatch,
                tmp_path,
                spec = "pytorch-triton-xpu==3.5.0",
                generic = "3.7.1",
                pinned = False,
                torch_label = label,
            )
            == []
        )

    def test_the_label_is_read_off_disk_not_imported(self, monkeypatch, tmp_path):
        # The fake torch/__init__.py raises, so reaching the swap proves the label came from version.py.
        # `import torch` loads the SYCL runtime and wedges on a stalled Intel driver.
        mod, _ = _load(
            monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1", pinned = False
        )
        assert mod.__dict__["_installed_torch_version_label"]() == "2.9.1+xpu"

    def test_an_explicit_pin_still_wins(self, monkeypatch, tmp_path):
        # A pinned mirror must be used verbatim, not replaced by the default index.
        mod, _ = _load(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        mod.__dict__["_ensure_xpu_triton"]()
        assert mod.__dict__["_test_index_urls"] == ["https://download.pytorch.org/whl/xpu"]


class TestTheFetchIgnoresTheUsersIndexEnvironment:
    """`pip download` honours PIP_* exactly like `pip install`, and that breaks the pin.

    PIP_NO_INDEX makes pip ignore --index-url outright, and PIP_EXTRA_INDEX_URL /
    PIP_FIND_LINKS are consulted IN ADDITION to it. Either the fetch fails, leaving generic
    triton shadowing the XPU build, or the wheel arrives from an index the pin never named.
    Every other pinned install in this file already routes through _install_env_for_cmd; this
    one is a raw subprocess.run, so it has to ask for the same scrub explicitly.
    """

    @pytest.mark.parametrize(
        "var, value",
        [
            ("PIP_NO_INDEX", "1"),
            ("PIP_INDEX_URL", "https://mirror.internal/simple"),
            ("PIP_EXTRA_INDEX_URL", "https://mirror.internal/simple"),
            ("PIP_FIND_LINKS", "/opt/wheels"),
            ("UV_INDEX_URL", "https://mirror.internal/simple"),
        ],
    )
    def test_the_fetch_drops_index_environment(self, monkeypatch, tmp_path, var, value):
        monkeypatch.setenv(var, value)
        mod, _ = _load(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        mod.__dict__["_ensure_xpu_triton"]()
        env = mod.__dict__["_test_download_envs"][0]
        assert env is not None, "the fetch inherited the ambient environment"
        assert var not in env

    def test_the_fetch_neutralises_the_pip_config_file(self, monkeypatch, tmp_path):
        # A config no-index outranks the CLI pin, and devnull is the only spelling that
        # reaches a SITE or GLOBAL file.
        mod, _ = _load(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        mod.__dict__["_ensure_xpu_triton"]()
        env = mod.__dict__["_test_download_envs"][0]
        assert env["PIP_CONFIG_FILE"] == os.devnull
        assert env["UV_NO_CONFIG"] == "1"

    def test_the_fetch_keeps_the_operators_build_policy(self, monkeypatch, tmp_path):
        # The pin is a wheel, so only-binary costs it nothing; this is `pip download`.
        monkeypatch.setenv("PIP_ONLY_BINARY", ":all:")
        monkeypatch.setenv("PIP_REQUIRE_HASHES", "1")
        monkeypatch.setenv("UV_EXCLUDE_NEWER", "2024-01-01T00:00:00Z")
        mod, _ = _load(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        mod.__dict__["_ensure_xpu_triton"]()
        env = mod.__dict__["_test_download_envs"][0]
        assert env["PIP_ONLY_BINARY"] == ":all:"
        assert "PIP_REQUIRE_HASHES" not in env
        # The pip leg cannot express an upload cutoff.
        assert "UV_EXCLUDE_NEWER" not in env

    def test_unrelated_environment_survives(self, monkeypatch, tmp_path):
        # Scrub the index vars, not the environment: HTTPS_PROXY and friends are how a corporate host reaches the index
        # at all.
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
        mod, _ = _load(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        mod.__dict__["_ensure_xpu_triton"]()
        env = mod.__dict__["_test_download_envs"][0]
        assert env["HTTPS_PROXY"] == "http://proxy.internal:8080"


class TestThePolicyOptOutDeclinesTheSwap:
    """UNSLOTH_RESPECT_PM_POLICY=1 skips the swap whole, before anything is touched.

    The swap uninstalls generic triton and installs a wheel fetched from the pinned index,
    and there is no expected hash to check that wheel against: hashing the artifact we just
    fetched and handing the digest back approves it with itself. So on a host whose operator
    has declined the installer's policy overrides, the swap is skipped rather than half
    done. Skipped WHOLE, not "fetch and then stop": generic triton stays, which costs an
    Intel host torch.compile on the XPU, where a venv with no triton at all is a state no
    rerun could repair.

    The guard's placement is the load-bearing part, and placement is only assertable by
    execution. It sits ABOVE _ensure_venv_pip(), which runs ensurepip and can pip install
    pip into a seedless venv: two steps a hash policy would refuse, and two mutations. A
    decline that has already changed the venv is not a decline. It is above the "replacing
    triton" line too, so the operator is not told the swap is happening and then told it is
    not.
    """

    SWAP = {"spec": "pytorch-triton-xpu==3.5.0", "generic": "3.7.1"}

    def test_the_swap_declines_and_touches_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setenv(_POLICY_OPT_OUT_ENV, "1")
        mod, log = _load(monkeypatch, tmp_path, **self.SWAP)
        mod.__dict__["_ensure_xpu_triton"]()
        assert log == ["WARN"], "the decline must warn and do nothing else"
        for action in ("DOWNLOAD", "UNINSTALL", "INSTALL"):
            assert action not in log
        # Nothing moved, so the pass's final pip check has nothing to report either.
        assert mod.__dict__["_test_counted"] == []
        assert mod.__dict__["_test_download_envs"] == []

    @pytest.mark.parametrize("ensurepip_works", [True, False])
    def test_the_pip_bootstrap_is_never_reached(self, monkeypatch, tmp_path, ensurepip_works):
        """The reason the guard sits above _ensure_venv_pip() rather than below it.

        `uv venv` is created without --seed, so this is the ordinary fresh-venv state, not
        a corner: has_pip=False means the bootstrap WOULD run. Both ensurepip outcomes,
        because the fallback `pip install pip` is the second mutation and a guard placed
        between the two would still pass a test that only watched ensurepip.
        """
        monkeypatch.setenv(_POLICY_OPT_OUT_ENV, "1")
        log = _run(
            monkeypatch,
            tmp_path,
            has_pip = False,
            ensurepip_works = ensurepip_works,
            **self.SWAP,
        )
        assert "ENSUREPIP" not in log, "ensurepip ran: the decline mutated a seedless venv"
        assert "BOOTSTRAP" not in log, "pip was installed before the swap declined"
        assert log == ["WARN"]

    def test_without_the_opt_out_the_same_scenario_still_swaps(self, monkeypatch, tmp_path):
        """The negative control. Every assertion above is satisfied for free by a swap that
        never triggers, so the identical call without the variable has to do the work."""
        monkeypatch.delenv(_POLICY_OPT_OUT_ENV, raising = False)
        assert _run(monkeypatch, tmp_path, **self.SWAP) == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    def test_without_the_opt_out_the_seedless_venv_is_still_bootstrapped(
        self, monkeypatch, tmp_path
    ):
        """The other half of the control: the very steps test_the_pip_bootstrap_is_never
        _reached asserts are absent do happen on the same scenario with the variable unset,
        so that test is watching a bootstrap that would otherwise have run."""
        monkeypatch.delenv(_POLICY_OPT_OUT_ENV, raising = False)
        log = _run(monkeypatch, tmp_path, has_pip = False, **self.SWAP)
        assert log[0] == "ENSUREPIP"
        assert log[-3:] == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    @pytest.mark.parametrize(
        "value, declines",
        [("1", True), ("on", True), ("TRUE", True), ("0", False), ("garbage", False)],
    )
    def test_the_boolish_set_is_the_modules_own(self, monkeypatch, tmp_path, value, declines):
        """One variable, one answer, across install.sh, install.ps1, setup.ps1 and here.
        An unrecognised value reads as OFF deliberately, so a typo takes the swap rather
        than silently leaving an Intel host with torch.compile off the XPU."""
        monkeypatch.setenv(_POLICY_OPT_OUT_ENV, value)
        log = _run(monkeypatch, tmp_path, **self.SWAP)
        assert (log == ["WARN"]) is declines, log


class TestADeadDriverIsNotAFlavourMismatch:
    """A wedged `import torch` under a SUPPORTED +xpu wheel is a driver, not a bad wheel.

    _ensure_xpu_torch used to read every inconclusive probe as "repair", and it runs at two
    repair points: on a stalled Arc host that is two 90-second hangs plus two force-reinstalls
    of the whole multi-gigabyte trio, every single update, fixing nothing. The disk answers
    the question the probe cannot, so an unsupported or missing wheel still repairs while a
    supported one gets the driver warning.
    """

    @pytest.mark.parametrize(
        "label, supported",
        [
            ("2.6.0+xpu", True),
            ("2.9.1+xpu", True),
            ("2.10.0+xpu", True),
            ("2.5.1+xpu", False),  # below the floor unsloth raises at
            ("2.11.0+xpu", False),  # past the tested ceiling
            ("3.0.0+xpu", False),
            ("2.9.1+cu128", False),
            ("2.9.1+rocm6.4", False),
            ("2.9.1", False),
            ("", False),
            (None, False),  # no torch on disk at all
        ],
    )
    def test_the_supported_range_matches_the_probe(self, monkeypatch, tmp_path, label, supported):
        mod, _ = _load(monkeypatch, tmp_path, spec = "", generic = "", torch_label = label)
        assert mod.__dict__["_xpu_wheel_supported_on_disk"]() is supported

    def test_the_disk_check_and_the_probe_agree_on_the_bounds(self):
        # Two copies of the range in different places; a drifted floor installs an environment that raises at import.
        src = STACK.read_text(encoding = "utf-8")
        assert src.count("(2, 6) <= _n < (2, 11)") == 1, "the probe's range moved"
        assert src.count("(2, 6) <= nums < (2, 11)") == 1, "the disk check's range moved"

    def test_a_timeout_on_a_supported_wheel_reinstalls_nothing(self):
        # Asserted on the source because _ensure_xpu_torch sits above the extracted slice: the early return must come
        # BEFORE the repair reason is set, or the repair runs anyway.
        src = STACK.read_text(encoding = "utf-8")
        start = src.index("def _ensure_xpu_torch() -> None:")
        body = src[start : src.index("def _installed_torch_version_label", start)]
        guard = body.index("_xpu_wheel_supported_on_disk()")
        armed = body.index('_why = "torch could not be probed"')
        assert guard < armed, "the guard runs after the repair is armed"
        assert "return" in body[guard : guard + 400], "the guard does not return"


class TestPlatformGuards:
    @pytest.mark.parametrize("flag", ["NO_TORCH", "IS_MACOS"])
    def test_skipped_where_it_does_not_apply(self, monkeypatch, tmp_path, flag):
        monkeypatch.delenv("UNSLOTH_EXPECTED_TORCH_TAG", raising = False)
        mod, log = _load(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        mod.__dict__[flag] = True
        mod.__dict__["_ensure_xpu_triton"].__globals__[flag] = True
        mod.__dict__["_ensure_xpu_triton"]()
        assert log == []

    def test_windows_defers_to_setup_ps1_when_setup_ps1_ran(self, monkeypatch, tmp_path):
        # setup.ps1 performs the same swap after this file exits, and publishes the handover variable immediately before
        # invoking it.
        monkeypatch.setenv("UNSLOTH_EXPECTED_TORCH_TAG", "xpu")
        mod, log = _load(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        mod.__dict__["_ensure_xpu_triton"].__globals__["IS_WINDOWS"] = True
        mod.__dict__["_ensure_xpu_triton"]()
        assert log == []

    def test_a_direct_windows_run_does_the_swap_itself(self, monkeypatch, tmp_path):
        # Bare `python install_python_stack.py` on Windows has no setup.ps1 postlude, so the core install leaves
        # triton-windows over torch's XPU triton. The absent handover variable is the signal that nobody else will fix
        # it.
        monkeypatch.delenv("UNSLOTH_EXPECTED_TORCH_TAG", raising = False)
        mod, log = _load(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        mod.__dict__["_ensure_xpu_triton"].__globals__["IS_WINDOWS"] = True
        mod.__dict__["_ensure_xpu_triton"]()
        assert "INSTALL" in log


def test_the_swap_is_wired_in_at_every_repair_point():
    src = STACK.read_text(encoding = "utf-8")
    assert src.count("        _ensure_xpu_triton()") == 3


def test_the_swap_runs_after_every_torch_migration():
    """Order, not just presence.

    The swap keys off the INSTALLED +xpu label. Run between two migrations, an explicit CPU pin
    over an XPU venv gets XPU triton installed and then torch replaced with the CPU build under
    it: a CPU environment whose top-level triton package is the XPU implementation, with its
    declared generic triton gone. Asserted on the AST so a reflow cannot fake it.
    """
    import ast as _ast

    blocks = []
    for node in _ast.walk(_ast.parse(STACK.read_text(encoding = "utf-8"))):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue  # Lambda / IfExp carry a single expression here, not a statement list
        calls = [
            s.value.func.id
            for s in body
            if isinstance(s, _ast.Expr)
            and isinstance(s.value, _ast.Call)
            and isinstance(s.value.func, _ast.Name)
            and s.value.func.id.startswith("_ensure_")
        ]
        if "_ensure_xpu_triton" in calls:
            blocks.append(calls)
    assert len(blocks) == 3, f"expected 3 repair blocks, found {len(blocks)}: {blocks}"
    for calls in blocks:
        assert calls[-1] == "_ensure_xpu_triton", calls
    # Step 13w's migration is _ensure_expected_torch_flavor, which the walk above does not collect (its result is
    # branched on, not discarded), so it is asserted on source.
    migrating = [c for c in blocks if "_ensure_cuda_torch" in c]
    assert len(migrating) == 2, blocks
    for calls in migrating:
        for migration in (
            "_ensure_cuda_torch",
            "_ensure_rocm_torch",
            "_ensure_xpu_torch",
            "_ensure_cpu_torch",
        ):
            assert calls.index(migration) < calls.index("_ensure_xpu_triton"), (migration, calls)
    # The final repair pass would otherwise silently undo the first. The third point is
    # step 13w, the Windows flavor invariant.
    src = STACK.read_text(encoding = "utf-8")
    windows = src[src.index("# 13w.") : src.index("# 14.")]
    assert windows.index("_ensure_expected_torch_flavor") < windows.index(
        "_ensure_xpu_triton"
    ), "the Windows swap must follow that platform's torch migration too"


def test_install_sh_does_not_carry_a_second_copy():
    # It used to. install.sh runs setup.sh, which runs this module, so a copy there is redundant and a place for the two
    # to drift apart.
    assert "replace generic Triton" not in (REPO / "install.sh").read_text(encoding = "utf-8")


class TestCpuRepairSeesAnXpuWheel:
    """An explicit CPU pin must be able to replace a +xpu wheel.

    `_ensure_cpu_torch` classifies the installed build and returns early on "already a CPU
    build". An XPU wheel sets neither `torch.version.cuda` nor `.hip`, so before this it read
    as CPU and the pin was silently ignored. The predicate is executed here, not re-implemented:
    it is pulled out of the module source so a future edit to it is what this test sees.
    """

    @staticmethod
    def _classify(
        ver,
        cuda = "",
        hip = "",
        runtime_xpu = "",
    ):
        src = STACK.read_text(encoding = "utf-8")
        start = src.index("def _ensure_cpu_torch() -> None:")
        seg = src[start : src.index("\n\ndef ", start)]
        # Read the predicate from the module source, so an edit to it is what this test sees rather than a copy that can
        # drift.
        marker = "_is_gpu_build = ("
        begin = seg.index(marker) + len(marker)
        depth, end = 1, begin
        for i in range(begin, len(seg)):
            if seg[i] == "(":
                depth += 1
            elif seg[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        # Re-wrapped in parentheses: the predicate spans several indented lines.
        expr = "(" + seg[begin:end] + ")"
        import re as _re

        return (
            "gpu"
            if eval(
                expr,
                {
                    "re": _re,
                    "_hip": hip,
                    "_cuda": cuda,
                    "_ver": ver.lower(),
                    # The probe's XPU marker, which the predicate reads as a module global.
                    # Empty unless a case states otherwise: torch.version has no word for XPU,
                    # so a wheel is only known to be one through this.
                    "_TORCH_RUNTIME_XPU": runtime_xpu,
                },
            )
            else "cpu"
        )

    def test_xpu_wheel_is_a_gpu_build(self):
        assert self._classify("2.9.1+xpu") == "gpu"

    def test_an_untagged_xpu_wheel_is_a_gpu_build_through_the_runtime_marker(self):
        # A private index serves XPU torch with no +xpu local version, so the tag says
        # nothing and torch.version stays empty on both fields. The runtime marker is the
        # only thing left that knows, and a CPU pin must still reinstall over that build.
        assert self._classify("2.9.1", runtime_xpu = "20250101") == "gpu"

    @pytest.mark.parametrize(
        "ver,cuda,hip,want",
        [
            ("2.9.1+cu128", "12.8", "", "gpu"),
            ("2.9.1+rocm6.4", "", "6.4", "gpu"),
            ("2.9.1+cpu", "", "", "cpu"),
            ("2.9.1", "", "", "cpu"),
        ],
    )
    def test_other_families_are_unchanged(self, ver, cuda, hip, want):
        # The XPU arm must be additive: a CPU build still reads as CPU, or every explicit CPU pin
        # force-reinstalls torch on every update.
        assert self._classify(ver, cuda, hip) == want


class TestCpuPinSurvivesAWedgedImport:
    """A hung `import torch` must not turn an explicit CPU pin into a no-op.

    The classifier probe has a 90s timeout, and on a wedged Intel driver `import torch` blocks
    in the SYCL runtime until it fires -- which is exactly the host the pin is meant to
    rescue. Returning there meant the one case that needed the repair never got it.
    """

    @staticmethod
    def _fn(name):
        src = STACK.read_text(encoding = "utf-8")
        start = src.index(f"def {name}(")
        body = src[start : src.index("\n\ndef ", start)]
        ns: dict = {
            "re": __import__("re"),
            "importlib": __import__("importlib.util", fromlist = ["util"]),
            "Path": Path,
        }
        exec(compile(body, str(STACK), "exec"), ns)
        return ns[name]

    @pytest.mark.parametrize(
        "label,want",
        [
            ("2.9.1+xpu", True),
            ("2.9.1+cu128", True),
            ("2.9.1+rocm6.4", True),
            ("2.9.1+cpu", False),
            ("2.9.1", False),
            ("", False),
        ],
    )
    def test_gpu_label_classification(self, label, want):
        # The CPU/untagged rows matter most: a slow but healthy CPU-only host must not force-reinstall torch on every
        # update.
        assert self._fn("_is_gpu_torch_label")(label) is want

    def test_timeout_falls_through_to_the_repair(self):
        src = STACK.read_text(encoding = "utf-8")
        start = src.index("def _ensure_cpu_torch() -> None:")
        body = src[start : src.index("\n\ndef ", start)]
        stalled = body.index("if not _ran:")
        guard = body.index("_is_gpu_torch_label(_installed_torch_label_on_disk())", stalled)
        # A merely slow CPU-only host returns; a GPU label on disk falls through...
        assert "return" in body[guard : guard + 200]
        # ...and the repair below must accept the probe-less path, or the one host that
        # needs the pin enforced is the one host that never gets it.
        repair = body.index("if not _ran or not _importable:")
        assert repair > guard

    def test_the_disk_read_launches_no_interpreter(self):
        # An interpreter here would reintroduce the hang the disk read exists to avoid.
        src = STACK.read_text(encoding = "utf-8")
        start = src.index("def _installed_torch_label_on_disk() -> str:")
        body = src[start : src.index("\n\ndef ", start)]
        assert "subprocess" not in body
        assert "find_spec" in body
