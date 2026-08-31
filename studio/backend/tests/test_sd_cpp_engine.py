# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the sd-cli engine + routing (``sd_cpp_engine.py``).

Hermetic: the binary finder is driven against a tmp filesystem, and ``generate``
runs a fake ``subprocess.Popen`` that emits canned lines and writes the output
PNG -- no real ``sd-cli``, no GPU.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
import types
from pathlib import Path

import pytest

from core.inference import sd_cpp_engine as eng
from core.inference.sd_cpp_engine import (
    ENGINE_DIFFUSERS,
    ENGINE_SD_CPP,
    SdCppEngine,
    find_sd_cpp_binary,
    find_sd_server_binary,
    runtime_env,
    select_diffusion_engine,
)
from core.inference.sd_cpp_args import SdCppGenParams, SdCppModelFiles, SdCppUpscaleParams


# ── binary discovery ────────────────────────────────────────────────────────


@pytest.fixture(autouse = True)
def _isolate_binary_discovery(tmp_path_factory, monkeypatch):
    """Point every hop of the finder at an empty tree, so a real install on the machine running the
    tests cannot satisfy it.

    Clearing ``SD_CLI_PATH`` / ``UNSLOTH_SD_CPP_PATH`` and patching ``Path.home`` is not enough:
    hop 3 goes through ``managed_install_root()``, which honors ``UNSLOTH_STUDIO_HOME`` /
    ``STUDIO_HOME`` and resolves to ``<studio home>/../stable-diffusion.cpp``. Anyone running the
    suite with an Unsloth home set -- which is the documented way to run side-by-side Unsloth instances -- gets
    a real binary back and every "nothing is installed" assertion here fails. Hop 4 (the in-tree
    developer build) has the same problem for anyone who built sd.cpp in the checkout.

    Autouse rather than a helper because the failure does not need a fixture to reach it:
    ``SdCppEngine(binary = None)`` calls the finder from its constructor.
    """
    eng._IDENTITY_MEMO.clear()  # a verdict from another test must never answer for this one
    root = tmp_path_factory.mktemp("no_sd_cpp")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(root / "studio"))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    monkeypatch.setattr(eng, "in_tree_install_root", lambda: root / "in_tree")
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: root / "nohome"))


def _clear_env(monkeypatch):
    monkeypatch.delenv("SD_CLI_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_SD_CPP_PATH", raising = False)


def test_find_prefers_sd_cli_path_env(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    binary = tmp_path / "sd-cli"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv("SD_CLI_PATH", str(binary))
    # even with PATH empty, the direct env wins
    monkeypatch.setattr(eng.shutil, "which", lambda *_a: None)
    assert find_sd_cpp_binary() == str(binary)


def test_find_custom_install_dir_build_layout(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    root = tmp_path / "sdcpp"
    built = root / "build" / "bin" / "sd-cli"
    built.parent.mkdir(parents = True)
    built.write_text("x")
    monkeypatch.setenv("UNSLOTH_SD_CPP_PATH", str(root))
    monkeypatch.setattr(eng.shutil, "which", lambda *_a: None)
    assert find_sd_cpp_binary() == str(built)


def test_find_falls_back_to_path(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(
        eng.shutil, "which", lambda stem: "/usr/bin/sd-cli" if stem == "sd-cli" else None
    )
    assert find_sd_cpp_binary() == "/usr/bin/sd-cli"


def test_find_rejects_unrelated_ubuntu_sd_on_path(tmp_path, monkeypatch, caplog):
    _clear_env(monkeypatch)
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(eng.shutil, "which", lambda stem: "/usr/bin/sd" if stem == "sd" else None)
    run_kwargs = {}
    hidden = object()

    def _run(*_args, **kwargs):
        run_kwargs.update(kwargs)
        return types.SimpleNamespace(
            stdout = "Find & replace occurrences of a pattern\nUsage: sd [OPTIONS] <FIND> <REPLACE>",
            stderr = "",
            returncode = 0,
        )

    monkeypatch.setattr(eng.subprocess, "run", _run)
    monkeypatch.setattr(eng, "windows_hidden_subprocess_kwargs", lambda: {"startupinfo": hidden})

    assert find_sd_cpp_binary() is None
    assert "does not identify stable-diffusion.cpp" in caplog.text
    assert run_kwargs["timeout"] == 10
    assert run_kwargs["startupinfo"] is hidden


def test_find_rejects_legacy_sd_when_identity_probe_times_out(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(eng.shutil, "which", lambda stem: "/usr/bin/sd" if stem == "sd" else None)

    def _timeout(*_args, **_kwargs):
        raise eng.subprocess.TimeoutExpired("sd", 10)

    monkeypatch.setattr(eng.subprocess, "run", _timeout)

    assert find_sd_cpp_binary() is None


def test_find_accepts_genuine_legacy_sd_on_path(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(
        eng.shutil, "which", lambda stem: "/opt/sd.cpp/bin/sd" if stem == "sd" else None
    )
    monkeypatch.setattr(
        eng.subprocess,
        "run",
        lambda *_a, **_k: types.SimpleNamespace(
            stdout = (
                "usage: /opt/sd.cpp/bin/sd [arguments]\n"
                "--model [MODEL]\n--negative-prompt PROMPT\n--cfg-scale SCALE\n--steps STEPS"
            ),
            stderr = "",
            returncode = 0,
        ),
    )

    assert find_sd_cpp_binary() == "/opt/sd.cpp/bin/sd"


def test_rejected_legacy_sd_allows_managed_install(tmp_path, monkeypatch):
    import core.inference.sd_cpp_backend as backend

    _clear_env(monkeypatch)
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(eng.shutil, "which", lambda stem: "/usr/bin/sd" if stem == "sd" else None)
    monkeypatch.setattr(
        eng.subprocess,
        "run",
        lambda *_a, **_k: types.SimpleNamespace(
            stdout = "Find & replace occurrences of a pattern\nUsage: sd [OPTIONS] <FIND> <REPLACE>",
            stderr = "",
            returncode = 0,
        ),
    )
    installed = tmp_path / "managed" / "sd-cli"
    installs = []

    def _install(**kwargs):
        installs.append(kwargs)
        return installed

    stub = types.ModuleType("install_sd_cpp_prebuilt")
    stub.install = _install
    monkeypatch.setitem(sys.modules, "install_sd_cpp_prebuilt", stub)

    assert backend.ensure_sd_cpp_binary(accelerator = "cpu") == str(installed)
    assert installs == [{"accelerator": "cpu"}]


def test_identity_probe_is_memoized_per_file_revision(tmp_path, monkeypatch):
    # Discovery runs on every load, and ensure_sd_cpp_binary resolves twice on its own, so an
    # unrelated `sd` was re-executed several times per load -- once per full 10s timeout when the
    # candidate hangs. The verdict is keyed on the file, not the path, so an in-place replacement
    # is still re-probed rather than answered from a stale entry.
    _clear_env(monkeypatch)
    candidate = tmp_path / "sd"
    candidate.write_text("#!/bin/sh\n")
    monkeypatch.setattr(eng.shutil, "which", lambda stem: str(candidate) if stem == "sd" else None)
    runs: list[list[str]] = []

    def _run(cmd, **_kwargs):
        runs.append(cmd)
        return types.SimpleNamespace(
            stdout = "Find & replace occurrences of a pattern\n", stderr = "", returncode = 0
        )

    monkeypatch.setattr(eng.subprocess, "run", _run)

    assert find_sd_cpp_binary() is None
    assert find_sd_cpp_binary() is None
    assert len(runs) == 1

    # Replaced in place by a genuine build: a new revision, so the old verdict does not apply.
    candidate.write_text("#!/bin/sh\n# a real stable-diffusion.cpp build now\n")
    os.utime(candidate, (0, 0))

    def _run_real(cmd, **_kwargs):
        runs.append(cmd)
        return types.SimpleNamespace(
            stdout = "stable-diffusion.cpp version unknown\n", stderr = "", returncode = 0
        )

    monkeypatch.setattr(eng.subprocess, "run", _run_real)
    assert find_sd_cpp_binary() == str(candidate)
    assert len(runs) == 2


def test_identity_probe_rekeys_a_timestamp_preserving_replacement(tmp_path, monkeypatch):
    # cp -p / shutil.copy2 / an archive carrying source timestamps restore the mtime of the file
    # they overwrite, so path + mtime + size alone would serve the old verdict for a different
    # program. The inode change time is not restorable that way.
    _clear_env(monkeypatch)
    candidate = tmp_path / "sd"
    candidate.write_text("A" * 64)
    stamp = os.stat(candidate)
    monkeypatch.setattr(eng.shutil, "which", lambda stem: str(candidate) if stem == "sd" else None)
    runs: list[list[str]] = []

    def _reject(cmd, **_kwargs):
        runs.append(cmd)
        return types.SimpleNamespace(stdout = "Find & replace\n", stderr = "", returncode = 0)

    monkeypatch.setattr(eng.subprocess, "run", _reject)
    assert find_sd_cpp_binary() is None

    # Same path, same size, mtime restored -- a different program underneath.
    candidate.write_text("B" * 64)
    os.utime(candidate, ns = (stamp.st_atime_ns, stamp.st_mtime_ns))
    assert os.stat(candidate).st_mtime_ns == stamp.st_mtime_ns

    def _accept(cmd, **_kwargs):
        runs.append(cmd)
        return types.SimpleNamespace(
            stdout = "stable-diffusion.cpp version unknown\n", stderr = "", returncode = 0
        )

    monkeypatch.setattr(eng.subprocess, "run", _accept)
    assert find_sd_cpp_binary() == str(candidate)
    assert len(runs) == 2


def test_identity_probe_does_not_memoize_a_nonzero_exit_it_learned_nothing_from(
    tmp_path, monkeypatch
):
    # A genuine sd.cpp that cannot load an adjacent shared library exits 127 from the dynamic
    # loader with nothing identifying on either stream. That is a CompletedProcess, not an
    # exception, so it would otherwise be cached as a definitive "not stable-diffusion.cpp"
    # against a file that never changed -- and installing the missing library would not get it
    # re-probed until Unsloth restarted.
    _clear_env(monkeypatch)
    candidate = tmp_path / "sd"
    candidate.write_text("#!/bin/sh\n")
    monkeypatch.setattr(eng.shutil, "which", lambda stem: str(candidate) if stem == "sd" else None)
    monkeypatch.setattr(
        eng.subprocess,
        "run",
        lambda *_a, **_k: types.SimpleNamespace(
            stdout = "",
            stderr = "sd: error while loading shared libraries: libggml.so: cannot open",
            returncode = 127,
        ),
    )
    assert find_sd_cpp_binary() is None
    assert eng._IDENTITY_MEMO == {}

    # The library is back. Same file, so the same key -- it must be probed again, not answered.
    monkeypatch.setattr(
        eng.subprocess,
        "run",
        lambda *_a, **_k: types.SimpleNamespace(
            stdout = "stable-diffusion.cpp version unknown\n", stderr = "", returncode = 0
        ),
    )
    assert find_sd_cpp_binary() == str(candidate)


def test_identity_probe_memoizes_an_identifying_build_that_exits_nonzero(tmp_path, monkeypatch):
    # The other half: older builds print usage and exit 1. Identifying output settles the question
    # whatever the exit code, so that verdict is decisive and worth keeping.
    _clear_env(monkeypatch)
    candidate = tmp_path / "sd"
    candidate.write_text("#!/bin/sh\n")
    monkeypatch.setattr(eng.shutil, "which", lambda stem: str(candidate) if stem == "sd" else None)
    runs = []
    monkeypatch.setattr(
        eng.subprocess,
        "run",
        lambda cmd, **_k: (
            runs.append(cmd),
            types.SimpleNamespace(
                stdout = "usage: sd\n --negative-prompt P\n --cfg-scale S\n --steps N\n",
                stderr = "",
                returncode = 1,
            ),
        )[1],
    )
    assert find_sd_cpp_binary() == str(candidate)
    assert find_sd_cpp_binary() == str(candidate)
    assert len(runs) == 1


def test_identity_verdict_expires(tmp_path, monkeypatch):
    # No stat tuple is a content hash. On Windows st_ctime is the CREATION time, which an in-place
    # overwrite preserves, so a same-sized write that also restores mtime leaves the whole key
    # unchanged. Hashing the binary on every lookup would cost a full read on a path walked for
    # every load; a short life bounds that staleness instead, and bounds whatever else the key
    # cannot see.
    _clear_env(monkeypatch)
    candidate = tmp_path / "sd"
    candidate.write_text("#!/bin/sh\n")
    monkeypatch.setattr(eng.shutil, "which", lambda stem: str(candidate) if stem == "sd" else None)
    runs = []

    def _reject(cmd, **_kwargs):
        runs.append(cmd)
        return types.SimpleNamespace(stdout = "Find & replace\n", stderr = "", returncode = 0)

    monkeypatch.setattr(eng.subprocess, "run", _reject)
    assert find_sd_cpp_binary() is None
    assert find_sd_cpp_binary() is None
    assert len(runs) == 1  # inside the window, the verdict answers

    clock = [time.monotonic() + eng._IDENTITY_MEMO_TTL_S + 1]
    monkeypatch.setattr(eng.time, "monotonic", lambda: clock[0])

    def _accept(cmd, **_kwargs):
        runs.append(cmd)
        return types.SimpleNamespace(
            stdout = "stable-diffusion.cpp version unknown\n", stderr = "", returncode = 0
        )

    monkeypatch.setattr(eng.subprocess, "run", _accept)
    # Past the window the same unchanged file is probed again, so a replacement the key could not
    # see is picked up rather than being answered from a verdict about the binary it replaced.
    assert find_sd_cpp_binary() == str(candidate)
    assert len(runs) == 2


def test_identity_probe_does_not_memoize_a_probe_that_failed(tmp_path, monkeypatch):
    # A timeout or a failed spawn does not touch the file, so its memo key does not change either.
    # Remembering that "no" would blacklist a genuine build for the life of the process over one
    # slow --help under disk or memory pressure -- Unsloth would have to be restarted to see it.
    _clear_env(monkeypatch)
    candidate = tmp_path / "sd"
    candidate.write_text("#!/bin/sh\n")
    monkeypatch.setattr(eng.shutil, "which", lambda stem: str(candidate) if stem == "sd" else None)

    def _timeout(*_args, **_kwargs):
        raise eng.subprocess.TimeoutExpired("sd", 10)

    monkeypatch.setattr(eng.subprocess, "run", _timeout)
    assert find_sd_cpp_binary() is None
    assert eng._IDENTITY_MEMO == {}

    # Same file, unchanged on disk, and the probe now answers: the earlier failure must not stand in.
    monkeypatch.setattr(
        eng.subprocess,
        "run",
        lambda *_a, **_k: types.SimpleNamespace(
            stdout = "stable-diffusion.cpp version unknown\n", stderr = "", returncode = 0
        ),
    )
    assert find_sd_cpp_binary() == str(candidate)


def test_identity_probe_does_not_memoize_a_candidate_it_cannot_stat(monkeypatch):
    # No key means no cache entry: a path that does not resolve yet must be re-probed once it does,
    # rather than being remembered as "not stable-diffusion.cpp" for the life of the process.
    _clear_env(monkeypatch)
    monkeypatch.setattr(
        eng.shutil, "which", lambda stem: "/nonexistent/sd" if stem == "sd" else None
    )
    runs = []
    monkeypatch.setattr(
        eng.subprocess,
        "run",
        lambda cmd, **_k: (
            runs.append(cmd),
            types.SimpleNamespace(stdout = "unrelated\n", stderr = "", returncode = 0),
        )[1],
    )

    assert find_sd_cpp_binary() is None
    assert find_sd_cpp_binary() is None
    assert len(runs) == 2
    assert eng._IDENTITY_MEMO == {}


def test_find_returns_none_when_absent(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(eng.shutil, "which", lambda *_a: None)
    assert find_sd_cpp_binary() is None


# ── sd-server discovery ──────────────────────────────────────────────────────


def _clear_server_env(monkeypatch):
    monkeypatch.delenv("SD_SERVER_PATH", raising = False)
    monkeypatch.delenv("SD_CLI_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_SD_CPP_PATH", raising = False)


def test_find_server_prefers_sd_server_path_env(tmp_path, monkeypatch):
    _clear_server_env(monkeypatch)
    binary = tmp_path / "sd-server"
    binary.write_text("x")
    monkeypatch.setenv("SD_SERVER_PATH", str(binary))
    monkeypatch.setattr(eng.shutil, "which", lambda *_a: None)
    assert find_sd_server_binary() == str(binary)


def test_find_server_build_layout(tmp_path, monkeypatch):
    _clear_server_env(monkeypatch)
    root = tmp_path / "sdcpp"
    built = root / "build" / "bin" / "sd-server"
    built.parent.mkdir(parents = True)
    built.write_text("x")
    monkeypatch.setenv("UNSLOTH_SD_CPP_PATH", str(root))
    monkeypatch.setattr(eng.shutil, "which", lambda *_a: None)
    assert find_sd_server_binary() == str(built)


def test_find_server_path_fallback(tmp_path, monkeypatch):
    _clear_server_env(monkeypatch)
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(
        eng.shutil, "which", lambda stem: "/usr/bin/sd-server" if stem == "sd-server" else None
    )
    assert find_sd_server_binary() == "/usr/bin/sd-server"


def test_find_server_not_confused_with_sd_cli(tmp_path, monkeypatch):
    # A tree with only sd-cli must NOT be reported as an sd-server (and vice versa), so the backend falls back to one-shot.
    _clear_server_env(monkeypatch)
    root = tmp_path / "sdcpp"
    (root / "build" / "bin").mkdir(parents = True)
    (root / "build" / "bin" / "sd-cli").write_text("x")
    monkeypatch.setenv("UNSLOTH_SD_CPP_PATH", str(root))
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(eng.shutil, "which", lambda *_a: None)
    assert find_sd_server_binary() is None
    assert find_sd_cpp_binary() == str(root / "build" / "bin" / "sd-cli")


# ── availability / version ──────────────────────────────────────────────────


def test_engine_unavailable_when_no_binary(monkeypatch):
    # Force the "no binary anywhere" condition so the test is hermetic on a host that happens to have sd-cli installed.
    monkeypatch.setattr(eng, "find_sd_cpp_binary", lambda: None)
    e = SdCppEngine(binary = None)
    assert e.is_available() is False
    assert e.version() is None


def test_engine_version_parsed_and_cached(tmp_path, monkeypatch):
    binary = tmp_path / "sd-cli"
    binary.write_text("x")
    e = SdCppEngine(binary = str(binary))
    calls = {"n": 0}

    def _fake_run(*_a, **_k):
        calls["n"] += 1
        return types.SimpleNamespace(
            stdout = "stable-diffusion.cpp version master-721\n", stderr = "", returncode = 0
        )

    monkeypatch.setattr(eng.subprocess, "run", _fake_run)
    assert e.version() == "stable-diffusion.cpp version master-721"
    assert e.version() == "stable-diffusion.cpp version master-721"
    assert calls["n"] == 1  # cached after the first probe


# ── runtime env (bundled shared libs) ───────────────────────────────────────


def test_runtime_env_prepends_binary_dir_to_lib_path():
    var = eng._lib_path_var()
    env = runtime_env("/opt/sdcpp/bin/sd-cli", {var: "/existing"})
    first = env[var].split(os.pathsep)[0]
    assert first == "/opt/sdcpp/bin"
    assert "/existing" in env[var]


def test_runtime_env_scrubs_native_path_lease_secret(monkeypatch):
    # The sd-cli child is an external process and must never receive the native-path lease secret; every launch funnels through runtime_env.
    monkeypatch.setenv("UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET", "top-secret")
    from_os = runtime_env("/opt/sdcpp/bin/sd-cli")
    assert "UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET" not in from_os
    from_base = runtime_env(
        "/opt/sdcpp/bin/sd-cli",
        {"UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET": "top-secret"},
    )
    assert "UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET" not in from_base


def test_runtime_env_handles_missing_lib_path():
    var = eng._lib_path_var()
    env = runtime_env("/opt/sdcpp/bin/sd-cli", {})
    assert env[var] == "/opt/sdcpp/bin"


def test_terminate_reaps_killed_child():
    # Cancellation/timeout paths call _terminate then raise, so it must reap the killed child itself or a burst of image
    # cancellations leaves zombies. After _terminate the returncode is set, so nothing lingers.
    import subprocess
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session = (os.name == "posix"),
    )
    try:
        eng._terminate(proc)
        assert proc.returncode is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ── generate (fake subprocess) ──────────────────────────────────────────────


class _FakePopen:
    """Stand-in for subprocess.Popen: streams ``lines`` then writes ``out_file``
    (unless ``write`` is False) and exits with ``returncode``."""

    captured_cmd: list[str] = []
    captured_env: dict = {}

    def __init__(
        self,
        cmd,
        *,
        lines,
        returncode,
        out_file,
        write,
        env = None,
    ):
        type(self).captured_cmd = list(cmd)
        type(self).captured_env = dict(env or {})
        self.pid = 424242  # a real Popen has one, and the lifetime record needs it
        self._lines = list(lines)
        self.returncode = returncode
        self._out_file = out_file
        self._write = write

    @property
    def stdout(self):
        return iter(self._lines)

    def wait(self, timeout = None):
        if self._write:
            Path(self._out_file).write_bytes(b"\x89PNG\r\n")
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        pass


def _patch_popen(
    monkeypatch,
    *,
    lines,
    returncode,
    out_file,
    write = True,
):
    def _factory(cmd, **kw):
        return _FakePopen(
            cmd,
            lines = lines,
            returncode = returncode,
            out_file = out_file,
            write = write,
            env = kw.get("env"),
        )

    monkeypatch.setattr(eng.subprocess, "Popen", _factory)


def _engine(tmp_path):
    binary = tmp_path / "sd-cli"
    binary.write_text("x")
    return SdCppEngine(binary = str(binary))


def test_generate_success_returns_path_and_collects_logs(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    _patch_popen(
        monkeypatch, lines = ["loading model", "step 1/8", "done"], returncode = 0, out_file = out
    )
    seen: list[str] = []
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf", vae = "/m/ae.sft", llm = "/m/q.gguf")
    params = SdCppGenParams(prompt = "a cat", steps = 8, seed = 1)

    result = e.generate(files, params, output_path = str(out), on_log = seen.append)

    assert result == out and out.is_file()
    assert seen == ["loading model", "step 1/8", "done"]
    # the real argv was built and handed to Popen
    assert "--diffusion-model" in _FakePopen.captured_cmd
    assert str(out) == _FakePopen.captured_cmd[_FakePopen.captured_cmd.index("--output") + 1]
    # the subprocess env carries the binary's dir on the library path
    var = eng._lib_path_var()
    assert str(Path(e.binary).resolve().parent) in _FakePopen.captured_env.get(var, "")


def test_default_run_log_is_compact_and_omits_user_text_and_paths(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(eng, "_verbose_native_logs", lambda: False)
    caplog.set_level("INFO", logger = eng.__name__)
    e = _engine(tmp_path)
    out = tmp_path / "private-output.png"
    _patch_popen(monkeypatch, lines = ["done"], returncode = 0, out_file = out)
    prompt = "private prompt " * 40
    negative = "private negative prompt " * 20

    e.generate(
        SdCppModelFiles(diffusion_model = "/private/models/z.gguf"),
        SdCppGenParams(
            prompt = prompt,
            negative_prompt = negative,
            width = 768,
            height = 512,
            steps = 8,
            seed = 7,
        ),
        output_path = str(out),
    )

    messages = [record.getMessage() for record in caplog.records if record.name == eng.__name__]
    assert messages[0] == (
        "sd-cli run started: mode=img_gen model=z.gguf size=768x512 steps=8 seed=7"
    )
    assert messages[1].startswith(
        "sd-cli run completed: mode=img_gen model=z.gguf size=768x512 steps=8 seed=7 elapsed="
    )
    rendered = "\n".join(messages)
    assert prompt not in rendered
    assert negative not in rendered
    assert "/private/models/z.gguf" not in rendered
    assert str(out) not in rendered


def test_verbose_run_log_keeps_argv_but_redacts_prompts(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(eng, "_verbose_native_logs", lambda: True)
    caplog.set_level("INFO", logger = eng.__name__)
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    _patch_popen(monkeypatch, lines = ["done"], returncode = 0, out_file = out)

    e.generate(
        SdCppModelFiles(diffusion_model = "/models/z.gguf"),
        SdCppGenParams(prompt = "a private cat", negative_prompt = "a private dog"),
        output_path = str(out),
        extra_args = ["-p", "a second private cat", "-n=a second private dog"],
    )

    rendered = "\n".join(
        record.getMessage() for record in caplog.records if record.name == eng.__name__
    )
    assert "/models/z.gguf" in rendered
    assert str(out) in rendered
    assert "--prompt <redacted>" in rendered
    assert "--negative-prompt <redacted>" in rendered
    assert "-p <redacted>" in rendered
    assert "-n=<redacted>" in rendered
    assert "a private cat" not in rendered
    assert "a private dog" not in rendered
    assert "a second private cat" not in rendered
    assert "a second private dog" not in rendered


def test_generate_raises_on_nonzero_exit(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    _patch_popen(monkeypatch, lines = ["boom: bad gguf"], returncode = 1, out_file = out, write = False)
    with pytest.raises(RuntimeError, match = "exited 1"):
        e.generate(
            SdCppModelFiles(diffusion_model = "/m/z.gguf"),
            SdCppGenParams(prompt = "x"),
            output_path = str(out),
        )


def test_generate_raises_when_no_output_despite_success(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    _patch_popen(monkeypatch, lines = ["ok"], returncode = 0, out_file = out, write = False)
    with pytest.raises(RuntimeError, match = "no image"):
        e.generate(
            SdCppModelFiles(diffusion_model = "/m/z.gguf"),
            SdCppGenParams(prompt = "x"),
            output_path = str(out),
        )


def test_generate_does_not_return_stale_preexisting_output(tmp_path, monkeypatch):
    # A leftover file at the target path must not satisfy the post-run output check when the run produced nothing: the target is cleared first.
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    out.write_bytes(b"stale")
    _patch_popen(monkeypatch, lines = ["ok"], returncode = 0, out_file = out, write = False)
    with pytest.raises(RuntimeError, match = "no image"):
        e.generate(
            SdCppModelFiles(diffusion_model = "/m/z.gguf"),
            SdCppGenParams(prompt = "x"),
            output_path = str(out),
        )
    assert not out.exists()


def test_generate_raises_when_binary_missing():
    e = SdCppEngine(binary = None)
    with pytest.raises(RuntimeError, match = "not found"):
        e.generate(
            SdCppModelFiles(diffusion_model = "/m/z.gguf"),
            SdCppGenParams(prompt = "x"),
            output_path = "/tmp/x.png",
        )


class _HangingPopen:
    """A child that runs but never prints and never exits -- the case a plain
    `for line in stdout` would block on forever, ignoring the timeout."""

    def __init__(self, cmd, **_kw):
        self._alive = True
        self.pid = 424243

    class _Blocking:
        def __init__(self, owner):
            self.owner = owner

        def __iter__(self):
            return self

        def __next__(self):
            while self.owner._alive:
                time.sleep(0.01)
            raise StopIteration

    @property
    def stdout(self):
        return self._Blocking(self)

    def poll(self):
        return None if self._alive else -9

    def wait(self, timeout = None):
        self._alive = False
        return -9

    def kill(self):
        self._alive = False


def test_generate_times_out_on_silent_hang(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    monkeypatch.setattr(eng.subprocess, "Popen", lambda cmd, **kw: _HangingPopen(cmd, **kw))
    t0 = time.time()
    with pytest.raises(RuntimeError, match = "timed out"):
        e.generate(
            SdCppModelFiles(diffusion_model = "/m/z.gguf"),
            SdCppGenParams(prompt = "x"),
            output_path = str(tmp_path / "x.png"),
            timeout = 0.3,
        )
    # The timeout is enforced promptly (not blocked until stdout EOF).
    assert time.time() - t0 < 5.0


def test_img2img_generate_passes_init_image(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n")
    _patch_popen(monkeypatch, lines = ["img2img"], returncode = 0, out_file = out)
    e.generate(
        SdCppModelFiles(diffusion_model = "/m/z.gguf"),
        SdCppGenParams(prompt = "x", init_img = str(src), strength = 0.5),
        output_path = str(out),
    )
    assert "--init-img" in _FakePopen.captured_cmd
    assert str(src) == _FakePopen.captured_cmd[_FakePopen.captured_cmd.index("--init-img") + 1]


def test_generate_native_speed_dedupes_against_offload(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    _patch_popen(monkeypatch, lines = ["ok"], returncode = 0, out_file = out)
    # offload already adds --diffusion-fa; native_speed="default" would add it again.
    e.generate(
        SdCppModelFiles(diffusion_model = "/m/z.gguf"),
        SdCppGenParams(prompt = "x"),
        output_path = str(out),
        offload = ["--offload-to-cpu", "--diffusion-fa"],
        native_speed = "default",
    )
    # --diffusion-fa appears exactly once (de-duped), not twice.
    assert _FakePopen.captured_cmd.count("--diffusion-fa") == 1


def test_generate_native_speed_adds_flag_when_not_offloaded(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    _patch_popen(monkeypatch, lines = ["ok"], returncode = 0, out_file = out)
    e.generate(
        SdCppModelFiles(diffusion_model = "/m/z.gguf"),
        SdCppGenParams(prompt = "x"),
        output_path = str(out),
        offload = [],  # fast/resident tier: no offload, but speed flag still applies
        native_speed = "default",
    )
    assert _FakePopen.captured_cmd.count("--diffusion-fa") == 1


def test_upscale_runs_and_returns_path(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "big.png"
    _patch_popen(monkeypatch, lines = ["upscaling", "done"], returncode = 0, out_file = out)
    result = e.upscale(
        SdCppUpscaleParams(input_image = "/in/small.png", upscale_model = "/m/esrgan.pth", repeats = 2),
        output_path = str(out),
    )
    assert result == out and out.is_file()
    assert _FakePopen.captured_cmd[_FakePopen.captured_cmd.index("--mode") + 1] == "upscale"
    assert "--upscale-model" in _FakePopen.captured_cmd


def test_upscale_raises_when_binary_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(eng, "find_sd_cpp_binary", lambda: None)
    e = SdCppEngine(binary = None)
    with pytest.raises(RuntimeError, match = "not found"):
        e.upscale(
            SdCppUpscaleParams(input_image = "/i.png", upscale_model = "/m/e.pth"),
            output_path = str(tmp_path / "x.png"),
        )


# ── engine routing ──────────────────────────────────────────────────────────


def test_routing_gpu_backends_use_diffusers():
    for backend in ("cuda", "rocm", "xpu"):
        assert select_diffusion_engine(backend, native_available = True) == ENGINE_DIFFUSERS


def test_routing_cpu_and_mps_use_native_when_available():
    assert select_diffusion_engine("cpu", native_available = True) == ENGINE_SD_CPP
    assert select_diffusion_engine("mps", native_available = True) == ENGINE_SD_CPP


def test_routing_cpu_falls_back_to_diffusers_without_binary():
    assert select_diffusion_engine("cpu", native_available = False) == ENGINE_DIFFUSERS


def test_routing_prefer_native_overrides_gpu():
    assert (
        select_diffusion_engine("cuda", native_available = True, prefer_native = True) == ENGINE_SD_CPP
    )
    # but only if a binary is actually available
    assert (
        select_diffusion_engine("cuda", native_available = False, prefer_native = True)
        == ENGINE_DIFFUSERS
    )


def test_native_generation_timeout_matches_the_ui_settle_window():
    # The native engine exists for slow CPU hosts: on GPU-less CI runners a 512x512 4-step Q2_K generation took 900 s (Linux)
    # and 1465 s (Windows), so the old 30-minute default killed still-progressing jobs. The ceiling now matches SETTLE_MAX_MS.
    from core.inference.sd_cpp_engine import NATIVE_GENERATION_TIMEOUT_S, SdCppEngine
    from core.inference import sd_cpp_backend

    assert NATIVE_GENERATION_TIMEOUT_S == 6 * 60 * 60
    for fn in (SdCppEngine.generate, SdCppEngine.upscale):
        assert (
            inspect.signature(fn).parameters["timeout"].default == NATIVE_GENERATION_TIMEOUT_S
        ), fn.__name__
    # The resident-server path shares the same ceiling, applied per request (see test_server_generate_splits_batches_above_server_limit).
    assert sd_cpp_backend.NATIVE_GENERATION_TIMEOUT_S == NATIVE_GENERATION_TIMEOUT_S


# ── in-place progress redraws ───────────────────────────────────────────────
# sd-cli redraws its sampling bar with a LEADING carriage return and closes each redraw with an
# erase-to-end-of-line, emitting a newline only on the final step:
#     printf("\r%s %i/%i - %s\033[K%s", bar, step, steps, speed, step == steps ? "\n" : "")
# so a reader that keys only on newlines reports nothing until sampling is already over.

_REDRAW = "\r  |=========>          | {}/{} - 21.50s/it\x1b[K"


def test_split_progress_records_treats_erase_as_a_terminator():
    """The redraw is complete the moment sd-cli flushes it, even though its own newline never
    comes and the NEXT redraw's carriage return has not arrived yet."""
    records, rest = eng.split_progress_records(_REDRAW.format(7, 30))
    assert records == ["", "  |=========>          | 7/30 - 21.50s/it\x1b[K"]
    assert rest == ""


def test_split_progress_records_keeps_unterminated_remainder():
    records, rest = eng.split_progress_records("done\nhalf a li")
    assert records == ["done"]
    assert rest == "half a li"


def test_split_progress_records_counts_crlf_as_one_terminator():
    records, rest = eng.split_progress_records("a\r\nb\r\n")
    assert records == ["a", "b"]
    assert rest == ""


def test_strip_ansi_removes_the_erase_sequence():
    assert eng.strip_ansi("  |==>  | 7/30 - 21.50s/it\x1b[K") == "  |==>  | 7/30 - 21.50s/it"


class _ChunkStream:
    """A text stream over a pipe: ``.buffer.read1`` returns whatever the child has flushed,
    exactly like a real subprocess pipe, and iteration would block until a newline. Counts reads
    so a test can prove WHEN a record was delivered, not merely that it arrived eventually."""

    class _Raw:
        def __init__(self, chunks, owner):
            self._chunks = list(chunks)
            self._owner = owner

        def read1(self, _n):
            if not self._chunks:
                return b""
            self._owner.reads += 1
            return self._chunks.pop(0)

    def __init__(self, chunks):
        self.reads = 0
        self.buffer = self._Raw(chunks, self)

    def __iter__(self):
        raise AssertionError("iteration would block on a redraw that carries no newline")


def test_iter_records_delivers_every_redraw():
    chunks = [_REDRAW.format(i, 3).encode() for i in (1, 2)]
    chunks.append((_REDRAW.format(3, 3) + "\n").encode())
    got = [r for r in eng.iter_sd_cpp_records(_ChunkStream(chunks)) if r.strip()]
    assert got == [
        "  |=========>          | 1/3 - 21.50s/it",
        "  |=========>          | 2/3 - 21.50s/it",
        "  |=========>          | 3/3 - 21.50s/it",
    ]


def test_iter_records_delivers_a_redraw_as_soon_as_it_is_flushed():
    """The actual regression: progress was not merely late-ish, it was one redraw behind, so a
    30-step job showed 0/30 until step 2 and never showed the last step before completion.

    Delivering after ONE read is the whole claim. A redraw carries no newline, and its carriage
    return sits at the front of the NEXT redraw, so a reader terminating only on CR/LF cannot
    produce step 1 until step 2 has been flushed -- which is a second read.
    """
    stream = _ChunkStream([_REDRAW.format(i, 3).encode() for i in (1, 2, 3)])
    records = eng.iter_sd_cpp_records(stream)
    first = next(r for r in records if r.strip())
    assert first == "  |=========>          | 1/3 - 21.50s/it"
    assert stream.reads == 1


def test_iter_records_decodes_utf8_split_across_reads():
    """A multi-byte character straddling two read1() boundaries must not become mojibake."""
    blob = "café\n".encode()
    stream = _ChunkStream([blob[:4], blob[4:]])
    assert list(eng.iter_sd_cpp_records(stream)) == ["café"]


def test_iter_records_falls_back_to_line_iteration_without_a_raw_buffer():
    """Test doubles (and non-pipe streams) hand us a plain iterable with no ``.buffer``."""
    lines = ["loading\n", _REDRAW.format(4, 4) + "\n"]
    got = [r for r in eng.iter_sd_cpp_records(iter(lines)) if r.strip()]
    assert got == ["loading", "  |=========>          | 4/4 - 21.50s/it"]


def test_run_forwards_clean_redraws_to_on_log(tmp_path, monkeypatch):
    """End of the engine's own chain: a redraw reaches on_log, with no escape left in it."""
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    _patch_popen(
        monkeypatch,
        lines = [_REDRAW.format(1, 2), _REDRAW.format(2, 2) + "\n"],
        returncode = 0,
        out_file = str(out),
    )
    seen: list[str] = []
    e.generate(
        SdCppModelFiles(diffusion_model = "/m/z.gguf"),
        SdCppGenParams(prompt = "p"),
        output_path = str(out),
        on_log = seen.append,
    )
    bars = [s for s in seen if "|" in s]
    assert bars == [
        "  |=========>          | 1/2 - 21.50s/it",
        "  |=========>          | 2/2 - 21.50s/it",
    ]
    assert not any("\x1b" in s for s in seen)
