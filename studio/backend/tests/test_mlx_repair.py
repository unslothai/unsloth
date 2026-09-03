# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MLX self-heal: on Apple Silicon with MLX missing, reinstall it by name on a background
thread (off the startup critical path). No-op elsewhere, but a present stack still overturns
a chat-only verdict that contradicts it, even when the reinstall is disabled. Models on
core.training.worker's runtime backend self-heal.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.mlx_repair as mr  # noqa: E402


@pytest.fixture(autouse = True)
def _reset_attempt_guard(monkeypatch):
    monkeypatch.setattr(mr, "_attempted", False)
    # Both halves, or a worker takes _run_repair_and_redetect's "install ran" branch on a
    # latch an earlier test left set and re-detects for real against the next test.
    monkeypatch.setattr(mr, "_environment_mutated", False)
    monkeypatch.delenv(mr.DISABLE_ENV_VAR, raising = False)
    yield
    # Join inside the test's stubs: an outliving worker would run the real detect_hardware()
    # against the next test's globals.
    for thread in threading.enumerate():
        if thread.name == "mlx-autorepair":
            thread.join(timeout = 5)
            assert not thread.is_alive(), (
                "an mlx-autorepair worker outlived its test; once these stubs are "
                "restored it runs the real repair and detection against another test"
            )


def test_uv_cmd_targets_this_interpreter_with_mlx_packages(monkeypatch):
    monkeypatch.setattr(mr, "_uv_executable", lambda: "/usr/bin/uv")
    cmd = mr._uv_install_cmd("--upgrade", *mr.MLX_PACKAGES)
    assert cmd is not None
    assert cmd[:5] == ["/usr/bin/uv", "pip", "install", "--python", sys.executable]
    assert set(mr.MLX_PACKAGES) <= set(cmd)
    # mlx-vlm keeps a floor so the resolver cannot backtrack to an old one that
    # imports but breaks VLM Train/Export, and a ceiling so this unattended
    # install cannot cross a major line on its own.
    assert "mlx-vlm>=0.4.4,<0.7.0" in cmd
    # Pinned, not floored: see _MLX_INSTALL_SPECS.
    assert "mlx==0.32.1" in cmd
    assert "mlx-lm==0.31.3" in cmd
    # Look the requirement up by name rather than by prefix. Asserting on
    # startswith("mlx==") could only ever be checked on a spec that already
    # pins, so it passed vacuously the moment the pin was relaxed, which is the
    # one case worth catching.
    for name in ("mlx", "mlx-lm"):
        spec = mr._MLX_INSTALL_SPECS[name]
        assert spec.startswith("=="), f"{name} must be pinned, not floored: got {spec}"


def test_uv_executable_finds_installer_location_when_path_is_minimal(monkeypatch, tmp_path):
    uv = tmp_path / ".local" / "bin" / "uv"
    uv.parent.mkdir(parents = True)
    uv.write_text("#!/bin/sh\n", encoding = "utf-8")
    uv.chmod(0o755)
    monkeypatch.setattr(mr.shutil, "which", lambda _x: None)
    monkeypatch.setattr(mr.Path, "home", lambda: tmp_path)
    assert mr._uv_executable() == str(uv)


def test_no_uv_repair_stays_chat_only_without_pip(monkeypatch):
    monkeypatch.setattr(mr, "_uv_executable", lambda: None)
    monkeypatch.setattr(mr, "_transformers_constraint_args", lambda: ([], None))
    called = {"run": False}

    def _fake_run(*_args, **_kwargs):
        called["run"] = True
        raise AssertionError("plain pip fallback must not run")

    monkeypatch.setattr(mr.subprocess, "run", _fake_run)
    assert mr.attempt_mlx_repair() is False
    assert called["run"] is False


def test_constraint_pins_installed_transformers(monkeypatch):
    transformers = pytest.importorskip("transformers")
    args, path = mr._transformers_constraint_args()
    try:
        assert args[:1] == ["--constraint"]
        assert args[1] == path
        assert Path(path).read_text().strip() == f"transformers=={transformers.__version__}"
    finally:
        if path:
            Path(path).unlink(missing_ok = True)


def test_repair_install_pins_transformers_and_cleans_up(monkeypatch):
    pytest.importorskip("transformers")
    captured = {}
    created_paths = []
    real_args = mr._transformers_constraint_args

    def _spy_args():
        args, path = real_args()
        if path:
            created_paths.append(path)
        return args, path

    monkeypatch.setattr(mr, "_transformers_constraint_args", _spy_args)
    monkeypatch.setattr(mr, "_uv_executable", lambda: "/usr/bin/uv")

    class _Result:
        returncode = 0
        stdout = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _Result()

    monkeypatch.setattr(mr.subprocess, "run", _fake_run)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: True)

    assert mr.attempt_mlx_repair() is True
    cmd = captured["cmd"]
    # transformers is pinned via a constraint file so the mlx install cannot
    # upgrade it underneath Unsloth, and the temp constraint file is cleaned up.
    assert "--constraint" in cmd
    assert "--upgrade" in cmd
    reinstall_pairs = set(zip(cmd, cmd[1:]))
    for name in mr._MLX_PACKAGE_NAMES:
        assert ("--reinstall-package", name) in reinstall_pairs
    for pkg in mr.MLX_PACKAGES:
        assert pkg in cmd
    assert created_paths and not Path(created_paths[0]).exists()
    # The install mirrors the main installer by relaxing the transformers pin via
    # UV_OVERRIDE so a current mlx-vlm can coexist with the Unsloth Transformers pin.
    env = captured["env"]
    assert env is not None
    assert env.get("UV_OVERRIDE", "").endswith("overrides-darwin-arm64.txt")


def test_install_requires_prebuilt_wheels(monkeypatch):
    # A source distribution's PEP 517 build backend runs arbitrary code at install
    # time, before the post-install stack check. The unattended self-heal must
    # require pre-built wheels so a malicious resolver-selected sdist cannot execute
    # during ordinary Unsloth startup. mlx/mlx-metal ship wheels only and
    # mlx-lm/mlx-vlm publish py3-none-any wheels, so a healthy self-heal still works.
    pytest.importorskip("transformers")
    captured = {}

    class _Result:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(mr, "_uv_executable", lambda: "/usr/bin/uv")
    monkeypatch.setattr(
        mr.subprocess, "run", lambda cmd, **k: captured.update(cmd = cmd) or _Result()
    )
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: True)

    assert mr.attempt_mlx_repair() is True
    assert mr._ONLY_BINARY_ARG in captured["cmd"]


def test_install_env_drops_secrets_and_source_redirects(monkeypatch):
    # The unattended self-heal must not hand resolver/build code the full Unsloth
    # environment: secrets and package-source redirects are dropped, while the
    # variables uv genuinely needs are forwarded.
    monkeypatch.setenv("HF_TOKEN", "secret-hf")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-aws")
    monkeypatch.setenv("WANDB_API_KEY", "secret-wandb")
    monkeypatch.setenv("UV_FIND_LINKS", "/tmp/evil")
    monkeypatch.setenv("UV_DEFAULT_INDEX", "file:///tmp/evil-index")
    monkeypatch.setenv("UV_INDEX_URL", "https://evil.example/simple")
    monkeypatch.setenv("PIP_INDEX_URL", "https://evil.example/simple")
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/evil-cache")
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/evil-xdg-cache")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/home/studio")

    env = mr._mlx_install_env()

    # Secrets never reach a (potentially malicious) build/install hook.
    for secret in ("HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "WANDB_API_KEY"):
        assert secret not in env
    # A poisoned process env cannot repoint the install at a hostile source or
    # an attacker-staged cache (cache poisoning / symlink writes).
    for redirect in (
        "UV_FIND_LINKS",
        "UV_DEFAULT_INDEX",
        "UV_INDEX_URL",
        "PIP_INDEX_URL",
        "UV_CACHE_DIR",
        "XDG_CACHE_HOME",
    ):
        assert redirect not in env
    # What uv genuinely needs is still forwarded.
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HOME"] == "/home/studio"
    # UV_OVERRIDE is set by us (not inherited), so a poisoned one is ignored.
    assert env.get("UV_OVERRIDE", "").endswith("overrides-darwin-arm64.txt")


def test_repair_rejects_inadequate_stack(monkeypatch):
    # A successful uv run that still leaves an old/missing mlx-vlm must NOT clear
    # chat-only: attempt_mlx_repair returns False so Train/Export stay disabled.
    class _Result:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _Result())
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: False)
    assert mr.attempt_mlx_repair() is False


def test_inadequate_stack_warning_names_the_floors_not_the_install_pins(monkeypatch):
    # The gate this message reports on is mlx_stack_available(), which tests the
    # floors. Quoting the install pins instead would tell an operator running a
    # perfectly usable mlx 0.33 that they need exactly 0.32.1.
    class _Result:
        returncode = 0
        stdout = ""

    warnings = []
    # Pin both, or this test measures the host. attempt_mlx_repair returns early
    # when _uv_executable() finds nothing, long before the message under test, so
    # on a machine without uv the warning list comes back empty and the unpack
    # below fails rather than the assertion. That is what took CI red while this
    # passed locally.
    monkeypatch.setattr(mr, "_uv_executable", lambda: "/usr/bin/uv")
    monkeypatch.setattr(mr, "_transformers_constraint_args", lambda: ([], None))
    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _Result())
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: False)
    monkeypatch.setattr(mr.logger, "warning", lambda msg, *args, **kw: warnings.append(msg % args))
    assert mr.attempt_mlx_repair() is False
    (message,) = [w for w in warnings if "incomplete or too-old" in w]
    for name, floor in mr._MLX_MIN_VERSIONS.items():
        assert f"{name}>={floor}" in message
    assert "==" not in message


def test_repair_invalidates_import_caches_before_stack_check(monkeypatch):
    events = []

    class _Result:
        returncode = 0
        stdout = ""

    def _stack_available():
        events.append("check")
        assert events == ["invalidate", "check"]
        return True

    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _Result())
    monkeypatch.setattr(mr, "_uv_executable", lambda: "/usr/bin/uv")
    monkeypatch.setattr(mr, "_transformers_constraint_args", lambda: ([], None))
    monkeypatch.setattr(mr.importlib, "invalidate_caches", lambda: events.append("invalidate"))
    monkeypatch.setattr(mr, "mlx_stack_available", _stack_available)

    assert mr.attempt_mlx_repair() is True
    assert events == ["invalidate", "check"]


def test_stack_unavailable_without_mlx(monkeypatch):
    import importlib.metadata as metadata

    def _missing(_name):
        raise metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(metadata, "version", _missing)
    assert mr.mlx_stack_available() is False


def test_stack_unavailable_checks_versions_before_imports(monkeypatch):
    import importlib.metadata as metadata

    def _version(name):
        if name == "mlx":
            return "0.21.0"
        return mr._MLX_MIN_VERSIONS[name]

    def _import_module(_name):
        raise AssertionError("MLX modules must not import before versions pass")

    monkeypatch.setattr(metadata, "version", _version)
    monkeypatch.setattr(mr.importlib, "import_module", _import_module)
    assert mr.mlx_stack_available() is False


def test_stack_unavailable_when_companion_import_fails(monkeypatch):
    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "version", lambda name: mr._MLX_MIN_VERSIONS[name])

    def _import_module(name):
        if name == "mlx_vlm":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr(mr.importlib, "import_module", _import_module)
    assert mr.mlx_stack_available() is False


def test_stack_available_requires_runtime_imports_and_versions(monkeypatch):
    import importlib.metadata as metadata

    imported = []

    def _import_module(name):
        imported.append(name)
        return object()

    monkeypatch.setattr(mr.importlib, "import_module", _import_module)
    monkeypatch.setattr(metadata, "version", lambda name: mr._MLX_MIN_VERSIONS[name])

    assert mr.mlx_stack_available() is True
    assert imported == list(mr._MLX_RUNTIME_IMPORTS)


def test_no_op_off_apple_silicon(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: False)
    called = {"n": 0}
    monkeypatch.setattr(
        mr, "attempt_mlx_repair", lambda **_k: called.__setitem__("n", called["n"] + 1) or True
    )
    assert mr.start_mlx_autorepair_if_needed() is False
    assert called["n"] == 0


def test_no_op_when_mlx_stack_present(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: True)
    started = mr.start_mlx_autorepair_if_needed()
    assert started is False


def test_disable_env_skips(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: False)
    monkeypatch.setenv(mr.DISABLE_ENV_VAR, "1")
    assert mr.start_mlx_autorepair_if_needed() is False


def test_apple_silicon_missing_mlx_starts_repair_and_redetects(monkeypatch):
    import threading

    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: False)

    repaired = {"called": False}

    def _fake_repair(**_kw):
        repaired["called"] = True
        return True

    redetected = {"called": False}

    # _run_repair_and_redetect imports utils.hardware.hardware lazily; stub repair
    # and capture that re-detection is invoked on success.
    monkeypatch.setattr(mr, "attempt_mlx_repair", _fake_repair)

    import utils.hardware.hardware as hw

    monkeypatch.setattr(hw, "detect_hardware", lambda: redetected.__setitem__("called", True))

    started = mr.start_mlx_autorepair_if_needed()
    assert started is True

    # Join the daemon thread deterministically.
    for thread in threading.enumerate():
        if thread.name == "mlx-autorepair":
            thread.join(timeout = 5)

    assert repaired["called"] is True
    assert redetected["called"] is True


def test_attempts_only_once_per_process(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: False)
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda **_k: False)

    first = mr.start_mlx_autorepair_if_needed()
    second = mr.start_mlx_autorepair_if_needed()
    assert first is True
    assert second is False  # guard prevents a second concurrent attempt


def test_mlx_install_env_routes_uv_override_through_safe_path(monkeypatch):
    # uv truncates UV_OVERRIDE at the first space (issue #6503).
    seen = {}

    def _spy(path):
        seen["path"] = path
        return "/space free/marker.txt".replace(" ", "_")

    monkeypatch.setattr(mr, "uv_safe_path", _spy)
    monkeypatch.delenv("UV_OVERRIDE", raising = False)

    env = mr._mlx_install_env()

    # The override file ships in the repo, so the helper must have run.
    assert "path" in seen
    assert str(seen["path"]).endswith("overrides-darwin-arm64.txt")
    assert env["UV_OVERRIDE"] == "/space_free/marker.txt"


def _fake_venv(tmp_path: Path) -> Path:
    """A venv-shaped directory: uv accepts a root that carries pyvenv.cfg."""
    (tmp_path / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding = "utf-8")
    (tmp_path / "bin").mkdir()
    return tmp_path


def test_venv_root_is_none_outside_a_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(mr.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(mr.sys, "base_prefix", str(tmp_path))
    assert mr._venv_root() is None


def test_venv_root_requires_the_marker_file(monkeypatch, tmp_path):
    # A half-deleted tree must not be offered to uv as an install target.
    monkeypatch.setattr(mr.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(mr.sys, "base_prefix", "/usr")
    assert mr._venv_root() is None
    _fake_venv(tmp_path)
    assert mr._venv_root() == str(tmp_path)


def test_install_env_names_the_target_venv_for_uv(monkeypatch, tmp_path):
    # VIRTUAL_ENV is set from sys.prefix, never forwarded from os.environ: it names
    # the environment uv installs into, so inheriting it would let a caller
    # redirect the install.
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(mr.sys, "prefix", str(venv))
    monkeypatch.setattr(mr.sys, "base_prefix", "/usr")
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/attacker-controlled")

    env = mr._mlx_install_env()

    assert env["VIRTUAL_ENV"] == str(venv)


def test_unresolvable_venv_reports_the_unsloth_repair_command(monkeypatch, tmp_path, capsys):
    # uv's own text tells the user to run `uv venv`, which would build an
    # environment Unsloth does not manage. Point at `unsloth studio update`.
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(mr.sys, "prefix", str(venv))
    monkeypatch.setattr(mr.sys, "base_prefix", "/usr")
    monkeypatch.setattr(mr, "_uv_executable", lambda: "/usr/bin/uv")
    monkeypatch.setattr(mr, "_transformers_constraint_args", lambda: ([], None))

    class _Result:
        returncode = 2
        stdout = "error: No virtual environment or system Python installation found\n"

    monkeypatch.setattr(mr.subprocess, "run", lambda cmd, **kw: _Result())

    assert mr.attempt_mlx_repair() is False

    # structlog renders to stdout, not through the stdlib logging caplog handler.
    text = capsys.readouterr().out
    assert "unsloth studio update" in text
    assert "uv venv" not in text.split("uv said:")[0]


def test_virtual_env_is_not_advertised_as_a_dangling_symlink_recovery():
    """The docstring must not re-sell the reverted placebo.

    _mlx_install_env once claimed VIRTUAL_ENV let uv "identify the target environment
    even when bin/python no longer resolves", and named a _uv_python_target helper. Both
    were removed: an explicit --python outranks VIRTUAL_ENV, so uv reports the same
    unresolved-interpreter error either way. The claim outlived the code and then misled a
    reviewer into asking for the mechanism back, so pin it rather than trusting prose.
    """
    src = (Path(mr.__file__)).read_text(encoding = "utf-8")
    assert "_uv_python_target" not in src, (
        "_uv_python_target was deleted with the venv-root retry; a reference to it means "
        "the placebo is back or the comment is stale again"
    )
    doc = mr._mlx_install_env.__doc__ or ""
    assert "even when bin/python no longer resolves" not in doc, (
        "the docstring claims VIRTUAL_ENV recovers a dangling interpreter, which uv "
        "disproves: --python outranks it and both paths report the same error"
    )


def test_an_unresolvable_interpreter_is_diagnosed_not_retried(monkeypatch, tmp_path):
    """One uv attempt, then a diagnosis -- never a second install with a different target.

    --target and --prefix do exit 0 against a broken venv, but resolve against whatever
    ambient interpreter uv finds and write a wrong-ABI or off-sys.path install. That looks
    like a repair while leaving mlx_stack_available() False, so it must never be reached.
    """
    venv = _fake_venv(tmp_path)
    monkeypatch.setattr(mr.sys, "prefix", str(venv))
    monkeypatch.setattr(mr.sys, "base_prefix", "/usr")
    monkeypatch.setattr(mr, "_uv_executable", lambda: "/usr/bin/uv")
    monkeypatch.setattr(mr, "_transformers_constraint_args", lambda: ([], None))

    calls = []

    class _Result:
        returncode = 2
        stdout = "error: No virtual environment or system Python installation found\n"

    def _run(cmd, **kw):
        calls.append(cmd)
        return _Result()

    monkeypatch.setattr(mr.subprocess, "run", _run)

    assert mr.attempt_mlx_repair() is False
    assert len(calls) == 1, f"uv was invoked {len(calls)} times; the retry is back: {calls}"
    flat = " ".join(str(part) for part in calls[0])
    assert (
        "--target" not in flat and "--prefix" not in flat
    ), f"a corrupting install target reached the uv command line: {flat}"


# ── Overturning a verdict a first-import race left behind (issue #9120) ───────


def _published_verdict(monkeypatch, *, chat_only: bool, reason):
    """Settled means a device and a set event beside the reason (a chat-only Mac measured its
    way to CPU, not to nothing), or a success check ignoring the verdict would pass. The state
    is monkeypatched, so nothing leaks to the next test."""
    import utils.hardware.hardware as hw

    settled = threading.Event()
    settled.set()
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU if chat_only else hw.DeviceType.MLX)
    monkeypatch.setattr(hw, "CHAT_ONLY", chat_only)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", reason)
    monkeypatch.setattr(hw, "DETECTION_COMPLETE", settled)
    monkeypatch.setattr(hw, "DETECTION_GENERATION", 0)

    redetects = []

    def _redetect():
        redetects.append(1)
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = hw.DeviceType.MLX, False, None
        return hw.DEVICE

    monkeypatch.setattr(hw, "_detect_hardware_locked", _redetect)
    return redetects


def _recorded_announcements(monkeypatch):
    lines = []

    class _Recorder:
        def info(self, message, *args, **kwargs):
            lines.append(message)

        def __getattr__(self, _name):
            return lambda *a, **k: None

    monkeypatch.setattr(mr, "logger", _Recorder())
    return lines


def _join_the_repair_worker():
    for thread in threading.enumerate():
        if thread.name == "mlx-autorepair":
            thread.join(timeout = 5)
            assert not thread.is_alive()


def test_a_stack_that_measures_usable_overturns_the_verdict(monkeypatch):
    # The #9120 shape: chat-only cached from a race the warm has since finished importing.
    import utils.hardware.hardware as hw

    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: True)
    redetects = _published_verdict(monkeypatch, chat_only = True, reason = "mlx_unavailable")
    announced = _recorded_announcements(monkeypatch)

    assert mr.start_mlx_autorepair_if_needed() is False
    assert len(redetects) == 1, "the verdict the stack contradicts was left published"
    assert (hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON) == (hw.DeviceType.MLX, False, None)
    assert any("Train/Export are back" in line for line in announced)


def test_a_stack_that_is_really_unusable_keeps_its_verdict(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: False)
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda **_k: False)
    redetects = _published_verdict(monkeypatch, chat_only = True, reason = "mlx_unavailable")

    assert mr.start_mlx_autorepair_if_needed() is True
    _join_the_repair_worker()
    assert redetects == []


def test_a_settled_verdict_that_does_not_blame_mlx_is_left_alone(monkeypatch):
    # Both were measured by something this cannot re-run.
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: True)

    for chat_only, reason in ((True, "no_gpu"), (False, None)):
        redetects = _published_verdict(monkeypatch, chat_only = chat_only, reason = reason)
        assert mr.start_mlx_autorepair_if_needed() is False
        assert redetects == [], f"re-detected over a {reason!r} verdict"


def test_declining_the_reinstall_does_not_mean_keeping_a_wrong_verdict(monkeypatch):
    # The opt-out declines changing the environment; re-detecting changes nothing on disk.
    monkeypatch.setenv(mr.DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: True)
    redetects = _published_verdict(monkeypatch, chat_only = True, reason = "mlx_unavailable")

    assert mr.start_mlx_autorepair_if_needed() is False
    assert len(redetects) == 1


@pytest.mark.parametrize("usable", (True, False))
def test_the_stack_is_measured_once_before_the_decision(monkeypatch, usable):
    """Two would disagree with each other, not only with the verdict: "not usable" then
    "usable" leaves it standing with no reinstall and nothing left to revisit it."""
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda **_k: False)
    _published_verdict(monkeypatch, chat_only = True, reason = "mlx_unavailable")
    probes = []
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: probes.append(1) or usable)

    mr.start_mlx_autorepair_if_needed()
    _join_the_repair_worker()
    assert probes == [1]


def test_the_overturn_cannot_republish_into_a_stopped_lifespan(monkeypatch):
    """detect_hardware() reads the current epoch when it owns none, so an unscoped re-detect
    adopts the one shutdown moved to and publishes for a dead lifespan, which the next then
    inherits instead of measuring for itself."""
    import utils.hardware.hardware as hw

    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    _published_verdict(monkeypatch, chat_only = True, reason = "mlx_unavailable")
    settled = (hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON)

    def _measure_while_shutdown_lands():
        hw.invalidate_detection()
        return True

    monkeypatch.setattr(mr, "mlx_stack_available", _measure_while_shutdown_lands)

    assert mr.start_mlx_autorepair_if_needed() is False
    assert (
        hw.DEVICE,
        hw.CHAT_ONLY,
        hw.CHAT_ONLY_REASON,
    ) == settled, "a retired lifespan's re-detect was published"


def test_a_redetect_that_publishes_nothing_is_not_announced(monkeypatch):
    """Nothing is published either way, and #9120 was diagnosed entirely from these lines:
    one claiming a recovery that did not happen is worse than silence."""
    import utils.hardware.hardware as hw

    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    _published_verdict(monkeypatch, chat_only = True, reason = "mlx_unavailable")
    announced = _recorded_announcements(monkeypatch)

    def _measure_while_shutdown_lands():
        hw.invalidate_detection()
        return True

    monkeypatch.setattr(mr, "mlx_stack_available", _measure_while_shutdown_lands)

    assert mr.start_mlx_autorepair_if_needed() is False
    assert announced == [], f"announced an overturn that never published: {announced}"

    # Retired mid-probe instead: the pass discards its healthy answer and leaves the reason
    # cleared, which "no longer the MLX verdict" reads as a win.
    hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = None, True, "mlx_unavailable"
    hw.DETECTION_COMPLETE.set()

    def _retired_under_the_probe():
        hw.invalidate_detection()
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = hw.DeviceType.MLX, False, None
        return hw.DEVICE

    monkeypatch.setattr(hw, "_detect_hardware_locked", _retired_under_the_probe)

    assert hw.overturn_the_mlx_verdict(hw.current_detection_epoch()) is False
    assert (hw.DEVICE, hw.CHAT_ONLY) == (None, True), "the discarded pass left state behind"

    # And shutdown clears DEVICE, then the event, then the verdict, unlocked: a read between
    # the first two sees a set event beside a device already gone.
    hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = None, True, "mlx_unavailable"
    hw.DETECTION_COMPLETE.set()

    def _torn_by_a_concurrent_shutdown():
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = None, False, None
        hw.DETECTION_COMPLETE.set()
        return None

    monkeypatch.setattr(hw, "_detect_hardware_locked", _torn_by_a_concurrent_shutdown)

    assert hw.overturn_the_mlx_verdict(hw.current_detection_epoch()) is False


def test_an_opted_out_host_with_nothing_to_overturn_imports_nothing(monkeypatch):
    """Under the warm's own kill switch join_background_warm() is a no-op, so detection has not
    run and this would be the process's first MLX import, for a reinstall that is opted out."""
    monkeypatch.setenv(mr.DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    _published_verdict(monkeypatch, chat_only = True, reason = None)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: pytest.fail("imported MLX for no one"))

    assert mr.start_mlx_autorepair_if_needed() is False


def test_the_repair_worker_is_scoped_to_the_epoch_read_before_the_measurement(monkeypatch):
    """The measurement imports the MLX runtime, so shutdown can land inside it: reading the
    epoch afterwards binds the repair to the one shutdown moved to."""
    import utils.hardware.hardware as hw

    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    _published_verdict(monkeypatch, chat_only = True, reason = "mlx_unavailable")
    before = hw.current_detection_epoch()

    def _measure_while_shutdown_lands():
        hw.invalidate_detection()
        return False

    monkeypatch.setattr(mr, "mlx_stack_available", _measure_while_shutdown_lands)
    scoped_to = []
    monkeypatch.setattr(mr, "_run_repair_and_redetect", lambda epoch = None: scoped_to.append(epoch))

    assert mr.start_mlx_autorepair_if_needed() is True
    _join_the_repair_worker()
    assert scoped_to == [before]
    assert hw.current_detection_epoch() != before, "the shutdown under test never happened"
