# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for unsloth.utils.dataset_num_proc.

The module under test is stdlib-only by design, and is loaded here straight off
disk rather than via ``import unsloth``. That keeps the policy tests meaningful
on a host whose torch/unsloth_zoo pair cannot import -- these assertions are
about worker-count arithmetic, not about the model stack. The separate
``test_rl_codegen_*`` cases below tie the module back to the import path that
``unsloth/models/rl.py`` actually generates, so a rename cannot slip through.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "unsloth" / "utils" / "dataset_num_proc.py"
RL_PATH = REPO_ROOT / "unsloth" / "models" / "rl.py"

# The dotted path unsloth/models/rl.py bakes into every generated trainer.
GENERATED_IMPORT_MODULE = "unsloth.utils.dataset_num_proc"
GENERATED_IMPORT_NAME = "get_dataset_num_proc"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "unsloth_dataset_num_proc_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dnp(monkeypatch):
    module = _load_module()
    module.reset_warning_state()
    monkeypatch.delenv(module.NUM_PROC_ENV_VAR, raising = False)
    # Pin the platform. macOS is refused by policy whatever the start method
    # says, so every assertion below that expects a worker count is really an
    # assertion about a forking platform; without this they pass on the Linux
    # runner and fail on the macOS one. Tests about the platform itself set
    # their own value afterwards, which wins.
    monkeypatch.setattr(module.sys, "platform", "linux")
    return module


def _force_start_method(monkeypatch, dnp, method):
    monkeypatch.setattr(dnp, "multiprocessing_start_method", lambda: method)


# ---------- start-method veto ----------


@pytest.mark.parametrize("method", ["spawn", "forkserver", None])
def test_non_fork_start_method_disables_multiprocessing(monkeypatch, dnp, method):
    # datasets ships the map function to workers as a dill pickle. Under
    # spawn/forkserver the child must also re-import the dynamically generated
    # trainer module, which has no importable name, so workers cannot run.
    _force_start_method(monkeypatch, dnp, method)
    assert dnp.get_dataset_num_proc(8) is None
    assert dnp.get_dataset_num_proc(None) is None


def test_non_fork_start_method_warns_once(monkeypatch, dnp, capsys):
    # Regression for eeffa4c065: an explicitly supplied value used to sail
    # straight through the guard. It must be vetoed, and the veto must be visible.
    _force_start_method(monkeypatch, dnp, "spawn")
    dnp.get_dataset_num_proc(8)
    dnp.get_dataset_num_proc(8)
    out = capsys.readouterr().out
    assert out.count("uses the 'spawn' start method") == 1
    assert "dataset_num_proc = 8" in out


def test_fork_start_method_honours_explicit_value(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(6) == 6


# ---------- the 1 -> None normalisation ----------


@pytest.mark.parametrize("value", [1, 0, -4])
def test_non_positive_and_one_normalise_to_none(monkeypatch, dnp, value):
    # `1` is a trap: callers pass it meaning "serial" and datasets >= 4.0 gives
    # them a Pool(1) instead. Only None is in-process on every supported release.
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(value) is None


def test_serial_as_none_false_preserves_an_explicit_one(monkeypatch, dnp):
    """The config layer must not collapse 1 to None.

    unsloth_zoo.sft_prepare_dataset reads a config ``None`` as "auto-size me",
    so writing None back for a user who asked for 1 would inflate it to the auto
    worker count -- the opposite of what they asked for.
    """
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(1, serial_as_none = False) == 1
    # 0 and negatives are not a coherent request, but they still mean "not
    # parallel", so they must land on the config layer's serial sentinel (1) and
    # not on None, which would auto-size them back up.
    assert dnp.get_dataset_num_proc(0, serial_as_none = False) == 1
    assert dnp.get_dataset_num_proc(-4, serial_as_none = False) == 1


def test_config_layer_never_returns_none_while_forking_is_available(monkeypatch, dnp):
    """On a fork host no path may write None back to a config.

    None means "auto-size me" downstream, so any route to it -- memory clamp,
    explicit serial -- would silently re-inflate.
    """
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 64)

    # memory clamp all the way down to serial
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("m", (), {"available": 1 * 1024**3})()
    )
    assert dnp.get_dataset_num_proc(16, serial_as_none = False) == 1
    assert dnp.get_dataset_num_proc(None, serial_as_none = False) == 1


@pytest.mark.parametrize("method", ["spawn", "forkserver", None])
@pytest.mark.parametrize("desired", [None, 1, 16])
def test_config_layer_is_none_not_one_on_a_non_fork_start_method(monkeypatch, dnp, method, desired):
    """Regression: the config sentinel 1 reached unpatched TRL map() call sites.

    Only SFT gets its map site rewritten (rl_replacements.py). DPO, KTO, CPO,
    ORPO, Reward and PRM pass ``args.dataset_num_proc`` straight into
    ``Dataset.map``, and on ``datasets`` >= 4.0 a ``1`` there builds a
    ``Pool(1)`` whose spawned child re-imports the user's ``__main__`` -- the
    Windows spawn loop (#3211 / #3397) this veto exists to prevent. Before this
    module those configs carried ``None`` on spawn hosts; they must still.

    Writing None back is safe here precisely because forking is unavailable:
    every auto-sizer that reads the config vetoes on a non-fork start method
    too, so there is nothing left to inflate it.
    """
    _force_start_method(monkeypatch, dnp, method)
    assert dnp.get_dataset_num_proc(desired, serial_as_none = False) is None


def test_config_layer_env_forced_serial_is_none_on_a_non_fork_start_method(monkeypatch, dnp):
    """UNSLOTH_DATASET_NUM_PROC=0 must not build a Pool(1) either."""
    _force_start_method(monkeypatch, dnp, "spawn")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "0")
    assert dnp.get_dataset_num_proc(None, serial_as_none = False) is None


def test_layering_config_then_map_site_is_correct(monkeypatch, dnp):
    """Composing the two layers must land on the right value for each intent."""
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 32)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 256 * 1024**3})(),
    )
    cfg = lambda v: dnp.get_dataset_num_proc(v, serial_as_none = False)  # noqa: E731
    site = dnp.get_dataset_num_proc

    # user asked for serial -> stays serial, never auto-inflated
    assert site(cfg(1)) is None
    # user asked for a specific count -> honoured end to end
    assert site(cfg(6)) == 6
    # user asked for nothing -> capped auto, and re-applying is idempotent
    assert cfg(None) == dnp.AUTO_NUM_PROC_CAP
    assert site(cfg(None)) == dnp.AUTO_NUM_PROC_CAP


def test_low_memory_auto_path_returns_none_not_one(monkeypatch, dnp):
    # The old heuristic returned 1 here, which still forked a Pool(1).
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 32)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 1 * 1024**3})(),
    )
    assert dnp.get_dataset_num_proc(None) is None


# ---------- auto sizing ----------


def test_auto_value_is_capped(monkeypatch, dnp):
    # Was min(max(cpu_count + 4, 2), 64) -- up to 64 forked workers, each handed
    # its own dill-pickled tokenizer closure.
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 128)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 512 * 1024**3})(),
    )
    assert dnp.get_dataset_num_proc(None) == dnp.AUTO_NUM_PROC_CAP
    assert dnp.AUTO_NUM_PROC_CAP < 64


def test_auto_value_clamped_by_available_memory(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 64)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 10 * 1024**3})(),
    )
    # 10 GB free, half of it budgeted, ~1 GB per worker -> 5.
    assert dnp.get_dataset_num_proc(None) == 5


def test_explicit_value_is_clamped_by_memory(monkeypatch, dnp, capsys):
    """The gap that actually caused issue #2693.

    Studio passes an explicit ``max(1, cpu_count // 4)``, which on a big-core
    machine is dozens of workers at ~680 MB each. The old heuristic capped only
    the auto path, so an explicit request sailed through no matter how little
    RAM there was.
    """
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 16 * 1024**3})(),
    )
    assert dnp.get_dataset_num_proc(48) == 8
    assert "reducing dataset_num_proc 48 -> 8" in capsys.readouterr().out


def test_explicit_value_is_not_capped_by_the_auto_cap(monkeypatch, dnp):
    # AUTO_NUM_PROC_CAP bounds auto-sizing only. A user who asks for 32 and has
    # the memory for it gets 32.
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 512 * 1024**3})(),
    )
    assert dnp.get_dataset_num_proc(32) == 32
    assert 32 > dnp.AUTO_NUM_PROC_CAP


def test_memory_clamp_is_skipped_without_psutil(monkeypatch, dnp):
    # No psutil means no memory reading. Honour the request rather than
    # silently serialising on a machine that may be perfectly capable.
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: None)
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(32) == 32


def test_bool_is_not_treated_as_an_int(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 8)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 64 * 1024**3})(),
    )
    assert dnp.get_dataset_num_proc(True) == 4


# ---------- environment escape hatch ----------


def test_env_override_beats_start_method_veto(monkeypatch, dnp):
    # A user who knows their workload is fork-safe is never silently downgraded.
    _force_start_method(monkeypatch, dnp, "spawn")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "24")
    assert dnp.get_dataset_num_proc(None) == 24


@pytest.mark.parametrize("raw", ["0", "none", "None", "false", ""])
def test_env_override_can_force_in_process(monkeypatch, dnp, raw):
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, raw)
    assert dnp.get_dataset_num_proc(16) is None


@pytest.mark.parametrize("raw", ["0", "none", "None", "false", "", "1"])
def test_env_override_in_process_is_encoded_for_the_config_layer(monkeypatch, dnp, raw):
    # Regression: the env override used to return before _serial(), so asking
    # for in-process tokenization wrote None into the *config*, which
    # unsloth_zoo.sft_prepare_dataset reads as "auto-size me" and inflates back
    # to its own uncapped cpu_count + 4 -> 64. The escape hatch the dead-worker
    # message recommends would have raised the worker count instead of removing
    # it. At the config layer serial is 1, never None.
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, raw)
    assert dnp.get_dataset_num_proc(16, serial_as_none = False) == 1


def test_env_override_is_uncapped(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "100")
    assert dnp.get_dataset_num_proc(None) == 100


def test_invalid_env_override_is_ignored_with_a_warning(monkeypatch, dnp, capsys):
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "banana")
    assert dnp.get_dataset_num_proc(4) == 4
    assert "is not an integer" in capsys.readouterr().out


# ---------- start-method probing must not mutate global state ----------


def test_start_method_probe_prefers_multiprocess_and_has_no_side_effects(dnp):
    """datasets does `from multiprocess import Pool`, so `multiprocess` -- not
    stdlib multiprocessing -- decides how map() spawns. Reading it must also not
    pin the context, which would make a later set_start_method() raise."""
    multiprocess = pytest.importorskip("multiprocess")
    import multiprocessing

    before_mp = multiprocess.get_start_method(allow_none = True)
    before_std = multiprocessing.get_start_method(allow_none = True)

    method = dnp.multiprocessing_start_method()

    # Must name a method this host actually offers. Windows offers only spawn,
    # and the private default-context chain the probe reads has been seen to
    # answer "fork" there; taking that at face value would read Windows as
    # forkable and let workers through.
    assert method in multiprocess.get_all_start_methods()
    assert multiprocess.get_start_method(allow_none = True) == before_mp
    assert multiprocessing.get_start_method(allow_none = True) == before_std


def test_start_method_probe_reports_an_explicit_setting(monkeypatch, dnp):
    import sys as _sys
    import types

    fake = types.ModuleType("multiprocess")
    fake.get_start_method = lambda allow_none = False: "forkserver"
    fake.get_all_start_methods = lambda: ["fork", "spawn", "forkserver"]
    monkeypatch.setitem(_sys.modules, "multiprocess", fake)
    assert dnp.multiprocessing_start_method() == "forkserver"


def _fake_multiprocess(listed, default_name):
    """A multiprocess stand-in with nothing pinned yet."""
    import types

    fake = types.ModuleType("multiprocess")
    fake.get_start_method = lambda allow_none = False: None
    fake.get_all_start_methods = lambda: list(listed)
    if default_name is not None:
        context = types.ModuleType("multiprocess.context")
        context._default_context = types.SimpleNamespace(
            _default_context = types.SimpleNamespace(_name = default_name),
            _actual_context = None,
        )
        fake.context = context
    return fake


def test_start_method_probe_prefers_the_real_default_over_list_order(monkeypatch, dnp):
    """macOS: multiprocess lists 'spawn' first but its default context is fork.

    ``multiprocess`` copies the stdlib ``get_all_start_methods()`` verbatim,
    darwin branch included, but keeps ``fork`` as its ``_default_context`` on
    every POSIX platform (``#FIXME: spawn`` in multiprocess/context.py). Since
    ``datasets`` builds its pool from ``multiprocess``, trusting the list would
    tell us 'spawn' on macOS while ``Dataset.map`` actually forks -- vetoing
    every worker and printing the wrong start method in the dead-worker
    diagnostics that exist to stop exactly that kind of misreport.
    """
    import sys as _sys

    darwin_order = ["spawn", "fork", "forkserver"]
    fake = _fake_multiprocess(darwin_order, "fork")
    monkeypatch.setitem(_sys.modules, "multiprocess", fake)
    assert dnp.multiprocessing_start_method() == "fork"


def test_macos_stays_in_process_even_though_multiprocess_forks(monkeypatch, dnp, capsys):
    """The probe reports fork on macOS; policy still refuses to use it.

    These are deliberately two separate things. ``multiprocess`` really does
    fork on darwin, so the diagnostics must say so, but CPython moved the macOS
    stdlib default to spawn in 3.8 (bpo-33725) on the grounds that forking there
    "can lead to crashes of the subprocess as macOS system libraries may start
    threads" -- and this parent has already loaded Torch and a threaded BLAS.
    Fixing the probe without this guard would have taken macOS from
    always-serial to up to AUTO_NUM_PROC_CAP forked workers.
    """
    _force_start_method(monkeypatch, dnp, "fork")
    # Pin memory so the contrast below is about the platform policy and not
    # about how much RAM the runner happens to have free.
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 1000)
    monkeypatch.setattr(dnp.sys, "platform", "darwin")

    # Serial at a map() call site, and None -- not 1 -- at the config layer, so
    # no Pool is built on datasets >= 4.1 either way.
    assert dnp.get_dataset_num_proc(8) is None
    assert dnp.get_dataset_num_proc(8, serial_as_none = False) is None
    assert dnp.get_dataset_num_proc(None) is None
    assert "macOS" in capsys.readouterr().out

    # The escape hatch still overrides it, and Linux is unaffected.
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "4")
    assert dnp.get_dataset_num_proc(8) == 4
    monkeypatch.delenv(dnp.NUM_PROC_ENV_VAR)
    monkeypatch.setattr(dnp.sys, "platform", "linux")
    assert dnp.get_dataset_num_proc(8) == 8


def test_start_method_probe_falls_back_to_list_order(monkeypatch, dnp):
    """The default context is private, so an unreadable one must not raise."""
    import sys as _sys

    fake = _fake_multiprocess(["spawn", "fork"], None)
    monkeypatch.setitem(_sys.modules, "multiprocess", fake)
    assert dnp.multiprocessing_start_method() == "spawn"


def test_start_method_probe_matches_the_pool_multiprocess_would_build(dnp):
    """The probe must agree with multiprocess's own default on this host."""
    multiprocess = pytest.importorskip("multiprocess")
    if multiprocess.get_start_method(allow_none = True) is not None:
        pytest.skip("a start method is already pinned in this process")
    assert (
        dnp.multiprocessing_start_method()
        == multiprocess.context._default_context._default_context._name
    )


# ---------- the generated trainer's import must match the real module ----------


def _rl_num_proc_snippet():
    tree = ast.parse(RL_PATH.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "num_proc_check":
            return ast.literal_eval(node.value)
    raise AssertionError("num_proc_check literal not found in unsloth/models/rl.py")


def test_rl_codegen_writes_back_without_collapsing_serial():
    # The config layer must pass serial_as_none = False; see the layering test.
    snippet = _rl_num_proc_snippet()
    assert "serial_as_none = False" in snippet


def test_zoo_sft_prepare_dataset_anchor_has_not_drifted():
    """unsloth/models/rl_replacements.py rewrites unsloth_zoo's
    sft_prepare_dataset by exact string match. A Zoo release that touches those
    lines makes _require_replace raise at import time, so catch drift here."""
    # Locate the file without importing unsloth_zoo: find_spec resolves the
    # package's path without executing its __init__, so this canary still runs
    # on a host whose torch/unsloth_zoo pair cannot import.
    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("unsloth_zoo not installed")
    zoo_file = Path(list(spec.submodule_search_locations)[0]) / "dataset_utils.py"
    if not zoo_file.is_file():
        pytest.skip("unsloth_zoo.dataset_utils not found")
    source = zoo_file.read_text(encoding = "utf-8")

    rr_tree = ast.parse(RL_PATH.with_name("rl_replacements.py").read_text(encoding = "utf-8"))

    def _anchor_and_count(where):
        found = [
            (
                ast.literal_eval(node.args[1]),
                next(
                    (ast.literal_eval(k.value) for k in node.keywords if k.arg == "count"),
                    1,
                ),
            )
            for node in ast.walk(rr_tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "_require_replace"
            and any(k.arg == "where" and ast.literal_eval(k.value) == where for k in node.keywords)
        ]
        assert len(found) == 1, f"expected exactly one _require_replace for {where!r}"
        return found[0]

    # Both edits this module owns. _require_replace raises when an anchor is
    # missing entirely, but it cannot notice that a count = 2 anchor dropped to
    # one occurrence, so assert the counts here.
    for where in (
        "sft_prepare_dataset dataset_num_proc selection",
        "sft_prepare_dataset tokenizing map() calls",
    ):
        anchor, count = _anchor_and_count(where)
        assert source.count(anchor) == count, (
            f"unsloth_zoo.dataset_utils has {source.count(anchor)} occurrences of "
            f"the {where!r} anchor, expected {count}; update rl_replacements.py"
        )


def test_rl_codegen_imports_the_module_that_exists():
    # The snippet is spliced into generated source as text, so a rename here is
    # only caught at trainer-construction time in production. Catch it now.
    snippet = _rl_num_proc_snippet()
    assert f"from {GENERATED_IMPORT_MODULE} import {GENERATED_IMPORT_NAME}" in snippet
    assert MODULE_PATH.is_file()
    module = _load_module()
    assert callable(getattr(module, GENERATED_IMPORT_NAME))


def test_rl_codegen_snippet_is_valid_python_at_method_indent():
    # rl.py re-indents extra_args to 8 spaces and drops it into __init__.
    snippet = _rl_num_proc_snippet()
    body = "\n".join(" " * 8 + line for line in snippet.split("\n"))
    source = (
        "class C:\n    def __init__(self, dataset_num_proc = None):\n" + body + "\n        pass\n"
    )
    ast.parse(source)


def test_rl_codegen_snippet_survives_an_unimportable_helper():
    # A generated file can outlive an unsloth downgrade. Constructing a config
    # must still work; it just leaves the caller's value alone.
    snippet = _rl_num_proc_snippet()
    namespace = {"dataset_num_proc": 7}
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("unsloth"):
            raise ImportError("simulated downgrade")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _blocked
    try:
        exec(snippet, namespace)
    finally:
        builtins.__import__ = real_import
    assert namespace["dataset_num_proc"] == 7


# ---------- worker-death diagnostics ----------


_DATASETS_MESSAGE = (
    "One of the subprocesses has abruptly died during map operation."
    "To debug the error, disable multiprocessing."
)


def test_worker_death_is_reraised_with_context(dnp):
    # datasets discards the child's exit status, so the original message cannot
    # distinguish an OOM kill from anything else. Supply what it dropped.
    with pytest.raises(RuntimeError) as caught:
        with dnp.map_failure_diagnostics(8):
            raise RuntimeError(_DATASETS_MESSAGE)

    message = str(caught.value)
    assert "dataset_num_proc = 8" in message
    assert "8 workers" in message
    assert "8GB" in message, "should estimate what those workers cost"
    assert dnp.NUM_PROC_ENV_VAR in message, "must name the escape hatch"
    assert "out-of-memory" in message
    # The child's traceback must survive for anyone who wants it.
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert _DATASETS_MESSAGE in str(caught.value.__cause__)


def test_worker_death_diagnostics_handles_in_process_runs(dnp):
    # num_proc=None still reaches the wrapper; it must not divide by a None.
    with pytest.raises(RuntimeError) as caught:
        with dnp.map_failure_diagnostics(None):
            raise RuntimeError(_DATASETS_MESSAGE)
    assert "dataset_num_proc = None" in str(caught.value)
    assert "1 worker," in str(caught.value)


def test_unrelated_errors_pass_through_untouched(dnp):
    # Only the dead-worker message is rewritten; nothing else is swallowed or
    # reworded, and non-RuntimeError types are not caught at all.
    original = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError) as caught:
        with dnp.map_failure_diagnostics(4):
            raise original
    assert caught.value is original

    key = KeyError("text")
    with pytest.raises(KeyError) as caught_key:
        with dnp.map_failure_diagnostics(4):
            raise key
    assert caught_key.value is key


def test_successful_map_is_not_disturbed(dnp):
    with dnp.map_failure_diagnostics(4):
        result = "tokenized"
    assert result == "tokenized"


def test_studio_num_proc_cap_has_not_drifted(dnp):
    """Studio duplicates AUTO_NUM_PROC_CAP; the two must stay equal.

    hardware.py cannot import this module -- that would pull in unsloth's whole
    __init__ during hardware detection -- so it carries its own copy, the same
    arrangement ZOO_MIN_ROWS_FOR_MULTIPROC uses in the other direction. Read it
    out of the source rather than importing studio, which needs a backend
    environment this suite does not have.
    """
    hardware = REPO_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
    if not hardware.is_file():
        pytest.skip("studio backend not present")

    tree = ast.parse(hardware.read_text(encoding = "utf-8"))
    found = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "_STUDIO_NUM_PROC_CAP" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert len(found) == 1, "expected exactly one _STUDIO_NUM_PROC_CAP assignment"
    assert found[0] == dnp.AUTO_NUM_PROC_CAP, (
        f"studio caps dataset workers at {found[0]} while this module caps the "
        f"auto path at {dnp.AUTO_NUM_PROC_CAP}; they must agree"
    )


def test_studio_bounds_its_own_computed_worker_count(dnp):
    """A backend heuristic must not outrank the cap.

    trainer.py asks for cpu_count // 4 and safe_num_proc's auto path is
    cpu_count // 3, so a 64-core host produced 16 workers and a 192-core one 48.
    Those are explicit ints by the time this module sees them, so it treated
    them as deliberate and only clamped them by free memory -- meaning a large
    machine with RAM to spare kept all 48. Nothing about that is user intent.
    """
    hardware = REPO_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"
    if not hardware.is_file():
        pytest.skip("studio backend not present")
    source = hardware.read_text(encoding = "utf-8")

    tree = ast.parse(source)
    safe = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "safe_num_proc"),
        None,
    )
    assert safe is not None, "safe_num_proc is where every studio map() count is decided"
    body = ast.dump(safe)
    assert "_STUDIO_NUM_PROC_CAP" in body, (
        "safe_num_proc no longer bounds its result; studio would send "
        "cpu_count // 3 workers straight to Dataset.map"
    )


def test_the_recovery_advice_does_not_promise_more_than_it_delivers(dnp):
    """UNSLOTH_DATASET_NUM_PROC=0 is not in-process on every path.

    The dead-worker message is the one place a user is told what to do next, so
    it has to be true. On fork, train_on_responses_only with a split at or over
    the Zoo's threshold gets ``1`` rather than ``None`` -- deliberately, since a
    bare None there is read as "size it for me" and would inflate to the Zoo's
    uncapped count -- and ``datasets`` >= 4.1 turns that into a Pool(1). So the
    advice is in-process almost everywhere and one worker in that one case, and
    saying "tokenize in-process" flatly was wrong for exactly the large-dataset
    runs that die in the first place.
    """
    with pytest.raises(RuntimeError) as excinfo:
        with dnp.map_failure_diagnostics(8):
            raise RuntimeError("One of the subprocesses has abruptly died during map operation.")
    message = str(excinfo.value)
    assert f"{dnp.NUM_PROC_ENV_VAR}=0" in message
    assert "one worker" in message, "the exception to in-process has to be stated"
    assert "train_on_responses_only" in message, "and which path it applies to"
    assert f"{dnp.ZOO_MIN_ROWS_FOR_MULTIPROC:,}" in message, "and above which size"


def test_the_advice_matches_what_the_resolver_actually_returns(dnp, monkeypatch):
    """Executed, not just read: the claim above is checked against the code.

    A message asserting a behaviour the resolver does not have would be worse
    than the vague one it replaced, so drive both branches and compare.
    """

    class _Split:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    class _Trainer:
        def __init__(self, split):
            self.train_dataset = split
            self.eval_dataset = None

    monkeypatch.setattr(dnp, "multiprocessing_start_method", lambda: "fork")
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 1000)
    monkeypatch.setattr(dnp.sys, "platform", "linux")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "0")

    over = dnp.resolve_responses_only_num_proc(
        _Trainer(_Split(dnp.ZOO_MIN_ROWS_FOR_MULTIPROC + 1)), None
    )
    under = dnp.resolve_responses_only_num_proc(
        _Trainer(_Split(dnp.ZOO_MIN_ROWS_FOR_MULTIPROC - 1)), None
    )
    assert over == 1, "over the threshold the best expressible request is one worker"
    assert under is None, "under it the Zoo's own guard already goes in-process"


def test_probe_rejects_a_start_method_the_host_does_not_offer(monkeypatch, dnp):
    """The private default-context chain is not trustworthy on its own.

    On a Windows runner it answered "fork" while get_all_start_methods() was
    ["spawn"]. A start method the platform does not offer cannot be the one in
    use, and believing it read Windows as forkable: _workers_unusable_reason()
    returned None and workers were allowed through, which is the spawn
    re-import loop of #3211 / #3397 that this module exists to prevent.
    """
    import sys as _sys
    import types

    fake = types.ModuleType("multiprocess")
    fake.get_start_method = lambda allow_none = False: None
    fake.get_all_start_methods = lambda: ["spawn"]
    context = types.ModuleType("multiprocess.context")
    context._default_context = types.SimpleNamespace(
        _default_context = types.SimpleNamespace(_name = "fork"),
    )
    fake.context = context
    monkeypatch.setitem(_sys.modules, "multiprocess", fake)

    assert dnp.multiprocessing_start_method() == "spawn"

    monkeypatch.setattr(dnp.sys, "platform", "win32")
    assert dnp.get_dataset_num_proc(8) is None
    assert dnp.get_dataset_num_proc(8, serial_as_none = False) is None
