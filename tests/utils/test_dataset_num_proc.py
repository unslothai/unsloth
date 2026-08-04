# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
    assert out.count("needs the 'fork' start method") == 1
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


def test_config_layer_never_returns_none(monkeypatch, dnp):
    """No path may write None back to a config, whatever the reason for it.

    None means "auto-size me" downstream, so any route to it -- start-method
    veto, memory clamp, explicit serial -- would silently re-inflate.
    """
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 64)

    # start-method veto
    _force_start_method(monkeypatch, dnp, "spawn")
    assert dnp.get_dataset_num_proc(16, serial_as_none = False) == 1

    # memory clamp all the way down to serial
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("m", (), {"available": 1 * 1024 ** 3})()
    )
    assert dnp.get_dataset_num_proc(16, serial_as_none = False) == 1
    assert dnp.get_dataset_num_proc(None, serial_as_none = False) == 1


def test_layering_config_then_map_site_is_correct(monkeypatch, dnp):
    """Composing the two layers must land on the right value for each intent."""
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 32)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 256 * 1024 ** 3})(),
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
        lambda: type("m", (), {"available": 1 * 1024 ** 3})(),
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
        lambda: type("m", (), {"available": 512 * 1024 ** 3})(),
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
        lambda: type("m", (), {"available": 10 * 1024 ** 3})(),
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
        lambda: type("m", (), {"available": 16 * 1024 ** 3})(),
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
        lambda: type("m", (), {"available": 512 * 1024 ** 3})(),
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
        lambda: type("m", (), {"available": 64 * 1024 ** 3})(),
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


# ---------- the generated trainer's import must match the real module ----------


def _rl_num_proc_snippet():
    tree = ast.parse(RL_PATH.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "num_proc_check"
        ):
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
                    (
                        ast.literal_eval(k.value)
                        for k in node.keywords
                        if k.arg == "count"
                    ),
                    1,
                ),
            )
            for node in ast.walk(rr_tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "_require_replace"
            and any(
                k.arg == "where" and ast.literal_eval(k.value) == where
                for k in node.keywords
            )
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
    source = "class C:\n    def __init__(self, dataset_num_proc = None):\n" + body + "\n        pass\n"
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
