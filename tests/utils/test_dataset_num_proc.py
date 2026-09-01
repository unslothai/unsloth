# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for unsloth.dataset_num_proc, the fallback copy of the policy.

unsloth_zoo.dataset_num_proc owns it; this copy only runs on a zoo that predates
the module, and ``test_the_two_copies_have_not_drifted`` holds them together.

The module is stdlib-only by design and is loaded straight off disk rather than
via ``import unsloth``, so these assertions stay meaningful on a host whose
torch/unsloth_zoo pair cannot import. The ``test_rl_codegen_*`` cases tie it back
to the import path ``unsloth/models/rl.py`` generates, catching a rename.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import textwrap
import types
from pathlib import Path

import pytest

try:
    import multiprocess  # noqa: F401
except ImportError:
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "unsloth" / "dataset_num_proc.py"
RL_PATH = REPO_ROOT / "unsloth" / "models" / "rl.py"

# The dotted paths unsloth/models/rl.py bakes into every generated trainer:
GENERATED_IMPORT_MODULE = "unsloth_zoo.dataset_num_proc"
GENERATED_FALLBACK_MODULE = "unsloth.dataset_num_proc"
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
    # Pin the platform: macOS is refused by policy whatever the start method says, so every assertion expecting a
    monkeypatch.setattr(module.sys, "platform", "linux")
    try:
        import psutil
        monkeypatch.setattr(
            psutil, "virtual_memory", lambda: type("m", (), {"available": 1024 * 1024**3})()
        )
    except ImportError:
        pass
    # Point this module's cgroup reader at a path that does not exist rather than stubbing the reader itself, so the
    monkeypatch.setattr(module, "CGROUP_ROOT", "/nonexistent-cgroup-root-for-tests")
    # Pin the memory ceiling too, at its two sources rather than at the reader, so the memory tests can still patch
    # either and win.
    # Every count this module returns is clamped by free RAM and by the cgroup budget, so on a memory-limited runner a
    # test about the start method or about an explicit value silently becomes a test of the clamp instead:
    # `get_dataset_num_proc(6) == 6` returns 4 in a small container.
    # unsloth_zoo's readers are neutralised by name instead, and without requiring the name to be there: pinning
    # hf_xet_tuning.CGROUP_ROOT alone only works on a zoo that has that global.
    try:
        from unsloth_zoo import hf_xet_tuning
    except Exception:
        # Importing the package can fail long after it has imported this submodule (__init__ pulls in hf_xet_tuning near
        # the top and only raises "Please install Unsloth" at the end), and the failure removes unsloth_zoo from
        # sys.modules while leaving unsloth_zoo.hf_xet_tuning behind.
        hf_xet_tuning = sys.modules.get("unsloth_zoo.hf_xet_tuning")
    if hf_xet_tuning is not None:
        for name, neutral in (
            ("CGROUP_ROOT", Path("/nonexistent-cgroup-root-for-tests")),
            ("_cgroup_v2_dirs", lambda: []),
            ("_cgroup_v1_dirs", lambda controller: []),
            ("cgroup_memory_limit", lambda: None),
            ("cgroup_cpu_limit", lambda: None),
        ):
            monkeypatch.setattr(hf_xet_tuning, name, neutral, raising = False)
    return module


def _force_start_method(monkeypatch, dnp, method):
    monkeypatch.setattr(dnp, "multiprocessing_start_method", lambda: method)


def _force_cpus(monkeypatch, dnp, count):
    """Pin the CPU count the auto path sizes from.

    Patching psutil alone is not enough: the count is the smallest of the host's
    CPUs, this process's affinity mask and any cgroup quota, so on a 4-vCPU runner
    a "128 CPU host" would still come out as 4.
    """
    monkeypatch.setattr(dnp, "_usable_cpus", lambda: count)




@pytest.mark.parametrize("method", ["spawn", "forkserver", None])
def test_non_fork_start_method_disables_multiprocessing(monkeypatch, dnp, method):
    # trainer module, which has no importable name, so workers cannot run.
    # Under spawn/forkserver the child must re-import the dynamically generated trainer module, which has no importable
    _force_start_method(monkeypatch, dnp, method)
    assert dnp.get_dataset_num_proc(8) is None
    assert dnp.get_dataset_num_proc(None) is None


def test_non_fork_start_method_warns_once(monkeypatch, dnp, capsys):
    # Regression for eeffa4c065:
    _force_start_method(monkeypatch, dnp, "spawn")
    dnp.get_dataset_num_proc(8)
    dnp.get_dataset_num_proc(8)
    out = capsys.readouterr().out
    assert out.count("uses the 'spawn' start method") == 1
    assert "dataset_num_proc = 8" in out


def test_fork_start_method_honours_explicit_value(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(6) == 6




@pytest.mark.parametrize("value", [1, 0, -4])
def test_non_positive_and_one_normalise_to_none(monkeypatch, dnp, value):
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(value) is None


def test_serial_as_none_false_preserves_an_explicit_one(monkeypatch, dnp):
    """The config layer must not collapse 1 to None.

    unsloth_zoo.sft_prepare_dataset reads a config ``None`` as "auto-size me",
    so writing None back for a user who asked for 1 would inflate it.
    """
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(1, serial_as_none = False) == 1
    # 0 and negatives are incoherent requests but still mean "not parallel", so they land on the config serial sentinel
    assert dnp.get_dataset_num_proc(0, serial_as_none = False) == 1
    assert dnp.get_dataset_num_proc(-4, serial_as_none = False) == 1


def test_config_layer_never_returns_none_while_forking_is_available(monkeypatch, dnp):
    """On a fork host no path may write None back to a config.

    None means "auto-size me" downstream, so any route to it -- memory clamp,
    explicit serial -- would re-inflate.
    """
    psutil = pytest.importorskip("psutil")
    _force_cpus(monkeypatch, dnp, 64)

    # `1` is a trap: callers mean "serial", datasets >= 4.0 gives a Pool(1).
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

    Only SFT gets its map site rewritten (rl_replacements.py); DPO, KTO, CPO,
    ORPO, Reward and PRM pass ``args.dataset_num_proc`` straight into
    ``Dataset.map``, where a ``1`` builds a ``Pool(1)`` whose spawned child
    re-imports the user's ``__main__`` (#3211 / #3397). None is safe here
    precisely because forking is unavailable, so every auto-sizer vetoes too.
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
    _force_cpus(monkeypatch, dnp, 32)
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
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    _force_cpus(monkeypatch, dnp, 32)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 1 * 1024**3})(),
    )
    assert dnp.get_dataset_num_proc(None) is None




def test_auto_value_is_capped(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    _force_cpus(monkeypatch, dnp, 128)
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
    _force_cpus(monkeypatch, dnp, 64)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 10 * 1024**3})(),
    )
    # 10 GB free, half of it budgeted, ~1 GB per worker -> 5.
    assert dnp.get_dataset_num_proc(None) == 5


def test_explicit_value_is_clamped_by_memory(monkeypatch, dnp, capsys):
    """The gap that caused issue #2693.

    Unsloth passes an explicit ``max(1, cpu_count // 4)``, dozens of workers at
    ~680 MB each on a big-core machine, and the old heuristic bounded only the
    auto path, so that sailed through however little RAM there was.
    """
    # The old heuristic returned 1 here, which still forked a Pool(1).
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
    # Was min(max(cpu_count + 4, 2), 64) -- up to 64 forked workers.
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
    # No psutil means no memory reading, so honour the request.
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: None)
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(32) == 32


def test_bool_is_not_treated_as_an_int(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    psutil = pytest.importorskip("psutil")
    _force_cpus(monkeypatch, dnp, 8)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: type("m", (), {"available": 64 * 1024**3})(),
    )
    assert dnp.get_dataset_num_proc(True) == 4




def test_env_override_beats_start_method_veto(monkeypatch, dnp):
    # A user who knows their workload is fork-safe is never downgraded.
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
    # Regression:
    # AUTO_NUM_PROC_CAP bounds auto-sizing only.
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, raw)
    assert dnp.get_dataset_num_proc(16, serial_as_none = False) == 1


def test_env_override_is_uncapped(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "100")
    assert dnp.get_dataset_num_proc(None) == 100
    # Above the auto cap is the easy half.
    # The memory clamp is the one that matters: the fixture leaves room for 512 workers, so without pinning it this
    # asserts nothing about the exemption.
    # map_failure_diagnostics points users at this hatch on exactly the host where the clamp would bite.
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 2)
    assert dnp.get_dataset_num_proc(None) == 100
    assert dnp.get_dataset_num_proc(4) == 100


def test_invalid_env_override_is_ignored_with_a_warning(monkeypatch, dnp, capsys):
    # Regression: the env override used to return before _serial(), writing None into the *config*, which
    # unsloth_zoo.sft_prepare_dataset reads as "auto-size me"
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

    # Must name a method this host offers.
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

    It copies stdlib ``get_all_start_methods()`` verbatim, darwin branch
    included, while keeping ``fork`` as its ``_default_context`` on every POSIX
    platform (``#FIXME: spawn`` in multiprocess/context.py). Since ``datasets``
    pools come from ``multiprocess``, trusting the list would veto every worker
    and misreport the start method in the dead-worker diagnostics.
    """
    import sys as _sys

    darwin_order = ["spawn", "fork", "forkserver"]
    fake = _fake_multiprocess(darwin_order, "fork")
    monkeypatch.setitem(_sys.modules, "multiprocess", fake)
    assert dnp.multiprocessing_start_method() == "fork"


def test_macos_stays_in_process_even_though_multiprocess_forks(monkeypatch, dnp, capsys):
    """The probe reports fork on macOS; policy still refuses to use it.

    ``multiprocess`` really does fork on darwin, so the diagnostics must say so,
    but CPython moved the macOS default to spawn in 3.8 (bpo-33725) because
    forking there "can lead to crashes of the subprocess as macOS system
    libraries may start threads" -- and this parent holds Torch and a threaded
    BLAS. Fixing the probe without the guard would take macOS from always-serial
    to AUTO_NUM_PROC_CAP forked workers.
    """
    _force_start_method(monkeypatch, dnp, "fork")
    # Pin memory so the contrast is about policy, not the runner's free RAM.
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 1000)
    monkeypatch.setattr(dnp.sys, "platform", "darwin")

    # Serial at a map() call site, None -- not 1 -- at the config layer, so no Pool is built on datasets >= 4.1 either
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




def _rl_serial_as_none(tree, source, trainer_file):
    """Evaluate rl.py's own serial_as_none rule rather than restating it.

    Restating it made every codegen case below self-fulfilling: they passed
    whatever rl.py decided, so flipping SFT to True -- losing the config
    sentinel these modules exist to protect -- was caught only by the one test
    that matches the emitted literal, which is the line a developer would edit
    in the same change.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_serial_as_none":
            return eval(  # noqa: S307
                "(" + ast.get_source_segment(source, node.value) + ")",
                {"trainer_file": trainer_file},
            )
    raise AssertionError("_serial_as_none assignment not found in unsloth/models/rl.py")


def _rl_num_proc_snippet(trainer_file = "sft_trainer"):
    """The snippet rl.py would splice into that trainer's generated config.

    The expression is evaluated rather than literal_eval'd because it is no
    longer one literal: the serial encoding depends on the trainer being
    patched, and the point of these tests is what each one ends up with.
    """
    source = RL_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "num_proc_check":
            # Parenthesised: the segment starts at the first literal, so its continuation lines are an IndentationError
            expression = "(" + ast.get_source_segment(source, node.value) + ")"
            return eval(  # noqa: S307
                expression, {"_serial_as_none": _rl_serial_as_none(tree, source, trainer_file)}
            )
    raise AssertionError("num_proc_check literal not found in unsloth/models/rl.py")


def test_rl_codegen_writes_back_without_collapsing_serial():
    # SFT's config is read by an auto-sizer, so serial has to survive as 1.
    assert "serial_as_none = False" in _rl_num_proc_snippet("sft_trainer")


@pytest.mark.parametrize(
    "trainer_file",
    ["dpo_trainer", "kto_trainer", "cpo_trainer", "orpo_trainer", "reward_trainer", "prm_trainer"],
)
def test_rl_codegen_keeps_serial_as_none_where_the_config_reaches_map(trainer_file):
    """Only SFT has a downstream auto-sizer to defend the sentinel against.

    These trainers hand args.dataset_num_proc straight to Dataset.map, so a
    config `1` is a Pool(1) on datasets >= 4.1 -- one worker holding its own
    tokenizer copy, on the low-memory host that just refused workers. Nothing
    inflates a None there, so None is what serial must be.
    """
    assert "serial_as_none = True" in _rl_num_proc_snippet(trainer_file)


def test_rl_codegen_only_sft_gets_the_config_sentinel():
    # give every trainer the None encoding, including the one that must not have it.
    # Pin the discriminator itself:
    source = RL_PATH.read_text(encoding = "utf-8")
    assert '_serial_as_none = "False" if trainer_file == "sft_trainer" else "True"' in source



# The tag rl_replacements.py gives the num_proc edit;
NUM_PROC_WHERE = "sft_prepare_dataset dataset_num_proc selection"

# The helpers that decide whether a source edit lands.
ANCHOR_HELPERS = ("_require_replace", "_replace_or_fallback", "_same_source")

# The module-level regex the narrow fallback anchor uses.
NARROW_ANCHOR_NAME = "_ZOO_MAP_NUM_PROC_ASSIGNMENT"


def _zoo_dataset_utils_source():
    # canaries still run where torch/unsloth_zoo cannot import.
    # find_spec resolves the package path without executing its __init__, so these canaries still run where
    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("unsloth_zoo not installed")
    zoo_file = Path(list(spec.submodule_search_locations)[0]) / "dataset_utils.py"
    if not zoo_file.is_file():
        pytest.skip("unsloth_zoo.dataset_utils not found")
    return zoo_file.read_text(encoding = "utf-8")


def _rl_replacements_tree():
    return ast.parse(RL_PATH.with_name("rl_replacements.py").read_text(encoding = "utf-8"))


def _anchor_calls(where):
    return [
        node
        for node in ast.walk(_rl_replacements_tree())
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") in ANCHOR_HELPERS
        and any(k.arg == "where" and ast.literal_eval(k.value) == where for k in node.keywords)
    ]


def _anchor_and_count(where):
    """The (anchor, expected occurrences) of the source edit tagged ``where``."""
    found = _anchor_calls(where)
    assert len(found) == 1, f"expected exactly one anchored edit for {where!r}"
    node = found[0]
    count = next((ast.literal_eval(k.value) for k in node.keywords if k.arg == "count"), 1)
    return ast.literal_eval(node.args[1]), count


def _keyword(where, name):
    node = _anchor_calls(where)[0]
    value = next(k.value for k in node.keywords if k.arg == name)
    return value


def _narrow_num_proc_pattern():
    """The compiled fallback regex, read out of rl_replacements.py by name.

    Read from the module-level ``re.compile`` rather than re-typed here: a test
    holding its own copy of the pattern would keep passing after the real one
    drifted away from the Zoo.
    """
    name = _keyword(NUM_PROC_WHERE, "fallback_pattern")
    assert (
        getattr(name, "id", "") == NARROW_ANCHOR_NAME
    ), f"the num_proc fallback no longer uses {NARROW_ANCHOR_NAME}"
    for node in ast.walk(_rl_replacements_tree()):
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == NARROW_ANCHOR_NAME for t in node.targets)
            and isinstance(node.value, ast.Call)
        ):
            args = [ast.literal_eval(a) for a in node.value.args]
            flags = next((k for k in node.value.keywords if k.arg == "flags"), None)
            assert flags is not None and "MULTILINE" in ast.dump(
                flags.value
            ), "the narrow anchor must be MULTILINE to match a line in a block"
            return re.compile(args[0], flags = re.MULTILINE)
    raise AssertionError(f"{NARROW_ANCHOR_NAME} not found in rl_replacements.py")


def _narrow_num_proc_replacement():
    return ast.literal_eval(_keyword(NUM_PROC_WHERE, "fallback_new"))


def test_zoo_sft_prepare_dataset_anchor_has_not_drifted():
    """unsloth/models/rl_replacements.py rewrites unsloth_zoo's
    sft_prepare_dataset by exact string match. A Zoo release that touches those
    lines makes _require_replace raise at import time, so catch drift here."""
    source = _zoo_dataset_utils_source()

    # _require_replace raises on a missing anchor but cannot notice a count = 2
    for where in (
        NUM_PROC_WHERE,
        "sft_prepare_dataset tokenizing map() calls",
    ):
        anchor, count = _anchor_and_count(where)
        assert source.count(anchor) == count, (
            f"unsloth_zoo.dataset_utils has {source.count(anchor)} occurrences of "
            f"the {where!r} anchor, expected {count}; update rl_replacements.py"
        )


def test_the_narrow_num_proc_anchor_still_matches_the_installed_zoo():
    """The num_proc site has a second, narrower anchor; it needs its own canary.

    The block anchor drifting is survivable precisely because this regex catches
    the assignment the block ends on. If the Zoo ever renames or restructures
    that line too, both anchors are gone and nothing rewrites the worker count --
    so pin it here, in CI, rather than discovering it from a user's Pool(1).
    """
    source = _zoo_dataset_utils_source()
    pattern = _narrow_num_proc_pattern()

    matches = pattern.findall(source)
    assert len(matches) == 1, (
        f"unsloth_zoo.dataset_utils has {len(matches)} lines matching the narrow "
        f"num_proc anchor {pattern.pattern!r}, expected 1; update rl_replacements.py"
    )

    block_anchor, _ = _anchor_and_count(NUM_PROC_WHERE)
    assert pattern.search(block_anchor) is not None

    # And the fallback has to leave the file parseable at the Zoo's indentation.
    rewritten = pattern.sub(_narrow_num_proc_replacement(), source)
    assert rewritten != source
    ast.parse(rewritten)




def _load_anchor_helpers():
    """Exec just the anchor helpers out of rl_replacements.py.

    That module imports torch and trl at its top and this file is torch-free by
    design, so lift the definitions the anchor logic needs instead of copying
    them: a test carrying its own copy would keep passing after the real helper
    changed. Returns the namespace plus the list the fake logger warns into.
    """
    tree = _rl_replacements_tree()
    wanted = set(ANCHOR_HELPERS) | {"_warn_once", "_WARNED_MISSING_ANCHORS", NARROW_ANCHOR_NAME}
    kept = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in wanted for t in node.targets)
        )
    ]
    assert {node.name for node in kept if isinstance(node, ast.FunctionDef)} >= set(
        ANCHOR_HELPERS
    ), "the anchor helpers were renamed"

    warnings = []
    namespace = {
        "re": re,
        "logger": types.SimpleNamespace(warning = warnings.append),
    }
    module = ast.fix_missing_locations(ast.Module(body = kept, type_ignores = []))
    exec(compile(module, str(RL_PATH.with_name("rl_replacements.py")), "exec"), namespace)  # noqa: S102
    return namespace, warnings


def _apply_num_proc_edit(source):
    """Run the real layered edit for NUM_PROC_WHERE over ``source``."""
    namespace, warnings = _load_anchor_helpers()
    node = _anchor_calls(NUM_PROC_WHERE)[0]
    assert (
        getattr(node.func, "id", "") == "_replace_or_fallback"
    ), "the num_proc edit lost its fallback and is a plain replace again"
    result = namespace["_replace_or_fallback"](
        source,
        ast.literal_eval(node.args[1]),
        ast.literal_eval(node.args[2]),
        fallback_pattern = namespace[NARROW_ANCHOR_NAME],
        fallback_new = _narrow_num_proc_replacement(),
        where = NUM_PROC_WHERE,
    )
    return result, warnings


def test_the_block_anchor_is_used_when_the_zoo_has_not_moved():
    source = _zoo_dataset_utils_source()
    result, warnings = _apply_num_proc_edit(source)
    assert "_unsloth_get_dataset_num_proc" in result
    assert 'map_kwargs["num_proc"] = dataset_num_proc' not in result
    # the fallback would rewrite something the primary edit never touched.
    # The block anchor and the narrow anchor have to describe the same site, or the fallback would rewrite something
    # The block replacement takes the Zoo's own sizing with it;
    block_anchor, _ = _anchor_and_count(NUM_PROC_WHERE)
    assert block_anchor not in result
    assert warnings == [], f"a matching anchor must not warn: {warnings}"
    ast.parse(result)


def test_the_narrow_anchor_takes_over_when_the_block_drifts():
    """A Zoo that rewrites the block but keeps the assignment must still be fixed.

    This is the case `required = False` used to swallow: the edit no-ops, the Zoo
    reads the config `1` the config layer writes for "serial", and datasets >= 4.1
    turns that into a Pool(1) on the host that asked for no workers.
    """
    source = _zoo_dataset_utils_source()
    # Drift the block without touching the line the fallback keys on.
    drifted = source.replace(
        "            import multiprocessing as _mp\n",
        "            import multiprocessing as _mp  # zoo refactor\n",
        1,
    )
    assert drifted != source
    assert _narrow_num_proc_pattern().findall(drifted)

    result, warnings = _apply_num_proc_edit(drifted)
    assert "_unsloth_get_dataset_num_proc" in result, "the fallback anchor did not apply"
    assert 'map_kwargs["num_proc"] = dataset_num_proc' not in result
    # Only the assignment was rewritten, so the Zoo's own sizing is still there, computing a value nothing reads.
    assert "if _mp.get_start_method() != 'fork':" in result
    assert len(warnings) == 1 and "moved in this unsloth_zoo" in warnings[0]
    ast.parse(result)


def test_neither_anchor_matching_only_warns():
    """The remaining escape hatch stays a warning, not a raise.

    Hard-failing here would break every SFT run on a Zoo whose text merely moved,
    which is a far bigger blast radius than the one worker at stake.
    """
    source = (
        _zoo_dataset_utils_source()
        .replace(
            '            map_kwargs["num_proc"] = dataset_num_proc\n',
            '            map_kwargs.update({"num_proc": dataset_num_proc})\n',
            1,
        )
        .replace(
            "            import multiprocessing as _mp\n",
            "            import multiprocessing as _mp  # zoo refactor\n",
            1,
        )
    )
    assert not _narrow_num_proc_pattern().findall(source)

    result, warnings = _apply_num_proc_edit(source)
    assert result == source, "nothing should be rewritten when both anchors miss"
    assert len(warnings) == 1 and "anchor not found" in warnings[0]


def test_the_narrow_anchor_keeps_indentation_and_yields_none(monkeypatch):
    """The injected fallback has to run, at the Zoo's indentation, and give None.

    Indentation comes from the match, so a Zoo that nests the assignment deeper
    still gets valid source; and the value it computes for a config `1` -- what
    the config layer writes for "serial" -- has to be None, never 1.
    """
    module = _load_module()
    module.reset_warning_state()
    monkeypatch.delenv(module.NUM_PROC_ENV_VAR, raising = False)
    # The injected snippet imports the zoo copy first;
    if "unsloth_zoo" not in sys.modules:
        monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.dataset_num_proc", module)

    pattern = _narrow_num_proc_pattern()
    replacement = _narrow_num_proc_replacement()

    for indent in ("            ", "                    "):
        snippet = pattern.sub(replacement, f'{indent}map_kwargs["num_proc"] = dataset_num_proc')
        for line in snippet.split("\n"):
            assert line.startswith(indent), f"lost the {len(indent)}-space indent: {line!r}"

        namespace = {
            "map_kwargs": {},
            "args": types.SimpleNamespace(dataset_num_proc = 1),
        }
        exec(compile(textwrap.dedent(snippet), "<fallback>", "exec"), namespace)  # noqa: S102
        assert (
            namespace["map_kwargs"]["num_proc"] is None
        ), "a config 1 has to become None at the map site; 1 is a Pool(1) on datasets >= 4.1"


def test_rl_codegen_imports_the_module_that_exists():
    snippet = _rl_num_proc_snippet()
    assert f"from {GENERATED_IMPORT_MODULE} import {GENERATED_IMPORT_NAME}" in snippet
    assert f"from {GENERATED_FALLBACK_MODULE} import {GENERATED_IMPORT_NAME}" in snippet
    assert snippet.index(GENERATED_IMPORT_MODULE) < snippet.index(
        GENERATED_FALLBACK_MODULE
    ), "the zoo has to be tried first; the unsloth copy is only the fallback"
    assert MODULE_PATH.is_file()
    module = _load_module()
    assert callable(getattr(module, GENERATED_IMPORT_NAME))


def test_generated_source_reaches_for_the_zoo_before_unsloth():
    """Generated trainer source must not import back into unsloth.

    unsloth/__init__.py generates that source, so a `from unsloth...` there is an
    import of the package mid-flight, and it drags unsloth/utils/__init__.py ->
    packing -> attention_dispatch -> models._utils (torch and the model stack)
    into a module needing none of it. Both call sites must try the zoo first.
    """
    tree = _rl_replacements_tree()
    injected = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", "") not in ANCHOR_HELPERS:
            continue
        if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
            injected.append(ast.literal_eval(node.args[2]))
        # The narrow fallback injects source too, as an re.sub template.
        injected += [
            ast.literal_eval(k.value)
            for k in node.keywords
            if k.arg == "fallback_new" and isinstance(k.value, ast.Constant)
        ]
    reaching = [text for text in injected if "dataset_num_proc" in text]
    assert (
        len(reaching) == 3
    ), "expected the num_proc selection, its narrow fallback and the map() wrapper"
    for text in reaching:
        assert f"from {GENERATED_IMPORT_MODULE} import" in text
        assert text.index(GENERATED_IMPORT_MODULE) < text.index(
            GENERATED_FALLBACK_MODULE
        ), f"this injection imports unsloth before the zoo:\n{text}"


def test_the_two_copies_have_not_drifted():
    """The zoo owns the policy; this package keeps a fallback copy.

    Two copies can disagree, and the failure would be silent: whichever import
    wins decides the worker count. Compare them with docstrings stripped, so
    prose may differ per repo but no branch, constant or message may.
    """
    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("unsloth_zoo not installed")
    zoo_file = Path(list(spec.submodule_search_locations)[0]) / "dataset_num_proc.py"
    if not zoo_file.is_file():
        pytest.skip("this unsloth_zoo predates dataset_num_proc")

    def _shape(path):
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
        return ast.dump(ast.parse(ast.unparse(tree)))

    assert _shape(MODULE_PATH) == _shape(zoo_file), (
        "unsloth/dataset_num_proc.py and unsloth_zoo/dataset_num_proc.py "
        "have diverged; the zoo copy is the source of truth"
    )


def test_rl_codegen_snippet_is_valid_python_at_method_indent():
    snippet = _rl_num_proc_snippet()
    body = "\n".join(" " * 8 + line for line in snippet.split("\n"))
    source = (
        "class C:\n    def __init__(self, dataset_num_proc = None):\n" + body + "\n        pass\n"
    )
    ast.parse(source)


def test_rl_codegen_snippet_survives_an_unimportable_helper():
    # must still work, just leaving the caller's value alone.
    # The snippet is spliced into generated source as text, so a rename would otherwise surface only at
    # rl.py re-indents extra_args to 8 spaces and drops it into __init__.
    # A generated file can outlive an unsloth downgrade:
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




_DATASETS_MESSAGE = (
    "One of the subprocesses has abruptly died during map operation."
    "To debug the error, disable multiprocessing."
)


def test_worker_death_is_reraised_with_context(dnp):
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
    with pytest.raises(RuntimeError) as caught:
        with dnp.map_failure_diagnostics(None):
            raise RuntimeError(_DATASETS_MESSAGE)
    assert "dataset_num_proc = None" in str(caught.value)
    assert "1 worker," in str(caught.value)


def test_unrelated_errors_pass_through_untouched(dnp):
    # Only the dead-worker message is rewritten, and non-RuntimeError types are not caught at all.
    original = RuntimeError("CUDA out of memory")
    # num_proc=None still reaches the wrapper; it must not divide by a None.
    # datasets discards the child's exit status, so the original message cannot distinguish an OOM kill from anything
    with pytest.raises(RuntimeError) as caught:
        with dnp.map_failure_diagnostics(4):
            raise original
    assert caught.value is original

    key = KeyError("text")
    with pytest.raises(KeyError) as caught_key:
        with dnp.map_failure_diagnostics(4):
            raise key
    assert caught_key.value is key

    # The identity assertions above hold under `except Exception` as well, since the guard re-raises the same object.
    lookalike = ValueError("One of the subprocesses has abruptly died during map operation.")
    with pytest.raises(ValueError) as caught_other:
        with dnp.map_failure_diagnostics(4):
            raise lookalike
    assert caught_other.value is lookalike


def test_successful_map_is_not_disturbed(dnp):
    with dnp.map_failure_diagnostics(4):
        result = "tokenized"
    assert result == "tokenized"


def test_studio_num_proc_cap_has_not_drifted(dnp):
    """Unsloth duplicates AUTO_NUM_PROC_CAP; the two must stay equal.

    hardware.py cannot import this module without pulling unsloth's whole
    __init__ into hardware detection, so it carries its own copy. Read it out of
    the source, since importing studio needs a backend environment this lacks.
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
    Those arrive downstream as explicit ints, only clamped by free memory, so a
    roomy machine kept all 48.
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
    """UNSLOTH_DATASET_NUM_PROC=0 is not in-process on every installation.

    The dead-worker message is the one place a user is told what to do next, so
    it has to be true. On fork, train_on_responses_only over the Zoo's threshold
    gets ``1`` -- a bare None would read as "size it for me". The Zoo release
    that ships with this reads that ``1`` as in-process, but an older one turns
    it into a Pool(1), and generated code runs against whichever Zoo is
    installed. Saying "tokenize in-process" flatly was wrong for exactly the
    large-dataset runs that die.
    """
    with pytest.raises(RuntimeError) as excinfo:
        with dnp.map_failure_diagnostics(8):
            raise RuntimeError("One of the subprocesses has abruptly died during map operation.")
    message = str(excinfo.value)
    assert f"{dnp.NUM_PROC_ENV_VAR}=0" in message
    assert "single worker" in message, "the exception to in-process has to be stated"
    assert "train_on_responses_only" in message, "and which path it applies to"
    assert f"{dnp.ZOO_MIN_ROWS_FOR_MULTIPROC:,}" in message, "and above which size"


def test_the_advice_matches_what_the_resolver_actually_returns(dnp, monkeypatch):
    """Executed, not just read: drive both branches of the claim above.

    A message asserting a behaviour the resolver does not have would be worse
    than the vague one it replaced.
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
    ["spawn"]. Believing it reads Windows as forkable, so
    _workers_unusable_reason() returns None and workers get through -- the spawn
    re-import loop of #3211 / #3397.
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


class _Split:
    """Minimal sized stand-in for a datasets.Dataset."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n




def test_memory_budget_follows_the_cgroup_not_the_host(monkeypatch, dnp):
    """psutil reports the HOST inside a container.

    A 2GB pod on a 512GB box read as having room for the full worker set and got
    OOM-killed, which is the failure the memory ceiling exists to prevent.
    """
    psutil = pytest.importorskip("psutil")
    _force_start_method(monkeypatch, dnp, "fork")
    _force_cpus(monkeypatch, dnp, 64)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("m", (), {"available": 512 * 1024**3})()
    )
    monkeypatch.setattr(dnp, "_cgroup_free_bytes", lambda: None)
    assert dnp.get_dataset_num_proc(None) == dnp.AUTO_NUM_PROC_CAP

    monkeypatch.setattr(dnp, "_cgroup_free_bytes", lambda: 2 * 1024**3)
    assert dnp.get_dataset_num_proc(None) is None, "a 2GB container has no room for workers"

    monkeypatch.setattr(dnp, "_cgroup_free_bytes", lambda: 8 * 1024**3)
    assert dnp.get_dataset_num_proc(None) == 4


def test_memory_already_spent_in_the_container_is_not_counted_as_free(monkeypatch, dnp):
    psutil = pytest.importorskip("psutil")
    _force_start_method(monkeypatch, dnp, "fork")
    _force_cpus(monkeypatch, dnp, 64)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("m", (), {"available": 512 * 1024**3})()
    )
    monkeypatch.setattr(dnp, "_cgroup_free_bytes", lambda: 32 * 1024**3)
    assert dnp.get_dataset_num_proc(None) == dnp.AUTO_NUM_PROC_CAP

    # 30 of the 32GB already spent leaves 2, which is not enough for workers.
    monkeypatch.setattr(dnp, "_cgroup_free_bytes", lambda: 2 * 1024**3)
    assert dnp.get_dataset_num_proc(None) is None


def test_cpu_count_follows_the_affinity_mask(monkeypatch, dnp):
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 128)
    monkeypatch.setattr(dnp, "_cgroup_cpu_quota", lambda: None)
    monkeypatch.setattr(dnp.os, "sched_getaffinity", lambda pid: set(range(4)), raising = False)
    assert dnp._usable_cpus() == 4


def test_cpu_count_follows_a_fractional_cgroup_quota(monkeypatch, dnp):
    # Kubernetes "cpu:
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 128)
    monkeypatch.setattr(dnp.os, "sched_getaffinity", lambda pid: set(range(128)), raising = False)
    monkeypatch.setattr(dnp, "_cgroup_cpu_quota", lambda: 0.5)
    assert dnp._usable_cpus() == 1


def test_a_single_usable_cpu_tokenizes_in_process(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    _force_cpus(monkeypatch, dnp, 1)
    assert dnp.get_dataset_num_proc(None) is None


def test_the_cgroup_readers_never_raise(dnp):
    # They run on every auto-sizing call, on hosts with no cgroup at all.
    free = dnp._cgroup_free_bytes()
    assert free is None or (isinstance(free, int) and free >= 0)
    quota = dnp._cgroup_cpu_quota()
    assert quota is None or isinstance(quota, float)




def _force_stdlib_start_method(monkeypatch, dnp, method):
    real = dnp._module_start_method
    monkeypatch.setattr(
        dnp,
        "_module_start_method",
        lambda name: method if name == "multiprocessing" else real(name),
    )


def test_serial_is_one_when_the_two_modules_disagree(monkeypatch, dnp, capsys):
    """A None handed to train_on_responses_only is "size it for me" to it.

    Its auto path asks stdlib multiprocessing. Where that says fork while
    multiprocess says spawn, None is not serial: it picks cpu_count + 4 workers,
    and datasets then builds that pool on the spawn context -- the #3211 / #3397
    re-import loop, multiplied.
    """
    _force_start_method(monkeypatch, dnp, "spawn")
    _force_stdlib_start_method(monkeypatch, dnp, "fork")

    trainer = type("t", (), {"train_dataset": _Split(dnp.ZOO_MIN_ROWS_FOR_MULTIPROC * 2)})()
    assert dnp.resolve_responses_only_num_proc(trainer, None) == 1
    assert dnp.resolve_responses_only_num_proc(trainer, 16) == 1
    assert "disagree about the start method" in capsys.readouterr().out


def test_serial_stays_none_when_the_zoo_would_refuse_workers_too(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setattr(dnp.sys, "platform", "darwin")
    _force_stdlib_start_method(monkeypatch, dnp, "spawn")

    trainer = type("t", (), {"train_dataset": _Split(dnp.ZOO_MIN_ROWS_FOR_MULTIPROC * 2)})()
    assert dnp.resolve_responses_only_num_proc(trainer, None) is None
    assert dnp.resolve_responses_only_num_proc(trainer, 16) is None


def test_agreeing_modules_are_left_alone(monkeypatch, dnp):
    # macOS: multiprocess forks, stdlib spawns.
    _force_start_method(monkeypatch, dnp, "fork")
    _force_stdlib_start_method(monkeypatch, dnp, "fork")
    _force_cpus(monkeypatch, dnp, 32)
    # Otherwise a container that has already spent most of its limit still reads as having the whole thing available.
    # Under taskset or Slurm pinning the host count is not what this process can run on, and workers would only contend
    # Kubernetes "cpu: 500m" is cpu.max "50000 100000" = 0.5 cores.
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("m", (), {"available": 256 * 1024**3})()
    )
    monkeypatch.setattr(dnp, "_cgroup_free_bytes", lambda: None)

    trainer = type("t", (), {"train_dataset": _Split(dnp.ZOO_MIN_ROWS_FOR_MULTIPROC * 2)})()
    assert dnp.resolve_responses_only_num_proc(trainer, None) == dnp.AUTO_NUM_PROC_CAP
    assert dnp.resolve_responses_only_num_proc(trainer, 1) == 1


def _fake_cgroup_module(
    monkeypatch,
    v2_dirs = (),
    v1_dirs = (),
):
    import types

    def _read_first_line(path):
        return path.read_text() if path.is_file() else None

    def _parse_limit(raw):
        if not raw or raw.strip() == "max":
            return None
        try:
            return int(raw.strip())
        except ValueError:
            return None

    fake = types.ModuleType("unsloth_zoo.hf_xet_tuning")
    fake._cgroup_v2_dirs = lambda: list(v2_dirs)
    fake._cgroup_v1_dirs = lambda controller: list(v1_dirs)
    fake._read_first_line = _read_first_line
    fake._parse_limit = _parse_limit
    fake.cgroup_memory_limit = lambda: None
    fake.cgroup_cpu_limit = lambda: None
    monkeypatch.setitem(sys.modules, "unsloth_zoo.hf_xet_tuning", fake)
    return fake


def test_free_memory_pairs_each_limit_with_its_own_usage(monkeypatch, dnp, tmp_path):
    """The binding limit is often an ancestor's, and so is the usage that fills it.

    A leaf's usage against a slice's limit reports memory that siblings have
    already spent as free. The other direction is worse: the root
    memory.current is the whole machine, and against a unit's own MemoryMax it
    leaves every run with nothing.
    """
    slice_dir = tmp_path / "user.slice"
    leaf = slice_dir / "session.scope"
    leaf.mkdir(parents = True)

    (slice_dir / "memory.max").write_text("34359738368\n")
    (slice_dir / "memory.current").write_text("32212254720\n")
    (leaf / "memory.max").write_text("17179869184\n")
    (leaf / "memory.current").write_text("1073741824\n")

    _fake_cgroup_module(monkeypatch, v2_dirs = [leaf, slice_dir])
    assert dnp._cgroup_free_bytes() == 2 * 1024**3


def test_free_memory_is_never_negative(monkeypatch, dnp, tmp_path):
    leaf = tmp_path / "scope"
    leaf.mkdir()
    (leaf / "memory.max").write_text("1073741824\n")
    (leaf / "memory.current").write_text("2147483648\n")
    _fake_cgroup_module(monkeypatch, v2_dirs = [leaf])
    assert dnp._cgroup_free_bytes() == 0


def test_an_unlimited_cgroup_is_not_a_ceiling(monkeypatch, dnp, tmp_path):
    leaf = tmp_path / "scope"
    leaf.mkdir()
    (leaf / "memory.max").write_text("max\n")
    (leaf / "memory.current").write_text("1073741824\n")
    _fake_cgroup_module(monkeypatch, v2_dirs = [leaf])
    assert dnp._cgroup_free_bytes() is None


def test_a_readable_limit_with_no_readable_usage_still_binds(monkeypatch, dnp, tmp_path):
    leaf = tmp_path / "scope"
    leaf.mkdir()
    (leaf / "memory.max").write_text("2147483648\n")
    _fake_cgroup_module(monkeypatch, v2_dirs = [leaf])
    assert dnp._cgroup_free_bytes() == 2 * 1024**3


def _old_zoo(monkeypatch, memory_limit = None):
    """An unsloth_zoo predating the private cgroup helpers, with only the public reader."""
    import types

    fake = types.ModuleType("unsloth_zoo.hf_xet_tuning")
    fake.cgroup_memory_limit = lambda: memory_limit
    fake.cgroup_cpu_limit = lambda: None
    monkeypatch.setitem(sys.modules, "unsloth_zoo.hf_xet_tuning", fake)
    return fake


def test_the_unaided_reader_subtracts_usage_too(monkeypatch, dnp, tmp_path):
    """An older unsloth_zoo must not turn the ceiling back into the raw limit.

    cgroup_memory_limit() alone reports an 8GB cgroup holding 6GB as 8GB free,
    which is the one case the ceiling exists for: sizing workers off memory that
    is already spent is how #2693's map() children get OOM-killed.
    """
    _old_zoo(monkeypatch, memory_limit = 8 * 1024**3)
    leaf = tmp_path / "kubepods" / "podabc"
    leaf.mkdir(parents = True)
    (leaf / "memory.max").write_text("8589934592\n")
    (leaf / "memory.current").write_text("6442450944\n")

    monkeypatch.setattr(dnp, "CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(dnp, "_proc_self_cgroup", lambda: ["0::/kubepods/podabc"])
    assert dnp._cgroup_free_bytes() == 2 * 1024**3


def test_the_unaided_reader_walks_to_the_binding_ancestor(monkeypatch, dnp, tmp_path):
    """Same pairing rule as the helper-backed path: the slice's limit binds, with the slice's usage."""
    _old_zoo(monkeypatch)
    slice_dir = tmp_path / "user.slice"
    leaf = slice_dir / "session.scope"
    leaf.mkdir(parents = True)
    # The slice caps 32GB and 30 of them are spent, mostly by a sibling;
    # this leaf has a 16GB cap of its own and has spent 1.
    (slice_dir / "memory.max").write_text("34359738368\n")
    (slice_dir / "memory.current").write_text("32212254720\n")
    (leaf / "memory.max").write_text("17179869184\n")
    (leaf / "memory.current").write_text("1073741824\n")

    monkeypatch.setattr(dnp, "CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(dnp, "_proc_self_cgroup", lambda: ["0::/user.slice/session.scope"])
    # 34 - 32 = 2GB from the slice, 16 - 1 = 15GB from the leaf.
    assert dnp._cgroup_free_bytes() == 2 * 1024**3


def test_the_unaided_reader_handles_cgroup_v1(monkeypatch, dnp, tmp_path):
    _old_zoo(monkeypatch)
    leaf = tmp_path / "memory" / "slurm" / "job_1"
    leaf.mkdir(parents = True)
    (leaf / "memory.limit_in_bytes").write_text("4294967296\n")
    (leaf / "memory.usage_in_bytes").write_text("3221225472\n")

    monkeypatch.setattr(dnp, "CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(
        dnp,
        "_proc_self_cgroup",
        lambda: ["7:memory,blkio:/slurm/job_1"],
    )
    assert dnp._cgroup_free_bytes() == 1024**3


def test_the_unaided_reader_ignores_the_unlimited_sentinels(monkeypatch, dnp, tmp_path):
    _old_zoo(monkeypatch)
    v2 = tmp_path / "scope"
    v2.mkdir()
    (v2 / "memory.max").write_text("max\n")
    (v2 / "memory.current").write_text("1073741824\n")
    v1 = tmp_path / "memory"
    v1.mkdir()
    # v1's "unlimited": a near-2^63 sentinel, not a 8-exabyte ceiling.
    (v1 / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    (v1 / "memory.usage_in_bytes").write_text("1073741824\n")

    monkeypatch.setattr(dnp, "CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(dnp, "_proc_self_cgroup", lambda: ["0::/scope", "7:memory:/"])
    assert dnp._cgroup_free_bytes() is None


def test_the_unaided_reader_is_never_negative(monkeypatch, dnp, tmp_path):
    _old_zoo(monkeypatch)
    # An over-committed cgroup reports more usage than its limit under pressure.
    leaf = tmp_path / "scope"
    leaf.mkdir()
    (leaf / "memory.max").write_text("1073741824\n")
    (leaf / "memory.current").write_text("2147483648\n")
    monkeypatch.setattr(dnp, "CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(dnp, "_proc_self_cgroup", lambda: ["0::/scope"])
    assert dnp._cgroup_free_bytes() == 0


def test_the_unaided_reader_keeps_the_public_limit_as_a_last_resort(monkeypatch, dnp, tmp_path):
    """No readable cgroup tree here, but an older zoo may still find one its own way.

    A limit with no usage beside it is still a ceiling, and is a tighter one than
    psutil's host-wide view inside a container.
    """
    _old_zoo(monkeypatch, memory_limit = 4 * 1024**3)
    monkeypatch.setattr(dnp, "CGROUP_ROOT", str(tmp_path / "absent"))
    monkeypatch.setattr(dnp, "_proc_self_cgroup", lambda: [])
    assert dnp._cgroup_free_bytes() == 4 * 1024**3


def test_the_unaided_reader_never_raises(monkeypatch, dnp):
    """It runs on every auto-sizing call under an older zoo, on hosts with no cgroup at all."""
    _old_zoo(monkeypatch)
    free = dnp._cgroup_free_bytes_unaided()
    assert free is None or (isinstance(free, int) and free >= 0)


def _no_hf_xet_tuning(monkeypatch):
    """Neither the private helpers nor the public readers: no unsloth_zoo at all."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "unsloth_zoo.hf_xet_tuning":
            raise ImportError("older unsloth_zoo")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)


def test_no_unsloth_zoo_and_no_cgroup_is_not_a_ceiling(monkeypatch, dnp, tmp_path):
    # The cgroup tree is pointed away from the host's on purpose:
    _no_hf_xet_tuning(monkeypatch)
    monkeypatch.setattr(dnp, "CGROUP_ROOT", str(tmp_path / "absent"))
    monkeypatch.setattr(dnp, "_proc_self_cgroup", lambda: [])
    assert dnp._cgroup_free_bytes() is None
    assert dnp._cgroup_cpu_quota() is None


def test_no_unsloth_zoo_still_reads_the_cgroup(monkeypatch, dnp, tmp_path):
    """The point of the unaided reader: the ceiling survives having no zoo to ask."""
    _no_hf_xet_tuning(monkeypatch)
    leaf = tmp_path / "scope"
    leaf.mkdir()
    (leaf / "memory.max").write_text("8589934592\n")
    (leaf / "memory.current").write_text("6442450944\n")
    monkeypatch.setattr(dnp, "CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(dnp, "_proc_self_cgroup", lambda: ["0::/scope"])
    assert dnp._cgroup_free_bytes() == 2 * 1024**3
    # The CPU quota reader has no unaided path, so it stays silent.
    assert dnp._cgroup_cpu_quota() is None


def test_env_forced_serial_is_in_process_on_a_small_split(monkeypatch, dnp):
    """The documented recovery has to actually recover.

    UNSLOTH_DATASET_NUM_PROC=0 with the config sentinel 1 arriving as an explicit
    count used to return 1, which bypasses the small-split guard and builds a
    Pool(1) on datasets >= 4.1. Under the threshold the guard is in-process, so
    None is what expresses the request exactly.
    """
    _force_start_method(monkeypatch, dnp, "fork")
    _force_stdlib_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "0")

    small = type("t", (), {"train_dataset": _Split(100)})()
    assert dnp.resolve_responses_only_num_proc(small, 1) is None
    assert dnp.resolve_responses_only_num_proc(small, None) is None

    # Over the threshold the guard is gone, and 1 is the least it can be given.
    big = type("t", (), {"train_dataset": _Split(dnp.ZOO_MIN_ROWS_FOR_MULTIPROC * 2)})()
    assert dnp.resolve_responses_only_num_proc(big, 1) == 1


def test_a_memory_starved_explicit_count_is_in_process_on_a_small_split(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    _force_stdlib_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 0)

    small = type("t", (), {"train_dataset": _Split(100)})()
    assert dnp.resolve_responses_only_num_proc(small, 16) is None


def test_an_explicit_count_the_host_can_afford_is_untouched_by_the_row_guard(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    _force_stdlib_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 1000)

    small = type("t", (), {"train_dataset": _Split(100)})()
    assert dnp.resolve_responses_only_num_proc(small, 4) == 4


def test_the_fallback_does_not_sit_behind_a_torch_import():
    """MLX hosts have no torch, and reach this module through chat_templates.

    unsloth/utils/__init__.py imports .packing, which imports torch, and
    .attention_dispatch, which imports unsloth.models._utils on top. A fallback
    under that package would raise on the torch-free host it exists to serve, so
    it lives at the top level and every reference to it has to stay there.
    """
    assert (
        MODULE_PATH.parent.name == "unsloth"
    ), "the fallback moved back under a package whose __init__ imports torch"

    utils_init = REPO_ROOT / "unsloth" / "utils" / "__init__.py"
    reached = {
        node.module.split(".")[0]
        if isinstance(node, ast.ImportFrom) and node.level == 0
        else (node.module or "").lstrip(".")
        for node in ast.parse(utils_init.read_text(encoding = "utf-8")).body
        if isinstance(node, ast.ImportFrom)
    }
    assert reached, "unsloth/utils/__init__.py stopped importing anything; re-check the premise"

    for path in (
        REPO_ROOT / "unsloth" / "chat_templates.py",
        REPO_ROOT / "unsloth" / "models" / "rl.py",
        REPO_ROOT / "unsloth" / "models" / "rl_replacements.py",
    ):
        source = path.read_text(encoding = "utf-8")
        assert (
            "unsloth.utils.dataset_num_proc" not in source
        ), f"{path.name} reaches it via unsloth.utils"
        assert ".utils.dataset_num_proc" not in source, f"{path.name} reaches it via unsloth.utils"


def test_the_fixture_really_neutralises_the_zoo_readers(dnp):
    """The fixture's patching must survive a package __init__ that raises.

    unsloth_zoo/__init__ imports hf_xet_tuning near the top and can raise at the
    end, which drops the package from sys.modules but leaves the submodule
    cached -- and that cache entry is what the policy imports. Patching only on
    a clean `from unsloth_zoo import hf_xet_tuning` leaves the real readers live
    on the runner's own /sys/fs/cgroup, so every sizing assertion silently
    becomes a test of the container's memory limit.
    """
    module = sys.modules.get("unsloth_zoo.hf_xet_tuning")
    if module is None:
        pytest.skip("unsloth_zoo.hf_xet_tuning is not reachable here")
    assert str(module.CGROUP_ROOT).startswith("/nonexistent"), module.CGROUP_ROOT
    assert module._cgroup_v2_dirs() == []
    assert module._cgroup_v1_dirs("memory") == []


def test_the_unaided_reader_picks_its_own_v1_line_too(monkeypatch, dnp, tmp_path):
    """The other half of the scan: a v1 line that is not the first line.

    The hybrid case above puts the memory controller first, so taking line 0
    would still land on it. Here a pids line comes first and the memory
    controller sits at a different path, which is what a Slurm step looks like:
    reading line 0 walks a directory that does not exist and loses the ceiling
    the step was given.
    """
    _no_hf_xet_tuning(monkeypatch)
    v2_leaf = tmp_path / "user.slice" / "app.scope"
    v2_leaf.mkdir(parents = True)
    (v2_leaf / "memory.max").write_text("8589934592\n")
    (v2_leaf / "memory.current").write_text("6442450944\n")
    # 4GB capped, 3 spent: 1GB free, which is less than the v2 side's 2GB.
    v1_leaf = tmp_path / "memory" / "slurm" / "job_1"
    v1_leaf.mkdir(parents = True)
    (v1_leaf / "memory.limit_in_bytes").write_text("4294967296\n")
    (v1_leaf / "memory.usage_in_bytes").write_text("3221225472\n")

    monkeypatch.setattr(dnp, "CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(
        dnp,
        "_proc_self_cgroup",
        lambda: [
            "11:pids:/user.slice/user-1000.slice/session-3.scope",
            "10:memory:/slurm/job_1",
            "0::/user.slice/app.scope",
        ],
    )
    assert dnp._cgroup_free_bytes() == 1024**3


def test_the_hatch_wins_on_a_small_split_too(monkeypatch, dnp):
    """A split under the threshold is where the resolver would otherwise stand
    down, and standing down there would silently discard the count a user set
    by hand to get workers on a small dataset."""
    # Same shape without the env var:
    _force_start_method(monkeypatch, dnp, "fork")
    _force_stdlib_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "16")

    small = type("t", (), {"train_dataset": _Split(100)})()
    assert dnp.resolve_responses_only_num_proc(small, None) == 16
