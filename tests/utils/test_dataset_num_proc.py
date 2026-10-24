# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Loaded straight off disk: the module is stdlib-only, so tests run where torch cannot import."""

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
    # Import before sys.platform is spoofed: multiprocess picks contexts at import.
    import multiprocess  # noqa: F401
except ImportError:
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "unsloth" / "dataset_num_proc.py"
RL_PATH = REPO_ROOT / "unsloth" / "models" / "rl.py"

# Zoo first so generated source does not import unsloth; this package as fallback.
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
    # macOS is refused by policy, so pin a forking platform; platform tests override.
    monkeypatch.setattr(module.sys, "platform", "linux")
    # Pin memory at its sources: free RAM and cgroup clamps would otherwise skew counts.
    try:
        import psutil
        monkeypatch.setattr(
            psutil, "virtual_memory", lambda: type("m", (), {"available": 1024 * 1024**3})()
        )
    except ImportError:
        pass
    monkeypatch.setattr(module, "CGROUP_ROOT", "/nonexistent-cgroup-root-for-tests")
    # Neutralise zoo readers by name; older zoos lack CGROUP_ROOT and read the real cgroup.
    try:
        from unsloth_zoo import hf_xet_tuning
    except Exception:
        # A failed package import can leave the submodule cached, so read it from sys.modules.
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
    """Patch _usable_cpus: psutil alone misses the affinity mask and cgroup quota, which also cap it."""
    monkeypatch.setattr(dnp, "_usable_cpus", lambda: count)


@pytest.mark.parametrize("method", ["spawn", "forkserver", None])
def test_non_fork_start_method_disables_multiprocessing(monkeypatch, dnp, method):
    # Spawned children cannot re-import the generated trainer module.
    _force_start_method(monkeypatch, dnp, method)
    assert dnp.get_dataset_num_proc(8) is None
    assert dnp.get_dataset_num_proc(None) is None


def test_non_fork_start_method_warns_once(monkeypatch, dnp, capsys):
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
    # `1` is a trap: callers mean serial, datasets >= 4.0 builds a Pool(1).
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(value) is None


def test_serial_as_none_false_preserves_an_explicit_one(monkeypatch, dnp):
    """Keep an explicit 1: a config None means auto-size downstream, which would inflate the worker
    count."""
    _force_start_method(monkeypatch, dnp, "fork")
    assert dnp.get_dataset_num_proc(1, serial_as_none = False) == 1
    # 0 and negatives map to the config serial sentinel (1), not None.
    assert dnp.get_dataset_num_proc(0, serial_as_none = False) == 1
    assert dnp.get_dataset_num_proc(-4, serial_as_none = False) == 1


def test_config_layer_never_returns_none_while_forking_is_available(monkeypatch, dnp):
    """On a fork host no path may return None, since downstream None means auto-size and re-inflates."""
    psutil = pytest.importorskip("psutil")
    _force_cpus(monkeypatch, dnp, 64)

    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("m", (), {"available": 1 * 1024**3})()
    )
    assert dnp.get_dataset_num_proc(16, serial_as_none = False) == 1
    assert dnp.get_dataset_num_proc(None, serial_as_none = False) == 1


@pytest.mark.parametrize("method", ["spawn", "forkserver", None])
@pytest.mark.parametrize("desired", [None, 1, 16])
def test_config_layer_is_none_not_one_on_a_non_fork_start_method(monkeypatch, dnp, method, desired):
    """On spawn-only hosts serial must be None, not 1: unpatched TRL maps would build a Pool(1)."""
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

    assert site(cfg(1)) is None
    assert site(cfg(6)) == 6
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
    # 10 GB free, half budgeted, ~1 GB per worker -> 5.
    assert dnp.get_dataset_num_proc(None) == 5


def test_explicit_value_is_clamped_by_memory(monkeypatch, dnp, capsys):
    """Explicit worker counts must also be memory-clamped; the old heuristic only bounded the auto path."""
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
    # Config serial is 1, never None: zoo reads None as auto-size.
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, raw)
    assert dnp.get_dataset_num_proc(16, serial_as_none = False) == 1


def test_env_override_is_uncapped(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "100")
    assert dnp.get_dataset_num_proc(None) == 100
    # Pin the memory clamp, or the fixture's room for 512 workers proves nothing.
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 2)
    assert dnp.get_dataset_num_proc(None) == 100
    assert dnp.get_dataset_num_proc(4) == 100


def test_invalid_env_override_is_ignored_with_a_warning(monkeypatch, dnp, capsys):
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "banana")
    assert dnp.get_dataset_num_proc(4) == 4
    assert "is not an integer" in capsys.readouterr().out


def test_start_method_probe_prefers_multiprocess_and_has_no_side_effects(dnp):
    """datasets does `from multiprocess import Pool`, so `multiprocess` -- not
    stdlib multiprocessing -- decides how map() spawns. Reading it must also not
    pin the context, which would make a later set_start_method() raise."""
    multiprocess = pytest.importorskip("multiprocess")
    import multiprocessing

    before_mp = multiprocess.get_start_method(allow_none = True)
    before_std = multiprocessing.get_start_method(allow_none = True)

    method = dnp.multiprocessing_start_method()

    # The private default context has answered "fork" on Windows, which only spawns.
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
    """On macOS multiprocess lists spawn first but defaults to fork; read the default, not the list."""
    import sys as _sys

    darwin_order = ["spawn", "fork", "forkserver"]
    fake = _fake_multiprocess(darwin_order, "fork")
    monkeypatch.setitem(_sys.modules, "multiprocess", fake)
    assert dnp.multiprocessing_start_method() == "fork"


def test_macos_stays_in_process_even_though_multiprocess_forks(monkeypatch, dnp, capsys):
    """macOS stays in-process: the probe reports fork, but forking there can crash with a threaded BLAS."""
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 1000)
    monkeypatch.setattr(dnp.sys, "platform", "darwin")

    assert dnp.get_dataset_num_proc(8) is None
    assert dnp.get_dataset_num_proc(8, serial_as_none = False) is None
    assert dnp.get_dataset_num_proc(None) is None
    assert "macOS" in capsys.readouterr().out

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
    """Evaluates rl.py's serial_as_none rule itself; a restated copy makes every case self-fulfilling."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_serial_as_none":
            # Parenthesised: continuation lines alone are an IndentationError.
            return eval(  # noqa: S307
                "(" + ast.get_source_segment(source, node.value) + ")",
                {"trainer_file": trainer_file},
            )
    raise AssertionError("_serial_as_none assignment not found in unsloth/models/rl.py")


def _rl_num_proc_snippet(trainer_file = "sft_trainer"):
    """Evaluated, not literal_eval'd: the serial encoding depends on whether the trainer is patched."""
    source = RL_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "num_proc_check":
            # Parenthesised: continuation lines alone are an IndentationError.
            expression = "(" + ast.get_source_segment(source, node.value) + ")"
            return eval(  # noqa: S307
                expression, {"_serial_as_none": _rl_serial_as_none(tree, source, trainer_file)}
            )
    raise AssertionError("num_proc_check literal not found in unsloth/models/rl.py")


def test_rl_codegen_writes_back_without_collapsing_serial():
    # SFT's config is read by an auto-sizer, so serial has to stay 1.
    assert "serial_as_none = False" in _rl_num_proc_snippet("sft_trainer")


@pytest.mark.parametrize(
    "trainer_file",
    ["dpo_trainer", "kto_trainer", "cpo_trainer", "orpo_trainer", "reward_trainer", "prm_trainer"],
)
def test_rl_codegen_keeps_serial_as_none_where_the_config_reaches_map(trainer_file):
    """Trainers that pass the config to Dataset.map need serial None; a 1 there builds a Pool(1)."""
    assert "serial_as_none = True" in _rl_num_proc_snippet(trainer_file)


def test_rl_codegen_only_sft_gets_the_config_sentinel():
    source = RL_PATH.read_text(encoding = "utf-8")
    assert '_serial_as_none = "False" if trainer_file == "sft_trainer" else "True"' in source


# Both anchors share this tag, so a rename shows up as a missing call.
NUM_PROC_WHERE = "sft_prepare_dataset dataset_num_proc selection"

ANCHOR_HELPERS = ("_require_replace", "_replace_or_fallback", "_same_source")

NARROW_ANCHOR_NAME = "_ZOO_MAP_NUM_PROC_ASSIGNMENT"


def _zoo_dataset_utils_source():
    # find_spec avoids executing __init__, so this runs without torch/unsloth_zoo.
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


def _num_proc_replacement():
    return ast.literal_eval(_anchor_calls(NUM_PROC_WHERE)[0].args[2])


def _zoo_adopted_the_num_proc_edit(source):
    return _num_proc_replacement() in source


def _zoo_source_as_anchored():
    """Installed Zoo source with an adopted num_proc block reverted to the anchor, so drift tests still bite on a newer Zoo."""
    source = _zoo_dataset_utils_source()
    if _zoo_adopted_the_num_proc_edit(source):
        anchor, _ = _anchor_and_count(NUM_PROC_WHERE)
        source = source.replace(_num_proc_replacement(), anchor, 1)
    return source


def _keyword(where, name):
    node = _anchor_calls(where)[0]
    value = next(k.value for k in node.keywords if k.arg == name)
    return value


def _narrow_num_proc_pattern():
    """Read from the module-level re.compile, not re-typed, so a drifted pattern cannot pass silently."""
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

    # _require_replace cannot notice a count-2 anchor dropping to one, so count here.
    for where in (
        NUM_PROC_WHERE,
        "sft_prepare_dataset tokenizing map() calls",
    ):
        anchor, count = _anchor_and_count(where)
        if where == NUM_PROC_WHERE and _zoo_adopted_the_num_proc_edit(source):
            # The Zoo ships the replacement itself; test_a_zoo_that_adopted_the_num_proc_edit_is_left_alone covers it.
            continue
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
    source = _zoo_source_as_anchored()
    pattern = _narrow_num_proc_pattern()

    matches = pattern.findall(source)
    assert len(matches) == 1, (
        f"unsloth_zoo.dataset_utils has {len(matches)} lines matching the narrow "
        f"num_proc anchor {pattern.pattern!r}, expected 1; update rl_replacements.py"
    )

    # Both anchors must describe the same site, or the fallback rewrites other code.
    block_anchor, _ = _anchor_and_count(NUM_PROC_WHERE)
    assert pattern.search(block_anchor) is not None

    rewritten = pattern.sub(_narrow_num_proc_replacement(), source)
    assert rewritten != source
    ast.parse(rewritten)


def _load_anchor_helpers():
    """Execs the anchor helpers lifted from rl_replacements.py, since that module imports torch and trl."""
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


def test_a_zoo_that_adopted_the_num_proc_edit_is_left_alone():
    """No rewrite and no warning; a narrow-anchor rewrite of its first line would leave continuation lines dangling."""
    source = _zoo_dataset_utils_source()
    adopted = source
    if not _zoo_adopted_the_num_proc_edit(adopted):
        anchor, _ = _anchor_and_count(NUM_PROC_WHERE)
        adopted = source.replace(anchor, _num_proc_replacement(), 1)
        assert adopted != source
    result, warnings = _apply_num_proc_edit(adopted)
    assert result == adopted
    assert warnings == [], f"an adopted block must not warn: {warnings}"
    ast.parse(result)


def test_the_block_anchor_is_used_when_the_zoo_has_not_moved():
    source = _zoo_source_as_anchored()
    result, warnings = _apply_num_proc_edit(source)
    assert "_unsloth_get_dataset_num_proc" in result
    assert 'map_kwargs["num_proc"] = dataset_num_proc' not in result
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
    source = _zoo_source_as_anchored()
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
    assert "if _mp.get_start_method() != 'fork':" in result
    assert len(warnings) == 1 and "moved in this unsloth_zoo" in warnings[0]
    ast.parse(result)


def test_neither_anchor_matching_only_warns():
    """Neither anchor matching only warns: a hard failure would break every SFT run whose Zoo text moved."""
    source = (
        _zoo_source_as_anchored()
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
    """The injected fallback must give None, never 1, for a config 1, and match the Zoo's indentation."""
    module = _load_module()
    module.reset_warning_state()
    monkeypatch.delenv(module.NUM_PROC_ENV_VAR, raising = False)
    # Point the zoo name at the copy under test to stay torch-free and offline.
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
    # Spliced as text, so a rename would otherwise surface only at trainer construction.
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
    """Generated trainer source tries unsloth_zoo first; importing unsloth mid-flight drags in torch."""
    tree = _rl_replacements_tree()
    injected = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", "") not in ANCHOR_HELPERS:
            continue
        if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
            injected.append(ast.literal_eval(node.args[2]))
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
    """Compares zoo and fallback copies with docstrings stripped; only prose may differ, not code."""
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
    # rl.py re-indents extra_args to 8 spaces into __init__.
    snippet = _rl_num_proc_snippet()
    body = "\n".join(" " * 8 + line for line in snippet.split("\n"))
    source = (
        "class C:\n    def __init__(self, dataset_num_proc = None):\n" + body + "\n        pass\n"
    )
    ast.parse(source)


def test_rl_codegen_snippet_survives_an_unimportable_helper():
    # A generated file can outlive an unsloth downgrade; config must still build.
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
    # datasets discards the child's exit status, so OOM kills are indistinguishable.
    with pytest.raises(RuntimeError) as caught:
        with dnp.map_failure_diagnostics(8):
            raise RuntimeError(_DATASETS_MESSAGE)

    message = str(caught.value)
    assert "dataset_num_proc = 8" in message
    assert "8 workers" in message
    assert "8GB" in message, "should estimate what those workers cost"
    assert dnp.NUM_PROC_ENV_VAR in message, "must name the escape hatch"
    assert "out-of-memory" in message
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert _DATASETS_MESSAGE in str(caught.value.__cause__)


def test_worker_death_diagnostics_handles_in_process_runs(dnp):
    with pytest.raises(RuntimeError) as caught:
        with dnp.map_failure_diagnostics(None):
            raise RuntimeError(_DATASETS_MESSAGE)
    assert "dataset_num_proc = None" in str(caught.value)
    assert "1 worker," in str(caught.value)


def test_unrelated_errors_pass_through_untouched(dnp):
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

    # Catches a widened except clause, which the identity assertions above cannot.
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
    """Reads studio's hardware.py copy of AUTO_NUM_PROC_CAP; importing it would load unsloth's __init__."""
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
    """Studio's worker heuristic must respect the cap: downstream explicit ints are only memory-clamped."""
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
    """UNSLOTH_DATASET_NUM_PROC=0 is not in-process on every install; the advice must not overpromise."""
    with pytest.raises(RuntimeError) as excinfo:
        with dnp.map_failure_diagnostics(8):
            raise RuntimeError("One of the subprocesses has abruptly died during map operation.")
    message = str(excinfo.value)
    assert f"{dnp.NUM_PROC_ENV_VAR}=0" in message
    assert "single worker" in message, "the exception to in-process has to be stated"
    assert "train_on_responses_only" in message, "and which path it applies to"
    assert f"{dnp.ZOO_MIN_ROWS_FOR_MULTIPROC:,}" in message, "and above which size"


def test_the_advice_matches_what_the_resolver_actually_returns(dnp, monkeypatch):
    """Runs both branches of the advice so the message cannot claim behaviour the resolver lacks."""

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
    """Reject a start method the host does not offer; the private default context is wrong on Windows."""
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
    """psutil reports the host's memory inside a container, so the budget must follow the cgroup instead."""
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

    monkeypatch.setattr(dnp, "_cgroup_free_bytes", lambda: 2 * 1024**3)
    assert dnp.get_dataset_num_proc(None) is None


def test_cpu_count_follows_the_affinity_mask(monkeypatch, dnp):
    # Under taskset or Slurm pinning, the host core count overstates what is usable.
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 128)
    monkeypatch.setattr(dnp, "_cgroup_cpu_quota", lambda: None)
    monkeypatch.setattr(dnp.os, "sched_getaffinity", lambda pid: set(range(4)), raising = False)
    assert dnp._usable_cpus() == 4


def test_cpu_count_follows_a_fractional_cgroup_quota(monkeypatch, dnp):
    # k8s "cpu: 500m" is cpu.max "50000 100000" = 0.5 cores.
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
    """On stdlib fork vs multiprocess spawn disagreement, None auto-sizes to many workers; return 1."""
    _force_start_method(monkeypatch, dnp, "spawn")
    _force_stdlib_start_method(monkeypatch, dnp, "fork")

    trainer = type("t", (), {"train_dataset": _Split(dnp.ZOO_MIN_ROWS_FOR_MULTIPROC * 2)})()
    assert dnp.resolve_responses_only_num_proc(trainer, None) == 1
    assert dnp.resolve_responses_only_num_proc(trainer, 16) == 1
    assert "disagree about the start method" in capsys.readouterr().out


def test_serial_stays_none_when_the_zoo_would_refuse_workers_too(monkeypatch, dnp):
    # macOS: multiprocess forks, stdlib spawns; its veto fires so None is in-process.
    _force_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setattr(dnp.sys, "platform", "darwin")
    _force_stdlib_start_method(monkeypatch, dnp, "spawn")

    trainer = type("t", (), {"train_dataset": _Split(dnp.ZOO_MIN_ROWS_FOR_MULTIPROC * 2)})()
    assert dnp.resolve_responses_only_num_proc(trainer, None) is None
    assert dnp.resolve_responses_only_num_proc(trainer, 16) is None


def test_agreeing_modules_are_left_alone(monkeypatch, dnp):
    _force_start_method(monkeypatch, dnp, "fork")
    _force_stdlib_start_method(monkeypatch, dnp, "fork")
    _force_cpus(monkeypatch, dnp, 32)
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
    """Pair each limit with usage from the same cgroup; mixing levels misreports free memory."""
    slice_dir = tmp_path / "user.slice"
    leaf = slice_dir / "session.scope"
    leaf.mkdir(parents = True)

    (slice_dir / "memory.max").write_text("34359738368\n")
    (slice_dir / "memory.current").write_text("32212254720\n")
    (leaf / "memory.max").write_text("17179869184\n")
    (leaf / "memory.current").write_text("1073741824\n")

    _fake_cgroup_module(monkeypatch, v2_dirs = [leaf, slice_dir])
    # Slice leaves 2GB, leaf 15GB: the slice binds.
    assert dnp._cgroup_free_bytes() == 2 * 1024**3


def test_free_memory_is_never_negative(monkeypatch, dnp, tmp_path):
    # An over-committed cgroup reports usage above its limit under pressure.
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
    """An older zoo's cgroup_memory_limit() reports the raw limit as free; usage must be subtracted."""
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
    (slice_dir / "memory.max").write_text("34359738368\n")
    (slice_dir / "memory.current").write_text("32212254720\n")
    (leaf / "memory.max").write_text("17179869184\n")
    (leaf / "memory.current").write_text("1073741824\n")

    monkeypatch.setattr(dnp, "CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(dnp, "_proc_self_cgroup", lambda: ["0::/user.slice/session.scope"])
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
    # v1's "unlimited" is a near-2^63 sentinel.
    (v1 / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    (v1 / "memory.usage_in_bytes").write_text("1073741824\n")

    monkeypatch.setattr(dnp, "CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(dnp, "_proc_self_cgroup", lambda: ["0::/scope", "7:memory:/"])
    assert dnp._cgroup_free_bytes() is None


def test_the_unaided_reader_is_never_negative(monkeypatch, dnp, tmp_path):
    _old_zoo(monkeypatch)
    leaf = tmp_path / "scope"
    leaf.mkdir()
    (leaf / "memory.max").write_text("1073741824\n")
    (leaf / "memory.current").write_text("2147483648\n")
    monkeypatch.setattr(dnp, "CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(dnp, "_proc_self_cgroup", lambda: ["0::/scope"])
    assert dnp._cgroup_free_bytes() == 0


def test_the_unaided_reader_keeps_the_public_limit_as_a_last_resort(monkeypatch, dnp, tmp_path):
    """With no readable cgroup tree, a bare limit is still a ceiling tighter than psutil's host view."""
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
    # Keep off the real /sys/fs/cgroup so results do not depend on the runner's container.
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
    assert dnp._cgroup_cpu_quota() is None


def test_env_forced_serial_is_in_process_on_a_small_split(monkeypatch, dnp):
    """UNSLOTH_DATASET_NUM_PROC=0 on a small split must give None: only None runs in-process everywhere."""
    _force_start_method(monkeypatch, dnp, "fork")
    _force_stdlib_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "0")

    small = type("t", (), {"train_dataset": _Split(100)})()
    assert dnp.resolve_responses_only_num_proc(small, 1) is None
    assert dnp.resolve_responses_only_num_proc(small, None) is None

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
    """Fallback stays top-level: unsloth/utils/__init__.py imports torch, and MLX hosts have none."""
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
    """Patch the cached submodule: a raising unsloth_zoo __init__ leaves hf_xet_tuning in sys.modules."""
    module = sys.modules.get("unsloth_zoo.hf_xet_tuning")
    if module is None:
        pytest.skip("unsloth_zoo.hf_xet_tuning is not reachable here")
    assert str(module.CGROUP_ROOT).startswith("/nonexistent"), module.CGROUP_ROOT
    assert module._cgroup_v2_dirs() == []
    assert module._cgroup_v1_dirs("memory") == []


def test_the_unaided_reader_picks_its_own_v1_line_too(monkeypatch, dnp, tmp_path):
    """Must select the memory v1 line, not line 0: a pids line can come first at a different path."""
    _no_hf_xet_tuning(monkeypatch)
    v2_leaf = tmp_path / "user.slice" / "app.scope"
    v2_leaf.mkdir(parents = True)
    (v2_leaf / "memory.max").write_text("8589934592\n")
    (v2_leaf / "memory.current").write_text("6442450944\n")
    # v1: 4GB capped, 3 spent -> 1GB free, below the v2 side's 2GB.
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
    _force_start_method(monkeypatch, dnp, "fork")
    _force_stdlib_start_method(monkeypatch, dnp, "fork")
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "16")

    small = type("t", (), {"train_dataset": _Split(100)})()
    assert dnp.resolve_responses_only_num_proc(small, None) == 16
