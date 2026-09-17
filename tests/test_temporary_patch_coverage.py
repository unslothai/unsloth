# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""A temporary patch that stops applying has to be caught, not just logged.

`_run_temporary_patches` calls each entry of `TEMPORARY_PATCHES` inside
`except Exception` and emits `logger.warning`, so `import unsloth` survives a
patch that blows up (#3130 ended the import outright on a `SyntaxError` from
one). The cost of that is a silent one: a patch can rot away against a new
transformers and every job stays green, because the only trace is a warning
line nobody asserts on.

The gate here reads the outcome that `_run_temporary_patches` now records and
fails on any patch that RAISED.

Why not assert a list of patch names, or a count. Which patches apply depends
entirely on what is installed. `patch_gemma4_moe` has nothing to attach itself
to on transformers 4.57.6 and returns without doing anything; the MoE
quantization patches check `is_transformers_v5_moe_quantization_available()`
first and decline on v4. A name list or a count would go red on the 4.57.6 leg
of the matrix for patches that are behaving exactly as designed, and would need
editing every time a patch is added. So the question asked is not "did patch X
apply" but "did any patch raise", which is version independent: declining
cleanly is a normal return, raising never is.
"""

import ast
import gc
import json
import os
import pathlib
import subprocess
import sys
import weakref

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_UTILS = _ROOT / "unsloth" / "models" / "_utils.py"

_BEGIN = "UNSLOTH_PATCH_REPORT_BEGIN"
_END = "UNSLOTH_PATCH_REPORT_END"

# Imports unsloth for real, then dumps what `_run_temporary_patches` recorded.
# The injection point is where a mutation control installs an extra patch and
# re-runs the pass, so the control travels the same code path as the gate.
_CHILD = """
import json
import unsloth  # noqa: F401
import unsloth.models._utils as _utils

{injection}

report = {{}}
for phase, outcome in _utils.TEMPORARY_PATCH_OUTCOMES.items():
    report[phase] = {{
        "completed": [getattr(p, "__name__", repr(p)) for p in outcome["completed"]],
        "raised": [
            [getattr(p, "__name__", repr(p)), type(e).__name__, str(e)]
            for p, e in outcome["raised"]
        ],
    }}
print({begin!r})
print(json.dumps(report))
print({end!r})
"""

_NO_ACCELERATOR = (
    "Unsloth cannot find any torch accelerator",
    "No CUDA GPUs are available",
    "Torch not compiled with CUDA enabled",
)


def _spawn(code, **extra_env):
    path = [str(_ROOT)]
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        env = dict(os.environ, PYTHONPATH = os.pathsep.join(path), **extra_env),
        timeout = 1800,
    )


def _collect(injection = ""):
    """The real import, in a fresh interpreter, reported as plain data.

    A host with no accelerator at all cannot finish `import unsloth` without
    `UNSLOTH_ALLOW_CPU=1`, and CI's CPU job is exactly that. Retrying on that
    one symptom rather than probing the host keeps this free of any assumption
    about which accelerator (or operating system) is present.
    """
    code = _CHILD.format(injection = injection, begin = _BEGIN, end = _END)
    result = _spawn(code)
    if result.returncode != 0 and any(m in result.stderr for m in _NO_ACCELERATOR):
        result = _spawn(code, UNSLOTH_ALLOW_CPU = "1")
    if result.returncode != 0 or _BEGIN not in result.stdout:
        raise AssertionError(
            "could not import unsloth to read the temporary patch outcomes\n"
            f"exit {result.returncode}\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    body = result.stdout.split(_BEGIN, 1)[1].split(_END, 1)[0]
    return json.loads(body)


def _assert_no_patch_raised(report):
    """The gate. Fails naming every patch that raised, and with what."""
    failures = []
    for phase in sorted(report):
        for name, exception_type, message in report[phase]["raised"]:
            failures.append(f"  {name} (phase {phase}): {exception_type}: {message}")
    assert not failures, (
        "PATCH COVERAGE LOST: these temporary patches raised and were swallowed by the "
        "fail-soft handler in _run_temporary_patches, so they are no longer applied. A "
        "patch that no longer has anything to patch on this library version must return "
        "cleanly, not raise.\n" + "\n".join(failures)
    )


@pytest.fixture(scope = "module")
def _clean_report():
    return _collect()


# ------------------------------------------------------------------ the gate


def test_no_temporary_patch_raised_on_this_library_version(_clean_report):
    assert _clean_report, "no phase was recorded at all, the bookkeeping is not running"
    assert "init" in _clean_report, "the import-time pass did not record an outcome"
    _assert_no_patch_raised(_clean_report)


def test_the_import_pass_actually_ran_some_patches(_clean_report):
    # Guards the degenerate green: an empty TEMPORARY_PATCHES would satisfy the
    # gate above without applying anything. No upper or exact bound, and no
    # names, so adding or removing a patch never touches this file.
    assert _clean_report["init"][
        "completed"
    ], "no temporary patch completed during import, so the gate above is vacuous"


# ------------------------------------------------- mutation controls
# The gate is only worth having if it is sensitive to a patch that breaks and
# insensitive to one that declines because this library version has nothing for
# it to patch. Both controls go through the real import and the real recording.


_RAISES = """
def _mutation_control_patch_that_raises():
    # What patch_merge_quantization_configs did in #3130: a SyntaxError out of
    # an exec'd string, from inside a patch body.
    raise SyntaxError("mutation control: this patch is broken")

_utils.TEMPORARY_PATCHES.append(_mutation_control_patch_that_raises)
_utils._run_temporary_patches("init")
"""

_DECLINES = """
def _mutation_control_patch_that_declines(phase):
    # The shape of every version-conditional patch in unsloth_zoo: look for the
    # thing to patch, and return without touching anything when this library
    # version does not have it.
    try:
        import a_module_no_transformers_release_ships  # noqa: F401
    except ImportError:
        return
    raise AssertionError("unreachable")

_utils.TEMPORARY_PATCHES.append(_mutation_control_patch_that_declines)
_utils._run_temporary_patches("init")
"""


def test_mutation_control_a_raising_patch_turns_the_gate_red():
    report = _collect(injection = _RAISES)
    with pytest.raises(AssertionError) as caught:
        _assert_no_patch_raised(report)
    message = str(caught.value)
    assert "_mutation_control_patch_that_raises" in message, message
    assert "SyntaxError" in message, message


def test_mutation_control_a_cleanly_declining_patch_does_not():
    report = _collect(injection = _DECLINES)
    _assert_no_patch_raised(report)
    assert (
        "_mutation_control_patch_that_declines" in report["init"]["completed"]
    ), "a patch that declined cleanly was not recorded as completed"


# ---------------------------------------------- the recording itself
# Fast, no import of unsloth: the function is loaded out of the file the way
# tests/test_import_time_floors.py does, so these run everywhere including a
# host that cannot import the model stack at all.


class _CollectingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


def _isolated(patches, logger, outcomes):
    source = _UTILS.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_temporary_patches":
            segment = ast.get_source_segment(source, node)
            break
    else:
        raise AssertionError("_run_temporary_patches is gone from unsloth/models/_utils.py")
    namespace = {
        "TEMPORARY_PATCHES": patches,
        "logger": logger,
        "TEMPORARY_PATCH_OUTCOMES": outcomes,
    }
    exec(compile(segment, str(_UTILS), "exec"), namespace)
    return namespace["_run_temporary_patches"]


def test_outcomes_separate_raising_from_returning():
    def declines():
        return None

    def explodes():
        raise RuntimeError("boom")

    outcomes = {}
    logger = _CollectingLogger()
    _isolated([declines, explodes], logger, outcomes)("init")

    assert [p.__name__ for p in outcomes["init"]["completed"]] == ["declines"]
    assert [p.__name__ for p, _ in outcomes["init"]["raised"]] == ["explodes"]
    assert len(logger.warnings) == 1, "fail-soft logging must be unchanged"


def test_outcomes_are_recorded_per_phase():
    def declines(phase):
        return None

    outcomes = {}
    run = _isolated([declines], _CollectingLogger(), outcomes)
    run("init")
    run("pre_compile")

    assert sorted(outcomes) == ["init", "pre_compile"]


def test_a_repeated_pass_replaces_rather_than_grows_its_phase():
    # `_run_temporary_patches` runs again on every model load, so an appending
    # record would be a leak in a long-lived process.
    def declines():
        return None

    outcomes = {}
    run = _isolated([declines], _CollectingLogger(), outcomes)
    for _ in range(5):
        run("pre_compile")

    assert len(outcomes["pre_compile"]["completed"]) == 1


def test_a_repeated_pass_releases_what_the_previous_failure_was_holding():
    # The count check above uses a patch that returns cleanly, which is the
    # cheap half of the question. The half that can actually cost memory is a
    # patch that RAISES: `raised` stores the exception object, the exception
    # carries its __traceback__, and the traceback keeps the raising frame and
    # every local in it alive. Measured: the frame's local is still reachable
    # for as long as the phase entry lives, and is released the moment the next
    # pass of that phase replaces the entry. So the bound on this record is not
    # just its length, it is that one pass never keeps the previous pass's
    # frames. Pinned here because a record that accumulated history would still
    # satisfy every other test in this file.
    class _Held:
        pass

    def explodes():
        heavy = _Held()  # noqa: F841  the local the traceback pins alive
        raise RuntimeError("boom")

    outcomes = {}
    run = _isolated([explodes], _CollectingLogger(), outcomes)
    run("pre_compile")

    _, exception = outcomes["pre_compile"]["raised"][0]
    traceback = exception.__traceback__
    assert traceback is not None, "the stored exception lost its traceback"
    frame = (traceback.tb_next or traceback).tb_frame
    held = weakref.ref(frame.f_locals["heavy"])
    del exception, traceback, frame
    assert held() is not None, "the probe never had a live reference to hold"

    run("pre_compile")
    gc.collect()
    assert held() is None, (
        "a second pass did not release the frames the previous failure was holding, so "
        "TEMPORARY_PATCH_OUTCOMES retains one traceback per failing model load"
    )


def test_an_unnamed_callable_does_not_break_the_recording():
    # The success path stores the callable itself and formats nothing, so a
    # callable with no __name__ cannot make the bookkeeping raise inside
    # `import unsloth`.
    class _Callable:
        def __call__(self):
            return None

    outcomes = {}
    logger = _CollectingLogger()
    _isolated([_Callable()], logger, outcomes)("init")

    assert len(outcomes["init"]["completed"]) == 1
    assert logger.warnings == []


def test_the_recording_is_wired_into_the_patch_loop():
    # DRIFT: the gate is only as good as its wiring, and a refactor that drops
    # these lines would leave every test above green while recording nothing.
    source = _UTILS.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_temporary_patches":
            segment = ast.get_source_segment(source, node)
            break
    else:
        raise AssertionError("_run_temporary_patches is gone from unsloth/models/_utils.py")

    assert "TEMPORARY_PATCH_OUTCOMES[phase]" in segment
    assert "raised.append(" in segment
    assert "completed.append(" in segment
    assert (
        "TEMPORARY_PATCH_OUTCOMES = {}" in source
    ), "the module level record the gate reads is gone"
