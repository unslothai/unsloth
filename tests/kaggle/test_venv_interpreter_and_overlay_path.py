# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Two defects that presented as one, and neither names this driver in its
traceback.

Measured over six sessions from ONE commit, deterministic per arm:

| arm | Default leg python | torch | datasets | verdict |
|---|---|---|---|---|
| A (3 runs) | 3.12.13 | 2.10.0+cu128 | 4.3.0 | pass 3/3 |
| B (3 runs) | **3.13.13** | 2.12.1+cu130 | 5.0.1 | **fail 3/3** |

Every other leg in both arms stayed on 3.12.13.

**Defect one: the venv interpreter was never pinned.** uv's default
python-preference is `managed`, so once any managed CPython exists under
``~/.local/share/uv/python`` a bare ``uv venv`` builds on that instead of the
Kaggle image's interpreter. ``--system-site-packages`` is still accepted and
still inherits nothing, because a 3.13 venv cannot see a 3.12 site-packages. The
leg then resolves torch, datasets and pyarrow from PyPI -- a stack no Kaggle
user has -- and pays minutes for the privilege, with nothing red.

**Defect two: the overlay directory was hostile to dill**, which is what turned
the wrong interpreter into a crash. ``dill._dill._is_builtin_module`` pickles a
module by REFERENCE only if its ``__file__`` starts with a sys prefix, ends with
an extension suffix, or contains the literal string ``site-packages``. A plain
``pip install --target /tmp/t4ci_venvs/overlay_X`` satisfies none of the three,
so everything in the overlay is pickled BY VALUE. ``Dataset.from_dict``
fingerprints through dill, ``datasets/utils/_dill.py:_save_arrowTable`` saves
``create_arrowTable``, dill walks its globals into the pyarrow module and dies
on pyarrow's Cython ``MonthDayNano``, whose ``__module__`` is ``builtins``:

    PicklingError: Can't pickle <class 'MonthDayNano'>:
    it's not found as builtins.MonthDayNano

Reproduced on CPU in seconds against a byte-identical package tree with the
DIRECTORY NAME as the only variable: the plain ``--target`` directory raised,
the copy named ``site-packages`` returned a fingerprint.

The rules below are asserted through the GENERATED driver source and, for the
dill half, by executing the real path expression against the real dill
predicate. A rule written against the template would not have caught either.
"""

from __future__ import annotations

import pathlib
import re
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SMOKE_DIR = ROOT / "tests" / "kaggle" / "t4_smoke"
sys.path.insert(0, str(ROOT / ".github" / "scripts"))

from kaggle_t4_ci import build_kernel, legs  # noqa: E402


def _driver_source() -> str:
    """The driver as the kernel will actually run it, overlays included."""
    names = ("default", "canary")
    payloads = {
        f"t4_{legs.LEGS[n].name}.ipynb": build_kernel.build_payload_notebook(
            SMOKE_DIR, legs.LEGS[n], unsloth_ref = "main", zoo_ref = "main"
        )
        for n in names
    }
    driver = build_kernel.build_driver(
        payloads,
        per_run_timeout = 3600,
        vram_source = {f"t4_{legs.LEGS[n].name}.ipynb": legs.LEGS[n] for n in names},
        overlays = {f"t4_{legs.LEGS[n].name}.ipynb": tuple(legs.LEGS[n].overlay) for n in names},
    )
    return "".join("".join(c["source"]) for c in driver["cells"])


# --------------------------------------------------------------------------
# Defect one: the interpreter
# --------------------------------------------------------------------------


def test_the_venv_is_built_on_the_drivers_own_interpreter():
    """`--system-site-packages` on a different python version is a silent
    no-op, so the flag alone is not the guard. The interpreter is."""
    source = _driver_source()
    body = source.split("def _make_venv(")[1].split("def run_one(")[0]
    cmd = re.search(r"venv_cmd = \[([^\]]*)\]", body)
    assert cmd, "the uv venv command is no longer a literal list; re-read this rule"
    assert '"--python"' in cmd.group(1) and "sys.executable" in cmd.group(1), (
        "uv venv runs without --python, so uv's `managed` python-preference "
        "decides the version. That is how the Default leg came up on 3.13 "
        "against a 3.12 image and inherited no system site-packages at all: "
        f"{cmd.group(1)}"
    )


def test_the_system_site_flag_is_still_passed():
    """The pin does not replace it. Both are needed: the right interpreter with
    no inheritance resolves the whole stack from PyPI just as expensively."""
    body = _driver_source().split("def _make_venv(")[1].split("def run_one(")[0]
    assert '"--system-site-packages"' in body


def test_the_resulting_python_version_is_REPORTED():
    """Three sessions went red before anyone read a python version out of a
    leg. A venv on the wrong interpreter is invisible without this."""
    body = _driver_source().split("def _make_venv(")[1].split("def run_one(")[0]
    sentinel = body.split("_VENV ")[1]
    assert '"python"' in sentinel, (
        "the _VENV record does not carry the venv's python version, so the "
        "next wrong-interpreter run is again only visible by inference"
    )
    assert (
        "sys.version_info" in body
    ), "nothing compares the venv against the driver's own interpreter"


# --------------------------------------------------------------------------
# Defect two: the overlay directory, driven against the real dill predicate
# --------------------------------------------------------------------------


def _overlay_dir_from_generated_source() -> pathlib.Path:
    """Evaluate the driver's OWN `_ov_dir` expression.

    Not a hand-written copy of the path: the rule below is only worth having if
    it fails when the generated expression changes.
    """
    source = _driver_source()
    line = [ln.strip() for ln in source.splitlines() if ln.strip().startswith("_ov_dir = ")]
    assert len(line) == 1, f"expected one _ov_dir assignment, found {len(line)}"
    scope = {
        "VENV_ROOT": pathlib.Path("/tmp/t4ci_venvs"),
        "pathlib": pathlib,
        "name": "t4_Default.ipynb",
    }
    exec(line[0], scope)  # noqa: S102 - the driver's own line, by design
    return scope["_ov_dir"]


def test_the_overlay_lands_where_dill_will_pickle_by_reference():
    """The real dill predicate, against the real generated path.

    `_is_builtin_module` needs only `__file__`, so a module object standing in
    for pyarrow is enough to execute the decision that crashed the leg.
    """
    dill_dill = pytest.importorskip("dill._dill")
    ov = _overlay_dir_from_generated_source()

    stand_in = types.ModuleType("pyarrow")
    stand_in.__file__ = str(ov / "pyarrow" / "__init__.py")
    assert dill_dill._is_builtin_module(stand_in), (
        f"a package installed at {ov} is pickled BY VALUE by dill, so any leg "
        "whose overlay carries pyarrow dies in Dataset.from_dict with "
        "PicklingError: Can't pickle <class 'MonthDayNano'>"
    )


def test_the_mutation_this_rule_exists_for_is_actually_caught():
    """The negative control. Without it the rule above could be passing because
    dill answers True for everything, which it does not."""
    dill_dill = pytest.importorskip("dill._dill")
    before = types.ModuleType("pyarrow")
    # The path the driver used until this was found: a --target directory that
    # is neither under a sys prefix nor named site-packages.
    before.__file__ = "/tmp/t4ci_venvs/overlay_t4_Default/pyarrow/__init__.py"
    assert not dill_dill._is_builtin_module(before), (
        "dill no longer distinguishes the two paths, so the fix above may have "
        "become unnecessary -- re-measure before simplifying it away"
    )


def test_the_path_is_not_merely_a_string_containing_site_packages():
    """`site-packages` has to be the directory packages are installed INTO, not
    a decoration further up the path, or pip lays the tree somewhere dill still
    reads by value."""
    ov = _overlay_dir_from_generated_source()
    assert ov.name == "site-packages", (
        f"the overlay --target is {ov}, whose last component is {ov.name!r}; "
        "dill's check is on the installed module's own __file__"
    )


def test_the_overlay_target_flag_uses_that_directory():
    """The path can be right and unused. This is the link between them."""
    source = _driver_source()
    install = source.split('"--target", str(_ov_dir)')
    assert len(install) == 2, (
        "the overlay install no longer targets _ov_dir, so the directory the "
        "rules above check is not where packages land"
    )
    assert (
        'env["PYTHONPATH"] = str(_ov_dir)' in source
    ), "the overlay is installed somewhere the payload never reads"
