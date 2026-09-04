# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the IPython pre_run_cell hook in docker/unsloth_nb_compat.py.

The hook only read the pip shim's marker file, which is a record of a PREVIOUS cell.
The notebooks pin a new model by installing and importing in ONE cell, and the shim
writes that marker from a child process partway through it, so the hook had already
returned: the cell ran the base transformers and every later cell was answered with
"already imported; cannot switch". The hook had no test coverage at all.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPAT_PATH = REPO_ROOT / "docker" / "unsloth_nb_compat.py"
RUN_PATH = REPO_ROOT / "docker" / "unsloth_run.py"

PIN = "5.10.2"
SIDECAR = "t_5_10_2"


@pytest.fixture()
def sidecar_root(tmp_path):
    root = tmp_path / "tf-sidecars"
    for name in ("t_5_5_0", SIDECAR):
        (root / name).mkdir(parents = True)
    (root / ".vllm_min_transformers").write_text("5.5.0\n")
    return root


@pytest.fixture()
def compat(sidecar_root, tmp_path, monkeypatch):
    """Fresh compat over fake sidecars, with sys.path and PYTHONPATH restored.

    activate() mutates both, so without the teardown one test's sidecar leaks into the
    next and a later assertion passes for the wrong reason.
    """
    monkeypatch.setenv("UNSLOTH_TF_SIDECAR_ROOT", str(sidecar_root))
    monkeypatch.delenv("UNSLOTH_TF_SIDECAR_MIN", raising = False)
    monkeypatch.setenv("UNSLOTH_NB_TF_MARKER", str(tmp_path / "marker" / "requested"))
    monkeypatch.setenv("PYTHONPATH", "")
    # activate() is a no-op once transformers is imported, and the session may well
    # have imported it before this file runs, which made the first test in the file
    # fail while the rest passed on the teardown of the one before it
    monkeypatch.delitem(sys.modules, "transformers", raising = False)
    path_before = list(sys.path)

    spec = importlib.util.spec_from_file_location("unsloth_nb_compat_under_test", COMPAT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._sidecar_dir = str(sidecar_root / SIDECAR)
    try:
        yield mod
    finally:
        sys.path[:] = path_before


def _fire(compat, source):
    """Run the hook the way IPython does and report the activated directory."""
    compat._pre_run_cell(SimpleNamespace(raw_cell = source))
    return compat._sidecar_dir if compat._sidecar_dir in sys.path else None


COMBINED_CELL = (
    "# Muse Glimmer needs transformers 5.10.2. Run before anything imports it\n"
    f'!pip install -q "transformers=={PIN}"\n'
    "\n"
    "import transformers\n"
    'print("transformers:", transformers.__version__)\n'
)


def test_a_combined_install_and_import_cell_gets_its_sidecar(compat):
    assert _fire(compat, COMBINED_CELL) == compat._sidecar_dir


def test_the_marker_only_hook_would_have_missed_it(compat):
    """Non-vacuity. Reproduces the OLD hook body against the same cell: if this ever
    activates, the test above proves nothing about the change."""
    v = compat.requested_version()
    assert v is None, "no previous cell has installed anything yet"
    if v and "transformers" not in sys.modules:
        compat.activate(v)
    assert compat._sidecar_dir not in sys.path


def test_the_marker_is_used_when_the_cell_pins_nothing_itself(compat, tmp_path):
    marker = Path(os.environ["UNSLOTH_NB_TF_MARKER"])
    marker.parent.mkdir(parents = True, exist_ok = True)
    marker.write_text(PIN)
    # a cell with no install at all, i.e. the shape the hook always handled
    assert _fire(compat, "import transformers\n") == compat._sidecar_dir


def test_the_cell_pin_outranks_a_stale_marker(compat, sidecar_root, tmp_path):
    """The marker records an install that has ALREADY run. An earlier cell in the same
    notebook can have pinned something else, and the marker path falls back to
    pid-<pid> when the connection file cannot be read, so a recycled pid inherits a
    stranger's pin. The cell about to run is the better authority either way."""
    marker = Path(os.environ["UNSLOTH_NB_TF_MARKER"])
    marker.parent.mkdir(parents = True, exist_ok = True)
    marker.write_text("5.5.0")
    other = str(sidecar_root / "t_5_5_0")

    compat._pre_run_cell(SimpleNamespace(raw_cell = COMBINED_CELL))

    assert compat._sidecar_dir in sys.path, "the cell's own pin was not honoured"
    assert other not in sys.path, "the stale marker won over the cell about to run"


@pytest.mark.parametrize(
    "cell",
    [
        pytest.param(f'# !pip install "transformers=={PIN}"\n', id = "commented-out"),
        pytest.param(f'"""\n!pip install transformers=={PIN}\n"""\n', id = "docstring"),
        pytest.param(f'note = "pip install transformers=={PIN}"\n', id = "string-literal"),
        pytest.param(f'print("upgrade from transformers=={PIN} if it breaks")\n', id = "mention"),
        pytest.param(f"REQUIRED = 'transformers=={PIN}'\n", id = "assigned-unused"),
        pytest.param("import transformers\n", id = "no-pin-at-all"),
    ],
)
def test_a_cell_that_installs_nothing_activates_nothing(compat, cell):
    """A wrong sidecar is unrecoverable once transformers is imported, while no sidecar
    only means the base venv, so anything ambiguous has to lose."""
    assert _fire(compat, cell) is None


@pytest.mark.parametrize(
    "cell",
    [
        pytest.param(f'!pip install "transformers=={PIN}"\n', id = "bang-pip"),
        pytest.param(f"%pip install transformers=={PIN}\n", id = "pip-magic"),
        pytest.param(f"!uv pip install -q transformers=={PIN}\n", id = "uv"),
        pytest.param(f"!python -m pip install transformers=={PIN}\n", id = "python-m-pip"),
        pytest.param(f"!pip install \\\n    transformers=={PIN}\n", id = "continuation"),
        pytest.param(f"  !pip install transformers=={PIN}\n", id = "indented"),
    ],
)
def test_every_real_install_shape_activates(compat, cell):
    assert _fire(compat, cell) == compat._sidecar_dir


def test_nothing_happens_once_transformers_is_imported(compat, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__ = "4.57.6"))
    compat._pre_run_cell(SimpleNamespace(raw_cell = COMBINED_CELL))
    assert compat._sidecar_dir not in sys.path
    # the hook fires on EVERY cell, so it must not narrate on each one
    assert capsys.readouterr().err == ""


def test_a_hook_call_with_no_info_is_harmless(compat):
    """IPython has passed the event object for a long time, but the parameter was
    optional here before and a bare call must not raise."""
    compat._pre_run_cell()
    compat._pre_run_cell(SimpleNamespace())


def test_pythonpath_does_not_accumulate_a_copy_per_cell(compat):
    for _ in range(3):
        compat.activate(PIN)
    entries = os.environ["PYTHONPATH"].split(os.pathsep)
    assert entries.count(compat._sidecar_dir) == 1, os.environ["PYTHONPATH"]


def test_the_hook_is_registered_under_pre_run_cell(compat):
    class _Events:
        def __init__(self):
            self.registered = []

        def register(self, name, fn):
            self.registered.append((name, fn))

    class _Shell:
        def __init__(self):
            self.events = _Events()

    shell = _Shell()
    compat.get_ipython = lambda: shell
    compat.register_ipython()
    compat.register_ipython()  # idempotent
    assert shell.events.registered == [("pre_run_cell", compat._pre_run_cell)]


def test_ipython_really_hands_the_hook_the_cell_source():
    """Premise pin: the fix reads info.raw_cell, so IPython has to supply it."""
    ipython = pytest.importorskip("IPython.core.interactiveshell")
    seen = []
    shell = ipython.InteractiveShell.instance()
    try:
        shell.events.register("pre_run_cell", lambda info: seen.append(info))
        shell.run_cell("x = 1\n")
    finally:
        ipython.InteractiveShell.clear_instance()
    assert seen, "pre_run_cell did not fire"
    assert getattr(seen[0], "raw_cell", None) == "x = 1\n"


def test_unsloth_run_scans_with_the_very_same_functions(compat, monkeypatch):
    """One scanner, not two. unsloth_run used to own these, and a copy in each module
    would let the headless kernel and the in-notebook hook pick different sidecars."""
    monkeypatch.setitem(sys.modules, "unsloth_nb_compat", compat)
    spec = importlib.util.spec_from_file_location("unsloth_run_same_scanner", RUN_PATH)
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    assert run.compat is compat
    for name in ("_PIN_RE", "_INSTALL_RE", "_strip_comment", "_live_source", "_install_lines"):
        assert getattr(run, name) is getattr(compat, name), name
    assert run._pin_from is compat.pin_from


# pip accepts every PEP 503 spelling of a requirement name, and unsloth_pip_shim
# canonicalises before it decides what to drop. A scanner that only knew the canonical
# form therefore left the pin unseen while the install was still suppressed, and the
# import in that same cell froze the base transformers for the life of the kernel.


@pytest.mark.parametrize(
    "spec",
    [
        "transformers==" + PIN,
        "Transformers==" + PIN,
        "TRANSFORMERS==" + PIN,
        "transformers[torch]==" + PIN,
        "transformers [torch] == " + PIN,
        "  transformers  ==  " + PIN,
    ],
)
def test_every_spelling_the_shim_drops_is_a_pin(compat, spec):
    assert compat.pin_in_cell("%pip install " + spec + "\nimport transformers\n") == PIN
    assert _fire(compat, "%pip install " + spec + "\nimport transformers\n") is not None


def test_the_shim_and_the_scanner_agree_on_the_name(compat):
    """The two halves this item found apart. Read the shim's own canonicaliser rather
    than restating its rule here, so a change on that side fails this instead of
    drifting silently."""
    shim_path = REPO_ROOT / "docker" / "unsloth_pip_shim.py"
    spec = importlib.util.spec_from_file_location("unsloth_pip_shim_names", shim_path)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    for spelling in ("transformers", "Transformers", "TRANSFORMERS", "transformers[torch]"):
        assert shim._canon(spelling + "==" + PIN) == "transformers", spelling
        assert compat.pin_in_cell("!pip install " + spelling + "==" + PIN) == PIN, spelling
    # and where the shim says a spelling is somebody else, so does the scanner: PEP 503
    # collapses a run of `-_.` to one `-`, it does not delete it
    for other in ("trans_formers", "trans.formers", "sentence-transformers"):
        assert shim._canon(other + "==" + PIN) != "transformers", other
        assert compat._norm_req(other) == shim._canon(other + "==" + PIN), other
        assert compat.pin_in_cell("!pip install " + other + "==" + PIN) is None, other


@pytest.mark.parametrize(
    "spec",
    [
        "transformers-stream-generator==" + PIN,
        "sentence-transformers==" + PIN,
        "trans-formers==" + PIN,
    ],
)
def test_a_different_distribution_is_not_the_pin(compat, spec):
    """Non-vacuity and the blast radius: matching the name loosely must not start
    reading somebody else's version as the transformers pin."""
    assert compat.pin_in_cell("%pip install " + spec) is None


def test_a_pin_still_has_to_come_from_an_install_line(compat):
    assert compat.pin_in_cell("# %pip install Transformers==" + PIN) is None
    assert compat.pin_in_cell('doc = """\n%pip install Transformers==' + PIN + '\n"""\n') is None
