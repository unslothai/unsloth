# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""`unsloth-run` must take its transformers pin from an install, not from prose.

Headless selection is `want = --transformers or install-cell pin or model tier`
(docker/unsloth_run.py), and the winner is prepended to the kernel's PYTHONPATH
as a sidecar before a single cell runs. `_scan` used to regex the ENTIRE source
of every code cell, so any text shaped like `transformers==X` won -- including a
commented-out line or a string literal, which install nothing. Commenting out an
install line is a routine notebook edit, and the damage is not hypothetical:

    # !pip install --no-deps transformers==4.57.6      <- stale, does not run
    ...
    FastModel.from_pretrained("unsloth/gemma-4-12b-it")

pin 4.57.6 beats tier 5.10.2, and 4.57.6 is below the vLLM floor so it clamps up
to the lowest eligible sidecar, 5.5.0. gemma4-unified landed in transformers
5.10.1, so the kernel comes up one tier short of the model it is about to load --
and because no install actually runs, the pip shim never gets to correct either
the marker or PYTHONPATH.

The fix must not narrow the scan too far: 430 of the 561 shipped notebooks carry
a pin, many of them on the CONTINUATION line of a multi-line `!uv pip install \\`
or indented inside the `if "COLAB_" not in ...` guard. Those shapes are pinned
below verbatim.

No docker, no GPU, no network.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = REPO_ROOT / "docker" / "unsloth_run.py"


def _load_run(sidecar_root):
    """Import a fresh unsloth_run (and its compat) bound to a synthetic root."""
    prev = {k: os.environ.get(k) for k in ("UNSLOTH_TF_SIDECAR_ROOT", "UNSLOTH_TF_SIDECAR_MIN")}
    os.environ["UNSLOTH_TF_SIDECAR_ROOT"] = str(sidecar_root)
    os.environ.pop("UNSLOTH_TF_SIDECAR_MIN", None)
    stale = sys.modules.pop("unsloth_nb_compat", None)
    try:
        spec = importlib.util.spec_from_file_location("unsloth_run_under_test", RUN_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("unsloth_nb_compat", None)
        if stale is not None:
            sys.modules["unsloth_nb_compat"] = stale
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod


@pytest.fixture()
def sidecar_root(tmp_path):
    """What the image actually ships: vLLM 0.26.0 drops 4.57.6 and 5.3.0, so the
    surviving sidecars are 5.5.0 and 5.10.2 and the recorded floor is 5.5.0."""
    root = tmp_path / "tf-sidecars"
    for name in ("t_5_5_0", "t_5_10_2"):
        (root / name).mkdir(parents = True)
    (root / ".vllm_min_transformers").write_text("5.5.0\n")
    return root


@pytest.fixture()
def run_mod(sidecar_root):
    return _load_run(sidecar_root)


def _nb(*sources):
    return {
        "cells": [
            {
                "cell_type": "code",
                "source": s,
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
            for s in sources
        ],
        "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Py"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


GEMMA4_12B = 'model, tok = FastModel.from_pretrained("unsloth/gemma-4-12b-it")\n'


# --------------------------------------------------------------------------
# Prose must never be mistaken for an install request.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell",
    [
        pytest.param(
            "# !pip install --no-deps transformers==4.57.6\n!pip install unsloth\n",
            id = "commented-out-install",
        ),
        pytest.param(
            "!pip install unsloth  # was transformers==4.57.6 before the 5.x bump\n",
            id = "trailing-comment",
        ),
        pytest.param(
            '"""Colab used to need transformers==4.57.6 here."""\n!pip install unsloth\n',
            id = "docstring",
        ),
        pytest.param(
            'print("upgrade from transformers==4.57.6 if you hit an import error")\n',
            id = "string-literal",
        ),
        pytest.param(
            "_legacy = 'transformers==4.57.6'  # no longer applied\n!pip install unsloth\n",
            id = "assigned-but-unused",
        ),
        pytest.param(
            "   # !uv pip install transformers==4.57.6\n!pip install unsloth\n",
            id = "indented-comment",
        ),
    ],
)
def test_a_mention_that_installs_nothing_is_not_a_pin(run_mod, cell):
    pin, model = run_mod._scan(_nb(cell, GEMMA4_12B))
    assert pin is None, f"{pin!r} came from text that never runs an install"
    assert model == "unsloth/gemma-4-12b-it"


# --------------------------------------------------------------------------
# ...and every real install shape must still be seen.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell, expected",
    [
        pytest.param(
            '!pip install --no-deps transformers==5.10.1 "tokenizers>=0.22.0"\n',
            "5.10.1",
            id = "bang-pip",
        ),
        pytest.param("%pip install transformers==5.5.0\n", "5.5.0", id = "percent-pip"),
        pytest.param("!pip3 install transformers==5.5.0\n", "5.5.0", id = "pip3"),
        pytest.param('!uv pip install --system -qqq "transformers==5.2.0"\n', "5.2.0", id = "uv-pip"),
        pytest.param("!python -m pip install transformers==5.3.0\n", "5.3.0", id = "python-m-pip"),
        pytest.param(
            "!{sys.executable} -m pip install transformers==5.3.0\n", "5.3.0", id = "sys-executable"
        ),
        pytest.param("!pip -q install transformers==5.5.0\n", "5.5.0", id = "opt-before-install"),
        pytest.param("pip install transformers==5.5.0\n", "5.5.0", id = "bare-shell-cell"),
        pytest.param(
            # Granite4.0.ipynb / the gpt-oss family: the pin lives on a
            # backslash continuation, several lines below the invocation.
            "!uv pip install -qqq \\\n"
            '    {_torch} "triton>=3.3.0" {_numpy} torchvision bitsandbytes "transformers==4.56.2" \\\n'
            '    "unsloth[base] @ git+https://github.com/unslothai/unsloth"\n',
            "4.56.2",
            id = "backslash-continuation",
        ),
        pytest.param(
            # Gemma4_(12B)_Text.ipynb: installs indented inside the Colab guard.
            "%%capture\n"
            "import os\n"
            'if "COLAB_" not in "".join(os.environ.keys()):\n'
            "    !pip install unsloth\n"
            "else:\n"
            "    !pip install --no-deps transformers==5.10.1\n",
            "5.10.1",
            id = "indented-inside-guard",
        ),
    ],
)
def test_real_install_shapes_still_yield_their_pin(run_mod, cell, expected):
    assert run_mod._scan(_nb(cell))[0] == expected


def test_an_install_still_outranks_the_model_tier(run_mod):
    # The pin is the notebook's own statement of what it was built against, so a
    # REAL install must keep winning; only prose stops counting.
    pin, model = run_mod._scan(_nb("!pip install transformers==5.5.0\n", GEMMA4_12B))
    assert (pin, model) == ("5.5.0", "unsloth/gemma-4-12b-it")


# --------------------------------------------------------------------------
# End to end: what the kernel is actually launched with.
# --------------------------------------------------------------------------
def _launch(run_mod, monkeypatch, tmp_path, nb, name):
    """Run main() up to the kernel launch and return what the kernel would see.

    The per-run marker is a temp file main() deletes on the way out, so read it
    at launch -- which is the only moment it matters."""
    src = tmp_path / f"{name}.ipynb"
    src.write_text(json.dumps(nb))
    seen = {}

    def fake_call(
        cmd,
        env = None,
        **kwargs,
    ):
        seen["cmd"] = cmd
        seen["env"] = dict(env or {})
        marker = seen["env"].get("UNSLOTH_NB_TF_MARKER")
        seen["marker"] = Path(marker).read_text().strip() if marker else None
        return 0

    # Swap the module's own handle, not the shared subprocess module.
    monkeypatch.setattr(run_mod, "subprocess", SimpleNamespace(call = fake_call))
    monkeypatch.setattr(sys, "argv", ["unsloth-run", str(src)])
    monkeypatch.delenv("UNSLOTH_NB_TF_MARKER", raising = False)
    with pytest.raises(SystemExit) as exc:
        run_mod.main()
    assert exc.value.code == 0
    return seen


def _sidecars_on_path(env):
    return [
        Path(p).name
        for p in env.get("PYTHONPATH", "").split(os.pathsep)
        if Path(p).name.startswith("t_")
    ]


def test_a_stale_commented_pin_does_not_downgrade_the_kernel(run_mod, monkeypatch, tmp_path):
    nb = _nb(
        "# legacy Colab workaround, no longer needed:\n"
        "# !pip install --no-deps transformers==4.57.6\n"
        "!pip install unsloth\n",
        GEMMA4_12B,
    )
    seen = _launch(run_mod, monkeypatch, tmp_path, nb, "stale")
    assert _sidecars_on_path(seen["env"]) == ["t_5_10_2"], (
        "the dead comment must not outrank the gemma-4-12b tier; 5.5.0 predates "
        "gemma4-unified, which landed in transformers 5.10.1"
    )
    assert seen["marker"] == "5.10.2"


def test_a_clean_notebook_is_unaffected(run_mod, monkeypatch, tmp_path):
    seen = _launch(
        run_mod, monkeypatch, tmp_path, _nb("!pip install unsloth\n", GEMMA4_12B), "clean"
    )
    assert _sidecars_on_path(seen["env"]) == ["t_5_10_2"]
    assert seen["marker"] == "5.10.2"


def test_a_real_pin_still_drives_the_kernel(run_mod, monkeypatch, tmp_path):
    nb = _nb("!pip install --no-deps transformers==5.10.1\n", GEMMA4_12B)
    seen = _launch(run_mod, monkeypatch, tmp_path, nb, "pinned")
    assert _sidecars_on_path(seen["env"]) == ["t_5_10_2"]
    assert seen["marker"] == "5.10.1"
