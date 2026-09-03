# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The notebook-view paths must survive being written into a Python config.

`studio_launch.sh` appends three settings to `jupyter_lab_config.py` built from
`UNSLOTH_NOTEBOOKS_VIEW_DIR`. Interpolated straight into a heredoc's string
literals, a path containing a double quote closed the literal and made the
config a SyntaxError, so the documented override stopped JupyterLab from
starting at all; a backslash silently produced a different path, since Python
reads `\\t` in a literal as a tab. Both characters are legal in a POSIX path.

The generator block is extracted from the shipped script and run for real, then
the result is compiled and executed the way Jupyter loads it, so the assertions
are about what Jupyter would actually see rather than about the text.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH = REPO_ROOT / "docker" / "studio_launch.sh"

behavioural = pytest.mark.skipif(
    shutil.which("bash") is None, reason = "needs bash"
)


@pytest.fixture(scope = "module")
def generator() -> str:
    """The `python - >> ... <<'PY' ... PY` block, verbatim from the script."""
    source = LAUNCH.read_text(encoding = "utf-8")
    match = re.search(
        r"^\s*UNSLOTH_VIEW_REL=.*?\n\s*python - >> .*?<<'PY'\n(.*?)\nPY$",
        source,
        re.S | re.M,
    )
    assert match, "the notebook-view config generator disappeared or changed shape"
    return match.group(1)


def _render(generator: str, rel: str, view: str) -> str:
    result = subprocess.run(
        [sys.executable, "-"],
        input = generator,
        capture_output = True,
        text = True,
        timeout = 120,
        env = dict(os.environ, UNSLOTH_VIEW_REL = rel, UNSLOTH_VIEW_DIR = view),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _load(config_text: str):
    """Execute the config the way Jupyter does, and hand back what it set."""

    class _Node(types.SimpleNamespace):
        pass

    config = types.SimpleNamespace(
        ServerApp = _Node(), LabApp = _Node(), PasswordIdentityProvider = _Node()
    )
    exec(compile(config_text, "jupyter_lab_config.py", "exec"), {"c": config})
    return config


NASTY = [
    pytest.param('My "Special" Notebooks', id = "double-quote"),
    pytest.param("Note\\tbooks", id = "backslash-t"),
    pytest.param("Note'books", id = "single-quote"),
    pytest.param("Note\\books", id = "trailing-backslash-segment"),
    pytest.param("Unsloth Notebooks", id = "the-default"),
]


@pytest.mark.parametrize("name", NASTY)
def test_the_config_stays_valid_python_and_keeps_the_path(generator: str, name: str):
    view = f"/workspace/{name}"
    rendered = _render(generator, name, view)

    config = _load(rendered)  # a SyntaxError here is the bug

    assert config.ServerApp.preferred_dir == view, (
        "the path Jupyter ends up with must be the one the user asked for"
    )
    assert config.ServerApp.default_url == f"/lab/tree/{name}"
    assert config.LabApp.default_url == config.ServerApp.default_url, (
        "LabApp otherwise overrides ServerApp back to /lab"
    )


def test_a_newline_in_the_path_cannot_inject_a_config_line(generator: str):
    # A newline would end the statement outright and let the rest of the value
    # be read as configuration.
    name = 'x"\nc.ServerApp.token = "pwned'
    config = _load(_render(generator, name, f"/workspace/{name}"))

    assert config.ServerApp.preferred_dir == f"/workspace/{name}"
    assert not hasattr(config.ServerApp, "token"), "the value was executed as config"


def test_the_generator_does_not_interpolate_the_paths_in_the_shell(generator: str):
    # The heredoc delimiter has to stay quoted and the values have to arrive
    # through the environment; an unquoted heredoc puts the shell's expansion
    # back in front of python's quoting and the fix is undone.
    source = LAUNCH.read_text(encoding = "utf-8")
    assert "python - >> \"${JUPYTER_CONFIG_DIR}/jupyter_lab_config.py\" <<'PY'" in source
    assert "${_view_rel}" not in generator and "${_view_dir}" not in generator
    assert 'os.environ["UNSLOTH_VIEW_REL"]' in generator
    assert 'os.environ["UNSLOTH_VIEW_DIR"]' in generator


@behavioural
def test_the_heredoc_form_this_replaced_really_was_broken(tmp_path: Path):
    """Pin the premise, so the test above is not guarding a hypothetical.

    Reproduces the previous construct exactly: an unquoted heredoc interpolating
    the value into a Python string literal. With a double quote in the path the
    generated config does not parse, which is JupyterLab failing to start.
    """
    name = 'My "Special" Notebooks'
    script = tmp_path / "old.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        '_view_rel="$1"\n'
        '_view_dir="$2"\n'
        "cat <<EOF\n"
        'c.ServerApp.default_url = "/lab/tree/${_view_rel}"\n'
        'c.LabApp.default_url = "/lab/tree/${_view_rel}"\n'
        'c.ServerApp.preferred_dir = "${_view_dir}"\n'
        "EOF\n",
        encoding = "utf-8",
    )
    result = subprocess.run(
        ["bash", str(script), name, f"/workspace/{name}"],
        capture_output = True, text = True, timeout = 120,
    )
    assert result.returncode == 0, result.stderr

    with pytest.raises(SyntaxError):
        compile(result.stdout, "jupyter_lab_config.py", "exec")
