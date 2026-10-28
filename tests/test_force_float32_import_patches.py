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
"""The import-time patch pass must not install the UNSLOTH_FORCE_FLOAT32 forwards.

The loader rewrites UNSLOTH_FORCE_FLOAT32 on every load and re-runs the patches in
pre_compile. If import honoured a user "1", a bf16 Gemma3 load (which resolves to "0")
kept the forced float32 embedding next to the generic bf16 attention, and training
died with "Expected query, key, and value to have the same dtype".
"""

import ast
import pathlib

import pytest

_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"


class _Logger:
    def warning(self, message):
        raise AssertionError(message)


def _run_temporary_patches(patches):
    source = _UTILS.read_text(encoding = "utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_temporary_patches":
            segment = ast.get_source_segment(source, node)
            break
    else:
        raise AssertionError("_run_temporary_patches is gone from unsloth/models/_utils.py")
    namespace = {
        "TEMPORARY_PATCHES": patches,
        "logger": _Logger(),
        "TEMPORARY_PATCH_OUTCOMES": {},
    }
    exec(compile(segment, str(_UTILS), "exec"), namespace)
    return namespace["_run_temporary_patches"]


def _recorder(seen):
    import os

    def patch(phase):
        seen.append((phase, os.environ.get("UNSLOTH_FORCE_FLOAT32")))

    return patch


@pytest.mark.parametrize("user_value", ["1", "0", None])
def test_init_pass_runs_unforced_and_restores_the_user_value(monkeypatch, user_value):
    if user_value is None:
        monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", user_value)
    seen = []
    run = _run_temporary_patches([_recorder(seen)])
    run("init")
    assert seen == [("init", "0")]
    import os

    assert os.environ.get("UNSLOTH_FORCE_FLOAT32") == user_value


@pytest.mark.parametrize("loader_value", ["1", "0"])
def test_compile_passes_follow_the_loader_decision(monkeypatch, loader_value):
    # The loader sets the variable before unsloth_compile_transformers; those passes must see it as is.
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", loader_value)
    seen = []
    run = _run_temporary_patches([_recorder(seen)])
    run("pre_compile")
    run("post_compile")
    assert seen == [("pre_compile", loader_value), ("post_compile", loader_value)]
