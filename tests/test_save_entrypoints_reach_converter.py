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

"""A save entry point must actually reach its converter.

A helper definition spliced into the middle of `unsloth_save_pretrained_gguf`
left both halves valid Python and the import working, while the function
returned before ever calling `save_to_gguf`. These are the structural
properties that break when a function gets cut in two, checked against the AST
so no unsloth import or GPU is needed.
"""

import ast
from pathlib import Path

import pytest

SAVE_PY = Path(__file__).resolve().parents[1] / "unsloth" / "save.py"


@pytest.fixture(scope = "module")
def tree():
    return ast.parse(SAVE_PY.read_text(encoding = "utf-8"))


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in save.py")


def _calls(node):
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


# the property that broke -------------------------------------------
def test_save_pretrained_gguf_calls_the_converter(tree):
    """The whole purpose of the function. Without this call it is an
    expensive no-op that reports success."""
    assert "save_to_gguf" in _calls(_func(tree, "unsloth_save_pretrained_gguf"))


def test_push_to_hub_gguf_reaches_a_converter(tree):
    """Same hazard, same shape, different entry point."""
    calls = _calls(_func(tree, "unsloth_push_to_hub_gguf"))
    assert calls & {"save_to_gguf", "unsloth_save_pretrained_gguf"}, sorted(calls)


def test_the_gguf_conversion_is_not_dead_code(tree):
    """Reaching the call is not enough -- it has to be reachable.

    In the regression the `save_to_gguf(...)` call still existed in the file,
    which is exactly why grepping for it looked reassuring. It had simply
    landed inside another function after a `return`.
    """
    fn = _func(tree, "unsloth_save_pretrained_gguf")
    assert "save_to_gguf" in _calls(fn)
    for owner in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n is not fn):
        if owner.name == "save_to_gguf":
            continue
        assert not _unreachable_calls(owner, "save_to_gguf"), (
            f"a save_to_gguf call is stranded as dead code inside "
            f"{owner.name} (line {owner.lineno})"
        )


def _unreachable(body):
    """Statements following an unconditional terminator in one block."""
    for i, stmt in enumerate(body):
        if isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
            return body[i + 1 :]
    return []


def _unreachable_calls(fn, name):
    for stmt in _unreachable(fn.body):
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == name:
                return True
    return False


def test_no_function_in_save_py_has_a_stranded_body(tree):
    """The splice signature, checked across the whole module.

    A function whose top-level block continues past an unconditional `return`
    is either dead code or, as here, someone else's body that got spliced in.
    Either way it is worth failing on, and it generalises past this one bug.
    """
    offenders = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        stranded = _unreachable(fn.body)
        # A lone trailing `pass` is this codebase's block terminator idiom.
        stranded = [s for s in stranded if not isinstance(s, ast.Pass)]
        if stranded:
            offenders.append(
                f"{fn.name} (line {fn.lineno} -> dead code at line {stranded[0].lineno})"
            )
    assert not offenders, "stranded function bodies:\n  " + "\n  ".join(offenders)


def test_the_disk_helper_is_a_module_level_function(tree):
    """It was nested by accident once; nesting it again would re-break the
    caller that uses it."""
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert "_gguf_failure_looks_like_disk" in names
    assert "unsloth_save_pretrained_gguf" in names


def test_the_disk_helper_is_actually_used(tree):
    """It exists to narrow the Kaggle disk claim; if nothing calls it, that
    claim has silently gone back to being unconditional."""
    assert "_gguf_failure_looks_like_disk" in _calls(_func(tree, "unsloth_save_pretrained_gguf"))


def test_module_still_parses_and_defines_the_public_entry_points(tree):
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for required in (
        "save_to_gguf",
        "unsloth_save_pretrained_gguf",
        "unsloth_push_to_hub_gguf",
        "unsloth_generic_save",
    ):
        assert required in names, required


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
