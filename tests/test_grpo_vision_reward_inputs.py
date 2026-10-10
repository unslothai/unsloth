# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The GRPO reward-call rewrite must work on TRL's VLM tool-image branch (no prompts_text) and on TRL 1.15
(no completions_text), while vision prompts keep getting decoded text as before."""

from __future__ import annotations

import ast
import os
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SOURCE = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _replacement():
    tree = ast.parse(open(SOURCE, encoding = "utf-8").read())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "replacement_string" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and "_calculate_rewards(inputs, prompts_text" in node.value.value
        ):
            return node.value.value
    raise AssertionError("reward replacement not found")


class _Proc:
    def batch_decode(
        self,
        ids,
        skip_special_tokens = True,
    ):
        return [f"decoded:{i}" for i in ids]


class _Self:
    processing_class = _Proc()

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        return prompts, completions


def _run(body_prefix):
    src = (
        "def f(self, inputs, prompts, completions, completion_ids, completion_ids_list, images):\n"
        + textwrap.indent(textwrap.dedent(body_prefix), "    ")
        + textwrap.indent(textwrap.dedent(_replacement()), "    ")
        + "\n    return rewards_per_func\n"
    )
    ns = {}
    exec(src, ns)
    return ns["f"]


def test_vision_prompts_get_text_trl_113():
    f = _run("prompts_text = ['p']\ncompletions_text = ['c']\n")
    assert f(_Self(), [], ["msgs"], ["cmsgs"], [1], [[1]], [["img"]]) == (["p"], ["c"])


def test_vision_prompts_get_text_trl_115_without_completions_text():
    f = _run("prompts_text = ['p']\n")
    assert f(_Self(), [], ["msgs"], ["cmsgs"], [7], [[7]], [["img"]]) == (["p"], ["decoded:7"])


def test_tool_image_branch_keeps_messages():
    f = _run("pass\n")
    assert f(_Self(), [], ["msgs"], ["cmsgs"], [7], [[7]], [["tool_img"]]) == (["msgs"], ["cmsgs"])


def test_text_only_keeps_messages():
    f = _run("prompts_text = ['p']\n")
    assert f(_Self(), [], ["msgs"], ["cmsgs"], [7], [[7]], None) == (["msgs"], ["cmsgs"])
