# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Guard against instruct/base drift in the ``__INT_TO_FLOAT_MAPPER`` registry.

Slot 1 of each entry names the upstream repo the unsloth build mirrors, and
``mapper.py`` turns it into two lookups: ``FLOAT_TO_INT_MAPPER[upstream] -> key``
and ``MAP_TO_UNSLOTH_16bit[upstream] -> slot 0``. So an instruct entry whose
upstream slot names the *base* repo silently hands anyone loading that base repo
the instruct weights instead, and leaves the real instruct repo unmapped.

Upstream cards make this easy to get wrong: an instruct model's ``base_model``
tag points at the pretrained base, so copying it into slot 1 looks right.

We inspect the source with ``ast`` so this needs no imports and no network.
"""

import ast
import os
import re

MAPPER_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models", "mapper.py")

# The suffixes upstreams use to mark a chat/instruction-tuned build.
_INSTRUCT_TAG = re.compile(r"(?i)(-it\b|-it-|instruct|-chat\b|-chat-)")


def _int_to_float_rows():
    """Yield (line number, key, upstream repo) for every entry with an upstream."""
    with open(MAPPER_PATH, encoding = "utf-8") as f:
        tree = ast.parse(f.read(), MAPPER_PATH)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        # Private names are mangled at code generation, which ``ast.parse``
        # never reaches, so the identifier reads exactly as written.
        if not any(
            isinstance(t, ast.Name) and t.id == "__INT_TO_FLOAT_MAPPER" for t in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue

        rows = []
        for key, value in zip(node.value.keys, node.value.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            # Entries are either a plain tuple or a per-precision dict whose
            # "16" tuple carries the 16bit names.
            tuples = []
            if isinstance(value, ast.Dict):
                for k, v in zip(value.keys, value.values):
                    if (
                        isinstance(k, ast.Constant)
                        and k.value == "16"
                        and isinstance(v, (ast.Tuple, ast.List))
                    ):
                        tuples.append(v)
            elif isinstance(value, (ast.Tuple, ast.List)):
                tuples.append(value)

            for entry in tuples:
                names = [
                    e.value
                    for e in entry.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
                if len(names) >= 2:
                    rows.append((entry.lineno, key.value, names[1]))
        return rows
    raise AssertionError("Could not find the __INT_TO_FLOAT_MAPPER dict literal in mapper.py")


def test_instruct_entries_do_not_point_at_a_base_repo():
    drifted = [
        f"line {lineno}: {key} -> {upstream}"
        for lineno, key, upstream in _int_to_float_rows()
        if _INSTRUCT_TAG.search(key) and not _INSTRUCT_TAG.search(upstream)
    ]
    assert not drifted, (
        "An instruct entry whose upstream slot names the base repo makes "
        "`FastLanguageModel.from_pretrained(<base repo>)` load the instruct "
        "weights, and leaves the instruct repo itself unmapped. Point slot 1 at "
        f"the instruct repo, in mapper.py: {drifted}"
    )


def test_base_entries_do_not_point_at_an_instruct_repo():
    drifted = [
        f"line {lineno}: {key} -> {upstream}"
        for lineno, key, upstream in _int_to_float_rows()
        if not _INSTRUCT_TAG.search(key) and _INSTRUCT_TAG.search(upstream)
    ]
    assert not drifted, (
        "A base entry whose upstream slot names the instruct repo swaps the two "
        "the other way round, so a request for the instruct model resolves to "
        f"base weights, in mapper.py: {drifted}"
    )
