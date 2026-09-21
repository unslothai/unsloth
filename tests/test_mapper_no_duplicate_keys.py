# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Guard against duplicate keys in the ``__INT_TO_FLOAT_MAPPER`` registry.

Duplicate keys in the dict literal silently overwrite earlier entries.
We inspect the source with ``ast`` to ensure there are no duplicates.
"""

import ast
import os

MAPPER_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models", "mapper.py")


def _duplicate_int_to_float_keys():
    with open(MAPPER_PATH, encoding = "utf-8") as f:
        tree = ast.parse(f.read(), MAPPER_PATH)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            # Private names are mangled at code generation, which ``ast.parse`` never reaches, so the identifier reads
            # exactly as written.
            if isinstance(target, ast.Name) and target.id == "__INT_TO_FLOAT_MAPPER":
                if not isinstance(node.value, ast.Dict):
                    continue
                # mapper.py reads the nested per-precision dicts directly, so check every dict.
                duplicates = {}
                for mapping in ast.walk(node.value):
                    if not isinstance(mapping, ast.Dict):
                        continue
                    seen = set()
                    for k in mapping.keys:
                        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                            continue
                        if k.value in seen:
                            duplicates.setdefault(k.value, []).append(k.lineno)
                        seen.add(k.value)
                return duplicates
    raise AssertionError("Could not find the __INT_TO_FLOAT_MAPPER dict literal in mapper.py")


def test_int_to_float_mapper_has_no_duplicate_keys():
    duplicates = _duplicate_int_to_float_keys()
    assert not duplicates, (
        "Duplicate keys in __INT_TO_FLOAT_MAPPER silently overwrite earlier "
        "entries and corrupt model resolution. Remove the redundant "
        f"literal(s), key -> line number(s) in mapper.py: {duplicates}"
    )
