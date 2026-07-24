"""Guard against duplicate keys in the ``__INT_TO_FLOAT_MAPPER`` registry.

Duplicate keys in the dict literal silently overwrite earlier entries.
We inspect the source with ``ast`` to ensure there are no duplicates.
"""

import ast
import os
from collections import Counter

MAPPER_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models", "mapper.py")


def _duplicate_int_to_float_keys():
    with open(MAPPER_PATH) as f:
        tree = ast.parse(f.read(), MAPPER_PATH)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            # The name is mangled at class scope, but at module scope it is the
            # plain ``__INT_TO_FLOAT_MAPPER`` identifier.
            if isinstance(target, ast.Name) and target.id == "__INT_TO_FLOAT_MAPPER":
                if not isinstance(node.value, ast.Dict):
                    continue
                keys = [
                    k.value
                    for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                ]
                counts = Counter(keys)
                return {key: n for key, n in counts.items() if n > 1}
    raise AssertionError("Could not find the __INT_TO_FLOAT_MAPPER dict literal in mapper.py")


def test_int_to_float_mapper_has_no_duplicate_keys():
    duplicates = _duplicate_int_to_float_keys()
    assert not duplicates, (
        "Duplicate keys in __INT_TO_FLOAT_MAPPER silently overwrite earlier "
        "entries and corrupt model resolution. Remove the redundant "
        f"literal(s): {duplicates}"
    )
