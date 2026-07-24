"""Guard against duplicate literal keys in the ``__INT_TO_FLOAT_MAPPER`` model
registry in ``unsloth/models/mapper.py``.

The registry is a hand-maintained dict literal with hundreds of entries. Python
keeps only the *last* value for a duplicated literal key, silently dropping the
earlier one -- with no error at parse, import, or runtime. When the duplicated
key carries a different value this corrupts model resolution everywhere
downstream (see ``test_gemma_2b_mapper_key.py`` for a concrete incident where the
base Gemma-2B repo resolved to the instruct model). Even an *exact* duplicate is
a latent trap: the next edit to one copy silently reintroduces the bug.

This test catches the whole class instead of one model at a time. We parse the
dict literal with ``ast`` rather than executing it so a duplicate is visible in
the *source* (post-exec, the duplicate is already gone), and so we never import
``unsloth`` (which requires a GPU).
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
