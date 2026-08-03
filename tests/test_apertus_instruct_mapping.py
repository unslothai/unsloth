"""Regression test for the Apertus Instruct entries in ``unsloth/models/mapper.py``.

Every 3-tuple in ``__INT_TO_FLOAT_MAPPER`` follows the invariant that the second
element (the original upstream repo) is the same model variant as the key and
the first element, with only the org prefix swapped. The two Apertus Instruct
entries broke it: the key, the Unsloth 16bit name, and the Unsloth 4bit name all
say ``Apertus-...-Instruct-2509``, but the upstream was the *base* repo
``swiss-ai/Apertus-...-2509`` (no ``Instruct``).

The build loop wires that upstream name into both ``FLOAT_TO_INT_MAPPER`` and
``MAP_TO_UNSLOTH_16bit``, so loading the base ``swiss-ai/Apertus-70B-2509`` was
silently redirected to the Unsloth *instruct* model, while the real instruct
upstream ``swiss-ai/Apertus-70B-Instruct-2509`` was registered nowhere and never
got the Unsloth-optimized version.

``mapper.py`` has no imports, so we exec it directly and inspect the built
mappers without importing ``unsloth`` (which requires a GPU).
"""

import os

MAPPER_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models", "mapper.py")


def _load_mappers():
    with open(MAPPER_PATH) as f:
        source = f.read()
    namespace = {}
    exec(compile(source, MAPPER_PATH, "exec"), namespace)
    return namespace


def test_apertus_instruct_upstream_is_the_instruct_repo():
    namespace = _load_mappers()
    map_to_16bit = namespace["MAP_TO_UNSLOTH_16bit"]
    float_to_int = namespace["FLOAT_TO_INT_MAPPER"]

    for size in ("70B", "8B"):
        instruct_upstream = f"swiss-ai/Apertus-{size}-Instruct-2509"
        base_upstream = f"swiss-ai/Apertus-{size}-2509"
        unsloth_16bit = f"unsloth/Apertus-{size}-Instruct-2509"
        unsloth_4bit = f"unsloth/Apertus-{size}-Instruct-2509-unsloth-bnb-4bit"

        # The genuine instruct upstream must map to the Unsloth instruct model.
        assert map_to_16bit.get(instruct_upstream) == unsloth_16bit, instruct_upstream
        assert float_to_int.get(instruct_upstream) == unsloth_4bit, instruct_upstream

        # The base upstream must not be redirected to the instruct model.
        assert map_to_16bit.get(base_upstream) != unsloth_16bit, base_upstream
        assert float_to_int.get(base_upstream) != unsloth_4bit, base_upstream
