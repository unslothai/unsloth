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

The two sizes then differ in what they can be redirected *to*. Unsloth published
``unsloth/Apertus-8B-Instruct-2509``, so the 8B row keeps the full 3-tuple and the
16bit redirect. At 70B only the GGUF and ``-unsloth-bnb-4bit`` repos exist, so the
row is a 1-tuple naming the real upstream, matching the other 70B and 405B rows;
that leaves 16bit loads on upstream instead of sending them to a repo that is not
published.

``mapper.py`` has no imports, so we exec it directly and inspect the built
mappers without importing ``unsloth`` (which requires a GPU).
"""

import os

MAPPER_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models", "mapper.py")

# Never published on the Hub: only the GGUF and -unsloth-bnb-4bit 70B repos exist.
UNPUBLISHED_16BIT = "unsloth/Apertus-70B-Instruct-2509"


def _load_mappers():
    with open(MAPPER_PATH, encoding = "utf-8") as f:
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
        unsloth_4bit = f"unsloth/Apertus-{size}-Instruct-2509-unsloth-bnb-4bit"

        # The genuine instruct upstream must reach the Unsloth instruct 4bit model.
        assert float_to_int.get(instruct_upstream) == unsloth_4bit, instruct_upstream

        # The base upstream must not be redirected to the instruct model.
        assert float_to_int.get(base_upstream) != unsloth_4bit, base_upstream
        assert map_to_16bit.get(base_upstream) != f"unsloth/Apertus-{size}-Instruct-2509", base_upstream

    # 8B has a published Unsloth 16bit repo, so the redirect stays.
    assert map_to_16bit.get("swiss-ai/Apertus-8B-Instruct-2509") == "unsloth/Apertus-8B-Instruct-2509"


def test_no_apertus_lookup_points_at_the_unpublished_70b_16bit_repo():
    namespace = _load_mappers()

    for mapper_name in ("MAP_TO_UNSLOTH_16bit", "INT_TO_FLOAT_MAPPER", "FLOAT_TO_INT_MAPPER"):
        for key, value in namespace[mapper_name].items():
            assert value.lower() != UNPUBLISHED_16BIT.lower(), f"{mapper_name}[{key}]"
