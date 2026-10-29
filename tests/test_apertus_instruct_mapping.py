"""Each 3-tuple's upstream must be the key's own variant; Apertus Instruct pointed at the base repo."""

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

        assert float_to_int.get(instruct_upstream) == unsloth_4bit, instruct_upstream

        assert float_to_int.get(base_upstream) != unsloth_4bit, base_upstream
        assert (
            map_to_16bit.get(base_upstream) != f"unsloth/Apertus-{size}-Instruct-2509"
        ), base_upstream

    assert (
        map_to_16bit.get("swiss-ai/Apertus-8B-Instruct-2509") == "unsloth/Apertus-8B-Instruct-2509"
    )


def test_no_apertus_lookup_points_at_the_unpublished_70b_16bit_repo():
    namespace = _load_mappers()

    for mapper_name in ("MAP_TO_UNSLOTH_16bit", "INT_TO_FLOAT_MAPPER", "FLOAT_TO_INT_MAPPER"):
        for key, value in namespace[mapper_name].items():
            assert value.lower() != UNPUBLISHED_16BIT.lower(), f"{mapper_name}[{key}]"
