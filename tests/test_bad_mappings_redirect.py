"""Regression test for BAD_MAPPINGS redirecting oversized dynamic quants.

get_model_name previously applied BAD_MAPPINGS only to the resolver's output,
but several listed names (the `-unsloth-bnb-4bit` dynamic quants, plus any name
the resolver doesn't map) come back as None, so their BAD_MAPPINGS entries were
dead and the oversized model loaded. Asserting over every entry catches all of
them. The mapper table and the resolver have no heavy imports of their own,
so we exec the import-free mapper module and ast-extract the resolver functions
rather than importing unsloth (which needs a GPU).
"""

import ast
import os

_MODELS = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models")


def _load_get_model_name():
    mapper_ns = {}
    with open(os.path.join(_MODELS, "mapper.py"), encoding = "utf-8") as f:
        exec(compile(f.read(), "mapper.py", "exec"), mapper_ns)

    with open(os.path.join(_MODELS, "loader_utils.py"), encoding = "utf-8") as f:
        tree = ast.parse(f.read())

    namespace = dict(mapper_ns)
    namespace["SUPPORTS_FOURBIT"] = True
    namespace["_env_says_offline"] = lambda: True
    namespace["_get_new_mapper"] = lambda: ({}, {}, {})

    wanted = {"__get_model_name", "_resolve_with_mappers", "get_model_name"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "BAD_MAPPINGS" for target in node.targets
        ):
            exec(compile(ast.Module([node], []), "<bad_mappings>", "exec"), namespace)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), node.name, "exec"), namespace)

    return namespace["get_model_name"], namespace["BAD_MAPPINGS"]


def test_bad_mappings_redirect_every_listed_name():
    get_model_name, bad_mappings = _load_get_model_name()
    assert bad_mappings, "BAD_MAPPINGS should not be empty"
    for name, expected in bad_mappings.items():
        assert get_model_name(name, load_in_4bit = True) == expected, name
