"""BAD_MAPPINGS must cover names the resolver maps to None, or oversized dynamic quants still load."""

import ast
import os

import pytest

_MODELS = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models")


_ENTRY = (
    "__get_model_name",
    "_resolve_with_mappers",
    "_prefer_legacy_lowercase_cache",
    "get_model_name",
)


def _module_defs(tree):
    """Top-level functions and plain assignments of loader_utils, by name (imports stay out)."""
    defs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defs[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defs[target.id] = node
    return defs


def _closure(defs, provided):
    """Every top-level function or assignment the entry points reach by name, minus what the test provides.
    A helper or constant a later change adds to get_model_name is picked up here instead of raising NameError."""
    seen, todo = set(), list(_ENTRY)
    while todo:
        name = todo.pop()
        if name in seen or name in provided or name not in defs:
            continue
        seen.add(name)
        todo.extend(n.id for n in ast.walk(defs[name]) if isinstance(n, ast.Name))
    return seen


def _load_get_model_name():
    mapper_ns = {}
    with open(os.path.join(_MODELS, "mapper.py"), encoding = "utf-8") as f:
        exec(compile(f.read(), "mapper.py", "exec"), mapper_ns)

    with open(os.path.join(_MODELS, "loader_utils.py"), encoding = "utf-8") as f:
        tree = ast.parse(f.read())

    namespace = dict(mapper_ns)
    # Environment probes: online, a current unsloth_zoo, nothing to fetch from the Hub.
    namespace["SUPPORTS_FOURBIT"] = True
    namespace["_env_says_offline"] = lambda: False
    namespace["_get_new_mapper"] = lambda: ({}, {}, {}, {}, {})
    namespace["_zoo_sets_bare_4bit_compute_dtype"] = lambda: True
    namespace["_complete_in_cache"] = lambda repo_id, cache_dir = None: True

    defs = _module_defs(tree)
    wanted = _closure(defs, namespace)
    for (
        node
    ) in tree.body:  # source order, so a constant exists before the code that reads it at import
        names = (
            {node.name}
            if isinstance(node, ast.FunctionDef)
            else {t.id for t in getattr(node, "targets", ()) if isinstance(t, ast.Name)}
        )
        if names & wanted:
            try:
                exec(compile(ast.Module([node], []), "<loader_utils>", "exec"), namespace)
            except NameError:
                # An assignment computed from an import (transformers_version = Version(...)); the
                # entry points only read it on paths these tests do not take.
                if isinstance(node, ast.FunctionDef):
                    raise

    return namespace["get_model_name"], namespace["BAD_MAPPINGS"]


def test_harness_extracts_every_constant_get_model_name_reads():
    """The NameError that broke these tests when _BARE_4BIT_EXPERT_REPOS was added (#19313) cannot recur."""
    with open(os.path.join(_MODELS, "loader_utils.py"), encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    wanted = _closure(
        _module_defs(tree), {"SUPPORTS_FOURBIT", "_env_says_offline", "_get_new_mapper"}
    )
    assert {"BAD_MAPPINGS", "_BARE_4BIT_EXPERT_REPOS"} <= wanted, sorted(wanted)


def test_bad_mappings_redirect_every_listed_name():
    get_model_name, bad_mappings = _load_get_model_name()
    assert bad_mappings, "BAD_MAPPINGS should not be empty"
    for name, expected in bad_mappings.items():
        assert get_model_name(name, load_in_4bit = True) == expected, name


def test_bad_mapping_redirects_keep_canonical_repo_case():
    get_model_name, _ = _load_get_model_name()
    assert get_model_name("Qwen/Qwen3-32B", load_in_4bit = True) == "unsloth/Qwen3-32B-bnb-4bit"
    assert get_model_name("Qwen/Qwen3-30B-A3B", load_in_4bit = True) == "unsloth/Qwen3-30B-A3B"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("upstage/SOLAR-10.7B-v1.0", "upstage/SOLAR-10.7B-v1.0"),
        ("allenai/Llama-3.1-Tulu-3-8B", "unsloth/Llama-3.1-Tulu-3-8B"),
        ("unsloth/Llama-3.1-Tulu-3-8B", "unsloth/Llama-3.1-Tulu-3-8B"),
        ("allenai/Llama-3.1-Tulu-3-70B", "allenai/Llama-3.1-Tulu-3-70B"),
        ("NousResearch/Hermes-3-Llama-3.1-405B", "NousResearch/Hermes-3-Llama-3.1-405B"),
        ("open-thoughts/OpenThinker-7B", "open-thoughts/OpenThinker-7B"),
        ("mistralai/Pixtral-12B-Base-2409", "unsloth/Pixtral-12B-Base-2409-bnb-4bit"),
    ],
)
def test_4bit_load_avoids_unpublished_bnb_repos(name, expected):
    get_model_name, _ = _load_get_model_name()
    assert get_model_name(name, load_in_4bit = True) == expected
