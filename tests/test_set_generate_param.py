"""GPU-free regression coverage for _set_generate_param, the helper that avoids
transformers' "Passing `generation_config` together with generation-related
arguments" deprecation warning (GenerationMixin._prepare_generation_config)
when unsloth_base_fast_generate needs to set a field that is also a recognized
GenerationConfig attribute (pad_token_id, cache_implementation, compile_config),
AST-extracted so no unsloth/CUDA import is needed."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace


VISION_PATH = Path(__file__).parents[1] / "unsloth" / "models" / "vision.py"


def _load_function(name, namespace):
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body = [function], type_ignores = []), str(VISION_PATH), "exec"), namespace)
    return namespace[name]


set_generate_param = _load_function("_set_generate_param", {"copy": copy})


def test_no_generation_config_sets_a_plain_kwarg():
    kwargs = {}
    set_generate_param(kwargs, "pad_token_id", 99)
    assert kwargs == {"pad_token_id": 99}


def test_no_generation_config_overwrite_false_still_sets_the_kwarg():
    # overwrite only governs behavior against an existing generation_config value;
    # with no config at all there is nothing to preserve, so the kwarg is set either way.
    kwargs = {"pad_token_id": 7}
    set_generate_param(kwargs, "pad_token_id", 99, overwrite = False)
    assert kwargs == {"pad_token_id": 99}


def test_config_present_unset_field_gets_filled_in():
    generation_config = SimpleNamespace(pad_token_id = None)
    kwargs = {"generation_config": generation_config}
    set_generate_param(kwargs, "pad_token_id", 99, overwrite = False)
    assert "pad_token_id" not in kwargs
    assert kwargs["generation_config"].pad_token_id == 99


def test_config_present_already_set_field_is_preserved_when_not_overwriting():
    generation_config = SimpleNamespace(pad_token_id = 7)
    kwargs = {"generation_config": generation_config}
    set_generate_param(kwargs, "pad_token_id", 99, overwrite = False)
    assert "pad_token_id" not in kwargs
    assert kwargs["generation_config"].pad_token_id == 7


def test_config_present_default_overwrite_forces_the_value():
    generation_config = SimpleNamespace(cache_implementation = "static")
    kwargs = {"generation_config": generation_config}
    set_generate_param(kwargs, "cache_implementation", "dynamic")
    assert "cache_implementation" not in kwargs
    assert kwargs["generation_config"].cache_implementation == "dynamic"


def test_no_leftover_kwarg_is_ever_left_alongside_an_explicit_config():
    # This is the actual bug being fixed: transformers flags any GenerationConfig
    # field present both on an explicit generation_config and as a raw kwarg. Neither
    # overwrite mode may ever leave one behind when a config was passed.
    for overwrite in (True, False):
        generation_config = SimpleNamespace(compile_config = None)
        kwargs = {"generation_config": generation_config, "compile_config": "stale"}
        set_generate_param(kwargs, "compile_config", "computed", overwrite = overwrite)
        assert "compile_config" not in kwargs


def test_a_raw_kwarg_next_to_a_config_still_wins_and_moves_onto_it():
    # transformers applies kwargs after the config merge, so the caller's value won before.
    generation_config = SimpleNamespace(pad_token_id = 7)
    kwargs = {"generation_config": generation_config, "pad_token_id": 99}
    set_generate_param(kwargs, "pad_token_id", kwargs.get("pad_token_id", 0), overwrite = False)
    assert "pad_token_id" not in kwargs
    assert kwargs["generation_config"].pad_token_id == 99

    generation_config = SimpleNamespace(cache_implementation = None, compile_config = None)
    kwargs = {
        "generation_config": generation_config,
        "cache_implementation": "offloaded",
        "compile_config": "mine",
    }
    set_generate_param(kwargs, "cache_implementation", "static")
    set_generate_param(kwargs, "compile_config", "computed")
    assert list(kwargs) == ["generation_config"]
    assert kwargs["generation_config"].cache_implementation == "offloaded"
    assert kwargs["generation_config"].compile_config == "mine"


def test_an_explicit_none_next_to_a_config_still_wins():
    # GenerationConfig.update writes None too, so cache_implementation=None used to disable the cache.
    generation_config = SimpleNamespace(cache_implementation = "static")
    kwargs = {"generation_config": generation_config, "cache_implementation": None}
    set_generate_param(kwargs, "cache_implementation", "static")
    assert "cache_implementation" not in kwargs
    assert kwargs["generation_config"].cache_implementation is None


def test_a_forced_value_beats_a_raw_kwarg_next_to_a_config():
    generation_config = SimpleNamespace(cache_implementation = "static")
    kwargs = {"generation_config": generation_config, "cache_implementation": "static"}
    set_generate_param(kwargs, "cache_implementation", "dynamic", force = True)
    assert "cache_implementation" not in kwargs
    assert kwargs["generation_config"].cache_implementation == "dynamic"


def test_the_callers_config_is_never_mutated():
    # TRL reuses one generation_config for every rollout; a value set for one call must not leak into the next.
    generation_config = SimpleNamespace(pad_token_id = None, cache_implementation = "static")
    kwargs = {"generation_config": generation_config, "pad_token_id": 11}
    set_generate_param(kwargs, "pad_token_id", kwargs.get("pad_token_id", 0), overwrite = False)
    set_generate_param(kwargs, "cache_implementation", "dynamic", force = True)
    assert kwargs["generation_config"] is not generation_config
    assert kwargs["generation_config"].pad_token_id == 11
    assert generation_config.pad_token_id is None
    assert generation_config.cache_implementation == "static"


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"OK: {len(tests)} _set_generate_param regression tests passed")
