# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""`backfill_fp8_modules_to_convert`: unsloth/Qwen3.8-Flash-Next-FP8's config omits the n-gram table from
`modules_to_convert`, so transformers 5.21 loaded it as raw fp8 and dropped its `weight_scale`."""

import ast, os, re, types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = open(os.path.join(HERE, "unsloth", "models", "loader_utils.py"), encoding = "utf-8").read()

# The real index keys of unsloth/Qwen3.8-Flash-Next-FP8, trimmed: one per-tensor scale, block scales elsewhere.
TABLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
WEIGHT_MAP = {
    f"{TABLE}.weight": "a",
    f"{TABLE}.weight_scale": "a",
    "model.language_model.layers.1.ple.key_proj.weight": "a",
    "model.language_model.layers.0.self_attn.q_proj.weight": "b",
    "model.language_model.layers.0.self_attn.q_proj.weight_scale_inv": "b",
}


def _backfill(weight_map):
    node = next(
        n
        for n in ast.parse(SRC).body
        if isinstance(n, ast.FunctionDef) and n.name == "backfill_fp8_modules_to_convert"
    )
    ns = {"re": re, "_load_fp8_weight_map": lambda *a, **k: weight_map}
    exec(ast.get_source_segment(SRC, node), ns)
    return ns["backfill_fp8_modules_to_convert"]


def _config(**quant):
    return types.SimpleNamespace(quantization_config = {"quant_method": "fp8", **quant})


def test_adds_the_table_the_unsloth_repo_left_out():
    config = _config(modules_to_not_convert = ["lm_head"])
    found = ["layers.1.ple.ple_embedding.ngram_embedding"]
    assert _backfill(WEIGHT_MAP)(config, "unsloth/Qwen3.8-Flash-Next-FP8") == found
    assert config.quantization_config["modules_to_convert"] == found
    # The quantizer matches by suffix: the runtime name must match, another layer's table must not.
    assert "model.language_model.layers.1.ple.ple_embedding.ngram_embedding".endswith(found[0])
    assert not "model.language_model.layers.11.ple.ple_embedding.ngram_embedding".endswith(found[0])


def test_mixed_precision_tables_keep_their_layer():
    weight_map = {
        "model.layers.0.table.weight": "a",
        "model.layers.0.table.weight_scale": "a",
        "model.layers.1.table.weight": "a",
        "model.embed_fp8.weight_scale": "a",
    }
    assert _backfill(weight_map)(_config(), "x") == ["layers.0.table", "model.embed_fp8"]


def test_leaves_configs_that_already_list_it_or_are_not_fp8_alone():
    listed = _config(modules_to_convert = ["ple.ple_embedding.ngram_embedding"])
    assert _backfill(WEIGHT_MAP)(listed, "Qwen/Qwen3.8-Flash-Next-FP8") is None
    bnb = types.SimpleNamespace(quantization_config = {"quant_method": "bitsandbytes"})
    assert _backfill(WEIGHT_MAP)(bnb, "x") is None
    assert "modules_to_convert" not in bnb.quantization_config
    assert _backfill(WEIGHT_MAP)(types.SimpleNamespace(), "x") is None


def test_block_fp8_only_and_skipped_tables_add_nothing():
    block_only = {k: v for k, v in WEIGHT_MAP.items() if not k.endswith(".weight_scale")}
    config = _config()
    assert (
        _backfill(block_only)(config, "x") is None
        and "modules_to_convert" not in config.quantization_config
    )
    skipped = _config(modules_to_not_convert = [TABLE])
    assert _backfill(WEIGHT_MAP)(skipped, "x") is None


def test_forwards_the_checkpoint_locator_options():
    calls = []
    node = next(
        n
        for n in ast.parse(SRC).body
        if isinstance(n, ast.FunctionDef) and n.name == "backfill_fp8_modules_to_convert"
    )
    ns = {"re": re, "_load_fp8_weight_map": lambda *a, **k: calls.append((a, k)) or WEIGHT_MAP}
    exec(ast.get_source_segment(SRC, node), ns)
    ns["backfill_fp8_modules_to_convert"](
        _config(), "repo", "tok", "rev", subfolder = "sub", cache_dir = "/c", variant = "fp8"
    )
    # subfolder / cache_dir / variant pick which index and shards are probed.
    ((args, kwargs),) = calls
    assert args == ("repo", False, "tok", "rev", "sub", "/c") and kwargs == {
        "variant": "fp8",
        "index_only": True,
    }


def test_index_only_never_downloads_a_single_file(monkeypatch):
    import sys

    node = next(
        n
        for n in ast.parse(SRC).body
        if isinstance(n, ast.FunctionDef) and n.name == "_load_fp8_weight_map"
    )
    ns = {"os": os}
    exec(ast.get_source_segment(SRC, node), ns)
    fetched = []

    def hf_hub_download(repo, filename, **k):
        fetched.append(filename)
        raise FileNotFoundError(filename)  # no index on this repo

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download = hf_hub_download)
    )
    assert ns["_load_fp8_weight_map"]("org/single-file-fp8", False, None, index_only = True) is None
    assert fetched == ["model.safetensors.index.json"]


def test_both_loaders_backfill_before_loading():
    # FastLanguageModel sends llama/mistral/gemma/qwen2/qwen3 through llama.py, everything else through vision.py.
    for name in ("llama.py", "vision.py"):
        src = open(os.path.join(HERE, "unsloth", "models", name), encoding = "utf-8").read()
        assert "backfill_fp8_modules_to_convert(" in src, name
        call = src.index("backfill_fp8_modules_to_convert(", src.index("def from_pretrained"))
        assert call < src.index(".from_pretrained(", call), name


def test_llama_backfill_reaches_the_load():
    # The plain llama.py load re-reads config.json unless it passes config=; a backfill must force that branch.
    src = open(os.path.join(HERE, "unsloth", "models", "llama.py"), encoding = "utf-8").read()
    assert "_fp8_backfilled = backfill_fp8_modules_to_convert(" in src
    routing = src[src.index("or _block_swap_saved is not None") :]
    routing = routing[: routing.index("):")]
    assert "or _fp8_backfilled" in routing
