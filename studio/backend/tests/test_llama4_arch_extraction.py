from types import SimpleNamespace

from utils.hardware.vram_estimation import (
    _compute_dense_layer_indices,
    _dense_mlp_size,
    _shared_expert_size,
    extract_arch_config,
)


def _llama4_text_config(**fields):
    base = dict(
        hidden_size = 2048,
        num_hidden_layers = 4,
        num_attention_heads = 16,
        num_key_value_heads = 4,
        intermediate_size = 8192,
        intermediate_size_mlp = 16384,
        vocab_size = 32000,
        tie_word_embeddings = True,
        num_local_experts = 4,
        num_experts_per_tok = 2,
    )
    base.update(fields)
    return SimpleNamespace(**base)


def test_llama4_moe_layers_dispatch_uses_explicit_indices():
    cfg = SimpleNamespace(num_hidden_layers = 4, moe_layers = [1, 3])
    assert _compute_dense_layer_indices(cfg, 4) == (0, 2)


def test_llama4_moe_layers_takes_priority_over_first_k_dense_replace():
    cfg = SimpleNamespace(
        num_hidden_layers = 6,
        moe_layers = [2, 4],
        first_k_dense_replace = 4,
    )
    assert _compute_dense_layer_indices(cfg, 6) == (0, 1, 3, 5)


def test_llama4_arch_extraction_picks_up_dense_intermediate_size_mlp():
    arch = extract_arch_config(_llama4_text_config(moe_layers = [1, 3]))
    assert arch is not None
    assert arch.intermediate_size == 8192
    assert arch.dense_intermediate_size == 16384
    assert _dense_mlp_size(arch) == 16384


def test_llama4_arch_extraction_auto_attaches_one_shared_expert():
    arch = extract_arch_config(_llama4_text_config(moe_layers = [1, 3]))
    assert arch is not None
    assert arch.n_shared_experts == 1
    # why: shared expert is built as Llama4TextMLP(config) at intermediate_size,
    # so _shared_expert_size falls back to the routed expert width.
    assert arch.shared_expert_intermediate_size is None
    assert _shared_expert_size(arch) == arch.intermediate_size


def test_llama4_dense_indices_drives_num_dense_layers():
    arch = extract_arch_config(_llama4_text_config(moe_layers = [1, 3]))
    assert arch is not None
    assert arch.dense_layer_indices == (0, 2)
    assert arch.num_dense_layers == 2


def test_non_llama4_config_leaves_dense_intermediate_size_none():
    cfg = SimpleNamespace(
        hidden_size = 1024,
        num_hidden_layers = 4,
        num_attention_heads = 8,
        num_key_value_heads = 2,
        intermediate_size = 4096,
        vocab_size = 32000,
        tie_word_embeddings = True,
    )
    arch = extract_arch_config(cfg)
    assert arch is not None
    assert arch.dense_intermediate_size is None
    # _dense_mlp_size falls back to intermediate_size for non-Llama4 configs
    assert _dense_mlp_size(arch) == 4096


def test_llama4_intermediate_size_mlp_only_no_moe_does_not_force_shared_expert():
    cfg = SimpleNamespace(
        hidden_size = 2048,
        num_hidden_layers = 4,
        num_attention_heads = 16,
        num_key_value_heads = 4,
        intermediate_size = 8192,
        intermediate_size_mlp = 16384,
        vocab_size = 32000,
        tie_word_embeddings = True,
    )
    arch = extract_arch_config(cfg)
    assert arch is not None
    assert arch.dense_intermediate_size == 16384
    assert arch.n_shared_experts == 0
