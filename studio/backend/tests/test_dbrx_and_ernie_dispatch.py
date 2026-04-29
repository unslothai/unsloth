from types import SimpleNamespace

from utils.hardware.vram_estimation import (
    _compute_dense_layer_indices,
    _compute_shared_moe_elements,
    _shared_expert_size,
    extract_arch_config,
)


def test_dbrx_extracts_moe_fields_from_ffn_subconfig():
    ffn = SimpleNamespace(moe_num_experts = 4, moe_top_k = 2, ffn_hidden_size = 1024)
    cfg = SimpleNamespace(
        hidden_size = 2048,
        num_hidden_layers = 4,
        num_attention_heads = 16,
        num_key_value_heads = 4,
        intermediate_size = 2048,
        vocab_size = 32000,
        tie_word_embeddings = False,
        ffn_config = ffn,
    )
    arch = extract_arch_config(cfg)
    assert arch is not None
    assert arch.num_experts == 4
    assert arch.num_experts_per_tok == 2
    assert arch.moe_intermediate_size == 1024


def test_dbrx_top_level_attrs_take_precedence_over_ffn_config():
    ffn = SimpleNamespace(moe_num_experts = 4, moe_top_k = 2, ffn_hidden_size = 1024)
    cfg = SimpleNamespace(
        hidden_size = 2048,
        num_hidden_layers = 4,
        num_attention_heads = 16,
        num_key_value_heads = 4,
        intermediate_size = 2048,
        vocab_size = 32000,
        tie_word_embeddings = False,
        ffn_config = ffn,
        num_local_experts = 16,
        num_experts_per_tok = 8,
    )
    arch = extract_arch_config(cfg)
    assert arch is not None
    assert arch.num_experts == 16
    assert arch.num_experts_per_tok == 8


def test_no_ffn_config_does_not_break_extraction():
    cfg = SimpleNamespace(
        hidden_size = 1024,
        num_hidden_layers = 4,
        num_attention_heads = 8,
        num_key_value_heads = 2,
        intermediate_size = 4096,
        vocab_size = 32000,
        tie_word_embeddings = True,
        num_local_experts = 8,
        moe_intermediate_size = 512,
        num_experts_per_tok = 2,
    )
    arch = extract_arch_config(cfg)
    assert arch is not None
    assert arch.num_experts == 8
    assert arch.moe_intermediate_size == 512


def test_ernie_phase_modulo_with_interval_two_matches_decoder():
    # transformers ERNIE 4.5 MoE: ((i + 1) % interval == 0) AND start <= i <= end.
    # With start=2, end=8 (=> exclusive 9), interval=2: MoE = {3, 5, 7}.
    cfg = SimpleNamespace(
        num_hidden_layers = 10,
        moe_layer_start_index = 2,
        moe_layer_end_index = 8,
        moe_layer_interval = 2,
    )
    assert _compute_dense_layer_indices(cfg, 10) == (0, 1, 2, 4, 6, 8, 9)


def test_ernie_phase_modulo_with_interval_three():
    # interval=3 with start=0, end=-1 (=> last): MoE = {2, 5, 8} since (i+1)%3==0.
    cfg = SimpleNamespace(
        num_hidden_layers = 9,
        moe_layer_start_index = 0,
        moe_layer_end_index = -1,
        moe_layer_interval = 3,
    )
    assert _compute_dense_layer_indices(cfg, 9) == (0, 1, 3, 4, 6, 7)


def test_ernie_default_interval_one_keeps_all_layers_after_start_as_moe():
    cfg = SimpleNamespace(
        num_hidden_layers = 4,
        moe_layer_start_index = 1,
        moe_layer_end_index = -1,
        moe_layer_interval = 1,
    )
    # interval=1 -> every i in [start, end] is MoE.
    assert _compute_dense_layer_indices(cfg, 4) == (0,)


def test_ernie_vl_shared_expert_width_uses_text_routed_not_vision():
    # ERNIE 4.5 VL MoE: moe_intermediate_size=[text_routed, vision_routed]; the
    # shared expert is sized from text_routed * moe_num_shared_experts and has
    # no shared_expert_gate.
    cfg = SimpleNamespace(
        text_config = SimpleNamespace(
            hidden_size = 1024,
            num_hidden_layers = 4,
            num_attention_heads = 8,
            num_key_value_heads = 4,
            intermediate_size = 2048,
            vocab_size = 32000,
            tie_word_embeddings = False,
            moe_num_experts = 8,
            moe_num_shared_experts = 2,
            moe_intermediate_size = [1536, 512],
        ),
        quantization_config = {},
    )
    arch = extract_arch_config(cfg)
    assert arch is not None
    assert arch.shared_expert_intermediate_size is None
    assert arch.moe_intermediate_size == 1536
    assert arch.n_shared_experts == 2
    assert _shared_expert_size(arch) == 1536
    assert _compute_shared_moe_elements(arch) == 1024 * 1536 * 3 * 2


def test_qwen_style_explicit_shared_expert_size_still_adds_gate():
    # Qwen3.5-MoE explicitly sets shared_expert_intermediate_size and uses a
    # shared_expert_gate Linear(hd, 1); _compute_shared_moe_elements adds the
    # gate iff shared_expert_intermediate_size is set.
    cfg = SimpleNamespace(
        hidden_size = 1024,
        num_hidden_layers = 4,
        num_attention_heads = 8,
        num_key_value_heads = 4,
        intermediate_size = 2048,
        vocab_size = 32000,
        tie_word_embeddings = False,
        num_local_experts = 8,
        moe_intermediate_size = 256,
        shared_expert_intermediate_size = 768,
    )
    arch = extract_arch_config(cfg)
    assert arch is not None
    assert arch.shared_expert_intermediate_size == 768
    # n_shared_experts is auto-set to 1 when shared_expert_intermediate_size is set
    assert arch.n_shared_experts == 1
    expected = 1024 * 768 * 3 + 1 * 1024
    assert _compute_shared_moe_elements(arch) == expected
