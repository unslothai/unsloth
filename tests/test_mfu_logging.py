# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FLOP accounting of unsloth/utils/mfu.py against hand counts on a tiny Llama."""

import importlib.util
import pathlib

import pytest
import torch

pytest.importorskip("transformers")

MFU = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "mfu.py"


def _load():
    spec = importlib.util.spec_from_file_location("_unsloth_mfu", MFU)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mfu = _load()

# hidden 16, 4 heads of 4, 2 kv heads, intermediate 32, vocab 64, 2 layers, untied head.
PER_LAYER = 16 * 16 + 16 * 8 + 16 * 8 + 16 * 16 + 3 * 16 * 32  # 2304
LAYERS = 2 * PER_LAYER
HEAD = 16 * 64
N = LAYERS + HEAD
# LoRA r=2 on q_proj (16->16) and v_proj (16->8): A 16*2 + B 2*16 and A 16*2 + B 2*8 per layer.
LORA = 2 * (32 + 32 + 32 + 16)


def _model():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 64,
        tie_word_embeddings = False,
        max_position_embeddings = 64,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


def test_full_finetune_is_6n():
    total, parts = mfu.count_linear_flops_per_token(_model())
    assert parts == {"forward": 2 * N, "input_grad": 2 * N, "weight_grad": 2 * N, "recompute": 0}
    assert total == 6 * N


def test_full_finetune_checkpointing_recomputes_layers_not_head():
    total, parts = mfu.count_linear_flops_per_token(_model(), gradient_checkpointing = True)
    assert parts["recompute"] == 2 * LAYERS
    assert total == 6 * N + 2 * LAYERS


def test_frozen_is_4n():
    model = _model().requires_grad_(False)
    total, parts = mfu.count_linear_flops_per_token(model)
    assert parts["weight_grad"] == 0
    assert total == 4 * N


def test_lora_counts_weight_grads_only_for_adapters():
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        _model(),
        peft.LoraConfig(r = 2, target_modules = ["q_proj", "v_proj"], lora_dropout = 0.0),
    )
    total, parts = mfu.count_linear_flops_per_token(model)
    assert parts == {
        "forward": 2 * (N + LORA),
        "input_grad": 2 * (N + LORA),
        "weight_grad": 2 * LORA,
        "recompute": 0,
    }
    total_gc, parts_gc = mfu.count_linear_flops_per_token(model, gradient_checkpointing = True)
    assert parts_gc["recompute"] == 2 * (LAYERS + LORA)
    assert total_gc == total + 2 * (LAYERS + LORA)


def test_attention_flops_per_pair():
    config = _model().config
    # QK^T and AV: 2 FLOPs per multiply add, head_dim 4, 4 heads, 2 layers.
    forward = 2 * 2 * 4 * 4 * 2
    assert mfu.count_attention_flops_per_pair(config) == 3 * forward
    assert mfu.count_attention_flops_per_pair(config, gradient_checkpointing = True) == 4 * forward


def test_sequence_lengths_exclude_padding_and_split_packing():
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    assert mfu._sequence_lengths({"attention_mask": mask}, ()).tolist() == [3, 2]
    packed = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    lengths = mfu._sequence_lengths({"input_ids": packed, "position_ids": packed}, ())
    assert lengths.tolist() == [3, 2, 4]
    ids = torch.zeros(2, 5, dtype = torch.long)
    assert mfu._sequence_lengths({"input_ids": ids}, ()).tolist() == [5, 5]


def test_peak_tflops_override_and_table(monkeypatch):
    monkeypatch.setenv("UNSLOTH_PEAK_TFLOPS", "123.5")
    assert mfu.get_peak_tflops() == 123.5
    monkeypatch.delenv("UNSLOTH_PEAK_TFLOPS")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device = None: "Tesla T4")
    assert mfu.get_peak_tflops(dtype = torch.float16) == 65.0
    assert mfu.get_peak_tflops(dtype = torch.bfloat16) is None
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device = None: "NVIDIA H100 PCIe")
    assert mfu.get_peak_tflops(dtype = torch.bfloat16) == 756.0
    assert mfu.get_peak_tflops(dtype = torch.float32) is None
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device = None: "Mystery GPU")
    assert mfu.get_peak_tflops() is None


def test_callback_logs_tflops_and_mfu(tmp_path, monkeypatch):
    from transformers import Trainer, TrainingArguments

    monkeypatch.setenv("UNSLOTH_PEAK_TFLOPS", "1")
    model = _model()
    rows = [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 0], "labels": [1, 2, 3, -100]},
        {"input_ids": [5, 6, 7, 8], "attention_mask": [1, 1, 0, 0], "labels": [5, 6, -100, -100]},
    ]
    dataset = [{k: torch.tensor(v) for k, v in row.items()} for row in rows]
    trainer = Trainer(
        model = model,
        args = TrainingArguments(
            output_dir = str(tmp_path),
            per_device_train_batch_size = 2,
            max_steps = 2,
            logging_steps = 1,
            report_to = [],
            use_cpu = True,
            save_strategy = "no",
        ),
        train_dataset = dataset,
    )
    callback = mfu.add_mfu_callback(trainer)
    assert mfu.add_mfu_callback(trainer) is callback
    assert trainer.callback_handler.callbacks[0] is callback

    seen = []
    original = callback._count_batch

    def spy(module, args, kwargs):
        before = int(callback._tokens)
        original(module, args, kwargs)
        seen.append(int(callback._tokens) - before)

    callback._count_batch = spy
    trainer.train()
    assert seen == [5, 5]
    logs = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert len(logs) == 2
    for entry in logs:
        assert entry["train_tflops_per_gpu"] > 0
        assert entry["train_mfu"] == pytest.approx(entry["train_tflops_per_gpu"])
    assert callback._handle is None


def test_gpt2_conv1d_projections_are_counted():
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(n_embd = 16, n_layer = 2, n_head = 4, vocab_size = 64, n_positions = 32)
    model = GPT2LMHeadModel(config)
    # c_attn 16->48, attn c_proj 16->16, c_fc 16->64, mlp c_proj 64->16 per layer, plus the head.
    expected = 2 * 12 * 16 * 16 + 16 * 64
    total, parts = mfu.count_linear_flops_per_token(model)
    assert parts["forward"] == 2 * expected


class _Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(8, 16, bias = False)
        self.down = torch.nn.Linear(16, 8, bias = False)


class _MoE(torch.nn.Module):
    def __init__(
        self,
        top_k = None,
        num_experts_per_tok = None,
    ):
        super().__init__()
        self.config = type("Config", (), {"num_experts_per_tok": num_experts_per_tok})()
        self.gate = torch.nn.Linear(8, 4, bias = False)
        self.experts = torch.nn.ModuleList(_Expert() for _ in range(4))
        if top_k is not None:
            self.top_k = top_k


def test_moe_experts_scaled_by_routed_share():
    gate, expert = 8 * 4, 2 * 8 * 16
    total, parts = mfu.count_linear_flops_per_token(_MoE(top_k = 2))
    assert parts["forward"] == 2 * (gate + 2 * expert)
    total, parts = mfu.count_linear_flops_per_token(_MoE(num_experts_per_tok = 1))
    assert parts["forward"] == 2 * (gate + expert)
    total, parts = mfu.count_linear_flops_per_token(_MoE())
    assert parts["forward"] == 2 * gate


class _State:
    def __init__(self):
        self.log_history = []
        self.global_step = 1


def test_eval_and_save_time_excluded_from_window(monkeypatch):
    clock = iter([0.0, 1.0, 11.0, 12.0, 21.0, 22.0, 22.0])
    monkeypatch.setattr(mfu.time, "perf_counter", lambda: next(clock))
    monkeypatch.setenv("UNSLOTH_PEAK_TFLOPS", "1")
    callback = mfu.UnslothMFUCallback()
    callback.on_train_begin(None, _State(), None, model = _model())
    linear = callback._linear_per_token
    callback.on_step_begin(None, None, None)
    callback.on_step_end(None, None, None)
    callback.on_evaluate(None, None, None)
    callback.on_step_begin(None, None, None)
    callback.on_step_end(None, None, None)
    callback.on_save(None, None, None)
    callback.on_step_begin(None, None, None)
    callback.on_step_end(None, None, None)
    callback._tokens = torch.tensor(10**12 // linear)
    logs = {"loss": 1.0}
    callback.on_log(None, _State(), None, logs = logs)
    # 22 s of wall time minus 10 s of evaluation and 9 s of checkpointing.
    assert logs["train_tflops_per_gpu"] == pytest.approx((10**12 // linear) * linear / 3 / 1e12)
    callback.on_train_end(None, None, None)


def test_model_split_over_gpus_reports_per_gpu_average(monkeypatch):
    monkeypatch.setenv("UNSLOTH_PEAK_TFLOPS", "1")
    model = _model()
    devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
    fake = [type("P", (), {"device": devices[i % 2]})() for i in range(4)]
    monkeypatch.setattr(model, "parameters", lambda *a, **k: iter(fake))
    callback = mfu.UnslothMFUCallback()
    callback.on_train_begin(None, _State(), None, model = model)
    assert callback._num_devices == 2
    monkeypatch.setattr(mfu.time, "perf_counter", lambda: 1.0)
    callback._window_start = 0.0
    callback._tokens = 1000
    logs = {"loss": 1.0}
    callback.on_log(None, _State(), None, logs = logs)
    assert logs["train_tflops_per_gpu"] == pytest.approx(
        1000 * callback._linear_per_token / 2 / 1e12
    )
    assert logs["train_mfu"] == pytest.approx(logs["train_tflops_per_gpu"])
    callback.on_train_end(None, None, None)


def test_encoder_decoder_models_are_not_reported(monkeypatch):
    from transformers import T5Config, T5ForConditionalGeneration

    monkeypatch.setenv("UNSLOTH_PEAK_TFLOPS", "1")
    config = T5Config(vocab_size = 64, d_model = 16, d_kv = 4, d_ff = 32, num_layers = 1, num_heads = 4)
    model = T5ForConditionalGeneration(config)
    callback = mfu.UnslothMFUCallback()
    callback.on_train_begin(None, _State(), None, model = model)
    assert callback._handle is None
    callback.on_step_begin(None, None, None)
    callback._tokens = 1000
    logs = {"loss": 1.0}
    callback.on_log(None, _State(), None, logs = logs)
    assert logs == {"loss": 1.0}


def test_modules_to_save_counts_only_the_active_copy():
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        _model(),
        peft.LoraConfig(
            r = 2, target_modules = ["q_proj", "v_proj"], modules_to_save = ["lm_head"], lora_dropout = 0.0
        ),
    )
    total, parts = mfu.count_linear_flops_per_token(model)
    assert parts["forward"] == 2 * (N + LORA)
    assert parts["weight_grad"] == 2 * (LORA + HEAD)


def test_on_log_synchronizes_every_participating_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device = None: calls.append(device))
    callback = mfu.UnslothMFUCallback(peak_tflops = 1)
    callback._devices = [0, 1]
    callback._linear_per_token = 1
    callback._window_start = 0.0
    callback._tokens = 10
    callback.on_log(None, _State(), None, logs = {"loss": 1.0})
    assert calls == [0, 1]


def test_vision_tower_is_not_charged_per_text_token():
    from transformers import (
        CLIPVisionConfig,
        LlamaConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
    )

    text = _model().config
    vision = CLIPVisionConfig(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        image_size = 8,
        patch_size = 4,
    )
    model = LlavaForConditionalGeneration(LlavaConfig(text_config = text, vision_config = vision))
    total, parts = mfu.count_linear_flops_per_token(model)
    assert parts["forward"] == 2 * N


def test_bidirectional_encoder_counts_square_pairs():
    from transformers import BertConfig, BertModel

    config = BertConfig(
        vocab_size = 64,
        hidden_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        intermediate_size = 32,
    )
    model = BertModel(config).train()
    callback = mfu.UnslothMFUCallback(peak_tflops = 1)
    callback.on_train_begin(None, _State(), None, model = model)
    callback._count_batch(model, (), {"attention_mask": torch.tensor([[1, 1, 1, 0]])})
    assert int(callback._pairs) == 9
    causal = mfu.UnslothMFUCallback(peak_tflops = 1)
    causal.on_train_begin(None, _State(), None, model = _model().train())
    causal._count_batch(model, (), {"attention_mask": torch.tensor([[1, 1, 1, 0]])})
    assert int(causal._pairs) == 6
    callback.on_train_end(None, None, None)
    causal.on_train_end(None, None, None)


def test_output_head_charged_only_for_logits_to_keep(monkeypatch):
    model = _model().train()
    callback = mfu.UnslothMFUCallback(peak_tflops = 1)
    callback.on_train_begin(None, _State(), None, model = model)
    assert callback._head_per_token == 6 * HEAD
    assert callback._linear_per_token == 6 * LAYERS
    mask = torch.ones(2, 4, dtype = torch.long)
    callback._count_batch(model, (), {"attention_mask": mask, "logits_to_keep": 1})
    callback._count_batch(model, (), {"attention_mask": mask})
    assert int(callback._tokens) == 16
    assert int(callback._head_tokens) == 2 + 8
    monkeypatch.setattr(mfu.time, "perf_counter", lambda: 1.0)
    callback._window_start = 0.0
    callback._pairs = 0
    logs = {"loss": 1.0}
    callback.on_log(None, _State(), None, logs = logs)
    assert logs["train_tflops_per_gpu"] == pytest.approx((16 * 6 * LAYERS + 10 * 6 * HEAD) / 1e12)
    callback.on_train_end(None, None, None)


def test_add_mfu_callback_moves_an_existing_callback_first(tmp_path):
    from transformers import Trainer, TrainingArguments

    existing = mfu.UnslothMFUCallback()
    trainer = Trainer(
        model = _model(),
        args = TrainingArguments(output_dir = str(tmp_path), report_to = [], use_cpu = True),
        callbacks = [existing],
    )
    assert trainer.callback_handler.callbacks[0] is not existing
    assert mfu.add_mfu_callback(trainer) is existing
    assert trainer.callback_handler.callbacks[0] is existing
    assert (
        sum(isinstance(c, mfu.UnslothMFUCallback) for c in trainer.callback_handler.callbacks) == 1
    )


def test_l40_is_not_matched_as_l4(monkeypatch):
    monkeypatch.delenv("UNSLOTH_PEAK_TFLOPS", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name, peak in (("NVIDIA L40S", 362.05), ("NVIDIA L40", 181.05), ("NVIDIA L4", 121.0)):
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device = None, n = name: n)
        assert mfu.get_peak_tflops(dtype = torch.bfloat16) == peak, name


def test_inactive_lora_adapters_are_not_counted():
    peft = pytest.importorskip("peft")
    config = peft.LoraConfig(r = 2, target_modules = ["q_proj", "v_proj"], lora_dropout = 0.0)
    model = peft.get_peft_model(_model(), config)
    model.add_adapter("other", config)
    model.set_adapter("default")
    _, parts = mfu.count_linear_flops_per_token(model)
    assert parts["forward"] == 2 * (N + LORA)
    with model.disable_adapter():
        _, parts = mfu.count_linear_flops_per_token(model)
    assert parts["forward"] == 2 * N


def test_hybrid_layer_types_charge_pairs_only_for_softmax_attention():
    from types import SimpleNamespace

    config = SimpleNamespace(
        num_hidden_layers = 4,
        num_attention_heads = 4,
        hidden_size = 16,
        head_dim = 4,
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    assert mfu.count_attention_flops_per_pair(config) == 4 * 1 * 4 * 4 * 3
    config.layer_types = ["sliding_attention", "full_attention", "full_attention", "full_attention"]
    assert mfu.count_attention_flops_per_pair(config) == 4 * 4 * 4 * 4 * 3


def test_unsloth_forward_head_window_matches_both_keep_spellings(monkeypatch):
    model = _model().train()
    callback = mfu.UnslothMFUCallback(peak_tflops = 1)
    monkeypatch.setattr(mfu, "_runs_unsloth_head_window", lambda m: True)
    callback.on_train_begin(None, _State(), None, model = model)
    mask = torch.ones(2, 4, dtype = torch.long)
    callback._count_batch(model, (), {"attention_mask": mask, "num_logits_to_keep": 1})
    assert int(callback._head_tokens) == 2
    callback._count_batch(
        model,
        (),
        {"attention_mask": mask, "logits_to_keep": torch.tensor([3]), "num_logits_to_keep": 0},
    )
    assert int(callback._head_tokens) == 2 + 8
    callback.on_train_end(None, None, None)


def test_runs_unsloth_head_window_reads_the_base_model_forward():
    model = _model()
    assert not mfu._runs_unsloth_head_window(model)

    def forward(self, *args, **kwargs):
        pass

    forward.__module__ = "unsloth.models.llama"
    patched = type("Patched", (type(model),), {"forward": forward})
    model.__class__ = patched
    assert mfu._runs_unsloth_head_window(model)


def test_pause_after_a_step_excludes_only_time_after_the_gpu_finished(monkeypatch):
    # Step ends at CPU t=1 with 3 s of kernels still queued; eval until t=11.
    clock = iter([0.0, 1.0, 11.0, 12.0, 12.0])
    monkeypatch.setattr(mfu.time, "perf_counter", lambda: next(clock))
    events = []

    class Event:
        def __init__(self, enable_timing = False):
            self.t = None
            events.append(self)

        def record(self, stream = None):
            # GPU completion times: step end at 4 s, resume probe at 11 s.
            self.t = 4.0 if len(events) == 1 else 11.0

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return (other.t - self.t) * 1000

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device = None: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device = None: None)
    callback = mfu.UnslothMFUCallback(peak_tflops = 1)
    callback._devices = [0]
    callback._linear_per_token = 10**12
    callback.on_step_begin(None, None, None)
    callback.on_step_end(None, None, None)
    callback.on_evaluate(None, None, None)
    callback.on_step_begin(None, None, None)
    callback._tokens = 5
    logs = {"loss": 1.0}
    callback.on_log(None, _State(), None, logs = logs)
    # 12 s of wall time minus the 7 s from GPU completion (4 s) to resume (11 s).
    assert logs["train_tflops_per_gpu"] == pytest.approx(5 / 5)


def test_positional_feature_dict_is_counted():
    # SentenceTransformer losses call model(features) with one feature dict.
    callback = mfu.UnslothMFUCallback(peak_tflops = 1)
    model = _model().train()
    callback.on_train_begin(None, _State(), None, model = model)
    features = {
        "input_ids": torch.zeros(2, 4, dtype = torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    }
    callback._count_batch(model, (features,), {})
    assert int(callback._tokens) == 5
    callback.on_train_end(None, None, None)


def test_mla_attention_uses_separate_qk_and_value_dims():
    from types import SimpleNamespace
    config = SimpleNamespace(
        num_hidden_layers = 2,
        num_attention_heads = 4,
        hidden_size = 64,
        head_dim = 64,
        qk_nope_head_dim = 128,
        qk_rope_head_dim = 64,
        v_head_dim = 128,
    )
    assert mfu.count_attention_flops_per_pair(config) == 3 * 2 * 2 * 4 * (192 + 128)


def test_sentence_transformer_attention_uses_the_inner_config():
    # SentenceTransformer is an nn.Sequential whose first module holds the HF model as auto_model.
    inner = _model()
    wrapper = torch.nn.Sequential(torch.nn.Module())
    wrapper[0].auto_model = inner
    plain = mfu.UnslothMFUCallback(peak_tflops = 1)
    plain.on_train_begin(None, _State(), None, model = inner)
    wrapped = mfu.UnslothMFUCallback(peak_tflops = 1)
    wrapped.on_train_begin(None, _State(), None, model = wrapper)
    assert wrapped._attention_per_pair == plain._attention_per_pair > 0
    plain.on_train_end(None, None, None)
    wrapped.on_train_end(None, None, None)


def test_model_off_the_gpu_takes_no_gpu_peak(monkeypatch):
    monkeypatch.delenv("UNSLOTH_PEAK_TFLOPS", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device = None: "NVIDIA H100 80GB HBM3")
    callback = mfu.UnslothMFUCallback()
    callback.on_train_begin(None, _State(), None, model = _model())
    assert callback.peak_tflops is None and callback._devices == []
    callback.on_train_end(None, None, None)
    monkeypatch.setenv("UNSLOTH_PEAK_TFLOPS", "100")
    callback = mfu.UnslothMFUCallback()
    callback.on_train_begin(None, _State(), None, model = _model())
    assert callback.peak_tflops == 100
    callback.on_train_end(None, None, None)


def test_sentence_transformer_dense_after_pooling_is_not_per_token():
    inner = _model()
    plain = mfu.count_linear_flops_per_token(inner)[0]
    wrapper = torch.nn.Sequential(torch.nn.Module(), torch.nn.Module(), torch.nn.Linear(16, 16))
    wrapper[0].auto_model = inner
    assert mfu.count_linear_flops_per_token(wrapper)[0] == plain
