# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from unsloth import FastLanguageModel
import unsloth.trainer as trainer_module
import unsloth.utils.packing as packing_module
from unsloth.utils import attention_dispatch as attention_dispatch_utils
from unsloth.utils.packing import (
    configure_padding_free,
    configure_sample_packing,
    enable_padding_free_metadata,
    enable_sample_packing,
    mask_packed_sequence_boundaries,
    patch_hybrid_linear_attention_varlen,
)

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset, IterableDataset
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


def _build_packed_training_setup(tmp_path, device):
    dtype = None
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    elif device.type == "xpu":
        dtype = torch.bfloat16
            
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM",
            max_seq_length = 64,
            load_in_4bit = False,
            dtype = dtype,
        )
    except OSError as exc:  # pragma: no cover - offline CI
        pytest.skip(f"Requires access to tiny llama checkpoint: {exc}")

    model.to(device)

    dataset = Dataset.from_dict(
        {
            "text": [
                "Hello world!",
                "Short sample.",
                "This is a slightly longer packed example to test batching.",
                "Another response to include in the batch.",
            ]
        }
    )

    training_args = SFTConfig(
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 1,
        dataset_text_field = "text",
        max_length = 64,
        logging_steps = 1,
        max_steps = 1,
        fp16 = dtype == torch.float16,
        bf16 = dtype == torch.bfloat16,
        dataset_num_proc = 1,
        output_dir = str(tmp_path),
        packing = True,
    )

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = training_args,
    )

    enable_sample_packing(model, trainer)

    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))

    model_device = next(model.parameters()).device

    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(model_device)

    from unsloth.models import llama as llama_mod

    return model, batch, trainer, llama_mod


def _trim_batch_to_total_tokens(data, total_tokens):
    def _trim_tensor(t: torch.Tensor):
        if t.ndim >= 2 and t.size(1) > total_tokens:
            return t[:, :total_tokens].contiguous()
        return t

    trimmed = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            trimmed[key] = _trim_tensor(value)
        else:
            trimmed[key] = value
    return trimmed


def test_mask_packed_sequence_boundaries_marks_single_row():
    shift_labels = torch.arange(6, dtype = torch.long).view(1, 6)
    changed = mask_packed_sequence_boundaries(
        shift_labels,
        torch.tensor([2, 1, 3], dtype = torch.int32),
    )
    assert changed is True
    flat = shift_labels.view(-1)
    assert flat[1].item() == -100
    assert flat[2].item() == -100
    assert flat[5].item() == -100
    assert flat[0].item() != -100


def test_mask_packed_sequence_boundaries_across_multiple_rows():
    shift_labels = torch.arange(10, dtype = torch.long).view(2, 5)
    lengths = torch.tensor([3, 2, 4, 1], dtype = torch.int32)
    changed = mask_packed_sequence_boundaries(shift_labels, lengths)
    assert changed is True
    flat = shift_labels.view(-1)
    for idx in (2, 4, 8, 9):
        assert flat[idx].item() == -100
    assert torch.any(flat != -100)


def test_configure_sample_packing():
    config = SimpleNamespace()
    configure_sample_packing(config)

    assert config.packing is True
    assert config.padding_free is True
    assert config.remove_unused_columns is False


def test_configure_padding_free():
    config = SimpleNamespace(remove_unused_columns = True)
    configure_padding_free(config)

    assert config.padding_free is True
    assert config.remove_unused_columns is False


# --- Hybrid linear-attention guard + varlen shim (PR #7211 / #7249) ---------------


def _hybrid_config_model():
    # Qwen3.5 / Qwen3-Next style: explicit linear_attention layer schedule.
    return SimpleNamespace(
        config = SimpleNamespace(layer_types = ["linear_attention", "full_attention"])
    )


def _gemma3_model():
    # Has layer_types but no linear_attention -> must NOT be flagged as hybrid.
    return SimpleNamespace(
        config = SimpleNamespace(
            model_type = "gemma3", layer_types = ["sliding_attention", "full_attention"]
        ),
    )


def _dense_qwen3_model():
    return SimpleNamespace(
        config = SimpleNamespace(model_type = "qwen3", architectures = ["Qwen3ForCausalLM"])
    )


class _FakeGatedDeltaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(4, 4, 3, groups = 4)
        self.A_log = torch.nn.Parameter(torch.zeros(4))

    def forward(self, hidden_states, **kwargs):  # dispatch through self.<kernel>
        return self.chunk_gated_delta_rule(self.causal_conv1d_fn(hidden_states))


class _FakeHybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()  # no markers -> forces module-level detection
        self.linear_attn = _FakeGatedDeltaNet()


def test_is_hybrid_linear_attention_detects_and_excludes():
    is_hybrid = trainer_module._is_hybrid_linear_attention_model
    assert is_hybrid(_hybrid_config_model()) is True
    assert is_hybrid(_FakeHybridModel()) is True  # module-structural evidence
    assert is_hybrid(_text_model()) is False  # Llama
    assert is_hybrid(_gemma3_model()) is False  # layer_types without linear_attention
    assert is_hybrid(_dense_qwen3_model()) is False  # dense Qwen3
    assert is_hybrid(None) is False


def test_varlen_from_position_ids():
    cu, seq_idx = packing_module._varlen_from_position_ids(torch.tensor([[0, 1, 0, 0, 1, 2]]))
    assert cu.tolist() == [0, 2, 3, 6]
    assert seq_idx.tolist() == [[0, 0, 1, 2, 2, 2]]
    assert (
        packing_module._varlen_from_position_ids(torch.tensor([[0, 1, 2, 3]])) is None
    )  # single sequence
    assert packing_module._varlen_from_position_ids(torch.tensor([[1, 2, 3]])) is None  # first != 0
    assert (
        packing_module._varlen_from_position_ids(torch.tensor([[0, 1], [0, 1]])) is None
    )  # normal 2-row batch
    assert packing_module._varlen_from_position_ids(None) is None


def test_seq_idx_from_cu_seqlens_handles_trailing_pad():
    cu = torch.tensor([0, 2, 5], dtype = torch.int32)
    boundaries, seq_idx = packing_module._seq_idx_from_cu_seqlens(cu, total = 8)  # pad_to_multiple_of
    assert boundaries.tolist() == [0, 2, 5, 8]
    assert seq_idx.tolist() == [[0, 0, 1, 1, 1, 2, 2, 2]]
    boundaries2, _ = packing_module._seq_idx_from_cu_seqlens(cu, total = 5)  # exact fit
    assert boundaries2.tolist() == [0, 2, 5]
    assert (
        packing_module._seq_idx_from_cu_seqlens(torch.tensor([1, 2], dtype = torch.int32), total = 2)
        is None
    )
    assert packing_module._seq_idx_from_cu_seqlens(cu, total = 3) is None  # boundaries exceed total


def test_hybrid_varlen_metadata_prefers_packed_seq_lengths():
    # A competing position_ids would segment [0, 3, 6]; packed_seq_lengths must win.
    kwargs = {
        "input_ids": torch.zeros(1, 6, dtype = torch.long),
        "packed_seq_lengths": torch.tensor([2, 1, 3], dtype = torch.int32),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]]),
    }
    cu, seq_idx = packing_module._hybrid_varlen_metadata(kwargs)
    assert cu.tolist() == [0, 2, 3, 6]
    assert seq_idx.tolist() == [[0, 0, 1, 2, 2, 2]]


def test_hybrid_varlen_metadata_suppressed_when_cached():
    base = {
        "input_ids": torch.zeros(1, 6, dtype = torch.long),
        "packed_seq_lengths": torch.tensor([2, 1, 3], dtype = torch.int32),
    }
    assert packing_module._hybrid_varlen_metadata({**base, "use_cache": True}) is None
    assert packing_module._hybrid_varlen_metadata({**base, "past_key_values": object()}) is None


def test_hybrid_varlen_metadata_none_for_plain_batch():
    kwargs = {
        "input_ids": torch.zeros(1, 4, dtype = torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),
    }
    assert packing_module._hybrid_varlen_metadata(kwargs) is None


def _make_fake_kernels():
    def causal_conv1d_fn(
        x,
        weight = None,
        bias = None,
        activation = None,
        seq_idx = None,
    ):
        causal_conv1d_fn.calls.append(seq_idx)
        return x

    causal_conv1d_fn.calls = []

    def chunk_gated_delta_rule(
        q,
        k = None,
        v = None,
        cu_seqlens = None,
        **kw,
    ):
        chunk_gated_delta_rule.calls.append(cu_seqlens)
        return q

    chunk_gated_delta_rule.calls = []
    return causal_conv1d_fn, chunk_gated_delta_rule


class _ShimGatedDeltaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(4, 4, 3, groups = 4)
        self.causal_conv1d_fn, self.chunk_gated_delta_rule = _make_fake_kernels()

    def forward(self, hidden_states, **kwargs):
        return self.chunk_gated_delta_rule(self.causal_conv1d_fn(hidden_states))


class _ShimHybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(layer_types = ["linear_attention", "full_attention"])
        self.linear_attn = _ShimGatedDeltaNet()

    def forward(
        self,
        input_ids = None,
        position_ids = None,
        packed_seq_lengths = None,
        use_cache = None,
        **kwargs,
    ):
        return self.linear_attn(input_ids.float())


def test_patch_hybrid_varlen_flag_off(monkeypatch):
    monkeypatch.delenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", raising = False)
    model = _ShimHybridModel()
    assert patch_hybrid_linear_attention_varlen(model) is False
    assert not getattr(model, "_unsloth_varlen_forward_wrapped", False)


def test_patch_hybrid_varlen_active_and_idempotent(monkeypatch):
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    model = _ShimHybridModel()
    conv_orig, scan_orig = (
        model.linear_attn.causal_conv1d_fn,
        model.linear_attn.chunk_gated_delta_rule,
    )

    assert patch_hybrid_linear_attention_varlen(model) is True
    assert model._unsloth_varlen_forward_wrapped is True
    assert model.linear_attn._unsloth_varlen_wrapped is True
    assert patch_hybrid_linear_attention_varlen(model) is True  # idempotent, no double-wrap

    conv_orig.calls.clear()
    scan_orig.calls.clear()
    packing_module._HYBRID_WARNED.clear()
    ids = torch.zeros(1, 6, dtype = torch.long)
    model(
        input_ids = ids,
        packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
        use_cache = False,
    )
    assert conv_orig.calls[-1] is not None  # seq_idx injected
    assert scan_orig.calls[-1].tolist() == [0, 2, 3, 6]  # cu_seqlens injected
    assert not packing_module._HYBRID_WARNED  # handshake passed, no rejection

    conv_orig.calls.clear()
    scan_orig.calls.clear()
    model(
        input_ids = ids, packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32), use_cache = True
    )
    assert conv_orig.calls[-1] is None  # cached forward -> no injection
    assert scan_orig.calls[-1] is None


def test_patch_hybrid_varlen_torch_fallback_fail_closed(monkeypatch):
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    model = _ShimHybridModel()

    def torch_chunk_gated_delta_rule(
        q,
        cu_seqlens = None,
        **kw,
    ):
        return q

    model.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    assert patch_hybrid_linear_attention_varlen(model) is False
    assert not getattr(model, "_unsloth_varlen_forward_wrapped", False)


def test_patch_hybrid_varlen_bad_signature_fail_closed(monkeypatch):
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    model = _ShimHybridModel()

    def scan_no_cu(q, **kw):  # missing cu_seqlens
        return q

    model.linear_attn.chunk_gated_delta_rule = scan_no_cu
    assert patch_hybrid_linear_attention_varlen(model) is False


def _hybrid_model_with_gdn(gdn_forward):
    # Build a fake hybrid model whose gated-delta mixer forward is `gdn_forward`.
    class _GatedDeltaNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = torch.nn.Conv1d(4, 4, 3, groups = 4)
            self.causal_conv1d_fn, self.chunk_gated_delta_rule = _make_fake_kernels()

        forward = gdn_forward

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(layer_types = ["linear_attention", "full_attention"])
            self.linear_attn = _GatedDeltaNet()

        def forward(
            self,
            input_ids = None,
            packed_seq_lengths = None,
            use_cache = None,
            **kwargs,
        ):
            return self.linear_attn(input_ids.float())

    return _Model()


def test_patch_hybrid_varlen_no_dispatch_aborts(monkeypatch):
    # Dispatch is verified at runtime, not statically. A mixer that never calls
    # self.<kernel> installs the shim, but the first packed forward aborts (both
    # boundary kernels are load-bearing).
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    model = _hybrid_model_with_gdn(lambda self, hidden_states, **kw: hidden_states)
    assert patch_hybrid_linear_attention_varlen(model) is True  # kernels valid -> installs
    with pytest.raises(RuntimeError, match = "both invoked"):
        model(
            input_ids = torch.zeros(1, 6),
            packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
            use_cache = False,
        )


def test_patch_hybrid_varlen_partial_dispatch_aborts(monkeypatch):
    # Only the conv fires; the scan would leak state. Both must be invoked, so abort.
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    conv_only = _hybrid_model_with_gdn(
        lambda self, hidden_states, **kw: self.causal_conv1d_fn(hidden_states)
    )
    assert patch_hybrid_linear_attention_varlen(conv_only) is True
    with pytest.raises(RuntimeError, match = "both invoked"):
        conv_only(
            input_ids = torch.zeros(1, 6),
            packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
            use_cache = False,
        )

    scan_only = _hybrid_model_with_gdn(
        lambda self, hidden_states, **kw: self.chunk_gated_delta_rule(hidden_states)
    )
    assert patch_hybrid_linear_attention_varlen(scan_only) is True
    with pytest.raises(RuntimeError, match = "both invoked"):
        scan_only(
            input_ids = torch.zeros(1, 6),
            packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
            use_cache = False,
        )


def test_varlen_from_position_ids_mrope_3d():
    pos = (
        torch.tensor([[0, 1, 0, 0, 1, 2]]).unsqueeze(0).expand(3, 1, 6).clone()
    )  # [3,1,T] text plane
    cu, seq_idx = packing_module._varlen_from_position_ids(pos)
    assert cu.tolist() == [0, 2, 3, 6]
    assert seq_idx.tolist() == [[0, 0, 1, 2, 2, 2]]


def test_hybrid_varlen_metadata_trailing_pad():
    # packed_seq_lengths sum to 6 but the flattened input is 8 (pad_to_multiple_of).
    kwargs = {
        "input_ids": torch.zeros(1, 8, dtype = torch.long),
        "packed_seq_lengths": torch.tensor([2, 1, 3], dtype = torch.int32),
    }
    cu, seq_idx = packing_module._hybrid_varlen_metadata(kwargs)
    assert cu.tolist() == [0, 2, 3, 6, 8]
    assert seq_idx.tolist() == [[0, 0, 1, 2, 2, 2, 3, 3]]


def _patch_fake_sft_trainer():
    class FakeSFTTrainer:
        def __init__(self, *args, **kwargs):
            self.model = args[0] if len(args) >= 1 else kwargs["model"]
            self.args = args[1] if len(args) >= 2 else kwargs["args"]
            self.data_collator = args[2] if len(args) >= 3 else kwargs.get("data_collator")

    trainer_module._patch_sft_trainer_auto_packing(SimpleNamespace(SFTTrainer = FakeSFTTrainer))
    return FakeSFTTrainer


def _vlm_model():
    return SimpleNamespace(
        config = SimpleNamespace(
            architectures = ["Gemma4ForConditionalGeneration"],
            model_type = "gemma4",
            vision_config = SimpleNamespace(),
        ),
        max_seq_length = 16,
    )


def _text_model():
    return SimpleNamespace(
        config = SimpleNamespace(
            architectures = ["LlamaForCausalLM"],
            model_type = "llama",
        ),
        max_seq_length = 16,
    )


class _CharacterTokenizer:
    bos_token = None
    eos_token = None
    chat_template = None

    def __call__(self, texts, **kwargs):
        is_batched = isinstance(texts, list)
        if not is_batched:
            texts = [texts]
        input_ids = [[ord(char) for char in text] for text in texts]
        if kwargs.get("truncation") and kwargs.get("max_length") is not None:
            input_ids = [ids[: kwargs["max_length"]] for ids in input_ids]
        return {"input_ids": input_ids if is_batched else input_ids[0]}


def test_vlm_text_dataset_allows_explicit_packing():
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)

    trainer = fake_trainer(
        model = _vlm_model(),
        args = config,
        processing_class = object(),
        train_dataset = Dataset.from_dict({"text": ["text-only CPT sample"]}),
    )

    assert config.packing is True
    assert config.padding_free is True
    assert trainer.model._unsloth_allow_packed_overlength is True


def test_vlm_without_processing_class_still_disables_packing():
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)

    fake_trainer(
        _vlm_model(),
        config,
        None,
        Dataset.from_dict({"text": ["text-only sample"]}),
    )

    assert config.packing is False
    assert config.padding_free is False


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    (
        ("t5", "T5ForConditionalGeneration"),
        ("bart", "BartForConditionalGeneration"),
        ("whisper", "WhisperForConditionalGeneration"),
    ),
)
def test_encoder_decoder_disables_packing(model_type, architecture):
    # Text-only encoder-decoder models are not VLMs, but their bidirectional encoder
    # attends across concatenated samples once padding-free drops attention_mask.
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    model = SimpleNamespace(
        config = SimpleNamespace(
            model_type = model_type,
            architectures = [architecture],
            is_encoder_decoder = True,
        ),
        max_seq_length = 16,
    )

    trainer = fake_trainer(model, config, None, Dataset.from_dict({"text": ["text-only sample"]}))

    assert config.packing is False
    assert config.padding_free is False


def test_decoder_only_conditional_generation_keeps_packing():
    # CSM is decoder-only despite the ForConditionalGeneration name -> packing stays on.
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    model = SimpleNamespace(
        config = SimpleNamespace(
            model_type = "csm",
            architectures = ["CsmForConditionalGeneration"],
            is_encoder_decoder = False,
        ),
        max_seq_length = 16,
    )

    trainer = fake_trainer(model, config, None, Dataset.from_dict({"text": ["text-only sample"]}))

    assert config.packing is True
    assert config.padding_free is True
    assert trainer.model._unsloth_allow_packed_overlength is True


def _hybrid_trainer_model():
    return SimpleNamespace(
        config = SimpleNamespace(
            model_type = "qwen3_next",
            architectures = ["Qwen3NextForCausalLM"],
            layer_types = ["linear_attention", "full_attention"],
        ),
        max_seq_length = 16,
    )


def test_hybrid_varlen_active_enables_packing(monkeypatch):
    # Baseline: shim active + no forward bypass -> hybrid packing is allowed.
    monkeypatch.setattr(trainer_module, "_chunked_loss_bypasses_forward", lambda config: False)
    monkeypatch.setattr(trainer_module, "patch_hybrid_linear_attention_varlen", lambda model: True)
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    fake_trainer(_hybrid_trainer_model(), config, None, Dataset.from_dict({"text": ["x"]}))
    assert config.packing is True
    assert config.padding_free is True


def test_hybrid_chunked_loss_stays_on_padded_path(monkeypatch):
    # TRL's chunked-loss forward bypass leaves the varlen shim off -> block packing.
    monkeypatch.setattr(trainer_module, "_chunked_loss_bypasses_forward", lambda config: True)
    monkeypatch.setattr(trainer_module, "patch_hybrid_linear_attention_varlen", lambda model: True)
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    fake_trainer(_hybrid_trainer_model(), config, None, Dataset.from_dict({"text": ["x"]}))
    assert config.packing is False
    assert config.padding_free is False


def test_string_hybrid_model_disables_packing(monkeypatch):
    # A string model= is materialized after init; a hybrid string is blocked because the
    # shim cannot patch a not-yet-built model.
    monkeypatch.setattr(
        trainer_module,
        "_resolve_string_model_config",
        lambda name, cfg: SimpleNamespace(
            model_type = "qwen3_next",
            architectures = ["Qwen3NextForCausalLM"],
            layer_types = ["linear_attention", "full_attention"],
        ),
    )
    monkeypatch.setattr(trainer_module, "patch_hybrid_linear_attention_varlen", lambda model: True)
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    fake_trainer("Qwen/Qwen3-Next-80B-A3B", config, None, Dataset.from_dict({"text": ["x"]}))
    assert config.packing is False
    assert config.padding_free is False


def test_vlm_vision_dataset_still_disables_packing():
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)

    fake_trainer(
        _vlm_model(),
        config,
        None,
        Dataset.from_dict({"images": [None], "text": ["multimodal sample"]}),
        None,
        object(),
    )

    assert config.packing is False
    assert config.padding_free is False


@pytest.mark.parametrize(
    "vision_column",
    ("pixel_values", "pixel_attention_mask", "image_grid_thw"),
)
def test_vlm_preprocessed_vision_dataset_disables_packing(vision_column):
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)

    fake_trainer(
        model = _vlm_model(),
        args = config,
        processing_class = object(),
        train_dataset = Dataset.from_dict({"input_ids": [[1]], vision_column: [None]}),
    )

    assert config.packing is False
    assert config.padding_free is False


@pytest.mark.parametrize("dict_eval", (False, True))
def test_vlm_vision_eval_dataset_disables_packing(dict_eval):
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    eval_dataset = Dataset.from_dict({"input_ids": [[1]], "pixel_values": [None]})
    if dict_eval:
        eval_dataset = {"vision": eval_dataset}

    fake_trainer(
        model = _vlm_model(),
        args = config,
        processing_class = object(),
        train_dataset = Dataset.from_dict({"text": ["text-only training sample"]}),
        eval_dataset = eval_dataset,
    )

    assert config.packing is False
    assert config.padding_free is False


def test_vlm_streaming_vision_dataset_without_metadata_disables_packing():
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    dataset = IterableDataset.from_generator(
        lambda: iter([{"images": [None], "text": "multimodal sample"}])
    )
    assert dataset.column_names is None

    fake_trainer(
        model = _vlm_model(),
        args = config,
        processing_class = object(),
        train_dataset = dataset,
    )

    assert config.packing is False
    assert config.padding_free is False
    assert next(iter(dataset))["text"] == "multimodal sample"


@pytest.mark.parametrize("data_collator", (None, object()))
def test_stateful_stream_is_not_consumed_during_detection(data_collator):
    class StatefulDataset:
        def __init__(self):
            self.rows = iter([{"text": "first"}, {"text": "second"}])

        def __iter__(self):
            return (row for row in self.rows)

    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    dataset = StatefulDataset()

    fake_trainer(
        model = _vlm_model(),
        args = config,
        processing_class = object(),
        data_collator = data_collator,
        train_dataset = dataset,
    )

    assert config.packing is False
    assert config.padding_free is False
    assert next(iter(dataset))["text"] == "first"


def test_text_model_stream_without_metadata_keeps_packing():
    class StatefulDataset:
        def __init__(self):
            self.rows = iter([{"text": "first"}, {"text": "second"}])

        def __iter__(self):
            return (row for row in self.rows)

    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)
    dataset = StatefulDataset()

    trainer = fake_trainer(
        model = _text_model(),
        args = config,
        processing_class = object(),
        train_dataset = dataset,
    )

    assert config.packing is True
    assert config.padding_free is True
    assert trainer.model._unsloth_allow_packed_overlength is True
    assert next(iter(dataset))["text"] == "first"


def test_bfd_packing_truncates_before_packing(monkeypatch):
    args = SimpleNamespace(
        dataset_num_proc = 1,
        dataset_text_field = "text",
        max_length = 4,
        packing_strategy = "bfd",
    )
    trainer = SimpleNamespace(model = None)
    dataset = Dataset.from_dict({"prompt": ["abc"], "completion": ["defghij"]})
    prepare_globals = SFTTrainer._prepare_dataset.__globals__

    def passthrough_pack_dataset(dataset, seq_length, strategy, map_kwargs):
        return dataset

    monkeypatch.setitem(prepare_globals, "pack_dataset", passthrough_pack_dataset)
    packed = SFTTrainer._prepare_dataset(
        trainer,
        dataset,
        _CharacterTokenizer(),
        args,
        True,
        None,
        "train",
    )

    assert len(packed["input_ids"][0]) == args.max_length


def test_wrapped_strategy_without_packing_still_truncates():
    args = SimpleNamespace(
        dataset_num_proc = 1,
        dataset_text_field = "text",
        max_length = 4,
        packing_strategy = "wrapped",
    )
    trainer = SimpleNamespace(model = None)
    dataset = Dataset.from_dict({"text": ["abcdefghi"]})

    prepared = SFTTrainer._prepare_dataset(
        trainer,
        dataset,
        _CharacterTokenizer(),
        args,
        False,
        None,
        "train",
    )

    assert len(prepared["input_ids"][0]) == args.max_length


@pytest.mark.parametrize("legacy_api", (False, True))
def test_wrapped_packing_preserves_overlength_tokens(monkeypatch, legacy_api):
    args_kwargs = {
        "dataset_num_proc": 1,
        "dataset_text_field": "text",
        "max_length": 4,
    }
    if not legacy_api:
        args_kwargs["packing_strategy"] = "wrapped"
    args = SimpleNamespace(**args_kwargs)
    trainer = SimpleNamespace(model = None)
    dataset = Dataset.from_dict({"text": ["abcdefghi"]})
    prepare_globals = SFTTrainer._prepare_dataset.__globals__
    pack_dataset = prepare_globals["pack_dataset"]

    def legacy_pack_dataset(
        dataset,
        seq_length,
        map_kwargs = None,
    ):
        return pack_dataset(dataset, seq_length, "wrapped", map_kwargs)

    if legacy_api:
        monkeypatch.setitem(prepare_globals, "pack_dataset", legacy_pack_dataset)

    packed = SFTTrainer._prepare_dataset(
        trainer,
        dataset,
        _CharacterTokenizer(),
        args,
        True,
        None,
        "train",
    )

    packed_ids = packed["input_ids"]
    assert sum(len(input_ids) for input_ids in packed_ids) == 9
    assert all(len(input_ids) <= args.max_length for input_ids in packed_ids)


class _DummyChild(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_seq_length = 8


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_seq_length = 16
        self.child = _DummyChild()
        self.config = SimpleNamespace(_attn_implementation = "sdpa")
        self.generation_config = SimpleNamespace(attn_implementation = "sdpa")


class _DummyTrainer:
    def __init__(self):
        self.args = SimpleNamespace(remove_unused_columns = True)
        collator_args = {
            "pad_token_id": 0,
            "completion_only_loss": False,
            "return_tensors": "pt",
        }
        optional_flags = [
            {"padding_free": True, "return_position_ids": False},
            {"padding_free": True},
            {},
        ]
        for extra in optional_flags:
            try:
                self.data_collator = DataCollatorForLanguageModeling(**collator_args, **extra)
                break
            except TypeError:
                continue
        # Ensure attributes exist even if the constructor rejected the flags.
        if not hasattr(self.data_collator, "padding_free"):
            self.data_collator.padding_free = True
        if not hasattr(self.data_collator, "return_position_ids"):
            self.data_collator.return_position_ids = False


class _PaddingFreeCollator:
    def __init__(self):
        self.padding_free = True
        self.return_position_ids = False
        self.calls = 0

    def torch_call(self, examples):
        self.calls += 1
        return {
            "input_ids": torch.tensor([[0]], dtype = torch.long),
            "examples_seen": self.calls,
        }


def test_enable_sample_packing():
    model = _DummyModel()
    trainer = _DummyTrainer()

    enable_sample_packing(model, trainer)

    # model hierarchy now allows packed overlength inputs
    assert getattr(model, "_unsloth_allow_packed_overlength") is True
    assert getattr(model.child, "_unsloth_allow_packed_overlength") is True

    collator = trainer.data_collator
    assert collator.return_position_ids is True
    assert getattr(collator, "_unsloth_packing_wrapped") is True

    examples = [
        {
            "input_ids": [0, 1, 2],
            "labels": [0, 1, 2],
            "seq_lengths": [2, 1],
        },
        {
            "input_ids": [3, 4, 5],
            "labels": [3, 4, 5],
            "seq_lengths": [3],
        },
    ]
    batch = collator.torch_call(examples)

    # packed lengths aggregated into one tensor
    assert "packed_seq_lengths" in batch
    assert torch.equal(batch["packed_seq_lengths"], torch.tensor([2, 1, 3], dtype = torch.int32))

    assert batch["input_ids"].shape == (1, 6)
    expected_positions = torch.tensor([0, 1, 0, 0, 1, 2], dtype = torch.long)
    assert torch.equal(batch["position_ids"].view(-1)[:6], expected_positions)


def test_enable_sample_packing_trl_collator(tmp_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    model, _, trainer, _ = _build_packed_training_setup(tmp_path, device)

    enable_sample_packing(model, trainer)

    examples = [
        {
            "input_ids": [0, 1, 2],
            "labels": [0, 1, 2],
            "seq_lengths": [2, 1],
        },
        {
            "input_ids": [3, 4, 5],
            "labels": [3, 4, 5],
            "seq_lengths": [3],
        },
    ]

    batch = trainer.data_collator.torch_call(examples)

    assert batch["input_ids"].shape == (1, 6)
    assert torch.equal(batch["packed_seq_lengths"], torch.tensor([2, 1, 3], dtype = torch.int32))

    expected_positions = torch.tensor([0, 1, 0, 0, 1, 2], dtype = torch.long)
    assert torch.equal(batch["position_ids"].view(-1)[:6], expected_positions)

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.free_memory()


def test_enable_padding_free_metadata():
    model = _DummyModel()
    trainer = SimpleNamespace(
        args = SimpleNamespace(remove_unused_columns = True),
        data_collator = _PaddingFreeCollator(),
    )

    enable_padding_free_metadata(model, trainer)

    assert getattr(model, "_unsloth_allow_packed_overlength") is True
    assert getattr(model.child, "_unsloth_allow_packed_overlength") is True

    collator = trainer.data_collator
    assert collator.return_position_ids is True
    assert getattr(collator, "_unsloth_padding_free_lengths_wrapped") is True

    examples = [
        {"input_ids": [0, 1, 2]},
        {"input_ids": [3, 4]},
    ]
    batch = collator.torch_call(examples)
    assert torch.equal(batch["packed_seq_lengths"], torch.tensor([3, 2], dtype = torch.int32))
    assert trainer.args.remove_unused_columns is False


def test_packing_sdpa(tmp_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    model, batch, trainer, llama_mod = _build_packed_training_setup(tmp_path, device)

    assert "packed_seq_lengths" in batch
    assert "attention_mask" not in batch
    assert batch["packed_seq_lengths"].dtype == torch.int32

    total_tokens = batch["input_ids"].size(-1)
    assert int(batch["packed_seq_lengths"].sum().item()) == total_tokens

    packed_tokens = int(batch["packed_seq_lengths"].sum().item())
    assert "position_ids" in batch
    flat_positions = batch["position_ids"].reshape(-1)[:packed_tokens]
    expected_positions = torch.cat(
        [torch.arange(length, dtype = torch.long) for length in batch["packed_seq_lengths"].tolist()]
    )
    assert torch.equal(flat_positions.cpu(), expected_positions)
    inputs = _trim_batch_to_total_tokens(batch, packed_tokens)

    seq_info = llama_mod.get_packed_info_from_kwargs(
        {"packed_seq_lengths": batch["packed_seq_lengths"]},
        inputs["input_ids"].device,
    )
    assert seq_info is not None

    original_mask = attention_dispatch_utils.build_sdpa_packed_attention_mask
    mask_calls = []
    captured_loss_labels = {}

    def _capture_mask(
        seq_info,
        dtype,
        device,
        *,
        sliding_window = None,
    ):
        mask_calls.append(tuple(seq_info[0].tolist()))
        return original_mask(
            seq_info,
            dtype = dtype,
            device = device,
            sliding_window = sliding_window,
        )

    def _capture_loss(*, logits, labels, **loss_kwargs):
        captured_loss_labels["labels"] = labels.detach().to("cpu")
        return torch.zeros((), device = logits.device, dtype = logits.dtype)

    with ExitStack() as stack:
        stack.enter_context(patch.object(attention_dispatch_utils, "HAS_FLASH_ATTENTION", False))
        stack.enter_context(patch.object(attention_dispatch_utils, "HAS_XFORMERS", False))
        stack.enter_context(
            patch.object(
                attention_dispatch_utils,
                "build_sdpa_packed_attention_mask",
                side_effect = _capture_mask,
            )
        )
        stack.enter_context(
            patch.object(
                llama_mod,
                "fast_cross_entropy_loss",
                side_effect = _capture_loss,
            )
        )
        with torch.no_grad():
            outputs = model(**inputs)

    assert mask_calls, "SDPA packed mask was not constructed"
    assert outputs.loss is not None
    assert "labels" in captured_loss_labels
    flat_loss_labels = captured_loss_labels["labels"].reshape(-1)
    boundaries = (
        torch.cumsum(batch["packed_seq_lengths"].to(device = "cpu", dtype = torch.long), dim = 0) - 1
    )
    for idx in boundaries.tolist():
        assert flat_loss_labels[idx].item() == -100
    assert torch.any(flat_loss_labels != -100)

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.free_memory()


# --- wrapped-packing source-injection robustness (reviewer.py / fork findings) --------


# fmt: off
# Named to match the unsloth_zoo helper (sourced by name, "def sft_prepare_dataset" ->
# "def _prepare_dataset"). Deliberately OMITS the "licensed under LGPLv3" header to
# emulate a newer Zoo whose header moved (dependency is only lower-bounded). Source only.
def sft_prepare_dataset(
    self, dataset, processing_class, args, packing, formatting_func, dataset_text_field
):
    do_truncation = True
    max_seq_length = 4
    used_column_names = ["text"]
    map_kwargs = {}
    dataset = processing_class(dataset, truncation = do_truncation,)
    if do_truncation and max_seq_length > 0:
        pass
    if packing:
        dataset = pack_dataset(
            dataset.select_columns(used_column_names),
            max_seq_length,
            getattr(args, "packing_strategy", "bfd"),
            map_kwargs,
        )
    return dataset
# fmt: on


def test_wrapped_packing_injection_is_drift_resistant(monkeypatch):
    # Regression: the setup used to anchor on the Zoo license comment, so a header
    # change silently no-op'd it while the truncation/pack edits still referenced its
    # variables -> NameError on every SFT prep. It must now install via the signature
    # before those references, and the pack edit must reuse the guarded
    # _unsloth_pack_has_strategy instead of re-calling _inspect.signature(pack_dataset).
    import ast
    import textwrap
    import unsloth.models.rl_replacements as rlr

    monkeypatch.setitem(rlr.RL_REPLACEMENTS, "sft_prepare_dataset", sft_prepare_dataset)

    source = (
        "def _prepare_dataset(self, dataset, processing_class, args, packing, "
        "formatting_func, dataset_text_field):\n    return dataset\n"
    )
    patched = rlr.sft_trainer_prepare_dataset("_prepare_dataset", source)

    # setup installed despite the missing header, and before it is referenced
    assert "_unsloth_wrapped_packing = packing" in patched
    assert "import inspect as _inspect" in patched
    assert patched.index("_unsloth_wrapped_packing = packing") < patched.index(
        "truncation = do_truncation and not _unsloth_wrapped_packing"
    )
    # the pack edit reuses the guarded flag (signature inspected exactly once, in setup)
    assert "if _unsloth_pack_has_strategy:" in patched
    assert patched.count("_inspect.signature(pack_dataset)") == 1
    ast.parse(textwrap.dedent(patched))


def test_require_replace_raises_on_missing_anchor():
    from unsloth.models.rl_replacements import _require_replace

    assert _require_replace("abc", "b", "B") == "aBc"
    with pytest.raises(RuntimeError):
        _require_replace("abc", "z", "Z", where = "unit test")
    # an optional edit warns once and returns the source unchanged (no dangling ref)
    assert _require_replace("abc", "z", "Z", required = False, where = "optional") == "abc"


def test_resolve_string_model_config_forwards_token(monkeypatch):
    import transformers

    captured = {}

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(name, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(is_encoder_decoder = False)

    monkeypatch.setattr(transformers, "AutoConfig", _FakeAutoConfig)

    config_arg = SimpleNamespace(
        model_init_kwargs = {
            "token": "hf_secret",
            "trust_remote_code": True,
            "cache_dir": "/tmp/cache",
            "torch_dtype": "bfloat16",  # not a config arg -> must NOT be forwarded
        }
    )
    result = trainer_module._resolve_string_model_config("org/private-hybrid", config_arg)

    assert result is not None
    assert captured.get("token") == "hf_secret"
    assert captured.get("trust_remote_code") is True
    assert captured.get("cache_dir") == "/tmp/cache"
    assert "torch_dtype" not in captured


def test_resolve_string_model_config_merges_top_level_trust_remote_code(monkeypatch):
    import transformers

    captured = {}

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(name, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(is_encoder_decoder = False)

    monkeypatch.setattr(transformers, "AutoConfig", _FakeAutoConfig)

    # SFTConfig(trust_remote_code=True) with no model_init_kwargs entry is honored
    config_arg = SimpleNamespace(model_init_kwargs = {}, trust_remote_code = True)
    trainer_module._resolve_string_model_config("org/remote-hybrid", config_arg)
    assert captured.get("trust_remote_code") is True

    # model_init_kwargs wins over the top-level flag (mirrors TRL's setdefault)
    captured.clear()
    config_arg = SimpleNamespace(
        model_init_kwargs = {"trust_remote_code": False}, trust_remote_code = True
    )
    trainer_module._resolve_string_model_config("org/remote-hybrid", config_arg)
    assert captured.get("trust_remote_code") is False
