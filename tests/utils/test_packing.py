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
    mask_packed_boundary_labels,
    mask_packed_sequence_boundaries,
    patch_hybrid_linear_attention_varlen,
)

import inspect
import logging
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset, IterableDataset
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


class _FakeConfig(SimpleNamespace):
    # get_transformers_model_type() resolves through to_dict(), which SimpleNamespace lacks.
    def to_dict(self):
        return dict(self.__dict__)


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
    return SimpleNamespace(config = _FakeConfig(layer_types = ["linear_attention", "full_attention"]))


def _gemma3_model():
    # Has layer_types but no linear_attention -> must NOT be flagged as hybrid.
    return SimpleNamespace(
        config = _FakeConfig(
            model_type = "gemma3", layer_types = ["sliding_attention", "full_attention"]
        ),
    )


def _dense_qwen3_model():
    return SimpleNamespace(
        config = _FakeConfig(model_type = "qwen3", architectures = ["Qwen3ForCausalLM"])
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


def _make_fake_mamba2_fused():
    def mamba2_split_conv1d_scan_combined(
        *args,
        seq_idx = None,
        **kwargs,
    ):
        mamba2_split_conv1d_scan_combined.calls.append(seq_idx)
        return args[0] if args else None

    mamba2_split_conv1d_scan_combined.calls = []
    return mamba2_split_conv1d_scan_combined


class _FakeNemotronHMamba2Mixer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(4, 4, 3, groups = 4)
        self.A_log = torch.nn.Parameter(torch.zeros(4))
        self.mamba2_split_conv1d_scan_combined = _make_fake_mamba2_fused()

    def forward(self, hidden_states, **kwargs):
        fused_kwargs = dict(kwargs)
        return self.mamba2_split_conv1d_scan_combined(hidden_states, **fused_kwargs)


class _FakeMamba2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()  # no markers -> module-level detection
        self.mixer = _FakeNemotronHMamba2Mixer()

    def forward(
        self,
        input_ids = None,
        packed_seq_lengths = None,
        use_cache = None,
        **kwargs,
    ):
        return self.mixer(input_ids.float(), **kwargs)


def test_is_hybrid_linear_attention_detects_mamba2_mixer():
    is_hybrid = trainer_module._is_hybrid_linear_attention_model
    assert is_hybrid(_FakeMamba2Model()) is True


def test_patch_mamba2_varlen_injects_seq_idx(monkeypatch):
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    model = _FakeMamba2Model()
    fused_orig = model.mixer.mamba2_split_conv1d_scan_combined

    assert patch_hybrid_linear_attention_varlen(model) is True
    assert patch_hybrid_linear_attention_varlen(model) is True  # idempotent

    fused_orig.calls.clear()
    packing_module._HYBRID_WARNED.clear()
    model(
        input_ids = torch.zeros(1, 6, dtype = torch.long),
        packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
        use_cache = False,
    )
    seq_idx = fused_orig.calls[-1]
    assert seq_idx is not None
    assert seq_idx.tolist() == [[0, 0, 1, 2, 2, 2]]
    assert not packing_module._HYBRID_WARNED

    fused_orig.calls.clear()
    model(
        input_ids = torch.zeros(1, 6, dtype = torch.long),
        packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
        use_cache = True,
    )
    assert fused_orig.calls[-1] is None


def test_patch_mamba2_varlen_kwargs_only_stub_fail_closed(monkeypatch):
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    model = _FakeMamba2Model()

    def stub(*args, **kwargs):
        return None

    model.mixer.mamba2_split_conv1d_scan_combined = stub
    assert patch_hybrid_linear_attention_varlen(model) is False


def test_patch_mamba2_varlen_rebinds_compiled_module_alias(monkeypatch):
    # Unsloth compiles mixer.forward into unsloth_compiled_cache with a module-global
    # fused kernel import. Wrapping only transformers.modeling_* leaves that alias
    # on the original and the handshake aborts on a real Nemotron-H train.
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    import sys
    import types

    fused = _make_fake_mamba2_fused()
    compiled = types.ModuleType("unsloth_compiled_cache.NemotronHMamba2Mixer")
    compiled.mamba_split_conv1d_scan_combined = fused
    sys.modules[compiled.__name__] = compiled

    class _CompiledNemotronHMamba2Mixer(_FakeNemotronHMamba2Mixer):
        def __init__(self):
            super().__init__()
            self.mamba2_split_conv1d_scan_combined = fused

        def forward(self, hidden_states, **fused_kwargs):
            return compiled.mamba_split_conv1d_scan_combined(
                hidden_states, seq_idx = None, **fused_kwargs
            )

    class _CompiledModel(_FakeMamba2Model):
        def __init__(self):
            super().__init__()
            self.mixer = _CompiledNemotronHMamba2Mixer()

    try:
        model = _CompiledModel()
        assert patch_hybrid_linear_attention_varlen(model) is True
        fused.calls.clear()
        model(
            input_ids = torch.zeros(1, 6, dtype = torch.long),
            packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
            use_cache = False,
        )
        assert fused.calls[-1] is not None
        assert fused.calls[-1].tolist() == [[0, 0, 1, 2, 2, 2]]
        assert compiled.mamba_split_conv1d_scan_combined is not fused
    finally:
        sys.modules.pop(compiled.__name__, None)


def _make_hub_mamba2_namespace(module_name):
    """A modeling module shaped like transformers' dynamic module loading.

    The module is registered in ``sys.modules`` and its functions are exec'd
    into the module dict, so ``cuda_kernels_forward.__globals__`` *is* that
    dict - which is what its LOAD_GLOBAL of the fused kernel resolves through.
    """
    import sys
    import types

    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    exec(
        """
def mamba_split_conv1d_scan_combined(*args, seq_idx=None, **kwargs):
    mamba_split_conv1d_scan_combined.calls.append(seq_idx)
    return args[0]
mamba_split_conv1d_scan_combined.calls = []
def cuda_kernels_forward(self, hidden_states, cache_params=None, attention_mask=None):
    return mamba_split_conv1d_scan_combined(hidden_states, seq_idx=None)
""",
        module.__dict__,
    )
    return module


def test_patch_mamba2_varlen_rewrites_cuda_kernels_forward_global(monkeypatch):
    # transformers 5.5 Nemotron-H LOAD_GLOBALs mamba_split_conv1d_scan_combined
    # from cuda_kernels_forward and hardcodes seq_idx=None. That is the H200 abort.
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    import sys
    import types

    module = _make_hub_mamba2_namespace("transformers_modules.fake_nemotron_h")
    ns = module.__dict__

    class _HubNemotronHMamba2Mixer(_FakeNemotronHMamba2Mixer):
        def __init__(self):
            super().__init__()
            del self.mamba2_split_conv1d_scan_combined
            self.cuda_kernels_forward = types.MethodType(ns["cuda_kernels_forward"], self)

        def forward(self, hidden_states, **kwargs):
            return self.cuda_kernels_forward(hidden_states)

    _HubNemotronHMamba2Mixer.__module__ = module.__name__

    class _HubModel(_FakeMamba2Model):
        def __init__(self):
            super().__init__()
            self.mixer = _HubNemotronHMamba2Mixer()

    try:
        model = _HubModel()
        assert patch_hybrid_linear_attention_varlen(model) is True
        ns["mamba_split_conv1d_scan_combined"].calls.clear()
        model(
            input_ids = torch.zeros(1, 6, dtype = torch.long),
            packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
            use_cache = False,
        )
        seq_idx = ns["mamba_split_conv1d_scan_combined"].calls[-1]
        assert seq_idx is not None
        assert seq_idx.tolist() == [[0, 0, 1, 2, 2, 2]]
    finally:
        sys.modules.pop(module.__name__, None)


def test_patch_mamba2_varlen_unreachable_kernel_namespace_aborts(monkeypatch):
    # The install reaches module globals by name; it deliberately does not hunt
    # references through gc or closures. A kernel reachable only from an
    # unregistered namespace must fail closed, never train unpacked.
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    import types

    ns: dict = {"__name__": "nowhere.fake_nemotron_h"}
    exec(
        """
def mamba_split_conv1d_scan_combined(*args, seq_idx=None, **kwargs):
    return args[0]
def cuda_kernels_forward(self, hidden_states, cache_params=None, attention_mask=None):
    return mamba_split_conv1d_scan_combined(hidden_states, seq_idx=None)
""",
        ns,
    )

    class _HiddenNemotronHMamba2Mixer(_FakeNemotronHMamba2Mixer):
        def __init__(self):
            super().__init__()
            del self.mamba2_split_conv1d_scan_combined
            self.cuda_kernels_forward = types.MethodType(ns["cuda_kernels_forward"], self)

        def forward(self, hidden_states, **kwargs):
            return self.cuda_kernels_forward(hidden_states)

    class _HiddenModel(_FakeMamba2Model):
        def __init__(self):
            super().__init__()
            self.mixer = _HiddenNemotronHMamba2Mixer()

    model = _HiddenModel()
    assert patch_hybrid_linear_attention_varlen(model) is True
    with pytest.raises(RuntimeError, match = "varlen conv/scan wrappers were not both invoked"):
        model(
            input_ids = torch.zeros(1, 6, dtype = torch.long),
            packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
            use_cache = False,
        )


def test_patch_mamba2_varlen_reaches_compiled_module_global(monkeypatch):
    # Unsloth compiles mixer methods into unsloth_compiled_cache as free
    # functions that LOAD_GLOBAL the fused kernel from the compiled module's own
    # globals. Reassigning the name there is what the call resolves through.
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    import sys
    import types

    compiled = types.ModuleType("unsloth_compiled_cache.NemotronHMamba2Mixer")
    ns = compiled.__dict__
    ns["__name__"] = compiled.__name__
    exec(
        """
def mamba_split_conv1d_scan_combined(*args, seq_idx=None, **kwargs):
    mamba_split_conv1d_scan_combined.calls.append(seq_idx)
    return args[0]
mamba_split_conv1d_scan_combined.calls = []
def NemotronHMamba2Mixer_cuda_kernels_forward(self, hidden_states, cache_params=None, attention_mask=None):
    return mamba_split_conv1d_scan_combined(hidden_states, seq_idx=None)
""",
        ns,
    )
    sys.modules[compiled.__name__] = compiled

    class _CompiledNemotronHMamba2Mixer(_FakeNemotronHMamba2Mixer):
        def __init__(self):
            super().__init__()
            del self.mamba2_split_conv1d_scan_combined
            self.cuda_kernels_forward = types.MethodType(
                ns["NemotronHMamba2Mixer_cuda_kernels_forward"], self
            )

        def forward(self, hidden_states, **kwargs):
            return self.cuda_kernels_forward(hidden_states)

    _CompiledNemotronHMamba2Mixer.__module__ = compiled.__name__

    class _CompiledModel(_FakeMamba2Model):
        def __init__(self):
            super().__init__()
            self.mixer = _CompiledNemotronHMamba2Mixer()

    try:
        model = _CompiledModel()
        assert patch_hybrid_linear_attention_varlen(model) is True
        ns["mamba_split_conv1d_scan_combined"].calls.clear()
        model(
            input_ids = torch.zeros(1, 6, dtype = torch.long),
            packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
            use_cache = False,
        )
        seq_idx = ns["mamba_split_conv1d_scan_combined"].calls[-1]
        assert seq_idx is not None
        assert seq_idx.tolist() == [[0, 0, 1, 2, 2, 2]]
    finally:
        sys.modules.pop(compiled.__name__, None)


def test_patch_mamba2_varlen_overwrites_stale_compiled_import(monkeypatch):
    # Compiler imports mamba_split_conv1d_scan_combined into unsloth_compiled_cache
    # before mixer __init__ stores the real kernel on the modeling module.
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    import sys
    import types

    def stale(
        *args,
        seq_idx = None,
        **kwargs,
    ):
        stale.calls.append(seq_idx)
        return args[0] if args else None

    stale.calls = []
    real = _make_fake_mamba2_fused()
    compiled = types.ModuleType("unsloth_compiled_cache.NemotronHMamba2Mixer")
    compiled.mamba_split_conv1d_scan_combined = stale
    sys.modules[compiled.__name__] = compiled

    class _StaleNemotronHMamba2Mixer(_FakeNemotronHMamba2Mixer):
        def __init__(self):
            super().__init__()
            self.mamba2_split_conv1d_scan_combined = real

        def cuda_kernels_forward(
            self,
            hidden_states,
            cache_params = None,
            attention_mask = None,
        ):
            return compiled.mamba_split_conv1d_scan_combined(hidden_states, seq_idx = None)

        def forward(self, hidden_states, **kwargs):
            return self.cuda_kernels_forward(hidden_states)

    _StaleNemotronHMamba2Mixer.__module__ = compiled.__name__

    class _StaleModel(_FakeMamba2Model):
        def __init__(self):
            super().__init__()
            self.mixer = _StaleNemotronHMamba2Mixer()

    try:
        model = _StaleModel()
        assert patch_hybrid_linear_attention_varlen(model) is True
        real.calls.clear()
        stale.calls.clear()
        model(
            input_ids = torch.zeros(1, 6, dtype = torch.long),
            packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
            use_cache = False,
        )
        assert stale.calls == []
        assert real.calls[-1] is not None
        assert real.calls[-1].tolist() == [[0, 0, 1, 2, 2, 2]]
        assert compiled.mamba_split_conv1d_scan_combined is not stale
    finally:
        sys.modules.pop(compiled.__name__, None)


def test_patch_mamba2_varlen_clears_padded_mask_for_fused_path(monkeypatch):
    # transformers 5.5 only calls mamba_split_conv1d_scan_combined when
    # attention_mask is all-ones. Packed batches still have pad zeros, so the
    # mixer takes mamba_chunk_scan_combined and the fused wrapper never runs.
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")

    def mamba_chunk_scan_combined(
        *args,
        seq_idx = None,
        **kwargs,
    ):
        mamba_chunk_scan_combined.calls.append(seq_idx)
        return args[0] if args else None

    mamba_chunk_scan_combined.calls = []

    class _MaskedNemotronHMamba2Mixer(_FakeNemotronHMamba2Mixer):
        def cuda_kernels_forward(
            self,
            hidden_states,
            cache_params = None,
            attention_mask = None,
        ):
            input_not_masked = attention_mask is None or bool(torch.all(attention_mask == 1))
            if self.training and cache_params is None and input_not_masked:
                return self.mamba2_split_conv1d_scan_combined(hidden_states, seq_idx = None)
            return mamba_chunk_scan_combined(hidden_states, seq_idx = None)

        def forward(
            self,
            hidden_states,
            cache_params = None,
            attention_mask = None,
            **kwargs,
        ):
            return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)

    class _MaskedModel(_FakeMamba2Model):
        def __init__(self):
            super().__init__()
            self.mixer = _MaskedNemotronHMamba2Mixer()

    model = _MaskedModel()
    fused_orig = model.mixer.mamba2_split_conv1d_scan_combined
    model.train()
    assert patch_hybrid_linear_attention_varlen(model) is True
    fused_orig.calls.clear()
    mamba_chunk_scan_combined.calls.clear()
    model(
        input_ids = torch.zeros(1, 6, dtype = torch.long),
        packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]]),
        use_cache = False,
    )
    assert mamba_chunk_scan_combined.calls == []
    assert fused_orig.calls[-1] is not None
    assert fused_orig.calls[-1].tolist() == [[0, 0, 1, 2, 2, 2]]


def test_patch_mamba2_varlen_clears_mask_on_compiled_mixer_forward(monkeypatch):
    # H200: compiled NemotronHMamba2Mixer_forward passes the padded mask through
    # to cuda_kernels_forward, which then skips the fused kernel.
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    import sys
    import types

    compiled = types.ModuleType("unsloth_compiled_module_nemotron_h")
    ns = compiled.__dict__
    ns["__name__"] = compiled.__name__
    ns["torch"] = torch
    exec(
        """
def mamba_split_conv1d_scan_combined(*args, seq_idx=None, **kwargs):
    mamba_split_conv1d_scan_combined.calls.append(seq_idx)
    return args[0]
mamba_split_conv1d_scan_combined.calls = []
def mamba_chunk_scan_combined(*args, seq_idx=None, **kwargs):
    mamba_chunk_scan_combined.calls.append(seq_idx)
    return args[0]
mamba_chunk_scan_combined.calls = []
def cuda_kernels_forward(self, hidden_states, cache_params=None, attention_mask=None):
    input_not_masked = attention_mask is None or bool(torch.all(attention_mask == 1))
    if self.training and cache_params is None and input_not_masked:
        return mamba_split_conv1d_scan_combined(hidden_states, seq_idx=None)
    return mamba_chunk_scan_combined(hidden_states, seq_idx=None)
def NemotronHMamba2Mixer_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)
""",
        ns,
    )
    sys.modules[compiled.__name__] = compiled

    class _CompiledNemotronHMamba2Mixer(_FakeNemotronHMamba2Mixer):
        def __init__(self):
            super().__init__()
            del self.mamba2_split_conv1d_scan_combined
            self.cuda_kernels_forward = types.MethodType(ns["cuda_kernels_forward"], self)

        def forward(
            self,
            hidden_states,
            cache_params = None,
            attention_mask = None,
            **kwargs,
        ):
            return ns["NemotronHMamba2Mixer_forward"](
                self,
                hidden_states = hidden_states,
                cache_params = cache_params,
                attention_mask = attention_mask,
                **kwargs,
            )

    _CompiledNemotronHMamba2Mixer.__module__ = compiled.__name__

    class _EmptyCache:
        def has_previous_state(self, layer_idx):
            return False

    class _CompiledModel(_FakeMamba2Model):
        def __init__(self):
            super().__init__()
            self.mixer = _CompiledNemotronHMamba2Mixer()

        def forward(
            self,
            input_ids = None,
            packed_seq_lengths = None,
            use_cache = None,
            **kwargs,
        ):
            # Nemotron-H builds an empty Cache inside the model and passes it
            # to mixers; it is not a trainer-batch kwarg.
            return self.mixer(
                input_ids.float(),
                cache_params = _EmptyCache(),
                **kwargs,
            )

    try:
        model = _CompiledModel()
        model.train()
        assert patch_hybrid_linear_attention_varlen(model) is True
        ns["mamba_split_conv1d_scan_combined"].calls.clear()
        ns["mamba_chunk_scan_combined"].calls.clear()
        model(
            input_ids = torch.zeros(1, 6, dtype = torch.long),
            packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
            attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]]),
            use_cache = False,
        )
        assert ns["mamba_chunk_scan_combined"].calls == []
        seq_idx = ns["mamba_split_conv1d_scan_combined"].calls[-1]
        assert seq_idx is not None
        assert seq_idx.tolist() == [[0, 0, 1, 2, 2, 2]]
    finally:
        sys.modules.pop(compiled.__name__, None)


def test_patch_mamba2_varlen_wraps_shared_kernel_once_per_model(monkeypatch):
    # Nemotron-H has one mixer per layer sharing a single fused kernel name.
    # Re-wrapping per mixer nested ~one frame per layer and hit the recursion
    # limit on the first packed forward.
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")

    fused = _make_fake_mamba2_fused()

    class _SharedNemotronHMamba2Mixer(_FakeNemotronHMamba2Mixer):
        def __init__(self):
            super().__init__()
            self.mamba2_split_conv1d_scan_combined = fused

    class _ManyLayerModel(_FakeMamba2Model):
        def __init__(self):
            super().__init__()
            self.mixer = _SharedNemotronHMamba2Mixer()
            self.layers = torch.nn.ModuleList([_SharedNemotronHMamba2Mixer() for _ in range(24)])

    model = _ManyLayerModel()
    assert patch_hybrid_linear_attention_varlen(model) is True

    installed = model.mixer.mamba2_split_conv1d_scan_combined
    assert installed is not fused
    assert installed.__wrapped__ is fused  # exactly one wrapper deep
    for layer in model.layers:
        assert layer.mamba2_split_conv1d_scan_combined is installed

    fused.calls.clear()
    model(
        input_ids = torch.zeros(1, 6, dtype = torch.long),
        packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
        use_cache = False,
    )
    assert fused.calls[-1].tolist() == [[0, 0, 1, 2, 2, 2]]


def test_patch_mamba2_varlen_skips_stale_seq_idx_after_training(monkeypatch):
    # generate() reaches the decoder without the wrapped model.forward, so the
    # mixers still held the training step's seq_idx and the kernel rejected it
    # with "seq_idx must have shape (batch_size, seqlen)".
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")

    class _GeneratingModel(_FakeMamba2Model):
        def generate(self, hidden_states):
            return self.mixer(hidden_states)

    model = _GeneratingModel()
    fused_orig = model.mixer.mamba2_split_conv1d_scan_combined
    assert patch_hybrid_linear_attention_varlen(model) is True

    model(
        input_ids = torch.zeros(1, 6, dtype = torch.long),
        packed_seq_lengths = torch.tensor([2, 1, 3], dtype = torch.int32),
        use_cache = False,
    )
    assert fused_orig.calls[-1].tolist() == [[0, 0, 1, 2, 2, 2]]

    # A shorter prompt must not inherit the 6-token packed boundaries.
    fused_orig.calls.clear()
    model.generate(torch.zeros(1, 3, 4))
    assert fused_orig.calls == [None]


def test_patch_mamba2_varlen_no_fused_dispatch_aborts(monkeypatch):
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")

    class _SilentNemotronHMamba2Mixer(_FakeNemotronHMamba2Mixer):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class _SilentModel(_FakeMamba2Model):
        def __init__(self):
            super().__init__()
            self.mixer = _SilentNemotronHMamba2Mixer()

    model = _SilentModel()
    assert patch_hybrid_linear_attention_varlen(model) is True
    with pytest.raises(RuntimeError, match = "not both invoked"):
        model(
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
        config = _FakeConfig(
            architectures = ["Gemma4ForConditionalGeneration"],
            model_type = "gemma4",
            vision_config = SimpleNamespace(),
        ),
        max_seq_length = 16,
    )


def _text_model():
    return SimpleNamespace(
        config = _FakeConfig(
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
        config = _FakeConfig(
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
        config = _FakeConfig(
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
        config = _FakeConfig(
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
        lambda name, cfg: _FakeConfig(
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


def _build_trl_language_modeling_collator():
    """Build TRL's SFT collator with only the fields the installed TRL accepts.

    The dataclass fields drift between TRL releases, so hardcoding a kwarg set
    breaks whenever upstream drops one: ``return_position_ids`` only existed
    around TRL 0.22, and ``completion_only_loss`` was removed from this collator
    in TRL 1.7.0 (huggingface/trl#6037, commit f9aeb59) when label masking moved
    into dataset preparation. Filtering against the live signature keeps the
    dummy trainer faithful to whatever TRL is installed.
    """
    wanted = {
        "pad_token_id": 0,
        "completion_only_loss": False,
        "return_tensors": "pt",
        "padding_free": True,
        "return_position_ids": False,
    }
    try:
        accepted = set(inspect.signature(DataCollatorForLanguageModeling).parameters)
    except (TypeError, ValueError):
        accepted = {"pad_token_id"}
    collator = DataCollatorForLanguageModeling(
        **{key: value for key, value in wanted.items() if key in accepted}
    )
    # Ensure attributes exist even when this TRL has no such field.
    if not hasattr(collator, "padding_free"):
        collator.padding_free = True
    if not hasattr(collator, "return_position_ids"):
        collator.return_position_ids = False
    return collator


class _DummyTrainer:
    def __init__(self):
        self.args = SimpleNamespace(remove_unused_columns = True)
        self.data_collator = _build_trl_language_modeling_collator()


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


def test_enable_sample_packing_only_requires_torch_call():
    """Packing must not depend on optional TRL collator fields.

    TRL keeps adding and removing fields on its SFT collator, so
    ``enable_sample_packing`` is only allowed to require ``torch_call``.
    """

    class _MinimalCollator:
        def torch_call(self, examples):
            return {"input_ids": torch.tensor([[0, 1, 2, 3, 4, 5]], dtype = torch.long)}

    trainer = SimpleNamespace(
        args = SimpleNamespace(remove_unused_columns = True),
        data_collator = _MinimalCollator(),
    )

    enable_sample_packing(_DummyModel(), trainer)

    collator = trainer.data_collator
    assert getattr(collator, "_unsloth_packing_wrapped") is True
    assert trainer.args.remove_unused_columns is False

    batch = collator.torch_call(
        [
            {"input_ids": [0, 1, 2], "seq_lengths": [2, 1]},
            {"input_ids": [3, 4, 5], "seq_lengths": [3]},
        ]
    )
    assert torch.equal(batch["packed_seq_lengths"], torch.tensor([2, 1, 3], dtype = torch.int32))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "builds a real 4bit model on an accelerator"
)
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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "builds a real 4bit model on an accelerator"
)
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
    max_seq_length = getattr(args, "max_length", 0)
    if max_seq_length == 0: max_seq_length = getattr(args, "max_seq_length", 0)
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
    # the max_length seed is normalised, or a padding-free None stops raw truncation
    assert 'max_seq_length = getattr(args, "max_length", 0) or 0' in patched
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


def _warn_text_model():
    return SimpleNamespace(
        config = _FakeConfig(architectures = ["LlamaForCausalLM"], model_type = "llama"),
        max_seq_length = 16,
    )


def test_packing_skip_warning_is_accurate(monkeypatch, caplog):
    # Two things the message used to get wrong: it blamed a "custom data collator" for
    # UNSLOTH_RETURN_LOGITS (which unsloth sets itself for compute_metrics), and it quoted a
    # token limit read before max_seq_length / max_length / the model limit are reconciled.
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(
        packing = True,
        padding_free = None,
        remove_unused_columns = True,
        max_seq_length = 4096,
        max_length = 512,
    )

    with caplog.at_level(logging.WARNING, logger = "unsloth.trainer"):
        fake_trainer(
            model = _warn_text_model(),
            args = config,
            train_dataset = Dataset.from_dict({"text": ["sample"]}),
        )

    messages = [r.message for r in caplog.records if "packing=True ignored" in r.message]
    assert len(messages) == 1
    assert "UNSLOTH_RETURN_LOGITS" in messages[0]
    assert "custom data collator" not in messages[0]
    assert "4096" not in messages[0] and "512" not in messages[0]
    # compute_metrics is one of several setters, so the message must not name it.
    assert "compute_metrics" not in messages[0]


def test_packing_skip_warning_keeps_custom_collator_reason(monkeypatch, caplog):
    # A passed collator must still be named as the cause; the env-var fallback is only for
    # the case where nothing else blocks packing.
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    fake_trainer = _patch_fake_sft_trainer()
    config = SimpleNamespace(packing = True, padding_free = None, remove_unused_columns = True)

    with caplog.at_level(logging.WARNING, logger = "unsloth.trainer"):
        fake_trainer(
            model = _warn_text_model(),
            args = config,
            data_collator = lambda features: features,
            train_dataset = Dataset.from_dict({"text": ["sample"]}),
        )

    messages = [r.message for r in caplog.records if "packing=True ignored" in r.message]
    assert len(messages) == 1
    assert "custom data collator" in messages[0]
    assert "UNSLOTH_RETURN_LOGITS" not in messages[0]


# --- packed-boundary guard on the fused-CE path ---------------------------------------
# mask_packed_sequence_boundaries needs shifted labels, so fused-CE paths (which shift
# internally) call mask_packed_boundary_labels, the pre-shift equivalent.


def test_mask_packed_boundary_labels_masks_next_document_first_token():
    labels = torch.arange(6, dtype = torch.long).view(1, 6)
    out = mask_packed_boundary_labels(labels, torch.tensor([2, 1, 3], dtype = torch.int32))
    # Docs start at 0, 2, 3; masking their first token stops the previous doc predicting
    # it. Slot 0 is the out-of-range redirect: harmless, the shift discards labels[0].
    assert out.reshape(-1).tolist() == [-100, 1, -100, -100, 4, 5]
    # out-of-place
    assert labels.reshape(-1).tolist() == [0, 1, 2, 3, 4, 5]
    assert out.shape == labels.shape
    assert out.dtype == labels.dtype


def test_mask_packed_boundary_labels_matches_the_shifted_guard():
    """The two entry points must mask exactly the same CE targets."""
    labels = torch.arange(100, 112, dtype = torch.long).view(1, 12)
    lengths = torch.tensor([5, 4, 3], dtype = torch.int32)

    # Route A: shift, then the in-place guard.
    shift_a = torch.empty_like(labels)
    shift_a[..., :-1] = labels[..., 1:]
    shift_a[..., -1] = -100
    mask_packed_sequence_boundaries(shift_a, lengths)

    # Route B: the raw-label guard, then shift (what fused CE does).
    masked = mask_packed_boundary_labels(labels, lengths)
    shift_b = torch.empty_like(masked)
    shift_b[..., :-1] = masked[..., 1:]
    shift_b[..., -1] = -100

    assert torch.equal(shift_a, shift_b)


def test_mask_packed_boundary_labels_is_idempotent_on_trl_masked_labels():
    """TRL already sets labels[position_ids == 0] = -100, so the guard is a no-op on it."""
    lengths = torch.tensor([2, 1, 3], dtype = torch.int32)
    labels = torch.arange(6, dtype = torch.long).view(1, 6)
    position_ids = torch.tensor([[0, 1, 0, 0, 1, 2]], dtype = torch.long)
    trl_labels = labels.clone()
    trl_labels[position_ids == 0] = -100

    once = mask_packed_boundary_labels(trl_labels, lengths)
    twice = mask_packed_boundary_labels(once, lengths)
    assert torch.equal(once, trl_labels)
    assert torch.equal(twice, once)


def test_mask_packed_boundary_labels_is_a_noop_without_packing():
    labels = torch.arange(6, dtype = torch.long).view(1, 6)
    assert mask_packed_boundary_labels(labels, None) is labels
    assert mask_packed_boundary_labels(labels, torch.tensor([], dtype = torch.int32)) is labels
    assert mask_packed_boundary_labels(None, torch.tensor([2, 4])) is None


def test_mask_packed_boundary_labels_tolerates_pad_to_multiple_of():
    # Trailing pad beyond sum(seq_lengths) stays -100, and no index goes OOB.
    labels = torch.tensor([[10, 11, 12, 13, -100, -100]], dtype = torch.long)
    out = mask_packed_boundary_labels(labels, torch.tensor([2, 2], dtype = torch.int32))
    assert out.reshape(-1).tolist() == [10, 11, -100, 13, -100, -100]


def test_mask_packed_boundary_labels_lengths_covering_whole_row():
    # cumsum == numel: the redirect must not corrupt a real target.
    labels = torch.arange(4, dtype = torch.long).view(1, 4)
    out = mask_packed_boundary_labels(labels, [2, 2])
    assert out.reshape(-1).tolist() == [-100, 1, -100, 3]


# ==========================================================================
# Each test below fails when its production hunk is reverted.
# 1 + 2. the fused-CE call sites (llama.py / mistral.py)
# ==========================================================================
class _StubInner(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden

    def forward(self, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state = self.hidden,
            past_key_values = None,
            hidden_states = None,
            attentions = None,
        )


def _make_stub_causal_lm(
    hidden_size = 8,
    vocab = 16,
    seq = 8,
):
    hidden = torch.zeros(1, seq, hidden_size)
    model = _StubInner(hidden)
    lm_head = torch.nn.Linear(hidden_size, vocab, bias = False)
    stub = SimpleNamespace(
        model = model,
        lm_head = lm_head,
        # Mistral's `elif self.training:` mask branch is only reached without xformers,
        # so omitting this passes locally but AttributeErrors on CI.
        training = True,
        config = SimpleNamespace(
            output_attentions = False,
            output_hidden_states = False,
            use_return_dict = True,
            model_type = "llama",
            final_logit_softcapping = 0,
            logit_scale = 0,
            torch_dtype = torch.float32,
        ),
    )
    return stub


@pytest.mark.parametrize("module_name", ["llama", "mistral"])
def test_fused_ce_branch_masks_packed_boundaries(monkeypatch, module_name):
    """The fused-CE branch must hand boundary-masked labels to the kernel."""
    import importlib

    mod = importlib.import_module(f"unsloth.models.{module_name}")
    seq = 8
    stub = _make_stub_causal_lm(seq = seq)

    seen = {}

    def _fake_fused(**kwargs):
        seen["labels"] = kwargs["labels"].clone()
        return torch.zeros((), requires_grad = False)

    monkeypatch.setattr(mod, "unsloth_fused_ce_loss", _fake_fused)
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)

    if module_name == "llama":
        forward = mod.CausalLM_fast_forward(lambda *a, **k: None)
    else:
        forward = mod.MistralForCausalLM_fast_forward

    labels = torch.arange(seq, dtype = torch.long).view(1, seq)
    forward(
        stub,
        input_ids = torch.zeros(1, seq, dtype = torch.long),
        labels = labels,
        packed_seq_lengths = torch.tensor([3, 5], dtype = torch.int32),
    )

    got = seen["labels"].reshape(-1).tolist()
    # slot 3 (first token of doc 2) is dropped; slot 0 is the harmless redirect.
    assert got == [-100, 1, 2, -100, 4, 5, 6, 7], got
    # out-of-place
    assert labels.reshape(-1).tolist() == list(range(seq))


# 3. the collator wrappers must leave boundary targets in place: unsloth_zoo counts
#    num_items_in_batch off this batch and already deducts them
class _UnmaskedPackingCollator:
    """Padding-free collator that does NOT pre-mask boundaries, like TRL < 0.24 - a test
    built on TRL 0.24+ output would pass either way."""

    def __init__(self):
        self.padding_free = True
        self.return_position_ids = False

    def torch_call(self, examples):
        ids = [i for ex in examples for i in ex["input_ids"]]
        return {
            "input_ids": torch.tensor([ids], dtype = torch.long),
            "labels": torch.tensor([ids], dtype = torch.long),
        }


def _zoo_num_items_in_batch(batch):
    """The count unsloth_zoo._unsloth_get_batch_samples derives from a batch."""
    count = int((batch["labels"][..., 1:] != -100).sum())
    lengths = batch.get("packed_seq_lengths")
    if lengths is not None:
        count -= int(torch.count_nonzero(lengths > 0)) - 1
    return count


@pytest.mark.parametrize("wrapper", [enable_sample_packing, enable_padding_free_metadata])
def test_collator_keeps_boundary_targets_for_the_num_items_deduction(wrapper):
    model = SimpleNamespace(max_seq_length = 16, children = lambda: [])
    trainer = SimpleNamespace(
        args = SimpleNamespace(remove_unused_columns = True),
        data_collator = _UnmaskedPackingCollator(),
    )
    wrapper(model, trainer)

    batch = trainer.data_collator.torch_call(
        [
            {"input_ids": [10, 11, 12], "seq_lengths": [2, 1]},
            {"input_ids": [13, 14, 15], "seq_lengths": [3]},
        ]
    )
    assert batch["labels"].reshape(-1).tolist() == [10, 11, 12, 13, 14, 15]
    # docs [10,11] [12] [13,14,15] -> 1 + 0 + 2 real CE targets
    assert _zoo_num_items_in_batch(batch) == 3


# 4. idempotence, discriminating (an identity helper must not pass)
def test_guard_is_idempotent_and_actually_masks():
    lengths = torch.tensor([2, 1, 3], dtype = torch.int32)
    labels = torch.arange(6, dtype = torch.long).view(1, 6)
    once = mask_packed_boundary_labels(labels, lengths)
    twice = mask_packed_boundary_labels(once, lengths)
    assert torch.equal(twice, once)
    # idempotence alone is trivial for an identity helper, so pin the values
    assert once.reshape(-1).tolist() == [-100, 1, -100, -100, 4, 5]
