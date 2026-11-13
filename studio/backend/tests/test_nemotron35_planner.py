# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Memory planning for NVIDIA Nemotron 3.5 Bolt and Super (arch nemotron_h_moe).

Measured on llama.cpp b11642 + ggml-org#30520 against Studio's estimates:

- The offload layout raised on the per-layer ``head_count_kv`` array and, past that,
  would have charged all 88 layers a KV cache and no Mamba state.
- The embedded NextN head (2987 MiB on Super UD-IQ2_XXS) is never loaded without
  ``--spec-type draft-mtp``, but the context fit priced it anyway.
- The routed experts run in ``moe_latent_size`` (1024), not n_embd (4096), so Super's
  compute width was over-priced; Bolt's per-layer ``expert_used_count`` was read as
  missing, which dropped the expert term and left Bolt's compute 6 MiB short.
- The DSpark draft context re-reserves a flat 528.5 MiB block graph that the drafter
  compute estimate missed, leaving Bolt + DSpark 177-332 MiB short.

Every other architecture must price exactly as before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

MIB = 1024 * 1024


class _StubField:
    def __init__(self, value):
        self._value = value

    def contents(self):
        return self._value


class _StubTensor:
    def __init__(self, name, n_bytes):
        self.name = name
        self.n_bytes = n_bytes


class _StubReader:
    def __init__(self, fields, tensors):
        self.fields = {k: _StubField(v) for k, v in fields.items()}
        self.tensors = tensors


# 8 layers in Nemotron-H order: Mamba (no attention, no FFN), MoE FFN, attention.
_HEADS = [0, 0, 2, 0, 0, 2, 0, 0]
_FFN = [0, 1536, 0, 0, 2048, 0, 0, 2304]
_SSM = {"inner_size": 8192, "state_size": 128, "group_count": 8, "conv_kernel": 4}


def _nemotron_fields(
    arch = "nemotron_h_moe",
    heads = _HEADS,
    used = (0, 9, 0, 0, 9, 0, 0, 9),
    **extra,
):
    fields = {
        "general.architecture": arch,
        f"{arch}.block_count": len(heads),
        f"{arch}.attention.head_count_kv": list(heads),
        f"{arch}.attention.head_count": 32,
        f"{arch}.embedding_length": 4096,
        f"{arch}.attention.key_length": 128,
        f"{arch}.attention.value_length": 128,
        f"{arch}.feed_forward_length": list(_FFN),
        f"{arch}.expert_count": 496,
        f"{arch}.expert_used_count": list(used),
        f"{arch}.context_length": 262144,
    }
    fields.update({f"{arch}.ssm.{k}": v for k, v in _SSM.items()})
    fields.update(extra)
    return fields


def _tensors(n_layers):
    return [_StubTensor(f"blk.{i}.ffn_up_exps.weight", MIB) for i in range(n_layers)] + [
        _StubTensor("token_embd.weight", 4 * MIB),
        _StubTensor("output.weight", 4 * MIB),
    ]


def _recurrent_bytes(n_recurrent):
    s = _SSM
    n_embd_r = (s["conv_kernel"] - 1) * (s["inner_size"] + 2 * s["group_count"] * s["state_size"])
    return n_recurrent * (n_embd_r + s["state_size"] * s["inner_size"]) * 4


def test_nemotron_layout_charges_kv_only_to_attention_layers():
    from core.inference.offload_layout import _layout_from_reader

    layout = _layout_from_reader(_StubReader(_nemotron_fields(), _tensors(8)))
    assert layout.complete
    assert layout.n_attention_layers == 2
    assert layout.kv_bytes_per_token_f16 == (2 + 2) * (128 + 128) * 2
    # Layers 0, 3, 6 carry neither attention nor an FFN: the Mamba layers.
    assert layout.recurrent_bytes == _recurrent_bytes(3)
    assert layout.n_expert_used == 9


def test_nemotron_layout_leaves_the_nextn_block_out_of_the_trunk():
    from core.inference.offload_layout import _layout_from_reader

    fields = _nemotron_fields(heads = _HEADS + [2], **{"nemotron_h_moe.nextn_predict_layers": 1})
    layout = _layout_from_reader(_StubReader(fields, _tensors(9)))
    assert layout.complete
    assert layout.n_attention_layers == 2


def test_per_layer_kv_heads_on_other_archs_still_abstain():
    """Other archs still raise, and layout_from_gguf turns that into an abstention."""
    from core.inference.offload_layout import _layout_from_reader

    fields = _nemotron_fields(arch = "jamba")
    with pytest.raises(TypeError):
        _layout_from_reader(_StubReader(fields, _tensors(8)))


def test_scalar_hybrid_layout_is_unchanged():
    from core.inference.offload_layout import _layout_from_reader

    fields = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 48,
        "qwen35moe.full_attention_interval": 4,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.head_count": 16,
        "qwen35moe.embedding_length": 2048,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.attention.value_length": 256,
        "qwen35moe.expert_count": 256,
        "qwen35moe.expert_used_count": 8,
        "qwen35moe.ssm.inner_size": 4096,
        "qwen35moe.ssm.state_size": 128,
        "qwen35moe.ssm.group_count": 16,
        "qwen35moe.ssm.conv_kernel": 4,
    }
    layout = _layout_from_reader(_StubReader(fields, _tensors(48)))
    assert layout.complete
    assert layout.n_attention_layers == 12
    assert layout.kv_bytes_per_token_f16 == 12 * 2 * (256 + 256) * 2
    assert layout.n_expert_used == 8
    n_embd_r = 3 * (4096 + 2 * 16 * 128)
    assert layout.recurrent_bytes == 36 * (n_embd_r + 128 * 4096) * 4


def _backend(**dims):
    from core.inference.llama_cpp import LlamaCppBackend

    b = LlamaCppBackend.__new__(LlamaCppBackend)
    for name, value in dims.items():
        setattr(b, f"_{name}", value)
    return b


def test_latent_moe_width_uses_the_latent_size():
    common = dict(
        embedding_length = 4096,
        expert_used_count = 22,
        expert_feed_forward_length = 2688,
        expert_shared_feed_forward_length = 5376,
        n_experts = 512,
    )
    latent = _backend(moe_latent_size = 1024, **common)._compute_activation_width()
    assert latent == 22 * (2 * 1024 + 3 * 2688) + 3 * 5376
    # No latent key: the formula every other MoE arch gets, unchanged.
    plain = _backend(**common)._compute_activation_width()
    assert plain == 22 * (2 * 4096 + 3 * 2688) + 3 * 5376


def _write_gguf(path, arch, kv, tensors):
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), arch)
    for key, value in kv.items():
        if isinstance(value, list):
            writer.add_array(key, value)
        else:
            writer.add_uint32(key, value)
    for name, n in tensors:
        writer.add_tensor(name, np.zeros(n, dtype = np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_per_layer_expert_dims_and_latent_size_are_read(tmp_path):
    from core.inference.llama_cpp import LlamaCppBackend

    path = tmp_path / "bolt.gguf"
    a = "nemotron_h_moe"
    _write_gguf(
        path,
        a,
        {
            f"{a}.block_count": 4,
            f"{a}.embedding_length": 64,
            f"{a}.expert_count": 8,
            f"{a}.expert_used_count": [0, 9, 0, 9],
            f"{a}.expert_feed_forward_length": [0, 1536, 0, 2560],
            f"{a}.moe_latent_size": 16,
        },
        [("blk.0.attn_norm.weight", 64)],
    )
    b = LlamaCppBackend(manages_processes = False)
    b._read_gguf_metadata(str(path))
    assert b._expert_used_count == 9
    assert b._expert_feed_forward_length == 2560
    assert b._moe_latent_size == 16


def _nextn_gguf(tmp_path, arch, nextn):
    path = tmp_path / f"{arch}.gguf"
    kv = {f"{arch}.block_count": 3, f"{arch}.embedding_length": 64}
    if nextn:
        kv[f"{arch}.nextn_predict_layers"] = nextn
    _write_gguf(
        path,
        arch,
        kv,
        [
            ("token_embd.weight", 256),
            ("blk.0.attn_norm.weight", 64),
            ("blk.1.ffn_up_exps.weight", 512),
            ("blk.2.nextn.eh_proj.weight", 1024),
            ("blk.2.ffn_up_exps.weight", 2048),
        ],
    )
    return str(path)


def test_unloaded_nextn_bytes_are_the_head_block_only(tmp_path):
    from core.inference.llama_cpp import LlamaCppBackend

    path = _nextn_gguf(tmp_path, "nemotron_h_moe", 1)
    b = LlamaCppBackend(manages_processes = False)
    b._read_gguf_metadata(path)
    assert b._unloaded_nextn_bytes(path) == 4 * (1024 + 2048)


@pytest.mark.parametrize("arch, nextn", [("qwen35moe", 1), ("glm4moe", 1), ("nemotron_h_moe", 0)])
def test_unloaded_nextn_bytes_are_zero_elsewhere(tmp_path, arch, nextn):
    from core.inference.llama_cpp import LlamaCppBackend

    path = _nextn_gguf(tmp_path, arch, nextn)
    b = LlamaCppBackend(manages_processes = False)
    b._read_gguf_metadata(path)
    assert b._unloaded_nextn_bytes(path) == 0


def test_tensor_scan_is_unchanged_without_also(tmp_path):
    from core.inference.llama_cpp import LlamaCppBackend

    path = _nextn_gguf(tmp_path, "nemotron_h_moe", 1)
    arch, named, experts = LlamaCppBackend._gguf_scan_tensor_bytes(path)
    assert arch == "nemotron_h_moe"
    assert named == {}
    assert experts == 4 * (512 + 2048)


class _Draft:
    def __init__(self, architecture):
        self._architecture = architecture

    def _compute_activation_width(self):
        return 0


@pytest.mark.parametrize("architecture, floored", [("dflash", True), ("qwen35", False)])
def test_dflash_draft_compute_is_floored(architecture, floored):
    b = _backend(embedding_length = 4096, vocab_size = 131072)
    b._draft_backend_for = lambda path: _Draft(architecture)
    got = b._mtp_draft_compute_bytes(32768, drafter_path = "dspark.gguf", spec_draft_n_max = 3)
    b._draft_backend_for = lambda path: None
    unfloored = b._mtp_draft_compute_bytes(32768, drafter_path = "dspark.gguf", spec_draft_n_max = 3)
    if floored:
        assert got == max(unfloored, b._DFLASH_DRAFT_COMPUTE_BYTES)
    else:
        assert got == unfloored


# Draft graph + target verify graph a DFlash drafter adds, measured on llama.cpp b11642 (MiB).
_DFLASH_MEASURED_COMPUTE_MIB = {
    "Bolt DSpark, block 8": 528.5 + 44.0,
    "Bolt LiLiCorr, block 8, SWA 1024": 307.0 + 60.1,
}


@pytest.mark.parametrize("drafter", sorted(_DFLASH_MEASURED_COMPUTE_MIB))
def test_dflash_compute_reserve_covers_the_measurement_without_double_counting(drafter):
    """Floors together must cover the 572 MiB measurement; charging it again overpriced Bolt + LiLiCorr."""
    from core.inference.llama_cpp import LlamaCppBackend

    reserve = (
        LlamaCppBackend._MTP_DRAFT_COMPUTE_BYTES + LlamaCppBackend._DFLASH_DRAFT_COMPUTE_BYTES
    ) / MIB
    measured = _DFLASH_MEASURED_COMPUTE_MIB[drafter]
    assert reserve >= measured
    assert reserve - max(_DFLASH_MEASURED_COMPUTE_MIB.values()) < 8
