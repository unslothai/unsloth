# SPDX-License-Identifier: AGPL-3.0-only
"""Mixed-generation multi-GPU hosts: the attention backend follows the device Q lives on, not
device 0, and homogeneous hosts never consult the per-device table."""

from types import SimpleNamespace

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.models import _utils as mu
from unsloth.utils import attention_dispatch as ad

AMPERE, TURING, PASCAL, HOPPER, B200, RTX50 = (8, 6), (7, 5), (6, 1), (9, 0), (10, 0), (12, 0)


def _cuda(index):
    return SimpleNamespace(type = "cuda", index = index)


def _table(
    monkeypatch,
    capabilities,
    flash = True,
    xformers = True,
    probe = lambda *a, **k: True,
):
    monkeypatch.setattr(ad, "HAS_FLASH_ATTENTION", flash)
    monkeypatch.setattr(ad, "HAS_XFORMERS", xformers)
    return ad._per_device_backends(capabilities, probe = probe)


@pytest.mark.parametrize(
    "capabilities",
    [(), (AMPERE,), (AMPERE, AMPERE), (TURING, TURING), (B200,) * 8, (AMPERE, HOPPER)],
)
def test_homogeneous_hosts_skip_the_table(monkeypatch, capabilities):
    assert _table(monkeypatch, capabilities) is None


def test_flash_is_kept_on_ampere_and_dropped_on_turing(monkeypatch):
    table = _table(monkeypatch, (AMPERE, TURING))
    monkeypatch.setattr(ad, "_DEVICE_BACKENDS", table)
    for backend in (ad.FLASH_DENSE, ad.FLASH_VARLEN):
        assert ad._backend_for_device(backend, _cuda(0)) == backend
        assert ad._backend_for_device(backend, _cuda(1)) == ad.SDPA


def test_turing_at_device_0_keeps_flash_off_and_sdpa_untouched(monkeypatch):
    # HAS_FLASH_ATTENTION is device-0 based, so a Turing device 0 leaves flash off everywhere.
    table = _table(monkeypatch, (TURING, AMPERE), flash = False)
    monkeypatch.setattr(ad, "_DEVICE_BACKENDS", table)
    assert ad._backend_for_device(ad.XFORMERS, _cuda(0), torch.float16) == ad.XFORMERS
    assert ad._backend_for_device(ad.SDPA, _cuda(1)) == ad.SDPA


def test_xformers_bf16_on_pre_ampere_goes_to_sdpa(monkeypatch):
    table = _table(monkeypatch, (AMPERE, PASCAL), flash = False)
    monkeypatch.setattr(ad, "_DEVICE_BACKENDS", table)
    assert ad._backend_for_device(ad.XFORMERS, _cuda(0), torch.bfloat16) == ad.XFORMERS
    assert ad._backend_for_device(ad.XFORMERS, _cuda(1), torch.float16) == ad.XFORMERS
    assert ad._backend_for_device(ad.XFORMERS, _cuda(1), torch.bfloat16) == ad.SDPA


def test_rtx50_beyond_device_0_is_probed_only_without_flash(monkeypatch):
    probed = []

    def probe(device, supports_bf16 = None):
        probed.append(device)
        return False

    table = _table(monkeypatch, (AMPERE, RTX50), flash = False, probe = probe)
    assert probed == ["cuda:1"]
    monkeypatch.setattr(ad, "_DEVICE_BACKENDS", table)
    assert ad._backend_for_device(ad.XFORMERS, _cuda(0), torch.bfloat16) == ad.XFORMERS
    assert ad._backend_for_device(ad.XFORMERS, _cuda(1), torch.bfloat16) == ad.SDPA

    probed.clear()
    _table(monkeypatch, (AMPERE, RTX50), flash = True, probe = probe)
    assert probed == []


def test_fp32_downcast_follows_the_device(monkeypatch):
    table = _table(monkeypatch, (AMPERE, B200), flash = False)
    assert [row[2] for row in table] == [False, True]


def test_non_cuda_and_out_of_range_devices_are_left_alone(monkeypatch):
    table = _table(monkeypatch, (AMPERE, TURING))
    monkeypatch.setattr(ad, "_DEVICE_BACKENDS", table)
    assert ad._backend_for_device(ad.FLASH_DENSE, SimpleNamespace(type = "cpu", index = None)) == (
        ad.FLASH_DENSE
    )
    assert ad._backend_for_device(ad.FLASH_DENSE, _cuda(5)) == ad.FLASH_DENSE


def _run(
    monkeypatch,
    table,
    config_kwargs = None,
):
    calls = []

    def fake_flash(Q, K, V, **kwargs):
        calls.append("flash")
        return torch.zeros_like(Q)

    real_sdpa = ad.scaled_dot_product_attention

    def fake_sdpa(*args, **kwargs):
        calls.append("sdpa")
        return real_sdpa(*args, **kwargs)

    monkeypatch.setattr(ad, "flash_attn_func", fake_flash, raising = False)
    monkeypatch.setattr(ad, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setattr(ad, "_DEVICE_BACKENDS", table)
    Q = torch.randn(1, 2, 4, 8, dtype = torch.float16, device = "cuda:0")
    config = ad.AttentionConfig(
        backend = ad.FLASH_DENSE,
        n_kv_heads = 2,
        n_groups = 1,
        flash_dense_kwargs = {"causal": True, **(config_kwargs or {})},
    )
    context = ad.AttentionContext(
        bsz = 1,
        q_len = 4,
        kv_seq_len = 4,
        n_heads = 2,
        head_dim = 8,
        requires_grad = False,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
    )
    ad.run_attention(config = config, context = context, Q = Q, K = Q, V = Q)
    return calls


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA tensor")


@requires_cuda
def test_run_attention_reroutes_only_the_incapable_device(monkeypatch):
    capable = ((True, True, False, True), (False, True, False, False))
    assert _run(monkeypatch, capable) == ["flash"]
    incapable = ((False, True, False, False), (True, True, False, True))
    assert _run(monkeypatch, incapable) == ["sdpa"]
    assert _run(monkeypatch, None) == ["flash"]


@requires_cuda
def test_run_attention_refuses_to_drop_a_softcap(monkeypatch):
    incapable = ((False, True, False, False), (True, True, False, True))
    with pytest.raises(RuntimeError, match = "softcaps"):
        _run(monkeypatch, incapable, {"softcap": 50.0})


class _Model(torch.nn.Module):
    _supports_sdpa = True

    def __init__(
        self,
        device_map,
        impl = "flash_attention_2",
        model_type = "llama",
    ):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation = impl, model_type = model_type)
        self.hf_device_map = device_map


def _capability(monkeypatch, capabilities):
    monkeypatch.setattr(mu, "HAS_FLASH_ATTENTION", True)
    monkeypatch.setattr(mu, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i = None: capabilities[i or 0])


@pytest.mark.parametrize(
    "device_map, expected",
    [
        ({"": 0}, "flash_attention_2"),
        ({"model.layers.0": 0, "model.layers.1": "cuda:2"}, "flash_attention_2"),
        ({"model.layers.0": 0, "model.layers.1": 1}, "sdpa"),
        ({"model.layers.0": "cuda:0", "model.layers.1": "cuda:1"}, "sdpa"),
        ({"model.layers.0": 0, "lm_head": "cpu", "model.norm": "disk"}, "flash_attention_2"),
        (None, "flash_attention_2"),
    ],
)
def test_hf_flash_falls_back_only_when_layers_reach_pre_ampere(monkeypatch, device_map, expected):
    _capability(monkeypatch, {0: AMPERE, 1: TURING, 2: HOPPER})
    model = _Model(device_map)
    mu.fall_back_from_flash_attention_on_pre_ampere(model)
    assert model.config._attn_implementation == expected


def test_hf_non_flash_models_are_untouched(monkeypatch):
    _capability(monkeypatch, {0: AMPERE, 1: TURING})
    model = _Model({"a": 0, "b": 1}, impl = "eager")
    assert mu.fall_back_from_flash_attention_on_pre_ampere(model) is False
    assert model.config._attn_implementation == "eager"


class _SelfAttention(torch.nn.Module):
    pass


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _SelfAttention()


def test_hf_flash_stays_when_only_non_attention_modules_reach_pre_ampere(monkeypatch):
    _capability(monkeypatch, {0: AMPERE, 1: TURING})
    model = _Model({"layers.0": 0, "lm_head": 1, "embed": "cuda:1"})
    model.layers = torch.nn.ModuleList([_Decoder()])
    model.lm_head = torch.nn.Linear(2, 2)
    model.embed = torch.nn.Embedding(2, 2)
    assert mu.fall_back_from_flash_attention_on_pre_ampere(model) is False
    assert model.config._attn_implementation == "flash_attention_2"
    model.hf_device_map = {"layers.0": 1, "lm_head": 0}
    assert mu.fall_back_from_flash_attention_on_pre_ampere(model) is True
    assert model.config._attn_implementation == "sdpa"


def test_hf_fallback_keeps_known_broken_sdpa_models_on_eager(monkeypatch):
    _capability(monkeypatch, {0: AMPERE, 1: TURING})
    model = _Model({"a": 0, "b": 1}, model_type = "gpt_oss")
    assert mu.fall_back_from_flash_attention_on_pre_ampere(model) is True
    assert model.config._attn_implementation == "eager"
