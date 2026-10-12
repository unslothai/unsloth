# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")

from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.diffusion_qwenimage21_vision import (
    configure_vision_attention,
    _ENV,
    _BACKEND,
    _eager_vision_attention,
)


def _target(backend = "rocm", ordinal = None):
    return DiffusionDeviceTarget(
        device = "cpu" if backend == "cpu" else "cuda",
        dtype = torch.bfloat16,
        backend = backend,
        vendor = None,
        supports_model_cpu_offload = True,
        supports_default_torch_compile = False,
        supports_pinned_transfer = True,
        ordinal = ordinal,
    )


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.delenv(_ENV, raising = False)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName = "gfx1151")
    )
    visual = type("Qwen3VLVisionModel", (), {})()
    visual.config = SimpleNamespace(_attn_implementation = "sdpa")
    visual.set_attn_implementation = Mock(
        side_effect = lambda value: setattr(visual.config, "_attn_implementation", value)
    )
    text = SimpleNamespace(config = SimpleNamespace(_attn_implementation = "sdpa"), visual = visual)
    pipe = SimpleNamespace(text_encoder = SimpleNamespace(model = text), transformer = object())
    return pipe, visual


def test_only_vision_changes_and_repeated_install_is_noop(runtime):
    pipe, visual = runtime
    assert configure_vision_attention(
        pipe, family = "qwen-image-2.1", target = _target(ordinal = 0), logger = Mock()
    )
    assert visual.config._attn_implementation == _BACKEND
    assert pipe.text_encoder.model.config._attn_implementation == "sdpa"
    assert not configure_vision_attention(
        pipe, family = "qwen-image-2.1", target = _target(ordinal = 0), logger = Mock()
    )
    visual.set_attn_implementation.assert_called_once_with(_BACKEND)


@pytest.mark.parametrize(
    "family, backend, arch, override, expected",
    [
        ("qwen-image", "rocm", "gfx1151", None, False),
        ("qwen-image-2.1", "cpu", "gfx1151", "1", False),
        ("qwen-image-2.1", "rocm", "gfx1100", None, False),
        ("qwen-image-2.1", "rocm", "gfx1100", "1", True),
        ("qwen-image-2.1", "rocm", "gfx1151:xnack-", None, True),
        ("qwen-image-2.1", "rocm", "gfx1151", "0", False),
    ],
)
def test_scope(runtime, monkeypatch, family, backend, arch, override, expected):
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName = arch)
    )
    if override is not None:
        monkeypatch.setenv(_ENV, override)
    assert (
        configure_vision_attention(
            runtime[0], family = family, target = _target(backend), logger = Mock()
        )
        is expected
    )


@pytest.mark.parametrize("backend", ["eager", "flash_attention_2", "custom"])
def test_preserves_other_attention_backends(runtime, backend):
    runtime[1].config._attn_implementation = backend
    assert not configure_vision_attention(
        runtime[0], family = "qwen-image-2.1", target = _target(), logger = Mock()
    )


def test_cuda_not_rocm(runtime, monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    assert not configure_vision_attention(
        runtime[0], family = "qwen-image-2.1", target = _target("cuda"), logger = Mock()
    )


@pytest.mark.parametrize("lengths", [[16], [7, 9]])
def test_real_vision_attention_parity_and_scope(monkeypatch, lengths):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLVisionConfig,
        Qwen3VLTextConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    config = Qwen3VLConfig(
        vision_config = Qwen3VLVisionConfig(
            depth = 1,
            hidden_size = 32,
            intermediate_size = 64,
            num_heads = 4,
            out_hidden_size = 32,
            deepstack_visual_indexes = [],
        ).to_dict(),
        text_config = Qwen3VLTextConfig(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 1,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 8,
            vocab_size = 64,
        ).to_dict(),
    )
    config._attn_implementation = "sdpa"
    model = Qwen3VLModel(config).eval()
    visual = model.visual
    text_backend = model.language_model.config._attn_implementation
    class_forward = type(visual.blocks[0].attn).forward
    flags = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    pipe = SimpleNamespace(text_encoder = SimpleNamespace(model = model))
    x = torch.randn(sum(lengths), 32)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype = torch.int32)
    positions = (torch.ones(sum(lengths), 8), torch.zeros(sum(lengths), 8))
    with torch.no_grad():
        reference = visual.blocks[0].attn(x, cu, position_embeddings = positions)
    monkeypatch.setenv(_ENV, "1")
    assert configure_vision_attention(
        pipe, family = "qwen-image-2.1", target = _target(), logger = Mock()
    )
    assert all(block.attn.config._attn_implementation == _BACKEND for block in visual.blocks)
    assert model.language_model.config._attn_implementation == text_backend
    assert type(visual.blocks[0].attn).forward is class_forward
    assert flags == (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    with torch.no_grad():
        actual = visual.blocks[0].attn(x, cu, position_embeddings = positions)
    torch.testing.assert_close(actual, reference, rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("mask_kind", ["none", "broadcast", "per_query"])
def test_chunked_attention_retains_all_keys_and_values(monkeypatch, mask_kind):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

    module = SimpleNamespace(num_key_value_groups = 1, training = False)
    q = torch.randn(1, 4, 1025, 8)
    k = torch.randn(1, 4, 1103, 8)
    v = torch.randn_like(k)
    mask = None
    if mask_kind != "none":
        mask = torch.zeros(1, 1, 1 if mask_kind == "broadcast" else 1025, 1103)
        mask[..., -3:] = -torch.inf
    original = qwen.eager_attention_forward
    expected, _ = original(module, q, k, v, mask, scaling = 0.25)
    chunks = []

    def tracked(module, query, key, value, attention_mask, **kwargs):
        chunks.append(query.shape[-2])
        assert key is k and value is v
        return original(module, query, key, value, attention_mask, **kwargs)

    monkeypatch.setattr(qwen, "eager_attention_forward", tracked)
    actual, weights = _eager_vision_attention(module, q, k, v, mask, scaling = 0.25)
    assert chunks == [512, 512, 1]
    assert weights is None
    torch.testing.assert_close(actual, expected, rtol = 1e-5, atol = 1e-6)


def test_dropout_keeps_stock_dispatch(monkeypatch):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

    native = Mock(return_value = ("output", "weights"))
    monkeypatch.setattr(qwen, "eager_attention_forward", native)
    q = torch.empty(1, 4, 1025, 8)
    assert _eager_vision_attention(None, q, q, q, None, dropout = 0.1) == ("output", "weights")
    native.assert_called_once()


def test_architecture_gate_uses_selected_device(runtime, monkeypatch):
    def properties(device):
        return SimpleNamespace(gcnArchName = "gfx1151" if device.index == 1 else "gfx1100")

    probe = Mock(side_effect = properties)
    monkeypatch.setattr(torch.cuda, "get_device_properties", probe)
    assert not configure_vision_attention(
        runtime[0], family = "qwen-image-2.1", target = _target(ordinal = 0), logger = Mock()
    )
    assert configure_vision_attention(
        runtime[0], family = "qwen-image-2.1", target = _target(ordinal = 1), logger = Mock()
    )
    assert [call.args[0] for call in probe.call_args_list] == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]


@pytest.mark.parametrize("setting", ["1", "true", "on", "yes", " TRUE "])
def test_boolean_opt_in_on_other_architecture(runtime, monkeypatch, setting):
    monkeypatch.setenv(_ENV, setting)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName = "gfx1100")
    )
    assert configure_vision_attention(
        runtime[0], family = "qwen-image-2.1", target = _target(), logger = Mock()
    )


@pytest.mark.parametrize("setting", ["0", "false", "off", "no", " OFF "])
def test_boolean_opt_out(runtime, monkeypatch, setting):
    monkeypatch.setenv(_ENV, setting)
    assert not configure_vision_attention(
        runtime[0], family = "qwen-image-2.1", target = _target(), logger = Mock()
    )


@pytest.mark.parametrize("setting", ["", "auto", " AUTO "])
def test_auto_override_keeps_architecture_gate(runtime, monkeypatch, setting):
    monkeypatch.setenv(_ENV, setting)
    assert configure_vision_attention(
        runtime[0], family = "qwen-image-2.1", target = _target(), logger = Mock()
    )


def test_invalid_override_is_actionable(runtime, monkeypatch):
    monkeypatch.setenv(_ENV, "treu")
    with pytest.raises(ValueError, match = _ENV):
        configure_vision_attention(
            runtime[0], family = "qwen-image-2.1", target = _target(), logger = Mock()
        )
    runtime[1].set_attn_implementation.assert_not_called()


@pytest.mark.parametrize("failure", ["register", "select"])
def test_install_failure_is_actionable_and_preserves_cause(runtime, monkeypatch, failure):
    from transformers import AttentionInterface

    error = RuntimeError("unsupported attention API")
    if failure == "register":
        monkeypatch.setattr(AttentionInterface, "register", Mock(side_effect = error))
    else:
        runtime[1].set_attn_implementation.side_effect = error
    with pytest.raises(RuntimeError, match = f"{_ENV}=0") as caught:
        configure_vision_attention(
            runtime[0], family = "qwen-image-2.1", target = _target(), logger = Mock()
        )
    assert caught.value.__cause__ is error
