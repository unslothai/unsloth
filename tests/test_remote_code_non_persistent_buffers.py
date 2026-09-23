# SPDX-License-Identifier: AGPL-3.0-only
"""transformers 5 builds a model on the meta device and gives each non-persistent buffer
empty storage for `_init_weights` to fill. Remote code written for 4.x computes those
buffers (RoPE inv_freq, lightning-attention slopes) in `__init__` and its `_init_weights`
only touches Linear / Embedding, so Ling-2.6-flash loaded with zero RoPE frequencies and
zero decay slopes: first-batch loss 5.03 in 16-bit (11.64 in 4-bit, where the storage
held garbage) against 1.12 for the same model on transformers 4.57.6."""

import importlib.util
import math
import os
import sys
import types

import pytest
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_helper():
    path = os.path.join(_ROOT, "unsloth", "models", "_remote_code_buffers.py")
    spec = importlib.util.spec_from_file_location("_unsloth_remote_code_buffers_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _slopes(n):
    start = 2 ** (-(2 ** -(math.log2(n) - 3)))
    return torch.tensor([start * start**i for i in range(n)], dtype = torch.float)


def _remote_module():
    """Classes shaped like 4.x remote code, living under transformers_modules.*"""
    from transformers import PretrainedConfig, PreTrainedModel

    name = "transformers_modules.unsloth_test_remote_buffers"
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    sys.modules.setdefault("transformers_modules", types.ModuleType("transformers_modules"))
    sys.modules[name] = module

    class TinyRemoteConfig(PretrainedConfig):
        model_type = "unsloth_tiny_remote_buffers"

        def __init__(
            self,
            hidden_size = 16,
            num_heads = 4,
            num_layers = 2,
            rope_theta = 10000.0,
            **kwargs,
        ):
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.rope_theta = rope_theta
            super().__init__(**kwargs)

    class TinyRotary(nn.Module):
        def __init__(
            self,
            config,
            device = None,
        ):
            super().__init__()
            dim = config.hidden_size // config.num_heads
            inv_freq = 1.0 / (
                config.rope_theta ** (torch.arange(0, dim, 2, dtype = torch.float) / dim)
            )
            self.config = config
            self.register_buffer("inv_freq", inv_freq, persistent = False)
            self.original_inv_freq = self.inv_freq

    class TinyLinearAttention(nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx
            self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
            slope = -_slopes(config.num_heads) * (
                1 - (layer_idx - 1) / (config.num_layers - 1) + 1e-5
            )
            self.register_buffer("slope", slope, persistent = False)
            self.rotary_emb = TinyRotary(config)

    class TinyRemotePreTrainedModel(PreTrainedModel):
        config_class = TinyRemoteConfig
        base_model_prefix = "model"

        def _init_weights(self, module):
            # What 4.x remote code ships: weights only, buffers assumed built in __init__.
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean = 0.0, std = 0.02)

    class TinyRemoteModel(TinyRemotePreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.layers = nn.ModuleList(
                TinyLinearAttention(config, i) for i in range(config.num_layers)
            )
            self.post_init()

    for cls in (
        TinyRemoteConfig,
        TinyRotary,
        TinyLinearAttention,
        TinyRemotePreTrainedModel,
        TinyRemoteModel,
    ):
        cls.__module__ = name
        setattr(module, cls.__name__, cls)
    return module


def _expected(layer_idx, config):
    dim = config.hidden_size // config.num_heads
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype = torch.float) / dim))
    slope = -_slopes(config.num_heads) * (1 - (layer_idx - 1) / (config.num_layers - 1) + 1e-5)
    return slope, inv_freq


def _saved_model(tmp_path):
    remote = _remote_module()
    config = remote.TinyRemoteConfig()
    remote.TinyRemoteModel(config).save_pretrained(tmp_path)
    return remote, config


def test_restores_buffers_after_a_transformers_load(tmp_path):
    helper = _load_helper()
    remote, config = _saved_model(tmp_path)
    model = remote.TinyRemoteModel.from_pretrained(tmp_path)
    restored = helper.restore_remote_code_non_persistent_buffers(model)
    if helper._transformers_builds_on_meta():
        assert restored == 2 * config.num_layers
    else:
        assert restored == 0
    for i, layer in enumerate(model.layers):
        slope, inv_freq = _expected(i, config)
        torch.testing.assert_close(layer.slope, slope)
        torch.testing.assert_close(layer.rotary_emb.inv_freq, inv_freq)
        # The alias 4.x code keeps next to the buffer points at the live buffer again.
        assert layer.rotary_emb.original_inv_freq is layer.rotary_emb.inv_freq


def test_meta_built_buffers_with_empty_storage_are_recomputed():
    # The transformers 5 sequence without a checkpoint: construct on meta, give the
    # non-persistent buffers empty storage, leave them for `_init_weights`.
    helper = _load_helper()
    if not helper._transformers_builds_on_meta():
        pytest.skip("transformers 4.x builds real buffers; the restore is a no-op there")
    remote = _remote_module()
    config = remote.TinyRemoteConfig()
    with torch.device("meta"):
        model = remote.TinyRemoteModel(config)
    for module in model.modules():
        for name in module._non_persistent_buffers_set:
            module._buffers[name] = torch.full(module._buffers[name].shape, 7.0)
    assert helper.restore_remote_code_non_persistent_buffers(model) == 2 * config.num_layers
    for i, layer in enumerate(model.layers):
        slope, inv_freq = _expected(i, config)
        torch.testing.assert_close(layer.slope, slope)
        torch.testing.assert_close(layer.rotary_emb.inv_freq, inv_freq)


def test_native_modules_and_unrecoverable_constructors_are_left_alone():
    helper = _load_helper()
    if not helper._transformers_builds_on_meta():
        pytest.skip("no-op on transformers 4.x")

    class Native(nn.Module):  # not remote code: transformers' own _init_weights owns its buffers
        def __init__(self):
            super().__init__()
            self.register_buffer("b", torch.full((2,), 3.0), persistent = False)

    native = Native()
    native.b.zero_()
    assert helper.restore_remote_code_non_persistent_buffers(native) == 0
    assert native.b.eq(0).all()

    class NeedsTensor(nn.Module):
        def __init__(self, table):
            super().__init__()
            self.register_buffer("b", table * 2, persistent = False)

    NeedsTensor.__module__ = "transformers_modules.unsloth_test_remote_buffers"
    module = NeedsTensor(torch.ones(2))
    module.b.zero_()
    # `table` is not recoverable from the instance, so the module is skipped, not guessed.
    assert helper._constructor_kwargs(module) is None
    assert helper.restore_remote_code_non_persistent_buffers(module) == 0

    class KeepsOnlyStride(nn.Module):
        def __init__(
            self,
            ratio = 2,
            device = None,
        ):
            super().__init__()
            self.stride = ratio  # `ratio` itself is not kept under its own name
            self.register_buffer("b", torch.full((2,), float(ratio)), persistent = False)

    KeepsOnlyStride.__module__ = "transformers_modules.unsloth_test_remote_buffers"
    module = KeepsOnlyStride(ratio = 4)
    module.b.zero_()
    # A non-default `ratio` cannot be told from the default, so nothing is rebuilt.
    assert helper._constructor_kwargs(module) is None
    assert helper.restore_remote_code_non_persistent_buffers(module) == 0
    assert module.b.eq(0).all()


def test_stored_tensor_arguments_skip_the_module():
    helper = _load_helper()
    if not helper._transformers_builds_on_meta():
        pytest.skip("no-op on transformers 4.x")

    class OptionalTable(nn.Module):
        def __init__(self, table = None):
            super().__init__()
            self.table = table
            base = torch.ones(2) if table is None else table
            self.register_buffer("b", base * 2, persistent = False)

    OptionalTable.__module__ = "transformers_modules.unsloth_test_remote_buffers"
    module = OptionalTable(torch.full((2,), 5.0))
    module.b.zero_()
    # Rebuilding with table=None would write 2.0 instead of 10.0, so the module is skipped.
    assert helper._constructor_kwargs(module) is None
    assert helper.restore_remote_code_non_persistent_buffers(module) == 0
    assert module.b.eq(0).all()


def test_a_stored_meta_device_is_not_passed_back():
    helper = _load_helper()
    if not helper._transformers_builds_on_meta():
        pytest.skip("no-op on transformers 4.x")

    class StoresDevice(nn.Module):
        def __init__(
            self,
            scale = 3.0,
            device = None,
        ):
            super().__init__()
            self.scale = scale
            self.device = device
            self.register_buffer("b", torch.full((2,), scale, device = device), persistent = False)

    StoresDevice.__module__ = "transformers_modules.unsloth_test_remote_buffers"
    module = StoresDevice(scale = 4.0, device = torch.device("meta"))
    module._buffers["b"] = torch.zeros(2)
    assert helper.restore_remote_code_non_persistent_buffers(module) == 1
    torch.testing.assert_close(module.b, torch.full((2,), 4.0))


def test_variadic_constructors_are_skipped():
    helper = _load_helper()
    if not helper._transformers_builds_on_meta():
        pytest.skip("no-op on transformers 4.x")

    class TakesKwargs(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            base = kwargs.get("base", 1.0)
            self.register_buffer("b", torch.full((2,), float(base)), persistent = False)

    TakesKwargs.__module__ = "transformers_modules.unsloth_test_remote_buffers"
    module = TakesKwargs(base = 6.0)
    module.b.zero_()
    # Rebuilding without `base` would write 1.0 instead of 6.0, so the module is skipped.
    assert helper._constructor_kwargs(module) is None
    assert helper.restore_remote_code_non_persistent_buffers(module) == 0
    assert module.b.eq(0).all()


def test_dtype_is_recovered_from_the_instance_or_the_module_is_skipped():
    helper = _load_helper()
    if not helper._transformers_builds_on_meta():
        pytest.skip("no-op on transformers 4.x")

    class EpsFromDtype(nn.Module):
        def __init__(
            self,
            dtype = torch.float32,
            keep = True,
        ):
            super().__init__()
            self.keep = keep
            if keep:
                self.dtype = dtype
            eps = torch.finfo(dtype).eps
            self.register_buffer("b", torch.full((2,), eps, dtype = torch.float32), persistent = False)

    EpsFromDtype.__module__ = "transformers_modules.unsloth_test_remote_buffers"
    module = EpsFromDtype(dtype = torch.float16)
    module.b.zero_()
    assert helper.restore_remote_code_non_persistent_buffers(module) == 1
    torch.testing.assert_close(module.b, torch.full((2,), torch.finfo(torch.float16).eps))

    module = EpsFromDtype(dtype = torch.float16, keep = False)
    module.b.zero_()
    assert helper._constructor_kwargs(module) is None
    assert helper.restore_remote_code_non_persistent_buffers(module) == 0


def test_equal_but_differently_typed_arguments_do_not_share_a_rebuild():
    helper = _load_helper()
    if not helper._transformers_builds_on_meta():
        pytest.skip("no-op on transformers 4.x")

    class TypeSensitive(nn.Module):
        def __init__(self, flag = False):
            super().__init__()
            self.flag = flag
            value = 2.0 if isinstance(flag, bool) else 3.0
            self.register_buffer("b", torch.full((2,), value), persistent = False)

    TypeSensitive.__module__ = "transformers_modules.unsloth_test_remote_buffers"
    parent = nn.Module()
    parent.first, parent.second = TypeSensitive(flag = True), TypeSensitive(flag = 1)
    parent.first.b.zero_()
    parent.second.b.zero_()
    assert helper.restore_remote_code_non_persistent_buffers(parent) == 2
    torch.testing.assert_close(parent.first.b, torch.full((2,), 2.0))
    torch.testing.assert_close(parent.second.b, torch.full((2,), 3.0))


def test_buffers_the_remote_init_weights_fills_are_left_alone():
    # A remote model whose own _init_weights fills a placeholder buffer already has the right
    # value; the constructor would only give the placeholder back.
    helper = _load_helper()
    if not helper._transformers_builds_on_meta():
        pytest.skip("no-op on transformers 4.x")

    class Placeholder(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("table", torch.zeros(2), persistent = False)

    class RemoteModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Placeholder()

        def _init_weights(self, module):
            if isinstance(module, Placeholder):
                module.table.fill_(5.0)

    for cls in (Placeholder, RemoteModel):
        cls.__module__ = "transformers_modules.unsloth_test_remote_buffers"
    model = RemoteModel()
    model._init_weights(model.block)
    assert helper.restore_remote_code_non_persistent_buffers(model) == 0
    torch.testing.assert_close(model.block.table, torch.full((2,), 5.0))


def test_a_module_that_did_not_keep_its_config_is_skipped():
    # In a composite model a child may have been built with text_config or vision_config;
    # rebuilding it from the root config could give different same-shaped buffers.
    helper = _load_helper()
    if not helper._transformers_builds_on_meta():
        pytest.skip("no-op on transformers 4.x")

    class ScaleFromConfig(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.register_buffer("b", torch.full((2,), float(config.scale)), persistent = False)

    ScaleFromConfig.__module__ = "transformers_modules.unsloth_test_remote_buffers"
    model = nn.Module()
    model.config = types.SimpleNamespace(scale = 1.0)
    model.child = ScaleFromConfig(types.SimpleNamespace(scale = 9.0))
    model.child.b.zero_()
    assert helper._constructor_kwargs(model.child) is None
    assert helper.restore_remote_code_non_persistent_buffers(model) == 0
    assert model.child.b.eq(0).all()


def test_remote_init_detection_is_scoped_to_the_class_it_names():
    helper = _load_helper()
    if not helper._transformers_builds_on_meta():
        pytest.skip("no-op on transformers 4.x")

    class Placeholder(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("table", torch.zeros(2), persistent = False)

    class ComputesTable(nn.Module):
        def __init__(self, base = 4.0):
            super().__init__()
            self.base = base
            self.register_buffer("table", torch.full((2,), base), persistent = False)

    class RemoteModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Placeholder()
            self.other = ComputesTable()

        def _init_weights(self, module):
            if isinstance(module, Placeholder):
                module.table.fill_(5.0)

    for cls in (Placeholder, ComputesTable, RemoteModel):
        cls.__module__ = "transformers_modules.unsloth_test_remote_buffers"
    model = RemoteModel()
    model._init_weights(model.block)
    model.other.table.zero_()
    assert helper.restore_remote_code_non_persistent_buffers(model) == 1
    torch.testing.assert_close(model.block.table, torch.full((2,), 5.0))
    torch.testing.assert_close(model.other.table, torch.full((2,), 4.0))


def test_loaders_restore_right_after_from_pretrained():
    for relative, calls in (("unsloth/models/vision.py", 1), ("unsloth/models/llama.py", 2)):
        with open(os.path.join(_ROOT, relative), encoding = "utf-8") as file:
            source = file.read()
        assert source.count("restore_remote_code_non_persistent_buffers(model)") == calls, relative
