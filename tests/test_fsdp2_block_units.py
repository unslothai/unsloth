# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Which module is a decoder block's FSDP2 unit. accelerate <= 1.15 shards the block inside an activation checkpoint
wrapper; accelerate >= 1.16 shards the wrapper itself, so the block is no FSDPModule and only its wrapper is. Both
shapes must count every block, or the last block is resharded after forward and gathered again for backward."""

import torch

from unsloth.models import fsdp2


class _FSDPModule:
    pass


class LlamaDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)


class _ShardedLayer(LlamaDecoderLayer, _FSDPModule):
    pass


class _Wrapper(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self._checkpoint_wrapped_module = inner


class _ShardedWrapper(_Wrapper, _FSDPModule):
    pass


class _Model(torch.nn.Module):
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


def _units(model):
    return fsdp2._block_units(model, fsdp2._parents(model), _FSDPModule)


def test_sharded_blocks_are_their_own_units():
    layers = [_ShardedLayer() for _ in range(3)]
    assert _units(_Model(layers)) == layers


def test_a_sharded_checkpoint_wrapper_is_the_unit_of_the_block_inside():
    wrappers = [_ShardedWrapper(LlamaDecoderLayer()) for _ in range(3)]
    assert _units(_Model(wrappers)) == wrappers


def test_an_unsharded_block_or_wrapper_is_no_unit():
    assert _units(_Model([LlamaDecoderLayer(), _Wrapper(LlamaDecoderLayer())])) == []
