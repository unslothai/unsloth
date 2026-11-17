# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Streaming generate must not stream image-only decoder blocks (Mllama's cross-attention layers):
text-only ranks skip them, so streaming them would pair one rank's gather with another block's."""

import torch.nn as nn

from unsloth.models.fsdp2_native import _stream_block_lists


class MllamaTextCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)


class MllamaCrossAttentionDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = MllamaTextCrossAttention()


class SelfLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 4)


class Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [SelfLayer(), MllamaCrossAttentionDecoderLayer(), SelfLayer(), SelfLayer()]
        )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = Text()


def test_image_only_decoder_blocks_are_not_streamed():
    model = Model()
    layers = model.language_model.layers
    [(found, blocks)] = _stream_block_lists(model, list(layers))
    assert found is layers
    assert sorted(blocks) == [0, 2, 3]


def test_all_image_only_means_no_streaming():
    model = Model()
    layers = model.language_model.layers
    assert _stream_block_lists(model, [layers[1]]) == []
