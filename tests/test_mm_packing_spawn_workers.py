# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The packed mRoPE collator's rope_index must pickle without the model: spawned DataLoader workers
(the default start method on Windows and macOS) pickle collate_fn. CPU only, tiny random Qwen2.5-VL."""

import functools
import io
import pickle

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")

from unsloth.utils import mm_packing  # noqa: E402


def _tiny_qwen2_5_vl():
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )

    config = Qwen2_5_VLConfig(
        text_config = dict(
            hidden_size = 32,
            intermediate_size = 64,
            vocab_size = 100,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            num_hidden_layers = 2,
            rope_scaling = dict(rope_type = "default", mrope_section = [1, 1, 2]),
        ),
        vision_config = dict(
            hidden_size = 32,
            intermediate_size = 64,
            depth = 1,
            num_heads = 4,
            out_hidden_size = 32,
            fullatt_block_indexes = [0],
        ),
    )
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(config)
    # PEFT / Unsloth checkpointed training installs this local-function hook on the model.
    model.enable_input_require_grads()
    return model


def _docs(config):
    start, image = config.vision_start_token_id, config.image_token_id
    # 1 x 4 x 4 patches, merged 2 x 2: four image tokens.
    with_image = [1, 2, start, image, image, image, image, 3, 4]
    return [
        {"input_ids": torch.tensor([with_image]), "image_grid_thw": torch.tensor([[1, 4, 4]])},
        {"input_ids": torch.tensor([[5, 6, 7]])},
    ]


class _NoModelPickler(pickle.Pickler):
    """Fails on any module or tensor reachable from the pickled object."""

    def persistent_id(self, obj):
        if isinstance(obj, (torch.nn.Module, torch.Tensor)):
            raise AssertionError(f"pickled a {type(obj).__name__}")
        return None


def test_the_module_under_test_is_this_tree():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert (
        pathlib.Path(mm_packing.__file__).resolve() == root / "unsloth" / "utils" / "mm_packing.py"
    )


def test_rope_index_pickles_without_the_model_and_keeps_positions():
    model = _tiny_qwen2_5_vl()
    rope_index = mm_packing.prepare_packed_multimodal_model(model)
    _NoModelPickler(io.BytesIO()).dump(rope_index)
    restored = pickle.loads(pickle.dumps(rope_index))
    docs = _docs(model.config)
    for doc in docs:
        assert torch.equal(restored(doc), rope_index(doc))
    # Images moved the mRoPE planes off the text positions, so the comparison is not trivial.
    assert not torch.equal(rope_index(docs[0])[0, 0], torch.arange(docs[0]["input_ids"].shape[1]))
    # The original still holds the live model.
    assert rope_index.get_rope_index.__self__ is model.model


def test_spawned_dataloader_workers_pack_like_the_main_process(monkeypatch):
    # The worker re-imports unsloth; let it use the CPU backend on a GPU-less host.
    monkeypatch.delenv("UNSLOTH_DISABLE_CPU_FALLBACK", raising = False)
    model = _tiny_qwen2_5_vl()
    rope_index = mm_packing.prepare_packed_multimodal_model(model)
    collate = functools.partial(mm_packing.pack_multimodal_documents, rope_index = rope_index)
    docs = _docs(model.config)
    expected = collate(docs)
    loader = torch.utils.data.DataLoader(
        docs,
        batch_size = 2,
        collate_fn = collate,
        num_workers = 1,
        multiprocessing_context = "spawn",
    )
    (batch,) = list(loader)
    assert torch.equal(batch["position_ids"], expected["position_ids"])
    assert torch.equal(batch["packed_seq_lengths"], expected["packed_seq_lengths"])
