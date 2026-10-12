# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""pack_multimodal_documents row layout (CPU only)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from unsloth.utils.mm_packing import (  # noqa: E402
    pack_multimodal_documents,
    prepare_packed_multimodal_model,
)


def _text(ids):
    t = torch.tensor([ids])
    return {
        "input_ids": t,
        "attention_mask": torch.ones_like(t),
        "mm_token_type_ids": torch.zeros_like(t),
        "labels": t.clone(),
    }


def _image(ids, n_img, patches):
    d = _text(ids)
    d["mm_token_type_ids"] = (torch.tensor([ids]) == 9).long()
    d["pixel_values"] = torch.full((n_img, patches, 4), float(len(ids)))
    d["image_position_ids"] = torch.ones(n_img, patches, 2, dtype = torch.long)
    return d


def _audio(ids, frames):
    d = _text(ids)
    d["input_features"] = torch.ones(1, frames, 3)
    d["input_features_mask"] = torch.ones(1, frames, dtype = torch.bool)
    return d


def test_layout_positions_lengths_and_boundary_labels():
    docs = [_text([1, 2, 3]), _image([4, 9, 9, 5], 1, 6), _text([6, 7])]
    out = pack_multimodal_documents(docs)
    assert out["input_ids"].tolist() == [[1, 2, 3, 4, 9, 9, 5, 6, 7]]
    assert out["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2, 3, 0, 1]]
    assert out["packed_seq_lengths"].tolist() == [3, 4, 2]
    assert out["packed_seq_lengths"].dtype == torch.int32
    assert out["labels"].tolist() == [[1, 2, 3, -100, 9, 9, 5, -100, 7]]
    assert out["mm_token_type_ids"].tolist() == [[0, 0, 0, 0, 1, 1, 0, 0, 0]]
    assert "attention_mask" not in out
    assert out["pixel_values"].shape == (1, 6, 4)


def test_media_concatenate_in_document_order_and_pad_with_the_model_padding():
    docs = [_image([9, 1], 2, 6), _text([2]), _image([9, 9, 3], 1, 8)]
    out = pack_multimodal_documents(docs)
    pv, pos = out["pixel_values"], out["image_position_ids"]
    assert pv.shape == (3, 8, 4) and pos.shape == (3, 8, 2)
    assert pv[:2, :6].eq(2.0).all() and pv[2].eq(3.0).all()
    assert pv[:2, 6:].eq(0).all()
    assert pos[:2, 6:].eq(-1).all() and pos[:2, :6].eq(1).all()  # (-1, -1) = padding patch


def test_audio_frames_pad_with_a_false_mask():
    out = pack_multimodal_documents([_audio([1, 2], 5), _audio([3], 7)])
    assert out["input_features"].shape == (2, 7, 3)
    assert out["input_features_mask"].tolist() == [[True] * 5 + [False] * 2, [True] * 7]


def test_padding_inside_a_document_is_stripped():
    d = _text([1, 2, 0, 0])
    d["attention_mask"] = torch.tensor([[1, 1, 0, 0]])
    out = pack_multimodal_documents([d, _text([5])])
    assert out["input_ids"].tolist() == [[1, 2, 5]]
    assert out["packed_seq_lengths"].tolist() == [2, 1]


@pytest.mark.parametrize("key", ["pixel_values_videos", "image_grid_thw", "pixel_attention_mask"])
def test_unproven_media_keys_are_refused(key):
    d = _text([1])
    d[key] = torch.ones(1, 2)
    with pytest.raises(ValueError, match = "cannot pack"):
        pack_multimodal_documents([d])


def test_unsupported_models_are_refused():
    qwen = SimpleNamespace(config = SimpleNamespace(model_type = "qwen2_5_vl"))
    with pytest.raises(ValueError, match = "only supported"):
        prepare_packed_multimodal_model(qwen)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def _collator(padding_free = True, **slots):
    from unsloth.trainer import UnslothVisionDataCollator

    collator = object.__new__(UnslothVisionDataCollator)
    collator._padding_free = padding_free
    collator._deferring_check = False
    collator._checked_video_paths = set()
    for name, value in slots.items():
        setattr(collator, name, value)
    return collator


def test_supervision_is_judged_over_the_batch_not_per_document(monkeypatch):
    from unsloth.trainer import UnslothVisionDataCollator, _UnslothVisionDataCollatorBase

    if not hasattr(_UnslothVisionDataCollatorBase, "_check_supervised"):
        pytest.skip("this unsloth_zoo has no supervision check")
    empty, trained = _text([1, 2, 3]), _text([4, 5])
    empty["labels"][:] = -100
    docs = iter([empty, trained])
    seen = []

    def collate(self, examples):
        doc = next(docs)
        self._check_supervised(doc["labels"], doc["attention_mask"])
        return doc

    monkeypatch.setattr(UnslothVisionDataCollator, "_collate_unpacked", collate)
    monkeypatch.setattr(
        _UnslothVisionDataCollatorBase,
        "_check_supervised",
        lambda self, labels, attention_mask = None: seen.append(labels.clone()),
    )
    collator = _collator(ignore_index = -100, train_on_responses_only = lambda batch: batch)
    out = collator([{}, {}])
    # One check, over both documents: the empty first one alone would read as an unsupervised batch.
    assert len(seen) == 1 and seen[0].shape == (2, 3)
    assert seen[0][0].eq(-100).all() and seen[0][1, :2].tolist() == [4, 5]
    assert out["packed_seq_lengths"].tolist() == [3, 2]


def _trainer(cp_enabled = False):
    return SimpleNamespace(
        accelerator = SimpleNamespace(parallelism_config = SimpleNamespace(cp_enabled = cp_enabled))
    )


def test_packed_collation_yields_to_eval_logits_and_context_parallelism(monkeypatch):
    from unsloth.trainer import _collator_packing_conflict

    # The trainer's own for_inference() leaves this set after __init__; only the pre-init read counts.
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    assert _collator_packing_conflict(_trainer(), False) is None
    assert "prediction row" in _collator_packing_conflict(_trainer(), True)
    assert "context parallelism" in _collator_packing_conflict(_trainer(cp_enabled = True), False)
    assert _collator_packing_conflict(SimpleNamespace(), False) is None
