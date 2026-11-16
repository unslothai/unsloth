# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastSentenceTransformer training fast paths: lean LoRA, prefetch, decoder packing."""

import threading
import types

import pytest

pytest.importorskip("sentence_transformers")
torch = pytest.importorskip("torch")

from unsloth import FastSentenceTransformer  # noqa: E402
from unsloth.models import _sentence_transformer_prefetch as prefetch  # noqa: E402
from unsloth.models import _sentence_transformer_unpadding as unpadding  # noqa: E402
from unsloth.models._lean_lora import lean_lora_forward  # noqa: E402


@pytest.fixture
def fast_encoder(tmp_path):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer
    from transformers import BertConfig, BertModel, BertTokenizerFast

    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "red", "square"]),
        encoding = "utf-8",
    )
    config = BertConfig(
        vocab_size = 8,
        hidden_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size = 32,
        max_position_embeddings = 64,
    )
    model_dir = tmp_path / "model"
    BertModel(config).save_pretrained(model_dir)
    BertTokenizerFast(vocab_file = str(vocab)).save_pretrained(model_dir)
    model = SentenceTransformer(modules = [Transformer(str(model_dir)), Pooling(16)], device = "cpu")
    model._unsloth_fast_encoder = True
    model._unsloth_trust_remote_code = False
    return model


def test_fast_encoder_lora_runs_lean_forward(fast_encoder):
    from peft.tuners.lora.layer import Linear

    model = FastSentenceTransformer.get_peft_model(
        fast_encoder,
        r = 4,
        lora_alpha = 8,
        target_modules = ["query", "key", "value"],
        use_gradient_checkpointing = False,
    )
    layers = [m for m in model[0].auto_model.modules() if hasattr(m, "_unsloth_peft_forward")]
    if Linear.forward.__name__ == "unsloth_forward":
        # An earlier FastModel load compiled PEFT's own forward, which the lean one must not replace.
        assert not layers
        return
    assert len(layers) == 6 and all(m.forward.__func__ is lean_lora_forward for m in layers)
    layer = layers[0]
    torch.manual_seed(0)
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_()
    x = torch.randn(3, 5, layer.in_features, requires_grad = True)
    lean = layer(x)
    (grad_lean,) = torch.autograd.grad(lean.square().sum(), x)
    peft = layer._unsloth_peft_forward(x)
    (grad_peft,) = torch.autograd.grad(peft.square().sum(), x)
    torch.testing.assert_close(lean, peft)
    torch.testing.assert_close(grad_lean, grad_peft)
    # A second adapter selected after install: the output must follow it, not the cached "default".
    from peft import LoraConfig

    peft_model = model[0].auto_model
    peft_model.add_adapter(
        "other", LoraConfig(r = 4, lora_alpha = 8, target_modules = ["query", "key", "value"])
    )
    with torch.no_grad():
        layer.lora_B["other"].weight.normal_()
    peft_model.set_adapter("other")
    torch.testing.assert_close(layer(x), layer._unsloth_peft_forward(x))
    assert not torch.allclose(layer(x), lean)


def test_trainer_collates_on_a_background_thread(fast_encoder, tmp_path, monkeypatch):
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )

    data = Dataset.from_dict(
        {"anchor": ["a red", "red square", "a square", "square a"] * 4, "positive": ["red"] * 16}
    )
    args = SentenceTransformerTrainingArguments(
        output_dir = str(tmp_path / "out"),
        per_device_train_batch_size = 4,
        report_to = "none",
        use_cpu = True,
        seed = 3407,
    )
    trainer = SentenceTransformerTrainer(
        model = fast_encoder,
        args = args,
        train_dataset = data,
        loss = losses.MultipleNegativesRankingLoss(fast_encoder),
    )
    loader = trainer.get_train_dataloader()
    assert isinstance(loader, prefetch.PrefetchedDataLoader)
    assert len(loader) == 4
    batches = list(loader)
    assert loader.prefetched_batches == 4
    monkeypatch.setenv("UNSLOTH_ST_PREFETCH", "0")
    plain = trainer.get_train_dataloader()
    assert not isinstance(plain, prefetch.PrefetchedDataLoader)
    for got, want in zip(batches, plain, strict = True):
        assert got.keys() == want.keys()
        for key in got:
            if torch.is_tensor(got[key]):
                assert torch.equal(got[key], want[key])


class _Loader:
    def __init__(
        self,
        items,
        fail_at = None,
    ):
        self.items, self.fail_at, self.closed, self.dataset = items, fail_at, 0, "data"

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        try:
            for i, item in enumerate(self.items):
                if i == self.fail_at:
                    raise ValueError("bad batch")
                yield item
        finally:
            self.closed += 1


def test_prefetched_loader_order_early_stop_and_errors():
    loader = _Loader(list(range(50)))
    wrapped = prefetch.prefetched(loader)
    assert list(wrapped) == list(range(50)) and len(wrapped) == 50 and wrapped.dataset == "data"
    before = threading.active_count()
    for i in wrapped:
        if i == 3:
            break
    assert threading.active_count() == before and loader.closed == 2
    with pytest.raises(ValueError, match = "bad batch"):
        list(prefetch.prefetched(_Loader(list(range(10)), fail_at = 4)))


class _FakeTransformer(torch.nn.Module):
    """Token embedding = f(token, position): any packing that drops or shifts tokens shows up."""

    model_forward_params = {"input_ids", "attention_mask", "position_ids"}

    def forward(self, features):
        ids = features["input_ids"]
        if "position_ids" in features:
            positions = features["position_ids"]
        else:
            positions = (features["attention_mask"].cumsum(1) - 1).clamp(min = 0)
        hidden = torch.stack((ids.float(), positions.float()), -1)
        return {**features, "token_embeddings": hidden}


@pytest.mark.parametrize("side", ["left", "right"])
def test_decoder_packing_restores_the_padded_layout(monkeypatch, side):
    transformer = _FakeTransformer().train()
    transformer._unsloth_original_forward = transformer.forward
    transformer._unsloth_packed_calls = 0
    monkeypatch.setattr(unpadding, "_packable", lambda *args: True)
    ids = torch.tensor([[5, 6, 7, 8], [9, 10, 0, 0], [11, 0, 0, 0]])
    mask = (ids != 0).long()
    if side == "left":
        ids, mask = ids.flip(1), mask.flip(1)
    out = unpadding._packed_transformer_forward(
        transformer, {"input_ids": ids, "attention_mask": mask, "modality": "text"}
    )
    want = transformer.forward({"input_ids": ids, "attention_mask": mask})["token_embeddings"]
    want = want * mask[..., None]
    torch.testing.assert_close(out["token_embeddings"], want)
    assert torch.equal(out["attention_mask"], mask) and torch.equal(out["input_ids"], ids)
    assert "position_ids" not in out and transformer._unsloth_packed_calls == 1


def test_prefetched_loader_keeps_accelerate_identity_for_resume():
    # Trainer resumes mid-epoch through accelerate.skip_first_batches, which dispatches on isinstance.
    accel = pytest.importorskip("accelerate.data_loader")
    loader = prefetch.prefetched(accel.DataLoaderShard(list(range(12)), batch_size = 2))
    assert isinstance(loader, accel.DataLoaderShard)
    assert [b.tolist() for b in loader] == [
        [2 * i, 2 * i + 1] for i in range(6)
    ] and loader.prefetched_batches == 6
    skipped = accel.skip_first_batches(loader, 2)
    assert isinstance(skipped, accel.DataLoaderShard)
    assert [b.tolist() for b in skipped] == [[2 * i, 2 * i + 1] for i in range(2, 6)]


def test_prefetch_is_off_under_a_distributed_process_group(monkeypatch):
    # Loader collectives (dispatcher broadcasts, RNG sync) must stay on the training thread.
    trainer = types.SimpleNamespace(
        model = types.SimpleNamespace(_unsloth_trust_remote_code = False),
        args = types.SimpleNamespace(dataloader_num_workers = 0),
    )
    assert prefetch.prefetch_enabled(trainer)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    assert not prefetch.prefetch_enabled(trainer)
