# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Packed / padding-free SFT rows must keep documents apart. Qwen2-VL / 2.5-VL drop the text row of 2D
position_ids (no packed mask), so they get [4, B, T] ids; remote code that ignores the boundaries falls back
to padded rows; Qwen3-VL, Llama and remote code that honors them stay packed unchanged (CPU, tiny models)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

try:
    import unsloth.trainer as trainer_module  # noqa: E402
except ImportError:
    import unsloth
    if getattr(unsloth, "DEVICE_TYPE", None) != "mlx":
        raise
    pytest.skip("unsloth.trainer is the MLX shim here", allow_module_level = True)

VOCAB = 160
TEXT = dict(
    vocab_size = VOCAB,
    hidden_size = 64,
    intermediate_size = 128,
    num_hidden_layers = 2,
    num_attention_heads = 4,
    num_key_value_heads = 2,
)
VISION = dict(depth = 1, hidden_size = 32, intermediate_size = 64, num_heads = 2, out_hidden_size = 64)


def _qwen_vl(name):
    torch.manual_seed(0)
    if name == "qwen3_vl":
        config_cls = getattr(transformers, "Qwen3VLConfig", None)
        model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        text = dict(
            TEXT,
            head_dim = 16,
            rope_scaling = {
                "rope_type": "default",
                "mrope_section": [2, 3, 3],
                "mrope_interleaved": True,
            },
        )
        vision = dict(VISION, deepstack_visual_indexes = [0], num_position_embeddings = 16)
    else:
        prefix = {"qwen2_vl": "Qwen2VL", "qwen2_5_vl": "Qwen2_5_VL"}[name]
        config_cls = getattr(transformers, f"{prefix}Config", None)
        model_cls = getattr(transformers, f"{prefix}ForConditionalGeneration", None)
        text = dict(TEXT, rope_scaling = {"type": "mrope", "mrope_section": [2, 3, 3]})
        vision = dict(VISION, embed_dim = 32) if name == "qwen2_vl" else VISION
    if config_cls is None:
        pytest.skip(f"transformers without {name}")
    config = config_cls(text_config = text, vision_config = vision, attn_implementation = "sdpa")
    return model_cls(config).float()


def _llama():
    torch.manual_seed(0)
    config = transformers.LlamaConfig(**TEXT, attn_implementation = "sdpa")
    return transformers.LlamaForCausalLM(config).float()


class _RemoteCfg:
    model_type = "tiny_remote"

    def __init__(self):
        self.vocab_size = VOCAB

    def to_dict(self):
        return {"model_type": self.model_type, "vocab_size": self.vocab_size}


class _RemoteForCausalLM(nn.Module):
    """A trust_remote_code-style decoder: causal SDPA over the whole row unless `segments`, where it reads
    the document boundaries from position_ids resets."""

    def __init__(self, segments):
        super().__init__()
        torch.manual_seed(0)
        self.config, self.segments = _RemoteCfg(), segments
        self.embed = nn.Embedding(VOCAB, 32)
        self.qkv = nn.Linear(32, 96)
        self.lm_head = nn.Linear(32, VOCAB)

    def forward(
        self,
        input_ids = None,
        position_ids = None,
        labels = None,
        **kwargs,
    ):
        h = self.embed(input_ids)
        q, k, v = (x.view(*x.shape[:2], 2, 16).transpose(1, 2) for x in self.qkv(h).chunk(3, -1))
        T = input_ids.shape[1]
        mask = torch.ones(T, T, dtype = torch.bool).tril()
        if self.segments and position_ids is not None:
            doc = (position_ids[0] == 0).cumsum(0)
            mask = mask & (doc[:, None] == doc[None, :])
        o = F.scaled_dot_product_attention(q, k, v, attn_mask = mask)
        return SimpleNamespace(logits = self.lm_head(h + o.transpose(1, 2).reshape(h.shape)))


_RemoteForCausalLM.__module__ = "transformers_modules.tiny_remote.modeling_tiny_remote"


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    for name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, name, lambda model, trainer: None)


def _trainer(
    model,
    packing,
    padding_free = None,
):
    class _Stub:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            self.model, self.args, self.data_collator = model, args, None

    module = SimpleNamespace(SFTTrainer = _Stub)
    trainer_module._patch_sft_trainer_auto_packing(module)
    args = SimpleNamespace(packing = packing, padding_free = padding_free, max_length = 512)
    ds = datasets.Dataset.from_dict({"input_ids": [[11, 12, 13], [14, 15]]})
    return module.SFTTrainer(model = model, args = args, train_dataset = ds, processing_class = object())


def _doc_losses(trainer, docs):
    """Summed next-token loss per document, on the batches the trainer would feed the model (TRL's compute_loss
    adds use_cache = False)."""
    model = trainer.model
    if trainer.args.packing or trainer.args.padding_free:
        ids = torch.cat(docs)[None]
        pos = torch.cat([torch.arange(len(d)) for d in docs])[None]
        lengths = torch.tensor([len(d) for d in docs], dtype = torch.int32)
        logits = model(
            input_ids = ids, position_ids = pos, packed_seq_lengths = lengths, use_cache = False
        ).logits[0]
        rows, start = [], 0
        for d in docs:
            rows.append(logits[start : start + len(d)])
            start += len(d)
    else:
        width = max(len(d) for d in docs)
        ids = torch.zeros(len(docs), width, dtype = torch.long)
        mask = torch.zeros(len(docs), width, dtype = torch.long)
        for i, d in enumerate(docs):
            ids[i, : len(d)], mask[i, : len(d)] = d, 1
        logits = model(input_ids = ids, attention_mask = mask, use_cache = False).logits
        rows = [logits[i, : len(d)] for i, d in enumerate(docs)]
    return [F.cross_entropy(r[:-1].float(), d[1:], reduction = "sum") for r, d in zip(rows, docs)]


def _alone(model, doc):
    pos = torch.arange(len(doc))[None]
    logits = model(input_ids = doc[None], position_ids = pos, use_cache = False).logits[0]
    return F.cross_entropy(logits[:-1].float(), doc[1:], reduction = "sum")


def _assert_documents_isolated(trainer):
    model = trainer.model.train()
    g = torch.Generator().manual_seed(1)
    a, a2, b = (torch.randint(10, VOCAB, (n,), generator = g) for n in (12, 12, 9))
    with torch.no_grad():
        packed_b = _doc_losses(trainer, [a, b])[1]
        other_a_b = _doc_losses(trainer, [a2, b])[1]
        alone_b = _alone(model, b)
    torch.testing.assert_close(packed_b, other_a_b, atol = 1e-4, rtol = 1e-4)
    torch.testing.assert_close(packed_b, alone_b, atol = 1e-4, rtol = 1e-4)
    params = [p for p in model.parameters() if p.requires_grad]
    packed_grads = torch.autograd.grad(sum(_doc_losses(trainer, [a, b])), params, allow_unused = True)
    ref_grads = torch.autograd.grad(_alone(model, a) + _alone(model, b), params, allow_unused = True)
    for got, ref in zip(packed_grads, ref_grads):
        if ref is None:
            assert got is None or not got.any()
            continue
        torch.testing.assert_close(got, ref, atol = 1e-4, rtol = 1e-3)


@pytest.mark.parametrize("packing", [False, True])
@pytest.mark.parametrize("name", ["qwen2_vl", "qwen2_5_vl"])
def test_text_row_dropping_mrope_keeps_packing_with_text_prepended_ids(name, packing):
    trainer = _trainer(_qwen_vl(name), packing)
    _assert_documents_isolated(trainer)
    assert trainer.args.padding_free is True and bool(trainer.args.packing) is packing
    assert trainer.model._unsloth_packed_rows_verdict == "text_prepended"


@pytest.mark.parametrize("packing", [False, True])
@pytest.mark.parametrize(
    "build,verdict",
    [
        (lambda: _qwen_vl("qwen3_vl"), "as_is"),
        (_llama, None),
        (lambda: _RemoteForCausalLM(True), "as_is"),
    ],
    ids = ["qwen3_vl", "llama", "remote_segments"],
)
def test_models_that_honor_packed_rows_are_unchanged(build, verdict, packing):
    trainer = _trainer(build(), packing)
    _assert_documents_isolated(trainer)
    assert trainer.args.padding_free is True and bool(trainer.args.packing) is packing
    assert not getattr(trainer.model, "_unsloth_text_prepended_position_ids", False)
    assert getattr(trainer.model, "_unsloth_packed_rows_verdict", None) == verdict


@pytest.mark.parametrize("packing", [False, True])
def test_remote_code_that_mixes_documents_falls_back_to_padded_rows(packing):
    trainer = _trainer(_RemoteForCausalLM(False), packing)
    _assert_documents_isolated(trainer)
    assert not trainer.args.padding_free and not trainer.args.packing


def test_probe_ignores_functional_attention_dropout():
    model = _qwen_vl("qwen3_vl")
    attention = [
        m for m in model.modules() if isinstance(getattr(m, "attention_dropout", None), float)
    ]
    assert attention
    for module in attention:
        module.attention_dropout = 0.5
    assert trainer_module._packed_rows_verdict(model, model.config) == "as_is"
    assert all(module.attention_dropout == 0.5 for module in attention)


def test_verified_hybrid_mrope_families_are_not_probed(monkeypatch):
    import unsloth.models.rl as rl

    if getattr(transformers, "Qwen3_5Config", None) is None:
        pytest.skip("transformers without qwen3_5")
    torch.manual_seed(0)
    rope = {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}
    text = dict(
        TEXT, head_dim = 16, layer_types = ["linear_attention", "full_attention"], rope_parameters = rope
    )
    config = transformers.Qwen3_5Config(text_config = text, vision_config = dict(VISION))
    model = transformers.Qwen3_5ForConditionalGeneration(config).float()
    monkeypatch.setattr(rl, "_packing_honored", lambda *a, **k: pytest.fail("probed"))
    assert trainer_module._uses_multimodal_rope(config)
    assert trainer_module._packed_rows_verdict(model, config) is None


def test_probe_runs_once_and_not_when_packing_is_off(monkeypatch):
    import unsloth.models.rl as rl

    calls, real = [], rl._packing_honored
    monkeypatch.setattr(rl, "_packing_honored", lambda *a, **k: calls.append(1) or real(*a, **k))
    model = _qwen_vl("qwen2_vl")
    assert not _trainer(model, False, padding_free = False).args.padding_free
    assert calls == []
    _trainer(model, True)
    _trainer(model, False)
    assert len(calls) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
