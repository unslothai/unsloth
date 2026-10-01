# SPDX-License-Identifier: AGPL-3.0-only
"""Packed vs padded parity for SentenceTransformer encoder unpadding on tiny BERT / RoBERTa.

"simulated" replaces flash_attn_varlen_func with per-segment SDPA (integration + gradients,
not FlashAttention numerics); "xformers" runs the real BlockDiagonalMask kernel.
"""

import copy
import importlib.util
import inspect
import subprocess
import sys

import pytest
import torch

from real_accelerator import has_real_accelerator

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("sentence_transformers") is None,
        reason = "needs the optional sentence-transformers package",
    ),
    pytest.mark.skipif(
        int(__import__("transformers").__version__.split(".")[0]) < 5,
        reason = "encoder attention interface needs Transformers 5",
    ),
]

LENGTHS = [3, 7, 12]


def _build(
    family,
    root,
    pooling = "mean",
):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    from transformers import RobertaConfig, RobertaModel

    pad = 0 if family == "bert" else 1
    config_class, model_class = {
        "bert": (BertConfig, BertModel),
        "roberta": (RobertaConfig, RobertaModel),
    }[family]
    config = config_class(
        vocab_size = 64,
        hidden_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size = 24,
        max_position_embeddings = 16,
        pad_token_id = pad,
        hidden_dropout_prob = 0.0,
        attention_probs_dropout_prob = 0.0,
    )
    torch.manual_seed(4460)
    checkpoint = root / family
    model_class(config).save_pretrained(checkpoint)
    vocab = {f"word{i}": i for i in range(64) if i not in (pad, 2)}
    vocab.update({"[PAD]": pad, "[UNK]": 2})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tokenizer, pad_token = "[PAD]", unk_token = "[UNK]"
    ).save_pretrained(checkpoint)
    key = (
        "model_kwargs"
        if "model_kwargs" in inspect.signature(Transformer).parameters
        else "model_args"
    )
    transformer = Transformer(str(checkpoint), **{key: {"attn_implementation": "sdpa"}})
    return SentenceTransformer(
        modules = [transformer, Pooling(16, pooling_mode = pooling)], device = "cpu"
    )


def _features(
    model,
    device = "cuda",
    lengths = LENGTHS,
):
    width = max(lengths)
    ids = torch.full((len(lengths), width), model[0].auto_model.config.pad_token_id)
    mask = torch.zeros_like(ids)
    for row, length in enumerate(lengths):
        ids[row, :length] = torch.arange(5 + row, 5 + row + length)
        mask[row, :length] = 1
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}


@pytest.fixture(params = ["bert", "roberta"])
def tiny(request, tmp_path):
    return _build(request.param, tmp_path)


@pytest.fixture(params = ["simulated", "xformers"])
def kernel(request, monkeypatch):
    """Forces the backend and records the packed token count of every attention call."""
    from unsloth.utils import attention_dispatch as ad
    from unsloth.utils import packing

    if not has_real_accelerator() or not torch.cuda.is_available():
        pytest.skip("packed path needs CUDA")
    calls = []
    if request.param == "xformers":
        if not ad.HAS_XFORMERS or packing._XFormersBidirectionalMask is None:
            pytest.skip("xFormers with BlockDiagonalMask is not installed")
        original = ad.xformers_attention

        def counted(query, key, value, **kwargs):
            assert type(kwargs["attn_bias"]) is packing._XFormersBidirectionalMask
            calls.append(query.shape[1])
            return original(query, key, value, **kwargs)

        monkeypatch.setattr(ad, "xformers_attention", counted)
        monkeypatch.setattr(ad, "select_attention_backend", lambda **_: ad.XFORMERS)
        return calls

    def simulated(query, key, value, cu_q, cu_k, max_q, max_k, **kwargs):
        assert kwargs["causal"] is False and torch.equal(cu_q, cu_k)
        calls.append(query.shape[0])
        bounds = cu_q.tolist()
        return torch.cat(
            [
                torch.nn.functional.scaled_dot_product_attention(
                    *(t[a:b].transpose(0, 1) for t in (query, key, value)),
                    dropout_p = kwargs.get("dropout_p", 0.0),
                    scale = kwargs.get("softmax_scale"),
                ).transpose(0, 1)
                for a, b in zip(bounds, bounds[1:])
            ]
        )

    monkeypatch.setattr(ad, "flash_attn_varlen_func", simulated, raising = False)
    monkeypatch.setattr(ad, "select_attention_backend", lambda **_: ad.FLASH_VARLEN)
    return calls


def _enable(model, auto = False):
    from unsloth.models._sentence_transformer_unpadding import enable_sentence_transformer_unpadding
    assert enable_sentence_transformer_unpadding(model, auto = auto)
    return model


def _relative_error(actual, expected):
    actual, expected = actual.float(), expected.float()
    return (
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1e-6)
    ).item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_training_step_matches_padded(tiny, kernel, dtype):
    reference = copy.deepcopy(tiny).cuda().to(dtype).train()
    packed = _enable(tiny).cuda().to(dtype).train()
    outputs = []
    for model in (reference, packed):
        out = model(_features(model))
        loss = out["sentence_embedding"].float().square().mean()
        loss.backward()
        outputs.append((out, loss))
    assert kernel == [sum(LENGTHS)] * 2
    (ref_out, ref_loss), (out, loss) = outputs
    tol = 0.02 if dtype == torch.float16 else 0.05
    assert _relative_error(out["sentence_embedding"], ref_out["sentence_embedding"]) < tol
    assert abs(loss.item() - ref_loss.item()) <= tol * abs(ref_loss.item())
    mask = out["attention_mask"].bool()
    assert torch.equal(
        out["token_embeddings"][~mask], torch.zeros_like(out["token_embeddings"][~mask])
    )
    ref_params = dict(reference.named_parameters())
    for name, param in packed.named_parameters():
        expected = ref_params[name].grad
        if expected is not None and expected.float().norm() > 1e-4:
            assert _relative_error(param.grad, expected) < 3 * tol, name


def test_packed_sentences_do_not_share_attention(tiny, kernel):
    model = _enable(tiny).cuda().half().train()
    features = _features(model)
    before = model(dict(features))["sentence_embedding"]
    features["input_ids"][1:, :3] = 40
    after = model(dict(features))["sentence_embedding"]
    assert torch.equal(before[0], after[0])
    assert not torch.equal(before[1:], after[1:])


@pytest.mark.parametrize("case", ["eval", "auto_small_batch", "no_padding", "fp32_no_autocast"])
def test_ineligible_batches_stay_padded(tiny, kernel, case):
    reference = copy.deepcopy(tiny).cuda()
    model = _enable(tiny, auto = case == "auto_small_batch").cuda()
    lengths = [12, 12, 12] if case == "no_padding" else LENGTHS
    for m in (reference, model):
        if case != "fp32_no_autocast":
            m.half()
        m.train(case != "eval")
    expected = reference(_features(reference, lengths = lengths))["sentence_embedding"]
    actual = model(_features(model, lengths = lengths))["sentence_embedding"]
    assert kernel == []
    torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)


def test_autocast_fp32_weights_are_packed(tiny, kernel):
    model = _enable(tiny).cuda().train()
    with torch.autocast("cuda", dtype = torch.bfloat16):
        model(_features(model))["sentence_embedding"].float().sum().backward()
    assert kernel == [sum(LENGTHS)] * 2


def test_installs_through_unsloth_zoo_bf16_autocast_wrapper(tiny, kernel):
    from unsloth_zoo.training_utils import _wrap_forward_in_bf16_autocast

    base = tiny[0].auto_model
    _wrap_forward_in_bf16_autocast(base, torch.bfloat16)
    assert type(base) is not type(base)._unsloth_autocast_base
    model = _enable(tiny).cuda().train()
    model(_features(model))["sentence_embedding"].float().sum().backward()
    assert kernel == [sum(LENGTHS)] * 2


@pytest.mark.parametrize("pooling", ["cls", "max", "mean_sqrt_len_tokens"])
def test_non_mean_pooling_is_not_installed(tmp_path, kernel, pooling):
    from unsloth.models._sentence_transformer_unpadding import enable_sentence_transformer_unpadding

    model = _build("bert", tmp_path, pooling = pooling)
    forward = model.forward
    assert not enable_sentence_transformer_unpadding(model)
    assert model.forward == forward
    assert model[0].auto_model.config._attn_implementation == "sdpa"


def test_disable_restores_stock_forwards(tiny, kernel):
    from unsloth.models._sentence_transformer_unpadding import (
        disable_sentence_transformer_unpadding,
    )

    encoder = tiny[0].auto_model.encoder
    forward, encoder_forward = tiny.forward, encoder.forward
    _enable(tiny)
    assert disable_sentence_transformer_unpadding(tiny)
    assert tiny.forward == forward and encoder.forward == encoder_forward
    assert tiny[0].auto_model.config._attn_implementation == "sdpa"
    tiny.cuda().half().train()
    tiny(_features(tiny))
    assert kernel == []


def test_invalid_policy_rejected_before_loading():
    from unsloth import FastSentenceTransformer
    for invalid in ("AUTO", "true", 1, None):
        with pytest.raises(ValueError, match = "use_unpadding"):
            FastSentenceTransformer.from_pretrained("does-not-exist", use_unpadding = invalid)


def test_saved_model_reloads_with_stock_sentence_transformers(kernel, tmp_path):
    model = _enable(_build("bert", tmp_path)).cuda().half().train()
    model(_features(model))["sentence_embedding"].float().square().mean().backward()
    torch.optim.SGD(model.parameters(), lr = 0.1).step()
    assert kernel
    model.save_pretrained(str(tmp_path / "saved"))
    model.eval().cpu().float()
    features = _features(model, "cpu")
    with torch.no_grad():
        expected = model(dict(features))["sentence_embedding"]
    torch.save({"features": features, "expected": expected}, tmp_path / "expected.pt")
    script = """import sys, torch
from sentence_transformers import SentenceTransformer
data = torch.load(sys.argv[1] + '/expected.pt', weights_only=True)
model = SentenceTransformer(sys.argv[1] + '/saved', device='cpu').float().eval()
with torch.no_grad():
    actual = model(data['features'])['sentence_embedding']
torch.testing.assert_close(actual, data['expected'], atol=1e-5, rtol=1e-4)
assert 'unsloth' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output = True, text = True, timeout = 300
    )
    assert result.returncode == 0, result.stdout + result.stderr
