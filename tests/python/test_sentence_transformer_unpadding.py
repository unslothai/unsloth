# SPDX-License-Identifier: Apache-2.0
"""Real tiny encoders exercise sentence unpadding without downloading checkpoints.

The simulated Flash function executes independent SDPA segments on CUDA. It proves
model integration and gradients, not the numerical behavior or speed of FlashAttention.
The real backend parameter is separately skipped when FlashAttention is unavailable.
"""

import copy
from contextlib import nullcontext
import importlib.util
import inspect
import subprocess
import sys

import pytest
import torch

from real_accelerator import has_real_accelerator


@pytest.fixture(params = ["bert", "roberta"])
def tiny_model(request, tmp_path):
    return _build_tiny_model(request.param, tmp_path)


def _build_tiny_model(
    family,
    tmp_path,
    hidden_size = 16,
):
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("tiny encoder fixtures require optional sentence-transformers")
    import transformers

    if int(transformers.__version__.split(".")[0]) < 5:
        pytest.skip("the encoder attention interface requires Transformers 5")
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        BertConfig,
        BertModel,
        PreTrainedTokenizerFast,
        RobertaConfig,
        RobertaModel,
    )

    pad = 0 if family == "bert" else 1
    config_class, model_class = (
        (BertConfig, BertModel) if family == "bert" else (RobertaConfig, RobertaModel)
    )
    config = config_class(
        vocab_size = 64,
        hidden_size = hidden_size,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size = 24,
        max_position_embeddings = 16,
        type_vocab_size = 2,
        pad_token_id = pad,
        hidden_dropout_prob = 0.0,
        attention_probs_dropout_prob = 0.0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(4460)
    checkpoint = tmp_path / family
    model_class(config).save_pretrained(checkpoint)
    vocab = {f"word{i}": i for i in range(64)}
    vocab.pop(f"word{pad}")
    vocab["[PAD]"] = pad
    vocab.pop("word2")
    vocab["[UNK]"] = 2
    tokenizer = Tokenizer(WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        pad_token = "[PAD]",
        unk_token = "[UNK]",
        model_max_length = 12,
    ).save_pretrained(checkpoint)
    model_options = (
        "model_kwargs"
        if "model_kwargs" in inspect.signature(Transformer).parameters
        else "model_args"
    )
    transformer = Transformer(str(checkpoint), **{model_options: {"attn_implementation": "sdpa"}})
    return SentenceTransformer(modules = [transformer, Pooling(hidden_size)], device = "cpu")


def test_missing_sentence_transformers_skips_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(pytest.skip.Exception, match = "optional sentence-transformers"):
        _build_tiny_model("bert", tmp_path)


def _features(model, device = "cpu"):
    lengths = [3, 7, 12]  # Packed length 22 exceeds the position table, each original row fits.
    ids = torch.full((3, 12), model[0].auto_model.config.pad_token_id, dtype = torch.long)
    mask = torch.zeros_like(ids)
    types = torch.zeros_like(ids)
    for row, length in enumerate(lengths):
        ids[row, :length] = torch.arange(5 + row, 5 + row + length)
        mask[row, :length] = 1
        types[row, 1:length:2] = 1
    return {
        key: value.to(device)
        for key, value in dict(input_ids = ids, attention_mask = mask, token_type_ids = types).items()
    }


def _copy_features(features):
    return {
        key: value.clone() if torch.is_tensor(value) else value for key, value in features.items()
    }


@pytest.fixture(params = ["simulated", "real"])
def kernel(request, monkeypatch):
    from unsloth.utils import attention_dispatch as ad

    if not has_real_accelerator() or not torch.cuda.is_available():
        pytest.skip("packed path requires CUDA")
    calls = []
    if request.param == "real":
        if not ad.HAS_FLASH_ATTENTION:
            pytest.skip("real FlashAttention is not installed")
        original = ad.flash_attn_varlen_func

        def counted(*args, **kwargs):
            calls.append(args[0].shape[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(ad, "flash_attn_varlen_func", counted)
    else:

        def simulated(query, key, value, cu_q, cu_k, max_q, max_k, **kwargs):
            assert kwargs["causal"] is False
            assert torch.equal(cu_q, cu_k)
            calls.append(query.shape[0])
            segments = []
            bounds = cu_q.tolist()
            for start, end in zip(bounds, bounds[1:]):
                q, k, v = (
                    tensor[start:end].transpose(0, 1).unsqueeze(0) for tensor in (query, key, value)
                )
                output = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p = kwargs.get("dropout_p", 0.0),
                    scale = kwargs.get("softmax_scale"),
                    is_causal = False,
                )
                segments.append(output.squeeze(0).transpose(0, 1))
            return torch.cat(segments)

        monkeypatch.setattr(ad, "flash_attn_varlen_func", simulated)
    monkeypatch.setattr(ad, "select_attention_backend", lambda **kwargs: ad.FLASH_VARLEN)
    return calls


def _enable(model):
    from unsloth.models._sentence_transformer_unpadding import enable_sentence_transformer_unpadding
    assert enable_sentence_transformer_unpadding(model)
    return model


def test_routed_attention_mask_override_keeps_padded_semantics(tiny_model, kernel):
    reference = tiny_model.cuda().half().train()
    reference.module_kwargs = {"0": ["attention_mask"]}
    candidate = _enable(copy.deepcopy(reference))
    features = _features(reference, "cuda")
    override = features["attention_mask"].clone()
    override[1, 3] = 0
    ordinary = reference(_copy_features(features))["sentence_embedding"]
    expected = reference(_copy_features(features), attention_mask = override)
    assert not torch.equal(ordinary, expected["sentence_embedding"])
    actual = candidate(_copy_features(features), attention_mask = override)
    assert not kernel, "routed mask overrides must not use stale feature-mask metadata"
    torch.testing.assert_close(actual["token_embeddings"], expected["token_embeddings"])
    torch.testing.assert_close(actual["sentence_embedding"], expected["sentence_embedding"])
    actual["sentence_embedding"].float().square().sum().backward()
    expected["sentence_embedding"].float().square().sum().backward()
    for (name, parameter), (reference_name, reference_parameter) in zip(
        candidate.named_parameters(), reference.named_parameters()
    ):
        assert name == reference_name
        if reference_parameter.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, reference_parameter.grad)


@pytest.mark.parametrize("invalid", [None, "true", "false", "AUTO", 0, 1, [], {}])
def test_invalid_unpadding_policy_rejected_before_loading(monkeypatch, invalid):
    from transformers import AutoConfig
    from unsloth import FastSentenceTransformer

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid use_unpadding reached model loading")

    monkeypatch.setattr(AutoConfig, "from_pretrained", unexpected_load)
    with pytest.raises(ValueError, match = "use_unpadding"):
        FastSentenceTransformer.from_pretrained("must-not-load", use_unpadding = invalid)


def test_automatic_size_boundary_and_forced_override(tiny_model, kernel):
    from unsloth import FastSentenceTransformer
    from unsloth.models._sentence_transformer_unpadding import enable_sentence_transformer_unpadding

    assert (
        inspect.signature(FastSentenceTransformer.from_pretrained)
        .parameters["use_unpadding"]
        .default
        is False
    )
    reference = tiny_model.cuda().half().train()
    candidate = copy.deepcopy(reference)
    assert enable_sentence_transformer_unpadding(candidate, auto = True)
    encoder = candidate[0].auto_model.encoder
    assert encoder._unsloth_min_tokens == 8192
    features = _features(candidate, "cuda")
    expected = reference(_copy_features(features))["token_embeddings"]
    actual = candidate(_copy_features(features))["token_embeddings"]
    assert not kernel
    torch.testing.assert_close(actual, expected)

    # Inject a bounded boundary after asserting the real policy constant;
    # this exercises the same comparison without 1024 simulated segments.
    size = features["attention_mask"].numel()
    encoder._unsloth_min_tokens = size + 1
    actual = candidate(_copy_features(features))["token_embeddings"]
    assert not kernel
    torch.testing.assert_close(actual, expected)
    encoder._unsloth_min_tokens = size
    actual = candidate(_copy_features(features))["token_embeddings"]
    assert len(kernel) == 2
    valid = features["attention_mask"].bool()
    torch.testing.assert_close(actual[valid], expected[valid], atol = 2e-3, rtol = 1e-2)

    forced = _enable(copy.deepcopy(reference))
    assert forced[0].auto_model.encoder._unsloth_min_tokens == 0
    kernel.clear()
    forced(_copy_features(features))
    assert len(kernel) == 2


@pytest.mark.parametrize("threshold,packed", [(0.38, True), (0.40, False)])
def test_padding_fraction_threshold(tiny_model, kernel, threshold, packed):
    from unsloth.models._sentence_transformer_unpadding import enable_sentence_transformer_unpadding

    reference = tiny_model.cuda().half().train()
    candidate = copy.deepcopy(reference)
    assert enable_sentence_transformer_unpadding(candidate, padding_threshold = threshold)
    features = _features(candidate, "cuda")  # 14 / 36 padding lies between both thresholds.
    expected = reference(_copy_features(features))["token_embeddings"]
    actual = candidate(_copy_features(features))["token_embeddings"]
    assert len(kernel) == (2 if packed else 0)
    valid = (
        features["attention_mask"].bool()
        if packed
        else torch.ones_like(features["attention_mask"], dtype = torch.bool)
    )
    torch.testing.assert_close(actual[valid], expected[valid], atol = 2e-3, rtol = 1e-2)


@pytest.mark.parametrize("family", ["bert", "roberta"])
def test_unsupported_flash_head_dimension_remains_padded(family, tmp_path, kernel):
    from unsloth.models._sentence_transformer_unpadding import enable_sentence_transformer_unpadding

    reference = _build_tiny_model(family, tmp_path, hidden_size = 528).cuda().half().train()
    candidate = copy.deepcopy(reference)
    assert reference[0].auto_model.config.hidden_size // 2 == 264
    assert not enable_sentence_transformer_unpadding(candidate)
    features = _features(candidate, "cuda")
    expected = reference(_copy_features(features))["token_embeddings"]
    actual = candidate(_copy_features(features))["token_embeddings"]
    assert not kernel
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("scope", ["whole", "inner"])
def test_torch_compile_eager_backend_preserves_padded_values(tiny_model, kernel, scope):
    reference = tiny_model.cuda().half().train()
    candidate = _enable(copy.deepcopy(reference))
    features = _features(candidate, "cuda")
    expected = reference(_copy_features(features))["token_embeddings"]
    if scope == "whole":
        compiled = torch.compile(candidate, backend = "eager")
    else:
        candidate[0].model = torch.compile(candidate[0].auto_model, backend = "eager")
        compiled = candidate
    actual = compiled(_copy_features(features))["token_embeddings"]
    assert not kernel
    torch.testing.assert_close(actual, expected)
    actual.float().square().sum().backward()
    assert any(p.grad is not None and bool(p.grad.norm() > 0) for p in candidate.parameters())
    torch._dynamo.reset()


@pytest.mark.parametrize("peft", [False, True])
@pytest.mark.parametrize("forward_params", [None, {"input_ids", "attention_mask"}])
def test_upstream_compile_restores_original_execution(
    tiny_model, monkeypatch, peft, forward_params
):
    from unsloth import FastSentenceTransformer
    from unsloth.utils import attention_dispatch as ad

    monkeypatch.setattr(ad, "select_attention_backend", lambda **kwargs: ad.FLASH_VARLEN)
    model = tiny_model
    transformer = model[0]
    base = transformer.auto_model
    original_forward, original_encoder = model.forward, base.encoder.forward
    transformer.model_forward_params = forward_params
    _enable(model)
    if forward_params is not None:
        transformer.model_forward_params.add("later_extension")
    if peft:
        from peft import LoraConfig, get_peft_model
        transformer.model = get_peft_model(base, LoraConfig(r = 2, target_modules = ["query", "value"]))
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    calls = []

    def compile_original(inner, mode):
        calls.append(inner)
        assert inner is transformer.auto_model
        assert model.forward == original_forward
        assert base.encoder.forward == original_encoder
        assert base.config._attn_implementation == "sdpa"
        assert not getattr(model, "_unsloth_unpadding_installed", False)
        assert not hasattr(base.encoder, "_unsloth_original_forward")
        expected_params = None if forward_params is None else forward_params | {"later_extension"}
        assert transformer.model_forward_params == expected_params
        return inner

    monkeypatch.setattr(torch, "compile", compile_original)
    assert FastSentenceTransformer._apply_torch_compile(model) is model
    assert FastSentenceTransformer._apply_torch_compile(model) is model
    assert len(calls) == 2
    assert parameter_ids == [id(parameter) for parameter in model.parameters()]


@pytest.mark.parametrize("owner", ["model", "encoder"])
def test_compile_restoration_preserves_user_forward(tiny_model, monkeypatch, owner):
    from unsloth.models._sentence_transformer_unpadding import (
        disable_sentence_transformer_unpadding,
    )
    from unsloth.utils import attention_dispatch as ad

    monkeypatch.setattr(ad, "select_attention_backend", lambda **kwargs: ad.FLASH_VARLEN)
    model = _enable(tiny_model)
    target = model if owner == "model" else model[0].auto_model.encoder

    def custom_forward(*args, **kwargs):
        return None

    target.forward = custom_forward
    assert not disable_sentence_transformer_unpadding(model)
    assert target.forward is custom_forward


def test_compile_restoration_preserves_changed_backend(tiny_model, monkeypatch):
    from unsloth.models._sentence_transformer_unpadding import (
        disable_sentence_transformer_unpadding,
    )
    from unsloth.utils import attention_dispatch as ad

    monkeypatch.setattr(ad, "select_attention_backend", lambda **kwargs: ad.FLASH_VARLEN)
    model = _enable(tiny_model)
    model[0].auto_model.config._attn_implementation = "eager"
    assert disable_sentence_transformer_unpadding(model)
    assert model[0].auto_model.config._attn_implementation == "eager"
    assert not disable_sentence_transformer_unpadding(model)


def _assert_gradients(
    reference,
    candidate,
    rtol = 0.03,
    autocast_dtype = None,
):
    expected = dict(reference.named_parameters())
    actual = dict(candidate.named_parameters())
    assert expected.keys() == actual.keys()
    relative_errors = []
    for name, parameter in expected.items():
        assert (parameter.grad is None) == (actual[name].grad is None), name
        if parameter.grad is not None:
            # FP16 key-bias gradients should cancel, but reductions can leave a
            # few subnormal quanta. Allow two per element, not a broad atol.
            # Autocast's FP16 key-bias reduction can be stored in an FP32 grad.
            floor = (
                max(2e-7, 2**-23 * parameter.numel() ** 0.5)
                if parameter.grad.dtype == torch.float16
                or (autocast_dtype == torch.float16 and name.endswith("attention.self.key.bias"))
                else 2e-7
            )
            ratio = _assert_relative_norm(
                actual[name].grad, parameter.grad, name, rtol = rtol, floor = floor
            )
            if parameter.grad.float().norm() > 1e-5:
                relative_errors.append(ratio)
    return max(relative_errors, default = 0.0)


def _assert_relative_norm(
    actual,
    expected,
    name,
    rtol = 0.03,
    floor = 2e-7,
):
    actual, expected = actual.float(), expected.float()
    error = torch.linalg.vector_norm(actual - expected)
    norm = torch.linalg.vector_norm(expected)
    assert (
        error <= rtol * norm + floor
    ), f"{name}: error={error.item():.6g}, reference norm={norm.item():.6g}"
    return (error / norm.clamp_min(1e-30)).item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_embeddings_valid_tokens_loss_gradients_and_update(
    tiny_model, kernel, dtype, record_property
):
    _check_training_step(tiny_model, kernel, dtype, record_property)


@pytest.mark.parametrize(
    "weight_dtype,autocast_dtype",
    [
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ],
    ids = ["fp32-amp-fp16", "fp32-amp-bf16", "fp16-amp-fp16", "bf16-amp-bf16"],
)
def test_autocast_embeddings_gradients_and_update(
    tiny_model, kernel, weight_dtype, autocast_dtype, record_property
):
    _check_training_step(
        tiny_model, kernel, weight_dtype, record_property, autocast_dtype = autocast_dtype
    )


def _check_training_step(
    tiny_model,
    kernel,
    dtype,
    record_property,
    autocast_dtype = None,
):
    reference = tiny_model.cuda().to(dtype = dtype).train()
    candidate = _enable(copy.deepcopy(reference))
    bf16 = (autocast_dtype or dtype) == torch.bfloat16
    features = _features(reference, "cuda")
    context = (
        torch.autocast("cuda", dtype = autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    with context:
        expected = reference(_copy_features(features))
        assert not kernel
        actual = candidate(_copy_features(features))
    assert kernel == [22, 22]
    for key in ("sentence_embedding", "token_embeddings"):
        assert actual[key].dtype == expected[key].dtype
        if autocast_dtype is not None:
            assert actual[key].dtype == torch.float32
    keep = features["attention_mask"].bool()
    for key in ("sentence_embedding", "token_embeddings"):
        left, right = actual[key], expected[key]
        if key == "token_embeddings":
            left, right = left[keep], right[keep]
        torch.testing.assert_close(left, right, atol = 1.5e-2 if bf16 else 2e-3, rtol = 1e-2)
    target = torch.linspace(-1, 1, 16, device = "cuda")
    expected_loss = (expected["sentence_embedding"].float() - target).square().mean()
    actual_loss = (actual["sentence_embedding"].float() - target).square().mean()
    torch.testing.assert_close(actual_loss, expected_loss, atol = 1e-3, rtol = 1e-2)
    expected_loss.backward()
    actual_loss.backward()
    gradient_error = _assert_gradients(
        reference, candidate, rtol = 0.05 if bf16 else 0.03, autocast_dtype = autocast_dtype
    )
    record_property("max_gradient_relative_l2_above_1e-5", gradient_error)
    initial = [parameter.detach().float().clone() for parameter in reference.parameters()]
    for model in (reference, candidate):
        torch.optim.SGD(model.parameters(), lr = 0.1 if bf16 else 0.01).step()
    expected_deltas = torch.cat(
        [
            (parameter.detach().float() - before).flatten()
            for parameter, before in zip(reference.parameters(), initial)
        ]
    )
    actual_deltas = torch.cat(
        [
            (parameter.detach().float() - before).flatten()
            for parameter, before in zip(candidate.parameters(), initial)
        ]
    )
    assert expected_deltas.norm() > 1e-5
    delta_error = _assert_relative_norm(
        actual_deltas, expected_deltas, "optimizer deltas", rtol = 0.1 if bf16 else 0.08
    )
    record_property("optimizer_delta_relative_l2", delta_error)


def test_autocast_state_is_rechecked_and_fp32_fallback_keeps_all_tokens(tiny_model, kernel):
    reference = tiny_model.cuda().float().train()
    candidate = _enable(copy.deepcopy(reference))
    features = _features(reference, "cuda")
    keep = features["attention_mask"].bool()
    for autocast_dtype in (None, torch.float16, None, torch.bfloat16, None):
        kernel.clear()
        context = (
            torch.autocast("cuda", dtype = autocast_dtype)
            if autocast_dtype is not None
            else torch.autocast("cuda", enabled = False)
        )
        with context:
            expected = reference(_copy_features(features))
            assert not kernel
            actual = candidate(_copy_features(features))
        assert (
            actual["token_embeddings"].dtype == expected["token_embeddings"].dtype == torch.float32
        )
        if autocast_dtype is None:
            assert not kernel
            torch.testing.assert_close(actual["token_embeddings"], expected["token_embeddings"])
            torch.testing.assert_close(actual["sentence_embedding"], expected["sentence_embedding"])
        else:
            assert kernel == [22, 22]
            atol = 1.5e-2 if autocast_dtype == torch.bfloat16 else 2e-3
            torch.testing.assert_close(
                actual["token_embeddings"][keep],
                expected["token_embeddings"][keep],
                atol = atol,
                rtol = 1e-2,
            )
            torch.testing.assert_close(
                actual["sentence_embedding"],
                expected["sentence_embedding"],
                atol = atol,
                rtol = 1e-2,
            )


def test_lora_frozen_base_then_unfrozen_gradients(tiny_model, kernel):
    from peft import LoraConfig, get_peft_model

    reference = tiny_model.cuda().half().train()
    candidate = _enable(copy.deepcopy(reference))
    for model in (reference, candidate):
        torch.manual_seed(71)
        transformer = model[0]
        peft_model = get_peft_model(
            transformer.auto_model,
            LoraConfig(
                r = 2,
                lora_alpha = 2,
                target_modules = ["query", "key", "value"],
                lora_dropout = 0.0,
                task_type = "FEATURE_EXTRACTION",
            ),
        )
        if isinstance(getattr(type(transformer), "auto_model", None), property):
            transformer.model = peft_model
        else:
            transformer.auto_model = peft_model
    for unfreeze in (False, True):
        for model in (reference, candidate):
            model.zero_grad(set_to_none = True)
            if unfreeze:
                model.requires_grad_(True)
            result = model(_features(model, "cuda"))["sentence_embedding"]
            result.float().square().mean().backward()
        _assert_gradients(reference, candidate)
        adapter_grads = [
            parameter.grad.float().norm()
            for name, parameter in candidate.named_parameters()
            if "lora_B" in name and parameter.grad is not None
        ]
        assert sum(adapter_grads) > 1e-7
        if unfreeze:
            base_gradient = candidate[0].auto_model.get_input_embeddings().weight.grad
            assert base_gradient is not None and base_gradient.float().norm() > 1e-7
    assert len(kernel) == 4


def test_training_dropout_has_finite_nonzero_gradients(tiny_model, kernel):
    model = _enable(tiny_model.cuda().half().train())
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.1
    output = model(_features(model, "cuda"))["sentence_embedding"]
    output.float().square().mean().backward()
    assert kernel == [22, 22]
    assert torch.isfinite(output).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model[0].auto_model.embeddings.word_embeddings.weight.grad.float().norm() > 1e-7


@pytest.mark.parametrize("reentrant", [False, True])
def test_checkpoint_replay_preserves_sequence_metadata(tiny_model, kernel, reentrant):
    reference = tiny_model.cuda().half().train()
    candidate = _enable(copy.deepcopy(reference))
    for model in (reference, candidate):
        model[0].auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs = {"use_reentrant": reentrant}
        )
        output = model(_features(model, "cuda"))["sentence_embedding"]
        output.float().square().mean().backward()
    _assert_gradients(reference, candidate)
    assert len(kernel) == 4  # Each layer's forward and recomputation use the packed path.


@pytest.mark.parametrize(
    "case", ["left_padding", "mask_hole", "custom_positions", "attended_pad_token"]
)
def test_original_embedding_positions_are_preserved(tiny_model, kernel, case):
    reference = tiny_model.cuda().half().train()
    candidate = _enable(copy.deepcopy(reference))
    features = _features(reference, "cuda")
    if case == "left_padding":
        features = {name: value.roll(5, dims = 1) for name, value in features.items()}
    elif case == "mask_hole":
        features["attention_mask"][1, 3] = 0
    elif case == "custom_positions":
        features["position_ids"] = torch.arange(12, device = "cuda").flip(0).unsqueeze(0)
    elif case == "attended_pad_token":
        features["input_ids"][1, 3] = reference[0].auto_model.config.pad_token_id
    expected = reference(_copy_features(features))
    actual = candidate(_copy_features(features))
    assert kernel == [int(features["attention_mask"].sum())] * 2
    keep = features["attention_mask"].bool()
    torch.testing.assert_close(
        actual["token_embeddings"][keep], expected["token_embeddings"][keep], atol = 2e-3, rtol = 1e-2
    )
    torch.testing.assert_close(
        actual["sentence_embedding"], expected["sentence_embedding"], atol = 2e-3, rtol = 1e-2
    )


def test_stock_trainer_and_mnrl_complete_one_update(tiny_model, kernel, tmp_path):
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    model = _enable(tiny_model.cuda().half().train())
    initial = model[0].auto_model.embeddings.word_embeddings.weight.detach().clone()
    dataset = Dataset.from_dict(
        {
            "anchor": ["word5 word6 word7", "word8 word9 word10 word11", "word12 word13"],
            "positive": ["word5 word7", "word8 word10", "word12 word14 word15 word16 word17"],
        }
    )
    trainer = SentenceTransformerTrainer(
        model = model,
        args = SentenceTransformerTrainingArguments(
            output_dir = str(tmp_path / "training"),
            max_steps = 1,
            per_device_train_batch_size = 3,
            learning_rate = 0.01,
            optim = "sgd",
            report_to = "none",
            save_strategy = "no",
            disable_tqdm = True,
        ),
        train_dataset = dataset,
        loss = MultipleNegativesRankingLoss(model),
    )
    result = trainer.train()
    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert len(kernel) >= 4
    assert not torch.equal(model[0].auto_model.embeddings.word_embeddings.weight, initial)


@pytest.mark.parametrize(
    "fallback",
    [
        "eval",
        "disabled",
        "hidden_states",
        "feature_hidden_states",
        "config_hidden_states",
        "attention_weights",
        "custom_module",
        "empty_row",
        "nonbinary",
        "runtime_chunk",
        "runtime_backend",
    ],
)
def test_unsupported_calls_remain_padded(tiny_model, kernel, fallback):
    from torch import nn

    model = _enable(tiny_model.cuda().half().train())
    features = _features(model, "cuda")
    kwargs = {}
    if fallback == "eval":
        model.eval()
    elif fallback == "disabled":
        model._unsloth_use_unpadding = False
    elif fallback == "hidden_states":
        kwargs["output_hidden_states"] = True
    elif fallback == "feature_hidden_states":
        features["output_hidden_states"] = True
    elif fallback == "config_hidden_states":
        model[0].auto_model.config.output_hidden_states = True
    elif fallback == "attention_weights":
        kwargs["output_attentions"] = True
    elif fallback == "custom_module":

        class InspectTokens(nn.Module):
            def forward(self, features, **kwargs):
                features["inspected_tokens"] = features["token_embeddings"].sum()
                return features

        model._modules = {"0": model[0], "intervening": InspectTokens(), "1": model[1]}
    elif fallback == "empty_row":
        features["attention_mask"][0] = 0
    elif fallback == "nonbinary":
        features["attention_mask"][0, 0] = 2
    elif fallback == "runtime_chunk":
        for layer in model[0].auto_model.encoder.layer:
            layer.chunk_size_feed_forward = 3
    elif fallback == "runtime_backend":
        model[0].auto_model.set_attn_implementation("sdpa")
    result = model(features, **kwargs)
    assert not kernel
    assert result["token_embeddings"].shape == (3, 12, 16)


def test_keyword_input_and_direct_encoder_positional_calls(tiny_model, kernel):
    reference = tiny_model.cuda().half().train()
    candidate = _enable(copy.deepcopy(reference))
    features = _features(reference, "cuda")
    expected = reference(input = _copy_features(features))
    actual = candidate(input = _copy_features(features))
    assert len(kernel) == 2
    torch.testing.assert_close(
        actual["sentence_embedding"], expected["sentence_embedding"], atol = 2e-3, rtol = 1e-2
    )
    kernel.clear()
    direct_expected = reference[0](_copy_features(features))["token_embeddings"]
    direct_actual = candidate[0](_copy_features(features))["token_embeddings"]
    torch.testing.assert_close(direct_actual, direct_expected)
    hidden = reference[0].auto_model.embeddings(
        input_ids = features["input_ids"],
        token_type_ids = features["token_type_ids"],
    )
    mask = features["attention_mask"][:, None, None, :].bool()
    direct_expected = reference[0].auto_model.encoder(hidden, mask, None, None, None, False)
    direct_actual = candidate[0].auto_model.encoder(hidden, mask, None, None, None, False)
    torch.testing.assert_close(direct_actual.last_hidden_state, direct_expected.last_hidden_state)
    assert not kernel


def test_deepcopy_after_install_uses_its_own_parameters(tiny_model, kernel):
    original = _enable(tiny_model.cuda().half().train())
    clone = copy.deepcopy(original)
    with torch.no_grad():
        clone[0].auto_model.embeddings.word_embeddings.weight[5, 0] += 1
    original.zero_grad(set_to_none = True)
    clone.zero_grad(set_to_none = True)
    features = _features(clone, "cuda")
    actual = clone(_copy_features(features))["sentence_embedding"]
    assert len(kernel) == 2
    clone._unsloth_use_unpadding = False
    reference = clone(_copy_features(features))["sentence_embedding"]
    torch.testing.assert_close(actual, reference, atol = 2e-3, rtol = 1e-2)
    actual.float().square().mean().backward()
    assert all(parameter.grad is None for parameter in original.parameters())
    assert clone[0].auto_model.embeddings.word_embeddings.weight.grad.float().norm() > 1e-7


@pytest.mark.parametrize("forward_filter", ["native", "explicit"])
def test_cpu_fallback_keeps_all_token_values(tiny_model, monkeypatch, forward_filter):
    from unsloth.utils import attention_dispatch as ad

    monkeypatch.setattr(ad, "select_attention_backend", lambda **kwargs: ad.FLASH_VARLEN)
    reference = tiny_model.train()
    if forward_filter == "explicit" and reference[0].model_forward_params is None:
        reference[0].model_forward_params = set(
            inspect.signature(reference[0].auto_model.forward).parameters
        )
    passthrough = reference[0].model_forward_params is None
    candidate = _enable(copy.deepcopy(reference))
    if passthrough:
        assert candidate[0].model_forward_params is None
    wrapper = candidate.forward
    _enable(candidate)
    assert candidate.forward is wrapper
    features = _features(reference)
    actual = candidate(_copy_features(features))
    expected = reference(_copy_features(features))
    torch.testing.assert_close(actual["token_embeddings"], expected["token_embeddings"])
    torch.testing.assert_close(actual["sentence_embedding"], expected["sentence_embedding"])


@pytest.mark.parametrize(
    "unsupported", ["no_flash", "eager", "decoder", "chunking", "custom_pipeline"]
)
def test_installation_keeps_unsupported_models_unchanged(tiny_model, monkeypatch, unsupported):
    from unsloth.models._sentence_transformer_unpadding import enable_sentence_transformer_unpadding
    from unsloth.utils import attention_dispatch as ad

    monkeypatch.setattr(ad, "select_attention_backend", lambda **kwargs: ad.FLASH_VARLEN)
    config = tiny_model[0].auto_model.config
    if unsupported == "no_flash":
        monkeypatch.setattr(ad, "select_attention_backend", lambda **kwargs: ad.SDPA)
    elif unsupported == "eager":
        config._attn_implementation = "eager"
    elif unsupported == "decoder":
        config.is_decoder = True
    elif unsupported == "chunking":
        config.chunk_size_feed_forward = 3
    elif unsupported == "custom_pipeline":
        tiny_model._modules = {
            "0": tiny_model[0],
            "custom": torch.nn.Identity(),
            "1": tiny_model[1],
        }
    original_forward = tiny_model.forward
    original_backend = config._attn_implementation
    assert not enable_sentence_transformer_unpadding(tiny_model)
    assert tiny_model.forward == original_forward
    assert config._attn_implementation == original_backend


def test_native_save_reloads_without_unsloth_in_fresh_process(kernel, tmp_path):
    checkpoint_roots = []
    for family in ("bert", "roberta"):
        family_root = tmp_path / family
        model = _enable(_build_tiny_model(family, family_root).cuda().half().train())
        initial = model[0].auto_model.embeddings.word_embeddings.weight.detach().clone()
        calls_before = len(kernel)
        result = model(_features(model, "cuda"))["sentence_embedding"]
        result.float().square().mean().backward()
        torch.optim.SGD(model.parameters(), lr = 0.01).step()
        assert not torch.equal(initial, model[0].auto_model.embeddings.word_embeddings.weight)
        assert len(kernel) == calls_before + 2
        before = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        output = family_root / "saved"
        model.save_pretrained(str(output))
        assert before == {name: tuple(value.shape) for name, value in model.state_dict().items()}
        model.eval().cpu().float()
        features = _features(model)
        expected = model(_copy_features(features))["sentence_embedding"].detach()
        torch.save({"features": features, "expected": expected}, family_root / "expected.pt")
        checkpoint_roots.append(str(family_root))
    script = """import sys, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
for checkpoint in map(Path, sys.argv[1:]):
    data = torch.load(checkpoint / 'expected.pt', weights_only=True)
    model = SentenceTransformer(str(checkpoint / 'saved'), device='cpu').float().eval()
    with torch.no_grad():
        actual = model(data['features'])['sentence_embedding']
    torch.testing.assert_close(actual, data['expected'], atol=1e-5, rtol=1e-4)
    assert 'unsloth' not in sys.modules
    print('STOCK_RELOAD_OK', checkpoint.name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *checkpoint_roots],
        cwd = tmp_path,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STOCK_RELOAD_OK bert" in result.stdout
    assert "STOCK_RELOAD_OK roberta" in result.stdout
