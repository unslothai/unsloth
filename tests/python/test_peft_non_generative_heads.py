# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""get_peft_model on heads that cannot generate (Wav2Vec2ForCTC) uses PEFT's generic PeftModel."""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VISION_PATH = REPO_ROOT / "unsloth" / "models" / "vision.py"


def _get_peft_model_source():
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FastBaseModel":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "get_peft_model":
                    return ast.unparse(item)
    raise AssertionError("FastBaseModel.get_peft_model not found")


def test_non_generative_heads_drop_the_causal_lm_task_type():
    source = _get_peft_model_source()
    assert "prepare_inputs_for_generation" in source
    assert "task_type = None" in source


def _tiny_ctc():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Wav2Vec2Config(
        vocab_size = 16,
        hidden_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 64,
        conv_dim = (16, 16),
        conv_stride = (5, 2),
        conv_kernel = (10, 3),
        num_feat_extract_layers = 2,
        num_conv_pos_embeddings = 16,
        num_conv_pos_embedding_groups = 2,
    )
    torch.manual_seed(0)
    return transformers.Wav2Vec2ForCTC(config)


def test_ctc_head_trains_as_peft_model_but_not_as_causal_lm():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    targets = r".*\.attention\.(?:q_proj|v_proj)"

    model = _tiny_ctc()
    assert not hasattr(model, "prepare_inputs_for_generation")
    with pytest.raises(AttributeError, match = "prepare_inputs_for_generation"):
        peft.get_peft_model(
            model, peft.LoraConfig(r = 4, target_modules = targets, task_type = "CAUSAL_LM")
        )

    model = peft.get_peft_model(
        _tiny_ctc(), peft.LoraConfig(r = 4, target_modules = targets, task_type = None)
    )
    audio = torch.randn(2, 4000)
    labels = torch.randint(1, 15, (2, 3))
    model(input_values = audio, labels = labels).loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


def test_causal_lms_keep_the_generation_hook():
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size = 32,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
    )
    assert hasattr(transformers.LlamaForCausalLM(config), "prepare_inputs_for_generation")


def _helper(name):
    import torch

    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    wanted = {name, "_is_audio_head_model", "_AUDIO_HEAD_SUFFIXES"}
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in wanted for t in n.targets))
    ]
    ns = {"torch": torch}
    exec(compile(ast.Module(nodes, []), str(VISION_PATH), "exec"), ns)
    return ns[name]


def test_ctc_and_classifier_heads_are_trained_and_saved():
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    heads = _helper("_trainable_head_modules")
    assert heads(_tiny_ctc()) == ["lm_head"]

    config = transformers.ASTConfig(
        hidden_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 32,
        num_mel_bins = 16,
        max_length = 32,
        num_labels = 3,
    )
    classifier = transformers.ASTForAudioClassification(config)
    assert heads(classifier) == ["classifier"]
    assert heads(transformers.ASTModel(config)) == []  # a bare encoder is its own backbone

    whisper = transformers.WhisperConfig(
        vocab_size = 64,
        d_model = 16,
        encoder_layers = 1,
        decoder_layers = 1,
        encoder_attention_heads = 2,
        decoder_attention_heads = 2,
        encoder_ffn_dim = 32,
        decoder_ffn_dim = 32,
        num_mel_bins = 8,
        max_source_positions = 8,
        num_labels = 3,
    )
    # backbone under `encoder`, not base_model_prefix
    assert heads(transformers.WhisperForAudioClassification(whisper)) == ["projector", "classifier"]

    xvector = _tiny_ctc().config
    xvector.tdnn_dim, xvector.tdnn_kernel, xvector.tdnn_dilation = (8, 8), (3, 1), (1, 1)
    xvector.xvector_output_dim = 8
    names = heads(transformers.Wav2Vec2ForXVector(xvector))
    assert (
        "feature_extractor" not in names and "classifier" in names
    )  # PEFT would also match wav2vec2.feature_extractor

    model = peft.get_peft_model(
        _tiny_ctc(),
        peft.LoraConfig(r = 4, target_modules = r".*\.attention\.q_proj", modules_to_save = ["lm_head"]),
    )
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("lm_head" in n for n in trainable)
    source = _get_peft_model_source()
    assert "modules_to_save = _trainable_head_modules(model) or None" in source
    assert (
        "_cannot_generate or _is_audio_head_model(model)" in source
    )  # ParakeetForCTC can generate
    assert "type(model) is PeftModel" in source  # a generic PeftModel is not wrapped twice


def test_num_labels_on_an_audio_classifier_loads_the_audio_head():
    source = VISION_PATH.with_name("loader.py").read_text(encoding = "utf-8")
    shortcut = source[source.index("if _num_labels is not None:") :][:900]
    assert "AutoModelForAudioClassification" in shortcut
