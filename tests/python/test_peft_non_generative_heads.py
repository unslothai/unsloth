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
