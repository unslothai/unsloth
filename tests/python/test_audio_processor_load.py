# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""Audio processors load as processors, not as their text tokenizer (Dia, Higgs Audio, Granite Speech)."""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VISION_PATH = REPO_ROOT / "unsloth" / "models" / "vision.py"


def _helpers():
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    wanted = {"_from_pretrained_left_padded", "_missing_audio_backend"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    ns = {"re": re}
    exec(compile(ast.Module(nodes, []), str(VISION_PATH), "exec"), ns)
    return ns["_from_pretrained_left_padded"], ns["_missing_audio_backend"]


class _Loader:
    """Mimics a processor whose from_pretrained forwards padding_side into a codec model's __init__."""

    def __init__(self, error = None):
        self.calls = []
        self.error = error

    def from_pretrained(self, name, **kwargs):
        self.calls.append(kwargs)
        if "padding_side" in kwargs and self.error is not None:
            raise self.error
        return ("processor", name)


def test_padding_side_is_dropped_only_when_it_breaks_the_load():
    left_padded, _ = _helpers()
    plain = _Loader()
    assert left_padded(plain, "repo", token = None) == ("processor", "repo")
    assert plain.calls == [{"padding_side": "left", "token": None}]

    codec = _Loader(
        TypeError("DacModel.__init__() got an unexpected keyword argument 'padding_side'")
    )
    assert left_padded(codec, "repo", token = None) == ("processor", "repo")
    assert codec.calls == [{"padding_side": "left", "token": None}, {"token": None}]

    other = _Loader(TypeError("expected str, bytes or os.PathLike object, not NoneType"))
    with pytest.raises(TypeError, match = "PathLike"):
        left_padded(other, "repo")


def test_missing_audio_backend_names_the_library():
    _, missing = _helpers()
    error = ImportError(
        "\nGraniteSpeechFeatureExtractor requires the torchaudio library but it was not found in your "
        "environment. Please install it and restart your\nruntime."
    )
    assert missing(error) == "torchaudio"
    assert missing(ImportError("No module named 'foo'")) is None
    assert missing(ValueError("requires the torchaudio library")) is None
    assert missing(None) is None


def test_processor_loads_use_the_retry():
    source = VISION_PATH.read_text(encoding = "utf-8")
    body = source[source.index("def _acquire_processor(lfo):") :]
    body = body[: body.index("return _tok, _err")]
    # Every primary auto_processor load goes through the retry; none passes padding_side itself.
    assert body.count("_from_pretrained_left_padded(") == 2
    assert "auto_processor.from_pretrained(" not in body


@pytest.mark.slow
def test_real_dia_processor_rejects_padding_side():
    """The defect being worked around, against the installed transformers (skipped offline)."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "DiaProcessor"):
        pytest.skip("transformers has no Dia")
    left_padded, _ = _helpers()
    try:
        processor = left_padded(transformers.AutoProcessor, "nari-labs/Dia-1.6B-0626")
    except OSError:
        pytest.skip("Hub not reachable")
    assert type(processor).__name__ == "DiaProcessor"


def _is_speech_forward():
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    nodes = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_is_speech_forward"
    ]
    import inspect

    ns = {"inspect": inspect}
    exec(compile(ast.Module(nodes, []), str(VISION_PATH), "exec"), ns)
    return ns["_is_speech_forward"]


def _tiny_whisper():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.WhisperConfig(
        vocab_size = 64,
        d_model = 32,
        encoder_layers = 1,
        decoder_layers = 1,
        encoder_attention_heads = 2,
        decoder_attention_heads = 2,
        encoder_ffn_dim = 64,
        decoder_ffn_dim = 64,
        num_mel_bins = 16,
        max_source_positions = 8,
        max_target_positions = 16,
        pad_token_id = 0,
        bos_token_id = 1,
        eos_token_id = 2,
        decoder_start_token_id = 1,
    )
    torch.manual_seed(0)
    model = transformers.WhisperForConditionalGeneration(config)
    batch = {"input_features": torch.randn(2, 16, 16), "labels": torch.randint(3, 64, (2, 5))}
    return model, batch


def test_speech_models_get_no_task_wrapper():
    is_speech = _is_speech_forward()
    transformers = pytest.importorskip("transformers")
    for cls in (
        "WhisperForConditionalGeneration",
        "Wav2Vec2ForCTC",
        "MoonshineForConditionalGeneration",
    ):
        cls = getattr(transformers, cls)
        assert is_speech(cls.__new__(cls))
    for cls in ("LlamaForCausalLM", "Qwen2AudioForConditionalGeneration"):
        cls = getattr(transformers, cls)
        assert not is_speech(cls.__new__(cls))

    source = VISION_PATH.read_text(encoding = "utf-8")
    assert "_is_speech_forward(model)" in source
    assert "task_type = None" in source


def test_causal_lm_wrapper_breaks_whisper_training():
    """The defect: PeftModelForCausalLM passes input_ids = None into Whisper's **kwargs."""
    peft = pytest.importorskip("peft")
    model, batch = _tiny_whisper()
    broken = peft.get_peft_model(
        model, peft.LoraConfig(r = 4, target_modules = ["q_proj", "v_proj"], task_type = "CAUSAL_LM")
    )
    with pytest.raises(TypeError, match = "multiple values for keyword argument 'input_ids'"):
        broken(**batch)

    model, batch = _tiny_whisper()
    fixed = peft.get_peft_model(
        model, peft.LoraConfig(r = 4, target_modules = ["q_proj", "v_proj"], task_type = None)
    )
    loss = fixed(**batch).loss
    loss.backward()
    assert loss.isfinite()
