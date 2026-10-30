"""GGUF conversion of LFM3-Audio reads the release config.json layout: llama.cpp's converter registers
Lfm3AudioForConditionalGeneration through an hparams loader that matches only `lfm` / `encoder` /
`depthformer`, while the remote config class saves `text_config` / `audio_encoder_config` /
`depth_config`."""

from __future__ import annotations

import json
import types

import pytest

save_mod = pytest.importorskip("unsloth.save")

RELEASE = {
    "architectures": ["Lfm3AudioForConditionalGeneration"],
    "model_type": "lfm3_audio",
    "codebooks": 8,
    "lfm": {"architectures": ["Lfm3ForCausalLM"], "vocab_size": 128000},
    "encoder": {"dim_in": 128},
    "depthformer": {"layers": 4},
}
SAVED = {
    "architectures": ["Lfm3AudioForConditionalGeneration"],
    "model_type": "lfm3_audio",
    "num_codebooks": 8,
    "text_config": {"model_type": "lfm3", "vocab_size": 128000},
    "audio_encoder_config": {"num_mel_bins": 128},
    "depth_config": {"model_type": "lfm3"},
}


def _write(directory, config):
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    return directory


def _peft_model(base):
    return types.SimpleNamespace(
        peft_config = {"default": types.SimpleNamespace(base_model_name_or_path = str(base))},
        config = types.SimpleNamespace(_name_or_path = str(base)),
    )


def _saved(directory):
    return json.loads((directory / "config.json").read_text(encoding = "utf-8"))


def test_the_transformers_layout_is_replaced_by_the_release_config(tmp_path):
    base = _write(tmp_path / "base", RELEASE)
    merged = _write(tmp_path / "merged", SAVED)
    assert save_mod._restore_lfm3_audio_release_config(_peft_model(base), str(merged))
    assert _saved(merged) == RELEASE


def test_a_release_layout_is_left_alone(tmp_path):
    base = _write(tmp_path / "base", {**RELEASE, "codebooks": 99})
    merged = _write(tmp_path / "merged", RELEASE)
    assert not save_mod._restore_lfm3_audio_release_config(_peft_model(base), str(merged))
    assert _saved(merged) == RELEASE


def test_resized_embeddings_keep_the_saved_config(tmp_path):
    base = _write(tmp_path / "base", RELEASE)
    resized = {**SAVED, "text_config": {**SAVED["text_config"], "vocab_size": 128008}}
    merged = _write(tmp_path / "merged", resized)
    assert not save_mod._restore_lfm3_audio_release_config(_peft_model(base), str(merged))
    assert _saved(merged) == resized


@pytest.mark.parametrize("model_type", ["lfm3_omni", "lfm3", "llama"])
def test_other_models_are_untouched(tmp_path, model_type):
    base = _write(tmp_path / "base", {**RELEASE, "model_type": model_type})
    other = {**SAVED, "model_type": model_type}
    merged = _write(tmp_path / "merged", other)
    assert not save_mod._restore_lfm3_audio_release_config(_peft_model(base), str(merged))
    assert _saved(merged) == other


def test_the_gguf_export_restores_it_before_converting():
    import inspect

    src = inspect.getsource(save_mod.unsloth_save_pretrained_gguf)
    assert src.index("_restore_lfm3_audio_release_config(") < src.index("save_to_gguf(")
