# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Component A tests: capability detection must never execute model repo code.

Covers: load_model_config defaults trust_remote_code False; the _VISION_CHECK_SCRIPT
subprocess literal keeps remote code off; registry-backed vision/audio detection from
raw config.json (repo-code VLMs detected without execution; ForConditionalGeneration
false positives fixed); and the model-details / GPU probes never enable remote code.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from utils.models.model_config import (
    load_model_config,
    is_vision_model,
    _is_vlm,
    _raw_config_has_vision_config,
    _vision_detection_cache,
    _VISION_CHECK_SCRIPT,
    _VLM_MODEL_TYPES,
    _AUDIO_ONLY_MODEL_TYPES,
    _VLM_CLASS_NAMES,
)


@pytest.fixture(autouse = True)
def _clear_vision_cache():
    _vision_detection_cache.clear()
    yield
    _vision_detection_cache.clear()


def _write_model_dir(
    tmp_path,
    cfg,
    with_evil_module = False,
):
    """Write a local model dir, optionally with an auto_map module that writes a sentinel
    on import so accidental code execution during detection shows up on disk."""
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    if with_evil_module:
        sentinel = tmp_path / "PWNED_SENTINEL"
        (tmp_path / "modeling_evil.py").write_text(
            "import os\n"
            f"open({str(sentinel)!r}, 'w').write('pwned')\n"
            "class EvilConfig: pass\n"
            "class EvilModel: pass\n"
        )
    return str(tmp_path)


# load_model_config default
class TestLoadModelConfigDefault:
    @patch("transformers.AutoConfig.from_pretrained")
    def test_default_off_with_token(self, fp):
        load_model_config("org/m", token = "hf_x")
        assert fp.call_args.kwargs["trust_remote_code"] is False

    @patch("utils.models.model_config.without_hf_auth")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_default_off_public(self, fp, no_auth):
        from contextlib import nullcontext

        no_auth.return_value = nullcontext()
        load_model_config("org/m", use_auth = False)
        assert fp.call_args.kwargs["trust_remote_code"] is False

    @patch("transformers.AutoConfig.from_pretrained")
    def test_default_off_cached_auth(self, fp):
        load_model_config("org/m", use_auth = True)
        assert fp.call_args.kwargs["trust_remote_code"] is False

    @patch("transformers.AutoConfig.from_pretrained")
    def test_explicit_true_forwarded(self, fp):
        load_model_config("org/m", token = "t", trust_remote_code = True)
        assert fp.call_args.kwargs["trust_remote_code"] is True


# subprocess script literal
def test_vision_check_script_disables_remote_code():
    assert '"trust_remote_code": False' in _VISION_CHECK_SCRIPT
    assert '"trust_remote_code": True' not in _VISION_CHECK_SCRIPT


# _is_vlm matrix (pure function, registry-backed)
def _cfg(**kw):
    return SimpleNamespace(**kw)


class TestIsVlm:
    def test_deepseek_ocr_vision_via_vision_config(self):
        # auto_map repo-code model; vision-ness is declarative.
        c = _cfg(
            model_type = "deepseek_vl_v2",
            architectures = ["DeepseekOCRForCausalLM"],
            vision_config = {},
            projector_config = {},
        )
        assert _is_vlm(c) is True

    def test_kimi_vision_via_vision_config(self):
        c = _cfg(
            model_type = "kimi_k25",
            architectures = ["KimiK25ForConditionalGeneration"],
            vision_config = {},
        )
        assert _is_vlm(c) is True

    def test_glm_flash_text_is_not_vision(self):
        c = _cfg(model_type = "glm4_moe_lite", architectures = ["Glm4MoeLiteForCausalLM"])
        assert _is_vlm(c) is False

    def test_gemma4_vision_via_vision_config(self):
        c = _cfg(
            model_type = "gemma4_unified",
            architectures = ["Gemma4UnifiedForConditionalGeneration"],
            vision_config = {},
            image_token_id = 1,
        )
        assert _is_vlm(c) is True

    def test_t5_not_misclassified_as_vision(self):
        # Regression: ForConditionalGeneration must NOT be a vision signal.
        c = _cfg(model_type = "t5", architectures = ["T5ForConditionalGeneration"])
        assert _is_vlm(c) is False

    def test_bart_not_misclassified_as_vision(self):
        c = _cfg(model_type = "bart", architectures = ["BartForConditionalGeneration"])
        assert _is_vlm(c) is False

    def test_whisper_audio_not_vision(self):
        c = _cfg(model_type = "whisper", architectures = ["WhisperForConditionalGeneration"])
        assert _is_vlm(c) is False

    def test_csm_audio_not_vision(self):
        c = _cfg(model_type = "csm", architectures = ["CsmForConditionalGeneration"])
        assert _is_vlm(c) is False

    def test_native_vlm_via_registry_model_type(self):
        # llava is in the transformers vision registry.
        assert "llava" in _VLM_MODEL_TYPES
        c = _cfg(model_type = "llava", architectures = ["LlavaForConditionalGeneration"])
        assert _is_vlm(c) is True

    def test_native_vlm_via_registry_class_name(self):
        # Class-name match works even if model_type were unknown.
        cls = next(iter(_VLM_CLASS_NAMES))
        c = _cfg(model_type = "something_unlisted", architectures = [cls])
        assert _is_vlm(c) is True

    def test_omni_audio_plus_vision_is_vision(self):
        # An audio-registry model_type with an explicit vision sub-config is still vision.
        audio_mt = next(iter(_AUDIO_ONLY_MODEL_TYPES - _VLM_MODEL_TYPES))
        c = _cfg(model_type = audio_mt, architectures = ["X"], vision_config = {})
        assert _is_vlm(c) is True


# _raw_config_has_vision_config (code-free reader, mocked HF download)
def _mock_raw_config(tmp_path, payload):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(payload))
    return p


class TestRawConfigVisionReader:
    @pytest.mark.parametrize(
        "payload,expected",
        [
            (
                {
                    "model_type": "deepseek_vl_v2",
                    "architectures": ["DeepseekOCRForCausalLM"],
                    "auto_map": {"AutoConfig": "modeling_deepseekocr.DeepseekOCRConfig"},
                    "vision_config": {},
                    "projector_config": {},
                },
                True,
            ),
            (
                {
                    "model_type": "kimi_k25",
                    "architectures": ["KimiK25ForConditionalGeneration"],
                    "auto_map": {"AutoConfig": "configuration_kimi_k25.KimiK25Config"},
                    "vision_config": {},
                },
                True,
            ),
            ({"model_type": "glm4_moe_lite", "architectures": ["Glm4MoeLiteForCausalLM"]}, False),
            (
                {
                    "model_type": "gemma4_unified",
                    "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                    "vision_config": {},
                },
                True,
            ),
            ({"model_type": "t5", "architectures": ["T5ForConditionalGeneration"]}, False),
            (
                {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]},
                False,
            ),
            ({"model_type": "csm", "architectures": ["CsmForConditionalGeneration"]}, False),
        ],
    )
    def test_reader(self, tmp_path, payload, expected):
        cfg_path = _mock_raw_config(tmp_path, payload)
        with (
            patch("utils.models.model_config.is_local_path", return_value = False),
            patch("huggingface_hub.hf_hub_download", return_value = str(cfg_path)),
        ):
            assert _raw_config_has_vision_config("org/model") is expected

    def test_reader_never_executes_remote_code(self, tmp_path):
        # Even with auto_map present, the reader only parses JSON: no AutoConfig touched.
        cfg_path = _mock_raw_config(
            tmp_path,
            {
                "model_type": "deepseek_vl_v2",
                "architectures": ["DeepseekOCRForCausalLM"],
                "auto_map": {"AutoConfig": "modeling_deepseekocr.DeepseekOCRConfig"},
                "vision_config": {},
            },
        )
        with (
            patch("utils.models.model_config.is_local_path", return_value = False),
            patch("huggingface_hub.hf_hub_download", return_value = str(cfg_path)),
            patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect = AssertionError("AutoConfig must not be called"),
            ),
        ):
            assert _raw_config_has_vision_config("org/deepseek-ocr") is True


# Probes: model-details + GPU estimate never execute remote code
def test_gpu_estimate_probe_is_code_free():
    from utils.hardware import hardware

    cfg = {
        "model_type": "glm4_moe_lite",
        "hidden_size": 4096,
        "num_hidden_layers": 40,
        "max_position_embeddings": 8192,
    }
    with (
        patch("utils.transformers_version._load_config_json", return_value = cfg),
        patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect = AssertionError("AutoConfig must not be called"),
        ),
    ):
        out = hardware._load_config_for_gpu_estimate("unsloth/GLM-4.7-Flash")
    assert out.max_position_embeddings == 8192
    assert out.hidden_size == 4096


def test_models_route_source_has_no_remote_code_probe():
    # The metadata probe must never build a trust_remote_code=True loader; referencing
    # the static consent scanner or the requires_trust_remote_code flag is fine.
    import inspect
    import routes.models as models_route

    src = inspect.getsource(models_route)
    assert "trust_remote_code = True" not in src
    assert "trust_remote_code=True" not in src


# Adversarial end-to-end: is_vision_model + the two metadata probes never run auto_map.
def test_no_code_execution_on_detection(tmp_path):
    # A malicious local auto_map -> modeling_evil must not execute through any probe.
    cfg = {
        "model_type": "deepseek_vl_v2",
        "architectures": ["DeepseekOCRForCausalLM"],
        "auto_map": {
            "AutoConfig": "modeling_evil.EvilConfig",
            "AutoModel": "modeling_evil.EvilModel",
        },
        "vision_config": {"image_size": 1024},
        "max_position_embeddings": 4096,
    }
    path = _write_model_dir(tmp_path, cfg, with_evil_module = True)
    sentinel = tmp_path / "PWNED_SENTINEL"

    from utils.hardware.hardware import _load_config_for_gpu_estimate
    from utils.transformers_version import _load_config_json

    result = is_vision_model(path)
    ns = _load_config_for_gpu_estimate(path)
    raw = _load_config_json(path)

    assert not sentinel.exists(), "SECURITY FAILURE: auto_map code executed during detection"
    assert result is True  # detected as vision via raw vision_config, no exec
    assert ns is not None and getattr(ns, "max_position_embeddings", None) == 4096
    assert raw is not None and raw.get("model_type") == "deepseek_vl_v2"


@pytest.mark.parametrize(
    "cfg, expected",
    [
        # repo-code VLMs (auto_map) detected via declarative vision_config
        (
            {
                "model_type": "deepseek_vl_v2",
                "architectures": ["DeepseekOCRForCausalLM"],
                "auto_map": {"AutoConfig": "x.Y"},
                "vision_config": {},
            },
            True,
        ),
        (
            {
                "model_type": "kimi_k25",
                "architectures": ["KimiK25ForConditionalGeneration"],
                "auto_map": {"AutoConfig": "x.Y"},
                "vision_config": {},
            },
            True,
        ),
        # newer-native vision via vision_config
        (
            {
                "model_type": "gemma4_unified",
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "vision_config": {},
                "image_token_id": 7,
            },
            True,
        ),
        # text / seq2seq / audio that share the ForConditionalGeneration suffix
        ({"model_type": "glm4_moe_lite", "architectures": ["Glm4MoeLiteForCausalLM"]}, False),
        ({"model_type": "t5", "architectures": ["T5ForConditionalGeneration"]}, False),
        ({"model_type": "bart", "architectures": ["BartForConditionalGeneration"]}, False),
        ({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}, False),
        ({"model_type": "csm", "architectures": ["CsmForConditionalGeneration"]}, False),
        # registry-native VLMs via model_type
        ({"model_type": "qwen2_vl", "architectures": ["Qwen2VLForConditionalGeneration"]}, True),
        ({"model_type": "llava", "architectures": ["LlavaForConditionalGeneration"]}, True),
    ],
)
def test_is_vision_model_end_to_end(tmp_path, cfg, expected):
    path = _write_model_dir(tmp_path, cfg)
    assert is_vision_model(path) is expected, f"{cfg['model_type']} expected vision={expected}"


def test_registry_derivation():
    # Registry-derived sets are large and include the curated repo-code VLMs.
    assert len(_VLM_MODEL_TYPES) >= 50, f"_VLM_MODEL_TYPES too small: {len(_VLM_MODEL_TYPES)}"
    assert (
        len(_AUDIO_ONLY_MODEL_TYPES) >= 20
    ), f"_AUDIO_ONLY too small: {len(_AUDIO_ONLY_MODEL_TYPES)}"
    for repo_vlm in ("deepseek_vl_v2", "kimi_k25", "phi3_v", "cogvlm2", "minicpmv"):
        assert repo_vlm in _VLM_MODEL_TYPES, f"curated repo-code VLM {repo_vlm} missing"
    for native in ("llava", "qwen2_vl"):
        assert native in _VLM_MODEL_TYPES, f"registry-native VLM {native} missing"
    for audio in ("whisper", "csm"):
        assert audio in _AUDIO_ONLY_MODEL_TYPES, f"audio type {audio} missing"
