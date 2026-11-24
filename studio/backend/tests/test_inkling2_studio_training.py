# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inkling 2 training in Studio: QLoRA survives the latest-transformers sidecar, and
"Train on completions" masks with Inkling's role tokens, so end-of-turn tokens are trained."""

import asyncio
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

INKLING = "unsloth/Inkling-2-Mini"
INKLING_CFG = {
    "model_type": "inkling2_loop",
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "text_config": {"model_type": "inkling2_loop_text"},
    "vision_config": {"model_type": "inkling2_vision"},
}
NEW_ARCH_CFG = {"model_type": "muse_glimmer", "architectures": ["MuseGlimmerForCausalLM"]}


def _sidecar(
    monkeypatch,
    cfg,
    active = True,
):
    import utils.transformers_version as tv

    monkeypatch.setattr(tv, "latest_tier_active_for", lambda *a, **k: active)
    monkeypatch.setattr(tv, "_load_config_json", lambda *a, **k: cfg)
    monkeypatch.setattr(tv, "_remote_lora_base", lambda *a, **k: None)
    return tv


@pytest.mark.parametrize("model_type", ["inkling2_loop", "inkling2"])
def test_latest_sidecar_keeps_4bit_for_inkling2(monkeypatch, model_type):
    tv = _sidecar(monkeypatch, {**INKLING_CFG, "model_type": model_type})
    assert tv.latest_tier_forces_16bit(INKLING) is False


def test_latest_sidecar_still_forces_16bit_for_other_new_architectures(monkeypatch):
    tv = _sidecar(monkeypatch, NEW_ARCH_CFG)
    assert tv.latest_tier_forces_16bit("unsloth/Muse-Glimmer-30B") is True
    # An unreadable config keeps the conservative 16-bit flip.
    tv = _sidecar(monkeypatch, None)
    assert tv.latest_tier_forces_16bit("unsloth/Muse-Glimmer-30B") is True


def test_no_sidecar_never_forces_16bit(monkeypatch):
    tv = _sidecar(monkeypatch, NEW_ARCH_CFG, active = False)
    assert tv.latest_tier_forces_16bit("unsloth/Muse-Glimmer-30B") is False


def test_training_load_stays_4bit_for_inkling2_on_the_sidecar(monkeypatch):
    """The flip /api/train/start and the worker both apply (provenance.effective_training_load_in_4bit)."""
    from core.training.provenance import effective_training_load_in_4bit

    _sidecar(monkeypatch, INKLING_CFG)
    assert effective_training_load_in_4bit({"load_in_4bit": True}, INKLING, None) is True
    _sidecar(monkeypatch, NEW_ARCH_CFG)
    assert effective_training_load_in_4bit({"load_in_4bit": True}, "x/new", None) is False
    # A 4-bit Inkling checkpoint resumes on the sidecar: its load mode is unchanged.
    _sidecar(monkeypatch, INKLING_CFG)
    assert (
        effective_training_load_in_4bit(
            {"load_in_4bit": True, "require_exact_model_resource": True}, INKLING, None
        )
        is True
    )


def _upgrade_route(monkeypatch, upgrade, latest_tier, cfg):
    pytest.importorskip("fastapi", reason = "inference stack not installed")
    inf_mod = pytest.importorskip("routes.inference", reason = "inference stack not installed")
    import utils.transformers_latest as latest_mod

    _sidecar(monkeypatch, cfg, active = latest_tier)
    monkeypatch.setattr(inf_mod, "_requires_trust_remote_code_for_model", lambda *a, **k: False)
    monkeypatch.setattr(
        inf_mod, "_hf_offline_if_unreachable", lambda: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(latest_mod, "check_upgrade_for_model", lambda *a, **k: upgrade)
    monkeypatch.setattr(
        "utils.models.model_config.get_base_model_from_lora_identifier", lambda *a, **k: None
    )
    from models.inference import TransformersUpgradeCheckRequest

    return asyncio.run(
        inf_mod.check_transformers_upgrade_route(
            TransformersUpgradeCheckRequest(model_name = INKLING), "tester"
        )
    )


def test_train_tab_is_not_told_inkling2_loses_4bit(monkeypatch):
    offer = {
        "model_type": "inkling2_loop",
        "pypi_version": "5.24.0",
        "supported_in_pypi": True,
        "supported_in_main": True,
    }
    before = _upgrade_route(monkeypatch, offer, False, INKLING_CFG)
    assert before.requires_transformers_upgrade is True
    assert before.forces_16bit is False
    after = _upgrade_route(monkeypatch, None, True, INKLING_CFG)
    assert after.latest_tier_active is True
    assert after.forces_16bit is False
    assert before.sidecar_keeps_4bit is True
    assert after.sidecar_keeps_4bit is True


def test_other_new_architectures_do_not_claim_4bit(monkeypatch):
    offer = {
        "model_type": "muse_glimmer",
        "pypi_version": "5.24.0",
        "supported_in_pypi": True,
        "supported_in_main": True,
    }
    before = _upgrade_route(monkeypatch, offer, False, NEW_ARCH_CFG)
    assert before.forces_16bit is True
    assert before.sidecar_keeps_4bit is False


def test_text_only_inkling2_config_keeps_4bit(monkeypatch):
    tv = _sidecar(monkeypatch, {"model_type": "inkling2_loop_text"})
    assert tv.latest_tier_forces_16bit("someone/inkling2-mini-text") is False


def test_bnb_checkpoint_is_not_called_16bit(monkeypatch):
    """A bnb checkpoint loads 4-bit from its own quantization_config, so a 16-bit claim would be false."""
    cfg = {
        **NEW_ARCH_CFG,
        "quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True},
    }
    tv = _sidecar(monkeypatch, cfg)
    assert tv.latest_tier_forces_16bit("someone/Muse-Glimmer-30B-bnb-4bit") is False


class _InklingTokenizer:
    def get_added_vocab(self):
        return {
            "<|message_system|>": 200000,
            "<|message_user|>": 200001,
            "<|message_model|>": 200002,
            "<|content_text|>": 200003,
            "<|end_message|>": 200004,
            "<|content_model_end_sampling|>": 200006,
        }


class _Holder:
    def __init__(self):
        self.processing_class = _InklingTokenizer()


def test_inkling2_masks_with_role_tokens_not_glued_auto_markers():
    from utils.datasets.completion_masking import apply_completion_masking

    calls = []

    def train_fn(trainer, **kwargs):
        calls.append(kwargs)
        return trainer

    # unsloth_zoo's auto-detect result for Inkling 2: both markers carry the previous turn's end tokens.
    def glued(processor):
        return (
            "<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>",
            "<|end_message|><|message_model|><|content_text|>",
        )

    _, applied = apply_completion_masking(_Holder(), INKLING, train_fn, detect_fn = glued)
    assert applied is True
    assert calls == [{"instruction_part": "<|message_user|>", "response_part": "<|message_model|>"}]


def test_other_tokenizers_keep_auto_detection():
    from utils.datasets.completion_masking import apply_completion_masking

    class _Tok:
        def get_added_vocab(self):
            return {"<|im_start|>": 1, "<|im_end|>": 2}

    class _H:
        processing_class = _Tok()

    calls = []
    apply_completion_masking(
        _H(),
        "unsloth/Qwen3-0.6B",
        lambda trainer, **kw: calls.append(kw) or trainer,
        detect_fn = lambda p: ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    )
    assert calls == [
        {"instruction_part": "<|im_start|>user\n", "response_part": "<|im_start|>assistant\n"}
    ]
