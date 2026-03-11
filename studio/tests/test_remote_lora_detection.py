# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""
Test remote LoRA adapter detection via HuggingFace Hub API.

Verifies that we can detect whether a remote HF model is a LoRA adapter
by checking for adapter_config.json in the repo file listing.
"""

import pytest
from huggingface_hub import model_info


def is_remote_lora_adapter(model_id: str, hf_token: str = None) -> bool:
    """
    Check if a remote HuggingFace model is a LoRA adapter
    by looking for adapter_config.json in its repo files.
    """
    try:
        info = model_info(model_id, token = hf_token)
        filenames = [s.rfilename for s in info.siblings]
        return "adapter_config.json" in filenames
    except Exception:
        return False


class TestRemoteLoRADetection:
    """Test remote LoRA adapter detection via HF Hub API."""

    def test_lora_adapter_detected(self):
        """edbeeching/llama-se-rl-adapter is a known LoRA adapter on HF."""
        result = is_remote_lora_adapter("edbeeching/llama-se-rl-adapter")
        assert (
            result is True
        ), "Expected edbeeching/llama-se-rl-adapter to be detected as a LoRA adapter"

    def test_base_model_not_detected_as_lora(self):
        """google/gemma-3-4b-it is a full base model, not a LoRA adapter."""
        result = is_remote_lora_adapter("google/gemma-3-4b-it")
        assert (
            result is False
        ), "Expected google/gemma-3-4b-it to NOT be detected as a LoRA adapter"

    def test_nonexistent_model_returns_false(self):
        """A nonexistent model should return False, not raise."""
        result = is_remote_lora_adapter("this-org-does-not-exist/fake-model-12345")
        assert result is False, "Expected nonexistent model to return False"
