# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "llm_compressor_consent_hub_push", Path(__file__).with_name("test_export_hub_push.py")
)
_HUB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HUB)


class _CompressedModel(_HUB._Model):
    def __init__(self):
        super().__init__()
        self.kwargs = []

    def save_pretrained_merged(
        self,
        save_directory,
        tokenizer,
        save_method = None,
        token = None,
        install_missing_dependencies = True,
    ):
        self.kwargs.append(install_missing_dependencies)
        output = Path(f"{save_directory}-FP8-Dynamic")
        output.mkdir(parents = True, exist_ok = True)
        (output / "model.safetensors").write_bytes(b"weights")


def _compressed_backend(
    monkeypatch,
    shadow_calls,
    shadow_disabled = False,
    shadow_pp = None,
):
    backend = _HUB._non_mlx_backend(monkeypatch, "test_llm_compressor_consent_backend", [], {})
    backend.current_model = _CompressedModel()
    export_module = sys.modules["test_llm_compressor_consent_backend"]
    monkeypatch.setattr(export_module, "_has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(export_module, "_compressed_export_supported", lambda: True)

    us = sys.modules["unsloth.save"]
    us._normalize_torchao_method = lambda method: None
    us._normalize_compressed_method = lambda alias: ("fp8", "FP8_DYNAMIC", "FP8-Dynamic")
    us._COMPRESSED_QUANTIZE_PYTHONPATH_ENV = "UNSLOTH_TEST_COMPRESSED_PYTHONPATH"
    us._transformers_exceeds_llm_compressor_ceiling = lambda: (False, "4.57.6")

    tv = types.ModuleType("utils.transformers_version")

    def llmcompressor_shadow_pythonpath(*, allow_provision = False):
        shadow_calls.append(allow_provision)
        return shadow_pp if allow_provision and not shadow_disabled else None

    tv.llmcompressor_shadow_pythonpath = llmcompressor_shadow_pythonpath
    tv._llmcompressor_main_disabled = lambda: shadow_disabled
    tv._env_offline = lambda: False
    monkeypatch.setitem(sys.modules, "utils.transformers_version", tv)
    return backend


@pytest.mark.parametrize(
    "consented, shadow_disabled, shadow_pp, expected",
    [
        (False, False, None, False),
        (False, True, None, False),
        (True, True, None, True),
        (True, False, "/shadow", True),
        (True, False, None, False),
    ],
)
def test_compressed_export_states_consent_explicitly(
    tmp_path, monkeypatch, consented, shadow_disabled, shadow_pp, expected
):
    # unsloth.save defaults install_missing_dependencies=True; must pass False.
    shadow_calls: list = []
    backend = _compressed_backend(monkeypatch, shadow_calls, shadow_disabled, shadow_pp)

    success, message, output_path = backend.export_merged_model(
        str(tmp_path / "export"),
        format_type = "FP8 (compressed-tensors)",
        install_missing_dependencies = consented,
    )

    assert success is True, message
    assert backend.current_model.kwargs == [expected]
    assert shadow_calls == [consented]
    assert output_path.endswith("export-FP8-Dynamic")
