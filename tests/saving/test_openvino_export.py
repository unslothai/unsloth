# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for OpenVINO export functionality in unsloth.save."""

from __future__ import annotations

import inspect
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from unsloth.save import (
    patch_saving_functions,
    unsloth_save_pretrained_openvino,
    unsloth_push_to_hub_openvino,
    _unsloth_save_openvino,
)


def test_openvino_export_signatures():
    """Verify export functions exist and accept expected parameters."""
    save_sig = inspect.signature(unsloth_save_pretrained_openvino)
    assert "save_directory" in save_sig.parameters
    assert "tokenizer" in save_sig.parameters
    assert "quantization_type" in save_sig.parameters
    assert "quantization_config" in save_sig.parameters
    assert "push_to_hub" in save_sig.parameters
    assert "token" in save_sig.parameters

    hub_sig = inspect.signature(unsloth_push_to_hub_openvino)
    assert "repo_id" in hub_sig.parameters
    assert "tokenizer" in hub_sig.parameters
    assert "quantization_type" in hub_sig.parameters


def test_openvino_methods_attached_by_patch_saving_functions():
    """Verify patch_saving_functions attaches OpenVINO methods to model."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.config = MagicMock()

        def push_to_hub(self, repo_id, **kwargs):
            """Push to hub docstring."""

    model = DummyModel()
    patched = patch_saving_functions(model)

    assert hasattr(patched, "save_pretrained_openvino")
    assert callable(patched.save_pretrained_openvino)
    assert hasattr(patched, "push_to_hub_openvino")
    assert callable(patched.push_to_hub_openvino)


def test_openvino_missing_dependency_error(tmp_path):
    """Verify helpful error is raised when optimum-intel is not installed."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

    model = DummyModel()
    with patch.dict("sys.modules", {"optimum.intel.openvino": None}):
        with pytest.raises(ImportError, match = "requires `optimum-intel` and `openvino`"):
            _unsloth_save_openvino(
                model = model,
                save_directory = str(tmp_path / "ov_out"),
            )


def test_openvino_invalid_quantization_type(tmp_path):
    """Verify ValueError on unsupported quantization type."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.save_pretrained = MagicMock()

    mock_ov_module = MagicMock()
    with patch.dict("sys.modules", {"optimum.intel.openvino": mock_ov_module}):
        with pytest.raises(ValueError, match = "Unknown OpenVINO quantization_type"):
            _unsloth_save_openvino(
                model = DummyModel(),
                save_directory = str(tmp_path / "ov_out"),
                quantization_type = "unsupported_quant_scheme",
            )
