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
#
# SPDX-License-Identifier: Apache-2.0
"""The diffusion runner must honour the GPU-layer split (#7574).

Studio used to accept a manual GPU-layers setting, drop it on the diffusion path, and launch
the visual server with every layer pinned to GPU, so a GGUF larger than VRAM OOMed in
cudaMalloc with no way out. Source-level contract, matching test_llama_cpp_wall_clock_cap.py:
importing the backend module pulls in the whole studio stack.
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "core"
    / "inference"
    / "llama_cpp.py"
)
SRC = SOURCE_PATH.read_text(encoding = "utf-8")
TREE = ast.parse(SRC)


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} missing")


def _kwonly_names(fn: ast.FunctionDef) -> set[str]:
    return {a.arg for a in fn.args.kwonlyargs} | {a.arg for a in fn.args.args}


def test_diffusion_server_accepts_the_layer_split():
    fn = _function("_start_diffusion_server")
    assert {"gpu_memory_mode", "gpu_layers"} <= _kwonly_names(fn)


def test_diffusion_server_forwards_ngl_to_the_shim():
    fn = _function("_start_diffusion_server")
    body = ast.get_source_segment(SRC, fn) or ""
    assert '"--ngl"' in body


def test_manual_zero_layers_is_not_swallowed_as_falsy():
    """gpu_layers = 0 (CPU-only) is the exact case the bug report hit; it must stay a real
    request rather than collapsing into the all-layers default."""
    fn = _function("_start_diffusion_server")
    body = ast.get_source_segment(SRC, fn) or ""
    assert "gpu_layers >= 0" in body


def test_diffusion_load_passes_the_users_split_through():
    fn = _function("load_model")
    body = ast.get_source_segment(SRC, fn) or ""
    start = body.index("_start_diffusion_server(")
    call = body[start : body.index(")", body.index("gpu_ids = gpu_ids", start))]
    assert "gpu_memory_mode = gpu_memory_mode" in call
    assert "gpu_layers = gpu_layers" in call


def test_diffusion_no_longer_hardcodes_auto_over_the_users_choice():
    fn = _function("_start_diffusion_server")
    body = ast.get_source_segment(SRC, fn) or ""
    assert 'self._gpu_memory_mode = "auto"' not in body
    assert "self._gpu_layers = -1" not in body
