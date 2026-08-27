# Copyright 2026-present the Unforgettable contributors.
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

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_HEAVY_IMPORT_PREFIXES = (
    "import unsloth",
    "from unsloth",
    "import torch",
    "from torch",
)


def test_package_does_not_import_studio():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding = "utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "import studio" in stripped or stripped.startswith("from studio"):
                offenders.append(f"{path}: {stripped}")
    assert offenders == []


def test_sidecar_has_no_eager_unsloth_or_torch():
    sidecar = ROOT / "sidecar"
    offenders = []
    for path in sidecar.rglob("*.py"):
        text = path.read_text(encoding = "utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            is_heavy = stripped.startswith(_HEAVY_IMPORT_PREFIXES)
            if path.name not in {"train.py", "export_gguf.py"}:
                if is_heavy or "unsloth" in stripped.lower():
                    offenders.append(f"{path}:{lineno}: {stripped}")
                continue
            if is_heavy and line[:1] not in (" ", "\t"):
                offenders.append(f"{path}:{lineno}: {stripped}")
    assert offenders == []


def test_importing_sidecar_does_not_load_unsloth_or_torch():
    import unforgettable.sidecar  # noqa: F401
    assert "unsloth" not in sys.modules
    assert "torch" not in sys.modules


def test_sft_path_does_not_import_dpo():
    text = (ROOT / "sidecar" / "train.py").read_text(encoding = "utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("from trl import") and "SFT" in stripped:
            assert "DPO" not in stripped, f"{lineno}: {stripped}"
        if "import" in stripped and "DPOTrainer" in stripped:
            assert line[:1] in (" ", "\t"), f"module-level DPO import at {lineno}"
