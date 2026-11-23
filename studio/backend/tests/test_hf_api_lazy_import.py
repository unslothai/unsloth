# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The account and token-validation modules import HfApi (huggingface_hub.hf_api) on first use, not at boot."""

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def test_boot_modules_do_not_import_hf_api():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(BACKEND)!r})\n"
        "import hub.services.models.account_access as access, utils.hf_token_validation\n"
        "assert 'huggingface_hub.hf_api' not in sys.modules\n"
        "import huggingface_hub\n"
        "huggingface_hub.HfApi = object\n"
        "assert access.HfApi.__module__ == 'huggingface_hub.hf_api'\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 300)
    assert out.returncode == 0, out.stderr[-3000:]
