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
"""ACCELERATE_MIXED_PRECISION = 'no' is a value, not an absence.

The GRPO replacements read it as a two-way switch:

    torch.float16 if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
    else torch.bfloat16

so 'no' comes out as bfloat16 and autocast is entered anyway. On a T4 or V100
torch does not merely ignore that, it raises

    RuntimeError: Current CUDA Device does not support bfloat16.
                  Please switch dtype to float16.

Two callers set 'no', and both land on exactly those GPUs: full finetuning
already did, and rl.py now does for a model the user explicitly loaded in
float32. So the branch meant to keep training in float32 could instead stop it.

The fix is `enabled`, not a different dtype: torch only validates bfloat16
when autocast is on, and turning it off is what 'no' means.
"""

import ast
import re
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RL_REPLACEMENTS = REPO_ROOT / "unsloth" / "models" / "rl_replacements.py"
SRC = RL_REPLACEMENTS.read_text(encoding = "utf-8")


# ---- the premise ---------------------------------------------------------

def test_bfloat16_autocast_raises_without_hardware_support():
    """Guards everything below: if torch ever downgraded this to a warning,
    the bug would be a silent precision change instead of a crash, and these
    tests would be asserting the wrong thing."""
    real = torch.cuda.is_bf16_supported
    torch.cuda.is_bf16_supported = lambda *args, **kwargs: False
    try:
        with pytest.raises(RuntimeError, match = "does not support bfloat16"):
            with torch.amp.autocast(device_type = "cuda", dtype = torch.bfloat16):
                pass
    finally:
        torch.cuda.is_bf16_supported = real


def test_disabling_autocast_skips_that_check():
    real = torch.cuda.is_bf16_supported
    torch.cuda.is_bf16_supported = lambda *args, **kwargs: False
    try:
        with torch.amp.autocast(
            device_type = "cuda", dtype = torch.bfloat16, enabled = False
        ):
            pass
    finally:
        torch.cuda.is_bf16_supported = real


# ---- _prepare_inputs, which is injected as source ------------------------

def _prepare_inputs_snippet() -> str:
    """The `with` header grpo_trainer__prepare_inputs splices into TRL."""
    start = SRC.index('"with torch.inference_mode(), "')
    end = SRC.index('",\n', start)
    return ast.literal_eval("(" + SRC[start:end + 1] + ")")


def test_the_injected_snippet_is_valid_python():
    ast.parse(_prepare_inputs_snippet() + "\n    pass\n")


@pytest.mark.parametrize(
    "precision,has_bf16,expect_enabled",
    [
        # The T4/V100 case, where the bug bites. accelerate never asks for
        # bf16 on this hardware, so that pairing is not a case.
        ("no", False, False),
        ("fp16", False, True),
        (None, False, True),
        ("no", True, False),
        ("bf16", True, True),
    ],
)
def test_the_injected_snippet_only_autocasts_when_asked(
    precision, has_bf16, expect_enabled
):
    """Run the real header and check both that it survives and that it did not
    quietly stop autocasting for everyone else."""
    from contextlib import nullcontext

    env = {"UNSLOTH_FORCE_FLOAT32": "0"}
    if precision is not None:
        env["ACCELERATE_MIXED_PRECISION"] = precision

    real = torch.cuda.is_bf16_supported
    torch.cuda.is_bf16_supported = lambda *args, **kwargs: has_bf16
    try:
        namespace = {
            "torch": torch, "os": type(sys)("os"), "nullcontext": nullcontext,
            "seen": [],
        }
        namespace["os"].environ = env
        exec(
            _prepare_inputs_snippet() + "\n    seen.append(torch.is_autocast_enabled('cuda'))\n",
            namespace,
        )
    finally:
        torch.cuda.is_bf16_supported = real
    assert namespace["seen"] == [expect_enabled]


# ---- _get_per_token_logps and friends, which run as ordinary code --------

def test_every_autocast_call_passes_enabled():
    """Five call sites share one `self._autocast_dtype`; one left behind would
    still raise, and only on the hardware nobody develops on."""
    calls = re.findall(r"torch\.amp\.autocast\((?:[^()]|\([^()]*\))*\)", SRC)
    using = [c for c in calls if "self._autocast_dtype" in c]
    assert using, "expected the shared autocast dtype to be used"
    missing = [c for c in using if "_autocast_enabled" not in c]
    assert missing == [], missing


def test_the_flag_is_recorded_beside_the_dtype():
    assert SRC.count("self._autocast_enabled = (") == 2, (
        "both _autocast_dtype initialisers must set it")


def test_reading_it_tolerates_an_older_trainer():
    """The dtype is cached under `hasattr`, so an instance built before this
    change can reach a call site with only half the state."""
    for call in re.findall(r"getattr\(self, \"_autocast_enabled\"[^)]*\)", SRC):
        assert call.endswith("True)"), call


def test_force_float32_still_autocasts_in_float16():
    """UNSLOTH_FORCE_FLOAT32 models (Gemma3, gpt_oss) are a different case: they
    keep fp16 autocast, so `no` must not swallow them."""
    for match in re.finditer(r'UNSLOTH_FORCE_FLOAT32", "0"\) == "1":\n(.+?)\n\n', SRC, re.S):
        block = match.group(1)
        if "_autocast_dtype" not in block:
            continue
        assert "self._autocast_enabled = True" in block, block


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
