# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""ACCELERATE_MIXED_PRECISION = 'no' is a value, not an absence.

The GRPO replacements read it as a two-way switch, fp16 else bfloat16, so 'no'
comes out as bfloat16 and autocast is entered anyway. On a T4 or V100 torch does
not merely ignore that, it raises

    RuntimeError: Current CUDA Device does not support bfloat16.
                  Please switch dtype to float16.

Two callers set 'no' and both land on exactly those GPUs: full finetuning
already did, and rl.py now does for a model explicitly loaded in float32, so the
branch meant to keep training in float32 could instead stop it. The fix is
`enabled`, not a different dtype: torch only validates bfloat16 when autocast is
on, and turning it off is what 'no' means.
"""

import ast
import re
import sys
import textwrap
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RL_REPLACEMENTS = REPO_ROOT / "unsloth" / "models" / "rl_replacements.py"
SRC = RL_REPLACEMENTS.read_text(encoding = "utf-8")


class _pretend_cuda:
    """torch.cuda answering as a card without bfloat16, or with it."""

    def __init__(self, has_bf16):
        self.has_bf16 = has_bf16

    def __enter__(self):
        self._saved = (torch.cuda.is_available, torch.cuda.is_bf16_supported)
        torch.cuda.is_available = lambda *args, **kwargs: True
        torch.cuda.is_bf16_supported = lambda *args, **kwargs: self.has_bf16
        return self

    def __exit__(self, *exc):
        torch.cuda.is_available, torch.cuda.is_bf16_supported = self._saved


def test_bfloat16_autocast_raises_without_hardware_support():
    """Guards everything below: if torch ever downgraded this to a warning,
    the bug would be a silent precision change instead of a crash, and these
    tests would be asserting the wrong thing."""
    with _pretend_cuda(has_bf16 = False):
        with pytest.raises(RuntimeError, match = "does not support bfloat16"):
            with torch.amp.autocast(device_type = "cuda", dtype = torch.bfloat16):
                pass


def test_disabling_autocast_skips_that_check():
    with _pretend_cuda(has_bf16 = False):
        with torch.amp.autocast(device_type = "cuda", dtype = torch.bfloat16, enabled = False):
            pass


def _prepare_inputs_snippet() -> str:
    """The `with` header grpo_trainer__prepare_inputs splices into TRL."""
    start = SRC.index('"with torch.inference_mode(), "')
    end = SRC.index('",\n', start)
    return ast.literal_eval("(" + SRC[start : end + 1] + ")")


def _autocast_helper_source() -> str:
    """Module level helpers the header calls, mirrored into the generated
    trainer through RL_PRE_ITEMS."""
    parts = [
        ast.get_source_segment(SRC, node)
        for node in ast.parse(SRC).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_unsloth_grpo_autocast")
    ]
    assert parts, "expected the shared autocast helper"
    return "\n\n".join(parts)


def _namespace(env):
    """What the generated trainer module offers the header. `self` carries no
    args here, which is the fallback path; a trainer with args is covered in
    test_grpo_autocast_per_trainer.py."""
    from contextlib import nullcontext

    namespace = {
        "torch": torch,
        "os": type(sys)("os"),
        "nullcontext": nullcontext,
        "self": type("Trainer", (), {})(),
        "seen": [],
    }
    namespace["os"].environ = env
    exec(_autocast_helper_source(), namespace)
    return namespace


def test_the_injected_snippet_is_valid_python():
    ast.parse(_prepare_inputs_snippet() + "\n    pass\n")


@pytest.mark.parametrize(
    "precision,has_bf16,expect_enabled",
    [
        # The T4/V100 case, where the bug bites. accelerate never asks for bf16
        ("no", False, False),
        ("fp16", False, True),
        (None, False, True),
        ("no", True, False),
        ("bf16", True, True),
    ],
)
def test_the_injected_snippet_only_autocasts_when_asked(precision, has_bf16, expect_enabled):
    """Run the real header and check both that it survives and that it did not
    quietly stop autocasting for everyone else."""
    env = {"UNSLOTH_FORCE_FLOAT32": "0"}
    if precision is not None:
        env["ACCELERATE_MIXED_PRECISION"] = precision

    namespace = _namespace(env)
    with _pretend_cuda(has_bf16 = has_bf16):
        exec(
            _prepare_inputs_snippet() + "\n    seen.append(torch.is_autocast_enabled('cuda'))\n",
            namespace,
        )
    assert namespace["seen"] == [expect_enabled]


# _get_per_token_logps and friends, which run as ordinary code --------
def test_every_autocast_call_passes_enabled():
    """Five call sites share one `self._autocast_dtype`; one left behind would
    still raise, and only on the hardware nobody develops on."""
    calls = re.findall(r"torch\.amp\.autocast\((?:[^()]|\([^()]*\))*\)", SRC)
    using = [c for c in calls if "self._autocast_dtype" in c]
    assert using, "expected the shared autocast dtype to be used"
    missing = [c for c in using if "_autocast_enabled" not in c]
    assert missing == [], missing


def test_the_flag_is_recorded_beside_the_dtype():
    """Every `_autocast_dtype` assignment must record the flag beside it. Both
    live in the one helper now, the default and the forced float32 override.

    Paired by position rather than by counting a literal spelling: the repo's
    ruff hook is free to collapse either assignment onto one line, and a test
    that pins `= (` would fail on formatting alone while a genuinely missing
    initialiser slipped through.
    """
    lines = SRC.splitlines()
    dtype_at = [i for i, l in enumerate(lines) if "self._autocast_dtype = " in l]
    flag_at = [i for i, l in enumerate(lines) if "self._autocast_enabled = " in l]
    assert len(dtype_at) == 2, dtype_at
    for i in dtype_at:
        assert any(0 < j - i <= 8 for j in flag_at), (i, lines[i].strip())


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


def test_forced_float32_still_autocasts_in_the_injected_header():
    """UNSLOTH_FORCE_FLOAT32 sets 'no' as well, but wants fp16 autocast, so the
    shared helper overrides both the dtype and the flag for it. Reading 'no'
    alone here would have left Gemma3 and gpt-oss generation in full float32."""
    namespace = _namespace(
        {
            "ACCELERATE_MIXED_PRECISION": "no",
            "UNSLOTH_FORCE_FLOAT32": "1",
        }
    )
    with _pretend_cuda(has_bf16 = False):
        exec(
            _prepare_inputs_snippet()
            + "\n    seen.append(torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else None)\n",
            namespace,
        )
    assert namespace["seen"] == [torch.float16]


def test_chunk_sizing_reads_the_model_dtype_when_autocast_is_off():
    """With autocast off the forward runs in the model's own dtype. That is
    float32 for an explicit float32 load but bfloat16 for pure bfloat16 full
    finetuning, so the flag alone would double the estimate for the latter."""
    line = next(l for l in SRC.splitlines() if "forward_dtype = (" in l)
    block = SRC[SRC.index(line) : SRC.index(line) + 400]
    assert "_autocast_enabled" in block, block
    assert "lm_head.dtype" in block, block
    assert "dtype_bytes = 16 if forward_dtype in" in block, block


def test_chunk_sizing_by_execution():
    """Both cases, evaluated rather than read: pure bfloat16 full finetuning
    keeps 16, an explicit float32 load gets 32."""
    for head_dtype, expected in ((torch.bfloat16, 16), (torch.float32, 32)):
        scope = {
            "torch": torch,
            "self": type(
                "T",
                (),
                {
                    "_autocast_dtype": torch.bfloat16,
                    "_autocast_enabled": False,
                },
            )(),
            "lm_head": torch.zeros(2, 2, dtype = head_dtype),
        }
        line = next(l for l in SRC.splitlines() if "forward_dtype = (" in l)
        start = SRC.index(line)
        end = SRC.index("dtype_bytes = 16 if forward_dtype in", start)
        end = SRC.index("\n", end)
        exec(textwrap.dedent(SRC[start:end]), scope)
        assert scope["dtype_bytes"] == expected, (head_dtype, scope["dtype_bytes"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
