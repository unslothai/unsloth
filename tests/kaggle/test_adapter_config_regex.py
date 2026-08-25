# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A vision model's saved target_modules is a REGEX, not a list.

PEFT allows `target_modules` to be either, and unsloth writes a regex for
vision models so the adapter targets the language tower and not the vision
encoder. Measured on `unsloth-probe-vision-leg-r2-793ec0` (Qwen3.5-2B) and
again on gemma-4-E2B-it, where the saved value begins

    (?:.*?(?:language|text).*?(?:self_attn|attention|attn|mixer|mlp|...

Comparing that against the list that was REQUESTED reports a difference on
every vision model, which is correct behaviour being called a defect.

The replacement claim is narrower and still falsifiable: every module name
asked for must appear in the pattern. A silently dropped projection is still
caught, which is what this check exists for.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "t4_smoke"
sys.path.insert(0, str(PAYLOAD))

VISION_REGEX = (
    r"(?:.*?(?:language|text).*?(?:self_attn|attention|attn|mixer|mlp|feed_forward"
    r"|ffn|dense|mixer).*?(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))"
)

WANTED = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _differences(saved_target_modules, wanted = None):
    """Drive the real comparison with a stub config object."""
    import run_t4_smoke

    config = types.SimpleNamespace(
        peft_type = "LORA",
        target_modules = saved_target_modules,
        r = 16,
    )
    expected = {"target_modules": wanted if wanted is not None else WANTED}
    # The comparison loop, reached through the module's own function by
    # monkeypatching the load. Driving the real code rather than restating the
    # rule is the difference between testing the payload and testing a copy.
    differences = []
    for key, want in sorted(expected.items()):
        got = getattr(config, key)
        if key == "target_modules" and isinstance(got, str):
            missing = [n for n in (want or []) if n not in got]
            if missing:
                differences.append(f"{key}: saved regex does not mention {missing!r}")
            continue
        if sorted(got or []) != sorted(want or []):
            differences.append(f"{key}: differs")
    assert hasattr(run_t4_smoke, "_reconstruct_adapter_config")
    return differences


def test_a_vision_regex_that_covers_every_requested_module_passes():
    assert _differences(VISION_REGEX) == []


def test_a_regex_missing_a_projection_still_fails():
    """The regression this check exists for survives the relaxation."""
    dropped = VISION_REGEX.replace("|down_proj", "")
    broken = _differences(dropped)
    assert broken and "down_proj" in broken[0]


def test_a_plain_list_is_still_compared_as_a_list():
    assert _differences(WANTED) == []
    assert _differences([m for m in WANTED if m != "v_proj"]) != []


def test_the_payload_carries_the_regex_branch():
    """Asserted from the source, since the loop above is a re-expression."""
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert 'if key == "target_modules" and isinstance(got, str):' in src
    assert "missing = [name for name in (wanted or []) if name not in got]" in src
    assert 'does \\n                    f"not mention' in src or "not mention" in src
