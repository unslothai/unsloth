# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fast-kernel provenance, and the three ways this check goes vacuous.

Every rule here is calibrated against what was MEASURED on a real 2xT4
(`unsloth-probe-vision-recon-c76ea3`), not against what the brief assumed:

* `fla` resolves to `unsloth_zoo/_vendored/fla`, version 0.5.1, and only AFTER
  the model load;
* attention resolves to `sdpa`, and `flash_attn` is not importable at all;
* `causal_conv1d` and `mamba_ssm` are NOT installed on this path, before or
  after the load.

The last one is why two obvious assertions are absent: asserting them present
would be red on correct behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "t4_smoke"
sys.path.insert(0, str(PAYLOAD))

from kernel_provenance import vision_kernel_failures  # noqa: E402

VENDORED = {
    "importable": True,
    "file": "/usr/local/lib/python3.12/dist-packages/unsloth_zoo/_vendored/fla/__init__.py",
    "version": "0.5.1",
    "vendored": True,
}


def test_the_measured_configuration_passes():
    """The exact shape the recon probe recorded. If this ever fails, the rule
    has drifted away from the hardware rather than the hardware from the rule."""
    kernels = {
        "fla": VENDORED,
        "causal_conv1d": {"importable": False, "error": "ModuleNotFoundError"},
        "mamba_ssm": {"importable": False, "error": "ModuleNotFoundError"},
        "flash_attn": {"importable": False, "error": "ModuleNotFoundError"},
    }
    assert vision_kernel_failures(kernels, {"config": "sdpa"}, capability = "7.5") == []


def test_a_pip_installed_fla_is_not_the_vendored_one():
    """ "importable" is the tempting assertion and it is the wrong one: it passes
    on a copy that is not what ships."""
    kernels = {
        "fla": {"importable": True, "file": "/site-packages/fla/__init__.py", "vendored": False}
    }
    broken = vision_kernel_failures(kernels, {"config": "sdpa"}, capability = "7.5")
    assert broken and "not the vendored copy" in broken[0]


def test_fla_missing_is_a_failure():
    kernels = {"fla": {"importable": False, "error": "ModuleNotFoundError: fla"}}
    broken = vision_kernel_failures(kernels, {"config": "sdpa"}, capability = "7.5")
    assert broken and "did not import after the model load" in broken[0]


def test_flash_attention_2_on_turing_is_a_failure():
    """The regression that matters. FA2 cannot execute on sm_75, so a stack that
    selects it fails at the first forward, and a leg that did not check would
    report that as an unexplained crash."""
    broken = vision_kernel_failures(
        {"fla": VENDORED}, {"config": "flash_attention_2"}, capability = "7.5"
    )
    assert broken and "cannot run it" in broken[0]


def test_flash_attention_2_is_not_flagged_off_turing():
    """The same choice is correct on Ampere. A rule that fired everywhere would
    be wrong rather than strict."""
    assert (
        vision_kernel_failures({"fla": VENDORED}, {"config": "flash_attention_2"}, capability = "8.6")
        == []
    )


def test_a_missing_attention_record_is_a_failure_not_a_silence():
    broken = vision_kernel_failures({"fla": VENDORED}, {}, capability = "7.5")
    assert broken and "no attention implementation was recorded" in broken[0]


def test_causal_conv1d_absence_is_reported_and_not_asserted():
    """Measured absent on this path. The wheel-first machinery in ssm_runtime.py
    belongs to Studio's training worker, which the notebook path never calls, so
    an assertion here would go red on correct behaviour."""
    kernels = {
        "fla": VENDORED,
        "causal_conv1d": {"importable": False, "error": "ModuleNotFoundError"},
        "mamba_ssm": {"importable": False, "error": "ModuleNotFoundError"},
    }
    assert vision_kernel_failures(kernels, {"config": "sdpa"}, capability = "7.5") == []


def test_no_provenance_at_all_is_a_failure():
    assert vision_kernel_failures(None, {"config": "sdpa"}, capability = "7.5")


def test_both_capability_spellings_reach_the_turing_rule():
    """The bug this catches was mine, and it was live for a few minutes.

    `environment_fingerprint()` records `"sm_75"`, while the recon probe and
    `torch.cuda.get_device_capability` give `"7.5"`. A `startswith("7.")` check
    against `"sm_75"` matches nothing, so the FA2 rule would never fire and the
    leg would report a clean pass while checking nothing at all.
    """
    for spelling in ("7.5", "sm_75", "75"):
        broken = vision_kernel_failures(
            {"fla": VENDORED}, {"config": "flash_attention_2"}, capability = spelling
        )
        assert broken, f"the Turing rule never fired for capability={spelling!r}"

    for spelling in ("8.6", "sm_86", "86", ""):
        assert (
            vision_kernel_failures(
                {"fla": VENDORED}, {"config": "flash_attention_2"}, capability = spelling
            )
            == []
        ), f"the Turing rule fired for capability={spelling!r}"


def test_the_payload_passes_the_capability_the_fingerprint_records():
    """Asserted from the source: the wiring is where the two spellings meet."""
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert 'capability = str(env.get("gpu_capability"' in src


def test_the_payload_never_calls_a_tokenizer_positionally():
    """A vision model's tokenizer IS a processor.

    `ProcessorMixin.__call__` is `(self, images=None, text=None, videos=None,
    ...)`, so a positional list of prompts is taken as IMAGES and transformers
    tries to fetch each string as an image URL. That is not hypothetical: it
    killed the Latest_compile leg on gemma-4-E2B-it after the model had already
    loaded and trained, on kernel unsloth-probe-latestcompile-r2-62b54d.

    `text` is also the first parameter of a plain tokenizer, so the keyword is
    correct everywhere and this is not a vision special case.
    """
    import re

    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    offenders = []
    for match in re.finditer(r"tokenizer\(\s*(?!text\s*=)(?!\))([^)\n]*)", src):
        arg = match.group(1).strip()
        # Keyword-only calls are fine; a bare `tokenizer(` opening a keyword
        # list is what the negative lookahead already allowed through.
        if arg and not arg.split(",")[0].strip().endswith("=") and "=" not in arg.split(",")[0]:
            offenders.append(match.group(0))
    assert offenders == [], f"positional tokenizer calls: {offenders}"


def test_prompt_token_lengths_index_past_the_batch_dimension():
    """The missing `[0]` made the whole padding check vacuous.

    A processor returns `input_ids` with a BATCH dimension, so `len(...)` on it
    is the number of sequences -- 1 -- for every prompt. Measured on kernel
    unsloth-probe-latestcompile-r3-cb1125, where gemma-4 reported
    `[1, 1, 1, 1, 1, 1, 1, 1]` and the run's own vacuity guard caught it:

        every batched prompt tokenised to the same length, so nothing was ever
        padded and the left-padding check proved nothing

    A plain tokenizer given one string returns a flat list, which is why this
    read correctly on every text model and broke on the first vision one.
    """
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert 'len(tokenizer(text = [p])["input_ids"][0]) for p in prompts' in src
    assert 'len(tokenizer(text = p)["input_ids"]) for p in prompts' not in src
