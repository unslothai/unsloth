# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Completions-only training and the gpt-oss GGUF export, and how each lies.

Both features share a failure shape: they can be requested, silently not
happen, and leave every other number in the report looking correct.

* A run that masks NOTHING trains on prompt and answer alike. It converges, its
  losses are finite, its grad_norm is healthy, and no loss-based assertion can
  tell it from a correct one. So the mask is read off a real collated batch and
  ruled on here.
* An export that is silently overridden to another format still writes a file
  and reports ok. gpt-oss answers `q8_0` with MXFP4 BY DESIGN, so the leg must
  ask for MXFP4 and accept only MXFP4; a wider accept list passes on the
  override as though the request had been honoured.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "t4_smoke"
sys.path.insert(0, str(PAYLOAD))
sys.path.insert(0, str(ROOT / ".github" / "scripts"))

from run_gptoss_t4 import masking_failures  # noqa: E402


def test_a_run_that_masked_nothing_is_a_failure():
    """The finding this file exists for. Every loss-based assertion in the leg
    passes in this state, so only this rule can catch it."""
    broken = masking_failures({"label_tokens": 512, "masked_tokens": 0}, expected = True)
    assert len(broken) == 1 and "NOTHING was masked" in broken[0]


def test_a_run_that_masked_everything_is_a_failure():
    broken = masking_failures({"label_tokens": 512, "masked_tokens": 512}, expected = True)
    assert broken and "no completion left to learn from" in broken[0]


def test_a_partly_masked_batch_passes():
    assert masking_failures({"label_tokens": 512, "masked_tokens": 120}, expected = True) == []


def test_missing_evidence_is_a_failure_not_a_silence():
    assert masking_failures(None, expected = True)
    assert masking_failures({"error": "boom"}, expected = True)
    assert masking_failures({"label_tokens": 0, "masked_tokens": 0}, expected = True)


def test_the_rule_is_inert_when_completions_only_was_not_requested():
    """A leg that did not ask for masking must not go red for not having it."""
    assert masking_failures(None, expected = False) == []
    assert masking_failures({"label_tokens": 512, "masked_tokens": 0}, expected = False) == []


def test_the_leg_asks_for_completions_and_for_mxfp4():
    """Asserted through the REGISTRY, because a payload flag nobody passes is
    coverage that does nothing. The default is on, so the check is that the leg
    does not turn it off."""
    from kaggle_t4_ci.legs import LEGS

    leg = LEGS["gptoss"]
    assert "--export-gguf" in leg.args
    assert "--no-train-on-completions" not in leg.args
    assert "gguf_export.py" in leg.files, "the export imports it lazily, so it must be declared"


def test_the_payload_requests_q8_and_accepts_only_mxfp4():
    """The pairing looks backwards and is the only one that works.

    gpt-oss overrides q8_0 to MXFP4 and says so. Asking for mxfp4 directly is
    the obvious response and unsloth REJECTS it as an input, measured on kernel
    unsloth-probe-gptoss-r3-832c85:

        Unsloth: Quant method = [mxfp4] not supported. Choose from below:
        [not_quantized] [fast_quantized] [quantized] [f32] [bf16] ...

    So the documented override is the only route to an MXFP4 file. Accepting
    ONLY mxfp4 keeps it honest: a run that produced a real q8_0 would fail,
    which is right, because gpt-oss q8_0 is documented impossible.
    """
    src = (PAYLOAD / "run_gptoss_t4.py").read_text(encoding = "utf-8")
    assert '"--gguf-quantization", default = "q8_0"' in src
    assert 'accept_quantizations = ("mxfp4",)' in src
    assert 'default = "mxfp4"' not in src, (
        "mxfp4 is not an accepted request value; unsloth rejects it before the conversion starts"
    )


def test_the_dataset_shape_and_the_text_field_cannot_both_be_set():
    """Naming a text field TRL cannot find is how a prompt-completion dataset
    silently falls back to training on everything."""
    src = (PAYLOAD / "run_gptoss_t4.py").read_text(encoding = "utf-8")
    assert '**({} if args.train_on_completions else {"dataset_text_field": "text"})' in src


def test_the_gptoss_export_does_not_land_in_the_artifact_volume():
    """Measured, not reasoned about.

    `/kaggle/working` is 21.0GB total. The gpt-oss export consumes 27.6GB of
    transient disk (three mxfp4 shards at 13.76GB, plus the GGUF). Exporting
    there fails in 2.8s with "Unsloth: Failed saving locally - no disk space
    left", which reads like an export bug and is a disk fact. Observed on
    kernel unsloth-probe-gptoss-comp-gguf-701d00.

    `/tmp` is the overlay: 8656.9GB total, 1102.5GB free. tempfile honours
    TMPDIR and lands there.
    """
    src = (PAYLOAD / "run_gptoss_t4.py").read_text(encoding = "utf-8")
    assert 'tempfile.mkdtemp(prefix = "gptoss_gguf_")' in src
    assert (
        'os.path.join(args.outdir, "gguf")' not in src
    ), "the export is back in the 21GB artifact volume and will fail on space"
