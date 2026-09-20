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
    assert "--no-train-on-completions" not in leg.args
    # The export is OFF on this leg now: 13153.7 MB of MXFP4 in 348.1s, the
    # most expensive conversion in the suite and the least representative of
    # the claim, which `default` makes on a 609.8 MB file in 40.6s. The payload
    # keeps the capability and the mxfp4 accept rule, so a dispatch can turn it
    # back on; what it does not do is pay for it on every PR.
    assert "--export-gguf" not in leg.args
    assert "gguf_export.py" in leg.files, (
        "the payload still imports it lazily behind --export-gguf, so it stays "
        "declared: a dispatch that turns the export back on must not fail on a "
        "missing file"
    )


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
    assert (
        'default = "mxfp4"' not in src
    ), "mxfp4 is not an accepted request value; unsloth rejects it before the conversion starts"


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


def test_the_off_gpu_walk_names_the_tensors_and_not_only_the_bytes():
    """Driven through the REAL `placement()`, against a stub model.

    `{'cpu': 579133440}` reached hardware twice and nobody could say which
    tensor it was, because the walk used `model.parameters()` and threw the
    names away. 579133440 is exactly 201088 x 2880 -- gpt-oss-20b's vocab by
    its hidden size -- so it was one embedding-shaped tensor all along, and
    that is a filable bug report where a byte count is not.

    A rule fed a hand-written dict would pass either way; this executes the
    producing code, which is the gap that let a missing `torch` import reach
    Kaggle on the Default leg.
    """
    import torch  # noqa: PLC0415

    from run_gptoss_t4 import _placement_failures, placement  # noqa: PLC0415

    class _Stub:
        def __init__(self):
            self._params = [
                ("model.embed_tokens.weight", torch.zeros(4, 3)),
                ("model.layers.0.mlp.weight", torch.zeros(2, 2)),
            ]

        def named_parameters(self):
            return iter(self._params)

    stub = _Stub()
    record = placement(stub)
    assert record["parameters_by_device"] == {"cpu": 16}
    assert record["off_gpu_parameter_count"] == 2
    # Largest first, so the reader sees the tensor that matters.
    assert [p["name"] for p in record["off_gpu_parameters"]] == [
        "model.embed_tokens.weight",
        "model.layers.0.mlp.weight",
    ]

    failures = _placement_failures(record)
    assert failures, "a wholly-CPU model must fail"
    assert "model.embed_tokens.weight" in failures[0], (
        "the failure message carries only byte counts again, which is the "
        "unactionable red this test exists to prevent"
    )


def test_a_walk_that_recorded_no_names_still_fails_and_says_so():
    """The refusal branch. An empty name list must not read as "no problem" --
    the byte counts already said there is one."""
    from run_gptoss_t4 import _placement_failures  # noqa: PLC0415

    failures = _placement_failures(
        {
            "parameters_by_device": {"cpu": 579133440, "cuda:0": 10461969984},
            "off_gpu_parameters": [],
            "offloaded": False,
        }
    )
    assert len(failures) == 1
    assert "the walk recorded no names" in failures[0]


def _placement_record(**over):
    """The shape a healthy gpt-oss run produces, measured on
    unsloth-probe-gptoss-names2-ae1968."""
    record = {
        "parameters_by_device": {"cpu": 579133440, "cuda:0": 10461969984},
        "off_gpu_parameters": [
            {"name": "model.embed_tokens.weight", "numel": 579133440, "device": "cpu"}
        ],
        "off_gpu_parameter_count": 1,
        "input_embedding": {
            "module": "model.embed_tokens",
            "weight_name": "model.embed_tokens.weight",
            "device": "cpu",
            "offload_hooks_installed": True,
        },
        "offloaded": False,
    }
    record.update(over)
    return record


def test_the_deliberate_embedding_offload_is_not_a_failure():
    """Measured, and the assertion was wrong rather than the stack.

    `Unsloth: Offloading embeddings to RAM to save 1.08 GB` is a documented
    optimisation: the input embedding moves to RAM and
    `_install_offload_embedding_hooks` carries ids down and vectors back up. It
    failed this leg twice while training converged, inference was coherent and
    the adapter moved, because the rule read a device count and the count
    cannot tell an optimisation from a spill.
    """
    from run_gptoss_t4 import _placement_failures  # noqa: PLC0415
    assert _placement_failures(_placement_record()) == []


def test_the_excuse_is_the_hook_flag_and_not_the_device():
    """The half that keeps it from being an excuse that can only excuse. An
    embedding on the CPU WITHOUT the hooks is a genuine bug -- the lookup either
    raises or silently synchronises -- and it is indistinguishable from the
    healthy case in `parameters_by_device`."""
    from run_gptoss_t4 import _placement_failures  # noqa: PLC0415

    embed = dict(_placement_record()["input_embedding"], offload_hooks_installed = False)
    failures = _placement_failures(_placement_record(input_embedding = embed))
    assert failures and "model.embed_tokens.weight" in failures[0]


def test_a_second_tensor_off_the_card_is_still_a_failure():
    """The excuse covers exactly one parameter. A real spill that happens to
    include the embedding must not ride in on its coat-tails."""
    from run_gptoss_t4 import _placement_failures  # noqa: PLC0415

    record = _placement_record(
        off_gpu_parameters = [
            {"name": "model.embed_tokens.weight", "numel": 579133440, "device": "cpu"},
            {"name": "model.layers.7.mlp.down_proj.weight", "numel": 8294400, "device": "cpu"},
        ]
    )
    failures = _placement_failures(record)
    assert failures
    assert "model.layers.7.mlp.down_proj.weight" in failures[0]
    # The bracketed list is the "what is wrong" half; the embedding is still
    # printed after it as context, which is what makes the verdict readable.
    listed = failures[0].split("[", 1)[1].split("]", 1)[0]
    assert (
        "model.embed_tokens.weight" not in listed
    ), "the list must name what is unexplained, not re-report the tensor that is accounted for"


def test_the_hook_flag_is_READ_off_the_module_rather_than_assumed():
    """Mutation found this one: hardcoding `offload_hooks_installed = True` in
    `placement()` satisfied every rule above, because they all judge the record
    and none of them produce it. The flag is the entire difference between an
    optimisation and a bug, so it has to come off the module."""
    import torch  # noqa: PLC0415

    from run_gptoss_t4 import placement  # noqa: PLC0415

    class _Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4, 3))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = _Embed()

        def get_input_embeddings(self):
            return self.embed_tokens

    model = _Model()
    assert placement(model)["input_embedding"]["offload_hooks_installed"] is False

    model.embed_tokens._unsloth_offload_hooks_installed = True
    read_back = placement(model)["input_embedding"]
    assert read_back["offload_hooks_installed"] is True
    assert read_back["weight_name"] == "embed_tokens.weight", (
        "the name has to come from the module walk, or it cannot be matched "
        "against the parameter that is off the card"
    )


def test_the_text_leg_exports_once_per_leg_and_not_once_per_cycle():
    """Measured: the conversion is 310.8s and 312.3s on the two Latest_compile
    cycles and 99.3s and 117.3s on the two vision ones, so the repeat is 47% of
    the longest leg in the suite. The second export is the same base weights
    plus an adapter trained by the same script with the same seed, so it re-runs
    llama.cpp rather than asking a new question; the cycles already prove
    reproducibility on the step tables and the generated text.
    """
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert 'if getattr(args, "export_gguf", False) and run_index > 0:' in src
    assert '"skipped": "exported on cycle 0' in src


def test_skipping_every_cycle_is_still_a_failure():
    """The saving must not be able to become missing coverage. A leg that asked
    for an export and produced no file anywhere has to say so, and a per-cycle
    excuse that fires on cycle 0 too would be silent."""
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert "every cycle skipped the GGUF export" in src
    # The excuse is keyed on a cycle having really exported, not on the flag.
    assert (
        'exported = [run for run in runs if not (run.get("gguf_export") or {}).get("skipped")]'
        in src
    )
    assert "for run in exported:" in src
