# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Completions-only masking and the gpt-oss GGUF override both fail silently, so each is checked."""

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
    # Export is off on this leg for cost; the payload keeps the capability.
    assert "--export-gguf" not in leg.args
    assert "gguf_export.py" in leg.files, (
        "the payload still imports it lazily behind --export-gguf, so it stays "
        "declared: a dispatch that turns the export back on must not fail on a "
        "missing file"
    )


def test_the_payload_requests_q8_and_accepts_only_mxfp4():
    """Only mxfp4 is accepted: asking for it directly is rejected, so the q8_0 override is the one route."""
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
    """The gpt-oss export needs 27.6GB of transient disk, more than the 21GB artifact volume holds."""
    src = (PAYLOAD / "run_gptoss_t4.py").read_text(encoding = "utf-8")
    assert 'tempfile.mkdtemp(prefix = "gptoss_gguf_")' in src
    assert (
        'os.path.join(args.outdir, "gguf")' not in src
    ), "the export is back in the 21GB artifact volume and will fail on space"


def test_the_off_gpu_walk_names_the_tensors_and_not_only_the_bytes():
    """placement() must keep parameter names, so an off-GPU byte count can be traced to a tensor."""
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
    """Embedding offload to RAM is a documented optimisation, not a spill, so it is not a failure."""
    from run_gptoss_t4 import _placement_failures  # noqa: PLC0415
    assert _placement_failures(_placement_record()) == []


def test_the_excuse_is_the_hook_flag_and_not_the_device():
    """The exemption keys on the hook flag, not the device: CPU without hooks is a real bug."""
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
    listed = failures[0].split("[", 1)[1].split("]", 1)[0]
    assert (
        "model.embed_tokens.weight" not in listed
    ), "the list must name what is unexplained, not re-report the tensor that is accounted for"


def test_the_hook_flag_is_READ_off_the_module_rather_than_assumed():
    """The hook flag must be read from the module: hardcoding it True passed every rule on mutation."""
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
    """Exports once per leg: a repeat cycle re-runs llama.cpp and asks no new question."""
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert 'if getattr(args, "export_gguf", False) and run_index > 0:' in src
    assert '"skipped": "exported on cycle 0' in src


def test_skipping_every_cycle_is_still_a_failure():
    """The saving must not be able to become missing coverage. A leg that asked
    for an export and produced no file anywhere has to say so, and a per-cycle
    excuse that fires on cycle 0 too would be silent."""
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert "every cycle skipped the GGUF export" in src
    assert (
        'exported = [run for run in runs if not (run.get("gguf_export") or {}).get("skipped")]'
        in src
    )
    assert "for run in exported:" in src
