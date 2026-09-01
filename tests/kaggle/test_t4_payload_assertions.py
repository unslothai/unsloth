# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU-only tests for what the T4 payloads actually ASSERT.

`test_t4_smoke_harness.py` covers the launcher, the gate and the shape of the
generated notebook; this covers the other half: given a result dict, does the
payload call it a pass or a failure? Every case here is one where a run that
measured nothing, or the wrong thing, used to report green.

Nothing here needs a GPU, which is the point: the pass/fail rule for a leg that
costs a Kaggle session has to be checkable without one.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = REPO_ROOT / "tests" / "kaggle" / "t4_smoke"
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"

sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(CI_DIR))

NAN = float("nan")
INF = float("inf")




def test_a_field_logged_by_only_one_run_is_a_difference():
    """grad_norm in cycle 0 and not in cycle 1 is nondeterminism, not a skip.

    ``check_reference`` already calls one-sided presence "a change in the SHAPE
    of what the trainer logged", and the exact comparator has to agree or the
    strong assertion is weaker than the tolerance band beside it.
    """
    from determinism import compare_metrics

    a = [{"step": 1, "loss": 1.0, "grad_norm": 2.0}]
    b = [{"step": 1, "loss": 1.0}]
    result = compare_metrics(a, b)
    assert result["identical"] is False
    assert result["first_diff_step"] == 1


def test_a_field_absent_from_both_runs_is_still_not_a_difference():
    from determinism import compare_metrics

    a = [{"step": 1, "loss": 1.0}]
    b = [{"step": 1, "loss": 1.0}]
    assert compare_metrics(a, b)["identical"] is True


def test_repeated_runs_must_agree_on_their_step_coordinates():
    """Same values in the same positions, against different step numbers."""
    from determinism import compare_metrics

    a = [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.5}, {"step": 3, "loss": 0.25}]
    b = [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.5}, {"step": 2, "loss": 0.25}]
    result = compare_metrics(a, b)
    assert result["identical"] is False
    assert result["step_mismatch"] == [{"index": 2, "a": 3, "b": 2}]


def test_matching_step_coordinates_are_not_reported_as_a_difference():
    from determinism import compare_metrics

    a = [{"step": 1, "loss": 1.0, "grad_norm": NAN}]
    b = [{"step": 1, "loss": 1.0, "grad_norm": NAN}]
    result = compare_metrics(a, b)
    assert result["identical"] is True
    assert result["step_mismatch"] == []


def test_two_runs_that_both_overflowed_on_the_same_step_agree():
    """An fp16 overflow logs infinity as readily as NaN, and abs(inf-inf) is NaN.

    NaN != 0.0, so the subtraction reported two identical traces as
    nondeterministic, with a max_abs_diff of 0.0 to show for it.
    """
    from determinism import compare_metrics

    a = [{"step": 1, "loss": 10.0, "grad_norm": INF}, {"step": 2, "loss": 9.0, "grad_norm": 4.0}]
    b = [{"step": 1, "loss": 10.0, "grad_norm": INF}, {"step": 2, "loss": 9.0, "grad_norm": 4.0}]
    result = compare_metrics(a, b)
    assert result["identical"] is True
    assert result["first_diff_step"] is None
    assert result["max_abs_diff"] == {"loss": 0.0, "grad_norm": 0.0}


@pytest.mark.parametrize(
    ("norm_a", "norm_b"),
    [(INF, 4.0), (4.0, INF), (INF, -INF), (-INF, INF)],
)
def test_an_infinity_only_one_run_logged_is_still_a_difference(norm_a, norm_b):
    from determinism import compare_metrics

    a = [{"step": 1, "loss": 1.0, "grad_norm": norm_a}]
    b = [{"step": 1, "loss": 1.0, "grad_norm": norm_b}]
    assert compare_metrics(a, b)["identical"] is False




def test_the_canary_must_be_the_whole_answer():
    from run_t4_smoke import CANARY, canary_failures

    run = {"run_index": 0, "generated": CANARY + "<|im_start|>and on it went"}
    failures = canary_failures(run, require = True)
    assert failures and "exactly" in failures[0]


def test_the_canary_tolerates_surrounding_whitespace_only():
    from run_t4_smoke import CANARY, canary_failures
    assert canary_failures({"run_index": 0, "generated": CANARY}, require = True) == []
    assert canary_failures({"run_index": 0, "generated": "\n" + CANARY + " \n"}, require = True) == []


def test_a_missing_canary_is_still_a_failure():
    from run_t4_smoke import canary_failures
    failures = canary_failures({"run_index": 1, "generated": "def my_function():"}, require = True)
    assert failures and "did not emit" in failures[0]


def test_the_canary_can_be_downgraded_to_a_warning():
    from run_t4_smoke import canary_failures
    assert canary_failures({"run_index": 1, "generated": "nope"}, require = False) == []




def test_an_infinite_gradient_norm_is_not_an_applied_update():
    """fp16 overflow reports the norm as inf as readily as NaN.

    ``inf == inf``, so the NaN-only test counted every skipped step as applied
    and a run that trained nothing reported green.
    """
    from run_t4_smoke import optimisation_failures

    metrics = [{"step": s, "loss": 10.0 - s, "grad_norm": INF} for s in (1, 2, 3)]
    failures = optimisation_failures(metrics)
    assert failures and "no optimizer update was applied" in failures[0]


def test_one_finite_gradient_norm_is_enough():
    from run_t4_smoke import optimisation_failures
    metrics = [
        {"step": 1, "loss": 10.0, "grad_norm": NAN},
        {"step": 2, "loss": 5.0, "grad_norm": INF},
        {"step": 3, "loss": 1.0, "grad_norm": 11.2},
    ]
    assert optimisation_failures(metrics) == []




def _adapter_state(**over) -> dict:
    state = {
        "dir": "/kaggle/working/smoke0/lora_run0",
        "files": ["adapter_config.json", "adapter_model.safetensors"],
        "weight_file": "adapter_model.safetensors",
        "config_readable": True,
        "tensors": 224,
        "parameters": 4325376,
        "non_finite_tensors": [],
        "nonzero_tensors": 224,
        "b_tensors": 112,
        "nonzero_b_tensors": 112,
        "keys_checked": True,
        "keys_missing": [],
        "keys_unexpected": [],
        "keys_extra": [],
    }
    state.update(over)
    return state


def test_a_saved_adapter_that_reloads_with_weights_passes():
    from run_t4_smoke import saved_adapter_failures
    assert saved_adapter_failures(_adapter_state()) == []


def test_an_adapter_whose_weight_file_cannot_be_read_is_a_failure():
    from run_t4_smoke import saved_adapter_failures
    failures = saved_adapter_failures(
        _adapter_state(tensors = None, error = "SafetensorError: header too small")
    )
    assert failures and "could not be read back" in failures[0]


def test_an_adapter_with_no_tensors_is_a_failure():
    from run_t4_smoke import saved_adapter_failures
    failures = saved_adapter_failures(_adapter_state(tensors = 0, parameters = 0))
    assert failures and "no tensors" in failures[0]


def test_an_adapter_with_non_finite_weights_is_a_failure():
    from run_t4_smoke import saved_adapter_failures
    failures = saved_adapter_failures(_adapter_state(non_finite_tensors = ["...lora_B.weight"]))
    assert failures and "non-finite" in failures[0]


def test_an_all_zero_adapter_is_a_failure():
    """lora_B starts at zero, so an all-zero file is an untrained adapter."""
    from run_t4_smoke import saved_adapter_failures

    failures = saved_adapter_failures(_adapter_state(nonzero_tensors = 0, nonzero_b_tensors = 0))
    assert failures and "saved lora_B matrices is zero" in failures[0]


def test_an_adapter_whose_b_matrices_are_all_zero_is_a_failure():
    """The A matrices alone keep `nonzero_tensors` up, and prove nothing.

    peft initialises lora_A randomly, so it is nonzero before a single step, and
    counting every tensor therefore accepted an adapter whose B matrices were
    all zero, contributing nothing since the update goes through B.
    """
    from run_t4_smoke import saved_adapter_failures

    failures = saved_adapter_failures(
        _adapter_state(nonzero_tensors = 112, b_tensors = 112, nonzero_b_tensors = 0)
    )
    assert failures and "saved lora_B matrices is zero" in failures[0]


def test_an_adapter_with_no_b_matrices_at_all_is_unusable_rather_than_fine():
    """Dropped B matrices, or a peft naming change: either way, no reading."""
    from run_t4_smoke import saved_adapter_failures

    failures = saved_adapter_failures(
        _adapter_state(nonzero_tensors = 112, b_tensors = 0, nonzero_b_tensors = 0)
    )
    assert failures and "is a lora_B matrix" in failures[0]


def test_an_adapter_nobody_checked_the_names_of_is_not_a_pass():
    """No oracle, no verdict.

    The tensor reading cannot see whether PEFT would consume these weights, so
    a run that could not derive the expected names learned nothing about the
    save. Recording that as an unchecked field would leave the strongest thing
    this function asserts silently switched off.
    """
    from run_t4_smoke import saved_adapter_failures

    failures = saved_adapter_failures(
        _adapter_state(keys_checked = False, keys_error = "ImportError: no such name")
    )
    assert failures and "never checked" in failures[0]


def test_an_adapter_missing_tensors_peft_names_is_a_failure():
    from run_t4_smoke import saved_adapter_failures
    failures = saved_adapter_failures(
        _adapter_state(keys_missing = ["base_model.model.q_proj.lora_B.weight"])
    )
    assert failures and "does not carry" in failures[0]


def test_lora_tensors_under_names_peft_does_not_use_are_a_failure():
    from run_t4_smoke import saved_adapter_failures
    failures = saved_adapter_failures(_adapter_state(keys_unexpected = ["q_proj.lora_B.weight"]))
    assert failures and "ignores them silently" in failures[0]


def test_a_non_lora_tensor_beside_the_adapter_is_not_a_failure(tmp_path):
    """The check is about names PEFT has to MATCH, not about extra tensors.

    ``save_pretrained`` can legitimately write more than the adapter -- an
    embedding copy when the vocabulary was resized, a modules_to_save entry --
    and none of that is a LoRA tensor PEFT's loader has to recognise. Failing on
    it turns a supported save into a red leg, which is why the two are sorted
    apart rather than failed together as "anything the oracle did not name".

    Through the real reading rather than a hand-built state, or the sorting this
    asserts is not the code that runs.
    """
    pytest.importorskip("safetensors")
    import torch
    from safetensors.torch import save_file

    from run_t4_smoke import saved_adapter_failures, verify_saved_adapter

    lora = "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
    embedding = "base_model.model.model.embed_tokens.weight"
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16}))
    save_file(
        {lora: torch.ones(8, 16) * 0.01, embedding: torch.ones(4, 4)},
        str(tmp_path / "adapter_model.safetensors"),
    )
    state = verify_saved_adapter(tmp_path, peft_keys = {"keys": [lora]})
    assert state["keys_extra"] == [embedding]
    assert state["keys_unexpected"] == []
    assert saved_adapter_failures(state) == []


def test_the_key_oracle_reports_a_failure_rather_than_raising():
    """It runs inside the payload, so a raise here would lose the whole leg."""
    from run_t4_smoke import peft_adapter_keys

    out = peft_adapter_keys(object())
    assert "keys" not in out and out["error"]


def test_an_adapter_peft_would_ignore_on_reload_does_not_pass(tmp_path):
    """The whole regression, through the real peft, on the CPU.

    A serialization regression that renames the tensors (dropping the
    ``base_model.model.`` prefix is what filtering ``model.state_dict()`` by
    hand instead of calling ``get_peft_model_state_dict`` produces) leaves a
    file that deserializes perfectly and holds the same nonzero lora_B matrices
    a trained adapter does. Every reading this payload took off the bytes is
    identical, PEFT raises nothing on reload, and the adapter contributes
    nothing -- the exact outcome the tensor counts exist to catch.

    The last assertion is the premise rather than the behaviour: if a future
    peft starts refusing unmatched keys, this test says so instead of quietly
    checking nothing.
    """
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import GPT2Config, GPT2LMHeadModel

    from run_t4_smoke import peft_adapter_keys, saved_adapter_failures, verify_saved_adapter

    def _base():
        torch.manual_seed(0)
        return GPT2LMHeadModel(GPT2Config(n_layer = 1, n_head = 2, n_embd = 32, vocab_size = 64))

    model = get_peft_model(
        _base(), LoraConfig(r = 4, target_modules = ["c_attn"], task_type = "CAUSAL_LM")
    )
    # lora_B starts at zero and only an applied optimizer step moves it.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.add_(0.25)
    model.save_pretrained(str(tmp_path))
    keys = peft_adapter_keys(model)
    assert keys["keys"], keys

    good = verify_saved_adapter(tmp_path, peft_keys = keys)
    assert good["keys_checked"] is True
    assert saved_adapter_failures(good) == []

    weights = tmp_path / "adapter_model.safetensors"
    tensors = load_file(str(weights))
    save_file(
        {name.removeprefix("base_model.model."): value for name, value in tensors.items()},
        str(weights),
    )
    broken = verify_saved_adapter(tmp_path, peft_keys = keys)

    assert broken["tensors"] == good["tensors"]
    assert broken["nonzero_b_tensors"] == good["nonzero_b_tensors"] > 0
    assert broken["non_finite_tensors"] == []
    assert broken["config_loadable"] is True

    failures = saved_adapter_failures(broken)
    assert failures, "the leg would have passed an adapter that reloads to the base model"
    assert "ignores them silently" in " ".join(failures)

    reloaded = PeftModel.from_pretrained(_base(), str(tmp_path))
    b_matrices = [p for n, p in reloaded.named_parameters() if "lora_B" in n]
    assert b_matrices and all(float(p.abs().sum()) == 0.0 for p in b_matrices), (
        "peft loaded the renamed keys after all, so this test no longer describes "
        "the regression it was written for"
    )


def test_a_missing_adapter_config_is_a_failure():
    from run_t4_smoke import saved_adapter_failures
    failures = saved_adapter_failures(_adapter_state(config_readable = False))
    assert failures and "adapter_config.json" in failures[0]


def test_the_adapter_check_reads_a_real_file_it_just_wrote(tmp_path):
    """End to end through the real writer, no GPU and no model.

    The first file is the exact artifact the old count accepted: a random
    lora_A beside a lora_B that never left zero, one nonzero tensor of two, and
    an adapter that reloads to the base model.
    """
    pytest.importorskip("safetensors")
    import torch
    from safetensors.torch import save_file

    from run_t4_smoke import saved_adapter_failures, verify_saved_adapter

    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16}))
    written = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(16, 8),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(8, 16),
    }
    peft_keys = {"keys": sorted(written)}
    save_file(written, str(tmp_path / "adapter_model.safetensors"))
    state = verify_saved_adapter(tmp_path, peft_keys = peft_keys)
    assert state["tensors"] == 2
    assert state["nonzero_tensors"] == 1
    assert state["b_tensors"] == 1
    assert state["nonzero_b_tensors"] == 0
    assert state["config_readable"] is True
    failures = saved_adapter_failures(state)
    assert failures and "saved lora_B matrices is zero" in failures[0]

    # The same file with a B matrix an optimizer moved, the only difference between an adapter that carries training
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(16, 8),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(8, 16) * 0.01,
        },
        str(tmp_path / "adapter_model.safetensors"),
    )
    trained = verify_saved_adapter(tmp_path, peft_keys = peft_keys)
    assert trained["nonzero_b_tensors"] == 1
    assert saved_adapter_failures(trained) == []

    save_file({"a.lora_B.weight": torch.zeros(4, 4)}, str(tmp_path / "adapter_model.safetensors"))
    failures = saved_adapter_failures(
        verify_saved_adapter(tmp_path, peft_keys = {"keys": ["a.lora_B.weight"]})
    )
    assert failures and "saved lora_B matrices is zero" in failures[0]


def test_a_syntactically_valid_but_empty_adapter_config_is_not_a_pass(tmp_path):
    """`{}` is valid JSON and PEFT cannot rebuild an adapter from it.

    The check used to be `json.loads` succeeding, which `{}` and `[]` both do,
    so a save that wrote no LoRA fields at all read as "config_readable" and
    the leg passed on a directory nothing can load. PEFT resolves the config
    class from `peft_type` and raises when it is absent, so the question is
    asked of PEFT rather than of a field list this file guessed at.
    """
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")
    import torch
    from safetensors.torch import save_file

    from run_t4_smoke import saved_adapter_failures, verify_saved_adapter

    save_file(
        {"q_proj.lora_B.weight": torch.ones(8, 16) * 0.01},
        str(tmp_path / "adapter_model.safetensors"),
    )
    for body in ("{}", "[]"):
        (tmp_path / "adapter_config.json").write_text(body)
        state = verify_saved_adapter(tmp_path, peft_keys = {"keys": ["q_proj.lora_B.weight"]})
        assert state["config_readable"] is True, "it IS readable JSON; that was never the question"
        assert state["config_loadable"] is False, body
        failures = saved_adapter_failures(state)
        assert failures and "PEFT cannot rebuild an adapter from it" in failures[0], body


def test_an_adapter_config_for_a_different_adapter_than_the_one_trained_fails(tmp_path):
    """A well-formed config that does not describe THIS run.

    Loadable is not enough on its own: a save that dropped `target_modules` or
    wrote a rank the run never used produces a file PEFT rebuilds happily into
    the wrong adapter. The expectation is the argument list the payload handed
    `get_peft_model`, so this compares the save against the request rather than
    against a copy of it.
    """
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")
    import torch
    from safetensors.torch import save_file

    from run_t4_smoke import saved_adapter_failures, verify_saved_adapter

    save_file(
        {"q_proj.lora_B.weight": torch.ones(8, 16) * 0.01},
        str(tmp_path / "adapter_model.safetensors"),
    )
    requested = {"r": 16, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]}
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj"],
            }
        )
    )
    keys = {"keys": ["q_proj.lora_B.weight"]}
    state = verify_saved_adapter(tmp_path, expected = requested, peft_keys = keys)
    assert state["config_loadable"] is True
    assert state["config_differences"], state
    failures = saved_adapter_failures(state)
    assert failures and "different adapter than the one that was trained" in failures[0]

    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", **requested}))
    good = verify_saved_adapter(tmp_path, expected = requested, peft_keys = keys)
    assert good["config_differences"] == []
    assert saved_adapter_failures(good) == []




def _write_reference(
    path: Path,
    *,
    config: dict,
    model: str = "unsloth/Qwen2.5-0.5B-Instruct",
):
    path.write_text(
        json.dumps(
            {
                "config": config,
                "model": model,
                "environment": {},
                "metrics": [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)],
            }
        ),
        encoding = "utf-8",
    )


def test_a_band_check_against_a_reference_from_another_card_is_refused(tmp_path):
    """The committed reference records the card it was captured on.

    references/t4_qwen2.5-0.5b.json carries gpu_name "Tesla T4" and
    gpu_capability "sm_75" because a loss trace belongs to its hardware: no
    bf16 and xformers attention on sm_75. Nothing compared it, so a run on
    another GPU was band-checked against a T4 trace and its deviations came
    back as a code regression. The only hardware check anywhere was the GPU
    COUNT, on the kernel, which never reads this file.
    """
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    ref.write_text(
        json.dumps(
            {
                "config": REFERENCE_CONFIG,
                "model": "unsloth/Qwen2.5-0.5B-Instruct",
                "environment": {"gpu_name": "Tesla T4", "gpu_capability": "sm_75"},
                "metrics": [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)],
            }
        ),
        encoding = "utf-8",
    )
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(
        observed,
        ref,
        0.10,
        0.05,
        max_steps = 3,
        config = REFERENCE_CONFIG,
        model = "unsloth/Qwen2.5-0.5B-Instruct",
        environment = {"gpu_name": "Tesla P100-PCIE-16GB", "gpu_capability": "sm_60"},
    )
    assert verdict["status"] == "hardware_mismatch", verdict
    # the reference, so a pass would look like a healthy run on the wrong card.
    # Refused BEFORE any number is compared:
    assert verdict["deviations"] == []
    failures = reference_failures(verdict, 0.10)
    assert failures and "not for this run" in failures[0]
    assert "Tesla P100-PCIE-16GB" in failures[0]

    # Same reference, the card it was captured on:
    same = check_reference(
        observed,
        ref,
        0.10,
        0.05,
        max_steps = 3,
        config = REFERENCE_CONFIG,
        model = "unsloth/Qwen2.5-0.5B-Instruct",
        environment = {"gpu_name": "Tesla T4", "gpu_capability": "sm_75"},
    )
    assert same["status"] == "ok", same
    assert reference_failures(same, 0.10) == []


def test_a_reference_that_records_no_hardware_is_unchecked_not_a_mismatch(tmp_path):
    """ "It does not say" is not "it differs", the rule the settings follow.

    An older reference captured before the environment block carried a GPU
    name must keep working rather than fail every run; the skip is recorded so
    it cannot read as a comparison that passed.
    """
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(
        observed,
        ref,
        0.10,
        0.05,
        max_steps = 3,
        config = REFERENCE_CONFIG,
        environment = {"gpu_name": "Tesla T4", "gpu_capability": "sm_75"},
    )
    assert verdict["status"] == "ok", verdict
    assert "gpu_name" in verdict["config_unchecked"]
    assert reference_failures(verdict, 0.10) == []


@pytest.mark.parametrize(
    "environment",
    [
        pytest.param({"error": "RuntimeError: no CUDA driver"}, id = "probe_raised"),
        pytest.param({"python": "3.12.13", "torch": "2.10.0"}, id = "probe_saw_no_gpu"),
        pytest.param({}, id = "empty"),
        pytest.param(None, id = "not_supplied"),
        pytest.param({"gpu_name": "Tesla T4", "gpu_capability": None}, id = "one_key_missing"),
    ],
)
def test_a_run_that_cannot_name_its_card_is_refused_not_waved_through(tmp_path, environment):
    """The hardware gate must not switch itself off when the probe fails.

    main() records ``environment = {"error": ...}`` for the whole block when
    environment_fingerprint() raises, and the fingerprint omits every gpu_* key
    outright when torch.cuda.is_available() is False. Either way the live values
    are absent while the reference still names Tesla T4 / sm_75, and treating
    that as "not compared" let the control leg report an ``ok`` reference check
    without ever establishing the card the trace belongs to -- the gate defeated
    by exactly the failure it exists to catch. What the REFERENCE does not say
    stays a skip; what the RUN cannot say about a key the reference does name is
    a refusal.
    """
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    ref.write_text(
        json.dumps(
            {
                "config": REFERENCE_CONFIG,
                "model": "unsloth/Qwen2.5-0.5B-Instruct",
                "environment": {"gpu_name": "Tesla T4", "gpu_capability": "sm_75"},
                "metrics": [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)],
            }
        ),
        encoding = "utf-8",
    )
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(
        observed,
        ref,
        0.10,
        0.05,
        max_steps = 3,
        config = REFERENCE_CONFIG,
        model = "unsloth/Qwen2.5-0.5B-Instruct",
        environment = environment,
    )
    assert verdict["status"] == "hardware_unverified", verdict
    assert verdict["deviations"] == [], "refused before any number was compared"
    assert "gpu_name" not in verdict["config_unchecked"], verdict
    failures = reference_failures(verdict, 0.10)
    assert failures and "not for this run" in failures[0], failures


def test_the_committed_reference_names_the_card_the_gate_reads(tmp_path):
    """The gate is derived from the file, so the file has to carry it.

    A reference recaptured without gpu_name silently turns the check above
    into a skip, which is the shape of every defect this suite keeps finding.
    """
    reference = SMOKE_DIR / "references" / "t4_qwen2.5-0.5b.json"
    environment = json.loads(reference.read_text(encoding = "utf-8"))["environment"]
    assert environment["gpu_name"] == "Tesla T4"
    assert environment["gpu_capability"] == "sm_75"


REFERENCE_CONFIG = {
    "max_steps": 3,
    # The rows, not the path: what trained is part of which experiment the trace is a trace of, and the payload records
    "dataset_digest": "d" * 64,
    "init_loss_scale": 0.0,
    "batch_size": 2,
    "grad_accum": 1,
    "max_seq_length": 512,
    "learning_rate": 0.001,
    "lora_r": 16,
    "lora_alpha": 32,
    "optim": "adamw_8bit",
    "gradient_checkpointing": "unsloth",
    "repeat": 2,
}


def test_a_reference_captured_with_another_learning_rate_is_refused(tmp_path):
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(
        observed,
        ref,
        0.10,
        0.05,
        max_steps = 3,
        config = dict(REFERENCE_CONFIG, learning_rate = 0.005),
    )
    assert verdict["status"] == "config_mismatch"
    assert verdict["config_differences"] == [
        {"key": "learning_rate", "reference": 0.001, "observed": 0.005}
    ]
    assert reference_failures(verdict, 0.10)


def test_a_reference_captured_for_another_model_is_refused(tmp_path):
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(
        observed,
        ref,
        0.10,
        0.05,
        max_steps = 3,
        config = REFERENCE_CONFIG,
        model = "unsloth/Llama-3.2-1B-Instruct",
    )
    assert verdict["status"] == "config_mismatch"
    assert reference_failures(verdict, 0.10)


def test_the_repeat_count_does_not_invalidate_a_reference(tmp_path):
    """Each cycle is a fresh process, so how many were run is not the run."""
    from run_t4_smoke import check_reference

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(
        observed, ref, 0.10, 0.05, max_steps = 3, config = dict(REFERENCE_CONFIG, repeat = 5)
    )
    assert verdict["status"] == "ok"


def test_a_reference_that_predates_a_setting_does_not_refuse_on_it(tmp_path):
    """An older file simply does not carry the key; that is not a mismatch."""
    from run_t4_smoke import check_reference

    ref = tmp_path / "ref.json"
    older = {k: v for k, v in REFERENCE_CONFIG.items() if k != "gradient_checkpointing"}
    _write_reference(ref, config = older)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(observed, ref, 0.10, 0.05, max_steps = 3, config = REFERENCE_CONFIG)
    assert verdict["status"] == "ok"
    # `model` too: the helper's reference names one and this call observed none, and a pin present on one side only did
    assert verdict["config_unchecked"] == ["gradient_checkpointing", "model"]


def test_the_reference_check_still_works_without_an_observed_config(tmp_path):
    """Backwards compatible: the step-count guard is what it always was."""
    from run_t4_smoke import check_reference

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    assert check_reference(observed, ref, 0.10, 0.05, max_steps = 3)["status"] == "ok"
    assert (
        check_reference(observed, ref, 0.10, 0.05, max_steps = 10)["status"] == "step_count_mismatch"
    )


def test_a_shifted_step_coordinate_is_refused_by_the_band_check(tmp_path):
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": 1, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(observed, ref, 0.10, 0.05, max_steps = 3)
    assert verdict["status"] == "step_mismatch"
    assert verdict["step_differences"] == [
        {"index": 1, "reference": 2, "observed": 1},
        {"index": 2, "reference": 3, "observed": 1},
    ]
    assert reference_failures(verdict, 0.10)


def test_the_model_revision_travels_with_the_reference(tmp_path):
    from run_t4_smoke import check_reference, reference_failures

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    payload = json.loads(ref.read_text())
    payload["resolved_checkpoint"] = "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit"
    payload["resolved_revision"] = "10413c288cb9629acdf60b3e0229f3ba75efe413"
    ref.write_text(json.dumps(payload))
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]

    same = check_reference(
        observed,
        ref,
        0.10,
        0.05,
        max_steps = 3,
        resolved_checkpoint = "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
        resolved_revision = "10413c288cb9629acdf60b3e0229f3ba75efe413",
    )
    assert same["status"] == "ok"

    moved = check_reference(
        observed,
        ref,
        0.10,
        0.05,
        max_steps = 3,
        resolved_checkpoint = "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
        resolved_revision = "0000000000000000000000000000000000000000",
    )
    assert moved["status"] == "config_mismatch"
    assert reference_failures(moved, 0.10)


def test_a_reference_with_no_recorded_revision_does_not_refuse(tmp_path):
    """The committed file predates this; unknown is unknown, not a mismatch."""
    from run_t4_smoke import check_reference

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(observed, ref, 0.10, 0.05, max_steps = 3, resolved_revision = "abc123")
    assert verdict["status"] == "ok"
    assert "resolved_revision" in verdict["config_unchecked"]


def test_a_pin_the_run_could_not_read_is_recorded_as_unchecked(tmp_path):
    """The other direction: the reference names a commit and the run does not.

    Skipping that silently is a checkpoint pin that stopped running with
    nothing to say so.
    """
    from run_t4_smoke import check_reference

    ref = tmp_path / "ref.json"
    _write_reference(ref, config = REFERENCE_CONFIG)
    payload = json.loads(ref.read_text())
    payload["resolved_revision"] = "10413c288cb9629acdf60b3e0229f3ba75efe413"
    ref.write_text(json.dumps(payload))
    observed = [{"step": s, "loss": 1.0 / s, "grad_norm": 3.0} for s in (1, 2, 3)]
    verdict = check_reference(observed, ref, 0.10, 0.05, max_steps = 3)
    assert verdict["status"] == "ok"
    assert "resolved_revision" in verdict["config_unchecked"]
    assert "resolved_checkpoint" not in verdict["config_unchecked"]


def test_what_the_band_did_not_compare_reaches_the_summary():
    """A skip nobody can see reads as a comparison that passed."""
    import report as report_module

    lines = report_module.render(
        {
            "label": "control",
            "model": "unsloth/Qwen2.5-0.5B-Instruct",
            "passed": True,
            "metrics": [],
            "reference_check": {
                "status": "ok",
                "worst_rel": {"loss": 0.01},
                "reference_max_steps": 10,
                "config_unchecked": ["resolved_revision"],
            },
        }
    )
    assert any("resolved_revision" in line for line in lines), lines


def test_the_committed_reference_pins_the_model_it_was_captured_on():
    """The one identity key that is knowable without another T4 session.

    The control leg passes no --model, so the reference belongs to DEFAULT_MODEL
    and the gate can be live now rather than from the next recapture.
    """
    from run_t4_smoke import DEFAULT_MODEL

    reference = json.loads(
        (SMOKE_DIR / "references" / "t4_qwen2.5-0.5b.json").read_text(encoding = "utf-8")
    )
    assert reference["model"] == DEFAULT_MODEL




def _batch_record(**over):
    """A healthy batched-generation record, with fields overridable per test."""
    base = {
        "prompt_token_lengths": [11, 14, 9, 17, 12, 20, 8, 15],
        "distinct_lengths": 8,
        "padding_side_observed": "left",
        "padding_side_after": "left",
        "singles": [f"out{i}" for i in range(8)],
        "batched": {
            "2": [f"out{i}" for i in range(8)],
            "4": [f"out{i}" for i in range(8)],
            "8": [f"out{i}" for i in range(8)],
        },
        "agrees": {"2": True, "4": True, "8": True},
        "empty_outputs": [],
    }
    base.update(over)
    return base


def _cycle(index: int, losses: list[float]) -> dict:
    return {
        "run_index": index,
        "metrics": [
            {"step": s, "loss": losses[s - 1], "grad_norm": 3.0} for s in range(1, len(losses) + 1)
        ],
        "generated": "__UNSLOTH__!!!",
        "canary_found": True,
        # A simulated HEALTHY cycle has to look healthy in every respect a real
        # one is judged on, this record included. Leaving it out made
        # `--check-batched-generation` (on by default, because a leg that
        # quietly skips it stops covering #3699/#1066/#1456/#2138) fail every
        # simulated run with "batched generation was never run" -- which is the
        # rule working, on a fixture that had not kept up.
        "batched_generation": _batch_record(),
        "adapter_files": ["adapter_config.json", "adapter_model.safetensors"],
        "saved_adapter": _adapter_state(),
        "determinism": {},
        "loss_scale": {},
        "timing_seconds": {},
        "peak_reserved_gb": 0.1,
    }


def _drive_main(monkeypatch, tmp_path, cycles: dict, argv: list[str]) -> tuple[int, dict]:
    import run_t4_smoke

    real_run = run_t4_smoke.subprocess.run

    def fake_run(cmd, *a, **kw):
        if "--cycle" not in cmd:
            return real_run(cmd, *a, **kw)
        index = int(cmd[cmd.index("--cycle") + 1])
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        outdir.mkdir(parents = True, exist_ok = True)
        payload = cycles.get(index)
        if payload is None:
            return types.SimpleNamespace(returncode = 1)
        (outdir / "cycle_report.json").write_text(json.dumps(payload), encoding = "utf-8")
        return types.SimpleNamespace(returncode = 0)

    monkeypatch.setattr(run_t4_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_t4_smoke.py", "--outdir", str(tmp_path)] + argv)
    code = run_t4_smoke.main()
    return code, json.loads((tmp_path / "t4_smoke_report.json").read_text(encoding = "utf-8"))


def test_every_requested_repeat_is_compared_against_the_baseline(monkeypatch, tmp_path):
    """--repeat 3 compared cycles 0 and 1 and threw cycle 2 away."""
    cycles = {
        0: _cycle(0, [3.0, 2.0, 1.0]),
        1: _cycle(1, [3.0, 2.0, 1.0]),
        2: _cycle(2, [3.0, 2.0, 99.0]),
    }
    code, report = _drive_main(monkeypatch, tmp_path, cycles, ["--repeat", "3", "--max-steps", "3"])
    assert code == 1
    assert report["passed"] is False
    assert any("cycle 2" in f for f in report["failures"])
    repro = report["reproducibility"]
    assert repro["compared_cycles"] == ["1", "2"]
    assert repro["cycles"]["1"]["identical"] is True
    assert repro["cycles"]["2"]["identical"] is False
    assert repro["identical"] is False
    assert repro["first_diff_step"] == 3
    assert repro["max_abs_diff"]["loss"] == 98.0


def test_two_agreeing_repeats_still_pass(monkeypatch, tmp_path):
    cycles = {0: _cycle(0, [3.0, 2.0, 1.0]), 1: _cycle(1, [3.0, 2.0, 1.0])}
    code, report = _drive_main(monkeypatch, tmp_path, cycles, ["--repeat", "2", "--max-steps", "3"])
    assert code == 0
    assert report["passed"] is True
    assert report["reproducibility"]["identical"] is True
    assert report["reproducibility"]["cycles"]["1"]["identical"] is True


def test_the_summary_renderer_still_reads_a_multi_repeat_report(monkeypatch, tmp_path):
    """report.py renders `identical` / `first_diff_step` / `max_abs_diff`
    straight off `reproducibility`; a shape change there is a silent
    "**DIFFERED**" on every healthy run."""
    from report import render

    cycles = {0: _cycle(0, [3.0, 2.0, 1.0]), 1: _cycle(1, [3.0, 2.0, 1.0])}
    _, report_dict = _drive_main(
        monkeypatch, tmp_path, cycles, ["--repeat", "2", "--max-steps", "3"]
    )
    rendered = "\n".join(render(report_dict))
    assert "agreed **bitwise**" in rendered
    assert "DIFFERED" not in rendered


def test_a_failed_cycle_still_reports_its_environment(monkeypatch, tmp_path):
    """The one case where knowing which library set died matters most."""
    code, report = _drive_main(monkeypatch, tmp_path, {}, ["--repeat", "2", "--max-steps", "3"])
    assert code == 1
    assert report["passed"] is False
    assert report["environment"]["resolved"]
    assert report["config"]["max_steps"] == 3
    assert report["label"] == "t4-smoke"




def _gptoss_args(**over):
    base = {"max_steps": 3, "require_compile": True}
    base.update(over)
    return types.SimpleNamespace(**base)


def _gptoss_result(**over) -> dict:
    result = {
        "metrics": [{"step": s, "loss": 3.0 - s, "grad_norm": 1.5} for s in (1, 2, 3)],
        "compile": {"available": True, "unique_graphs": 32, "unique_graphs_delta": 30},
        "generated": "hello there",
        "precision": {
            "fp16": False,
            "bf16": False,
            "force_float32_env": "1",
            "custom_dtype_env": "down_projs;mlp.router",
        },
        "environment": {"bf16_supported": False, "gpu_name": "Tesla T4"},
        # What the feasibility probe measured:
        "placement_after_load": {
            "parameters_by_device": {"cuda:0": 20_900_000_000},
            "hf_device_map_devices": None,
            "offloaded": False,
        },
    }
    result.update(over)
    return result


def test_a_healthy_gptoss_run_passes():
    from run_gptoss_t4 import failures_for
    assert failures_for(_gptoss_result(), _gptoss_args()) == []


def test_gptoss_fails_when_the_forced_float32_path_stopped_firing():
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        precision = {
            "fp16": True,
            "bf16": False,
            "force_float32_env": None,
            "custom_dtype_env": None,
        }
    )
    failures = failures_for(result, _gptoss_args())
    assert failures and any("float32" in f for f in failures)


def test_gptoss_does_not_demand_float32_on_a_card_with_bf16():
    """The force path exists for cards without bf16; elsewhere it is right
    that it does not fire, and a red there would be the harness's own bug."""
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        environment = {"bf16_supported": True, "gpu_name": "NVIDIA A100"},
        precision = {
            "fp16": False,
            "bf16": True,
            "force_float32_env": None,
            "custom_dtype_env": None,
        },
    )
    assert failures_for(result, _gptoss_args()) == []


def test_gptoss_fails_when_precision_was_never_recorded():
    from run_gptoss_t4 import failures_for

    result = _gptoss_result()
    result.pop("precision")
    failures = failures_for(result, _gptoss_args())
    assert failures and any("precision" in f for f in failures)


def test_gptoss_measures_compilation_across_training_only():
    """Loading compiled 32 graphs and training compiled none."""
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        compile = {"available": True, "unique_graphs": 32, "unique_graphs_delta": 0}
    )
    failures = failures_for(result, _gptoss_args())
    assert failures and any("eager" in f for f in failures)


@pytest.mark.parametrize("value", ["0", "", None, "true"])
def test_gptoss_requires_the_forcing_to_be_on_rather_than_merely_recorded(value):
    """`"0"` is what the loader writes on its ordinary branch.

    models/loader.py sets UNSLOTH_FORCE_FLOAT32 to "0" BEFORE deciding whether
    to force and overwrites it with "1" only when the forcing fires, and every
    production consumer reads `== "1"`. A truthiness check therefore accepts the
    one regression this leg uniquely covers: forcing off, fp16 and bf16 still
    false, leg green.
    """
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        precision = {
            "fp16": False,
            "bf16": False,
            "force_float32_env": value,
            "custom_dtype_env": None,
        }
    )
    failures = failures_for(result, _gptoss_args())
    assert failures and any("UNSLOTH_FORCE_FLOAT32" in f for f in failures), failures


def test_gptoss_refuses_a_compile_check_that_has_no_baseline():
    """A post-training read with no pre-training one to subtract."""
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(compile = {"available": True, "unique_graphs": 32})
    failures = failures_for(result, _gptoss_args())
    assert failures and any("pre-training dynamo counters" in f for f in failures)


def test_gptoss_compile_counters_report_a_delta():
    from run_gptoss_t4 import compile_counters

    before = compile_counters()
    after = compile_counters(before = before)
    assert after["unique_graphs_delta"] == 0
    fake = compile_counters(before = {"available": True, "unique_graphs": 5, "calls_captured": 9})
    assert fake["unique_graphs_delta"] == fake["unique_graphs"] - 5


def test_gptoss_fails_when_no_optimizer_update_was_applied():
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        metrics = [{"step": s, "loss": 3.0 - s, "grad_norm": 0.0} for s in (1, 2, 3)]
    )
    failures = failures_for(result, _gptoss_args())
    assert failures and any("optimizer update" in f for f in failures)


def _moved(**over) -> dict:
    """An adapter reading from a run that trained."""
    reading = {
        "ok": True,
        "changed": True,
        "tensors": 168,
        "abs_sum_before": 100.0,
        "abs_sum_after": 101.5,
        "b_abs_sum_before": 0.0,
        "b_abs_sum_after": 1.5,
    }
    reading.update(over)
    return reading


def test_gptoss_does_not_infer_a_verdict_from_an_unlogged_grad_norm():
    """A trainer that stops logging grad_norm still says nothing either way.

    That has not changed; what changed is that the leg no longer depends on the
    field, the trained adapter being fingerprinted before and after, so silence
    from the trainer is answered from the weights rather than guessed at.
    """
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        metrics = [{"step": s, "loss": 3.0 - s} for s in (1, 2, 3)],
        adapter_update = _moved(),
    )
    assert failures_for(result, _gptoss_args()) == []


def test_gptoss_fails_when_nothing_at_all_can_say_the_adapter_moved():
    """No grad_norm logged AND no adapter reading is not a pass.

    Every other number in the report (finite losses, captured graphs, non-empty
    generation) is produced by the base model and the loader on their own, so
    with both instruments gone the leg has nothing left to show for the LoRA
    training it covers.
    """
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        metrics = [{"step": s, "loss": 3.0 - s} for s in (1, 2, 3)],
        adapter_update = {"ok": False, "error": "RuntimeError: meta tensor"},
    )
    failures = failures_for(result, _gptoss_args())
    assert any("could not be established" in f for f in failures), failures


def test_gptoss_fails_when_the_adapter_is_the_one_it_started_with():
    """The grad norms can look healthy and the weights still not move.

    Gradients flowing into weights nobody updated is a run that trained nothing,
    so the adapter reading decides it rather than the telemetry.
    """
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        adapter_update = _moved(changed = False, abs_sum_after = 100.0, b_abs_sum_after = 0.0),
    )
    failures = failures_for(result, _gptoss_args())
    assert any("no optimizer update was applied" in f for f in failures), failures


@pytest.mark.parametrize(
    "placement, expected",
    [
        (
            {
                "parameters_by_device": {"cuda:0": 18_000_000_000, "cpu": 2_900_000_000},
                "hf_device_map_devices": ["0", "cpu"],
                "offloaded": True,
            },
            "off the GPU",
        ),
        (
            {
                "parameters_by_device": {"cuda:0": 18_000_000_000, "disk": 2_900_000_000},
                "hf_device_map_devices": ["0", "disk"],
                "offloaded": True,
            },
            "off the GPU",
        ),
        (
            {
                "parameters_by_device": {"cuda:0": 18_000_000_000, "meta": 2_900_000_000},
                "hf_device_map_devices": None,
                "offloaded": False,
            },
            "off the GPU",
        ),
        (
            {
                "parameters_by_device": {"error": "RuntimeError: no"},
                "hf_device_map_devices": None,
                "offloaded": False,
            },
            "could not be walked",
        ),
        (
            {"parameters_by_device": {}, "hf_device_map_devices": None, "offloaded": False},
            "recorded no devices",
        ),
        (None, "never recorded"),
    ],
)
def test_gptoss_fails_when_the_checkpoint_did_not_stay_on_the_gpu(placement, expected):
    """The leg's documented result is that 20B fits on one T4. Assert it.

    Every other number in the report survives an offload: the losses are finite,
    the adapter moves, compilation engages and generation returns text, just
    slower. So a loader or memory regression that spills to CPU, disk or meta
    reports green while the thin-memory condition this leg exists to hold has
    stopped holding. `placement_after_load` recorded that on every run and
    `failures_for` never read it.
    """
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(placement_after_load = placement)
    failures = failures_for(result, _gptoss_args())
    assert any(expected in f for f in failures), failures


def test_gptoss_accepts_the_placement_the_probe_measured():
    """The floor under the test above: a healthy run is not red for placement.

    Every parameter on the one visible CUDA device and no accelerate dispatch,
    which is what kernels 8161ceb9 / 7ab727f1 reported.
    """
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        placement_after_load = {
            "parameters_by_device": {"cuda:0": 20_900_000_000},
            "hf_device_map_devices": None,
            "offloaded": False,
        },
    )
    assert failures_for(result, _gptoss_args()) == []


def test_gptoss_refuses_a_placement_record_it_cannot_read():
    """An offload flag that is neither True nor False is not a pass.

    Same three-way rule the bf16 reading gets: the check that switches itself
    off when its instrument breaks is the one that never fires.
    """
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(
        placement_after_load = {"parameters_by_device": {"cuda:0": 20_900_000_000}},
    )
    failures = failures_for(result, _gptoss_args())
    assert any("whether the loader offloaded" in f for f in failures), failures


def test_grpo_fails_when_nothing_at_all_can_say_the_adapter_moved():
    """The same hole, and the same reason it is a hole on this leg too.

    Reward, reward_std and the completions all come from generating and scoring,
    which the base model does without a single optimizer step.
    """
    from run_grpo_t4 import failures_for

    result = _grpo_result(
        metrics = [{"step": s, "loss": 0.0} for s in (1, 2, 3)],
    )
    failures = failures_for(result, _grpo_args())
    assert any("could not be established" in f for f in failures), failures


def test_grpo_reads_the_adapter_when_the_trainer_logged_no_norms():
    from run_grpo_t4 import failures_for
    result = _grpo_result(
        metrics = [{"step": s, "loss": 0.0} for s in (1, 2, 3)],
        adapter_update = _moved(),
    )
    assert failures_for(result, _grpo_args()) == []




class _Tensor:
    """The two calls adapter_fingerprint makes on a parameter, and no more."""

    def __init__(self, value):
        self._value = value

    def detach(self):
        return self

    def float(self):
        return self

    def abs(self):
        return self

    def sum(self):
        return self

    def item(self):
        return self._value


def test_an_empty_norm_list_is_not_the_same_answer_as_a_useless_one():
    """The distinction the `if norms and not applied` spelling collapsed."""
    from training_evidence import update_verdict

    useless = [{"step": 1, "loss": 1.0, "grad_norm": 0.0}]
    silent = [{"step": 1, "loss": 1.0}]
    assert update_verdict(useless)["verdict"] == "not_applied"
    assert update_verdict(silent)["verdict"] == "unverifiable"
    assert update_verdict(useless, _moved())["verdict"] == "not_applied"
    assert update_verdict(silent, _moved())["verdict"] == "applied"


def test_the_adapter_reading_beats_the_grad_norms_it_is_a_proxy_for():
    """Healthy norms over weights that did not move is still nothing trained."""
    from training_evidence import update_verdict

    healthy = [{"step": s, "loss": 1.0, "grad_norm": 2.0} for s in (1, 2)]
    frozen = _moved(changed = False, abs_sum_after = 100.0, b_abs_sum_after = 0.0)
    verdict = update_verdict(healthy, frozen)
    assert verdict["verdict"] == "not_applied"
    assert "bitwise identical" in verdict["detail"]


def test_a_fingerprint_of_a_model_with_no_lora_parameters_is_not_an_answer():
    """`ok: False` means the question could not be answered, never "no"."""
    from training_evidence import adapter_fingerprint, adapter_update

    class _Bare:
        def named_parameters(self):
            return iter([("model.layers.0.mlp.up_proj.weight", _Tensor(1.0))])

    from training_evidence import update_verdict

    reading = adapter_fingerprint(_Bare())
    assert reading["ok"] is False
    assert adapter_update(reading, reading)["ok"] is False
    assert update_verdict([{"step": 1, "loss": 1.0}], reading)["verdict"] == "unverifiable"


def test_a_fingerprint_that_raises_is_reported_rather_than_propagated():
    """A diagnostic that kills the payload it diagnoses is worse than the gap."""
    from training_evidence import adapter_fingerprint

    class _Broken:
        def named_parameters(self):
            raise RuntimeError("Cannot copy out of meta tensor")

    reading = adapter_fingerprint(_Broken())
    assert reading["ok"] is False
    assert "RuntimeError" in reading["error"]


def test_the_zero_initialised_b_matrices_are_what_make_the_comparison_safe():
    """Both sums are compared, so an exact float equality can be a red."""
    from training_evidence import adapter_fingerprint, adapter_update

    class _Adapter:
        def __init__(self, b):
            self._b = b

        def named_parameters(self):
            return iter(
                [
                    ("base.layers.0.q_proj.lora_A.default.weight", _Tensor(4.0)),
                    ("base.layers.0.q_proj.lora_B.default.weight", _Tensor(self._b)),
                ]
            )

    before = adapter_fingerprint(_Adapter(0.0))
    assert before == {"ok": True, "tensors": 2, "abs_sum": 4.0, "b_abs_sum": 0.0}
    # A B matrix that moved while the total happened to come out the same.
    after = adapter_fingerprint(_Adapter(0.0))
    after["b_abs_sum"] = 0.5
    assert adapter_update(before, after)["changed"] is True
    assert adapter_update(before, before)["changed"] is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_lora_weight_is_not_read_as_a_successful_update(bad):
    """The strongest possible pass, produced by the worst possible run.

    `NaN != finite` and `inf != finite` both read as "the adapter changed", so a
    run whose optimizer corrupted the weights reported `applied` on exactly the
    no-telemetry path this module decides, while generation still returned text.
    """
    from training_evidence import adapter_fingerprint, adapter_update, update_verdict

    class _Corrupt:
        def __init__(self, b):
            self._b = b

        def named_parameters(self):
            return iter(
                [
                    ("base.layers.0.q_proj.lora_A.default.weight", _Tensor(4.0)),
                    ("base.layers.0.q_proj.lora_B.default.weight", _Tensor(self._b)),
                ]
            )

    before = adapter_fingerprint(_Corrupt(0.0))
    after = adapter_fingerprint(_Corrupt(bad))
    assert after["ok"] is False
    assert after["non_finite"] is True
    assert "lora_B" in after["error"]

    update = adapter_update(before, after)
    assert update["ok"] is False
    assert update["non_finite"] is True

    # And it beats a healthy grad_norm, which says nothing about weights that went non-finite two steps later.
    healthy = [{"step": s, "loss": 1.0, "grad_norm": 2.0} for s in (1, 2)]
    assert update_verdict(healthy, update)["verdict"] == "non_finite"
    assert update_verdict([], update)["verdict"] == "non_finite"


@pytest.mark.parametrize("field", ["abs_sum", "b_abs_sum"])
def test_a_non_finite_sum_is_refused_by_the_comparison_as_well(field):
    """Checked where the exact `!=` happens, not only where it was summed."""
    from training_evidence import adapter_update, update_verdict

    before = {"ok": True, "tensors": 4, "abs_sum": 1.0, "b_abs_sum": 0.0}
    after = dict(before, **{field: float("nan")})
    verdict = adapter_update(before, after)
    assert verdict["ok"] is False
    assert verdict["non_finite"] is True
    assert update_verdict([], verdict)["verdict"] == "non_finite"


def test_a_non_finite_adapter_turns_both_legs_red():
    """The verdict has to reach a failure string, or it is not a check."""
    from run_gptoss_t4 import failures_for as gptoss_failures
    from run_grpo_t4 import failures_for as grpo_failures

    corrupt = {
        "ok": False,
        "non_finite": True,
        "error": "1 of 4 LoRA tensors hold non-finite weights",
    }
    gptoss = gptoss_failures(_gptoss_result(adapter_update = corrupt), _gptoss_args())
    assert gptoss and any("non-finite weights" in f for f in gptoss)
    grpo = grpo_failures(_grpo_result(adapter_update = corrupt), _grpo_args())
    assert grpo and any("non-finite weights" in f for f in grpo)


def test_a_verdict_neither_leg_has_been_taught_about_is_still_a_failure():
    """`elif != "applied"`, not `elif == "unverifiable"`.

    A verdict added to training_evidence.py and not wired here would otherwise
    be a silent pass.
    """
    import run_gptoss_t4
    import run_grpo_t4

    unknown = {"verdict": "something_new", "detail": "d", "grad_norms": []}
    for module, args in ((run_gptoss_t4, _gptoss_args()), (run_grpo_t4, _grpo_args())):
        original = module.update_verdict
        module.update_verdict = lambda *a, **k: unknown
        try:
            result = _gptoss_result() if module is run_gptoss_t4 else _grpo_result()
            assert module.failures_for(result, args), module.__name__
        finally:
            module.update_verdict = original


def test_two_fingerprints_over_different_tensor_counts_are_not_compared():
    from training_evidence import adapter_update

    before = {"ok": True, "tensors": 168, "abs_sum": 1.0, "b_abs_sum": 0.0}
    after = {"ok": True, "tensors": 12, "abs_sum": 1.0, "b_abs_sum": 0.0}
    verdict = adapter_update(before, after)
    assert verdict["ok"] is False
    assert "not comparable" in verdict["error"]




def _grpo_args(**over):
    base = {"max_steps": 3}
    base.update(over)
    return types.SimpleNamespace(**base)


def _grpo_result(**over) -> dict:
    result = {
        "log_history": [
            {"step": s, "loss": 0.0, "reward": 0.5, "reward_std": 0.2} for s in (1, 2, 3)
        ],
        "metrics": [{"step": s, "loss": 0.0, "grad_norm": 1.25} for s in (1, 2, 3)],
        "completions": [["four score and seven"], ["a completion with 7 in it"]],
        "fast_generate": "some generated text",
        "fast_generate_lora": {"requested": True, "applied": True},
    }
    result.update(over)
    return result


def test_a_healthy_grpo_run_passes():
    from run_grpo_t4 import failures_for
    assert failures_for(_grpo_result(), _grpo_args()) == []


def test_grpo_fails_when_a_step_logged_no_reward():
    from run_grpo_t4 import failures_for

    history = [
        {"step": 1, "loss": 0.0, "reward": 0.5, "reward_std": 0.2},
        {"step": 2, "loss": 0.0},
        {"step": 3, "loss": 0.0, "reward": 0.5, "reward_std": 0.2},
    ]
    failures = failures_for(_grpo_result(log_history = history), _grpo_args())
    assert failures and any("every step" in f for f in failures)


def test_grpo_ignores_the_summary_entry_that_carries_no_loss():
    """`train()` appends a train_runtime row; it is not a training step."""
    from run_grpo_t4 import failures_for

    history = [{"step": s, "loss": 0.0, "reward": 0.5, "reward_std": 0.2} for s in (1, 2, 3)]
    history.append({"step": 3})
    assert failures_for(_grpo_result(log_history = history), _grpo_args()) == []


def test_grpo_fails_when_every_gradient_norm_is_nan():
    from run_grpo_t4 import failures_for

    result = _grpo_result(metrics = [{"step": s, "loss": 0.0, "grad_norm": NAN} for s in (1, 2, 3)])
    failures = failures_for(result, _grpo_args())
    assert failures and any("optimizer update" in f for f in failures)


def test_grpo_fails_when_the_final_generation_skipped_the_trained_adapter():
    from run_grpo_t4 import failures_for

    result = _grpo_result(
        fast_generate_lora = {"requested": True, "applied": False, "error": "AttributeError: x"}
    )
    failures = failures_for(result, _grpo_args())
    assert failures and any("adapter" in f for f in failures)


def test_grpo_records_the_engine_it_built_before_a_later_failure(monkeypatch, tmp_path):
    """Construction succeeded and training raised: the report must say so."""
    import run_grpo_t4

    def exploding_train(args, report = None):
        if report is not None:
            report["engine_built"] = True
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(run_grpo_t4, "train", exploding_train)
    monkeypatch.setattr(run_grpo_t4, "make_libcuda_linkable", lambda: {"needed": False})
    monkeypatch.setattr(run_grpo_t4, "vllm_facts", lambda: {"version": "0.11.2"})
    monkeypatch.setattr(sys, "argv", ["run_grpo_t4.py", "--outdir", str(tmp_path)])
    code = run_grpo_t4.main()
    report = json.loads((tmp_path / "t4_smoke_report.json").read_text(encoding = "utf-8"))
    assert code == 1
    assert report["engine_built"] is True
    assert any("illegal memory access" in f for f in report["failures"])


def test_grpo_still_reports_an_engine_that_never_built(monkeypatch, tmp_path):
    import run_grpo_t4

    def exploding_train(args, report = None):
        raise RuntimeError("EngineCore failed to start")

    monkeypatch.setattr(run_grpo_t4, "train", exploding_train)
    monkeypatch.setattr(run_grpo_t4, "make_libcuda_linkable", lambda: {"needed": False})
    monkeypatch.setattr(run_grpo_t4, "vllm_facts", lambda: {"version": "0.11.2"})
    monkeypatch.setattr(sys, "argv", ["run_grpo_t4.py", "--outdir", str(tmp_path)])
    assert run_grpo_t4.main() == 1
    report = json.loads((tmp_path / "t4_smoke_report.json").read_text(encoding = "utf-8"))
    assert report["engine_built"] is False




def test_the_recapture_recipe_selects_the_control_report_by_label():
    """`reports[0]` is whichever kernel slug sorted first, not the control."""
    readme = (SMOKE_DIR / "references" / "README.md").read_text(encoding = "utf-8")
    assert 'reports"][0]' not in readme.replace(" ", "")
    assert '"control"' in readme
    recipe = readme[readme.index("## Capturing one") :]
    assert "label" in recipe


def _recapture_recipe() -> str:
    readme = (SMOKE_DIR / "references" / "README.md").read_text(encoding = "utf-8")
    body = readme.split("python - <<'PY'\n", 1)[1]
    return body.split("\nPY\n", 1)[0]


def test_the_recapture_recipe_records_the_kernel_it_came_from(tmp_path, monkeypatch):
    """The recipe is EXECUTED here, against a two-kernel evidence tree.

    `source_kernel` is the only field pointing at the hardware run the band was
    measured on, and is what makes a recapture auditable while the evidence
    artifact is still around. Writing the leg label into it names something
    every reference has and no run in particular.
    """
    evidence = tmp_path / "kaggle_evidence"
    kernels = [
        ("danielhanchen/unsloth-t4-ci-9f0e1a2b", ("control", "canary")),
        ("danielhanchen/unsloth-t4-ci-0a1b2c3d", ("gptoss", "frontier")),
    ]
    reports = []
    for slug, leg_names in kernels:
        directory = evidence / slug.rsplit("/", 1)[-1]
        directory.mkdir(parents = True)
        for leg in leg_names:
            (directory / f"t4_{leg}_output.ipynb").write_text("{}", encoding = "utf-8")
            reports.append(
                {
                    "label": leg,
                    "passed": True,
                    "model": "unsloth/Qwen2.5-0.5B-Instruct",
                    "metrics": [{"step": 1, "loss": 1.0}],
                    "environment": {"gpu_name": "Tesla T4"},
                    "config": {"max_steps": 10},
                    "resolved_checkpoint": "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
                    "resolved_revision": "0123456789abcdef",
                }
            )
    (evidence / "launch_result.json").write_text(
        json.dumps({"reports": reports, "kernels": [{"slug": s} for s, _ in kernels]}),
        encoding = "utf-8",
    )
    (tmp_path / "tests" / "kaggle" / "t4_smoke" / "references").mkdir(parents = True)

    monkeypatch.chdir(tmp_path)
    exec(compile(_recapture_recipe(), "<recapture-recipe>", "exec"), {"__name__": "__main__"})

    written = json.loads(
        (tmp_path / "tests/kaggle/t4_smoke/references/t4_qwen2.5-0.5b.json").read_text(
            encoding = "utf-8"
        )
    )
    assert written["source_kernel"] == "danielhanchen/unsloth-t4-ci-9f0e1a2b"
    assert written["model"] == "unsloth/Qwen2.5-0.5B-Instruct"
    assert written["resolved_revision"] == "0123456789abcdef"


def test_the_committed_reference_names_a_kernel_and_not_a_leg():
    reference = json.loads(
        (SMOKE_DIR / "references" / "t4_qwen2.5-0.5b.json").read_text(encoding = "utf-8")
    )
    assert "/" in reference["source_kernel"], reference["source_kernel"]


def test_reports_are_not_ordered_control_first(tmp_path):
    """The fact behind the item above, driven through the real extractor."""
    from launch import extract_reports

    def notebook(label: str) -> str:
        line = "T4_SMOKE_REPORT " + json.dumps(
            {"label": label, "model": "m", "metrics": [], "environment": {}, "config": {}}
        )
        return json.dumps({"cells": [{"outputs": [{"text": line + "\n"}]}]})

    for slug, legs in (
        ("unsloth-t4-ci-9f0e1a2b", ("control", "canary")),
        ("unsloth-t4-ci-0a1b2c3d", ("gptoss", "frontier")),
    ):
        kernel = tmp_path / slug
        kernel.mkdir(parents = True)
        for leg in legs:
            (kernel / f"t4_{leg}_output.ipynb").write_text(notebook(leg), encoding = "utf-8")

    labels = [r["label"] for r in extract_reports(tmp_path)]
    assert labels[0] != "control"
    assert sorted(labels) == ["canary", "control", "frontier", "gptoss"]


@pytest.mark.parametrize(
    "environment",
    [
        {"error": "RuntimeError: is_bf16_supported() failed"},
        {},
        {"gpu_name": "Tesla T4"},
        {"bf16_supported": None},
        {"bf16_supported": "false"},
        {"bf16_supported": 0},
    ],
)
def test_gptoss_fails_when_the_cards_bf16_support_is_unreadable(environment):
    """`is False` alone made the float32 assertion optional.

    main() records `environment = {"error": ...}` for the whole probe when it
    raises (a torch build that changed or failed `is_bf16_supported()` is
    enough), and the block below was then skipped entirely while training,
    finite losses, an updated adapter, compilation and generation all passed --
    green without ever establishing the float32 path this leg uniquely covers.

    Not just the `error` shape: anything that is not a literal True or False is
    unverifiable, including a plausible-looking string or 0.
    """
    from run_gptoss_t4 import failures_for

    result = _gptoss_result(environment = environment)
    failures = failures_for(result, _gptoss_args())
    assert failures and any("bf16" in f for f in failures), failures


def test_gptoss_still_reads_a_bf16_card_and_a_t4_the_way_it_did():
    """The two literal readings keep their meanings, so the widening above
    cannot be satisfied by turning every card into a failure."""
    from run_gptoss_t4 import failures_for

    assert failures_for(_gptoss_result(), _gptoss_args()) == []
    assert (
        failures_for(
            _gptoss_result(
                environment = {"bf16_supported": True, "gpu_name": "NVIDIA A100"},
                precision = {"fp16": False, "bf16": True, "force_float32_env": None},
            ),
            _gptoss_args(),
        )
        == []
    )


def test_batched_generation_runs_end_to_end_against_a_stub_model():
    """Every other test in this section feeds `batched_generation_failures` a
    dict someone typed. That checks the RULE and never once executes the code
    that produces the dict, which is how kernel unsloth-probe-defaultleg-723c28
    trained all ten steps on a real T4 and then died on

        NameError: name 'torch' is not defined

    inside `batched_generation` itself. Every torch user in run_t4_smoke.py
    imports it inside the function; that one did not, and no CPU test noticed
    because none of them ever called it.

    So drive the real function with a stub tokenizer and model. The stub echoes
    a deterministic continuation per row, so agreement across batch sizes is
    guaranteed and this asserts the plumbing rather than the model: shapes,
    the padded-width slice, and that the function runs at all.
    """
    import torch

    from run_t4_smoke import batched_generation, batched_generation_failures

    class _Enc(dict):
        def to(self, _device):
            return self

    class _Tok:
        padding_side = "right"
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0

        def __call__(
            self,
            text,
            return_tensors = None,
            padding = False,
        ):
            texts = [text] if isinstance(text, str) else list(text)
            # One token per character, so a length spread in the prompts is a
            # length spread in the ids and the padding is real.
            ids = [[ord(c) % 100 + 1 for c in t] for t in texts]
            if return_tensors is None:
                return {"input_ids": ids[0] if isinstance(text, str) else ids}
            width = max(len(i) for i in ids)
            # LEFT padding, which is what the function asks for and what the
            # padded-width slice below depends on.
            padded = [[0] * (width - len(i)) + i for i in ids]
            return _Enc(
                input_ids = torch.tensor(padded),
                attention_mask = torch.tensor([[0] * (width - len(i)) + [1] * len(i) for i in ids]),
            )

        def decode(
            self,
            row,
            skip_special_tokens = False,
        ):
            return "".join(chr(int(v)) for v in row if int(v) != 0)

    class _Model:
        device = "cpu"

        def generate(
            self,
            input_ids = None,
            attention_mask = None,
            max_new_tokens = 8,
            **_kw,
        ):
            # Append the same continuation to every row, derived from that
            # row's own unpadded content, so batching cannot change it.
            outs = []
            for row in input_ids:
                real = [int(v) for v in row if int(v) != 0]
                tail = [(sum(real) % 26) + 65] * max_new_tokens
                outs.append([int(v) for v in row] + tail)
            return torch.tensor(outs)

    # Eight, because BATCH_SIZES tops out at 8 and the rule rejects a record
    # whose largest batch could never have been formed. Lengths 1..8 so every
    # batch pads and the padded-width slice is exercised rather than skipped.
    prompts = ["abcdefgh"[:n] * 1 for n in range(1, 9)]
    record = batched_generation(_Model(), _Tok(), prompts, max_new_tokens = 4)

    assert record["distinct_lengths"] == 8, "the stub prompts must actually pad"
    assert record["padding_side_observed"] == "left"
    assert len(record["singles"]) == len(prompts)
    assert all(s for s in record["singles"]), "the padded-width slice dropped everything"
    assert batched_generation_failures(record) == []


def test_a_healthy_batched_generation_record_reports_no_failures():
    from run_t4_smoke import batched_generation_failures
    assert batched_generation_failures(_batch_record()) == []


def test_a_batch_whose_prompts_are_all_one_length_is_reported_as_proving_nothing():
    from run_t4_smoke import batched_generation_failures

    """The vacuity case, and the reason this check exists in this shape.

    Left padding only happens when the prompts in a batch differ in length. Feed
    eight identical-length prompts and every batch agrees with every single --
    perfectly, every time, having never padded once. That is a green
    left-padding assertion covering nothing, which is the failure mode this
    whole file keeps being caught by.
    """
    failures = batched_generation_failures(
        _batch_record(prompt_token_lengths = [12] * 8, distinct_lengths = 1)
    )
    assert any("nothing was ever padded" in f for f in failures), failures


def test_a_padding_side_silently_flipped_to_right_is_a_failure():
    from run_t4_smoke import batched_generation_failures

    """#2138: a release forced the tokenizer padding side to right in inference.

    Checked twice, before and after generating, because the override that
    caused it happened INSIDE the inference path -- so the value this payload
    set is not evidence of the value that was used.
    """
    assert any(
        "padding_side_after" in f
        for f in batched_generation_failures(_batch_record(padding_side_after = "right"))
    )
    assert any(
        "padding_side_observed" in f
        for f in batched_generation_failures(_batch_record(padding_side_observed = "right"))
    )


def test_a_batch_that_disagrees_with_one_at_a_time_is_a_failure():
    from run_t4_smoke import batched_generation_failures

    """#3699 / #1456: batched output diverging from sequential greedy output."""
    broken = _batch_record()
    broken["batched"]["4"] = ["different"] * 8
    broken["agrees"]["4"] = False
    failures = batched_generation_failures(broken)
    assert any("batch size 4 did not reproduce" in f for f in failures), failures


def test_empty_generations_are_a_failure_even_when_every_batch_agrees():
    from run_t4_smoke import batched_generation_failures

    """#1066: gibberish, of which "nothing at all" is the degenerate case.

    Agreement alone is not health: a model that emits an empty string for every
    prompt agrees with itself at every batch size.
    """
    failures = batched_generation_failures(
        _batch_record(
            singles = [""] * 8,
            empty_outputs = [0, 1, 2, 3, 4, 5, 6, 7],
            batched = {"2": [""] * 8, "4": [""] * 8, "8": [""] * 8},
        )
    )
    assert any("generated nothing at all" in f for f in failures), failures


def test_too_few_prompts_to_fill_the_largest_batch_is_reported():
    from run_t4_smoke import batched_generation_failures

    """A batch size larger than the prompt list silently becomes one small batch."""
    failures = batched_generation_failures(
        _batch_record(singles = ["a", "b"], prompt_token_lengths = [5, 9], distinct_lengths = 2)
    )
    assert any("never actually formed" in f for f in failures), failures


def test_a_missing_batched_record_is_a_failure_not_a_pass():
    from run_t4_smoke import batched_generation_failures
    """A leg where the check never ran must not look like a leg where it passed."""
    assert batched_generation_failures(None) == ["batched generation was never run"]


# ------------------------------------------------- run_grpo_t4.py: the reward


def _completions(*lengths):
    """GRPO hands reward functions a list of message lists."""
    return [[{"role": "assistant", "content": "x" * n}] for n in lengths]


def test_the_length_reward_still_discriminates_at_the_lengths_the_model_emits():
    """The leg's only instrument, and it was broken in the least visible way.

    `reward_length` was `min(len(t), 200) / 200.0` while its docstring claimed
    to be "SENSITIVE to a group's diversity". Kernels
    unsloth-probe-grpo-rep2-b03be8 and -rep3-bc3828 recorded completions of
    2534 to 3396 characters, so every completion scored exactly 1.0, every
    group tied, and the leg failed with `reward_std was zero on every step`.
    Two runs in three died on it.

    A reward that saturates below the range the model actually occupies reads
    as a broken generation path in the report. So assert discrimination at the
    OBSERVED lengths, not at convenient small ones: the old function passes any
    test written with 10- and 20-character completions.
    """
    sys.path.insert(0, str(SMOKE_DIR))
    from run_grpo_t4 import reward_length

    for a, b in ((2534, 3396), (3013, 3328), (2738, 2633)):
        scores = reward_length(_completions(a, b))
        assert scores[0] != scores[1], (
            f"lengths {a} and {b} scored identically ({scores}); the reward has "
            f"saturated in the range the model actually produces"
        )


def test_the_length_reward_is_strictly_increasing_and_never_saturates():
    """The property that makes the test above hold for lengths nobody has seen
    yet. A cap is a length at which this stops being true, which is exactly how
    the previous version passed review."""
    from run_grpo_t4 import reward_length

    lengths = [0, 1, 50, 200, 1000, 3000, 10000, 100000]
    scores = reward_length(_completions(*lengths))
    assert all(x < y for x, y in zip(scores, scores[1:])), scores
    assert all(0.0 <= s < 1.0 for s in scores), scores


def test_two_identical_completions_still_tie_because_that_is_real():
    """Guard against overcorrecting. A group whose completions are genuinely
    identical SHOULD tie -- that is degenerate generation and the leg is right
    to fail on it. The bug was ties between DIFFERENT completions."""
    from run_grpo_t4 import reward_length

    scores = reward_length(_completions(2534, 2534))
    assert scores[0] == scores[1]


# --------------------------------------------------- gguf_export.py


def _gguf_record(**over):
    """A healthy export, shaped like the one measured on
    unsloth-probe-gguf-q8-peft-920e3e: the GGUF in the SIBLING directory."""
    record = {
        "save_dir": "/tmp/q8p",
        "requested_quantization": "q8_0",
        "ok": True,
        "seconds": 40.6,
        "ggufs": [
            {
                "path": "/tmp/q8p_gguf/qwen3-0.6b.Q8_0.gguf",
                "mb": 609.8,
                "found_in": "/tmp/q8p_gguf",
                "suffix": "_gguf",
            }
        ],
    }
    record.update(over)
    return record


def test_a_healthy_gguf_export_reports_no_failures():
    from gguf_export import export_failures
    assert export_failures(_gguf_record(), accept_quantizations = ("q8_0",)) == []


def test_an_export_that_reported_ok_but_wrote_no_gguf_is_a_failure():
    """The trap this module exists for. save_pretrained_gguf writes the merged
    safetensors into the directory it was given and the GGUF into a SIBLING, so
    code that globs the directory it passed finds nothing, raises nothing, and
    calls it a successful export."""
    from gguf_export import export_failures

    failures = export_failures(_gguf_record(ggufs = []), accept_quantizations = ("q8_0",))
    assert failures and "no .gguf" in failures[0]
    assert "_gguf sibling" in failures[0]


def test_a_gguf_search_finds_the_file_in_the_sibling_directory(tmp_path):
    """Executed against a real directory pair rather than a hand-written dict,
    because the whole point is WHERE the file is."""
    from gguf_export import find_ggufs

    save = tmp_path / "out"
    save.mkdir()
    (save / "model.safetensors").write_bytes(b"0" * 2048)
    sibling = tmp_path / "out_gguf"
    sibling.mkdir()
    (sibling / "qwen3-0.6b.Q8_0.gguf").write_bytes(b"0" * 4096)

    found = find_ggufs(str(save))
    assert len(found) == 1, found
    assert found[0]["suffix"] == "_gguf"
    assert found[0]["path"].endswith("qwen3-0.6b.Q8_0.gguf")


def test_a_gguf_that_is_only_a_header_is_not_an_export():
    from gguf_export import export_failures
    failures = export_failures(
        _gguf_record(
            ggufs = [
                {
                    "path": "/tmp/q8p_gguf/x.Q8_0.gguf",
                    "mb": 0.1,
                    "found_in": "/tmp/q8p_gguf",
                    "suffix": "_gguf",
                }
            ]
        ),
        accept_quantizations = ("q8_0",),
    )
    assert failures and "header and no weights" in failures[0]


def test_a_model_allowed_to_override_the_quantization_still_passes():
    """gpt-oss answers q8_0 with "GPT-OSS does not support GGUF quantization
    (requested: q8_0). Overriding to MXFP4 format." That is documented
    behaviour, so a leg that accepts MXFP4 must not fail on it."""
    from gguf_export import export_failures

    record = _gguf_record(
        ggufs = [
            {
                "path": "/tmp/g_gguf/gpt-oss-20b.MXFP4.gguf",
                "mb": 11800.0,
                "found_in": "/tmp/g_gguf",
                "suffix": "_gguf",
            }
        ]
    )
    assert export_failures(record, accept_quantizations = ("mxfp4",)) == []
    # ... and a leg that does NOT accept it still says so.
    failures = export_failures(record, accept_quantizations = ("q8_0",))
    assert failures and "accepted quantization" in failures[0]


def test_a_gguf_that_no_runner_could_execute_is_a_failure():
    from gguf_export import run_failures
    failures = run_failures(
        {
            "gguf": "/tmp/q8p_gguf/x.gguf",
            "bench": {"seconds": 240.0, "error": "TimeoutExpired: ..."},
            "completion": {"seconds": 240.0, "returncode": 1, "stderr": "bad magic"},
        }
    )
    assert failures and "produced no output from any runner" in failures[0]


def test_one_successful_runner_is_enough():
    from gguf_export import run_failures
    assert (
        run_failures(
            {
                "gguf": "/tmp/q8p_gguf/x.gguf",
                "bench": {"seconds": 12.0, "returncode": 0, "stdout": "tg128 ... 41.2"},
            }
        )
        == []
    )


def test_a_bundle_with_no_runners_at_all_is_reported_as_that():
    """Distinct from "the file does not run": a missing binary is a bundle
    problem and blaming the model file for it would send the reader to the
    wrong place."""
    from gguf_export import run_failures

    failures = run_failures(
        {
            "gguf": "/tmp/x.gguf",
            "bench": {"skipped": "no llama-bench in the bundle"},
            "completion": {"skipped": "no llama-completion in the bundle"},
        }
    )
    assert failures and "missing from the llama.cpp bundle" in failures[0]


def test_the_llama_cpp_facts_read_a_tuple_not_a_directory(tmp_path):
    """install_llama_cpp returns (llama-quantize, convert_hf_to_gguf.py). An
    earlier probe treated the return value as a bin directory and reported
    "0 binaries", which was the probe being wrong, not the bundle being
    empty."""
    from gguf_export import llama_cpp_facts

    quant = tmp_path / "llama-quantize"
    conv = tmp_path / "convert_hf_to_gguf.py"
    quant.write_text("")
    conv.write_text("")

    facts = llama_cpp_facts(
        "Unsloth: Installing prebuilt llama.cpp b10472-mix-4b653db "
        "(app-b10472-mix-4b653db-linux-x64-cpu.tar.gz) - skipping compilation.",
        (str(quant), str(conv)),
    )
    assert facts["all_exist"] is True
    assert facts["dir"] == str(tmp_path)
    assert facts["prebuilt"] is True
    assert facts["source_build_markers"] == []


def test_a_source_build_is_visible_even_though_it_succeeds():
    """The failure to catch is not "kernel missing" but "kernel built from
    source": silent, correct, and many minutes on 4 vCPUs."""
    from gguf_export import llama_cpp_facts

    facts = llama_cpp_facts("-- Configuring done\ncmake --build . -j 4\n", ())
    assert facts["prebuilt"] is False
    assert "cmake" in facts["source_build_markers"]


# ------------------------------- run_t4_smoke.py: parent -> child argv


# Options the PARENT alone acts on, so their absence from the child command is
# correct rather than a leak. Each is here for a stated reason; an entry added
# without one is how this guard stops working.
PARENT_ONLY_DESTS = {
    "outdir",  # the parent gives each cycle its own subdirectory
    "cycle",  # set by the parent per child, never forwarded verbatim
    "repeat",  # how many children to launch
    "reference",  # band check runs in the parent, over collected cycles
    "rel_tol",  # ... and its tolerances
    "abs_floor",
    "require_canary",  # evaluated by the parent's failure collector
    "check_batched_generation",
    "export_gguf",  # forwarded as a bare flag, asserted separately below
    # The pin check reads the report the cycles produced, in the parent
    # (run_t4_smoke.py:1741), so the children have nothing to do with it.
    "pins",
    # The plain-TRL control arm is spawned BY the parent, after the cycles, and
    # ruled on there. A cycle child neither runs it nor judges it.
    "compare_naive_trl",
    "control_oom_is_ok",
    # Kernel provenance IS collected in the child (it needs the loaded model),
    # but the flag reaches it through the bare-flag block rather than the
    # name/value pairs this check walks; asserted separately in
    # test_kernel_provenance.py.
    "kernel_provenance",
    # The vision run is spawned BY the parent, after the cycles and in a
    # process of its own: it loads a second model, and two 4bit models resident
    # at once on a 14.56GB T4 is how a leg becomes an OOM blamed on the thing
    # it was testing.
    "vision_run",
}


def _smoke_source() -> str:
    return (SMOKE_DIR / "run_t4_smoke.py").read_text(encoding = "utf-8")


def _child_command_block() -> str:
    """The argv the parent builds for each cycle."""
    source = _smoke_source()
    start = source.index("cmd = [\n            sys.executable,")
    end = source.index("proc = subprocess.run(cmd)", start)
    return source[start:end]


def test_every_option_the_child_needs_actually_reaches_the_child():
    """The class of bug, not one instance of it.

    Cycles run in fresh child processes and the parent rebuilds their argv from
    an explicit list. A flag added to the parser but not to that list is
    accepted on the command line, parsed, logged in the driver's exec line, and
    silently ignored -- which is exactly what happened to --export-gguf on
    kernel unsloth-probe-default-gguf-637565: the leg failed with "GGUF export
    was never run" while the driver log showed --export-gguf right there in the
    command.

    --check-batched-generation escaped this only because it defaults to True, so
    the child got it without being told. That is luck, not design.
    """
    import argparse
    import importlib

    sys.path.insert(0, str(SMOKE_DIR))
    module = importlib.import_module("run_t4_smoke")

    # Build the parser the same way main() does, by calling it with a sentinel
    # that makes it return rather than run.
    parser = argparse.ArgumentParser()
    source = _smoke_source()
    dests = set(re.findall(r'dest\s*=\s*"([a-z_0-9]+)"', source))
    dests |= {
        m.replace("-", "_") for m in re.findall(r'ap\.add_argument\(\s*"--([a-z0-9-]+)"', source)
    }
    assert "export_gguf" in dests, "the parser no longer defines --export-gguf"
    assert "model" in dests, "the dest scrape found nothing; fix the scrape"

    block = _child_command_block()
    forwarded = {m.replace("-", "_") for m in re.findall(r'"--([a-z0-9-]+)"', block)}

    missing = sorted(d for d in dests - forwarded - PARENT_ONLY_DESTS if not d.startswith("no_"))
    assert not missing, (
        f"these options are parsed but never forwarded to the cycle child, so "
        f"setting them does nothing: {missing}. Either forward them or list "
        f"them in PARENT_ONLY_DESTS with a reason."
    )
    del parser, module


def test_the_export_flag_is_forwarded_as_a_bare_flag():
    """store_true options cannot ride the (flag, value) loop -- `--export-gguf
    True` is not a thing -- so they need their own append, and that is the line
    that was missing."""
    block = _child_command_block()
    assert 'cmd.append("--export-gguf")' in block
    assert "if args.export_gguf:" in block


def test_the_export_settings_ride_the_value_loop():
    block = _child_command_block()
    assert '("--gguf-quantization", args.gguf_quantization)' in block
    assert '("--gguf-accept", args.gguf_accept)' in block


def test_a_second_cycle_cannot_report_an_already_installed_llama_cpp_as_a_source_build():
    """Measured on unsloth-probe-visleg-full-b3a317, and it is a trap.

    The prebuilt banner is printed once, by the install that downloads the
    bundle. A second cycle in the same session finds llama.cpp already there
    and prints nothing, so the field read `prebuilt: true` on cycle 0 and
    `prebuilt: false` on cycle 1 for the SAME installation -- and `false` reads
    as "built from source", which is the one thing this field exists to catch.

    None is the third state: this run did not install it, so it cannot say.
    """
    from gguf_export import llama_cpp_facts

    quiet = llama_cpp_facts("", ())
    assert (
        quiet["prebuilt"] is None
    ), "an install that printed nothing must not claim a source build"
    assert quiet["source_build_markers"] == []

    # And the two real answers are unchanged.
    assert (
        llama_cpp_facts(
            "Unsloth: Installing prebuilt llama.cpp b10472 - skipping compilation.", ()
        )["prebuilt"]
        is True
    )
    assert llama_cpp_facts("cmake --build . -j 4\n", ())["prebuilt"] is False


def test_the_text_leg_gguf_export_does_not_land_in_the_artifact_directory():
    """`/kaggle/working` is 21.0 GB and is what `kernels output` ships back.

    Measured on unsloth-probe-lcleg-final-a90fbb, which is the run that found
    this:

        RuntimeError: Unsloth: Not enough disk space to convert to GGUF.
        The export needs about 16.6GB on the filesystem holding
        `/kaggle/working/t4_out_Latest_compile/cycle0/gguf_run0`

    Two failures in one. A merge too big for that volume kills the leg, and a
    merge that DOES fit is downloaded as part of the artifact, which nobody
    wanted. gpt-oss and the vision run were both moved to a tempdir for exactly
    this reason and this path was missed -- so a small model kept passing and
    hid it.
    """
    src = _smoke_source()
    call = src[src.index("gguf_export_record = export_gguf(") :]
    call = call[: call.index(")")]
    assert "tempfile.mkdtemp" in call, "the export writes into the artifact directory"
    assert (
        "args.outdir" not in call
    ), "args.outdir is /kaggle/working, which is 21 GB and is collected"
