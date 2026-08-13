# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""gpt-oss-20b LoRA on a single T4: does the compile-and-offload path hold?

This is the payload behind the `gptoss` leg of the Kaggle T4 notebook CI,
and it was written first as a FEASIBILITY PROBE. The question it exists to
answer is not "is the loss right" -- it is "can a 20B checkpoint be loaded,
LoRA-trained and generated from at all on 16GB of sm_75, with the compiled
float32 path this card forces".

What the probe found, so the report can be read against it
-----------------------------------------------------------
Kernels `unsloth-t4-ci-8161ceb9` and `unsloth-t4-ci-7ab727f1`, 2026-08-11,
Tesla T4 / sm_75 / 14.56 GB. It works, and three of the four things that
looked most likely to break turned out not to be in the path at all.

* **MXFP4 is never reached.** `unsloth/gpt-oss-20b` is an MXFP4 checkpoint
  and MXFP4 has no backward pass in unsloth_zoo at all, but
  `load_in_4bit=True` makes Unsloth's FLOAT_TO_INT_MAPPER redirect the load
  to `unsloth/gpt-oss-20b-unsloth-bnb-4bit`, which is NF4. The probe
  confirmed the redirect from `model.config._name_or_path`, and that is
  recorded on every run: a change to that mapping would move this leg onto
  a checkpoint that cannot train, and nothing else here would notice.
* **No bf16, and Unsloth already knows.** gpt-oss is in `FORCE_FLOAT32`.
  The probe saw `UNSLOTH_FORCE_FLOAT32=1`, `fp16=False`, `bf16=False`, and
  `UNSLOTH_FORCE_CUSTOM_DTYPE` pinning `down_projs` and `mlp.router` to
  float32. That is the path this leg exists to keep working; it exists for
  this card and nothing else in CI exercises it.
* **No offload.** 12.78 GB reserved of 14.56, every parameter on `cuda:0`,
  no `hf_device_map`. It fits, with about 1.8 GB of headroom, which is thin
  enough that placement is still counted on every run rather than assumed.
* **torch.compile engages**: 32 unique graphs, 779 calls captured, 2 graph
  breaks, both of them `_warnings.warn`. A silent fall back to eager would
  leave every other number in this report looking healthy while the thing
  the leg covers was not exercised, so this is asserted, not just recorded.

What it asserts
---------------
1. The model loads, and the report says in what dtype and across which
   devices.
2. Training runs for the requested number of steps, every logged loss is
   finite, and at least one logged `grad_norm` is finite and non-zero -- a
   run whose every gradient was zero produces healthy numbers everywhere
   else and trained nothing. There is no committed reference band here: the
   run is too short and the model too large for a per-step trace to be worth
   capturing, and a band nobody can recapture cheaply is a check that gets
   disabled the first time it is inconvenient.
2a. The forced-float32 path was the path taken. On a card without bf16,
   `fp16`/`bf16` must both be off and `UNSLOTH_FORCE_FLOAT32` must be set.
   This is the coverage the leg uniquely claims, and it was recorded on
   every run and asserted on none.
3. `torch.compile` captured at least one graph DURING TRAINING
   (`--require-compile`, on by default). The Dynamo counters are
   process-global and loading a 20B checkpoint fills them, so the assertion
   is on the delta across `trainer.train()`, not on the total.
4. Generation after training returns non-empty text without raising. This is
   the assertion that catches a training run which "succeeds" and leaves the
   model unusable, which on a quantised offloaded path is a real outcome.

`--probe` records every one of those and fails on none of them. That mode is
for the deliberate one-off feasibility runs: a probe whose job is to find
out whether the payload is viable must come back with evidence, not with a
nonzero exit and a truncated report.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from versions import (  # noqa: E402
    GOAL_PACKAGES,
    flatten_versions,
    resolved_versions,
)

CANARY = "__UNSLOTH__!!!"
SEED = 3407

DEFAULT_MODEL = "unsloth/gpt-oss-20b"


def _log(msg: str) -> None:
    print(f"[gptoss-t4] {msg}", flush = True)


def compile_counters(before: dict | None = None) -> dict:
    """What `torch.compile` actually did, from Dynamo's own bookkeeping.

    `unique_graphs` is the number that decides whether compilation engaged:
    zero means every region fell back to eager, whatever the banner said.
    `graph_breaks` is kept beside it because a run that captured graphs and
    broke a hundred times is a different, and reportable, state from one
    that captured cleanly -- on a card with no bf16 a break is often the
    first visible symptom of a dtype the compiled path refused.

    These counters are process-global and never reset, and loading a model
    through Unsloth compiles plenty before `trainer.train()` is called. So
    the absolute number cannot answer "did TRAINING compile" -- a training
    path that fell back to eager entirely still leaves the loader's graphs
    standing. Pass the reading taken before training as ``before`` and the
    delta is the answer; `failures_for` asserts on the delta.
    """
    state: dict = {"available": False}
    try:
        import torch._dynamo.utils as dynamo_utils

        counters = dynamo_utils.counters
        stats = dict(counters.get("stats", {}))
        breaks = dict(counters.get("graph_break", {}))
        state = {
            "available": True,
            "unique_graphs": int(stats.get("unique_graphs", 0)),
            "calls_captured": int(stats.get("calls_captured", 0)),
            "graph_breaks_total": sum(int(v) for v in breaks.values()),
            # Truncated: the reasons are free text and a pathological run
            # produces hundreds of distinct ones.
            "graph_break_reasons": sorted(breaks)[:10],
        }
    except Exception as exc:  # noqa: BLE001
        state = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if before and before.get("available") and state.get("available"):
        for key in ("unique_graphs", "calls_captured", "graph_breaks_total"):
            state[f"{key}_delta"] = state[key] - int(before.get(key, 0))
        state["before"] = {
            k: before.get(k) for k in ("unique_graphs", "calls_captured", "graph_breaks_total")
        }
    return state


def placement(model) -> dict:
    """Where the weights ended up, counted rather than trusted.

    A 20B checkpoint on a 16GB card either offloads or does not fit, and
    "did it offload" is answerable only by looking. `hf_device_map` is the
    accelerate-side answer and is absent when nothing dispatched; the
    parameter walk is the answer that is always available, and it is the one
    that distinguishes a model that quietly landed on the CPU (correct, slow)
    from one that landed on meta (loaded nothing at all).
    """
    counts: dict = {}
    try:
        for param in model.parameters():
            key = str(param.device)
            counts[key] = counts.get(key, 0) + param.numel()
    except Exception as exc:  # noqa: BLE001
        counts = {"error": f"{type(exc).__name__}: {exc}"}
    device_map = getattr(model, "hf_device_map", None)
    return {
        "parameters_by_device": counts,
        "hf_device_map_devices": (
            sorted({str(v) for v in device_map.values()}) if isinstance(device_map, dict) else None
        ),
        "offloaded": bool(
            isinstance(device_map, dict)
            and any(str(v) in ("cpu", "disk") for v in device_map.values())
        ),
    }


def memory() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    props = torch.cuda.get_device_properties(0)
    return {
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "total_gb": round(props.total_memory / 1024**3, 2),
    }


def build_dataset(tokenizer, rows: list[dict]):
    """The canary rows as chat turns, through the model's own template.

    Going through `apply_chat_template` rather than hand-rolling a prompt is
    deliberate: gpt-oss has a template with channels and a reasoning-effort
    knob, and a payload that bypassed it would exercise a text format no user
    of this notebook ever produces.
    """
    from datasets import Dataset

    texts = []
    for row in rows:
        texts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ],
                tokenize = False,
                add_generation_prompt = False,
            )
        )
    return Dataset.from_dict({"text": texts})


def train_and_infer(args) -> dict:
    import torch
    from unsloth import FastLanguageModel

    result: dict = {}

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        dtype = None,  # let the loader choose; T4 has no bf16
        max_seq_length = args.max_seq_length,
        load_in_4bit = True,
        full_finetuning = False,
    )
    result["load_seconds"] = round(time.time() - t0, 1)
    result["model_dtype"] = str(getattr(model, "dtype", None))
    # What was REALLY loaded. `unsloth/gpt-oss-20b` is an MXFP4 checkpoint,
    # and MXFP4 has no backward pass at all (unsloth_zoo raises
    # "Backwards pass using MXFP4 is still under construction"). Asking for
    # load_in_4bit=True makes Unsloth's FLOAT_TO_INT_MAPPER redirect the
    # request to `unsloth/gpt-oss-20b-unsloth-bnb-4bit`, an NF4 checkpoint
    # that does train. So the name in the config is not the name that was
    # asked for, and a change to that redirect would silently move this leg
    # onto a path that cannot train. Record it and let the report show it.
    model_config = getattr(model, "config", None)
    result["resolved_checkpoint"] = getattr(model_config, "_name_or_path", None)
    quant = getattr(model_config, "quantization_config", None)
    result["quantization"] = (
        {
            "method": str(getattr(quant, "quant_method", None)),
            "type": str(getattr(quant, "bnb_4bit_quant_type", None)),
        }
        if quant is not None
        else None
    )
    result["placement_after_load"] = placement(model)
    result["memory_after_load"] = memory()
    _log(f"loaded in {result['load_seconds']}s, dtype {result['model_dtype']}")

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha = args.lora_r * 2,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = SEED,
    )

    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding = "utf-8").splitlines()
        if line.strip()
    ]
    dataset = build_dataset(tokenizer, rows)

    from trl import SFTConfig, SFTTrainer

    config = SFTConfig(
        output_dir = str(Path(args.outdir) / "trainer"),
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        max_length = args.max_seq_length,
        max_steps = args.max_steps,
        learning_rate = 2e-4,
        warmup_steps = 0,
        lr_scheduler_type = "constant",
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        seed = SEED,
        data_seed = SEED,
        # fp16/bf16 are deliberately NOT set here, and that is the opposite
        # of what the tiny SFT payload does. gpt-oss is in Unsloth's
        # FORCE_FLOAT32 list: on a card without bf16 the loader sets
        # UNSLOTH_FORCE_FLOAT32=1 and the RL/SFT patch switches the run to
        # float32 ("Unsloth: Switching to float32 training since model cannot
        # work with float16"), because fp16 autocast through the MXFP4-derived
        # weights produces infinities. Asking for fp16 here would fight that
        # patch, and whichever won, the run would no longer be the one a
        # notebook user gets. What was actually chosen is recorded below.
        dataloader_num_workers = 0,
        dataloader_pin_memory = False,
        report_to = "none",
        save_strategy = "no",
    )
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = config,
    )

    # Which precision the run ended up in, after Unsloth's patches have had
    # their say. On a T4 this is expected to be float32 and NOT fp16; if it
    # ever reads fp16 here, the FORCE_FLOAT32 path stopped firing and the
    # infinities it exists to prevent are back.
    result["precision"] = {
        "fp16": bool(getattr(trainer.args, "fp16", None)),
        "bf16": bool(getattr(trainer.args, "bf16", None)),
        "force_float32_env": os.environ.get("UNSLOTH_FORCE_FLOAT32"),
        "custom_dtype_env": os.environ.get("UNSLOTH_FORCE_CUSTOM_DTYPE"),
    }
    _log(f"precision {json.dumps(result['precision'])}")

    # The counters as they stand BEFORE training, so what training itself
    # compiled is a subtraction rather than an inference. Loading a 20B
    # checkpoint through Unsloth compiles a great deal, and every one of
    # those graphs sits in the same process-global counter.
    compile_before = compile_counters()
    _log(f"compile counters before training: {json.dumps(compile_before)}")

    t0 = time.time()
    stats = trainer.train()
    result["train_seconds"] = round(time.time() - t0, 1)
    result["metrics"] = [
        {"step": entry.get("step"), "loss": entry.get("loss"), "grad_norm": entry.get("grad_norm")}
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    result["train_metrics"] = {k: v for k, v in (stats.metrics or {}).items()}
    result["compile"] = compile_counters(before = compile_before)
    result["memory_after_train"] = memory()
    _log(
        f"trained {len(result['metrics'])} logged steps in "
        f"{result['train_seconds']}s; compile {result['compile']}"
    )

    # Inference on the trained model. The notebook's own shape: chat
    # template with a reasoning effort, greedy decode, short.
    FastLanguageModel.for_inference(model)
    try:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": rows[0]["question"]}],
            add_generation_prompt = True,
            return_tensors = "pt",
            return_dict = True,
            reasoning_effort = "low",
        ).to("cuda")
    except TypeError:
        # reasoning_effort is a gpt-oss template keyword. A template that
        # does not take it is a finding worth recording, not a crash.
        result["reasoning_effort_supported"] = False
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": rows[0]["question"]}],
            add_generation_prompt = True,
            return_tensors = "pt",
            return_dict = True,
        ).to("cuda")
    else:
        result["reasoning_effort_supported"] = True

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens = args.max_new_tokens,
            do_sample = False,
            temperature = None,
            top_p = None,
            top_k = None,
            use_cache = True,
        )
    result["infer_seconds"] = round(time.time() - t0, 1)
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens = True)
    result["generated"] = generated
    result["canary_found"] = CANARY in generated
    result["memory_peak"] = memory()
    _log(f"generated {generated!r}")
    return result


def _is_finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def failures_for(result: dict, args) -> list[str]:
    """The assertions, separated from the run so they can be unit-tested.

    Nothing here needs a GPU, which is the point: the pass/fail rule for a
    leg that costs a Kaggle session has to be checkable without one.
    """
    failures: list[str] = []
    metrics = result.get("metrics") or []
    if len(metrics) != args.max_steps:
        failures.append(f"expected {args.max_steps} logged steps, got " f"{len(metrics)}")
    losses = [m.get("loss") for m in metrics if m.get("loss") is not None]
    if not losses:
        failures.append("no loss was logged at all, so nothing trained")
    bad = [l for l in losses if l != l or l in (float("inf"), float("-inf"))]
    if bad:
        failures.append(f"non-finite loss: {losses}")

    # Did the optimizer apply anything? Every number above stays healthy on a
    # run whose gradients are all zero -- the loss is finite, compilation
    # engaged, and the untrained base model still generates text -- so a leg
    # that claims to cover LoRA training on this path has to look. Only
    # decidable where grad_norm was logged at all, exactly as in the SFT
    # leg: a trainer that stopped logging it says nothing either way, and
    # inferring "nothing applied" from silence would be a failure this check
    # invented rather than found.
    norms = [m.get("grad_norm") for m in metrics if m.get("grad_norm") is not None]
    applied = [g for g in norms if _is_finite(g) and float(g) != 0.0]
    if norms and not applied:
        failures.append(
            f"no optimizer update was applied: every logged grad_norm is zero or "
            f"non-finite ({norms}), so the adapter is the adapter it started with "
            f"and this leg measured a forward pass rather than LoRA training"
        )

    # The float32 path, which is the coverage this leg uniquely claims. It is
    # recorded on every run and was asserted on none: a run that quietly went
    # through fp16 instead logs finite losses, compiles and generates, and
    # reports green while the path this leg exists for was never exercised.
    #
    # Conditioned on the card, not hardcoded to T4. FORCE_FLOAT32 exists
    # because this hardware has no bf16; on a card that has it, the patch not
    # firing is correct, and a red there would be this check's own bug.
    environment = result.get("environment") or {}
    if environment.get("bf16_supported") is False:
        precision = result.get("precision")
        if not precision:
            failures.append(
                "the training precision was never recorded, so whether the forced "
                "float32 path fired could not be established"
            )
        else:
            if precision.get("fp16") or precision.get("bf16"):
                failures.append(
                    f"the run trained in reduced precision on a card without bf16: "
                    f"{precision}. gpt-oss is in Unsloth's FORCE_FLOAT32 list because "
                    f"fp16 autocast through these weights produces infinities, so this "
                    f"is the patch having stopped firing, not a slow pass."
                )
            elif not precision.get("force_float32_env"):
                failures.append(
                    f"UNSLOTH_FORCE_FLOAT32 was not set: {precision}. The float32 path "
                    f"is the only thing this leg covers that nothing else in CI does."
                )

    if args.require_compile:
        compiled = result.get("compile") or {}
        # The DELTA across training, not the process-global total. The total
        # is nonzero the moment the loader has compiled anything, so a
        # training path that fell back to eager entirely used to satisfy it.
        graphs = compiled.get("unique_graphs_delta", compiled.get("unique_graphs", 0))
        if not compiled.get("available"):
            failures.append(
                "torch._dynamo counters were unreadable, so whether "
                "torch.compile engaged could not be established: "
                f"{compiled.get('error')}"
            )
        elif graphs < 1:
            failures.append(
                "torch.compile captured zero graphs across training itself, so the "
                "training path ran eager whatever was compiled at load time. That is "
                "the silent fallback this leg exists to catch; it is a failure, not a "
                f"slow pass. Counters: {compiled}"
            )

    generated = result.get("generated")
    if generated is None:
        failures.append("generation did not run")
    elif not generated.strip():
        failures.append(
            "generation after training returned empty text, so the trained model is unusable"
        )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = DEFAULT_MODEL)
    ap.add_argument("--dataset", default = str(_HERE / "canary_dataset.jsonl"))
    ap.add_argument("--outdir", required = True)
    ap.add_argument("--label", default = "gptoss")
    ap.add_argument("--max-steps", type = int, default = 3)
    ap.add_argument("--max-seq-length", type = int, default = 1024)
    ap.add_argument("--lora-r", type = int, default = 8)
    ap.add_argument("--max-new-tokens", type = int, default = 32)
    ap.add_argument("--require-compile", dest = "require_compile", action = "store_true", default = True)
    ap.add_argument("--no-require-compile", dest = "require_compile", action = "store_false")
    ap.add_argument(
        "--probe",
        action = "store_true",
        help = "record everything, assert nothing. For the "
        "one-off feasibility runs, whose job is to come "
        "back with evidence rather than an exit code",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)

    report: dict = {
        "label": args.label,
        "model": args.model,
        "leg": "gptoss",
        "probe": args.probe,
        "config": {
            k: getattr(args, k)
            for k in ("max_steps", "max_seq_length", "lora_r", "max_new_tokens", "require_compile")
        },
        "versions": {},
        "environment": {},
        "failures": [],
    }

    # Versions first, and before anything can crash. A payload that died in
    # the loader still has to say which library set it died with, or the
    # crash is unattributable and the session was spent for nothing.
    report["versions"] = resolved_versions(
        GOAL_PACKAGES, import_check = ("torch", "transformers", "trl")
    )
    report["versions_flat"] = flatten_versions(report["versions"])
    _log("versions " + json.dumps(report["versions_flat"]))

    try:
        import torch
        props = torch.cuda.get_device_properties(0)
        report["environment"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": props.name,
            "gpu_capability": f"sm_{props.major}{props.minor}",
            "gpu_total_gb": round(props.total_memory / 1024**3, 1),
            "gpu_count_visible": torch.cuda.device_count(),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "compile_disabled_env": os.environ.get("UNSLOTH_COMPILE_DISABLE"),
        }
    except Exception as exc:  # noqa: BLE001
        report["environment"] = {"error": f"{type(exc).__name__}: {exc}"}

    failures: list[str] = []
    try:
        result = train_and_infer(args)
        report.update(result)
        # The metrics key the launcher and report renderer already know how
        # to display, so this leg needs no special case downstream.
        report["metrics"] = result.get("metrics", [])
        failures = failures_for(report, args)
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        report["traceback"] = traceback.format_exc()[-6000:]
        failures = [f"{type(exc).__name__}: {str(exc)[:600]}"]
        _log("EXCEPTION\n" + report["traceback"])
        report["memory_peak"] = memory()
        report["compile"] = compile_counters()

    report["observed_failures"] = failures
    if args.probe:
        # A probe reports; it does not judge. The verdict is read off
        # observed_failures by a human, and the leg is not wired into CI
        # until that reading says it can be.
        report["failures"] = []
        report["passed"] = True
    else:
        report["failures"] = failures
        report["passed"] = not failures

    (outdir / "t4_smoke_report.json").write_text(
        json.dumps(report, indent = 2, default = str), encoding = "utf-8"
    )
    print("T4_SMOKE_REPORT " + json.dumps(report, default = str), flush = True)
    for entry in failures:
        _log(f"OBSERVED FAILURE: {entry}")
    _log("T4_SMOKE_RESULT " + ("PASS" if report["passed"] else "FAIL"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
