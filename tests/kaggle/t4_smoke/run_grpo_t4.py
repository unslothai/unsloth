# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Qwen3-4B GRPO with vLLM on a single T4: does the engine fit and generate?

The payload behind the `grpo` leg of the Kaggle T4 notebook CI, written
first as a FEASIBILITY PROBE. Two things are genuinely in doubt on this
hardware and this file is built to answer both with evidence.

**Does vLLM run on sm_75 at all, at the version installed?** vLLM selects an
attention backend by compute capability. Turing has no FlashAttention, so it
depends on the xformers backend, and that backend has not survived every
release. A version that dropped it does not fail at install time and does not
fail at import time: it fails when the engine is constructed, deep inside
platform selection, which is why the resolved vLLM version, the selected
backend and the engine-construction outcome are all recorded separately.

**Does it fit?** The notebook this leg comes from sets `load_in_4bit=False`,
which is roughly 8GB of 16-bit weights, and then asks a vLLM engine with
`gpu_memory_utilization=0.9` and a LoRA training loop to share the same 16GB
card. `--load-in-4bit` is therefore a first-class switch here rather than a
constant, and the probe's job is to say which setting the card actually
tolerates. Peak reserved and allocated memory are reported for whichever was
used.

What it asserts, and what it deliberately does not
--------------------------------------------------
**Not the loss.** With `num_iterations=1` and `beta=0.0` the TRL GRPO
objective is zero by construction on a healthy run: the policy that
generated the completions is the policy being updated, so the importance
ratio is exactly 1, and with no KL term the whole loss cancels. A check on
loss here would either always pass or fire on arithmetic noise. It is
recorded and never asserted, and that is not an oversight.

**Reward, reward_std and the completions instead.**

* `reward` must be logged and finite on every step. Absent means the reward
  functions never ran, which means generation produced nothing.
* `reward_std` must be non-zero on at least one step. Zero across a whole
  group is the real bug signal on this path: it means every completion in
  the group was scored identically, which in practice means every completion
  was IDENTICAL -- a sampler that ignored its temperature, a seed applied
  per-completion instead of per-group, or an engine returning the same
  cached text N times. The gradient is exactly zero in that state, so
  training "succeeds" while learning nothing, and no other number here moves.
* At least one completion must be non-empty, and the completions actually
  seen are captured and reported. An engine that returns N empty strings
  scores them all the same, so the reward checks alone would call that a
  clean run.

`--probe` records everything and asserts nothing, for the deliberate one-off
feasibility runs.
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

SEED = 3407
DEFAULT_MODEL = "unsloth/Qwen3-4B-Base"

SYSTEM_PROMPT = "You are given a question. Answer it as briefly as you can."

# Every completion the reward functions were shown, in order. A module global
# rather than a closure because TRL calls the reward functions itself and
# hands back nothing but the scores; without capturing here, the only record
# of what the engine produced would be the scores it happened to earn.
SEEN_COMPLETIONS: list[list[str]] = []


def _log(msg: str) -> None:
    print(f"[grpo-t4] {msg}", flush = True)


def _texts(completions) -> list[str]:
    """TRL hands completions back as plain strings or as chat turns."""
    out = []
    for completion in completions:
        if isinstance(completion, str):
            out.append(completion)
        elif isinstance(completion, list) and completion:
            out.append(str(completion[0].get("content", "")))
        else:
            out.append("")
    return out


def reward_length(completions, **kwargs) -> list[float]:
    """Longer completions score higher, saturating at 200 characters.

    Chosen because it is deterministic given the text and yet SENSITIVE to
    the diversity of a group. A constant reward would make `reward_std` zero
    on a perfectly healthy run and destroy the only instrument this leg has.
    """
    texts = _texts(completions)
    SEEN_COMPLETIONS.append(texts)
    return [min(len(t), 200) / 200.0 for t in texts]


def reward_digit(completions, **kwargs) -> list[float]:
    """A second, differently shaped signal, so `reward` is not one function.

    A single reward function that happened to be broken would look exactly
    like a broken generation path. Two disagreeing sources make that
    distinguishable in the report.
    """
    return [1.0 if any(c.isdigit() for c in t) else 0.0 for t in _texts(completions)]


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


def vllm_facts() -> dict:
    """Which vLLM, and which attention backend it would choose here.

    Recorded BEFORE the engine is built, so a payload that dies constructing
    the engine still says what it was trying to construct. The backend is
    read from vLLM's own selector where that is reachable and from the
    environment override otherwise; both are reported, because an override
    silently deciding the answer is itself worth seeing.
    """
    facts: dict = {"env_override": os.environ.get("VLLM_ATTENTION_BACKEND")}
    try:
        import vllm
        facts["version"] = getattr(vllm, "__version__", "unknown")
    except BaseException as exc:  # noqa: BLE001
        facts["version"] = None
        facts["import_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return facts
    try:
        from vllm.platforms import current_platform
        facts["platform"] = str(current_platform.device_name)
        facts["capability"] = str(current_platform.get_device_capability())
    except Exception as exc:  # noqa: BLE001
        facts["platform_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    # The backend enum has moved between releases and its absence is a
    # finding, not a crash: a build with no xformers backend to name is
    # exactly the state this leg is probing for.
    for path in (
        "vllm.attention.backends.registry",
        "vllm.attention.selector",
        "vllm.platforms.interface",
    ):
        try:
            module = __import__(path, fromlist = ["*"])
        except Exception:  # noqa: BLE001
            continue
        backends = getattr(module, "_Backend", None) or getattr(module, "Backend", None)
        if backends is not None:
            names = sorted(getattr(b, "name", str(b)) for b in backends)
            facts["backend_enum_source"] = path
            facts["backends_available"] = names
            facts["xformers_backend_present"] = any("XFORMERS" in n.upper() for n in names)
            break
    return facts


def build_dataset(rows: list[dict]):
    from datasets import Dataset
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                ]
                for row in rows
            ],
            "answer": [row["answer"] for row in rows],
        }
    )


def train(args) -> dict:
    import torch
    from unsloth import FastLanguageModel

    result: dict = {}
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        load_in_4bit = args.load_in_4bit,
        fast_inference = True,
        max_lora_rank = args.lora_rank,
        gpu_memory_utilization = args.gpu_memory_utilization,
    )
    result["load_seconds"] = round(time.time() - t0, 1)
    result["engine_built"] = True
    result["memory_after_load"] = memory()
    _log(f"loaded in {result['load_seconds']}s, " f"memory {result['memory_after_load']}")

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = SEED,
    )

    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding = "utf-8").splitlines()
        if line.strip()
    ]
    dataset = build_dataset(rows)

    from trl import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        output_dir = str(Path(args.outdir) / "trainer"),
        temperature = 1.0,
        learning_rate = 5e-6,
        weight_decay = 0.001,
        warmup_steps = 0,
        lr_scheduler_type = "constant",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = args.num_generations,
        gradient_accumulation_steps = 1,
        num_generations = args.num_generations,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_seq_length - args.max_prompt_length,
        max_steps = args.max_steps,
        seed = SEED,
        fp16 = True,
        bf16 = False,
        report_to = "none",
        save_strategy = "no",
    )
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [reward_length, reward_digit],
        args = config,
        train_dataset = dataset,
    )

    t0 = time.time()
    trainer.train()
    result["train_seconds"] = round(time.time() - t0, 1)
    result["log_history"] = [
        {
            k: v
            for k, v in entry.items()
            if k
            in (
                "step",
                "loss",
                "reward",
                "reward_std",
                "kl",
                "completions/mean_length",
                "frac_reward_zero_std",
            )
            or k.startswith("rewards/")
        }
        for entry in trainer.state.log_history
    ]
    # The shape the launcher and the summary renderer already understand.
    result["metrics"] = [
        {"step": entry.get("step"), "loss": entry.get("loss"), "grad_norm": entry.get("grad_norm")}
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    result["completions"] = SEEN_COMPLETIONS[: args.max_steps * 2]
    result["memory_peak"] = memory()
    _log(
        f"trained in {result['train_seconds']}s; " f"log {json.dumps(result['log_history'])[:1500]}"
    )

    # Generation through the vLLM path after training. `fast_generate` is
    # what the notebook uses and it is a different code path from the
    # trainer's own rollouts, so a failure here is not covered above.
    try:
        from vllm import SamplingParams

        params = SamplingParams(temperature = 1.0, top_k = 50, max_tokens = 32, seed = SEED)
        out = model.fast_generate([rows[0]["question"]], sampling_params = params, lora_request = None)
        result["fast_generate"] = out[0].outputs[0].text
    except BaseException as exc:  # noqa: BLE001
        result["fast_generate"] = None
        result["fast_generate_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"

    del trainer, model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def failures_for(result: dict, args) -> list[str]:
    """The GRPO assertions. See this file's docstring for why not the loss."""
    failures: list[str] = []
    history = result.get("log_history") or []
    rewards = [e["reward"] for e in history if e.get("reward") is not None]
    if not rewards:
        failures.append(
            "no reward was logged on any step, so the reward functions never "
            "ran and generation produced nothing to score"
        )
    elif any(r != r or r in (float("inf"), float("-inf")) for r in rewards):
        failures.append(f"non-finite reward: {rewards}")

    stds = [e["reward_std"] for e in history if e.get("reward_std") is not None]
    if not stds:
        failures.append(
            "reward_std was never logged, so group diversity could not be established"
        )
    elif not any(s > 0 for s in stds):
        failures.append(
            f"reward_std was zero on every step ({stds}): every completion in "
            f"each group scored identically, which means the group was not "
            f"diverse. The GRPO advantage is exactly zero in that state, so "
            f"the run trained on nothing while reporting a healthy loss."
        )

    seen = result.get("completions") or []
    flat = [text for group in seen for text in group]
    if not flat:
        failures.append("no completion was ever produced")
    elif not any(t.strip() for t in flat):
        failures.append(
            f"every one of the {len(flat)} completions was empty, so the "
            f"identical rewards above are not evidence of anything"
        )

    if len(result.get("metrics") or []) != args.max_steps:
        failures.append(
            f"expected {args.max_steps} logged steps, got " f"{len(result.get('metrics') or [])}"
        )

    if result.get("fast_generate") is None:
        failures.append(
            "fast_generate (the vLLM inference path the notebook uses after "
            f"training) failed: {result.get('fast_generate_error')}"
        )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = DEFAULT_MODEL)
    ap.add_argument("--dataset", default = str(_HERE / "canary_dataset.jsonl"))
    ap.add_argument("--outdir", required = True)
    ap.add_argument("--label", default = "grpo")
    ap.add_argument("--max-steps", type = int, default = 3)
    ap.add_argument("--max-seq-length", type = int, default = 2048)
    ap.add_argument("--max-prompt-length", type = int, default = 256)
    ap.add_argument("--num-generations", type = int, default = 4)
    ap.add_argument("--lora-rank", type = int, default = 32)
    ap.add_argument("--gpu-memory-utilization", type = float, default = 0.9)
    # The switch the probe exists to decide. The notebook says False, which
    # is ~8GB of 16-bit weights plus an engine plus a trainer on a 16GB card.
    ap.add_argument("--load-in-4bit", dest = "load_in_4bit", action = "store_true", default = False)
    ap.add_argument("--no-load-in-4bit", dest = "load_in_4bit", action = "store_false")
    ap.add_argument("--probe", action = "store_true", help = "record everything, assert nothing")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)

    report: dict = {
        "label": args.label,
        "model": args.model,
        "leg": "grpo",
        "probe": args.probe,
        "config": {
            k: getattr(args, k)
            for k in (
                "max_steps",
                "max_seq_length",
                "max_prompt_length",
                "num_generations",
                "lora_rank",
                "gpu_memory_utilization",
                "load_in_4bit",
            )
        },
        "failures": [],
    }
    report["versions"] = resolved_versions(
        GOAL_PACKAGES, import_check = ("torch", "transformers", "trl", "vllm")
    )
    report["versions_flat"] = flatten_versions(report["versions"])
    _log("versions " + json.dumps(report["versions_flat"]))
    report["vllm"] = vllm_facts()
    _log("vllm " + json.dumps(report["vllm"]))

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
            "vllm_standby": os.environ.get("UNSLOTH_VLLM_STANDBY"),
        }
    except Exception as exc:  # noqa: BLE001
        report["environment"] = {"error": f"{type(exc).__name__}: {exc}"}

    failures: list[str] = []
    try:
        result = train(args)
        report.update(result)
        failures = failures_for(report, args)
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        report["traceback"] = traceback.format_exc()[-6000:]
        report["engine_built"] = report.get("engine_built", False)
        failures = [f"{type(exc).__name__}: {str(exc)[:600]}"]
        _log("EXCEPTION\n" + report["traceback"])
        report["memory_peak"] = memory()

    report["observed_failures"] = failures
    if args.probe:
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
