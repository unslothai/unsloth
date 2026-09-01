# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Qwen3-4B GRPO with vLLM on a single T4: does the engine fit and generate?

The payload behind the `grpo` leg, written first as a FEASIBILITY PROBE. Two
things are genuinely in doubt on this hardware.

**Does vLLM run on sm_75 at all, at the version installed?** vLLM selects an
attention backend by compute capability, and Turing has neither FlashAttention
nor FlashInfer. It used to depend on the xformers backend, deleted in 0.12.0,
and at the version this leg installs the ladder in `vllm/platforms/cuda.py`
falls through the two unavailable ones to TRITON_ATTN. The leg names TRITON_ATTN
in `VLLM_ATTENTION_BACKEND` rather than trusting that order, so a release that
reorders or drops it goes red instead of silently substituting. None of this
fails at install or import time; it fails when the engine is constructed, deep
inside platform selection, so the resolved vLLM version, the backends the build
offers, the backend selected and the construction outcome are recorded
separately.

**Does it fit?** The notebook this leg comes from sets `load_in_4bit=False`,
roughly 8GB of 16-bit weights, and then asks a vLLM engine at
`gpu_memory_utilization=0.9` and a LoRA training loop to share one 16GB card.
`--load-in-4bit` is therefore a first-class switch, and peak reserved and
allocated memory are reported for whichever setting ran.

What it asserts, and what it deliberately does not:

**Not the loss.** With `num_iterations=1` and `beta=0.0` the TRL GRPO objective
is zero by construction on a healthy run -- the policy that generated the
completions is the one being updated, so the importance ratio is exactly 1, and
with no KL term the loss cancels. A check on it would always pass or fire on
arithmetic noise, so it is recorded and never asserted.

**Reward, reward_std and the completions instead.**

* `reward` must be logged and finite on every step. Absent means the reward
  functions never ran, so generation produced nothing.
* `reward_std` must be non-zero on at least one step. Zero across a group means
  every completion scored identically, which in practice means they were
  IDENTICAL: a sampler ignoring its temperature, a seed applied per-completion
  instead of per-group, or an engine returning the same cached text N times. The
  gradient is exactly zero then, so training "succeeds" while learning nothing
  and no other number moves.
* At least one completion must be non-empty, and the completions seen are
  captured and reported: an engine returning N empty strings scores them all the
  same, so the reward checks alone would call that clean.
* The optimizer must have applied something. Under fp16 here every step can
  overflow and be skipped while loss, reward and reward_std are still logged.
  Decided on the LoRA weights, fingerprinted before and after training, with
  `grad_norm` as fallback so a TRL version that stops logging it cannot take the
  assertion with it. See training_evidence.py.
* The final `fast_generate` runs with the TRAINED adapter, transferred in with
  `save_lora` + `load_lora`. `lora_request=None` reads the base weights and
  passes whether or not the adapter can reach vLLM, which is the second of the
  two questions above.

`--probe` records everything and asserts nothing, for the one-off feasibility
runs.
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

from training_evidence import (  # noqa: E402
    adapter_fingerprint,
    adapter_update,
    update_verdict,
)
from versions import (  # noqa: E402
    GOAL_PACKAGES,
    flatten_versions,
    resolved_versions,
)

SEED = 3407
DEFAULT_MODEL = "unsloth/Qwen3-4B-Base"

SYSTEM_PROMPT = "You are given a question. Answer it as briefly as you can."

# Every completion the reward functions were shown, in order.
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


# Chosen so the reward can never saturate. len/(len+SCALE) is strictly
# increasing on [0, inf), maps into (0, 1), and equals 0.5 at SCALE characters,
# so it stays informative at every length instead of only below a cap.
_LENGTH_SCALE = 200.0


def reward_length(completions, **kwargs) -> list[float]:
    """Longer completions score higher, without ever saturating.

    Deterministic given the text, yet SENSITIVE to a group's diversity: a
    constant reward zeroes `reward_std` on a healthy run and destroys the only
    instrument this leg has.

    THAT IS NOT HYPOTHETICAL - this function used to be

        min(len(t), 200) / 200.0

    and the docstring above already claimed it was diversity-sensitive. It was
    not. Kernels unsloth-probe-grpo-rep2-b03be8 and -rep3-bc3828 recorded
    completions of 2534 to 3396 characters, an order of magnitude past the cap,
    so every completion scored exactly 1.0, every group tied, and the leg failed
    with `reward_std was zero on every step` - a red that looks like a broken
    generation path and is really a reward that stopped measuring anything.
    Two of three runs died this way, which made it the leg's dominant failure.

    The cap was the whole problem: it was set BELOW the lengths the model
    actually produces, so the one region it discriminated in was the one region
    the run never visited.
    """
    texts = _texts(completions)
    SEEN_COMPLETIONS.append(texts)
    return [len(t) / (len(t) + _LENGTH_SCALE) for t in texts]


def reward_digit(completions, **kwargs) -> list[float]:
    """A second, differently shaped signal, so `reward` is not one function.

    One broken reward function looks exactly like a broken generation path; two
    disagreeing sources make that distinguishable in the report.
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

    Recorded BEFORE the engine is built, so a payload that dies constructing it
    still says what it was trying to construct. The backend comes from vLLM's
    own selector where reachable and from the environment override otherwise;
    both are reported, since an override silently deciding it is worth seeing.
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
    # The backend enum has moved between releases, and its absence is a finding rather than a crash.
    # Both names are recorded, not asserted: at the version this leg installs, xformers is EXPECTED missing and
    # TRITON_ATTN is where the ladder should land, so which world we are in is answerable from the report alone.
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
            facts["triton_attn_backend_present"] = any("TRITON_ATTN" in n.upper() for n in names)
            facts["requested_backend"] = os.environ.get("VLLM_ATTENTION_BACKEND", "")
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


def train(args, report: dict | None = None) -> dict:
    """One GRPO cycle. Writes progress into ``report`` as it goes.

    The second argument makes the feasibility verdict survive a crash.
    Everything used to be returned at the end, so an exception after the engine
    was built threw away the fact that it HAD been, and the report said
    `engine_built: false` -- the opposite of what happened, and the exact
    distinction this probe draws between "vLLM cannot start on sm_75" and "vLLM
    started and GRPO failed later". Facts are published as soon as known.
    """
    import torch
    from unsloth import FastLanguageModel

    result: dict = {}
    if report is None:
        report = {}

    def record(key, value):
        result[key] = value
        report[key] = value

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        load_in_4bit = args.load_in_4bit,
        fast_inference = True,
        max_lora_rank = args.lora_rank,
        gpu_memory_utilization = args.gpu_memory_utilization,
    )
    record("load_seconds", round(time.time() - t0, 1))
    record("engine_built", True)

    # `unsloth/Qwen3-4B-Base` is a BASE model and ships no chat template, so TRL's `maybe_apply_chat_template` raises on
    # the first training step: ValueError: Cannot use chat template functions because tokenizer.chat_template is not set
    # Measured on kernel unsloth-t4-ci-27b0dc2e, the first probe to get that far (the vLLM engine had built and the
    # trainer was inside `_run_epoch`).
    # The base model is the RIGHT choice and is not what to change: GRPO on an instruct model would measure the instruct
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + "
            "'<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        result["chat_template"] = "set by the payload (base model ships none)"
    else:
        result["chat_template"] = "shipped with the tokenizer"
    _log("chat template: " + result["chat_template"])
    result["memory_after_load"] = memory()
    _log(f"loaded in {result['load_seconds']}s, memory {result['memory_after_load']}")

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

    # The adapter before a single step, so whether the optimizer applied anything is a subtraction rather than a reading
    # of what TRL chose to log.
    # Under fp16 here every step can overflow and be skipped while loss, reward and reward_std are still logged, and
    # grad_norm was the only field that used to say so.
    # See training_evidence.py.
    adapter_before = adapter_fingerprint(model)
    _log(f"adapter before training: {json.dumps(adapter_before)}")

    t0 = time.time()
    trainer.train()
    result["train_seconds"] = round(time.time() - t0, 1)
    result["adapter_update"] = adapter_update(adapter_before, adapter_fingerprint(model))
    _log(f"adapter update: {json.dumps(result['adapter_update'])}")
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
    result["metrics"] = [
        {"step": entry.get("step"), "loss": entry.get("loss"), "grad_norm": entry.get("grad_norm")}
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    result["completions"] = SEEN_COMPLETIONS[: args.max_steps * 2]
    result["memory_peak"] = memory()
    _log(f"trained in {result['train_seconds']}s; log {json.dumps(result['log_history'])[:1500]}")

    # Generation through the vLLM path after training.
    # `fast_generate` is what the notebook uses and is a different code path from the trainer's own rollouts, so a
    # failure here is not covered above.
    # WITH the trained adapter: `fast_generate` is the engine's own generate, and the trained LoRA reaches it only
    # through a lora_request built by `save_lora` + `load_lora`.
    # Passing None generates from the base weights and passes whether or not the adapter can be transferred at all,
    # which is the second question this leg answers, so the request is built, recorded, and asserted on in
    # `failures_for`.
    lora_state: dict = {"requested": True, "applied": False}
    lora_request = None
    try:
        lora_dir = str(Path(args.outdir) / "grpo_lora")
        model.save_lora(lora_dir)
        lora_request = model.load_lora(lora_dir)
        lora_state.update(applied = lora_request is not None, path = lora_dir)
        if lora_request is None:
            lora_state["error"] = "load_lora returned None"
    except BaseException as exc:  # noqa: BLE001
        lora_state["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    record("fast_generate_lora", lora_state)

    try:
        from vllm import SamplingParams

        params = SamplingParams(temperature = 1.0, top_k = 50, max_tokens = 32, seed = SEED)
        out = model.fast_generate(
            [rows[0]["question"]], sampling_params = params, lora_request = lora_request
        )
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
    # The training steps, not every row:
    steps = [e for e in history if "loss" in e]
    rewards = [e["reward"] for e in history if e.get("reward") is not None]
    if not rewards:
        failures.append(
            "no reward was logged on any step, so the reward functions never "
            "ran and generation produced nothing to score"
        )
    elif any(r != r or r in (float("inf"), float("-inf")) for r in rewards):
        failures.append(f"non-finite reward: {rewards}")
    else:
        # used to satisfy it, so a step whose reward functions never ran was
        # ON EVERY STEP, as this file's docstring claims.
        missing = [e.get("step") for e in steps if e.get("reward") is None]
        if missing:
            failures.append(
                f"the reward functions did not run on every step: no reward was "
                f"logged for step(s) {missing} of {len(steps)}. A reward on some "
                f"steps is not the invariant this leg asserts."
            )

    stds = [e["reward_std"] for e in history if e.get("reward_std") is not None]
    if not stds:
        failures.append("reward_std was never logged, so group diversity could not be established")
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

    metrics = result.get("metrics") or []
    if len(metrics) != args.max_steps:
        failures.append(f"expected {args.max_steps} logged steps, got {len(metrics)}")

    # Under fp16 here every step can overflow and be skipped while loss, reward and reward_std are still logged and
    # base-model generation still returns text, so the length check above is satisfied by a run that applied no
    # optimizer update at all.
    # The earlier spelling `if norms and not applied` passes on an EMPTY list: a TRL version that stops logging
    # grad_norm removes the only instrument this leg had, and it saves no adapter to read back either.
    # See training_evidence.py.
    update = update_verdict(metrics, result.get("adapter_update"))
    if update["verdict"] == "not_applied":
        failures.append(
            f"no optimizer update was applied: {update['detail']}. The fp16 scaler "
            f"skips the step it overflowed on, so this run generated, scored and "
            f"updated nothing."
        )
    elif update["verdict"] == "non_finite":
        failures.append(
            f"the adapter holds non-finite weights after training: {update['detail']}. "
            f"An optimizer step landed and produced NaN or infinity, which is a "
            f"broken run rather than an untrained one, and generation still returns "
            f"text either way."
        )
    elif update["verdict"] != "applied":
        failures.append(
            f"whether the optimizer applied anything could not be established: "
            f"{update['detail']}. Every other number this leg reports is produced "
            f"by generation and scoring, which the base model does on its own."
        )

    lora = result.get("fast_generate_lora")
    if lora is not None and lora.get("requested") and not lora.get("applied"):
        failures.append(
            f"the trained adapter could not be handed to the vLLM engine, so the "
            f"final generation ran on the base weights and proves nothing about "
            f"training: {lora.get('error')}"
        )

    if result.get("fast_generate") is None:
        # Not `== "unverifiable"`: any verdict this file has not been taught about is a failure rather than a silent
        failures.append(
            "fast_generate (the vLLM inference path the notebook uses after "
            f"training) failed: {result.get('fast_generate_error')}"
        )
    return failures


def make_libcuda_linkable() -> dict:
    """Let the linker find `-lcuda`, so flashinfer's JIT can link what it built.

    Measured twice on real T4 sessions (kernels unsloth-t4-ci-e2d9ce9b and
    -916d5986). flashinfer 0.6.6 JIT-compiles its sampling ops on first use, and
    on Kaggle all three .cu files COMPILE cleanly for
    `-gencode=arch=compute_75,code=sm_75` -- nothing here is a Turing problem --
    and then the link dies:

        /usr/bin/ld: cannot find -lcuda

    `-L/usr/local/cuda/lib64/stubs` is already on that command line; the image
    simply ships no `libcuda.so`, only the runtime `libcuda.so.1`, a versioned
    soname the linker will not resolve `-lcuda` against. Normally the CUDA
    toolkit's driver STUB fills that gap; this image has the directory and not
    the file.

    `VLLM_USE_FLASHINFER_SAMPLER=0` was tried first and did not help, which is
    the useful part: the JIT is not reached only through the sampler, so
    switching off one consumer is whack-a-mole, while making `-lcuda` resolvable
    fixes every flashinfer op at once.

    `LIBRARY_PATH` rather than a symlink into /usr/local: gcc and ld search it
    for `-l`, it needs no root, and it cannot damage the image for anything else
    in the session. Linking against the real driver rather than a stub is
    correct, the driver being present is the whole reason a stub would have
    substituted for it.

    Returns what it did, so the report says so rather than the next reader
    inferring it from an absence of failure.
    """
    facts: dict = {"needed": False, "applied": False}
    try:
        import ctypes.util
        import subprocess

        # ONLY the directories flashinfer passes with -L.
        # Measured on kernel unsloth-t4-ci-d0d480b6: an earlier version also accepted /usr/local/cuda/compat, found
        # libcuda.so there, concluded "already linkable" and did nothing, and the link failed anyway because compat is
        link_dirs = ["/usr/local/cuda/lib64", "/usr/local/cuda/lib64/stubs"]
        for d in link_dirs:
            if os.path.exists(os.path.join(d, "libcuda.so")):
                facts["already_linkable"] = d
                return facts
        facts["needed"] = True
        facts["searched"] = link_dirs

        # Where the real driver lives.
        real = None
        try:
            out = subprocess.run(
                ["/sbin/ldconfig", "-p"], capture_output = True, text = True, timeout = 60
            ).stdout
            for line in out.splitlines():
                if "libcuda.so.1" in line and "=>" in line:
                    real = line.split("=>")[-1].strip()
                    break
        except Exception:
            real = None
        if real is None or not os.path.exists(real):
            found = ctypes.util.find_library("cuda")
            real = found if found and os.path.exists(found) else None
        if real is None:
            # The compat tree is the usual place on Kaggle: useless as a -L target since nothing passes it, fine as a
            # symlink TARGET.
            for candidate in (
                "/usr/local/cuda/compat/libcuda.so",
                "/usr/local/cuda/compat/libcuda.so.1",
            ):
                if os.path.exists(candidate):
                    real = candidate
                    break
        if real is None:
            facts["error"] = "no libcuda.so.1 on this machine"
            return facts

        shim = os.path.join(os.environ.get("TMPDIR") or "/tmp", "unsloth_libcuda_shim")
        os.makedirs(shim, exist_ok = True)
        link = os.path.join(shim, "libcuda.so")
        if not os.path.exists(link):
            os.symlink(real, link)
        existing = os.environ.get("LIBRARY_PATH", "")
        os.environ["LIBRARY_PATH"] = f"{shim}:{existing}" if existing else shim
        facts.update(applied = True, real = real, shim = shim)
    except Exception as exc:  # noqa: BLE001
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts


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
    # The switch the probe exists to decide.
    # The notebook says False: ~8GB of 16-bit weights plus an engine plus a trainer on a 16GB card.
    ap.add_argument("--load-in-4bit", dest = "load_in_4bit", action = "store_true", default = False)
    ap.add_argument("--no-load-in-4bit", dest = "load_in_4bit", action = "store_false")
    ap.add_argument("--probe", action = "store_true", help = "record everything, assert nothing")
    # An illegal memory access surfaces at whatever CUDA call synchronises next, which on the first T4 GRPO run was
    # `empty_cache()` inside vLLM standby, nowhere near the faulting kernel.
    ap.add_argument("--cuda-launch-blocking", action = "store_true")
    args = ap.parse_args()

    if args.cuda_launch_blocking:
        # Must precede any CUDA context creation, so before this process's first `import torch` rather than merely
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # Before anything imports vLLM: flashinfer JITs on first use and the link step is what fails on this image.
    libcuda = make_libcuda_linkable()
    _log(f"libcuda link shim: {libcuda}")

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
                # Reported because it changes what a timing or a traceback means.
                "cuda_launch_blocking",
            )
        },
        "failures": [],
    }
    report["versions"] = resolved_versions(
        GOAL_PACKAGES, import_check = ("torch", "transformers", "trl", "vllm")
    )
    report["versions_flat"] = flatten_versions(report["versions"])
    _log("versions " + json.dumps(report["versions_flat"]))
    report["libcuda_shim"] = libcuda
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
        result = train(args, report)
        report.update(result)
        failures = failures_for(report, args)
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        # Head AND tail: the last probe's 6000-char tail was entirely ninja's output, dropping the Python frames that
        _tb = traceback.format_exc()
        report["traceback"] = (
            _tb if len(_tb) <= 12000 else _tb[:6000] + "\n...[middle elided]...\n" + _tb[-6000:]
        )
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
