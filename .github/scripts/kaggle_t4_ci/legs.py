# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The payloads this CI can run, and what each one is FOR.

A Kaggle session is 2xT4 and the account allows 2 concurrent batch kernels, so four payloads at once is the ceiling. Each entry is a *leg*: an install recipe, a script and its arguments; `build_kernel.py` turns a list of legs into kernel notebooks.

**control** and **canary** are a matched pair: the SAME payload, seed, dataset and step count on the same card, differing only in the transformers/trl/peft/accelerate/bitsandbytes versions. Control pins them (tests/kaggle/t4_smoke/pins/control.txt); canary takes the newest release Unsloth's declared constraints allow. Canary red with control green means a library RELEASE broke Unsloth, and the canary's report names every resolved version; both red means the base image, the model download, Kaggle or Unsloth's own code; control red with canary green means the pins no longer resolve. All three readings break at once if the legs differ in anything but versions, which is why `--smoke-args` and the reference are shared rather than per leg.

**gptoss** covers `torch.compile` and the forced-float32 path (gpt-oss is in FORCE_FLOAT32 precisely because this card has no bf16). **grpo** covers vLLM, which `fast_inference=True` puts on the same 16GB card as the training loop.

To add a leg, append an entry and name it in a `KERNELS` kernel. tests/kaggle/test_t4_smoke_harness.py builds every leg and parses every generated cell, so a leg that cannot produce valid Python never reaches Kaggle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ZOO = "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@{zoo_ref}"
UNSLOTH = "unsloth @ git+https://github.com/unslothai/unsloth@{unsloth_ref}"

# Expanded at build time so the notebook states its versions.
PINS = "@PINS:{file}"

# Named explicitly so torch and the CUDA stack do not move too.
CANARY_UPGRADES = ("transformers", "trl", "peft", "accelerate", "bitsandbytes")


@dataclass(frozen = True)
class Leg:
    """One payload: what to install, what to run, and what it is for."""

    name: str
    summary: str
    install: tuple[tuple[str, ...], ...]
    entry: str
    args: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    imports: tuple[str, ...] = (
        "torch",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "bitsandbytes",
        "unsloth",
        "unsloth_zoo",
    )
    # Measured peak_reserved_gb on a T4; the driver uses it to decide card sharing.
    vram_gb: float = 1.0
    # Pure-Python pins installed --no-deps into a per-leg dir prepended on PYTHONPATH.
    # Cannot replace torch, triton or bitsandbytes.
    overlay: tuple[str, ...] = ()
    # Removed after install: FlashInfer cannot link on Kaggle (no libcuda.so stub).
    uninstall: tuple[str, ...] = ()
    reference: str = ""
    env: dict = field(default_factory = dict)
    # False for legs that replace torch: visible image NVIDIA packages would yield an unimportable torch.
    system_site_packages: bool = True
    # For multi_gpu: DEVICE_COUNT > 1 code paths only run when every card is visible.
    all_cards: bool = False


COMMON_FILES = (
    "versions.py",
    "canary_dataset.jsonl",
    "training_evidence.py",
    "phase_timers.py",
)

# unsloth resolves its own dependencies, as `pip install unsloth` does, so pyproject.toml is tested.
BASE_INSTALL = ((ZOO,), (UNSLOTH,), ("bitsandbytes",))

PACKAGE_UNDER_TEST = UNSLOTH.split("@", 1)[0].strip()

SMOKE_FILES = COMMON_FILES + (
    "run_t4_smoke.py",
    "determinism.py",
    "gguf_export.py",
    # Shipped to every leg: run_t4_smoke imports it at module scope.
    "naive_trl_compare.py",
    # Imported at module scope by run_t4_smoke.
    "kernel_provenance.py",
)


LEGS: dict[str, Leg] = {
    "control": Leg(
        vram_gb = 0.7,
        name = "control",
        summary = "tiny SFT determinism run, pinned library set",
        # Pins go last so they beat what earlier groups resolved.
        install = BASE_INSTALL + ((PINS.format(file = "control.txt"),),),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES + ("pins/control.txt",),
        reference = "t4_qwen2.5-0.5b.json",
        args = ("--pins", "@ROOT/pins/control.txt"),
    ),
    "canary": Leg(
        vram_gb = 0.7,
        name = "canary",
        summary = "the same SFT run on the newest permitted library set",
        # One resolution with zoo present so pip respects zoo's constraints.
        install = BASE_INSTALL + ((("--upgrade", ZOO) + CANARY_UPGRADES),),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        # No reference: different library sets do not share one fp16 trajectory.
        reference = "",
    ),
    "latest_compile": Leg(
        # Measured 12.73 GB peak; lower values let the scheduler co-tenant this leg and OOM.
        vram_gb = 12.8,
        name = "Latest_compile",
        summary = "gemma-4-E2B-it SFT on the newest transformers and trl, against plain TRL",
        # With deps: --no-deps overshoots transformers' tokenizers and safetensors floors.
        install = BASE_INSTALL + ((("--upgrade", "transformers", "trl")),),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        args = (
            "--model",
            "unsloth/gemma-4-E2B-it",
            "--max-steps",
            "10",
            # The plain-TRL comparison reports both traces rather than asserting equality across versions.
            "--compare-naive-trl",
            # The plain arm OOMs at load on a T4 (E2B checkpoint carries E4B weights); training OOMs still fail.
            "--control-oom-is-ok",
        ),
        reference = "",
    ),
    # Not wired; see UNWIRED.
    "vision_fla_compile": Leg(
        vram_gb = 2.84,
        name = "Vision_FLA_compile",
        summary = "Qwen3.5-2B on the newest stack: vendored FLA, sdpa on Turing, completions-only",
        install = BASE_INSTALL + ((("--upgrade", "transformers", "trl")),),
        entry = "run_t4_smoke.py",
        # Separate entry script so vision-only setup stays out of the text legs.
        files = SMOKE_FILES + ("run_vision_t4.py",),
        args = (
            "--model",
            "unsloth/Qwen3.5-2B",
            "--max-steps",
            "10",
            "--kernel-provenance",
            "--vision-run",
            "--export-gguf",
        ),
        reference = "",
    ),
    "frontier": Leg(
        vram_gb = 0.7,
        name = "frontier",
        summary = "the same SFT run on the newest transformers and trl on PyPI",
        # The canary is capped by zoo's metadata, so this leg covers newer transformers and trl.
        # With deps, not --no-deps: that broke on tokenizers and safetensors floors.
        # Resolving also moves datasets and huggingface_hub.
        install = BASE_INSTALL + ((("--upgrade", "transformers", "trl")),),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        reference = "",
    ),
    "default": Leg(
        vram_gb = 0.7,
        name = "Default",
        summary = "Qwen3-0.6B on the pinned default set, plus batched inference",
        # Version pins arrive as an overlay below, not resolved into the venv.
        install = BASE_INSTALL,
        overlay = ("transformers==4.57.6", "trl~=0.22.0"),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        # Covers the released `pip install unsloth` lower bound so the range is not tested twice at one point.
        args = (
            "--model",
            "unsloth/Qwen3-0.6B",
            # 20 steps: at 10 the model does not learn to stop after the canary.
            "--max-steps",
            "20",
            "--export-gguf",
        ),
        # No reference yet: the committed band is for a different model.
        reference = "",
    ),
    "multi_gpu": Leg(
        # Small on purpose: this leg reserves its share on every card.
        vram_gb = 1.2,
        name = "Multi_GPU",
        summary = "Qwen3-0.6B with BOTH T4s visible: unsloth's DEVICE_COUNT > 1 bindings",
        install = BASE_INSTALL,
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        # The point: DEVICE_COUNT > 1 code paths only run with multiple visible cards.
        all_cards = True,
        args = (
            "--model",
            "unsloth/Qwen3-0.6B",
            # 20 steps, as in Default.
            "--max-steps",
            "20",
            "--require-multi-gpu",
            "--expected-cards",
            "2",
            # Weights on one card: sharding across both currently fails and is not a per-PR check.
            "--single-device",
            # No --export-gguf: the bundled llama.cpp is backend CPU, so a tensor-split check could not fail.
        ),
        reference = "",
    ),
    "gptoss": Leg(
        vram_gb = 12.78,
        name = "gptoss",
        summary = "gpt-oss-20b LoRA: torch.compile and the float32 path",
        # No triton_kernels: load_in_4bit redirects to the NF4 checkpoint, so MXFP4 is never reached.
        install = BASE_INSTALL,
        entry = "run_gptoss_t4.py",
        # gguf_export.py is imported lazily, so declare it explicitly.
        files = COMMON_FILES + ("run_gptoss_t4.py", "gguf_export.py"),
        args = (
            "--max-steps",
            "3",
            "--max-seq-length",
            "1024",
            # No export: gpt-oss forces MXFP4, the slowest and least representative export.
        ),
    ),
    # Not wired; see UNWIRED.
    "grpo": Leg(
        vram_gb = 13.8,
        name = "grpo",
        summary = "Qwen3-4B GRPO through a vLLM engine on the same card",
        # vLLM first and alone: it pins torch. 0.19.1 is the newest pinning the image's torch 2.10.0.
        install = (("vllm==0.19.1",),) + BASE_INSTALL,
        entry = "run_grpo_t4.py",
        files = COMMON_FILES + ("run_grpo_t4.py",),
        imports = (
            "torch",
            "transformers",
            "trl",
            "peft",
            "datasets",
            "bitsandbytes",
            "vllm",
            "unsloth",
            "unsloth_zoo",
        ),
        # All five values are load-bearing on a 14.56 GB card.
        args = (
            "--max-steps",
            "3",
            "--load-in-4bit",
            "--gpu-memory-utilization",
            "0.95",
            "--max-seq-length",
            "1024",
            "--num-generations",
            "2",
            "--lora-rank",
            "16",
        ),
        env = {
            "UNSLOTH_VLLM_STANDBY": "1",
            # Named rather than probed so a backend reorder turns this leg red.
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            # Avoid flashinfer JIT: it cannot link on Kaggle and costs wall clock.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            # Required: unsloth_zoo's patch_vllm overwrites the two settings above unless this is set.
            "UNSLOTH_VLLM_NO_FLASHINFER": "1",
        },
        # Kept alongside the env var so a future image still takes the deliberate path.
        uninstall = ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"),
        system_site_packages = True,
    ),
}


# VRAM admission, not card count, decides packing; wall clock against the 12h kill caps legs.
MAX_LEGS_PER_KERNEL = 6

# Legs defined but not run; every entry must say what was measured.
UNWIRED: dict[str, str] = {
    "frontier": (
        "SUPERSEDED by vision_fla_compile rather than broken. frontier was "
        "latest transformers and trl on Qwen2.5-0.5B; the vision leg is latest "
        "transformers and trl on Qwen3.5-2B and asserts everything frontier "
        "did plus the vendored FLA kernels, the Turing attention choice, a "
        "real vision training run, the merged vision export, a Q8_0 GGUF with "
        "its mmproj sidecar and inference on the exported file. Everything "
        "frontier proved is a subset, so keeping both spends a card on the "
        "smaller claim against MAX_LEGS_PER_KERNEL = 5.\n"
        "The definition stays rather than being deleted: it is the cheapest "
        "latest-everything leg there is, so if the vision leg ever has to come "
        "out for a reason of its own, this is what goes back in that hour "
        "instead of being reconstructed from a commit message."
    ),
    "multi_gpu": (
        "MEASURED AND REJECTED for the per-PR kernel and RUNNING NIGHTLY "
        "instead, which is where the coverage is free. The leg passes: it is "
        "green in BOTH arms of the ab3 A/B and in unsloth-probe-venvfix-r1 and "
        "-r2, with the DEVICE_COUNT > 1 bindings live for the first time in "
        "this CI.\n"
        "ab3, four sessions, one commit, both accounts, arm A the wired five "
        "legs plus Studio and arm B the same plus this leg:\n"
        "    account   A        B        delta\n"
        "    r1        1511.8   1684.2   +172.4\n"
        "    r2        1508.7   1548.4    +39.7\n"
        "Slower in 2 of 2, mean +106.1s on a ~1510s makespan. Arm A reproduced "
        "to 3.1s across accounts, so the baseline is not the noisy part; arm B "
        "varies by 135.8s, which is why the delta is quoted as a range and a "
        "sign rather than as one number. The cost is ATTRIBUTABLE: "
        "vision_fla_compile sets the makespan alone on gpu0 and is the leg that "
        "slows, 1486.2 -> 1644.8 and 1480.3 -> 1518.7, so a sixth concurrent "
        "install on 4 vCPUs is what this buys. The brief was multi-GPU coverage "
        "at no wall-clock cost, and +106s is a cost.\n"
        "WITHDRAWN, and recorded because it stood in this file as a finding: an "
        "earlier note said the Default leg failed 3 of 3 whenever this leg was "
        "present, with PicklingError on pyarrow's MonthDayNano, and blamed the "
        "sixth install perturbing another leg's overlay. That was two defects "
        "in build_kernel.py -- an unpinned venv interpreter and an overlay "
        "directory dill pickles by value -- both fixed. Default now passes in "
        "every arm-B session run since, and no session contains a "
        "PicklingError anywhere.\n"
        "WHAT WOULD UNBLOCK IT for per-PR: a makespan that stops being set by "
        "one leg. While vision_fla_compile owns gpu0 for 1480s of a 1510s run "
        "there is no slack for a sixth install to hide in, and no scheduler "
        "change recovers CPU that is not there."
    ),
    "latest_compile": (
        "STILL UNKNOWN: nothing about the leg. It is GREEN end to end and is "
        "held out by a DEPENDENCY, which is why this note now reads as it "
        "does. unsloth-probe-lcleg-tmpdir-ac53ca: passed true, failures [], "
        "naive_trl_failures [], peak 12.73GB on one card on both cycles, Q8_0 "
        "4725.1MB plus a 940.0MB F16-mmproj sidecar from the prebuilt "
        "llama.cpp bundle, and llama-bench rc=0 on the exported file at 16.18 "
        "t/s pp8. Both of the questions this note used to carry are answered: "
        "vram_gb is measured at 12.8 rather than the 6.0 placeholder that "
        "would have co-tenanted a card it cannot share, and the plain-TRL arm "
        "runs -- its load-time OOM is a fact about a MatFormer checkpoint on a "
        "14.56GB card and is REPORTED, via --control-oom-is-ok, rather than "
        "failed.\n"
        "THE BLOCKER: unsloth-zoo #1103. That run was built with --zoo-ref "
        "fix/heterogeneous-config-dtype-walk. Against zoo main the leg cannot "
        "load gemma-4 at all -- patching_utils.py:467 walks a config key that "
        "transformers 5.15 refuses to read back, and "
        "AmbiguousGlobalPerLayerAttributeError is not an AttributeError, so "
        "the `getattr(..., None)` default does not suppress it. Measured "
        "twice, unsloth-probe-latestcompile-a07f60 and -lcleg-r6-616b6f, same "
        "line and same call path.\n"
        "#1103 IS MERGED (2026-08-27, 5a017838), so that blocker is gone and "
        "the sentence this note used to end with -- 'this moves into KERNELS "
        "the day #1103 merges' -- is WITHDRAWN. It was written without the one "
        "number that decides where the leg goes.\n"
        "MEASURED AND REJECTED for the per-PR kernel, RUNNING NIGHTLY instead. "
        "The leg's own DONE record is 1323.0s, and at 12.73GB peak against "
        "CARD_VRAM_BUDGET_GB = 13.0 it admits no co-tenant, so it wants a card "
        "to itself for 22 minutes. The per-PR kernel has exactly one block of "
        "slack -- gpu1 idle 776.3s, from 1358.7 to 2126.7, while Studio holds "
        "gpu0 -- and 1323 does not fit in 776. Wiring it there adds roughly "
        "580s to a 2101.8s makespan, which is the same trade multi_gpu was "
        "refused for at a fifth the cost.\n"
        "WHAT WOULD UNBLOCK IT for per-PR: the leg getting under ~776s, or the "
        "kernel gaining a second free card in that window. Most of the 1323s "
        "is install and load rather than the 52.0s and 25.4s of training, so "
        "the shared-base-venv work is where that would come from; dropping the "
        "GGUF export would also buy back time, at the cost of the one claim "
        "only this leg makes on a MatFormer checkpoint. Measure before "
        "believing either.\n"
        "It stays in UNWIRED rather than moving to KERNELS because UNWIRED is "
        "what the guard reads: the leg runs, nightly, and nothing about the "
        "leg itself is open."
    ),
    "grpo": (
        "vLLM standby sleep hits an illegal memory access on Turing, and it is "
        "INTERMITTENT. Three sessions on a real Tesla T4, identical to the "
        "flag and identical in every recorded version (torch 2.10.0+cu128, "
        "transformers 5.5.0, trl 0.24.0, peft 0.19.1, vllm 0.19.1, unsloth "
        "2026.8.15, zoo 2026.8.10) and at the same 13.8GB/13.6GB peak of "
        "14.56GB: unsloth-t4-ci-53efcc4e PASSED (engine_built true, reward_std "
        "0.707 and grad_norm 0.772 at step 2, three steps in 192s), then "
        "unsloth-t4-ci-70a2f4eb and unsloth-t4-ci-c98f14be both FAILED with "
        "engine_built false and\n"
        "  unsloth_zoo/vllm_utils.py:601 sleep() -> torch.cuda.empty_cache()\n"
        "  torch.AcceleratorError: CUDA error: an illegal memory access was "
        "encountered\n"
        "UNSLOTH_VLLM_STANDBY=1 is set in all three. One pass in three is not "
        "a leg CI can spend a session on: it would go red for a reason no "
        "reader could act on.\n"
        "The --cuda-launch-blocking run is done, kernel unsloth-t4-ci-b1f23e34, "
        "and it did NOT localise the fault: with blocking on there was no "
        "illegal memory access at all. engine_built true, three steps, same "
        "13.8GB peak. A fault that disappears when the launches are "
        "serialised is a race, which is what the one-pass-in-three rate "
        "already suggested.\n"
        "RE-MEASURED on the current stack (2026-08-25), NINE sessions, and it "
        "changes which problem is the blocker. Crash axis, with the flashinfer "
        "uninstall in place: grpo-shipped-d15695, rep2-b03be8, rep3-bc3828, "
        "reward3-045dba and crashax-b-27d9e3 clean; reward-158b39, crashax-a, "
        "crashax-c2 and crashax-d2 CRASHED. FOUR IN NINE, 44%. Without the "
        "uninstall: noun-05777b crashed, one in one. This note said ONE IN "
        "FOUR at five sessions, off a window that happened to open on three "
        "clean runs; four more sessions moved the rate the wrong way, which is "
        "what a 44% fault looks like some of the time. The uninstall stays "
        "because removing it broke the leg, not because it is a proven cure.\n"
        "The crash is still what the --cuda-launch-blocking run said it was: a "
        "race, not a capacity problem. Peak was 13.39-13.40GB of 14.56 in every "
        "one of the nine, crashed or not, so memory does not distinguish them "
        "either, and it survives BOTH the flashinfer uninstall and "
        "UNSLOTH_VLLM_NO_FLASHINFER=1. Nine sessions at a 44% reproduction "
        "rate is worth filing upstream.\n"
        "It also exposed a SECOND problem, and the two are separate. That run "
        "failed on reward_std = [0.0, 0.0, 0.0] with grad_norm 0.0 at every "
        "step. The completions recorded in the report are coherent prose, not "
        "degenerate, so this is not the model collapsing -- it is two "
        "completions scoring identically. The leg runs num_generations = 2, "
        "shrunk to fit a 14.56GB card, and at two samples a tie on a coarse "
        "reward is ordinary rather than a bug. So the leg's own pass "
        "criterion is fragile at the size it has to be to fit.\n"
        "THAT SECOND PROBLEM IS NOW FIXED AND CONFIRMED ON HARDWARE, TWICE, "
        "and it was the dominant failure rather than the crash: two of the "
        "three clean-crash runs still went red on reward_std zero. The cause "
        "was reward_length saturating at 200 characters while the model emits "
        "2534-3396, so every completion scored 1.0 and every group tied. It is "
        "now len/(len+200), which cannot saturate. Two independent sessions on "
        "two accounts then recorded non-zero reward_std on every step -- "
        "reward3-045dba and crashax-b-27d9e3 -- with frac_reward_zero_std 0.0 "
        "throughout. The reward blocker is closed.\n"
        "SO WHAT KEEPS THIS OUT OF THE PER-PR KERNEL IS THE CRASH ALONE, and "
        "44% is the number that decides it: a coin-flip red in front of every "
        "PR, for a fault no reader can act on, is how a check gets switched "
        "off before the day it is right. The nightly is where it runs instead, "
        "and it does -- kaggle-t4-notebook-ci.yml dispatches 'grpo,multi_gpu' "
        "on the schedule, where a red costs nobody a merge.\n"
        "STILL UNKNOWN: where the race is (the standby wake/sleep cycle on "
        "sm_75 is the suspect, and UNSLOTH_VLLM_STANDBY=1 is set in every "
        "session), and what pass criterion is honest at num_generations = 2."
    ),
}

# One kernel holds every leg so the shared account keeps a free session slot.
# gptoss sits third on purpose so the prefetch lane can finish its download first.
KERNELS: tuple[tuple[str, ...], ...] = (
    ("canary", "control", "gptoss", "vision_fla_compile", "default"),
)


# Must match what legs LOAD after redirects, not the declared names.
PREFETCH_REPOS: tuple[str, ...] = (
    # Critical path first: vision_fla_compile starts earliest and sets the makespan.
    "unsloth/Qwen3.5-2B",
    # Both names, because FLOAT_TO_INT_MAPPER redirects at load time.
    "unsloth/Qwen3-0.6B",
    "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    "unsloth/Qwen2.5-0.5B-Instruct",
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
)

LOAD_REDIRECTS: dict[str, str] = {
    "unsloth/gpt-oss-20b": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    # Case matters: the HF cache keys on the literal string. Measured as
    # `resolved_checkpoint: unsloth/Qwen3-0.6B-unsloth-bnb-4bit` on kernel unsloth-t4-ci-n361d0b65ec92-0d7d.
    "unsloth/Qwen3-0.6B": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
}


def expand_install(
    leg: Leg, *, unsloth_ref: str, zoo_ref: str, payload_dir: Path
) -> list[list[str]]:
    """Resolve a leg's install groups into concrete pip argument lists."""
    groups: list[list[str]] = []
    for group in leg.install:
        expanded: list[str] = []
        for item in group:
            if item.startswith("@PINS:"):
                expanded.extend(_read_pins(payload_dir / "pins" / item[len("@PINS:") :]))
                continue
            expanded.append(item.format(unsloth_ref = unsloth_ref, zoo_ref = zoo_ref))
        if expanded:
            groups.append(expanded)
    return groups


def _read_pins(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"leg names a pin file that is not there: {path}")
    out = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    if not out:
        raise ValueError(
            f"pin file {path} names no versions at all, so the control leg would pin nothing"
        )
    return out


def resolve(names) -> list[Leg]:
    """Legs by name, in the order given. Unknown names fail loudly here, at build time rather than on the kernel: a typo in a workflow input must cost a runner second, not a Kaggle session."""
    legs = []
    for name in names:
        if name not in LEGS:
            raise SystemExit(f"unknown leg {name!r}; known legs are {', '.join(sorted(LEGS))}")
        legs.append(LEGS[name])
    return legs
