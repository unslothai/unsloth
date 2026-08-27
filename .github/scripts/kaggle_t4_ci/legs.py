# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The payloads this CI can run, and what each one is FOR.

A Kaggle session is 2xT4 and the account allows 2 concurrent batch kernels, so
the ceiling is four payloads at once, and this is the registry of what those
four are. Each entry is a *leg*: an install recipe, a script and its arguments.
`build_kernel.py` turns a list of legs into kernel notebooks; nothing else here
knows what a leg contains.

**control** and **canary** are a matched pair and the core of the design: the
SAME payload, seed, dataset and step count on the same card, differing only in
the transformers/trl/peft/accelerate/bitsandbytes versions installed. Control
pins them (tests/kaggle/t4_smoke/pins/control.txt); canary takes the newest
release of each that Unsloth's declared constraints allow. Either leg failing
alone says something specific:

* canary red, control green -> a library RELEASE broke Unsloth, and the
  canary's report names every resolved version, so the bisect is a diff of two
  reports rather than an investigation.
* both red -> not a version bump: the base image, the model download, the
  Kaggle side, or Unsloth's own code.
* control red, canary green -> the pins no longer resolve, a maintenance signal
  for the pin file rather than a regression.

All three readings break at once if the legs differ in anything but versions,
which is why `--smoke-args` and the reference are shared rather than per leg.

**gptoss** covers `torch.compile` and the forced-float32 path: gpt-oss is in
Unsloth's FORCE_FLOAT32 list precisely because this card has no bf16, and
nothing else in CI exercises it. **grpo** covers vLLM, which
`fast_inference=True` puts on the same 16GB card as the training loop, and
whose sm_75 support is a version-by-version question.

To add a leg, append an entry and name it in a `KERNELS` kernel. The tests in
tests/kaggle/test_t4_smoke_harness.py build every leg and parse every generated
cell, so a leg that cannot produce valid Python never reaches Kaggle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Placeholders substituted at build time from --unsloth-ref / --zoo-ref.
ZOO = "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@{zoo_ref}"
UNSLOTH = "unsloth @ git+https://github.com/unslothai/unsloth@{unsloth_ref}"

# Expands to a pin file's contents, one requirement per argument. Expanded at
# BUILD time, not read on the kernel, so the generated notebook states the
# versions it will install and the tests can read them without executing it.
PINS = "@PINS:{file}"

# What the canary upgrades. Named explicitly rather than "--upgrade
# everything", which would move torch and the CUDA stack too: a leg that
# changes ten things at once cannot attribute a failure to any of them.
CANARY_UPGRADES = ("transformers", "trl", "peft", "accelerate", "bitsandbytes")


@dataclass(frozen = True)
class Leg:
    """One payload: what to install, what to run, and what it is for."""

    name: str
    summary: str
    # pip argument groups, run in order. Each group is one `pip install`.
    install: tuple[tuple[str, ...], ...]
    entry: str
    args: tuple[str, ...] = ()
    # Files copied verbatim from the payload directory into the notebook.
    files: tuple[str, ...] = ()
    # Modules the fail-fast probe imports before spending the session.
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
    # Filename under <payload-dir>/references to band-check against, if any.
    reference: str = ""
    # Extra environment for the child process.
    env: dict = field(default_factory = dict)
    # Does this leg's virtualenv see the Kaggle image's site-packages?
    #
    # True by default, which is what makes the control leg honest: it runs
    # against the image's torch, the situation a notebook user is in, and
    # installs only what the image lacks.
    #
    # False is for a leg that REPLACES torch, and it is measured rather than
    # preferred. `vllm==0.11.2` pins `torch==2.9.0`, downgrading the image's
    # 2.10.0, and with the image's site-packages visible pip treats torch's
    # pinned NVIDIA runtime packages as satisfied by the 2.10 copies, giving a
    # torch that installs cleanly and cannot be imported. Two probe kernels
    # found two faces of it:
    #
    #   libcusparseLt.so.0: cannot open shared object file
    #   libtorch_cuda.so: undefined symbol: ncclCommWindowRegister
    #
    # Naming the packages fixes them one at a time, and the list has no reason
    # to be short. A venv that cannot see the image fixes the class: pip must
    # resolve the whole stack, so it resolves a consistent one, at the cost of
    # a download of a few minutes.
    system_site_packages: bool = True


# Files every leg needs: the version recorder, which makes a red leg
# attributable to a version; the canary dataset; and the "did the optimizer
# apply anything" evidence, without which every payload here can pass on a run
# that trained nothing.
COMMON_FILES = ("versions.py", "canary_dataset.jsonl", "training_evidence.py")

# The install prefix shared by every leg. unsloth_zoo first and WITH deps, then
# unsloth on top, then bitsandbytes, which neither pulls and the image does not
# carry, and without which `import unsloth` raises.
#
# UNSLOTH RESOLVES ITS DEPENDENCIES, and used to carry --no-deps so the overlay
# could not walk the set zoo had just resolved. That made the one file this
# workflow watches for packaging changes -- pyproject.toml is in its trigger
# paths -- the one thing it could not test: pip enforces the requirements of
# packages IN a resolution (see the frontier leg), so with the tested
# distribution outside every resolution, a dependency it adds is never
# installed, one it tightens is never checked against what is here, and the
# import probe still passes whenever the dependency is reached by a delayed
# code path. `pip install unsloth` is what a user runs, and this is now the
# same call.
#
# The --no-deps concern is answered by what unsloth actually declares: typer,
# rich, pydantic, pyyaml, nest-asyncio, structlog and click, none of which zoo
# resolves and none of which any leg pins, so there is nothing here for pip to
# fight over. A pyproject that DOES name one of zoo's packages would move it,
# and that is the regression this exists to show rather than a side effect to
# suppress -- a user's install would move it too.
BASE_INSTALL = ((ZOO,), (UNSLOTH,), ("bitsandbytes",))

# The distribution under test, read off the requirement above rather than
# restated: the verify cell asks pip whether THIS distribution's declared
# requirements are satisfied, and a name that drifted from the one actually
# installed would check nothing and say so quietly.
PACKAGE_UNDER_TEST = UNSLOTH.split("@", 1)[0].strip()

SMOKE_FILES = COMMON_FILES + ("run_t4_smoke.py", "determinism.py")


LEGS: dict[str, Leg] = {
    "control": Leg(
        name = "control",
        summary = "tiny SFT determinism run, pinned library set",
        # Pins go in LAST, as their own resolution step, so they beat what the
        # preceding groups resolved; first, zoo's dependency set would quietly
        # walk them forward again.
        install = BASE_INSTALL + ((PINS.format(file = "control.txt"),),),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES + ("pins/control.txt",),
        reference = "t4_qwen2.5-0.5b.json",
        args = ("--pins", "@ROOT/pins/control.txt"),
    ),
    "canary": Leg(
        name = "canary",
        summary = "the same SFT run on the newest permitted library set",
        # One resolution with the zoo requirement present, so pip picks the
        # newest release of each that zoo's constraints allow. A separate
        # upgrade call would let pip install a version zoo forbids and merely
        # warn, measuring an environment Unsloth never claimed to support.
        install = BASE_INSTALL + ((("--upgrade", ZOO) + CANARY_UPGRADES),),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        # No reference: two library sets do not produce the same fp16
        # trajectory, so band-checking against the control's committed trace
        # would fail on drift rather than on a regression. The canary asserts
        # the version-independent things instead -- the canary string, that the
        # optimizer applied updates, that two fresh processes agreed bitwise
        # WITH EACH OTHER, and that nothing raised. See
        # tests/kaggle/t4_smoke/references/README.md.
        reference = "",
    ),
    "frontier": Leg(
        name = "frontier",
        summary = "the same SFT run on the newest transformers and trl on PyPI",
        # WHY THIS EXISTS, given the canary already says "newest": the canary
        # installs the newest set zoo's metadata ALLOWS, and that ceiling is
        # low. unsloth_zoo/pyproject.toml pins
        #
        #     transformers >=4.51.3,...,<=5.5.0
        #     trl          >=0.18.2,!=0.19.0,<=0.24.0
        #
        # so on 2026-08-11 the canary resolved transformers 5.5.0 against a
        # PyPI latest of 5.15.0, and trl 0.24.0 against a latest of 1.9.2, a
        # whole major version. It moved peft 0.19.1 -> 0.20.0 and accelerate
        # 1.13.0 -> 1.14.0 (both latest, both uncapped), which made the leg look
        # like it was working; two of the five never moved, and they are the two
        # that break most. So with only the canary this CI CANNOT detect a
        # transformers 5.6+ or trl 1.x regression, having never installed one,
        # and the cap is raised only after someone checks -- this is what
        # checks.
        #
        # WITH dependencies, NOT --no-deps. `--no-deps transformers trl` plus a
        # blanket `--upgrade tokenizers` did reach transformers 5.15.0 and trl
        # 1.9.2 (kernel unsloth-t4-ci-bd0c49e5, the first time this CI installed
        # either) and then died before running anything:
        #
        #   tokenizers<=0.23.0,>=0.22.0 is required, but found tokenizers==0.23.1
        #   safetensors>=0.8.0 is required, but found safetensors==0.7.0
        #
        # An unbounded upgrade overshoots transformers' declared ceiling and
        # --no-deps leaves nothing to repair it. Resolving the deps fixes both,
        # because pip enforces only the requirements of packages IN the
        # resolution: unsloth_zoo is merely installed, so its `<=5.5.0` is a
        # warning rather than a ceiling. Dry run against an environment with zoo
        # installed: "Would install datasets-5.0.1 huggingface_hub-1.27.0
        # transformers-5.15.0 trl-1.9.2". So this leg moves whatever
        # transformers and trl now require, which is the honest scope of taking
        # the new version.
        #
        # I expected this leg to go red. IT DOES NOT. Kernel from
        # temp/frontier_kernel2.ipynb on a real T4: transformers 5.15.0, trl
        # 1.9.2, datasets 5.0.1, ten steps, canary emitted, two fresh processes
        # agreeing BITWISE (max_abs_diff 0.0 on both loss and grad_norm).
        # Unsloth trains and generates correctly a whole trl major above what
        # zoo's metadata permits. A red here would be a to-do about the next
        # version bump rather than a broken main, and should be wired so a
        # reader can tell those apart.
        #
        # WHAT IT DOES NOT CATCH, worth a reader's eye: the loss trajectory is
        # not the control's.
        #
        #   control  tf 5.5.0  trl 0.24.0: 10.3222 10.4956 9.9563 10.3892 5.0523 ...
        #   frontier tf 5.15.0 trl 1.9.2 :  6.4367  6.6086 5.9956  3.6721 2.0265 ...
        #
        # Step 1 is computed before any update, on identical initial weights,
        # data and seed, so 10.32 against 6.44 is not optimisation drift: the
        # loss FUNCTION differs, in masking or normalisation. Both converge, so
        # neither is obviously wrong, and this leg has no reference band (see
        # the canary), which is why it passes without noticing. Settling which
        # objective is intended is separate work.
        install = BASE_INSTALL + ((("--upgrade", "transformers", "trl")),),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        # Same reasoning as the canary, more so: this set is further still from
        # the committed trace.
        reference = "",
    ),
    "gptoss": Leg(
        name = "gptoss",
        summary = "gpt-oss-20b LoRA: torch.compile and the float32 path",
        # The base install and nothing else, specifically WITHOUT the
        # `triton_kernels` git dependency the gpt-oss notebook installs. That
        # omission is measured: two probe kernels on 2026-08-11 ran this leg on
        # a T4, one with triton_kernels and torchao (the notebook's own install
        # cell) and one with neither, and produced the SAME three losses to the
        # last bit (5.76492166519165, 4.781009674072266, 4.027626991271973), the
        # same peak memory (12.78 GB) and the same compile counters (32 graphs,
        # 779 calls, 2 breaks).
        #
        # They agree because `load_in_4bit=True` never reaches MXFP4: Unsloth's
        # FLOAT_TO_INT_MAPPER redirects `unsloth/gpt-oss-20b` to the NF4
        # `unsloth/gpt-oss-20b-unsloth-bnb-4bit`, and MXFP4 has no backward pass
        # to reach. triton_kernels would be a pinned git checkout of a
        # third-party repo on every run for no observable effect.
        install = BASE_INSTALL,
        entry = "run_gptoss_t4.py",
        files = COMMON_FILES + ("run_gptoss_t4.py",),
        args = ("--max-steps", "3", "--max-seq-length", "1024"),
    ),
    # NOT WIRED, for a smaller reason than it used to be. See UNWIRED below:
    # the install that killed three probe sessions is re-solved, and what
    # remains is a runtime question needing one session on a real T4.
    "grpo": Leg(
        name = "grpo",
        summary = "Qwen3-4B GRPO through a vLLM engine on the same card",
        # vLLM FIRST and alone: it pins torch, and resolving it after unsloth
        # walks torch underneath an already installed stack.
        #
        # THE VERSION MATCHES THE IMAGE, IT IS NOT MERELY OLD. Kaggle ships
        # torch 2.10.0+cu128, and vLLM's torch pin by release is
        #
        #   0.11.2 .. 0.16.0   torch==2.9.0 / 2.9.1
        #   0.17.0 .. 0.19.1   torch==2.10.0      <- the whole window
        #   0.20.0 .. 0.26.0   torch==2.11.0
        #   0.27.0 ..          torch==2.13.0
        #
        # Every other choice REPLACES the image's torch, which is what all three
        # probe sessions died of: the image's NVIDIA runtime packages belong to
        # 2.10 and pip treats them as satisfying the new torch's pins. 0.19.1 is
        # the newest release needing no replacement, so the install is ordinary
        # and the leg keeps `system_site_packages`.
        #
        # No xformers: its vLLM attention backend was deleted in 0.12.0, so it
        # would install a package nothing selects. sm_75 has no FlashAttention
        # and no FlashInfer, and the ladder in vllm/platforms/cuda.py falls
        # through those to TRITON_ATTN, pinned below rather than left to a probe
        # order that moves between releases. sm_75 is still in
        # CUDA_SUPPORTED_ARCHS at v0.19.1, and fp16 is supported below
        # capability 8.0.
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
        # ALL FIVE ARE LOAD-BEARING ON A 14.56GB CARD, and are the values that
        # passed rather than the ones that look reasonable. Two probes with the
        # notebook's own settings (seq 2048, 4 generations, rank 32, utilization
        # 0.9) died in the BACKWARD at
        # unsloth_zoo/gradient_checkpointing.py:1013, peaking at 15.97GB in
        # 16-bit and 19.25GB in 4-bit.
        #
        # 4-bit is not the lever it looks like -- it peaked HIGHER than 16-bit,
        # since quantizing weights does nothing for activations while
        # utilization 0.9 still hands vLLM ~13GB up front. UNSLOTH_VLLM_STANDBY
        # returns the weights during training but not the KV cache reservation,
        # so utilization is what decides whether a backward has anywhere to run.
        #
        # Measured on kernel unsloth-t4-ci-53efcc4e: peak 13.60GB allocated of
        # 14.56GB, three steps in 192s.
        args = (
            "--max-steps",
            "3",
            "--load-in-4bit",
            "--gpu-memory-utilization",
            "0.5",
            "--max-seq-length",
            "1024",
            "--num-generations",
            "2",
            "--lora-rank",
            "16",
        ),
        env = {
            "UNSLOTH_VLLM_STANDBY": "1",
            # See the install comment. Named rather than probed so a release
            # reordering the ladder turns this leg red instead of silently
            # selecting something else.
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            # Kaggle cannot link what flashinfer JIT-compiles. On kernel
            # unsloth-t4-ci-e2d9ce9b, vLLM 0.19.1 reached engine construction on
            # a real T4, flashinfer 0.6.6 compiled all three sampling .cu files
            # CLEANLY for `-gencode=arch=compute_75,code=sm_75`, and the link
            # died on
            #
            #   /usr/bin/ld: cannot find -lcuda
            #
            # `-L/usr/local/cuda/lib64/stubs` is on the command line, so the
            # driver stub `libcuda.so` is simply absent from this image; only
            # the runtime `libcuda.so.1` is there. That is not an sm_75 problem
            # and is not fixable from here. So do not JIT: the sampler has a
            # native path, and skipping the build saves a four-file nvcc compile
            # in a session billed by wall clock.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        },
        # Now true, which is the point of the version choice above: this leg no
        # longer replaces torch, so it shares the image's view instead of
        # resolving a whole CUDA stack from scratch. Probe 3 spent about an hour
        # of quota doing that and never got past venv creation.
        system_site_packages = True,
    ),
}


# How many legs one kernel packs. This used to be 2, the card count, because
# legs were started all at once and one per card: a third would have SHARED a
# card and quietly changed what both of the legs on it measured.
#
# They now queue -- build_kernel.py runs one worker per card, and a card takes
# its next leg only when the previous one has exited -- so a leg beyond the
# second waits rather than sharing, and the card count no longer caps this.
# What caps it is wall clock: every leg past the second adds its whole runtime
# to one card's column, the session is killed at 12 hours, and the launcher's
# own ceiling is lower still. At 4 legs the longest column is about 11 minutes,
# so there is room, but this is a number to raise deliberately and measure
# after, not to grow by accident.
MAX_LEGS_PER_KERNEL = 4

# Legs defined here and deliberately NOT run, with the reason. A leg is unwired
# rather than deleted when the payload is right and the environment is not, so
# the next person to try owes nothing but a working install. Every entry must
# say what was measured.
UNWIRED: dict[str, str] = {
    # A leg belongs here only while a specific unanswered question about it
    # would be answered by a session. "Not tried yet" is not that.
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
        "It also exposed a SECOND problem, and the two are separate. That run "
        "failed on reward_std = [0.0, 0.0, 0.0] with grad_norm 0.0 at every "
        "step. The completions recorded in the report are coherent prose, not "
        "degenerate, so this is not the model collapsing -- it is two "
        "completions scoring identically. The leg runs num_generations = 2, "
        "shrunk to fit a 14.56GB card, and at two samples a tie on a coarse "
        "reward is ordinary rather than a bug. So the leg's own pass "
        "criterion is fragile at the size it has to be to fit.\n"
        "STILL UNKNOWN: where the race is (the standby wake/sleep cycle on "
        "sm_75 is the suspect, and UNSLOTH_VLLM_STANDBY=1 is set in all four "
        "sessions), and what pass criterion is honest at num_generations = 2."
    ),
}

# Which legs travel in which kernel. ONE kernel, holding every leg, which its
# 2xT4 session works through two at a time -- one per card, the next leg
# starting on a card only when that card's previous leg has exited.
#
# This was two kernels, and the reason it is one now is NOT quota. A session
# bills its wall clock once rather than per card, so two kernels of two legs
# cost 662s of billing against 646s for one kernel of four: a rounding error.
# What two kernels cost is the whole ACCOUNT. Kaggle allows two concurrent GPU
# sessions, two kernels take both, and kaggle-t4-studio-gpu-ci.yml runs on the
# same account -- so the notebook leg locked Unsloth out entirely for as long as
# it ran (measured: Unsloth's run 32607617804 queued ~40 minutes behind notebook
# run 32607621452). One kernel holds one session and leaves the other free, and
# the two workflows now hold separate GitHub concurrency groups so they can
# actually use it. Splitting the group without packing the kernel, or packing
# without splitting the group, each make things worse on their own.
#
# Ordered LONGEST EXPECTED LEG FIRST, and that ordering is load-bearing.
# build_kernel.py hands this order to the driver as its start order and a card
# takes work greedily, so putting gptoss (384s, and the leg that sets the
# makespan) anywhere but first leaves the schedule unable to balance around it.
# Measured on run 32607621452: gptoss 384.1s, frontier 312.2s, canary 265.3s,
# control 262.2s, which this order packs as 384.1+262.2 = 646.3s against
# 312.2+265.3 = 577.5s. That is the optimal split of these four; the next best
# pairing is 649.4s, and perfect balance would be 611.9s, so the 68.8s of idle
# at the end is 34.4s of genuinely unavoidable imbalance and not a packing bug.
#
# control and canary stay in the same kernel, which is what their comparison
# needs: same image, same driver, same hour. They no longer run on the two
# cards SIMULTANEOUSLY, which is fine -- they were never compared against each
# other, only each against its own committed reference, and neither reads a
# clock. What would break them is landing in different SESSIONS, and packing
# everything into one kernel makes that impossible rather than merely unlikely.
#
# `grpo` returns here once the illegal memory access in UNWIRED is understood.
# It briefly had a kernel of its own, on the reasoning that pairing with gpt-oss
# broke it: it failed paired and had passed alone. Running it ALONE again
# (unsloth-t4-ci-c98f14be) reproduced the paired failure exactly, same stack,
# same 13.8GB peak, same engine_built false, so the pairing was never the
# variable and one contrasting observation was not enough to blame a shared
# host. It stays unwired rather than re-paired, since a leg passing one session
# in three tells CI nothing either way.
KERNELS: tuple[tuple[str, ...], ...] = (("gptoss", "frontier", "canary", "control"),)


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
    """Legs by name, in the order given. Unknown names fail loudly here.

    At build time rather than on the kernel: a typo in a workflow input must
    cost a runner second, not a Kaggle session.
    """
    legs = []
    for name in names:
        if name not in LEGS:
            raise SystemExit(f"unknown leg {name!r}; known legs are {', '.join(sorted(LEGS))}")
        legs.append(LEGS[name])
    return legs
