# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The payloads this CI can run, and what each one is FOR.

A Kaggle session is 2xT4 and the account allows 2 concurrent batch kernels,
so the ceiling is four payloads at once. This file is the registry of what
those four are. Each entry is a *leg*: an install recipe, a script, and the
arguments it runs with. `build_kernel.py` turns a list of legs into kernel
notebooks; nothing else in this directory knows what a leg contains.

The four legs, and why this particular four
-------------------------------------------
**control** and **canary** are a matched pair and the core of the design.
They run the SAME payload, seed, dataset and step count on the same card.
The only difference between them is the version of transformers, trl, peft,
accelerate and bitsandbytes that gets installed: the control pins them
(tests/kaggle/t4_smoke/pins/control.txt), the canary takes the newest
release of each that Unsloth's own declared constraints allow.

That pairing is the whole instrument. Either leg failing alone says
something specific:

* canary red, control green -> a library RELEASE broke Unsloth. The canary's
  report names the resolved version of every package in the set, so the
  bisect is a diff of two reports rather than an investigation.
* both red -> not a version bump. The base image, the model download, the
  Kaggle side, or Unsloth's own code.
* control red, canary green -> the pins no longer resolve, which is a
  maintenance signal for the pin file and not a regression.

If they ever differ in anything but versions, all three of those readings
become wrong at once, so `--smoke-args` and the reference are deliberately
shared between them rather than configured per leg.

**gptoss** covers `torch.compile` and the forced-float32 path. gpt-oss is in
Unsloth's FORCE_FLOAT32 list precisely because this card has no bf16, and
that path exists for T4 and is exercised by nothing else in CI.

**grpo** covers vLLM. `fast_inference=True` puts a vLLM engine and a
training loop on the same 16GB card, and vLLM's support for sm_75 is a
version-by-version question rather than a settled one.

Adding a leg
------------
Append an entry and add its name to a kernel in `KERNELS`. The tests in
tests/kaggle/test_t4_smoke_harness.py build every leg and parse every
generated cell, so a leg that cannot produce valid Python never reaches a
Kaggle session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Placeholders substituted at build time from --unsloth-ref / --zoo-ref.
ZOO = "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@{zoo_ref}"
UNSLOTH = "unsloth @ git+https://github.com/unslothai/unsloth@{unsloth_ref}"

# The sentinel that expands to the contents of a pin file, one requirement
# per argument. Expanded at BUILD time rather than read on the kernel, so the
# generated notebook states the versions it is about to install and the test
# suite can read them without executing anything.
PINS = "@PINS:{file}"

# The version set the canary leg upgrades. Named explicitly rather than
# "--upgrade everything": upgrading the whole environment would move torch
# and the CUDA stack too, and a leg that changes ten things at once cannot
# attribute a failure to any of them.
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
    # True is the default and is what makes the control leg honest: it runs
    # against the image's torch, which is the situation a notebook user is
    # in, and installs only what the image lacks.
    #
    # False is for a leg that REPLACES torch, and it is not a preference. It
    # is the measured consequence of trying the other way. `vllm==0.11.2`
    # pins `torch==2.9.0`, so installing it downgrades the image's 2.10.0,
    # and with the image's site-packages visible pip treats torch's pinned
    # NVIDIA runtime packages as already satisfied by the copies that belong
    # to 2.10. The result is a torch that installs cleanly and cannot be
    # imported. Two probe kernels found two different faces of it:
    #
    #   libcusparseLt.so.0: cannot open shared object file
    #   libtorch_cuda.so: undefined symbol: ncclCommWindowRegister
    #
    # Naming the packages individually fixes them one at a time and there is
    # no reason to think the list is short. A venv that cannot see the image
    # fixes the class: pip has to resolve the whole stack, so it resolves a
    # consistent one. The cost is the download, which is minutes.
    system_site_packages: bool = True


# Files every leg needs: the version recorder and the canary dataset. The
# recorder is what makes any red leg attributable to a version, so no leg is
# allowed to ship without it.
COMMON_FILES = ("versions.py", "canary_dataset.jsonl")

# The install prefix shared by every leg. unsloth_zoo first and WITH deps,
# then unsloth --no-deps on top, so the overlay does not fight the dependency
# set zoo resolved; then bitsandbytes, which neither of them pulls and the
# Kaggle image does not carry, and without which `import unsloth` raises.
BASE_INSTALL = ((ZOO,), ("--no-deps", UNSLOTH), ("bitsandbytes",))

SMOKE_FILES = COMMON_FILES + ("run_t4_smoke.py", "determinism.py")


LEGS: dict[str, Leg] = {
    "control": Leg(
        name = "control",
        summary = "tiny SFT determinism run, pinned library set",
        # The pins go in LAST and as their own resolution step, so they beat
        # whatever the preceding groups resolved. Installing them first would
        # let zoo's own dependency set quietly walk them forward again.
        install = BASE_INSTALL + ((PINS.format(file = "control.txt"),),),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES + ("pins/control.txt",),
        reference = "t4_qwen2.5-0.5b.json",
        args = ("--pins", "@ROOT/pins/control.txt"),
    ),
    "canary": Leg(
        name = "canary",
        summary = "the same SFT run on the newest permitted library set",
        # One resolution, with the zoo requirement present, so pip picks the
        # newest release of each that zoo's constraints actually allow.
        # Upgrading them in a separate call would let pip install a version
        # zoo forbids and merely warn about it, and the leg would then be
        # measuring an environment Unsloth never claimed to support.
        install = BASE_INSTALL + ((("--upgrade", ZOO) + CANARY_UPGRADES),),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        # No reference. Two library sets do not produce the same fp16
        # trajectory, so band-checking the canary against the control's
        # committed trace would fail on drift rather than on a regression.
        # What the canary asserts instead is everything that is
        # version-independent: the canary string, that the optimizer applied
        # updates, that two fresh processes agreed bitwise WITH EACH OTHER,
        # and that nothing raised. See tests/kaggle/t4_smoke/references/README.md.
        reference = "",
    ),
    "gptoss": Leg(
        name = "gptoss",
        summary = "gpt-oss-20b LoRA: torch.compile and the float32 path",
        # The base install and nothing else, and specifically WITHOUT the
        # `triton_kernels` git dependency the gpt-oss notebook installs.
        #
        # That omission is measured, not assumed. Two probe kernels on
        # 2026-08-11 ran this leg on a T4, one with triton_kernels and
        # torchao (the notebook's own install cell) and one with neither,
        # and they produced the SAME three losses to the last bit:
        # 5.76492166519165, 4.781009674072266, 4.027626991271973. They
        # agreed on the peak memory (12.78 GB) and on the compile counters
        # (32 graphs, 779 calls, 2 breaks) too.
        #
        # The reason they agree is that `load_in_4bit=True` never reaches
        # MXFP4 at all: Unsloth's FLOAT_TO_INT_MAPPER redirects
        # `unsloth/gpt-oss-20b` to `unsloth/gpt-oss-20b-unsloth-bnb-4bit`,
        # an NF4 checkpoint, and MXFP4 has no backward pass to reach. So
        # triton_kernels would be a pinned git checkout of a third-party
        # repository, on every run, for no observable effect.
        install = BASE_INSTALL,
        entry = "run_gptoss_t4.py",
        files = COMMON_FILES + ("run_gptoss_t4.py",),
        args = ("--max-steps", "3", "--max-seq-length", "1024"),
    ),
    # NOT WIRED, but for a smaller reason than it used to be. See UNWIRED
    # below: the install that killed three probe sessions has been re-solved
    # and what remains is a runtime question that needs one session on a
    # real T4.
    "grpo": Leg(
        name = "grpo",
        summary = "Qwen3-4B GRPO through a vLLM engine on the same card",
        # vLLM FIRST and alone, because it pins torch and letting it resolve
        # after unsloth means pip walks torch underneath an already installed
        # stack.
        #
        # THE VERSION IS CHOSEN TO MATCH THE IMAGE, NOT TO BE OLD. Kaggle's
        # image ships torch 2.10.0+cu128. vLLM's torch pin by release:
        #
        #   0.11.2 .. 0.16.0   torch==2.9.0 / 2.9.1
        #   0.17.0 .. 0.19.1   torch==2.10.0      <- the whole window
        #   0.20.0 .. 0.26.0   torch==2.11.0
        #   0.27.0 ..          torch==2.13.0
        #
        # Every other choice REPLACES the image's torch, and replacing it is
        # what all three probe sessions died of: the image's NVIDIA runtime
        # packages belong to 2.10 and pip treats them as satisfying the new
        # torch's pins. 0.19.1 is the newest release that needs no
        # replacement at all, so the install is a normal one and the leg can
        # keep `system_site_packages`.
        #
        # No xformers. Its vLLM attention backend was deleted in 0.12.0, so
        # carrying it here would install a package nothing selects. sm_75 has
        # no FlashAttention and no FlashInfer, and the backend ladder in
        # vllm/platforms/cuda.py falls through those to TRITON_ATTN, which is
        # what this leg pins below rather than leaving to a probe order that
        # moves between releases. sm_75 is still in CUDA_SUPPORTED_ARCHS at
        # v0.19.1, and fp16 is a supported dtype below capability 8.0.
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
        # EVERY ONE OF THESE FIVE IS LOAD-BEARING ON A 14.56GB CARD, and the
        # values are the ones that passed rather than the ones that look
        # reasonable. Two probes with the notebook's own settings -- seq 2048,
        # 4 generations, rank 32, utilization 0.9 -- died in the BACKWARD at
        # unsloth_zoo/gradient_checkpointing.py:1013, peaking at 15.97GB in
        # 16-bit and 19.25GB in 4-bit.
        #
        # 4-bit is not the lever it looks like: it peaked HIGHER than 16-bit,
        # because quantizing weights does nothing for activations while
        # utilization 0.9 still hands vLLM ~13GB up front. UNSLOTH_VLLM_STANDBY
        # returns the weights during training but not the KV cache
        # reservation, so the utilization figure is the one that decides
        # whether a backward has anywhere to run.
        #
        # Measured on kernel unsloth-t4-ci-53efcc4e: peak 13.60GB allocated of
        # 14.56GB, three steps in 192s.
        args = (
            "--max-steps", "3",
            "--load-in-4bit",
            "--gpu-memory-utilization", "0.5",
            "--max-seq-length", "1024",
            "--num-generations", "2",
            "--lora-rank", "16",
        ),
        env = {
            "UNSLOTH_VLLM_STANDBY": "1",
            # See the install comment. Named rather than probed so that a
            # release reordering the ladder shows up as this leg going red,
            # not as it silently selecting something else.
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            # Kaggle cannot link what flashinfer JIT-compiles. Measured on
            # kernel unsloth-t4-ci-e2d9ce9b: vLLM 0.19.1 reached engine
            # construction on a real T4, flashinfer 0.6.6 started building its
            # sampling ops, all three .cu files compiled CLEANLY for
            # `-gencode=arch=compute_75,code=sm_75`, and the link step died on
            #
            #   /usr/bin/ld: cannot find -lcuda
            #
            # `-L/usr/local/cuda/lib64/stubs` is on the command line, so the
            # driver stub `libcuda.so` is simply not in this image; only the
            # runtime `libcuda.so.1` is. Nothing about that is sm_75, and
            # nothing about it is fixable from here.
            #
            # So do not JIT at all. The sampler has a native path, and skipping
            # the build also saves a four-file nvcc compile inside a session
            # billed by wall clock.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        },
        # Now true, and that is the point of the version choice above: this
        # leg no longer replaces torch, so it can share the image's view
        # instead of resolving a whole CUDA stack from scratch. Probe 3 spent
        # about an hour of quota doing exactly that and never got past venv
        # creation.
        system_site_packages = True,
    ),
}


# GPUs in a Kaggle session, and therefore legs one kernel can carry. A third
# leg in a kernel would not fail; it would share a card with another and
# quietly change what both of them measure.
MAX_LEGS_PER_KERNEL = 2

# Legs that are defined here and deliberately NOT run, with the reason.
#
# A leg is unwired rather than deleted when the payload is right and the
# environment is not: the next person to try owes nothing but a working
# install, and deleting it would mean rediscovering the install problem from
# scratch. Every entry must say what was measured.
UNWIRED: dict[str, str] = {
    # Empty on purpose. Every leg in LEGS is in KERNELS.
    #
    # `grpo` was the last entry here and came out on 2026-08-11 after kernel
    # unsloth-t4-ci-53efcc4e passed on a real Tesla T4: reward_std 0.707 and
    # grad_norm 0.772 at step 2, peak 13.60GB of 14.56GB, three steps in 192s.
    # The four blockers it cleared on the way are recorded where each fix
    # lives -- the vLLM/torch pin and the attention backend in the leg's
    # install comment, the flashinfer link shim and the base-model chat
    # template in run_grpo_t4.py, and the T4-sized training config in the
    # leg's args.
    #
    # A leg belongs here only while there is a specific unanswered question
    # about it that a session would answer. "Not tried yet" is not that.
}

# Which legs travel in which kernel. One entry per kernel, and a kernel runs
# its legs one per T4 of its session.
#
# The pairing is not arbitrary. control and canary share a kernel so they run
# on the two cards of the SAME session: same image, same driver, same hour.
# Splitting them across sessions would put an uncontrolled variable between
# the only two legs whose comparison has to be clean.
#
# The second kernel carries one leg and has a free T4, which costs nothing:
# a session bills its wall clock once, not per card. That card is where
# `grpo` goes when its install works.
KERNELS: tuple[tuple[str, ...], ...] = (
    ("control", "canary"),
    ("gptoss", "grpo"),
)


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

    Failing at build time rather than on the kernel is the point: a typo in
    a workflow input must cost a runner second, not a Kaggle session.
    """
    legs = []
    for name in names:
        if name not in LEGS:
            raise SystemExit(f"unknown leg {name!r}; known legs are {', '.join(sorted(LEGS))}")
        legs.append(LEGS[name])
    return legs
