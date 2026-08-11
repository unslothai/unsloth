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
    imports: tuple[str, ...] = ("torch", "transformers", "trl", "peft",
                                "datasets", "bitsandbytes", "unsloth",
                                "unsloth_zoo")
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
    # NOT WIRED. See UNWIRED below for what three probe sessions found and
    # what has to change before this can run on quota anyone is watching.
    "grpo": Leg(
        name = "grpo",
        summary = "Qwen3-4B GRPO through a vLLM engine on the same card",
        # vLLM FIRST and alone. It pins torch, and letting it resolve after
        # unsloth means pip walks torch backwards underneath an already
        # installed stack. The version is pinned because vLLM's support for
        # sm_75 is a per-release question; see the leg's payload docstring.
        # xformers travels WITH vllm and in the same resolution. On sm_75
        # there is no FlashAttention, so the xformers backend is the only
        # one vLLM can select; installing it afterwards would let the
        # resolver pick a build against a torch vllm had already replaced.
        install = (("vllm==0.11.2", "xformers"),) + BASE_INSTALL,
        entry = "run_grpo_t4.py",
        files = COMMON_FILES + ("run_grpo_t4.py",),
        imports = ("torch", "transformers", "trl", "peft", "datasets",
                   "bitsandbytes", "vllm", "unsloth", "unsloth_zoo"),
        args = ("--max-steps", "3"),
        env = {"UNSLOTH_VLLM_STANDBY": "1"},
        # See the field's own comment. This leg replaces torch, so it cannot
        # share a view of the image that still holds torch's old NVIDIA
        # runtime packages.
        system_site_packages = False,
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
    "grpo": (
        "Three probe sessions on 2026-08-11 never reached a training step, "
        "and none of them failed for a reason to do with sm_75, memory or "
        "GRPO. `vllm==0.11.2` pins `torch==2.9.0`, so this leg has to REPLACE "
        "the Kaggle image's torch 2.10.0+cu128, and that is where all three "
        "died.\n\n"
        "  1. venv seeing the image (kernels 8161ceb9, 7ab727f1): both cards "
        "     died at `import torch` with `libcusparseLt.so.0: cannot open "
        "     shared object file`. pip had treated torch's pinned NVIDIA "
        "     runtime packages as satisfied by the image's copies, which "
        "     belong to 2.10.\n"
        "  2. that package named and force-reinstalled (kernel f88c929b): "
        "     the cusparseLt error cleared and the next one appeared one "
        "     package along, `libtorch_cuda.so: undefined symbol: "
        "     ncclCommWindowRegister`. IDENTICAL on vllm 0.11.2 and 0.15.1, "
        "     so at this stage it is not a vLLM-version question at all.\n"
        "  3. fully isolated venv, pip resolving the whole stack (kernel "
        "     9ac72efe): the session produced no payload output past venv "
        "     creation, Kaggle's own nbconvert of the kernel failed at t=406s "
        "     with NotJSONError, and the session then sat in RUNNING past its "
        "     own 5400s ceiling until it was deleted by hand, about an hour "
        "     of quota later.\n\n"
        "Wiring it in this state would make the check permanently red and "
        "would spend the budget doing it. What is still unknown, and needs a "
        "session that gets further than these did: whether 8GB of 16-bit "
        "weights plus a vLLM engine plus a LoRA trainer fit in 14.56GB, and "
        "whether sm_75 still has an attention backend at either vLLM "
        "version. The payload asserts on reward and reward_std rather than "
        "loss and is ready for that session; the install is not."),
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
    ("gptoss",),
)


def expand_install(leg: Leg, *, unsloth_ref: str, zoo_ref: str,
                   payload_dir: Path) -> list[list[str]]:
    """Resolve a leg's install groups into concrete pip argument lists."""
    groups: list[list[str]] = []
    for group in leg.install:
        expanded: list[str] = []
        for item in group:
            if item.startswith("@PINS:"):
                expanded.extend(_read_pins(payload_dir / "pins"
                                           / item[len("@PINS:"):]))
                continue
            expanded.append(item.format(unsloth_ref = unsloth_ref,
                                        zoo_ref = zoo_ref))
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
        raise ValueError(f"pin file {path} names no versions at all, so the "
                         f"control leg would pin nothing")
    return out


def resolve(names) -> list[Leg]:
    """Legs by name, in the order given. Unknown names fail loudly here.

    Failing at build time rather than on the kernel is the point: a typo in
    a workflow input must cost a runner second, not a Kaggle session.
    """
    legs = []
    for name in names:
        if name not in LEGS:
            raise SystemExit(f"unknown leg {name!r}; known legs are "
                             f"{', '.join(sorted(LEGS))}")
        legs.append(LEGS[name])
    return legs
