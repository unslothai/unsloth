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

# Expands to a pin file's contents, one requirement per argument, at BUILD time so the generated notebook states the versions it will install and the tests can read them without executing it.
PINS = "@PINS:{file}"

# What the canary upgrades. Named explicitly rather than "--upgrade everything", which would move torch and the CUDA stack too: a leg that changes ten things at once cannot attribute a failure to any of them.
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
    # Measured peak_reserved_gb on a Tesla T4 (run 32611343797), NOT an estimate and not a request: the driver uses it to decide whether two legs may share a card. gptoss 12.78 of 14.56 (88%, so it never shares); the three Qwen legs 0.70 each. A leg whose real appetite grows past what it declares here is the failure mode, so test_the_declared_vram_matches_what_the_legs_reported checks these against the evidence.
    vram_gb: float = 1.0
    # Pure-Python version pins this leg wants OVER the shared base, installed into a per-leg directory and put on PYTHONPATH ahead of it. Same mechanism Studio ships as `.venv_t5_*` (studio/backend/utils/transformers_version.py): --no-deps, prepended, propagated to children via PYTHONPATH. Measured on kernel unsloth-probe-overlay-t4-r2-38ac4d, `transformers==4.57.6 + trl~=0.22.0` resolved to three packages, 115.9 MB, in 10.0s, against 158-207s for a leg's own install groups. Sound only for pure-Python or wheel-shipped distributions: it CANNOT replace torch, triton or bitsandbytes, and nothing may import the ambient copy before the child starts, which is why the overlay reaches the payload as an env var rather than a sys.path edit inside a running interpreter. Empty means "use the base as-is".
    overlay: tuple[str, ...] = ()
    # Distributions to REMOVE after the install groups run; empty for every leg but grpo. FlashInfer cannot LINK on the Kaggle image: it ships the runtime libcuda.so.1 but not the driver stub libcuda.so, so `-lcuda` fails after nvcc has succeeded. Four ladder probes established that no environment variable covers this: VLLM_USE_FLASHINFER_SAMPLER=0 changed nothing, and UNSLOTH_VLLM_NO_FLASHINFER=1 stopped the sampler build but the failure moved to the attention prefill kernels, for which vLLM exposes no knob. Removing the three flashinfer distributions after installing vLLM passed at the first rung, covers every caller, and flips unsloth's own find_spec("flashinfer") test to False, a branch its code already handles.
    uninstall: tuple[str, ...] = ()
    reference: str = ""
    env: dict = field(default_factory = dict)
    # Does this leg's virtualenv see the Kaggle image's site-packages? True by default, which is what makes the control leg honest: it runs against the image's torch, as a notebook user does. False is for a leg that REPLACES torch, and it is measured: `vllm==0.11.2` pins `torch==2.9.0`, downgrading the image's 2.10.0, and with the image's site-packages visible pip treats torch's pinned NVIDIA runtime packages as satisfied by the 2.10 copies, giving a torch that installs cleanly and cannot be imported (libcusparseLt.so.0 missing, and libtorch_cuda.so undefined symbol ncclCommWindowRegister). A venv that cannot see the image forces pip to resolve the whole stack, so it resolves a consistent one, at the cost of a few minutes of download.
    system_site_packages: bool = True
    # Does this leg want EVERY card visible rather than being pinned to one? False for every leg but `multi_gpu`, and it exists for a branch no pinned leg can reach: unsloth/kernels/utils.py:170 binds torch_gpu_device to torch.cuda.device only when DEVICE_COUNT > 1, and build_kernel.py pins every payload with CUDA_VISIBLE_DEVICES, so every unsloth kernel this CI has run took the nullcontext shim. The same holds for the per-device rotary caches (llama.py:1838, gemma.py:280), the DEVICE_COUNT-sized CUDA_STREAMS / WEIGHT_BUFFERS / ABSMAX_BUFFERS arrays and the temp_mlp device tuples (llama.py:1300). It is FREE because the divergence is triggered by VISIBILITY, not by using both cards: the model still fits on one.
    all_cards: bool = False


# Files every leg needs: the version recorder, which makes a red leg attributable to a version; the canary dataset; and the "did the optimizer apply anything" evidence, without which every payload here can pass on a run that trained nothing.
COMMON_FILES = (
    "versions.py",
    "canary_dataset.jsonl",
    "training_evidence.py",
    # Splits the one `load` figure into fetch and weight load. Shipped to every leg, because the number it produces is only comparable across legs if every leg can produce it.
    "phase_timers.py",
)

# The install prefix shared by every leg: unsloth_zoo first and WITH deps, then unsloth on top, then bitsandbytes, which neither pulls and the image does not carry, and without which `import unsloth` raises. UNSLOTH RESOLVES ITS DEPENDENCIES: it used to carry --no-deps, which made pyproject.toml, the one file this workflow watches for packaging changes, the one thing it could not test, since pip enforces only the requirements of packages IN a resolution. `pip install unsloth` is what a user runs, and this is now the same call. What unsloth declares (typer, rich, pydantic, pyyaml, nest-asyncio, structlog, click) is nothing zoo resolves and nothing any leg pins, so pip has nothing to fight over; a pyproject that DOES name one of zoo's packages would move it, which is the regression this exists to show.
BASE_INSTALL = ((ZOO,), (UNSLOTH,), ("bitsandbytes",))

# The distribution under test, read off the requirement above rather than restated: the verify cell asks pip whether THIS distribution's declared requirements are satisfied, and a name that drifted from the one actually installed would check nothing and say so quietly.
PACKAGE_UNDER_TEST = UNSLOTH.split("@", 1)[0].strip()

SMOKE_FILES = COMMON_FILES + (
    "run_t4_smoke.py",
    "determinism.py",
    "gguf_export.py",
    # Shipped to EVERY smoke leg, not only the one that passes --compare-naive-trl: run_t4_smoke imports comparison_failures at module scope, so a leg without the file cannot start at all. The module itself imports nothing heavier than the stdlib.
    "naive_trl_compare.py",
    # Imported at module scope by run_t4_smoke, like naive_trl_compare, so a leg without it cannot start even though only one leg passes the flag.
    "kernel_provenance.py",
)


LEGS: dict[str, Leg] = {
    "control": Leg(
        vram_gb = 0.7,
        name = "control",
        summary = "tiny SFT determinism run, pinned library set",
        # Pins go in LAST, as their own resolution step, so they beat what the preceding groups resolved; otherwise zoo's dependency set would quietly walk them forward again.
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
        # One resolution with the zoo requirement present, so pip picks the newest release of each that zoo's constraints allow. A separate upgrade call would let pip install a version zoo forbids and merely warn, measuring an environment Unsloth never claimed to support.
        install = BASE_INSTALL + ((("--upgrade", ZOO) + CANARY_UPGRADES),),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        # No reference: two library sets do not produce the same fp16 trajectory, so band-checking against the control's committed trace would fail on drift rather than on a regression. The canary asserts the version-independent things instead (the canary string, that the optimizer applied updates, that two fresh processes agreed bitwise WITH EACH OTHER, and that nothing raised). See tests/kaggle/t4_smoke/references/README.md.
        reference = "",
    ),
    "latest_compile": Leg(
        # MEASURED on one card, twice: 12.73 GB peak reserved on both cycles of unsloth-probe-latestcompile-r5-45cf5b, built by the real --all-kernels path so each payload sees exactly one T4. The placeholder was 6.0, at which the scheduler co-tenants this leg and the pair asks for ~19 GB of a 14.56 GB card, so the OOM comes back reading like a code failure. E2B is a MatFormer submodel of E4B and the checkpoint carries the larger weights, which is why the 4bit resident set is nowhere near "far smaller". 12.8 rather than 12.73 leaves a hair of headroom while staying under CARD_VRAM_BUDGET_GB.
        vram_gb = 12.8,
        name = "Latest_compile",
        summary = "gemma-4-E2B-it SFT on the newest transformers and trl, against plain TRL",
        # Same resolution shape as `frontier`, WITH dependencies: an unbounded `--no-deps` upgrade overshoots transformers' own tokenizers and safetensors floors and dies before running anything (measured on kernel unsloth-t4-ci-bd0c49e5; see the frontier comment below).
        install = BASE_INSTALL + ((("--upgrade", "transformers", "trl")),),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        args = (
            "--model",
            "unsloth/gemma-4-E2B-it",
            "--max-steps",
            "10",
            # The plain-TRL comparison runs in a SECOND process on the same venv with unsloth never imported, and it REPORTS both step traces rather than asserting they match: `frontier` showed transformers 5.5.0 and 5.15.1 producing different step-1 losses on identical weights, data and seed (10.3222 against 6.4367), so an equality assertion would be red on drift. What it DOES assert is that both paths train and both converge. NO EXPORT here, deliberately: it worked (unsloth-probe-lcleg-tmpdir-ac53ca produced a 4725.1MB gemma-4-e2b-it.Q8_0.gguf plus a 940.0MB F16-mmproj, llama-bench rc=0 at 16.18 t/s pp8) but cost 310.8s, the largest line item in the suite, for a claim `default` makes in 40.6s and `vision_fla_compile` in 99.3s through the same prebuilt binaries. Put it back with --export-gguf on a dispatch if a gemma-specific conversion is ever in doubt.
            "--compare-naive-trl",
            # Measured on kernels unsloth-probe-latestcompile-r4-e67ef2 and -r5-45cf5b: the plain arm asks for 8.75GiB with 8.96GiB already resident on a 14.56GiB T4, at LOAD, and gradient checkpointing changed the number not at all. E2B is a MatFormer SUBMODEL of E4B and the checkpoint carries the larger weights, so a loader that does not extract the submodel materialises all of them: a fact about the card and the checkpoint, not about either training stack. An OOM DURING training is still a failure; this only excuses the load.
            "--control-oom-is-ok",
        ),
        # No reference, for the frontier/canary reason: two library sets do not produce one fp16 trajectory, and a band here would go red on ordinary cross-version drift. See references/README.md.
        reference = "",
    ),
    # NOT WIRED. The recon probe (unsloth-probe-vision-recon-c76ea3) answered the kernel questions and left ONE open, which is the one the scheduler needs. See UNWIRED.
    "vision_fla_compile": Leg(
        # MEASURED on ONE card on the COMPLETE leg (text cycles, vision training run and Q8_0 export): unsloth-probe-visleg-full-b3a317 read 2.84 GB peak reserved on both cycles. The earlier 0.92 GB recon figure was one card's share of a two-card placement, since a probe body sees both cards where a leg gets one.
        vram_gb = 2.84,
        name = "Vision_FLA_compile",
        summary = "Qwen3.5-2B on the newest stack: vendored FLA, sdpa on Turing, completions-only",
        install = BASE_INSTALL + ((("--upgrade", "transformers", "trl")),),
        entry = "run_t4_smoke.py",
        # The vision payload travels with this leg as a SEPARATE entry script, not a flag on run_t4_smoke: a vision run needs FastVisionModel, UnslothVisionDataCollator and four SFTConfig settings that would be dead weight in every text leg, and the branching to keep both in one file is how the text path acquires a vision-shaped bug nobody notices.
        files = SMOKE_FILES + ("run_vision_t4.py",),
        args = (
            "--model",
            "unsloth/Qwen3.5-2B",
            "--max-steps",
            "10",
            # The point of the leg, measured on the recon probe: `fla` resolves to unsloth_zoo/_vendored/fla 0.5.1 and is importable only AFTER the model load, and attention resolves to `sdpa` while `flash_attn` is not importable at all. Deliberately NOT asserted: that causal_conv1d or mamba_ssm are installed (neither is on this path, since the wheel-first machinery in studio/backend/utils/ssm_runtime.py belongs to Studio's training worker and the notebook path never calls it), and not that FlashAttention-2 ran, since sm_75 cannot execute it. The rule instead fails if anything selects FA2 on a Turing card.
            "--kernel-provenance",
            # The image path itself, spawned by the parent after the cycles in a process of its own. Without it this leg trains TEXT and asserts kernels, which is a vision leg in name only.
            "--vision-run",
            # Q8_0 through the prebuilt llama.cpp binaries, then inference on the result; measured here for the first time on a vision checkpoint, whose merged export is the half the text path cannot exercise.
            "--export-gguf",
        ),
        reference = "",
    ),
    "frontier": Leg(
        vram_gb = 0.7,
        name = "frontier",
        summary = "the same SFT run on the newest transformers and trl on PyPI",
        # WHY THIS EXISTS given the canary already says "newest": the canary installs the newest set zoo's METADATA allows, and that ceiling is low. unsloth_zoo/pyproject.toml pins transformers <=5.5.0 and trl <=0.24.0, so on 2026-08-11 the canary resolved transformers 5.5.0 against a PyPI latest of 5.15.0 and trl 0.24.0 against 1.9.2, a whole major version, while peft and accelerate did move and made the leg look like it was working. So with only the canary this CI CANNOT detect a transformers 5.6+ or trl 1.x regression.
        # WITH dependencies, NOT --no-deps: `--no-deps transformers trl` plus a blanket `--upgrade tokenizers` reached transformers 5.15.0 and trl 1.9.2 (kernel unsloth-t4-ci-bd0c49e5) and then died on tokenizers 0.23.1 against a <=0.23.0 requirement and safetensors 0.7.0 against >=0.8.0. Resolving the deps fixes both, because pip enforces only the requirements of packages IN the resolution and unsloth_zoo is merely installed, so its <=5.5.0 is a warning rather than a ceiling. Dry run against an environment with zoo installed: "Would install datasets-5.0.1 huggingface_hub-1.27.0 transformers-5.15.0 trl-1.9.2". So this leg moves whatever transformers and trl now require, which is the honest scope of taking the new version.
        # This leg does NOT go red: transformers 5.15.0, trl 1.9.2, datasets 5.0.1, ten steps, canary emitted, two fresh processes agreeing bitwise. What it does not catch is that the loss trajectory is not the control's (control 10.3222 10.4956 ... against frontier 6.4367 6.6086 ...). Step 1 is computed before any update on identical weights, data and seed, so the loss FUNCTION differs in masking or normalisation; both converge, and this leg has no reference band.
        install = BASE_INSTALL + ((("--upgrade", "transformers", "trl")),),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        # Same reasoning as the canary, more so: this set is further still from the committed trace.
        reference = "",
    ),
    "default": Leg(
        vram_gb = 0.7,
        name = "Default",
        summary = "Qwen3-0.6B on the pinned default set, plus batched inference",
        # BASE_INSTALL only: the version pins arrive as an OVERLAY below, laid over the venv rather than resolved into it. Measured on kernel unsloth-probe-overlay-t4-r2-38ac4d, the overlay is three packages, 115.9 MB, in 10.0s, against 158-207s for a leg's own install groups.
        install = BASE_INSTALL,
        overlay = ("transformers==4.57.6", "trl~=0.22.0"),
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        # WHY THIS MODEL AND THESE VERSIONS: `control` pins transformers 5.5.0 / trl 0.24.0, exactly zoo's own ceiling, so on run 32703162400 `canary` resolved the SAME pair and two legs measured one library set. This leg holds the version pair a released `pip install unsloth` gives a user (PyPI unsloth 2026.8.19 declares transformers <=5.5.0,>=4.51.3 and trl <=0.24.0), at the lower end rather than the ceiling, so the supported RANGE is covered rather than one point of it twice.
        args = (
            "--model",
            "unsloth/Qwen3-0.6B",
            # 20, not the default 10, and measured: at 10 (unsloth-probe-defaultleg-r2-563e31) Qwen3-0.6B learns the canary but not to STOP after it, emitting `__UNSLOTH__!!!__` and failing the exact-match rule; at 20 (unsloth-probe-defaultleg-s20-002d25) both repeats emit `__UNSLOTH__!!!` exactly. The alternative, --no-require-canary, would have made the check vacuous.
            "--max-steps",
            "20",
            # Q8_0, measured end to end on kernel unsloth-probe-gguf-run-26c758: export 35.7s, a 609.8MB qwen3-0.6b.Q8_0.gguf in the SIBLING directory, and llama-bench reporting pp8 68.85 t/s. So this leg covers the export AND that the exported file runs, which are different claims.
            "--export-gguf",
        ),
        # No reference yet. The committed band (references/t4_qwen2.5-0.5b.json) is a Qwen2.5-0.5B trajectory and says nothing about Qwen3-0.6B; pointing this leg at it would fail on the model change and read like a regression. Capture on a real T4 in its own run, as references/README.md requires, and wire it in then.
        reference = "",
    ),
    "multi_gpu": Leg(
        # Small, and it has to be: this leg reserves its share on EVERY card, so a big appetite would block gptoss (12.78 of the 13.0 budget) for its whole life. 1.2 MEASURED on kernel unsloth-probe-multigpu-r2-a280e2, both cycles. It was 0.7 first, copied from the other Qwen legs, which under-declared the card the weights are on, since this leg holds a CUDA context on both cards.
        vram_gb = 1.2,
        name = "Multi_GPU",
        summary = "Qwen3-0.6B with BOTH T4s visible: unsloth's DEVICE_COUNT > 1 bindings",
        install = BASE_INSTALL,
        entry = "run_t4_smoke.py",
        files = SMOKE_FILES,
        # THE WHOLE POINT, and it is a branch rather than a capacity: unsloth/kernels/utils.py:170 binds torch_gpu_device to torch.cuda.device only when DEVICE_COUNT > 1, and build_kernel.py pins every other payload with CUDA_VISIBLE_DEVICES, so every unsloth kernel this CI has run took the shim. The same is true of the DEVICE_COUNT-sized CUDA_STREAMS / WEIGHT_BUFFERS / ABSMAX_BUFFERS arrays (kernels/utils.py:211-250), the per-device rotary caches (models/llama.py:1838, gemma.py:280) and the temp_mlp device tuples (llama.py:1300). It is FREE because the divergence is triggered by VISIBILITY: Qwen3-0.6B fits on one card, so the leg co-tenants at 0.7 GB per card.
        all_cards = True,
        args = (
            "--model",
            "unsloth/Qwen3-0.6B",
            # 20 for the same measured reason as the Default leg: at 10 this model learns the canary but not to stop after it, and the exact-match rule then fails on a training length rather than on a regression (unsloth-probe-defaultleg-r2-563e31 vs -s20-002d25).
            "--max-steps",
            "20",
            "--require-multi-gpu",
            "--expected-cards",
            "2",
            # Both cards VISIBLE, weights on ONE. Measured on unsloth-probe-multigpu-r1-18beab: with two T4s visible accelerate sharded this model across both, unsloth announced `Num GPUs used = 2`, and step 0 died at unsloth/models/llama.py:972 with `index is on cuda:0, different from other tensors on cuda:1`. Whether that is a DEFECT is unsettled (#2467 closed with "unsloth runs on only 1 gpu", #2882 closed as fixed), so it is not something a per-PR check should go red on, and the bindings this leg covers do not depend on it.
            "--single-device",
            # NO --export-gguf, and this is a measurement rather than a saving: the bundle `install_llama_cpp` fetches for the notebook legs is the CPU one (on unsloth-probe-full-concurrent-417238 llama-bench reported `backend CPU`), and a CPU llama.cpp cannot split across cards, so a two-card tensor-split assertion here could not fail and would not mean anything. The GGUF path is already covered by four other legs.
        ),
        # Same situation as `canary` and `Default`: two library versions do not produce one fp16 trajectory, and this leg additionally changes how many cards the kernels see, so a band would go red on ordinary cross-version drift.
        reference = "",
    ),
    "gptoss": Leg(
        vram_gb = 12.78,
        name = "gptoss",
        summary = "gpt-oss-20b LoRA: torch.compile and the float32 path",
        # The base install and nothing else, specifically WITHOUT the `triton_kernels` git dependency the gpt-oss notebook installs. Measured: two probe kernels on 2026-08-11, one with triton_kernels and torchao and one with neither, produced the SAME three losses to the last bit, the same 12.78 GB peak and the same compile counters (32 graphs, 779 calls, 2 breaks). They agree because `load_in_4bit=True` never reaches MXFP4: FLOAT_TO_INT_MAPPER redirects `unsloth/gpt-oss-20b` to the NF4 `unsloth/gpt-oss-20b-unsloth-bnb-4bit`, and MXFP4 has no backward pass to reach.
        install = BASE_INSTALL,
        entry = "run_gptoss_t4.py",
        # gguf_export.py travels too, now that this leg exports. It is imported lazily inside the export branch, so a leg that never exports would not notice it missing, which is why it is declared here rather than left to chance.
        files = COMMON_FILES + ("run_gptoss_t4.py", "gguf_export.py"),
        args = (
            "--max-steps",
            "3",
            "--max-seq-length",
            "1024",
            # NO EXPORT here either, and for this leg it was never Q8_0: gpt-oss overrides any request ("GPT-OSS does not support GGUF quantization ... Overriding to MXFP4 format"), so this produced a 13153.7MB MXFP4 file in 348.1s across 27.6GB of transient disk (unsloth-probe-gptoss-final-e9bd76), the most expensive export in the suite and the least representative of the claim under test, which `default` makes in 40.6s and `vision_fla_compile` in 99.3s on files small enough to load afterwards.
        ),
    ),
    # NOT WIRED, for a smaller reason than it used to be. See UNWIRED below: the install that killed three probe sessions is re-solved, and what remains is a runtime question needing one session on a real T4.
    "grpo": Leg(
        vram_gb = 13.8,
        name = "grpo",
        summary = "Qwen3-4B GRPO through a vLLM engine on the same card",
        # vLLM FIRST and alone: it pins torch, and resolving it after unsloth walks torch underneath an already installed stack. THE VERSION MATCHES THE IMAGE, IT IS NOT MERELY OLD: Kaggle ships torch 2.10.0+cu128, and only vLLM 0.17.0 .. 0.19.1 pin torch==2.10.0 (0.11.2-0.16.0 want 2.9.x, 0.20.0-0.26.0 want 2.11.0, 0.27.0+ want 2.13.0). Every other choice REPLACES the image's torch, which is what all three probe sessions died of, since the image's NVIDIA runtime packages belong to 2.10 and pip treats them as satisfying the new torch's pins. 0.19.1 is the newest release needing no replacement, so the leg keeps `system_site_packages`. No xformers: its vLLM attention backend was deleted in 0.12.0. sm_75 has no FlashAttention and no FlashInfer, and the ladder in vllm/platforms/cuda.py falls through to TRITON_ATTN, pinned below rather than left to a probe order that moves between releases; sm_75 is still in CUDA_SUPPORTED_ARCHS at v0.19.1 and fp16 is supported below capability 8.0.
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
        # ALL FIVE ARE LOAD-BEARING ON A 14.56GB CARD, and are the values that passed rather than the ones that look reasonable. Two probes with the notebook's own settings (seq 2048, 4 generations, rank 32, utilization 0.9) died in the BACKWARD at unsloth_zoo/gradient_checkpointing.py:1013, peaking at 15.97GB in 16-bit and 19.25GB in 4-bit. 4-bit is not the lever it looks like: quantizing weights does nothing for activations while utilization 0.9 still hands vLLM ~13GB up front, and UNSLOTH_VLLM_STANDBY returns the weights during training but not the KV cache reservation. Measured on kernel unsloth-t4-ci-53efcc4e: peak 13.60GB allocated of 14.56GB, three steps in 192s.
        args = (
            "--max-steps",
            "3",
            "--load-in-4bit",
            "--gpu-memory-utilization",
            # 0.95, measured on BOTH platforms rather than assumed: Colab T4 torch 2.11.0 peak reserved 11.76 GB and Kaggle T4 torch 2.10.0 peak reserved 11.30 GB, sleep/wake 3/3, both passing on the FIRST rung of a 0.95/0.8/0.6/0.5 ladder, ~3 GB under the card. The earlier argument against it came from Qwen3-4B probes that OOMed at 0.9; at 0.6B the activation budget is a different problem.
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
            # See the install comment. Named rather than probed so a release reordering the ladder turns this leg red instead of silently selecting something else.
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            # Kaggle cannot link what flashinfer JIT-compiles: on kernel unsloth-t4-ci-e2d9ce9b, flashinfer 0.6.6 compiled all three sampling .cu files cleanly for sm_75 and the link died on `/usr/bin/ld: cannot find -lcuda`. The stub is NOT absent and this IS fixable from here: kernel unsloth-probe-grpo-shipped-d15695 reports libcuda_shim applied against /usr/local/cuda/compat/libcuda.so, which run_grpo_t4.py:597 symlinks onto LIBRARY_PATH and has done since #8440. The two `-L` paths flashinfer passes do not include /usr/local/cuda/compat, which is why the raw link fails and why the hand-written ladder probes, which had no shim, hit it. The setting stays regardless: not JITting is cheaper than JITting in a session billed by wall clock on 4 vCPUs.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            # The two settings above are NOT sufficient on their own: a probe that set only them failed identically to one that set neither (unsloth-probe-grpo-ladder-kaggle-7e7697 and -r2-ddcb18). unsloth_zoo's patch_vllm OVERWRITES both during model load (vllm_utils.py:2494 sets VLLM_ATTENTION_BACKEND="FLASHINFER", :2502 sets VLLM_USE_FLASHINFER_SAMPLER="1"), reached via `elif Version(vllm_version) >= Version("0.11.0")`, and the guard at 2452-2460 tests only that nvcc and ninja EXIST, which on Kaggle they do. This knob is read at 2449, before any of the assignments, and skips the entire flashinfer block.
            "UNSLOTH_VLLM_NO_FLASHINFER": "1",
        },
        # See the field comment. The env var above is kept as well: it is what stops unsloth reaching for FlashInfer at all, and keeping both means a future image that DOES ship the stub still takes the deliberate path rather than whichever one happens to link.
        uninstall = ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"),
        # Now true, which is the point of the version choice above: this leg no longer replaces torch, so it shares the image's view instead of resolving a whole CUDA stack from scratch. Probe 3 spent about an hour of quota doing that and never got past venv creation.
        system_site_packages = True,
    ),
}


# How many legs one kernel packs. This was 2, the card count, because legs were started all at once and one per card. They now queue (build_kernel.py runs one worker per card, and a card takes its next leg only when the previous one has exited), so the card count no longer caps this; wall clock does, since every leg past the second adds its runtime to one card's column against a 12 hour session kill. MEASURED at five legs on unsloth-probe-all5-kernel-e28818: makespan 792.2s. Note that legs do NOT simply queue one per card: the VRAM admission lets light legs CO-TENANT (frontier and canary hold gpu0 together at 0.7GB each against a 13.0GB budget), while gptoss at 12.78GB is admitted only once gpu0 is empty, which is why it starts at 492.7s. So ADMISSION, not the card count and not a queue, decides, and a fifth HEAVY leg would be a different question. Raised 5 -> 6 for multi_gpu, the cheapest addition yet: it co-tenants on BOTH cards at 1.2GB rather than taking one. Whether the makespan holds is a MEASUREMENT, not this comment.
MAX_LEGS_PER_KERNEL = 6

# Legs defined here and deliberately NOT run, with the reason. A leg is unwired rather than deleted when the payload is right and the environment is not, so the next person to try owes nothing but a working install. Every entry must say what was measured.
UNWIRED: dict[str, str] = {
    # A leg belongs here only while a specific unanswered question about it would be answered by a session. "Not tried yet" is not that.
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

# Which legs travel in which kernel. ONE kernel holding every leg, which its 2xT4 session works through two at a time. This was two kernels, and the reason it is one now is NOT quota (a session bills wall clock once, so two kernels of two legs cost 662s against 646s for one of four) but the ACCOUNT: Kaggle allows two concurrent GPU sessions, two kernels take both, and kaggle-t4-studio-gpu-ci.yml runs on the same account, so the notebook leg locked Unsloth out entirely while it ran (run 32607617804 queued ~40 minutes behind notebook run 32607621452). The two workflows now hold separate GitHub concurrency groups; splitting the group without packing the kernel, or packing without splitting, each make things worse alone.
# gptoss sits THIRD, and that position is load-bearing. The driver prefetches models on a lane that takes no card (kaggle_prefetch.py), and a prefetch only pays for work it finishes BEFORE the leg that wants the model starts, so gptoss, the only leg with a 12 GB download, must not start at t=0. Third is first pick of the second wave, at ~190-220s. Simulated over every permutation against the durations measured on run 32611343797 and a range of the unknown download time D: this order is never worse than the old one at any D tested, prefetching WITHOUT reordering is a REGRESSION (gptoss at t=0 has no window in front of it), and gptoss LAST is worse than doing nothing for any D under ~120s, since ending on the longest leg leaves the other card idle. Chosen on WORST CASE rather than mean, because D is not measured yet. If the prefetch fails outright the schedule degrades to ~568s against the old 563.1s.
# control and canary stay in the same kernel, which is what their comparison needs: same image, same driver, same hour. They no longer run on the two cards SIMULTANEOUSLY, which is fine, since they were never compared against each other, only each against its own committed reference. What would break them is landing in different SESSIONS, which packing everything into one kernel makes impossible.
# `grpo` returns here once the illegal memory access in UNWIRED is understood. Running it ALONE (unsloth-t4-ci-c98f14be) reproduced the paired failure exactly, same stack, same 13.8GB peak, same engine_built false, so pairing with gpt-oss was never the variable.
# `frontier` is REPLACED by `vision_fla_compile` rather than joined by it: everything frontier proved is a subset of the vision leg, which also asserts the vendored FLA kernels, the attention choice on Turing, a real vision training run, the merged vision export, a Q8_0 GGUF with its mmproj sidecar and inference on the exported file. Verified green as a whole leg before wiring (unsloth-probe-visleg-final-2b896b, 2.84GB peak reserved).
KERNELS: tuple[tuple[str, ...], ...] = (
    ("canary", "control", "gptoss", "vision_fla_compile", "default"),
)


# What the prefetch lane warms, in order, into the Kaggle image's default HF cache. See kaggle_prefetch.py for the mechanism and KERNELS above for why gptoss is third. The SMALL model first, which inverts what looks obvious: gpt-oss is bigger but is wanted by ONE leg whose own setup does not finish until ~160s, while Qwen2.5-0.5B gates THREE legs and the first starts at t=0, so it is the only fetch with no lead time in front of it.
# These must match what the legs LOAD, not what they ASK FOR. The gptoss payload declares DEFAULT_MODEL = "unsloth/gpt-oss-20b", but that is an MXFP4 checkpoint and sm_75 cannot read MXFP4, so unsloth redirects to `-unsloth-bnb-4bit` at load time; the first version of this list prefetched the declared name and pulled 55.1 GB of a checkpoint no leg opens while the leg downloaded the real one anyway. Nothing at runtime notices: a prefetch of the wrong repo downloads happily, warms a cache nobody reads, and reports success.
PREFETCH_REPOS: tuple[str, ...] = (
    # CRITICAL PATH FIRST, a reversal of the original gpt-oss-first order, which was chosen for margin while D (how long 12.5 GB takes) was unknown. D is now measured at 61.7s on kernel unsloth-probe-prefetch-verify-9568-7a0bdd, so every repo here lands inside ~90s whatever the order, and the leg that needs one EARLIEST should not be the one waiting. vision_fla_compile is that leg: it starts at t~21 and it sets the makespan.
    "unsloth/Qwen3.5-2B",
    # Default's model and the 4bit sibling `load_in_4bit=True` actually resolves to. Both, because FLOAT_TO_INT_MAPPER redirects at load time and warming only the name in the args warms a cache the leg never reads.
    "unsloth/Qwen3-0.6B",
    "unsloth/qwen3-0.6b-unsloth-bnb-4bit",
    "unsloth/Qwen2.5-0.5B-Instruct",
    # Last now rather than first: gptoss is admitted only once a card empties, around t~500 on the measured schedule, so it has the most slack of anything here.
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
)

# Declared name -> what actually gets loaded on an sm_75 card. Kept beside the list it corrects so the two cannot drift apart silently.
LOAD_REDIRECTS: dict[str, str] = {
    "unsloth/gpt-oss-20b": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    # CASE MATTERS HERE, and it is not a typo: two runs report `resolved_checkpoint: unsloth/qwen3-0.6b-unsloth-bnb-4bit` in lower case while gpt-oss above resolves with its capitals intact. The HF cache keys on the literal string, so prefetching the pretty spelling warms a directory the leg never reads and the download happens twice. Copied from the report rather than typed.
    "unsloth/Qwen3-0.6B": "unsloth/qwen3-0.6b-unsloth-bnb-4bit",
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
