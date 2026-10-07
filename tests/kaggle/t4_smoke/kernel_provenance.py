# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Which fast kernels a model actually got, and where each one came from.

**The failure this is built to catch is not "kernel missing".** It is "kernel
built from source", which is silent, correct, slow, and reports the same
`__version__` as the wheel. Only the module's `__file__` and the install log
tell them apart, so a version check here would pass while proving nothing.

**Read AFTER the model load, not before.** `fla` now comes from unsloth_zoo's
vendored copy, injected at `import unsloth`, so it is importable from that point
on and the answer is the vendored path rather than a pip install; the other
kernels still resolve late. Measured on kernel
`unsloth-probe-vision-recon-c76ea3`: a probe that read provenance only up front
reported them absent, which is the opposite of the truth.

Nothing here raises. A payload that dies collecting a diagnostic reports nothing
at all, which is the one outcome worse than a missing field.
"""

from __future__ import annotations

# Not all must be present: two are absent on this path by design. See vision_kernel_failures.
_KERNELS = ("fla", "causal_conv1d", "mamba_ssm", "flash_attn", "triton", "xformers")


def probe_kernels() -> dict:
    """Import each kernel and record where it resolved from."""
    out: dict = {}
    for name in _KERNELS:
        entry: dict = {"importable": False}
        try:
            module = __import__(name)
            entry["importable"] = True
            entry["file"] = getattr(module, "__file__", None)
            entry["version"] = getattr(module, "__version__", None)
            entry["vendored"] = "_vendored" in (entry["file"] or "")
        except BaseException as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
        out[name] = entry

    # A package can be installed but not importable (wrong CUDA ABI).
    try:
        from importlib import metadata

        dists = {}
        for dist in metadata.distributions():
            name = (dist.metadata["Name"] or "").lower()
            if name in (
                "causal-conv1d",
                "mamba-ssm",
                "flash-attn",
                "fla-core",
                "flash-linear-attention",
                "xformers",
            ):
                dists[name] = dist.version
        out["_distributions"] = dists
    except Exception as exc:  # noqa: BLE001
        out["_distributions"] = {"error": str(exc)[:200]}
    return out


def attention_choice(model) -> dict:
    """What attention resolved to, read off the config.

    The config records the choice; a module walk was tried on the recon probe
    and returned an empty set, so this reports the one source that answers.
    """
    record: dict = {}
    try:
        config = getattr(model, "config", None)
        record["config"] = getattr(config, "_attn_implementation", None)
        text = getattr(config, "text_config", None)
        if text is not None:
            record["text_config"] = getattr(text, "_attn_implementation", None)
    except BaseException as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return record


def _is_turing(capability) -> bool:
    """True for compute capability 7.x, whichever way it was spelled."""
    text = str(capability or "").strip().lower().replace("sm_", "").replace("sm", "")
    if not text:
        return False
    head = text.split(".")[0]
    if "." in text:
        return head == "7"
    return len(head) >= 2 and head[0] == "7"


def vision_kernel_failures(
    kernels: dict | None, attention: dict | None, *, capability: str
) -> list:
    """The pass rule, as a pure function so it is checkable without a GPU.

    Three claims, each chosen so it is neither false nor vacuous on a T4:

    1. **FLA is present and VENDORED.** Measured: it resolves to
       `unsloth_zoo/_vendored/fla`, version 0.5.1, after the load. Asserting
       merely "importable" would pass on a pip-installed copy that is not what
       ships, which is a different thing being tested.
    2. **Attention is a valid Turing choice, and it is NOT flash_attention_2.**
       FA2 supports Ampere, Ada and Hopper. On sm_75 it cannot execute, so
       asserting it ran would be false and asserting it was "selected" would be
       vacuous. Asserting `sdpa` (or another real Turing path) is the claim
       that can be both true and informative -- and it catches the regression
       that matters, which is unsloth choosing a backend the card cannot run.
    3. **Nothing was built from source.** A source build is silent and costs
       many minutes on 4 vCPUs.

    `causal_conv1d` and `mamba_ssm` are deliberately NOT asserted present.
    Measured on the recon probe: neither is installed on the notebook path,
    before or after the load. The wheel-first machinery in
    `studio/backend/utils/ssm_runtime.py` belongs to Studio's training worker
    and this path never calls it. Asserting them would be red on correct
    behaviour; they are REPORTED so a change shows up in the diff.
    """
    if not kernels:
        return ["no kernel provenance was collected at all"]

    failures = []

    fla = kernels.get("fla") or {}
    if not fla.get("importable"):
        failures.append(
            f"fla did not import after the model load, so the vendored fast "
            f"kernels are not reachable: {fla.get('error')}"
        )
    elif not fla.get("vendored"):
        failures.append(
            f"fla imported from {fla.get('file')!r}, which is not the vendored "
            f"copy under unsloth_zoo/_vendored. This leg is about the vendored "
            f"kernels; a pip-installed fla is a different thing"
        )

    # Normalised: "sm_75" and "7.5" are both live spellings.
    turing = _is_turing(capability)
    if turing:
        chosen = (attention or {}).get("config")
        if chosen in (None, ""):
            failures.append("no attention implementation was recorded")
        elif "flash_attention_2" in str(chosen):
            failures.append(
                f"attention resolved to {chosen!r} on capability {capability}. "
                f"FlashAttention-2 supports Ampere, Ada and Hopper; a Turing "
                f"card cannot run it, so this would fail at the first forward"
            )

    return failures
