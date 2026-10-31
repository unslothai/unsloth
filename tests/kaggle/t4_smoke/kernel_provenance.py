# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Read after the model load: the other kernels resolve late, so an early read reports them absent."""

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
    """Reads attention from the config, since a module walk on the recon probe returned an empty set."""
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
    """Attention is asserted as sdpa, not flash_attention_2, since FA2 cannot execute on sm_75."""
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
