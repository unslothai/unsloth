# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fingerprints taken before and after training answer whether the optimizer ran, unlike grad_norm."""

from __future__ import annotations

# Matched lowercased so a capitalisation change cannot silently empty the set.
LORA_MARKER = "lora_"
LORA_B_MARKER = "lora_b"


def _is_finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def adapter_fingerprint(model) -> dict:
    """Non-finite sums are flagged, because NaN compares as changed and would read as a strong pass."""
    try:
        total = 0.0
        b_total = 0.0
        tensors = 0
        non_finite: list[str] = []
        for name, param in model.named_parameters():
            lowered = name.lower()
            if LORA_MARKER not in lowered:
                continue
            tensors += 1
            value = float(param.detach().float().abs().sum().item())
            if not _is_finite(value):
                non_finite.append(name)
            total += value
            if LORA_B_MARKER in lowered:
                b_total += value
        if not tensors:
            return {"ok": False, "error": "no parameter name carries a LoRA marker"}
        if non_finite:
            return {
                "ok": False,
                "non_finite": True,
                "tensors": tensors,
                "error": (
                    f"{len(non_finite)} of {tensors} LoRA tensors hold non-finite "
                    f"weights: {sorted(non_finite)[:10]}"
                ),
            }
        return {"ok": True, "tensors": tensors, "abs_sum": total, "b_abs_sum": b_total}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def adapter_update(before, after) -> dict:
    """Two sums make exact comparison safe: B starts at 0.0 and both sums would need to cancel bitwise."""
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    if not (before.get("ok") and after.get("ok")):
        return {
            "ok": False,
            "non_finite": bool(before.get("non_finite") or after.get("non_finite")),
            "error": before.get("error")
            or after.get("error")
            or "the adapter was not fingerprinted",
        }
    # Rechecked here: `!=` on NaN always reports a change.
    unusable = [
        f"{side}.{key}={reading[key]}"
        for side, reading in (("before", before), ("after", after))
        for key in ("abs_sum", "b_abs_sum")
        if not _is_finite(reading.get(key))
    ]
    if unusable:
        return {
            "ok": False,
            "non_finite": True,
            "error": f"the adapter fingerprints are not finite ({', '.join(unusable)})",
        }
    if before.get("tensors") != after.get("tensors"):
        return {
            "ok": False,
            "error": (
                f"the adapter had {before.get('tensors')} LoRA tensors before training "
                f"and {after.get('tensors')} after, so the two readings are not "
                f"comparable"
            ),
        }
    changed = (after["abs_sum"] != before["abs_sum"]) or (after["b_abs_sum"] != before["b_abs_sum"])
    return {
        "ok": True,
        "changed": bool(changed),
        "tensors": after["tensors"],
        "abs_sum_before": before["abs_sum"],
        "abs_sum_after": after["abs_sum"],
        "b_abs_sum_before": before["b_abs_sum"],
        "b_abs_sum_after": after["b_abs_sum"],
    }


def update_verdict(metrics, adapter = None) -> dict:
    """The adapter reading wins over grad_norm; non_finite is decided first and beats a healthy norm."""
    rows = metrics or []
    norms = [row.get("grad_norm") for row in rows if row.get("grad_norm") is not None]
    usable = [g for g in norms if _is_finite(g) and float(g) != 0.0]
    weights = adapter if isinstance(adapter, dict) else {}
    moved = weights.get("changed") if weights.get("ok") else None

    if weights.get("non_finite"):
        return {
            "verdict": "non_finite",
            "detail": str(weights.get("error") or "the adapter holds non-finite weights"),
            "grad_norms": norms,
        }
    if moved is False:
        return {
            "verdict": "not_applied",
            "detail": (
                f"the {weights.get('tensors')} LoRA tensors are bitwise identical to "
                f"the ones training started with (|w| {weights.get('abs_sum_before')} "
                f"-> {weights.get('abs_sum_after')}, of which the zero-initialised B "
                f"matrices {weights.get('b_abs_sum_before')} -> "
                f"{weights.get('b_abs_sum_after')})"
            ),
            "grad_norms": norms,
        }
    if norms and not usable:
        return {
            "verdict": "not_applied",
            "detail": f"every logged grad_norm is zero or non-finite ({norms})",
            "grad_norms": norms,
        }
    if usable or moved:
        return {"verdict": "applied", "detail": "", "grad_norms": norms}
    return {
        "verdict": "unverifiable",
        "detail": (
            "no grad_norm was logged on any step, and the adapter weights could not "
            f"be compared either ({weights.get('error') or 'not fingerprinted'})"
        ),
        "grad_norms": norms,
    }
