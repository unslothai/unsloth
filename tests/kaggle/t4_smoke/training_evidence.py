# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Did the optimizer actually change the adapter? Shared by the payloads.

A run that applied no update at all still reports healthily: the loss is finite
because a forward pass computes one, `torch.compile` engages on the forward,
and the BASE model generates text without any adapter. So each payload asserts
some logged ``grad_norm`` was finite and non-zero -- which is only decidable
where ``grad_norm`` was logged. A Trainer change that stops emitting the field
leaves an EMPTY list, and the obvious `if norms and not applied` spelling
collapses "no usable norm" and "no norm logged" into a pass.

Failing on that silence would be a failure invented rather than found, so this
stops depending on trainer telemetry: LoRA's B matrices are exactly zero until
an optimizer step lands on them, so fingerprints taken before and after
training answer the question directly.

The tiny SFT payload (``run_t4_smoke.py``) instead reads its saved adapter back
off disk and fails on an all-zero one (``verify_saved_adapter``); gptoss and
grpo save no adapter, which is why the silence there was covered by nothing.

Nothing here raises: a diagnostic that kills the payload it diagnoses leaves
the leg reporting nothing at all.
"""

from __future__ import annotations

# peft names them `...lora_A.default.weight` / `lora_B...`;
# Substrings marking a LoRA parameter and, narrower, the B matrices.
LORA_MARKER = "lora_"
LORA_B_MARKER = "lora_b"


def _is_finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def adapter_fingerprint(model) -> dict:
    """Sum ``|w|`` over every LoRA parameter of ``model``.

    Cheap: a rank-8 adapter is a few hundred small matrices, microseconds beside
    the training step it brackets. Returns ``{"ok": False, "error": ...}``
    rather than raising; ``ok`` false means the question could not be answered,
    not that the answer was no.

    A non-finite sum is the one refusal that IS an answer. NaN or infinite LoRA
    weights are a broken run, and left as a number they are the strongest
    possible pass (``NaN != finite`` reads as "the adapter changed"), so they
    are flagged ``non_finite`` for ``update_verdict`` to name rather than
    compare.
    """
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
    """Compare two fingerprints. ``changed`` is the whole answer.

    Two sums, not one, is what makes an exact float comparison safe to turn red
    on. ``abs_sum`` moving is the general signal; ``b_abs_sum`` starts at
    exactly 0.0 because peft zero-initialises every B matrix. For BOTH to be
    bitwise unchanged after a real update, the optimizer would have to land
    deltas cancelling to the last bit in two different summations at once.

    A differing tensor count between readings is unusable rather than a change:
    whatever it is, it is not evidence about the optimizer.
    """
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
    # Rechecked here, not only where the sums were taken:
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
    """Was an optimizer update applied? ``applied`` / ``not_applied`` /
    ``non_finite`` / ``unverifiable``.

    The adapter reading wins where it exists, being the thing grad norms are a
    proxy FOR: gradients flowing into weights nobody updated is still a run that
    trained nothing.

    ``unverifiable`` -- no usable grad_norm logged AND no adapter reading -- is
    a failure at the call sites: not because nothing was applied, which is
    unknown, but because the leg can no longer show it exercised the training
    path.

    ``non_finite`` is decided FIRST and beats a healthy grad_norm: a finite norm
    at step 1 says nothing about weights that went NaN at step 3.

    Every call site treats anything but ``applied`` as a failure, so a verdict
    added here cannot be silently dropped.
    """
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
