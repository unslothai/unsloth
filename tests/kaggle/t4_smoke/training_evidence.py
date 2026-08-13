# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Did the optimizer actually change the adapter? Shared by the payloads.

Every leg here can produce a completely healthy-looking report from a run
that applied no update at all. The loss is finite because a forward pass
computes one, `torch.compile` engages because compilation happens on the
forward, and generation returns text because the BASE model generates text
perfectly well without any adapter. So each payload asserts that at least one
logged ``grad_norm`` was finite and non-zero.

That assertion has a hole, and this module exists to close it: it is only
decidable where ``grad_norm`` was logged. A Trainer or integration change
that stops emitting the field leaves an EMPTY list, and "no logged norm was
usable" and "no norm was logged" are opposite situations that the obvious
`if norms and not applied` spelling collapses into a pass.

Inferring "nothing was applied" from that silence would be a failure the
check invented rather than found, so the answer is not to fail on silence but
to stop depending on the trainer's telemetry. The adapter itself is the
ground truth: LoRA's B matrices are initialised to exactly zero and stay
exactly zero until an optimizer step lands on them, so a fingerprint taken
before training and after answers the question directly, whatever the trainer
chose to log.

The tiny SFT payload (``run_t4_smoke.py``) does not use this. It reads its
saved adapter back off disk and fails on an all-zero one
(``verify_saved_adapter``), which is the same question answered by a
different, already-committed instrument. gptoss and grpo save no adapter,
which is why the silence there was covered by nothing.

Nothing in here raises. A diagnostic that can kill the payload it is
diagnosing is worse than the gap it closes -- the leg would then report
nothing at all, which is the outcome every report in this directory exists to
prevent.
"""

from __future__ import annotations

# Substring that marks a LoRA parameter, and the narrower one for the B
# matrices. peft names them `...lora_A.default.weight` / `lora_B...`; the
# match is lowercased so a future capitalisation does not silently empty the
# set (a fingerprint over zero tensors reports itself as unusable rather than
# as "nothing changed", see below).
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

    Cheap by construction: a rank-8 adapter is a few hundred small matrices,
    so this is microseconds beside the training step it brackets, and it
    allocates nothing that outlives the call.

    Returns ``{"ok": False, "error": ...}`` rather than raising, on anything
    at all. ``ok`` false means the question could not be answered here, not
    that the answer was no.

    A non-finite sum is the one refusal that is NOT "could not answer". A
    LoRA weight that has gone NaN or infinite is a broken run, and left as a
    number it is the strongest possible pass: ``NaN != finite`` and
    ``inf != finite`` both read as "the adapter changed", so the corrupted
    run reports an applied update on exactly the no-telemetry path this
    module was written to decide. It is flagged with ``non_finite`` so
    ``update_verdict`` can call it what it is instead of comparing it.
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

    Two sums are compared, not one, and the second is what makes an exact
    float comparison safe to turn red on. ``abs_sum`` moving is the general
    signal; ``b_abs_sum`` is the specific one, and it starts at exactly 0.0
    because peft zero-initialises every B matrix. For BOTH to be bitwise
    unchanged after a real update, an optimizer would have to land a set of
    deltas that cancels to the last bit in two different summations at once.

    A tensor count that differs between the two readings is reported as
    unusable rather than as a change: whatever that is, it is not evidence
    about the optimizer.
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
    # Checked again on this side, not only where the sums were taken: an
    # exact `!=` on a NaN is the most confident "it changed" this file can
    # produce, so the comparison refuses to run on one whatever produced it.
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

    The adapter reading wins where it exists, because it is the thing the
    grad norms are a proxy FOR: gradients that flowed into weights nobody
    updated is still a run that trained nothing.

    ``unverifiable`` is the state this module was written for -- no usable
    grad_norm was logged AND the adapter could not be read -- and it is a
    failure at the call sites. Not because nothing was applied, which is
    unknown, but because the leg's whole claim is that it exercised the
    training path and it can no longer show that it did.

    ``non_finite`` is decided FIRST and beats a healthy grad_norm, because it
    is the one adapter reading that is an answer rather than a refusal. A
    finite norm logged at step 1 says nothing about weights that went NaN at
    step 3, and the trained adapter is the artifact the leg exists to
    produce.

    Every call site treats anything other than ``applied`` as a failure, so a
    verdict added here cannot be silently dropped by one that has not been
    taught about it yet.
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
