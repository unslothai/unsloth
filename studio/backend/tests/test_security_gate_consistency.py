# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic consistency guards for the model-load security gate.

The gate is threaded through many PARALLEL sites (validate / load / status, the
inference / training / export workers, the preflight route). Every past regression in
this area was the same shape: a fix applied to one site while a sibling site silently
kept the old behavior (a dropped token, a trust_remote_code requirement read from the
wrong source). A stochastic review eventually finds one such sibling per pass, which is
why the reviews felt never-ending.

These guards enumerate the sites MECHANICALLY (AST + source), so a newly added site
that drops the token or mis-reports the requirement fails here immediately instead of
waiting for a review round to notice it. They are intentionally cheap and network-free.
"""

import ast
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent

# Capability probes read repo config from the Hub to classify a model, so a caller must
# pass the hf_token or a gated/private model 404s and is misclassified. We scan every
# CALLER under routes/ and core/; the probe DEFINITIONS live in utils/ (excluded).
_PROBE_FUNCS = {"is_vision_model", "is_embedding_model", "detect_audio_type"}
_PROBE_CALLER_ROOTS = ("routes", "core")


def _iter_caller_files():
    for root in _PROBE_CALLER_ROOTS:
        yield from (_BACKEND / root).rglob("*.py")


def _passes_token(call: ast.Call) -> bool:
    """True if the call passes an hf_token, by keyword or as the second positional arg
    (the hf_token slot for all three probes)."""
    if any(kw.arg in ("hf_token", "token") for kw in call.keywords if kw.arg is not None):
        return True
    return len(call.args) >= 2


def _call_name(call: ast.Call):
    fn = call.func
    return fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)


def test_capability_probes_thread_the_hf_token():
    """Every is_vision_model / is_embedding_model / detect_audio_type caller passes the
    token. A token-less probe 404s on a gated/private model and silently classifies it as
    a plain text model -- the /check-vision regression. A new caller that drops the token
    fails here, not in a later review round."""
    offenders = []
    for path in _iter_caller_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) in _PROBE_FUNCS:
                if not _passes_token(node):
                    rel = path.relative_to(_BACKEND)
                    offenders.append(f"{rel}:{node.lineno} {_call_name(node)}() drops the hf_token")
    assert not offenders, (
        "A capability probe must pass the hf_token so gated/private models classify "
        "correctly:\n  " + "\n  ".join(offenders)
    )


def test_gguf_trust_remote_code_reported_inert_not_from_yaml():
    """A GGUF loads via llama.cpp and never executes auto_map Python, so every response
    path reports requires_trust_remote_code via the resolver (non-GGUF) or False (GGUF) --
    never the raw YAML default ``bool(inference_config.get("trust_remote_code", ...))``,
    the round-6 regression that re-flagged an inert GGUF as needing trust_remote_code."""
    src = (_BACKEND / "routes" / "inference.py").read_text()
    assert "requires_trust_remote_code = bool(" not in src, (
        "Report requires_trust_remote_code via _resolve_loaded_trust_remote_code "
        "(non-GGUF) or set it False (GGUF); never bool(inference_config.get(...))."
    )


def test_capability_detection_caches_are_token_aware():
    """Every per-model capability cache (vision / audio / embedding) is keyed by a
    tuple that includes the token, so an unauthenticated miss on a gated/private repo
    cannot poison a later authenticated lookup with a stale result. A cache re-declared
    as Dict[str, ...] (keyed by model name only) fails here -- the round-final audio
    cache regression."""
    src = (_BACKEND / "utils" / "models" / "model_config.py").read_text()
    offenders = []
    for line in src.splitlines():
        stripped = line.strip()
        if "_detection_cache:" in stripped and stripped.endswith("= {}"):
            if "Dict[Tuple" not in stripped and "Dict[tuple" not in stripped:
                offenders.append(stripped)
    assert not offenders, (
        "A capability cache must be keyed by (model, token_fingerprint), not the bare "
        "model name:\n  " + "\n  ".join(offenders)
    )


def test_malware_and_consent_gates_cover_the_lora_base():
    """Every worker that runs the malware or consent gate resolves the LoRA base too, so
    a poisoned base or a base that ships its own auto_map code is never skipped. Each
    gate's target list is built from the model AND its resolved base; a worker that gates
    only the selected model (no base resolution) fails here."""
    gated_workers = [
        "core/inference/worker.py",
        "core/export/worker.py",
        "core/training/worker.py",
    ]
    offenders = []
    for rel in gated_workers:
        src = (_BACKEND / rel).read_text()
        runs_gate = "evaluate_file_security(" in src or "evaluate_remote_code_consent" in src
        resolves_base = "get_base_model_from_lora_identifier(" in src or "base_model" in src
        if runs_gate and not resolves_base:
            offenders.append(f"{rel} runs a load gate but never resolves the LoRA base")
    assert not offenders, "\n".join(offenders)
