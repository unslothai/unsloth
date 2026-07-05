# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic consistency guards for the model-load security gate.

The gate spans many parallel sites (validate/load/status, the inference/training/export
workers, the preflight route); past regressions were a fix at one site with a sibling
left behind. These guards enumerate the sites mechanically (AST + source) so a new site
that drops the token or mis-reports the requirement fails here, not in a later review.
"""

import ast
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent

# Probes read Hub config to classify a model; a token-less call 404s on a gated repo.
# Scan callers under routes/ and core/ (probe definitions live in utils/).
_PROBE_FUNCS = {"is_vision_model", "is_embedding_model", "detect_audio_type"}
_PROBE_CALLER_ROOTS = ("routes", "core")


def _iter_caller_files():
    for root in _PROBE_CALLER_ROOTS:
        yield from (_BACKEND / root).rglob("*.py")


def _passes_token(call: ast.Call) -> bool:
    """True if the call passes an hf_token (keyword, or the 2nd positional slot)."""
    if any(kw.arg in ("hf_token", "token") for kw in call.keywords if kw.arg is not None):
        return True
    return len(call.args) >= 2


def _call_name(call: ast.Call):
    fn = call.func
    return fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)


def test_capability_probes_thread_the_hf_token():
    """Every capability-probe caller passes the token; a token-less probe misclassifies
    a gated model (the /check-vision regression)."""
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
    """GGUF never executes auto_map, so requires_trust_remote_code is reported via the
    resolver or False, never the raw YAML bool() (the round-6 regression)."""
    src = (_BACKEND / "routes" / "inference.py").read_text()
    assert "requires_trust_remote_code = bool(" not in src, (
        "Report requires_trust_remote_code via _resolve_loaded_trust_remote_code "
        "(non-GGUF) or set it False (GGUF); never bool(inference_config.get(...))."
    )


def test_capability_detection_caches_are_token_aware():
    """Every capability cache is keyed by (model, token_fingerprint) so an unauthenticated
    miss cannot poison a later authenticated lookup (the audio-cache regression)."""
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
    """Every worker that runs a load gate also resolves the LoRA base, so a poisoned or
    custom-code base is never skipped."""
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


def test_rag_embedding_path_runs_the_malware_gate():
    """The RAG embedding model is set through /settings and later loaded by
    SentenceTransformer, which deserializes pickles; both sites must run the malware gate
    or a flagged repo loads unscanned (bypassing the normal model-load protections)."""
    offenders = []
    for rel in ("routes/settings.py", "core/rag/embeddings.py"):
        if "evaluate_file_security(" not in (_BACKEND / rel).read_text():
            offenders.append(
                f"{rel} loads/persists an embedding model without evaluate_file_security"
            )
    assert not offenders, "\n".join(offenders)
