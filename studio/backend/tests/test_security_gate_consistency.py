# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The guards enumerate every model-load gate site, so a sibling that drops the token fails here."""

import ast
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent

# A token-less probe 404s on a gated repo; scan callers under routes/ and core/ (probes live in utils/).
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
    """Capability probes must thread the HF token; without it gated models misclassify."""
    offenders = []
    for path in _iter_caller_files():
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
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
    """For GGUF, requires_trust_remote_code is inert and never taken from raw YAML bool()."""
    src = (_BACKEND / "routes" / "inference.py").read_text(encoding = "utf-8")
    assert "requires_trust_remote_code = bool(" not in src, (
        "Report requires_trust_remote_code via _resolve_loaded_trust_remote_code "
        "(non-GGUF) or set it False (GGUF); never bool(inference_config.get(...))."
    )


def test_capability_detection_caches_are_token_aware():
    """Every capability cache is keyed by (model, token_fingerprint) so an unauthenticated
    miss cannot poison a later authenticated lookup (the audio-cache regression)."""
    src = (_BACKEND / "utils" / "models" / "model_config.py").read_text(encoding = "utf-8")
    # Resolve type aliases first: an aliased cache is still tuple-keyed, so matching "Dict[Tuple" fails.
    tuple_aliases = {
        line.split("=", 1)[0].strip()
        for line in src.splitlines()
        if "=" in line
        and not line.startswith((" ", "\t"))
        and ("Tuple[" in line.split("=", 1)[1] or "tuple[" in line.split("=", 1)[1])
    }
    offenders = []
    for line in src.splitlines():
        stripped = line.strip()
        if "_detection_cache:" in stripped and stripped.endswith("= {}"):
            key = stripped.split("Dict[", 1)[-1].split(",", 1)[0].strip()
            if not ("Dict[Tuple" in stripped or "Dict[tuple" in stripped or key in tuple_aliases):
                offenders.append(stripped)
    assert not offenders, (
        "A capability cache must be keyed by (model, token_fingerprint), not the bare "
        "model name:\n  " + "\n  ".join(offenders)
    )


def test_malware_and_consent_gates_cover_the_lora_base():
    """Gated workers must resolve the LoRA base too, so a poisoned base cannot be skipped."""
    gated_workers = [
        "core/inference/worker.py",
        "core/export/worker.py",
        "core/training/worker.py",
    ]
    offenders = []
    for rel in gated_workers:
        src = (_BACKEND / rel).read_text(encoding = "utf-8")
        runs_gate = "evaluate_file_security(" in src or "evaluate_remote_code_consent" in src
        resolves_base = "get_base_model_from_lora_identifier(" in src or "base_model" in src
        if runs_gate and not resolves_base:
            offenders.append(f"{rel} runs a load gate but never resolves the LoRA base")
    assert not offenders, "\n".join(offenders)


def test_rag_embedding_path_runs_the_malware_gate():
    """RAG embeddings reach SentenceTransformer, which unpickles; both sites must run the malware gate."""
    offenders = []
    for rel in ("routes/settings.py", "core/rag/embeddings.py"):
        if "evaluate_file_security(" not in (_BACKEND / rel).read_text(encoding = "utf-8"):
            offenders.append(
                f"{rel} loads/persists an embedding model without evaluate_file_security"
            )
    assert not offenders, "\n".join(offenders)
