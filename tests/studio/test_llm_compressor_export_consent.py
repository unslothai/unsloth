"""Static guards for Studio llm-compressor export consent (#8904)."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding = "utf-8")


def test_probe_route_is_registered():
    routes = _read("studio/backend/routes/export.py")
    assert "/llm-compressor-probe" in routes
    assert "LlmCompressorExportProbeResponse" in routes


def test_merged_request_accepts_install_missing_dependencies():
    models = _read("studio/backend/models/export.py")
    assert "install_missing_dependencies: bool = Field(" in models


def test_export_pipeline_threads_consent_flag():
    export_py = _read("studio/backend/core/export/export.py")
    assert "install_missing_dependencies" in export_py
    assert "allow_provision = install_missing_dependencies" in export_py
    worker = _read("studio/backend/core/export/worker.py")
    assert 'cmd.get("install_missing_dependencies"' in worker
    orch = _read("studio/backend/core/export/orchestrator.py")
    assert '"install_missing_dependencies": install_missing_dependencies' in orch


def test_shadow_pythonpath_requires_explicit_provision():
    tv = _read("studio/backend/utils/transformers_version.py")
    assert "def llmcompressor_shadow_pythonpath(*, allow_provision: bool = False)" in tv


def test_frontend_export_wires_consent_dialog():
    root = _read("studio/frontend/src/app/routes/__root.tsx")
    assert "LlmCompressorConsentDialog" in root
    page = _read("studio/frontend/src/features/export/export-page.tsx")
    assert "confirmLlmCompressorInstallIfNeeded" in page
    assert "installMissingDependencies" in page
