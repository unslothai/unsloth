# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ARM64 inference-only tier ships without ``datasets``, so everything that
reaches it has to degrade rather than 500.

Windows on ARM has no win_arm64 wheel for pyarrow -- ``datasets``' storage engine
-- at any version, so when no x64 interpreter can be obtained the installer drops
the library entirely (issue #8495). These tests pin the two halves of that
promise: startup does not touch ``datasets``, and the endpoints that do answer
503 with a stated reason.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import datasets_availability  # noqa: E402


class TestAvailabilityProbe:
    def test_module_imports_without_importing_datasets(self):
        """find_spec only. Importing datasets here would cost a pyarrow load on
        every environment that has one, on a module read from request paths."""
        source = (_BACKEND / "utils" / "datasets_availability.py").read_text(encoding = "utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(name.split(".")[0] in {"datasets", "pyarrow"} for name in imported)

    def test_env_override_forces_unavailable(self, monkeypatch):
        """The tier is a Windows-on-ARM install, so every other platform needs a
        way to reach the unavailable path -- in tests and when reproducing a report."""
        monkeypatch.setenv("UNSLOTH_FORCE_NO_DATASETS", "1")
        assert datasets_availability.datasets_available() is False
        with pytest.raises(datasets_availability.DatasetsUnavailable):
            datasets_availability.require_datasets()

    def test_unavailable_is_a_runtime_error(self):
        """Subclasses RuntimeError so existing `except RuntimeError` callers keep
        working; mirrors rag_db.RagExtensionUnavailable."""
        assert issubclass(datasets_availability.DatasetsUnavailable, RuntimeError)

    def test_detail_names_the_fix_not_just_the_problem(self):
        detail = datasets_availability.unavailable_detail()
        assert "x64" in detail.lower()
        assert "arm64" in detail.lower()

    def test_http_dependency_raises_503(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_FORCE_NO_DATASETS", "1")
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as excinfo:
            datasets_availability.require_datasets_http()
        assert excinfo.value.status_code == 503
        assert "datasets" in str(excinfo.value.detail).lower()

    def test_available_by_default_in_a_normal_install(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_FORCE_NO_DATASETS", raising = False)
        # This test environment has datasets installed; the point is that the
        # normal path is unchanged and no gate fires.
        assert (
            datasets_availability.datasets_available() is datasets_availability.DATASETS_AVAILABLE
        )
        datasets_availability.require_datasets_http()


class TestStartupDoesNotNeedDatasets:
    """The premise the whole tier rests on: nothing imported at server startup
    reaches ``datasets`` or ``pyarrow``, so chat and model downloads work without
    them. Static, over module-level imports only -- a runtime import graph would
    need the whole torch stack loaded."""

    # Deliberately not the full transitive closure: this pins the modules that
    # historically were one edit away from dragging the training stack into
    # startup. trainer.py imports datasets at module scope and must stay off
    # every one of these import lists.
    STARTUP_MODULES = (
        "main.py",
        "run.py",
        "routes/__init__.py",
        "hub/routes/__init__.py",
        "routes/training.py",
        "routes/inference.py",
        "utils/hardware/hardware.py",
    )

    @staticmethod
    def _module_level_imports(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        names: set[str] = set()
        for node in tree.body:  # body only: nested = function-local = lazy
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module.split(".")[0])
            elif isinstance(node, (ast.If, ast.Try)):
                # `if TYPE_CHECKING:` and guarded imports still execute (or are
                # skipped) at module scope, so walk them too.
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Import):
                        names.update(alias.name.split(".")[0] for alias in sub.names)
                    elif isinstance(sub, ast.ImportFrom) and sub.module and sub.level == 0:
                        names.add(sub.module.split(".")[0])
        return names

    @pytest.mark.parametrize("relative", STARTUP_MODULES)
    def test_no_module_level_datasets_import(self, relative):
        names = self._module_level_imports(_BACKEND / relative)
        assert "datasets" not in names, f"{relative} imports datasets at module scope"
        assert "pyarrow" not in names, f"{relative} imports pyarrow at module scope"

    def test_trainer_is_the_known_exception_and_stays_out_of_startup(self):
        """core/training/trainer.py DOES import datasets at module scope. That is
        allowed because it only ever loads inside the spawned training worker; if
        a route ever imports it at module scope, the tier's server stops booting."""
        trainer = self._module_level_imports(_BACKEND / "core" / "training" / "trainer.py")
        assert "datasets" in trainer  # if this changes, the note above is stale
        for relative in self.STARTUP_MODULES:
            source = (_BACKEND / relative).read_text(encoding = "utf-8")
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert (
                        "core.training.trainer" not in node.module
                    ), f"{relative} imports the training worker at module scope"


class TestRoutesAreGated:
    def test_hub_datasets_router_is_gated(self):
        source = (_BACKEND / "hub" / "routes" / "datasets.py").read_text(encoding = "utf-8")
        assert "require_datasets_http" in source
        assert "APIRouter(dependencies = [Depends(require_datasets_http)])" in source

    @pytest.mark.parametrize("route", ['"/start"', '"/diffusion/start"'])
    def test_training_start_routes_are_gated(self, route):
        """Only the start endpoints: /status, /progress and /metrics must stay
        reachable so a UI already polling them keeps working."""
        source = (_BACKEND / "routes" / "training.py").read_text(encoding = "utf-8")
        index = source.index(route)
        decorator = source[index : index + 400]
        assert "require_datasets_http" in decorator

    def test_status_route_is_not_gated(self):
        source = (_BACKEND / "routes" / "training.py").read_text(encoding = "utf-8")
        index = source.index('@router.get("/status")')
        assert "require_datasets_http" not in source[index : index + 300]


class TestHealthVerdict:
    def test_snapshot_reports_datasets_unavailable(self):
        """An install without datasets cannot train on ANY device, so the health
        verdict must say so rather than repeat the hardware pass's answer."""
        source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
        assert 'return True, "datasets_unavailable", datasets_unavailable_detail()' in source

    def test_frontend_explains_the_reason(self):
        frontend = _BACKEND.parent / "frontend" / "src" / "components" / "app-sidebar.tsx"
        source = frontend.read_text(encoding = "utf-8")
        assert '"datasets_unavailable"' in source
        assert "x64" in source
