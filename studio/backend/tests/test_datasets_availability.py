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
from unittest import mock
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
        """Whichever text is chosen, it must say what to DO, not only what is wrong."""
        detail = datasets_availability.unavailable_detail().lower()
        assert "installer" in detail or "install" in detail

    def test_the_arm64_windows_advice_is_confined_to_arm64_windows(self):
        """The tier is the reason this gate exists, not the only way to reach it.

        Any environment missing `datasets` lands in the same branch, including an
        ordinary Linux box with a half-finished venv. Telling that user to install
        x64 Python from python.org because it "runs emulated" is advice for a
        machine they are not sitting at, and it buries the real remedy. This test
        exists because the message used to be a single constant that said exactly
        that to everyone.
        """
        arm64 = datasets_availability._ARM64_WINDOWS_MSG.lower()
        assert "arm64" in arm64 and "x64" in arm64

        generic = datasets_availability._GENERIC_MSG.lower()
        assert "arm64" not in generic, "non-ARM64 users must not be told about ARM64"
        assert "x64" not in generic, "nor sent to download a different interpreter"
        assert "pip install datasets" in generic

        # Both halves say what is lost, since that is the part every caller needs.
        for message in (arm64, generic):
            assert "training" in message and "chat" in message

    def test_the_tier_predicate_reads_the_interpreter_not_the_hardware(self):
        """`sysconfig.get_platform()`, because the question is which wheels this
        interpreter can install. An x64 Python emulated on ARM hardware answers
        `win-amd64` and CAN install pyarrow, so it must not get the tier text."""
        import sysconfig

        with mock.patch.object(datasets_availability.sys, "platform", "win32"):
            with mock.patch.object(sysconfig, "get_platform", return_value="win-arm64"):
                assert datasets_availability._is_arm64_windows() is True
            with mock.patch.object(sysconfig, "get_platform", return_value="win-amd64"):
                assert datasets_availability._is_arm64_windows() is False

        # Linux and macOS never take the tier branch, whatever their arch says.
        for platform_name in ("linux", "darwin"):
            with mock.patch.object(datasets_availability.sys, "platform", platform_name):
                with mock.patch.object(sysconfig, "get_platform", return_value="win-arm64"):
                    assert datasets_availability._is_arm64_windows() is False

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

    def test_verdict_is_published_before_detection_settles(self):
        """It does not depend on the hardware pass and never changes, so gating it
        on DETECTION_COMPLETE would leave the first replies with chat_only set and
        no reason, and the UI polling for a verdict that is already known."""
        source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
        function = source[source.index("def _hardware_snapshot(") :]
        function = function[: function.index("\n\n\n")]
        assert function.index("datasets_unavailable") < function.index(
            "DETECTION_COMPLETE.is_set()"
        )

    def test_frontend_explains_the_reason(self):
        frontend = _BACKEND.parent / "frontend" / "src" / "components" / "app-sidebar.tsx"
        source = frontend.read_text(encoding = "utf-8")
        assert '"datasets_unavailable"' in source
        assert "x64" in source
