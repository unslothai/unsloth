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
import asyncio
import sys
import tempfile
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
            with mock.patch.object(sysconfig, "get_platform", return_value = "win-arm64"):
                assert datasets_availability._is_arm64_windows() is True
            with mock.patch.object(sysconfig, "get_platform", return_value = "win-amd64"):
                assert datasets_availability._is_arm64_windows() is False

        # Linux and macOS never take the tier branch, whatever their arch says.
        for platform_name in ("linux", "darwin"):
            with mock.patch.object(datasets_availability.sys, "platform", platform_name):
                with mock.patch.object(sysconfig, "get_platform", return_value = "win-arm64"):
                    assert datasets_availability._is_arm64_windows() is False

    def test_http_dependency_raises_503(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_FORCE_NO_DATASETS", "1")
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(datasets_availability.require_datasets_http())
        assert excinfo.value.status_code == 503
        assert "datasets" in str(excinfo.value.detail).lower()

    def test_available_by_default_in_a_normal_install(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_FORCE_NO_DATASETS", raising = False)
        # This test environment has datasets installed; the point is that the
        # normal path is unchanged and no gate fires.
        assert (
            datasets_availability.datasets_available() is datasets_availability.DATASETS_AVAILABLE
        )
        asyncio.run(datasets_availability.require_datasets_http())

    def test_the_gate_is_async_so_it_never_queues_for_a_worker(self):
        """FastAPI runs a SYNCHRONOUS dependency in the AnyIO worker pool. On a
        healthy install this gate only reads a bool, but as a sync dependency every
        gated request would have to win a worker token before it could even be
        rejected as unauthenticated, which under load from this app's sync routes
        puts a queue in front of a dict lookup."""
        import inspect

        assert inspect.iscoroutinefunction(datasets_availability.require_datasets_http)
        # And the one direct caller awaits it, or the tier would accept MCP training
        # jobs again: an un-awaited coroutine raises nothing.
        mcp = (_BACKEND / "mcp_server.py").read_text(encoding = "utf-8")
        assert "await require_datasets_http()" in mcp


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


class TestTheVerdictIsKeyedOffTheTierNotTheLibrary:
    """chat_only is a statement about the DEVICE: the frontend hides safetensors
    models, Video and the Hub's Run button on it (pickers.tsx, model-inspector.tsx).

    True on the ARM64 tier, which is a torch-less install. False on a GPU box whose
    venv merely lost `datasets` to a half-finished update, and answering chat_only
    there would strip features from a machine this change must not touch. So the
    health verdict asks about the tier while the route gates ask about the library.
    """

    def test_a_missing_library_alone_is_not_the_tier(self, monkeypatch):
        monkeypatch.setattr(datasets_availability, "DATASETS_AVAILABLE", False)
        monkeypatch.setattr(datasets_availability, "_probe", lambda: False)
        monkeypatch.delenv("UNSLOTH_FORCE_NO_DATASETS", raising = False)
        monkeypatch.setattr(datasets_availability, "_is_arm64_windows", lambda: False)
        monkeypatch.setattr(datasets_availability.sys, "prefix", tempfile.mkdtemp())
        assert datasets_availability.datasets_available() is False
        assert datasets_availability.is_inference_only_tier() is False

    def test_a_native_arm64_windows_interpreter_is_the_tier(self, monkeypatch):
        monkeypatch.setattr(datasets_availability, "DATASETS_AVAILABLE", False)
        monkeypatch.setattr(datasets_availability, "_probe", lambda: False)
        monkeypatch.delenv("UNSLOTH_FORCE_NO_DATASETS", raising = False)
        monkeypatch.setattr(datasets_availability, "_is_arm64_windows", lambda: True)
        assert datasets_availability.is_inference_only_tier() is True

    def test_the_installer_marker_is_the_tier(self, monkeypatch):
        """An x64 interpreter can carry the tier too: UNSLOTH_NO_DATASETS=1 is an
        opt-in, and the marker is what survives the installer's own process."""
        root = tempfile.mkdtemp()
        (Path(root) / ".unsloth-no-datasets").write_text("", encoding = "utf-8")
        monkeypatch.setattr(datasets_availability, "DATASETS_AVAILABLE", False)
        monkeypatch.setattr(datasets_availability, "_probe", lambda: False)
        monkeypatch.delenv("UNSLOTH_FORCE_NO_DATASETS", raising = False)
        monkeypatch.setattr(datasets_availability, "_is_arm64_windows", lambda: False)
        monkeypatch.setattr(datasets_availability.sys, "prefix", root)
        assert datasets_availability.is_inference_only_tier() is True

    def test_an_ordinary_install_is_never_the_tier(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_FORCE_NO_DATASETS", raising = False)
        assert datasets_availability.is_inference_only_tier() is False

    def test_the_force_hook_reaches_the_tier_path(self, monkeypatch):
        """Otherwise the degraded UI could only be exercised on ARM64 Windows."""
        monkeypatch.setenv("UNSLOTH_FORCE_NO_DATASETS", "1")
        assert datasets_availability.is_inference_only_tier() is True


class TestHealthVerdict:
    def test_snapshot_reports_datasets_unavailable(self):
        """The tier cannot train on ANY device, so the health verdict must say so
        rather than repeat the hardware pass's answer."""
        source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
        assert 'return True, "datasets_unavailable", datasets_unavailable_detail()' in source

    def test_the_verdict_asks_about_the_tier_and_the_gates_about_the_library(self):
        """The one-line version of the class above, pinned against a future edit
        that "simplifies" the predicate back to datasets_available()."""
        source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
        function = source[source.index("def _hardware_snapshot(") :]
        function = function[: function.index("\n\n\n")]
        assert "is_inference_only_tier()" in function
        assert "datasets_available()" not in function
        gate = (_BACKEND / "utils" / "datasets_availability.py").read_text(encoding = "utf-8")
        require = gate[gate.index("def require_datasets_http(") :]
        assert "datasets_available()" in require

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


class TestProbeRequiresPyarrow:
    """``datasets`` on disk is not the same as ``datasets`` importable.

    pyarrow is the whole reason this tier exists -- no win_arm64 wheel at any
    version -- so the environment where the distribution is present and its storage
    engine is not is the expected failure, not an exotic one: a pass that installed
    datasets and then died building pyarrow leaves exactly that. ``import datasets``
    reaches pyarrow eagerly through arrow_dataset, so probing the name alone
    published "available" for an install whose first ``from datasets import ...``
    still raises ModuleNotFoundError -- the 500 this gate replaces.
    """

    @staticmethod
    def _probe_with(present: set[str]) -> bool:
        def fake_find_spec(name: str):
            return object() if name in present else None

        with mock.patch.object(
            datasets_availability.importlib.util, "find_spec", side_effect = fake_find_spec
        ):
            return datasets_availability._probe()

    def test_datasets_without_pyarrow_is_unavailable(self):
        assert self._probe_with({"datasets"}) is False

    def test_pyarrow_without_datasets_is_unavailable(self):
        assert self._probe_with({"pyarrow"}) is False

    def test_both_present_is_available(self):
        assert self._probe_with({"datasets", "pyarrow"}) is True

    def test_neither_present_is_unavailable(self):
        assert self._probe_with(set()) is False

    def test_probe_still_never_imports_datasets(self):
        """The pyarrow check must stay a find_spec too: importing either to find
        out costs the multi-second load this module exists to avoid."""
        source = (_BACKEND / "utils" / "datasets_availability.py").read_text(encoding = "utf-8")
        assert "import pyarrow" not in source
        assert '_spec_present("pyarrow")' in source


class TestCompatibilityRoutersAreGated:
    """The gate has to cover every mount that reaches the library, not just the
    newest one. ``/api/datasets`` is the retained compatibility alias an older
    client still calls and it reaches the same formatting service; Data Recipes
    reads seeds through pandas and ``datasets.load_dataset``. Both were mounted
    ungated, so both answered 500 where the tier promises 503."""

    @staticmethod
    def _mount(name: str) -> str:
        """Exactly one include_router() call. A window wide enough to spill into the
        next mount would read that one's dependency as this one's."""
        source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
        index = source.index(f"app.include_router(\n    {name},")
        return source[index : source.index("\n)", index)]

    @pytest.mark.parametrize("router", ["datasets_router", "data_recipe_router"])
    def test_legacy_router_carries_the_dependency(self, router):
        assert "Depends(require_datasets_http)" in self._mount(router)

    def test_import_example_route_is_gated(self):
        """_materialize_hf_dataset() runs `from datasets import load_dataset`, so
        an ungated import surfaces as a 502 rather than the tier's 503."""
        source = (_BACKEND / "routes" / "training.py").read_text(encoding = "utf-8")
        index = source.index('"/diffusion/dataset/import-example"')
        assert "require_datasets_http" in source[index : index + 300]

    def test_mcp_start_training_enforces_the_gate(self):
        """mcp_server calls start_training() directly, so FastAPI never runs the
        decorator dependency: the tool accepted a job whose trainer then died on
        `import datasets`."""
        source = (_BACKEND / "mcp_server.py").read_text(encoding = "utf-8")
        index = source.index("async def start_training(")
        body = source[index : source.index("async def stop_training(", index)]
        assert "require_datasets_http()" in body


class TestChatOnlyVerdictDoesNotHideModels:
    """Which model FORMATS run here is the hardware's question, not this tier's.

    The tier machine is CPU-only, so the hardware pass reports chat_only anyway and
    the picker restricts to GGUF exactly as it does on every other CPU-only host.
    Exempting `datasets_unavailable` from that filter made safetensors selectable on
    a box the rest of the UI treats as GGUF-only."""

    @staticmethod
    def _adapter() -> str:
        path = (
            _BACKEND.parent / "frontend" / "src" / "features" / "chat" / "api" / "chat-adapter.ts"
        )
        return path.read_text(encoding = "utf-8")

    def test_no_reason_is_exempted_from_the_format_filter(self):
        source = self._adapter()
        assert "datasets_unavailable" not in source

    @pytest.mark.parametrize(
        "fn", ["function runsOnThisPlatform(", "function cachedModelsRunOnThisPlatform("]
    )
    def test_filters_ask_the_hardware_verdict(self, fn):
        source = self._adapter()
        index = source.index(fn)
        body = source[index : source.index("\n}", index)]
        assert "isChatOnly()" in body

    def test_sidebar_disables_data_recipes_in_the_tier(self):
        """Every Data Recipes seed path reads pandas or datasets.load_dataset, and
        this tier ships neither: an enabled entry that only fails on click is worse
        than a greyed-out one with a reason."""
        source = (
            _BACKEND.parent / "frontend" / "src" / "components" / "app-sidebar.tsx"
        ).read_text(encoding = "utf-8")
        index = source.index("const datasetsUnavailable =")
        assert 'chatOnlyReason === "datasets_unavailable"' in source[index : index + 200]
        recipes = source[source.index("    recipes: {") :]
        recipes = recipes[: recipes.index("\n    },")]
        assert "disabled: datasetsUnavailable" in recipes
        assert "recipesDisabledHint" in recipes


class TestTheGateReopensAfterAnInstall:
    """The 503 says `pip install datasets`. Cached for the life of the process, that
    advice does nothing until Studio is restarted, which the message never mentions."""

    def test_availability_is_reprobed_while_it_is_false(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_FORCE_NO_DATASETS", raising = False)
        monkeypatch.setattr(datasets_availability, "DATASETS_AVAILABLE", False)
        monkeypatch.setattr(datasets_availability, "_probe", lambda: False)
        assert datasets_availability.datasets_available() is False

        monkeypatch.setattr(datasets_availability, "_probe", lambda: True)
        assert datasets_availability.datasets_available() is True

    def test_a_true_answer_is_not_reprobed(self, monkeypatch):
        """The cost stays on the failing path: an install that has the library must
        not pay two find_spec calls per gated request."""
        monkeypatch.delenv("UNSLOTH_FORCE_NO_DATASETS", raising = False)
        monkeypatch.setattr(datasets_availability, "DATASETS_AVAILABLE", True)
        calls = []
        monkeypatch.setattr(datasets_availability, "_probe", lambda: calls.append(1) or True)
        assert datasets_availability.datasets_available() is True
        assert calls == []
