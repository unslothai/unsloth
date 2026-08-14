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
import re
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
    def test_hub_dataset_endpoints_are_gated(self):
        """Per endpoint now, not on the router: see TestOnlyDatasetRoutesAreGated."""
        source = (_BACKEND / "hub" / "routes" / "datasets.py").read_text(encoding = "utf-8")
        assert "require_datasets_http" in source
        assert "needs_datasets = Depends(require_datasets_http)" in source

    @pytest.mark.parametrize("route", ['"/start"'])
    def test_training_start_routes_are_gated(self, route):
        """The LLM start endpoint, which loads a dataset. /diffusion/start trains from
        a data_dir and is not gated; /status, /progress and /metrics stay reachable so
        a UI already polling them keeps working."""
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
    ungated, so both answered 500 where the tier promises 503 -- but per route,
    because each mount also carries handlers that never touch the library."""

    @staticmethod
    def _mount(name: str) -> str:
        """Exactly one include_router() call. A window wide enough to spill into the
        next mount would read that one's dependency as this one's."""
        source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
        index = source.index(f"app.include_router(\n    {name},")
        return source[index : source.index("\n)", index)]

    @pytest.mark.parametrize("router", ["datasets_router", "data_recipe_router"])
    def test_neither_mount_is_gated_wholesale(self, router):
        assert "Depends(require_datasets_http)" not in self._mount(router)

    def test_the_legacy_alias_gates_check_format(self):
        """The one handler behind it that reaches `from datasets import`, matching
        the hub router it aliases."""
        source = (_BACKEND / "routes" / "datasets.py").read_text(encoding = "utf-8")
        assert "require_datasets_http" in TestOnlyDatasetRoutesAreGated._decorator(
            source, "/check-format"
        )

    @pytest.mark.parametrize(
        "path", ['"/upload"', '"/local"', '"/download-progress"', '"/ai-assist-mapping"']
    )
    def test_the_legacy_alias_leaves_the_rest_open(self, path):
        source = (_BACKEND / "routes" / "datasets.py").read_text(encoding = "utf-8")
        decorator = TestOnlyDatasetRoutesAreGated._decorator(source, path.strip('"'))
        assert "require_datasets_http" not in decorator, path

    @pytest.mark.parametrize(
        "path", ['"/seed/inspect"', '"/seed/inspect-upload"', '"/jobs"', '"/jobs/{job_id}/dataset"']
    )
    def test_data_recipe_gates_the_routes_that_load_data(self, path):
        """Per route since the blanket mount went: these three reach load_dataset or
        pandas, while the seed-file deletes only unlink under the upload root."""
        for module in ("seed", "jobs"):
            source = (_BACKEND / "routes" / "data_recipe" / f"{module}.py").read_text(
                encoding = "utf-8"
            )
            if path not in source:
                continue
            index = source.index(path)
            assert "require_datasets_http" in source[index : index + 400], path
            return
        raise AssertionError(f"{path} not found in the data recipe routes")

    @pytest.mark.parametrize(
        "path",
        ['"/seed/unstructured-file/{block_id}/{file_id}"', '"/seed/unstructured-block/{block_id}"'],
    )
    def test_data_recipe_cleanup_stays_open(self, path):
        source = (_BACKEND / "routes" / "data_recipe" / "seed.py").read_text(encoding = "utf-8")
        index = source.index(path)
        # Decorator only: the handler may be sync or async, so end at whichever
        # definition keyword comes first rather than assuming one of them.
        ends = [source.find(marker, index) for marker in ("\ndef ", "\nasync def ")]
        end = min(position for position in ends if position != -1)
        assert "require_datasets_http" not in source[index:end]

    def test_import_example_reaches_the_gate(self):
        """_materialize_hf_dataset() runs `from datasets import load_dataset`, so an
        ungated import surfaces as a 502 rather than the tier's 503. Inside the
        handler now, per loader: see TestGatesFollowTheLoader."""
        source = (_BACKEND / "routes" / "training.py").read_text(encoding = "utf-8")
        index = source.index('"/diffusion/dataset/import-example"')
        assert "require_datasets_http" in source[index : index + 2000]

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

    def test_a_recovery_is_not_latched(self, monkeypatch):
        """A probe taken mid-install can see the spec before the package works.
        Latching that would trade this gate's 503 for a permanent 500."""
        monkeypatch.delenv("UNSLOTH_FORCE_NO_DATASETS", raising = False)
        monkeypatch.setattr(datasets_availability, "DATASETS_AVAILABLE", False)
        monkeypatch.setattr(datasets_availability, "_probe", lambda: True)
        assert datasets_availability.datasets_available() is True
        assert datasets_availability.DATASETS_AVAILABLE is False

        monkeypatch.setattr(datasets_availability, "_probe", lambda: False)
        assert datasets_availability.datasets_available() is False

    def test_a_true_answer_is_not_reprobed(self, monkeypatch):
        """The cost stays on the failing path: an install that has the library must
        not pay two find_spec calls per gated request."""
        monkeypatch.delenv("UNSLOTH_FORCE_NO_DATASETS", raising = False)
        monkeypatch.setattr(datasets_availability, "DATASETS_AVAILABLE", True)
        calls = []
        monkeypatch.setattr(datasets_availability, "_probe", lambda: calls.append(1) or True)
        assert datasets_availability.datasets_available() is True
        assert calls == []


class TestOnlyDatasetRoutesAreGated:
    """The tier still downloads models, so it still accumulates dataset caches.

    Gating the whole hub router took away the endpoints that reclaim that disk space
    and manage download jobs, none of which touch the datasets library.
    """

    @staticmethod
    def _source() -> str:
        return (_BACKEND / "hub" / "routes" / "datasets.py").read_text(encoding = "utf-8")

    def test_the_router_itself_is_not_gated(self):
        source = self._source()
        assert "router = APIRouter()" in source
        assert "APIRouter(dependencies" not in source

    @staticmethod
    def _decorator(source: str, path: str) -> str:
        """The whole decorator for `path`, not just the line the path sits on.

        A decorator that outgrows one line puts `dependencies` on a line of its own, so a
        per-line check would report every such route as ungated whether it is or not.
        """
        index = source.index(f'"{path}"')
        # `\ndef ` alone would run past every `async def` route and swallow the decorators
        # after it, so a gated route could read as ungated.
        end = re.compile(r"\n(?:async )?def ").search(source, index)
        return source[index : end.start() if end else len(source)]

    @pytest.mark.parametrize("path", ["/check-format"])
    def test_dataset_paths_keep_the_gate(self, path):
        assert "needs_datasets" in self._decorator(self._source(), path)

    @pytest.mark.parametrize(
        "path",
        [
            "/cached",
            "/download",
            # Uploads write a file, /local walks the filesystem, /local-options
            # reimplements split inference without the library on purpose, and
            # ai-assist-mapping reads samples the client already sent.
            "/upload",
            "/local",
            "/local-options",
            "/ai-assist-mapping",
            "/download/cancel",
            "/download-status",
            "/active-downloads",
            "/transport-status",
        ],
    )
    def test_cache_and_download_paths_stay_open(self, path):
        """Starting a download included: download_dataset_response and
        hub/workers/hf_download.py::_download_dataset reach huggingface_hub and the cache
        utilities only, so the tier that downloads models can populate dataset caches too."""
        assert "needs_datasets" not in self._decorator(self._source(), path), path


class TestGatesFollowTheLoader:
    """A gate on a route that does not need the library is a feature removed for
    nothing. The curated diffusion examples are two loaders, and only one of them
    reaches datasets."""

    def test_the_example_import_gates_inside_not_on_the_route(self):
        source = (_BACKEND / "routes" / "training.py").read_text(encoding = "utf-8")
        index = source.index('"/diffusion/dataset/import-example"')
        decorator = source[index : index + 220]
        assert "require_datasets_http" not in decorator
        body = source[index : index + 2000]
        assert 'entry.get("loader") == "hf_dataset"' in body
        assert "await require_datasets_http()" in body

    def test_the_imagefolder_example_is_not_gated(self):
        """tarot-1920 materializes through huggingface_hub and file copies."""
        source = (_BACKEND / "routes" / "training.py").read_text(encoding = "utf-8")
        assert '"loader": "imagefolder_jsonl"' in source


class TestVerificationAsksTheTargetVenv:
    """verify_install() also checks OTHER venvs, so the lifts follow the manifest's
    recorded interpreter tag rather than the one doing the checking."""

    def test_the_recorded_tag_wins(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "im_gate",
            _BACKEND.parent / "install_manifest.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module._is_windows_arm64_python({"platform_tag": "win-arm64"}) is True
        assert module._is_windows_arm64_python({"platform_tag": "win-amd64"}) is False
        # No key: an older manifest, so fall back to this interpreter (Linux here).
        assert module._is_windows_arm64_python({}) is False
        assert "platform_tag" in (_BACKEND.parent / "install_manifest.py").read_text(
            encoding = "utf-8"
        )


class TestGatesLeaveWorkingFeaturesAlone:
    """Each of these paths runs without the datasets library, so gating it removed a
    feature from installs that merely lost the package, for nothing."""

    def test_diffusion_training_is_not_gated(self):
        """Image training reads a data_dir; no diffusion module imports datasets."""
        source = (_BACKEND / "routes" / "training.py").read_text(encoding = "utf-8")
        index = source.index('"/diffusion/start"')
        assert "require_datasets_http" not in source[index : index + 300]

    def test_starting_a_download_is_not_gated(self):
        """hf_download._download_dataset() is huggingface_hub and cache utilities, so a
        tier install can still fill a cache for the x64 environment it will move to."""
        source = (_BACKEND / "hub" / "routes" / "datasets.py").read_text(encoding = "utf-8")
        index = source.index('"/download"')
        assert "needs_datasets" not in source[index : source.index("\n", index)]

    def test_download_progress_is_not_gated(self):
        """Snapshot accounting over the cache dir, so a tier install can still watch
        the downloads it is allowed to start and cancel."""
        source = (_BACKEND / "hub" / "routes" / "datasets.py").read_text(encoding = "utf-8")
        index = source.index('"/download-progress"')
        assert "needs_datasets" not in source[index : source.index("\n", index)]

    def test_paging_a_finished_recipe_is_gated_not_500(self):
        """duckdb .fetchdf() is pandas, and the except arm around it catches only
        duckdb errors, so an ungated read surfaced a 422 naming pandas."""
        source = (_BACKEND / "routes" / "data_recipe" / "jobs.py").read_text(encoding = "utf-8")
        index = source.index('"/jobs/{job_id}/dataset"')
        assert "require_datasets_http" in source[index : source.index("\ndef ", index)]

    def test_the_data_recipe_router_is_not_gated_wholesale(self):
        """Its cleanup endpoints only unlink files under the upload root; a blanket
        gate stopped users reclaiming that space."""
        source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
        index = source.index("data_recipe_router,")
        assert "require_datasets_http" not in source[index : index + 200]

    def test_setup_forces_a_pass_when_the_tier_changes(self):
        """verify_install() judges a venv against the tier it already has, so the fast
        path skipped the pass that would carry out a requested transition."""
        source = (_BACKEND.parents[1] / "studio" / "setup.ps1").read_text(encoding = "utf-8")
        index = source.index("requested install tier differs")
        block = source[index - 1600 : index + 200]
        assert "$_noDatasetsRequested" in block
        assert "recorded_no_datasets" in block


class TestTheCapabilityIsPublishedAndActedOn:
    """The tier is only useful if the client learns about it in every reply, can act
    on it before a page renders, and can see it clear again without a reload."""

    def test_health_publishes_it_outside_the_hardware_branch(self):
        """It is answered from the interpreter, so a deferred or provisional hardware
        reply still carries it -- the client treats those as settled and would keep
        its default (available) for the session."""
        source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
        published = source.index('authed["datasets_available"] = datasets_available()')
        branch = source.index("    if snapshot is not None:", source.index("authed = {"))
        assert published < branch

    def test_the_route_guard_redirects_the_two_gated_pages(self):
        """Disabling a sidebar row is not a guard: a bookmark or a reload still lands
        on a page whose every call answers 503."""
        source = (_BACKEND.parent / "frontend" / "src" / "app" / "routes" / "__root.tsx").read_text(
            encoding = "utf-8"
        )
        assert '["/studio", "/data-recipes"]' in source
        index = source.index("needsDatasets(location.pathname)")
        assert "!unmeasured && !datasetsAvailable" in source[index - 120 : index]

    def test_the_recovery_poll_treats_missing_datasets_as_unsettled(self):
        """Off ARM64 the host is not chat-only, so `!chatOnly` alone settled it and the
        rows stayed disabled until a reload after the 503's own advice was followed."""
        source = (
            _BACKEND.parent / "frontend" / "src" / "components" / "app-sidebar.tsx"
        ).read_text(encoding = "utf-8")
        index = source.index("const selfHealSettled =")
        assert "!datasetsMissing &&" in source[index : index + 200]
        # And in the dependency array, or the effect keeps a stale reading of it.
        deps = source.index("}, [", index)
        assert "datasetsMissing," in source[deps : source.index("]", deps)]

    def test_the_recipes_hint_prefers_the_dataset_detail(self):
        """chatOnlyDetail is null on a host that merely lost the library, and the
        fallback text is the ARM64 remedy -- advice for a machine they are not at."""
        source = (
            _BACKEND.parent / "frontend" / "src" / "components" / "app-sidebar.tsx"
        ).read_text(encoding = "utf-8")
        assert "const recipesDetail = datasetsDetail ?? chatOnlyDetail;" in source

    def test_winget_rechecks_for_an_x64_python_by_path(self):
        """winget does not reorder PATH, so its x64 build can be installed while
        `python` still resolves the native one that sent us here."""
        source = (_BACKEND.parents[1] / "studio" / "setup.ps1").read_text(encoding = "utf-8")
        assert "function Find-X64SetupPython" in source
        index = source.index("$_x64Python = Find-X64SetupPython")
        assert "Add-PythonDirToProcessPath $_x64Python" in source[index : index + 300]
        assert source.index("function Find-X64SetupPython") < index


class TestTheTierRemovesTrainingNotTheDevice:
    """UNSLOTH_NO_DATASETS=1 on an x64 GPU host keeps the GPU, safetensors inference,
    Video and Hub Run. Only training and the data features go, so the health payload
    carries that as its own capability instead of calling the device chat-only."""

    def test_health_publishes_the_capability(self):
        source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
        assert 'authed["datasets_available"] = datasets_available()' in source
        assert "datasets_unavailable_detail" in source

    def test_chat_only_is_only_forced_on_native_arm64(self):
        source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
        index = source.index('return True, "datasets_unavailable"')
        assert "is_inference_only_tier() and _is_arm64_windows()" in source[index - 400 : index]

    def test_the_frontend_gates_training_on_it(self):
        source = (
            _BACKEND.parent / "frontend" / "src" / "components" / "app-sidebar.tsx"
        ).read_text(encoding = "utf-8")
        assert "const trainingBlocked = chatOnlyMeasured || datasetsMissing;" in source
        assert "disabled: trainingBlocked," in source

    def test_the_store_keeps_a_measured_false(self):
        """A provisional or unauthenticated reply omits the field, and must not flip a
        measured false back to true and re-enable Train on a tier install."""
        source = (_BACKEND.parent / "frontend" / "src" / "config" / "env.ts").read_text(
            encoding = "utf-8"
        )
        assert "data.datasets_available ?? previous.datasetsAvailable" in source
