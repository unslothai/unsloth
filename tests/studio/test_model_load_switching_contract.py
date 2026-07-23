# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source-level contracts for superseding an in-flight chat model load."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "studio" / "frontend" / "src"


def _read(relative: str) -> str:
    return (FRONTEND / relative).read_text()


def test_unload_is_awaited_and_failure_blocks_replacement():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "await unloadModel({ model_path: model.id });" in runtime
    assert "await unloadModel({ model_path: model.id }).catch" not in runtime
    cancel = runtime.split("const cancelLoadRun = useCallback(", 1)[1]
    cancel = cancel.split("const cancelLoading = useCallback(", 1)[0]
    assert cancel.index("await unloadModel({ model_path: model.id });") < cancel.index(
        "await run.completionPromise;"
    )
    assert "const stopped = await cancelLoadRun(activeRun, true);" in runtime
    assert "if (!stopped)" in runtime
    assert "await refresh();" in runtime
    assert "if (!preserveCheckpoint) clearCheckpoint();" in cancel
    assert "useTransformersUpgradeDialogStore.getState().cancelPending();" in cancel
    assert "useRemoteCodeConsentDialogStore.getState().resolve(false);" in cancel
    assert "useHfTokenWarningStore.getState().resolve(\"cancel\");" in cancel
    assert "await run.completionPromise;" in cancel


def test_late_callbacks_are_bound_to_their_originating_run():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "activeLoadRunRef" in runtime
    assert "if (!ownsModelLoadRun(activeLoadRunRef.current, run)) return;" in runtime
    assert runtime.count("ownsModelLoadRun(activeLoadRunRef.current, run)") >= 6
    assert runtime.count("resetLoadingUi(run)") >= 2
    assert "if (run.cancelPromise) return;" in runtime
    assert "loadAttemptRef.current === run.attemptId" in runtime
    assert "loadIntentRef.current === run.intentId" in runtime
    assert "if (loadIntentRef.current !== loadIntentId) return;" in runtime
    assert "loadIntentRef.current += 1;" in runtime


def test_cancelled_load_does_not_report_success_to_callers():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    hub = _read("features/hub/hub-page.tsx")
    assert "if (abortCtrl.signal.aborted) return false;" in runtime
    assert "return true;" in runtime
    assert "if (loaded !== true) return;" in hub


def test_picker_preserves_background_downloads_but_switches_cached_models():
    page = _read("features/chat/chat-page.tsx")
    stage_or_load = page.split("const stageOrLoad = useCallback(", 1)[1]
    stage_or_load = stage_or_load.split("useRepoDownload({", 1)[0]
    assert "if (wantBackgroundDownload)" in stage_or_load
    assert "await cancelLoading()" not in stage_or_load
    assert "await selectModel({" in stage_or_load
    assert "const stopped = await cancelLoadRun(activeRun, true);" in _read(
        "features/chat/hooks/use-chat-model-runtime.ts"
    )


def test_external_selection_invalidates_older_local_intent():
    page = _read("features/chat/chat-page.tsx")
    external = page.split(
        'if (meta?.source === "external" || isExternalModelId(value)) {',
        1,
    )[1]
    assert external.index("invalidatePendingModelSelection()") < external.index(
        "store.setCheckpoint(value, null);"
    )
    assert external.index("await cancelLoading();") < external.index(
        "store.setCheckpoint(value, null);"
    )
    assert "isModelSelectionIntentCurrent(selectionIntentId)" in external


def test_replacement_carries_forward_an_already_unloaded_rollback_target():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "previousCheckpointWasUnloaded: replacementNeedsRollback" in runtime
    assert (
        "replacementNeedsRollback ||\n"
        "          activeRun.previousCheckpointWasUnloaded"
    ) in runtime
    assert "let previousWasUnloaded = run.previousCheckpointWasUnloaded;" in runtime
