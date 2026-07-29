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
    assert "await unloadModel({ model_path: backendLoadModelId });" in runtime
    cancel = runtime.split("const cancelLoadRun = useCallback(", 1)[1]
    cancel = cancel.split("const cancelLoading = useCallback(", 1)[0]
    assert cancel.index("await unloadModel({ model_path: backendLoadModelId });") < cancel.index(
        "await run.completionPromise;"
    )
    assert "const stopped = await cancelLoadRun(activeRun, true);" in runtime
    assert "if (!stopped)" in runtime
    assert "await refresh();" in runtime
    standalone = cancel.split("if (!preserveCheckpoint) {", 1)[1].split("}", 1)[0]
    assert standalone.index("clearCheckpoint();") < standalone.index("await refresh();")
    no_active = runtime.split(
        "} else if (!statusRes.active_model && !isExternalSelectionActive) {", 1
    )[1].split("}", 1)[0]
    assert "clearCheckpoint();" not in no_active
    assert "loadedIsDiffusion: false" in no_active
    assert 'useHfTokenWarningStore.getState().resolve("cancel", run);' in cancel
    assert "useRemoteCodeConsentDialogStore.getState().resolve(false, run);" in cancel
    assert "useTransformersUpgradeDialogStore.getState().cancelPending(run);" in cancel
    assert cancel.count("await run.completionPromise;") >= 2


def test_late_callbacks_are_bound_to_their_originating_run():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "activeLoadRunRef" in runtime
    assert "if (!ownsModelLoadRun(activeLoadRunRef.current, run)) return;" in runtime
    assert runtime.count("ownsModelLoadRun(activeLoadRunRef.current, run)") >= 6
    assert runtime.count("resetLoadingUi(run)") >= 2
    assert "if (run.cancelPromise) return;" in runtime
    assert "loadAttemptRef.current === run.attemptId" in runtime
    assert "loadIntentRef.current === run.intentId" in runtime
    assert runtime.count("loadIntentRef.current !== loadIntentId") >= 2
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
    assert external.index("await cancelLoadingForReplacement();") < external.index(
        "store.setCheckpoint(value, null);"
    )
    assert external.index("clearPendingReplacementRollback();") < external.index(
        "store.setCheckpoint(value, null);"
    )
    assert "isModelSelectionIntentCurrent(selectionIntentId)" in external


def test_replacement_carries_forward_an_already_unloaded_rollback_target():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "previousCheckpointWasUnloaded: replacementNeedsRollback" in runtime
    assert runtime.count("inheritCancelledRunRollback(activeRun);") >= 2
    assert "replacementNeedsRollback = pendingReplacementRollback != null;" in runtime
    assert "let previousWasUnloaded = run.previousCheckpointWasUnloaded;" in runtime


def test_cancelled_preflight_does_not_open_late_owned_dialogs():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    remote_code = _read("features/security/hooks/use-remote-code-consent.ts")
    hf_token = _read("features/hf-auth/confirm-token.ts")
    assert runtime.count("signal: abortCtrl.signal") >= 3
    staged_token = runtime.split(
        "const preparedToken = await prepareHfTokenForUse(", 1
    )[1].split(");", 1)[0]
    assert "dialogOwner: run" in staged_token
    assert "signal: abortCtrl.signal" in staged_token
    assert "if (signal?.aborted)" in remote_code
    assert "if (options.signal?.aborted)" in hf_token
    assert hf_token.index("await validateHfToken(normalized, options.signal)") < hf_token.index(
        "if (options.signal?.aborted)"
    )
    assert remote_code.index("await getRemoteCodeScan") < remote_code.index("if (signal?.aborted)")


def test_other_runtime_surface_can_cancel_the_shared_load():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "let sharedModelLoadHandle: SharedModelLoadHandle | null = null;" in runtime
    assert "sharedModelLoadHandle = {" in runtime
    assert "return shared ? shared.cancel(preserveCheckpoint) : Promise.resolve(false);" in runtime
    assert "if (sharedModelLoadHandle?.run === run)" in runtime
    assert "const stopped = await shared.cancel(true);" in runtime
    assert runtime.count("inheritCancelledRunRollback(shared.run);") >= 2
    assert "supersedeOwnerIntent: () => void;" in runtime
    assert "shared.supersedeOwnerIntent();" in runtime
    assert "loadIntentRef.current += 1;" in runtime
    assert "!initialSharedRun?.cancelPromise" in runtime
    assert "!sharedRun?.cancelPromise" in runtime
    assert "const cancelLoading = useCallback(\n    (): Promise<boolean>" in runtime
    assert "cancelLoadingWithCheckpointPolicy(false)" in runtime
    assert "cancelLoadingWithCheckpointPolicy(true)" in runtime


def test_superseded_replacement_keeps_the_working_model_config():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "rollbackConfig?: PerModelConfig;" in runtime
    assert "rollbackConfig: previousConfig" in runtime
    assert "previousConfig = cancelledRun.rollbackConfig;" in runtime
    assert "inheritCancelledRunRollback(activeRun);" in runtime
    assert "inheritCancelledRunRollback(shared.run);" in runtime
    assert "previousConfig?.maxSeqLength ??" in runtime
    assert "rollbackState.params.maxSeqLength" in runtime
    assert "applyPerModelConfigToRuntime(previousConfig" in runtime
    assert "previousConfig.nParallel ?? null" in runtime


def test_loaded_checkpoint_noop_does_not_hide_an_inflight_replacement():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    checkpoint_noop = runtime.split("params.checkpoint === modelId", 1)[1].split(
        "restorePreviousConfig();", 1
    )[0]
    assert "!initialInFlightLoad" in checkpoint_noop
    assert "!pendingReplacementRollback" in checkpoint_noop


def test_unloaded_rollback_target_survives_pending_selection_intents():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "let pendingReplacementRollback: PendingReplacementRollback | null" in runtime
    assert "pendingReplacementRollback = {" in runtime
    assert "const inheritedPendingRollback = pendingReplacementRollback;" in runtime
    assert "pendingReplacementRollback = null;" in runtime
    assert "checkpoint: cancelledRun.rollbackCheckpoint" in runtime
    assert "state: cancelledRun.rollbackState" in runtime


def test_replacement_inherits_the_rollback_models_complete_runtime_state():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "rollbackState: ChatRuntimeStateSnapshot;" in runtime
    assert "rollbackState: rollbackStateForRun" in runtime
    assert "inheritedRollbackState = cancelledRun.rollbackState;" in runtime
    rollback = runtime.split("const rollbackResponse = await loadModel(", 1)[1]
    rollback = rollback.split("await refresh();", 1)[0]
    assert "rollbackState.activeNativePathToken" in runtime
    assert "rollbackState.loadedGpuMemoryMode" in rollback
    assert "rollbackState.loadedGpuIds" in rollback
    assert "rollbackState.loadedChatTemplateOverride" in rollback


def test_throwing_callers_learn_when_their_selection_is_superseded():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert (
        runtime.count('throw new Error("Model selection was superseded by a newer choice.");') >= 2
    )


def test_shared_loading_pick_stays_visible_until_cancel_settles():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    cancel = runtime.split("const cancelLoadRun = useCallback(", 1)[1]
    cancel = cancel.split("const cancelLoading = useCallback(", 1)[0]
    assert cancel.count("clearLoadingModelPick(pickOf(model))") == 1
    assert cancel.index("await run.completionPromise;") < cancel.index(
        "clearLoadingModelPick(pickOf(model))"
    )
    failed_unload = cancel.split("} catch (error) {", 1)[1].split("} finally {", 1)[0]
    assert "await run.completionPromise;" in failed_unload


def test_backend_load_request_settles_before_cancellation_releases_its_slot():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    api = _read("features/chat/api/chat-api.ts")
    cancel = runtime.split("const cancelLoadRun = useCallback(", 1)[1]
    cancel = cancel.split("const cancelLoadingWithCheckpointPolicy", 1)[0]
    assert cancel.count("await unloadModel(") >= 2
    assert cancel.index("await run.completionPromise;") < cancel.rindex(
        "await unloadModel("
    )
    assert runtime.count("retainRequestOnAbort: true") >= 2
    assert "signal: options?.retainRequestOnAbort ? undefined : options?.signal" in api


def test_active_model_reload_cancellation_marks_rollback_unloaded():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    cancel = runtime.split("const cancelLoadRun = useCallback(", 1)[1]
    cancel = cancel.split("const cancelLoading = useCallback(", 1)[0]
    assert "run.rollbackCheckpoint.toLowerCase() === model.id.toLowerCase()" in cancel
    assert "run.previousCheckpointWasUnloaded = true;" in cancel


def test_successful_rollback_requires_the_replacement_to_unload_it():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    rollback = runtime.split("const rollbackResponse = await loadModel(", 1)[1]
    rollback = rollback.split("const rollbackSpeculativeType", 1)[0]
    assert "previousWasUnloaded = false;" in rollback
    assert "run.previousCheckpointWasUnloaded = false;" in rollback


def test_abort_signal_reaches_validation_and_scan_cleanup():
    api = _read("features/chat/api/chat-api.ts")
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    hf_api = _read("features/hf-auth/api.ts")
    hf_token = _read("features/hf-auth/confirm-token.ts")
    remote_api = _read("features/security/api/remote-code-api.ts")
    remote_code = _read("features/security/hooks/use-remote-code-consent.ts")
    assert api.count("signal: options?.signal") >= 2
    assert "validateHfToken(normalized, options.signal)" in hf_token
    assert "signal?: AbortSignal" in hf_api
    assert "signal," in hf_api
    assert "getRemoteCodeScan(modelName, hfToken);" in remote_code
    assert "getRemoteCodeScan(modelName, hfToken, signal)" not in remote_code
    assert "signal?: AbortSignal" in remote_api
    assert "signal," in remote_api
    assert "signal: options?.signal" in api
    staged_metadata = runtime.split("await fetchGgufStagedMetadata({", 1)[1].split(
        ").isDiffusion", 1
    )[0]
    assert "{ signal: abortCtrl.signal }" in staged_metadata
    assert "const discardScanDownloads = async (): Promise<void> =>" in remote_code
    aborted = remote_code.split("if (signal?.aborted)", 1)[1].split("// No custom code", 1)[0]
    assert "await discardScanDownloads();" in aborted
    assert "await Promise.all(toPurge.map" in remote_code


def test_cancellation_targets_an_inflight_rollback_load():
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "backendLoadModelId: string | null;" in runtime
    assert "const backendLoadModelId = run.backendLoadModelId ?? model.id;" in runtime
    assert "run.backendLoadModelId = modelId;" in runtime
    rollback = runtime.split("const rollbackResponse = await loadModel(", 1)[0]
    assert "run.backendLoadModelId = previousCheckpoint;" in rollback
