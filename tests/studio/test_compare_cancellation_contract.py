# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source contracts for stable compare cancellation and layout ownership."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHAT = ROOT / "studio" / "frontend" / "src" / "features" / "chat"


def _read(name: str) -> str:
    return (CHAT / name).read_text()


def test_cleanup_reconciles_the_origin_checkpoint_before_clearing_it():
    composer = _read("shared-composer.tsx")
    catch = composer.split("} catch (err) {", 1)[1].split("} finally {", 1)[0]
    assert "await run.cleanup;" in catch
    assert "cleanupError = error;" in catch
    assert "const status = await getInferenceStatus();" in catch
    assert catch.index("await run.cleanup;") < catch.index(
        "const status = await getInferenceStatus();"
    )
    assert "!run.cleanup" not in catch
    assert "(!originIsExternal && run.cleanup)" not in catch
    cancel = composer.split("async function cancelCompareBackendLoad(", 1)[1].split(
        "function fileToBase64DataURL", 1
    )[0]
    assert "useChatRuntimeStore.getState().clearCheckpoint();" in cancel
    assert "status.loading.some(" in cancel
    assert "COMPARE_CANCEL_SETTLED_OBSERVATIONS" in cancel
    assert "throw new Error(`Could not cancel the backend model load:" in cancel


def test_compare_layout_waits_for_inventory_then_freezes():
    page = _read("chat-page.tsx")
    compare = page.split("const CompareContent = memo(", 1)[1]
    compare = compare.split("return isLoraCompare ?", 1)[0]
    assert "state.modelRuntimeHydrated" in compare
    assert "state.modelsError" in compare
    assert "onRefreshModelInventories: () => Promise<void>" in compare
    assert "const [inventoryRefreshComplete, setInventoryRefreshComplete]" in compare
    assert "void onRefreshModelInventories().finally(" in compare
    assert "COMPARE_INVENTORY_WAIT_MS" in compare
    assert "window.setTimeout(" in compare
    assert "current ?? false" in compare
    assert "const [storedThreads, setStoredThreads]" in compare
    assert "COMPARE_THREAD_LOOKUP_WAIT_MS" in compare
    assert "listStoredChatThreads({ pairId })" in compare
    assert "if (isLoraCompare !== null || !storedThreadsReady) return;" in compare
    assert 'thread.modelType === "model1"' in compare
    assert 'thread.modelType === "model2"' in compare
    assert 'thread.modelType === "base"' in compare
    assert 'thread.modelType === "lora"' in compare
    assert "if (!inventoryRefreshComplete) return;" in compare
    assert "if (modelsError)" in compare
    assert "setIsLoraCompare(false);" in compare
    assert "const [isLoraCompare, setIsLoraCompare]" in compare
    assert "getIsLoraCompareFromState(" in compare
    assert "useChatRuntimeStore.getState()" in compare
    assert 'aria-busy="true"' in compare

    generalized = page.split("const GeneralCompareContent = memo(", 1)[1]
    generalized = generalized.split("const model2LoraBase", 1)[0]
    assert "lora.id === globalCheckpoint" in generalized
    assert "lora.id === current.id" in generalized
    assert 'lora.exportType === "lora"' in generalized
    assert "return { ...current, isLora: true };" in generalized


def test_overlapping_inventory_refreshes_restore_hydration_as_a_group():
    runtime = _read("hooks/use-chat-model-runtime.ts")
    sync = runtime.split("async function syncInferenceStatusToStore(", 1)[1]
    sync = sync.split("export async function resyncInferenceStatus", 1)[0]
    assert "beginFullInventoryRefresh();" in sync
    assert "finishFullInventoryRefresh(true);" in sync
    assert sync.count("finishFullInventoryRefresh(false);") == 3
    assert "fullInventoryRefreshSucceeded || fullInventoryHydrationBaseline" in runtime
    assert "if (fullInventoryRefreshSucceeded)" in runtime
    assert "store.setModelsError(null);" in runtime
    failed = sync.split("} catch (error) {", 1)[1]
    assert failed.index("setModelsError(message);") < failed.rindex(
        "finishFullInventoryRefresh(false);"
    )


def test_same_model_reload_retains_origin_for_stop_reconciliation():
    composer = _read("shared-composer.tsx")
    assert "modelSwitchState.originCheckpoint = previousCheckpoint || null;" in composer
    assert "same-model reload" in composer


def test_compare_waiter_tracks_default_key_until_remote_id_handoff():
    composer = _read("shared-composer.tsx")
    waiter = composer.split("waitForRunEnd: () =>", 1)[1].split(
        "};\n    return () =>", 1
    )[0]
    assert '...(remoteId ? [] : ["__default"])' in waiter
    assert waiter.index("const remoteId") < waiter.index('["__default"]')
    assert "aui.thread().getState().isRunning" in waiter
    assert "aui.subscribe(check)" in waiter


def test_overlapping_compare_sends_are_rejected_before_file_conversion():
    composer = _read("shared-composer.tsx")
    send = composer.split("async function send() {", 1)[1].split(
        "async function sendImpl() {", 1
    )[0]
    assert "sendInProgressRef.current" in send
    assert "sendInProgressRef.current = true;" in send
    assert "await sendImpl();" in send
    assert "sendInProgressRef.current = false;" in send


def test_initial_thread_lookup_is_bounded_and_applied_before_mount():
    page = _read("chat-page.tsx")
    compare = page.split("const CompareContent = memo(", 1)[1].split(
        "return isLoraCompare ?", 1
    )[0]
    assert "const settle = (threads: StoredCompareThread[])" in compare
    assert "if (settled) return;" in compare
    assert "() => settle([])" in compare
    assert "COMPARE_THREAD_LOOKUP_WAIT_MS" in compare
    assert "setStoredThreads(threads);" in compare
    assert "setStoredThreadsReady(true);" in compare
    assert 'toast.error("Could not restore compare conversations"' in compare
    general = page.split("const GeneralCompareContent = memo(", 1)[1]
    assert "initialThreads.find(" in general
    assert "submissionReady={true}" in general


def test_compare_load_target_starts_at_the_request_boundary():
    api = _read("api/chat-api.ts")
    load_model = api.split("export async function loadModel(", 1)[1].split(
        "export async function validateModel(", 1
    )[0]
    assert "onRequestStart: options?.onRequestStart" in load_model
    assert "onAuthenticationRequired: options?.onAuthenticationRequired" in load_model
    assert load_model.index("const preparedToken = await prepareHfTokenForUse") < (
        load_model.index("const response = await authFetch")
    )

    auth = (
        ROOT / "studio" / "frontend" / "src" / "features" / "auth" / "api.ts"
    ).read_text()
    auth_fetch = auth.split("export async function authFetch(", 1)[1].split(
        "async function postLogout(", 1
    )[0]
    assert auth_fetch.index("lifecycle?.onRequestStart?.();") < auth_fetch.index(
        "response = await fetchWithTauriNetworkRetry"
    )
    assert auth_fetch.index("lifecycle?.onAuthenticationRequired?.();") < (
        auth_fetch.index("const refreshed = await refreshSession();")
    )

    composer = _read("shared-composer.tsx")
    assert "onRequestStart: () => {" in composer
    assert "onAuthenticationRequired: () => {" in composer
    assert "compareRunsRef.current.setLoadingModel(run, sel);" in composer
    assert "compareRunsRef.current.setLoadingModel(run, null);" in composer


def test_prompt_queue_waits_for_compare_thread_restore():
    composer = _read("shared-composer.tsx")
    queue = composer.split("onRunList={(items) => {", 1)[1].split(
        "const hasCompareHandles", 1
    )[0]
    assert "if (!submissionReady)" in queue
    assert 'toast.info("Restoring compare conversations…"' in queue
