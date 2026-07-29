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
    assert "if (run.cleanup) await run.cleanup;" in catch
    assert "const status = await getInferenceStatus();" in catch
    assert catch.index("if (run.cleanup) await run.cleanup;") < catch.index(
        "const status = await getInferenceStatus();"
    )
    assert "!run.cleanup" not in catch
    assert "(!originIsExternal && run.cleanup)" not in catch


def test_compare_layout_waits_for_inventory_then_freezes():
    page = _read("chat-page.tsx")
    compare = page.split("const CompareContent = memo(", 1)[1]
    compare = compare.split("return isLoraCompare ?", 1)[0]
    assert "state.modelRuntimeHydrated" in compare
    assert "state.modelsError" in compare
    assert "usedInventoryFallbackRef" in compare
    assert "layoutCheckpointRef" in compare
    assert "layoutCheckpointCapturedRef" in compare
    assert "handleCompareActiveChange" in compare
    assert "if (compareActive) return;" in compare
    assert "if (!modelsError || isLoraCompare !== null) return;" in compare
    assert "setIsLoraCompare(false);" in compare
    assert "setIsLoraCompare(detected);" in compare
    assert "const [isLoraCompare, setIsLoraCompare]" in compare
    assert "getIsLoraCompareFromState(" in compare
    assert "useChatRuntimeStore.getState()" in compare
    assert "current ??" in compare
    assert 'aria-busy="true"' in compare


def test_overlapping_inventory_refreshes_restore_hydration_as_a_group():
    runtime = _read("hooks/use-chat-model-runtime.ts")
    sync = runtime.split("async function syncInferenceStatusToStore(", 1)[1]
    sync = sync.split("export async function resyncInferenceStatus", 1)[0]
    assert "beginFullInventoryRefresh();" in sync
    assert "finishFullInventoryRefresh(true);" in sync
    assert sync.count("finishFullInventoryRefresh(false);") == 2
    assert "fullInventoryRefreshSucceeded || fullInventoryHydrationBaseline" in runtime


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


def test_initial_thread_lookup_blocks_submission_until_ids_are_applied():
    page = _read("chat-page.tsx")
    initial_lookup = page.split(
        "// Resolve the persisted pair independently of submission state.", 1
    )[1].split("// Once the initial lookup is known", 1)[0]
    assert "applyCompareThreadIds(ids);" in initial_lookup
    assert "setInitialThreadLookupComplete(true);" in initial_lookup
    assert initial_lookup.index("applyCompareThreadIds(ids);") < initial_lookup.index(
        "setInitialThreadLookupComplete(true);"
    )
    assert "if (!isActive) return;" in initial_lookup
    assert 'toast.error("Could not restore compare conversations"' in initial_lookup
    general = page.split("const GeneralCompareContent = memo(", 1)[1]
    assert "submissionReady={initialThreadLookupComplete}" in general


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
