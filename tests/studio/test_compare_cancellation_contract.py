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


def test_initial_thread_lookup_does_not_replace_active_submission_ids():
    page = _read("chat-page.tsx")
    initial_lookup = page.split(
        "// Resolve the persisted pair independently of submission state.", 1
    )[1].split("// Once the initial lookup is known", 1)[0]
    assert "initialThreadLookupCompleteRef.current = true;" in initial_lookup
    assert "if (compareSubmittingRef.current) return;" in initial_lookup
    assert initial_lookup.index("if (compareSubmittingRef.current) return;") < (
        initial_lookup.index("applyCompareThreadIds(ids);")
    )
