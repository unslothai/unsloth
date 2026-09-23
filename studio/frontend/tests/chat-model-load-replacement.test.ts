// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Replacing an in-flight chat model load: picking model B while model A is still
// loading must stop A and start B, instead of refusing the pick. Asserted at the
// source level because the decision lives inside a React hook whose awaits span
// the consent dialogs, the backend unload and the load itself.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RUNTIME = fileURLToPath(
  new URL("../src/features/chat/hooks/use-chat-model-runtime.ts", import.meta.url),
);
const CHAT_PAGE = fileURLToPath(
  new URL("../src/features/chat/chat-page.tsx", import.meta.url),
);

function read(path: string): string {
  return readFileSync(path, "utf8");
}

function section(source: string, start: string, end: string): string {
  const from = source.indexOf(start);
  assert.notEqual(from, -1, `expected to find ${start}`);
  const to = source.indexOf(end, from + start.length);
  assert.notEqual(to, -1, `expected to find ${end} after ${start}`);
  return source.slice(from, to);
}

test("only the run that owns the slot may update or clear it", async () => {
  // The helper is what makes a superseded run stop touching shared loading state.
  const { ownsModelLoadRun, releaseOwnedModelLoadRun } = await import(
    "../src/features/chat/utils/model-load-run.ts"
  );
  const run = { attemptId: 7 };
  assert.equal(ownsModelLoadRun(run, { attemptId: 7 }), true);
  assert.equal(ownsModelLoadRun(run, { attemptId: 8 }), false);
  assert.equal(ownsModelLoadRun(null, { attemptId: 7 }), false);
  // A losing run must not clear the winner's slot.
  const other = { attemptId: 9, label: "newer" };
  assert.equal(releaseOwnedModelLoadRun(other, { attemptId: 7 }), other);
  assert.equal(releaseOwnedModelLoadRun(other, { attemptId: 9 }), null);
});

test("a different pick supersedes the pending load instead of being rejected", () => {
  const runtime = read(RUNTIME);

  // The old behaviour: refuse the pick and tell the user to wait.
  assert.equal(
    runtime.includes("Another model is already loading"),
    false,
    "the 'Another model is already loading' bail-out must be gone",
  );

  // The new behaviour: cancel the pending run, then keep looping so the last
  // selection wins even when several are waiting on the same cancellation.
  const loop = section(
    runtime,
    "// A different pick supersedes the load in flight.",
    "// Ask the backend, not params.checkpoint",
  );
  assert.match(loop, /while \(true\) \{/);
  assert.match(loop, /const stopped = await cancelLoadRun\(activeRun, true\);/);
  assert.match(loop, /if \(!stopped\) \{/);
  // The loop keeps waiting while a run still owns the slot, even after its picker
  // was cleared by cancellation.
  assert.match(loop, /if \(!inFlightLoad && !activeRun\) break;/);
});

test("the pending load is stopped at the backend before the replacement claims the slot", () => {
  const runtime = read(RUNTIME);
  const cancel = section(
    runtime,
    "const cancelLoadRun = useCallback(",
    "const cancelLoadingWithCheckpointPolicy = useCallback(",
  );
  // An abort signal cannot stop a load that is already POSTing, so the awaited
  // /unload is what actually interrupts it.
  assert.match(cancel, /await unloadModel\(\{/);
  assert.match(cancel, /model_path: run\.loadAttemptPath,/);
  assert.match(cancel, /const cancelPromise = \(async \(\): Promise<boolean> => \{/);
  assert.match(cancel, /return false;/);
  assert.match(cancel, /run\.cancelPromise = cancelPromise;/);
  // The slot is released only by the run that still owns it.
  assert.match(cancel, /ownsModelLoadRun\(activeLoadRunRef\.current, run\)/);
  assert.match(cancel, /releaseOwnedModelLoadRun\(/);
});

test("the matching request ID is threaded to both /load and its cancel", () => {
  const runtime = read(RUNTIME);
  const registration = section(
    runtime,
    "const loadRun: ActiveModelLoadRun = {",
    "activeLoadRunRef.current = loadRun;",
  );
  // Each run owns one stable opaque ID, minted once and never reused across runs.
  assert.match(registration, /requestId: crypto\.randomUUID\(\),/);
  // It is sent on the load itself...
  const load = section(runtime, "const loadResponse = await loadModel({", "});");
  assert.match(load, /load_request_id: loadRun\.requestId,/);
  // ...and the cancel names the same ID plus the exact path that was POSTed, so the
  // backend binds the unload to this attempt and never to a newer same-model load.
  const cancel = section(
    runtime,
    "const cancelLoadRun = useCallback(",
    "const cancelLoadingWithCheckpointPolicy = useCallback(",
  );
  assert.match(cancel, /cancel_load_request_id: run\.requestId,/);
  assert.match(cancel, /model_path: run\.loadAttemptPath,/);
  // The path is recorded at the load boundary, so a cancel only fires for a load
  // that actually reached the backend.
  assert.match(runtime, /loadRun\.loadAttemptPath = loadPath;/);
  // The old unscoped double-unload, which let cancellation race the load it was
  // meant to stop, must be gone.
  assert.equal(
    cancel.includes("await unloadModel({ model_path: backendLoadModelId })"),
    false,
    "the unscoped double-unload race must be gone",
  );
});

test("the winning selection reserves the slot for its own run", () => {
  const runtime = read(RUNTIME);
  assert.match(runtime, /const activeLoadRunRef = useRef<ActiveModelLoadRun \| null>\(null\)/);
  assert.match(runtime, /const loadIntentId = \+\+modelSelectionIntentEpoch;/);
  const registration = section(runtime, "const loadRun: ActiveModelLoadRun = {", "activeLoadRunRef.current = loadRun;");
  assert.match(registration, /abortController: abortCtrl,/);
  assert.match(registration, /cancelPromise: null,/);
  assert.match(runtime, /activeLoadRunRef\.current = loadRun;/);
  // A selection that lost the race must not go on to start a load.
  assert.match(runtime, /modelSelectionIntentEpoch !== loadIntentId/);
});

test("a downloaded replacement pick falls through to selectModel", () => {
  const page = read(CHAT_PAGE);
  const guard = section(
    page,
    "if (store.modelLoading) {",
    "if (wantManagerStaging) {\n        setPendingHubAutoLoad(",
  );
  // Only two early exits may remain: the duplicate click of the pick already
  // loading, and the download-manager handoff. A different, already-downloaded
  // pick must fall through the guard to the selectModel call below it.
  const returns = guard.match(/\breturn;/g) ?? [];
  assert.equal(
    returns.length,
    2,
    "the mid-load guard may only return for the duplicate click and the handoff",
  );
  assert.match(guard, /The duplicate click is the only pick this guard refuses/);
  assert.match(page, /await selectModel\(\{/);
  // The guard must end without a trailing bail-out: the old unconditional return
  // used to sit right before its closing brace and swallowed every replacement pick.
  assert.equal(
    /return;\s*\}\s*$/.test(guard),
    false,
    "the mid-load guard must not bail out before selectModel",
  );
});

test("a superseded run cannot clear a newer run's loading state", () => {
  const runtime = read(RUNTIME);
  // Both terminal paths must go through the ownership-checked helper, so a cancelled
  // load settling late cannot clear the replacement's refs or release its lease.
  assert.match(runtime, /resetLoadingUiForRun\(loadRun\);/);
  assert.equal(
    /\n\s{10}resetLoadingUi\(\);/.test(runtime),
    false,
    "no terminal cleanup may call the unowned resetLoadingUi() directly",
  );
  const helper = section(
    runtime,
    "const resetLoadingUiForRun = useCallback(",
    "const renderLoadDescription = useCallback(",
  );
  assert.match(helper, /ownsModelLoadRun\(activeLoadRunRef\.current, run\)/);
  // Cancellation owns the slot until its /unload settles.
  assert.match(helper, /if \(run\.cancelPromise\) return;/);
});

test("rollback state is inherited from the run being replaced", () => {
  const runtime = read(RUNTIME);
  const inherit = section(
    runtime,
    "const inheritCancelledRunRollback = (",
    "// A different pick supersedes the load in flight.",
  );
  // The cancelled run's own rollback target is adopted, not just its pending marker.
  assert.match(inherit, /cancelledRun: ActiveModelLoadRun/);
  assert.match(inherit, /cancelledRun\.rollbackConfig/);
  assert.match(inherit, /pendingReplacementRollback = \{/);
  assert.match(inherit, /checkpoint: cancelledRun\.rollbackCheckpoint/);
  // Called with the run, so the target is populated rather than only read.
  assert.match(runtime, /inheritCancelledRunRollback\(activeRun\);/);
  // The replacement's own rollback checkpoint comes from the inherited target.
  assert.match(runtime, /inheritedPendingRollback\?\.config/);
  assert.match(
    runtime,
    /previousCheckpoint = inheritedPendingRollback\n\s*\? inheritedPendingRollback\.checkpoint/,
  );
});

test("cacheRam stays in the load tuning snapshot", () => {
  const runtime = read(RUNTIME);
  const tuning = section(
    runtime,
    "let loadServerTuning: ServerTuningValues = {",
    "try {",
  );
  // Dropping it here silently omits cache_ram from /load whenever per-model settings
  // are not being reset, so a configured value is never sent or committed.
  assert.match(
    tuning,
    /cacheRam: pendingLoadConfig\?\.cacheRam \?\? stateBeforeUnload\.cacheRam,/,
  );
});

test("the picker no longer refuses a different model mid-load", () => {
  const page = read(CHAT_PAGE);
  const guard = section(
    page,
    "if (store.modelLoading) {",
    "if (wantManagerStaging) {\n        setPendingHubAutoLoad(",
  );
  assert.equal(
    guard.includes("Another model is already loading"),
    false,
    "the mid-load rejection toast must be gone from the picker",
  );
  // Only the same-pick duplicate click and the download-manager handoff return early.
  assert.match(guard, /This model is already loading/);
  assert.match(guard, /return;/);
});


test("a pick arriving mid-cancel waits for the run that still holds the slot", () => {
  const runtime = read(RUNTIME);
  const loop = section(
    runtime,
    "// A different pick supersedes the load in flight.",
    "// A local pick that is superseded by a later selection must not keep the slot.",
  );
  // Cancellation clears the picker before its unload settles while the run keeps the
  // slot and the lifecycle lease, so the loop must not break on the picker alone.
  assert.match(loop, /if \(!inFlightLoad && !activeRun\) break;/);
  assert.equal(
    /if \(!inFlightLoad\) break;/.test(loop),
    false,
    "the loop must not break while a cancelling run still owns the slot",
  );
  assert.match(loop, /if \(inFlightLoad\) \{/);
  // The run it waits for still has to be the one the pick asked to cancel.
  assert.match(loop, /const stopped = await cancelLoadRun\(activeRun, true\);/);
});

test("cancelling during the preliminary unload reconciles the removed resident model", () => {
  const runtime = read(RUNTIME);
  // Only a real preliminary /unload removes the resident model; the forced path leaves
  // it to /load, so it must not be recorded as gone.
  assert.match(runtime, /residentModelUnloaded: boolean;/);
  // A fresh run starts unloaded only when it inherits that fact from the run it replaces;
  // its own preliminary unload flips the flag below.
  assert.match(runtime, /residentModelUnloaded: inheritedPendingRollback\?\.residentUnloaded === true,/);
  const pre = section(
    runtime,
    "if (!forceCancelActive) {",
    "// Set either way: /load can still leave no model resident",
  );
  const unload = pre.indexOf("await unloadModel({ model_path: currentCheckpoint });");
  const mark = pre.indexOf("loadRun.residentModelUnloaded = true;");
  assert.notEqual(unload, -1, "expected the preliminary unload");
  assert.notEqual(mark, -1, "the preliminary unload must record the removal");
  assert.ok(mark > unload, "the flag must be set after the unload, not before");
  // A run that stopped before POSTing its own /load leaves nothing registered, so the
  // store's checkpoint has to be reconciled rather than preserved.
  const cancel = section(
    runtime,
    "const cancelLoadRun = useCallback(",
    "const cancelLoadingWithCheckpointPolicy = useCallback(",
  );
  assert.match(
    cancel,
    /if \([\s\S]*?!run\.loadAttemptPath && run\.residentModelUnloaded[\s\S]*?clearCheckpoint\(\);[\s\S]*?await refresh\(\);/,
    "an aborted pre-load cancellation must reconcile the removed resident model",
  );
  // The reconciliation has to happen before the slot is handed over.
  const reconcile = cancel.indexOf("run.residentModelUnloaded");
  const release = cancel.indexOf("activeLoadRunRef.current = releaseOwnedModelLoadRun(");
  assert.notEqual(reconcile, -1, "expected the reconciliation");
  assert.notEqual(release, -1, "expected the slot release");
  assert.ok(reconcile < release, "the slot must be released only after reconciling");
});

test("a cancelled preflight keeps the slot until its coroutine unwinds", () => {
  const runtime = read(RUNTIME);
  // The old run may be parked inside an unabortable preflight, so the slot is released
  // only once the coroutine that owns it has actually unwound.
  assert.match(runtime, /settledPromise: Promise<void>;/);
  assert.match(runtime, /markLoadRunSettled = resolve;/);
  assert.match(runtime, /settledPromise: loadRunSettled,/);
  assert.match(runtime, /markSettled: markLoadRunSettled,/);
  assert.match(runtime, /\} finally \{\n\s*\/\/ Last act of this run's coroutine[\s\S]*?markLoadRunSettled\(\);/);
  const cancel = section(
    runtime,
    "const cancelLoadRun = useCallback(",
    "const cancelLoadingWithCheckpointPolicy = useCallback(",
  );
  assert.match(
    cancel,
    /if \(ownsModelLoadRun\(activeLoadRunRef\.current, run\)\) \{[\s\S]*?await run\.settledPromise;/,
    "the slot must be held until the cancelled run settles",
  );
});

test("the cancelled coroutine stops applying config after its preflight awaits", () => {
  const runtime = read(RUNTIME);
  const ABORT_CHECK = 'if (abortCtrl.signal.aborted) throw new Error("Cancelled");';
  // Staged metadata gates the config pre-apply, so the abort check has to sit between them.
  const staged = section(
    runtime,
    "if (isGguf && isDiffusion === undefined) {",
    "const targetIsDiffusion = isDiffusion === true;",
  );
  const stagedAwait = staged.indexOf("await fetchGgufStagedMetadata(");
  const stagedAbort = staged.indexOf(ABORT_CHECK);
  assert.notEqual(stagedAwait, -1, "expected the staged-metadata await");
  assert.notEqual(stagedAbort, -1, "staged metadata must be followed by an abort check");
  assert.ok(stagedAbort > stagedAwait, "the abort check must follow the await");
  // validateModel is the other unabortable preflight that precedes shared-state writes.
  const validation = section(
    runtime,
    "const validation = await validateModel({",
    "if (validation.mlx_loads_base_model) {",
  );
  assert.ok(
    validation.includes(ABORT_CHECK),
    "validateModel must be followed by an abort check before it writes shared state",
  );
});

test("a superseded preflight yields instead of starting the stale load", () => {
  const runtime = read(RUNTIME);
  // A newer pick cannot take the lifecycle lease while this preflight holds it, so it
  // leaves no picker entry behind: the epoch, not only the picker, has to be re-checked.
  const recheck = section(
    runtime,
    "// Re-check the tracked picker for a load that was already starting",
    "const forceCancelActive = stopDecision.forceCancelActive;",
  );
  assert.match(
    recheck,
    /if \(rivalLoadStarted\(\) \|\| modelSelectionIntentEpoch !== loadIntentId\) \{/,
  );
  assert.match(recheck, /releasePreflightLifecycleLease\(\);/);
});

test("a superseded run does not restore its config over the replacement", () => {
  const runtime = read(RUNTIME);
  const terminal = section(
    runtime,
    "await performLoad();",
    "// Last act of this run's coroutine",
  );
  assert.match(
    terminal,
    /if \(modelSelectionIntentEpoch === loadIntentId\) restorePreviousConfig\(\);/,
    "only a run that still owns the selection may roll the shared config back",
  );
});

// The four exact-head Codex findings on the previous revision: each asserts the code path
// that used to leave the store and the backend disagreeing.

test("a failed unload reconciles only after the cancelled run has settled", () => {
  const runtime = read(RUNTIME);
  const cancel = section(
    runtime,
    "const cancelLoadRun = useCallback(",
    "const cancelLoadingWithCheckpointPolicy = useCallback(",
  );
  // The unabortable /load can still make this run's own target resident after /unload failed.
  // Reading status before the run stops would let that later response survive unreconciled,
  // leaving the store naming the old checkpoint while the new model serves prompts.
  const failure = section(
    cancel,
    "// The request failed, so reconcile against the backend before releasing the",
    "return false;",
  );
  const settle = failure.indexOf("await run.settledPromise;");
  const refresh = failure.indexOf("await refresh();");
  assert.notEqual(settle, -1, "the failed-cancel path must wait for the run to settle");
  assert.notEqual(refresh, -1, "the failed-cancel path must still reconcile status");
  assert.ok(settle < refresh, "status must be read only after the run has stopped");
  // The pre-fix order refreshed against a still-moving backend.
  assert.equal(
    /toast\.error\(message, \{ description: detail \}\);\s*\/\/[^\n]*\n[\s\S]{0,200}?try \{\n\s*await refresh\(\);/.test(
      cancel,
    ),
    false,
    "the refresh must not run before the settlement await",
  );
});

test("clearing a cancelled run's checkpoint restores its rollback config first", () => {
  const runtime = read(RUNTIME);
  // clearCheckpoint remembers whatever the store holds under params.checkpoint. After the
  // preliminary unload that checkpoint still names the FORMER resident while the store already
  // holds the cancelled target's settings, so clearing first persisted the wrong model's entry.
  const helper = section(
    runtime,
    "function restoreRollbackConfigForClear(",
    "const approvedRemoteCodeFingerprints",
  );
  assert.match(helper, /applyPerModelConfigToRuntime\(run\.rollbackConfig,/);
  const cancel = section(
    runtime,
    "const cancelLoadRun = useCallback(",
    "const cancelLoadingWithCheckpointPolicy = useCallback(",
  );
  const reconcile = section(
    cancel,
    "if (\n              (!run.loadAttemptPath",
    "activeLoadRunRef.current = releaseOwnedModelLoadRun(",
  );
  const restore = reconcile.indexOf("restoreRollbackConfigForClear(run);");
  const clear = reconcile.indexOf("clearCheckpoint();");
  assert.notEqual(restore, -1, "the reconciliation must restore the rollback config");
  assert.notEqual(clear, -1, "the reconciliation must clear the checkpoint");
  assert.ok(restore < clear, "the rollback config must be restored before the clear");
});

test("an inherited rollback carries the already-unloaded state into the replacement", () => {
  const runtime = read(RUNTIME);
  // The reconciliation above clears the store checkpoint, so the replacement can no longer read
  // the unloaded fact from the store. It has to ride the inherited record, or performLoad would
  // skip its compensating reload and leave no model resident.
  assert.match(runtime, /residentUnloaded\?: boolean;/);
  const inherit = section(
    runtime,
    "const inheritCancelledRunRollback = (",
    "// A different pick supersedes the load in flight.",
  );
  assert.match(inherit, /residentUnloaded: cancelledRun\.residentModelUnloaded,/);
  const registration = section(
    runtime,
    "const loadRun: ActiveModelLoadRun = {",
    "activeLoadRunRef.current = loadRun;",
  );
  assert.match(
    registration,
    /residentModelUnloaded: inheritedPendingRollback\?\.residentUnloaded === true,/,
  );
  // The gate itself has to read the inherited fact, not only the store checkpoint.
  const gate = section(
    runtime,
    "async function performLoad\(\): Promise<void> {",
    "const pendingLoadConfig =",
  );
  assert.match(
    gate,
    /let previousWasUnloaded =\n\s*inheritedPendingRollback\?\.residentUnloaded === true;/,
  );
  // The compensating reload must still be gated on that flag with a rollback target present.
  assert.match(runtime, /if \(previousWasUnloaded && previousCheckpoint\) \{/);
});

test("a failed cancellation clears the rollback the replacement would inherit", () => {
  const runtime = read(RUNTIME);
  const loop = section(
    runtime,
    "// A different pick supersedes the load in flight.",
    "// A local pick that is superseded by a later selection must not keep the slot.",
  );
  const failed = section(loop, "if \(!stopped\) \{", "return;\n        }");
  // A failed /unload leaves an uncertain backend, so the checkpoint/config captured before it
  // no longer describes what is resident; inheriting it would restore the wrong model later.
  assert.match(
    failed,
    /pendingReplacementRollback = null;/,
    "the failed-cancellation path must drop the pending rollback marker",
  );
  const drop = failed.indexOf("pendingReplacementRollback = null;");
  const bail = failed.indexOf("if (throwOnError) throw new Error(message);");
  assert.notEqual(bail, -1, "expected the bail-out");
  assert.ok(drop < bail, "the marker must be cleared on the way out");
});

test("a pick parked on a preflight lease waits for the holder instead of being lost", () => {
  const runtime = read(RUNTIME);
  // The holder owns the lease without having published a run or a picker entry, so a pick
  // arriving now used to read null from the gate and return -- and the holder then yielded as
  // stale, so neither selection loaded. The latest intent waits and re-checks until it can claim.
  const claim = section(
    runtime,
    "// Hold the lifecycle lease through confirmation and loading.",
    "if (lifecycleLease === null) {",
  );
  assert.match(claim, /beginModelLoading\("preparing"\)/);
  assert.match(
    claim,
    /if \(modelSelectionIntentEpoch !== loadIntentId\) return;/,
    "a superseded waiter must yield rather than load",
  );
  assert.doesNotMatch(claim, /leaseWaitDeadline|PREFLIGHT_LEASE_WAIT_MS|Date\.now\(\) >=/);
  assert.match(
    claim,
    /if \(settled \|\| state\.modelLoading\) return;/,
    "the wait must end when the holder releases the lease",
  );
  // Once the lease is claimed, normal failure paths still restore the prior config.
  assert.match(runtime, /restorePreviousConfig\(\);\n\s*toast\.info\("A model is loading"/);
});

test("the cancelled run's GGUF variant survives into the replacement's rollback", () => {
  const runtime = read(RUNTIME);
  // clearCheckpoint() drops the store's activeGgufVariant, so a replacement that must reload the
  // former resident would send gguf_variant: null and restore the wrong artifact of that repo.
  const inherit = section(
    runtime,
    "const inheritCancelledRunRollback = (",
    "// A different pick supersedes the load in flight.",
  );
  assert.match(
    inherit,
    /variant: cancelledRun\.rollbackVariant \?\? null,/,
    "the cancelled run's captured variant must ride the inherited rollback",
  );
  assert.match(runtime, /rollbackVariant: string \| null;/);
  const variant = section(
    runtime,
    "const previousVariant = inheritedPendingRollback",
    "const reloadingSameModel =",
  );
  assert.match(variant, /inheritedPendingRollback\.variant \?\? null/);
  // The run has to capture it, since the store no longer holds it by then.
  const registration = section(
    runtime,
    "const loadRun: ActiveModelLoadRun = {",
    "activeLoadRunRef.current = loadRun;",
  );
  assert.match(registration, /rollbackVariant: previousVariant,/);
});

test("an unload that survived the load POST still counts as unloaded", () => {
  const runtime = read(RUNTIME);
  // The /load POST does not put the removed resident back, so the unloaded fact must not be
  // discarded merely because the run has since POSTed. Otherwise a replacement cancelled before
  // its own preliminary unload skips the compensating reload and nothing is resident.
  const inherit = section(
    runtime,
    "const inheritCancelledRunRollback = (",
    "// A different pick supersedes the load in flight.",
  );
  assert.equal(
    /loadAttemptPath === null/.test(inherit),
    false,
    "the unloaded marker must not be gated on the run not having POSTed",
  );
  assert.match(inherit, /residentUnloaded: cancelledRun\.residentModelUnloaded,/);
});

test("every rollback path reads the inherited config, not this pick's own previousConfig", () => {
  const runtime = read(RUNTIME);
  // selection.previousConfig belongs to the pick's own predecessor; when this pick superseded a
  // load that had already pre-applied its settings, that field holds the superseded TARGET's
  // transient config, so a rollback through it restores the resident wearing the wrong settings.
  const body = section(
    runtime,
    "// Every rollback read below uses the INHERITED target, never this pick's own",
    "if (isGguf && isDiffusion === undefined)",
  );
  // Comment lines name the field to explain the rule, so only code lines are judged here.
  const bodyCode = body
    .split("\n")
    .filter((line) => !line.trimStart().startsWith("//"))
    .join("\n");
  assert.equal(
    /selection\.previousConfig/.test(bodyCode),
    false,
    "the rollback reads must not consult this pick's own previousConfig",
  );
  assert.match(body, /const rollbackConfig = previousConfigForReplacement;/);
  // The decline path and the maxSeqLength snapshot are rollback reads too.
  const decline = section(
    runtime,
    "if (!stopDecision.proceed) {",
    "// Re-check the tracked picker for a load that was already starting",
  );
  assert.match(decline, /restorePreviousConfig\(\);/);
  assert.match(
    runtime,
    /previousConfigForReplacement\?\.maxSeqLength \?\? maxSeqLength;/,
  );
});

test("an external pick cancels the local load it replaces", () => {
  const page = read(CHAT_PAGE);
  const external = section(
    page,
    'if (isExternalSelection) {',
    'const selectedExternal = parseExternalModelId(value);',
  );
  // Without this the local run keeps modelLoading true -- the composer then treats even the
  // external checkpoint as unavailable -- and its completion overwrites the capability fields
  // this branch sets. The intent is invalidated BEFORE the cancellation so a run still parked in
  // its preflight yields on wakeup instead of adopting its status and starting anyway.
  assert.match(external, /if \(isActiveModelLoad\) \{/);
  const invalidate = external.indexOf("invalidatePendingModelSelection()");
  const cancel = external.indexOf("cancelLoadingForReplacement(externalIntentId)");
  assert.notEqual(invalidate, -1, "expected the intent invalidation");
  assert.notEqual(cancel, -1, "expected the replacement cancellation");
  assert.ok(invalidate < cancel, "the intent must be invalidated before cancelling");
  // A stopped run's own reconciliation may clear the checkpoint, so the pick is re-asserted;
  // and the stale-intent guard keeps a superseded pick from writing to the store.
  assert.match(external, /if \(!isModelSelectionIntentCurrent\(externalIntentId\)\) return;/);
  assert.match(external, /live\.setCheckpoint\(value, null\);/);
  assert.match(external, /restoreConfigForExternalReplacement\(externalIntentId\);/);
  assert.match(external, /discardExternalReplacement\(externalIntentId\);/);
  assert.match(page, /isModelSelectionIntentCurrent,/);
});


test("Hub credential cancellation cannot strand an inherited unloaded resident", () => {
  const runtime = read(RUNTIME);
  const selection = section(
    runtime,
    "const loadIntentId = ++modelSelectionIntentEpoch;",
    "if (!stopped) {",
  );
  const prepare = selection.indexOf("await prepareHfTokenForUse(");
  const cancel = selection.indexOf("const stopped = await cancelLoadRun(activeRun, true);");
  assert.notEqual(prepare, -1, "Hub credentials are prepared before replacing a run");
  assert.notEqual(cancel, -1, "the replacement still cancels the prior run");
  assert.ok(prepare < cancel, "a declined credential prompt must leave the current run untouched");
  assert.match(selection, /if \(!preparedToken\.proceed\) \{[\s\S]*?return;/);
});

test("inherited rollback preserves the resident's pin and native-path lease", () => {
  const runtime = read(RUNTIME);
  const inherit = section(
    runtime,
    "const inheritCancelledRunRollback = (",
    "// A different pick supersedes the load in flight.",
  );
  assert.match(inherit, /loadId: cancelledRun\.rollbackLoadId,/);
  assert.match(inherit, /nativePathToken: cancelledRun\.rollbackNativePathToken,/);
  assert.match(inherit, /nativePathExpiresAtMs: cancelledRun\.rollbackNativePathExpiresAtMs,/);
  const registration = section(
    runtime,
    "const loadRun: ActiveModelLoadRun = {",
    "activeLoadRunRef.current = loadRun;",
  );
  assert.match(registration, /rollbackLoadId:/);
  assert.match(registration, /rollbackNativePathToken:/);
  assert.match(registration, /rollbackNativePathExpiresAtMs:/);
  const payload = section(runtime, "const previousActiveNativePathToken =", "const previousIsGguf =");
  assert.match(payload, /inheritedPendingRollback\s*\?\s*inheritedPendingRollback\.nativePathToken/);
  assert.match(payload, /inheritedPendingRollback\s*\?\s*inheritedPendingRollback\.loadId/);
  assert.match(payload, /inheritedPendingRollback\s*\?\s*inheritedPendingRollback\.nativePathExpiresAtMs/);
});

test("external picks invalidate every pending local preflight and restore full external capabilities", () => {
  const page = read(CHAT_PAGE);
  const external = section(
    page,
    'if (isExternalSelection) {',
    "const selectedExternal = parseExternalModelId(value);",
  );
  const invalidate = external.indexOf("invalidatePendingModelSelection()");
  const conditionalCancel = external.indexOf("if (isActiveModelLoad)", invalidate);
  assert.notEqual(invalidate, -1);
  assert.ok(invalidate < conditionalCancel, "invalidate even before a run/loading flag exists");
  assert.notEqual(invalidate, -1);
  assert.match(page, /externalCapabilityPatch = \{/);
  assert.match(page, /useChatRuntimeStore\.setState\(externalCapabilityPatch\);/);
  assert.match(external, /if \(externalCapabilityPatch\) \{[\s\S]*?setState\(externalCapabilityPatch\)/);
});

test("latest model pick remains queued until the preflight lifecycle lease is released", () => {
  const runtime = read(RUNTIME);
  const wait = section(runtime, "// Hold the lifecycle lease through confirmation", "loadLifecycleLeaseRef.current = lifecycleLease;");
  assert.match(wait, /while \(lifecycleLease === null\)/);
  assert.match(wait, /PREFLIGHT_LEASE_RETRY_MS/);
  assert.doesNotMatch(wait, /leaseWaitDeadline|PREFLIGHT_LEASE_WAIT_MS|Date\.now\(\) >=/);
});

test("inherited rollback carries loaded launch settings into compensating reload", () => {
  const runtime = read(RUNTIME);
  assert.match(runtime, /rollbackLoadedState: inheritedPendingRollback\?\.loadedState \?\? currentRollbackState/);
  assert.match(runtime, /loadedState: cancelledRun\.rollbackLoadedState/);
  const rollback = section(runtime, "const rollbackResponse = await loadModel({", "await refresh();");
  assert.match(rollback, /rollbackState\.loadedGpuMemoryMode/);
  assert.match(rollback, /rollbackState\.loadedSpeculativeType/);
  assert.match(rollback, /rollbackState\.loadedGpuLayers/);
});

test("a Transformers upgrade unload is recorded on the active run for cancellation reconciliation", () => {
  const runtime = read(RUNTIME);
  const upgrade = section(
    runtime,
    "if (validation.requires_transformers_upgrade) {",
    "if (!upgraded) {",
  );
  const consumedUnload = upgrade.indexOf(".consumeServerUnloadedChat()");
  const runMarker = upgrade.indexOf("loadRun.residentModelUnloaded = true;");
  const priorFlag = upgrade.indexOf("previousWasUnloaded = true;");
  assert.notEqual(consumedUnload, -1, "the upgrade installer reports when it unloaded the resident");
  assert.notEqual(runMarker, -1, "cancellation reconciliation needs the run-level unloaded marker");
  assert.ok(consumedUnload < runMarker && runMarker < priorFlag);
});

test("declined Hub credentials restore a failed superseded run's resident config", () => {
  const runtime = read(RUNTIME);
  const selection = section(
    runtime,
    "const loadIntentId = ++modelSelectionIntentEpoch;",
    "if (!stopped) {",
  );
  const token = selection.indexOf("const activeRunBeforeCredentials = activeLoadRunRef.current;");
  const prompt = selection.indexOf("await prepareHfTokenForUse(hfToken)");
  assert.notEqual(token, -1, "capture the in-flight run before opening credentials");
  assert.ok(token < prompt, "capture the run before awaiting the credential dialog");
  const decline = section(selection, "if (!preparedToken.proceed) {", "hfToken = preparedToken.token;");
  assert.match(decline, /activeRunBeforeCredentials\.settledPromise\.then\(/);
  assert.match(decline, /modelSelectionIntentEpoch !== loadIntentId/);
  assert.match(decline, /!activeRunBeforeCredentials\.residentModelUnloaded/);
  assert.match(decline, /current\.params\.checkpoint ===\s*activeRunBeforeCredentials\.rollbackCheckpoint/);
  assert.match(decline, /restoreRollbackConfigForClear\(activeRunBeforeCredentials\)/);
});

test("reselecting the resident external model invalidates a pending local preflight", () => {
  const page = read(CHAT_PAGE);
  const selection = section(
    page,
    "const isExternalSelection =",
    "if (isExternalSelection) {",
  );
  assert.match(selection, /const isExternalSelection =/);
  assert.match(selection, /if \(isSameLoadedModel && !meta\?\.forceReload\) \{/);
  assert.match(selection, /if \(!isExternalSelection\) return;/);
  assert.match(
    selection,
    /if \(!isActiveModelLoad\) \{\s*invalidatePendingModelSelection\(\);\s*return;/,
    "same-model external picks must invalidate a preflight even before loading flags appear",
  );
});


test("status adoption consumes the inherited rollback after hydrating the resident model", () => {
  const runtime = read(RUNTIME);
  const adoption = section(
    runtime,
    "if (confirmedStatus && adoptable(confirmedStatus)) {",
    "// Hold the lifecycle lease through confirmation",
  );
  const hydrated = adoption.indexOf("syncModelCapabilities(modelId, confirmedStatus);");
  const cleared = adoption.indexOf("pendingReplacementRollback = null;");
  const completed = adoption.indexOf("void refreshContextUsage({ afterModelLoad: true });");
  assert.notEqual(hydrated, -1, "the resident status must be applied before consuming rollback");
  assert.notEqual(cleared, -1, "status adoption must consume the inherited rollback marker");
  assert.ok(hydrated < cleared && cleared < completed);
});


test("successful external replacement discards inherited rollback", () => {
  const runtime = read(RUNTIME);
  const discard = section(
    runtime,
    "const discardExternalReplacement =",
    "const restoreConfigForExternalReplacement =",
  );
  assert.match(discard, /pendingExternalReplacement = null;\s*pendingReplacementRollback = null;/);
});

test("forced cancellation reconciles resident status before preserving rollback", () => {
  const runtime = read(RUNTIME);
  const cancel = section(runtime, "await run.settledPromise;", "activeLoadRunRef.current = releaseOwnedModelLoadRun(");
  assert.match(cancel, /run\.forceCancelActive/);
  assert.match(cancel, /await getInferenceStatus\(\)/);
  assert.match(cancel, /residentModelMatchesPick\(status/);
  assert.match(cancel, /status\.loading\?\.length/);
});

test("approved Hub credentials inherit a settled run rollback before replacement", () => {
  const runtime = read(RUNTIME);
  const selection = section(
    runtime,
    "const loadIntentId = ++modelSelectionIntentEpoch;",
    "if (!stopped) {",
  );
  const prompt = selection.indexOf("await prepareHfTokenForUse(hfToken)");
  const adoption = selection.indexOf("activeLoadRunRef.current !== activeRunBeforeCredentials");
  const inherit = selection.indexOf("previousConfigForReplacement = activeRunBeforeCredentials.rollbackConfig;");
  const restore = selection.indexOf("restoreRollbackConfigForClear(activeRunBeforeCredentials);", adoption);
  assert.ok(prompt < adoption && adoption < inherit && inherit < restore);
  assert.match(selection, /!activeRunBeforeCredentials\.loadAttemptPath/);
  assert.match(selection, /current\.params\.checkpoint ===\s*activeRunBeforeCredentials\.rollbackCheckpoint/);
});
