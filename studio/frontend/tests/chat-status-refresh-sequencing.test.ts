// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A media load announces itself twice, so two status refreshes race; the last issued must win,
// or an older answer re-pins a model that was just released.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const SOURCE = readSrc("features/chat/hooks/use-chat-model-runtime.ts");

const SYNC = SOURCE.slice(
  SOURCE.indexOf("async function syncInferenceStatusToStore("),
  SOURCE.indexOf("/**\n * Reconcile the UI after the SERVER unloaded"),
);

const LORA_REQUEST = SYNC.slice(
  SYNC.indexOf("listLoras().then("),
  SYNC.indexOf("options?.preserveIdleUnloaded"),
);

test("every refresh takes a generation, and the newest one wins", () => {
  assert.match(SOURCE, /let syncGeneration = 0;/);
  assert.match(SOURCE, /let loraSyncGeneration = 0;/);
  assert.match(SYNC, /const generation = \+\+syncGeneration;/);
  assert.match(SYNC, /const superseded = \(\) => generation !== syncGeneration;/);
});

test("a superseded refresh writes no stale model or status state", () => {
  const guard = SYNC.slice(0, SYNC.indexOf("setModels("));
  assert.match(
    guard,
    /if \(signal\?\.aborted \|\| superseded\(\)\) return;/,
    "the check must sit before model and status writes",
  );
});

test("a superseded refresh does not report its failure either", () => {
  const catchBlock = SYNC.slice(SYNC.indexOf("} catch (error) {"));
  assert.match(catchBlock, /if \(signal\?\.aborted \|\| superseded\(\)\) return;/);
  assert.ok(
    catchBlock.indexOf("superseded()") < catchBlock.indexOf("toast.error"),
    "the guard must precede the error toast",
  );
});

test("the lora inventory settles from its own request, not from a sibling's", () => {
  assert.match(
    SYNC,
    /const loraGeneration = includeLoras \? \+\+loraSyncGeneration : null;/,
  );
  // Both outcomes hang off listLoras() itself so a sibling rejection cannot discard a good list.
  assert.match(LORA_REQUEST, /setLoras\(lorasRes\.loras\.map\(toLoraSummary\)\)/);
  assert.match(LORA_REQUEST, /loraInventorySettled: true/);
  assert.match(LORA_REQUEST, /!loraSuperseded\(\)/);
  const catchBlock = SYNC.slice(SYNC.indexOf("} catch (error) {"));
  assert.doesNotMatch(catchBlock, /loraInventorySettled|setLoras/);
});

test("the eviction branch is behind the same guard", () => {
  const evictionAt = SYNC.indexOf("residentCheckpoint: null,");
  const guardAt = SYNC.indexOf("superseded()");
  assert.ok(guardAt !== -1 && guardAt < evictionAt);
});

test("residency is deferred only while a settlement wait is outstanding", () => {
  assert.doesNotMatch(
    SYNC,
    /if \(statusLoading\) return;/,
    "the lifecycle bus owns settlement itself; generic hydration must still publish residency",
  );
  assert.match(SYNC, /if \(statusLoading && serverModelWaitOutstanding\(\)\) return;/);
  const handoff = SOURCE.slice(
    SOURCE.indexOf("async function refreshAndWaitForServerModel("),
    SOURCE.indexOf("* Reconcile the UI after the SERVER unloaded"),
  );
  assert.ok(
    handoff.indexOf("beginServerModelWait(signal)") <
      handoff.indexOf("await syncInferenceStatusToStore(options);"),
  );
  assert.match(handoff, /await waitForServerModel\(signal\);/);
  assert.match(handoff, /\} finally \{\s*release\(\);\s*\}/);
});

test("a completed UI load still publishes residency while holding its lease", () => {
  const selectionGuard = SYNC.slice(
    SYNC.indexOf("const selectionChanged"),
    SYNC.indexOf("const chatActiveModel"),
  );
  assert.match(selectionGuard, /selectedCheckpoint !== selectedAtStart/);
  assert.doesNotMatch(
    selectionGuard,
    /modelLoading/,
    "the load lease must not suppress the successful load's own settled refresh",
  );

  const activeBranch = SYNC.slice(
    SYNC.indexOf("if (\n      chatActiveModel"),
    SYNC.indexOf("} else if (", SYNC.indexOf("if (\n      chatActiveModel")),
  );
  assert.match(activeBranch, /applyActiveModelStatusToStore\(statusRes,/);
});

test("the mount observer adopts only a settled model", () => {
  const wait = SOURCE.slice(
    SOURCE.indexOf("async function waitForServerModel("),
    SOURCE.indexOf("function parseTrailingEpoch("),
  );
  assert.match(
    wait,
    /if \(!loading && status\.active_model\) \{\s*await tryAdoptServerActiveModel\(\{ status \}\);/,
  );
  assert.match(
    wait,
    /!useChatRuntimeStore\.getState\(\)\.params\.checkpoint &&\s*!useChatRuntimeStore\.getState\(\)\.modelLoading/,
  );
  assert.match(wait, /const poll = statusPollSignal\(signal\);/);
  assert.match(wait, /await getInferenceStatus\(poll\.signal\)/);
  assert.match(wait, /\} finally \{\s*poll\.dispose\(\);\s*\}/);
});

test("normal startup status adoption owns the resident model's globals", () => {
  assert.match(
    SYNC,
    /adoptingExistingServerModel: selectedCheckpoint === ""/,
  );
});
