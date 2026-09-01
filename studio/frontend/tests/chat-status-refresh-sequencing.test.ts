// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A media load announces itself twice: once before its POST, so the row appears
// with the toast, and once after, which is the instant the GPU arbiter has
// committed the eviction. Chat re-reads its status on both, so two refreshes
// are in flight within the POST's own duration -- measured at 1.8s against a
// live backend, which is far wider than the gap between the two reads.
//
// They read the status at different moments and answer in whatever order the
// network gives, so the older one landing last would re-pin the model the newer
// one had just seen released: chat would claim a model that 400s on send, until
// the load finally settled hours later. Last issued has to win.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SOURCE = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/hooks/use-chat-model-runtime.ts", import.meta.url),
  ),
  "utf8",
);

const SYNC = SOURCE.slice(
  SOURCE.indexOf("async function syncInferenceStatusToStore("),
  SOURCE.indexOf("/**\n * Reconcile the UI after the SERVER unloaded"),
);

/** The lora slot of that function's Promise.all, both handlers included. */
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
  // Otherwise a read nobody would have applied still raises a toast.
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
  // Both outcomes hang off listLoras() itself. Read out of the shared Promise.all, a
  // sibling rejection discarded a good list and still marked the inventory settled,
  // which classified a resident LoRA as a base model and pinned a new pair generalized.
  assert.match(LORA_REQUEST, /setLoras\(lorasRes\.loras\.map\(toLoraSummary\)\)/);
  assert.match(LORA_REQUEST, /loraInventorySettled: true/);
  assert.match(LORA_REQUEST, /!loraSuperseded\(\)/);
  const catchBlock = SYNC.slice(SYNC.indexOf("} catch (error) {"));
  assert.doesNotMatch(catchBlock, /loraInventorySettled|setLoras/);
});

test("the eviction branch is behind the same guard", () => {
  // This is the branch that clears residency and drops the pick, so a stale
  // answer reaching it is the expensive case.
  const evictionAt = SYNC.indexOf("residentCheckpoint: null,");
  const guardAt = SYNC.indexOf("superseded()");
  assert.ok(guardAt !== -1 && guardAt < evictionAt);
});
