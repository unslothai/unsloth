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
    new URL(
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      import.meta.url,
    ),
  ),
  "utf8",
);
const ADAPTER = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  ),
  "utf8",
);

const SYNC = SOURCE.slice(
  SOURCE.indexOf("async function syncInferenceStatusToStore("),
  SOURCE.indexOf("/**\n * Reconcile the UI after the SERVER unloaded"),
);

test("every refresh takes a generation, and the newest one wins", () => {
  assert.match(SOURCE, /let syncGeneration = 0;/);
  assert.match(SYNC, /const generation = \+\+syncGeneration;/);
  assert.match(
    SYNC,
    /const superseded = \(\) => generation !== syncGeneration;/,
  );
});

test("a superseded refresh writes nothing back to the store", () => {
  const guard = SYNC.slice(0, SYNC.indexOf("setModels("));
  assert.match(
    guard,
    /if \(aborted\(\) \|\| superseded\(\)\) return;/,
    "the check must sit between the await and the first write",
  );
});

test("a superseded refresh does not report its failure either", () => {
  const catchBlock = SYNC.slice(SYNC.indexOf("} catch (error) {"));
  assert.match(catchBlock, /if \(aborted\(\) \|\| superseded\(\)\) return;/);
  // Otherwise a read nobody would have applied still raises a toast.
  assert.ok(
    catchBlock.indexOf("superseded()") < catchBlock.indexOf("toast.error"),
    "the guard must precede the error toast",
  );
});

test("the eviction branch is behind the same guard", () => {
  // This is the branch that clears residency and drops the pick, so a stale
  // answer reaching it is the expensive case.
  const evictionAt = SYNC.indexOf("residentCheckpoint: null,");
  const guardAt = SYNC.indexOf("superseded()");
  assert.ok(guardAt !== -1 && guardAt < evictionAt);
});

test("a superseding refresh inherits the mount poll", () => {
  assert.match(SOURCE, /let activeCliLoadPoll:/);
  assert.match(SYNC, /const inheritedPoll =/);
  assert.match(
    SYNC,
    /activeCliLoadPoll = \{ signal: pollSignal, generation \};/,
  );
  assert.match(SYNC, /const shouldPollForCliLoad =\s*pollUntilActiveModel &&/);
  assert.match(
    SYNC,
    /if \(activeCliLoadPoll\?\.generation === generation\) \{\s*activeCliLoadPoll = null;/,
  );
});

test("the mount poll waits for a replacement beside a resident model", () => {
  assert.match(
    SYNC,
    /\(!statusRes\?\.active_model \|\|\s*inferenceStatusShowsLoadInFlight\(statusRes\)\)/,
  );
});

test("known server load evidence survives an unavailable status probe", () => {
  const evidence = ADAPTER.slice(
    ADAPTER.indexOf("async function serverLoadEvidence("),
    ADAPTER.indexOf("// Slow downloads and llama-server warm-up"),
  );
  const adoption = ADAPTER.slice(
    ADAPTER.indexOf("async function adoptInFlightServerLoad("),
    ADAPTER.indexOf("async function autoLoadSmallestModel("),
  );
  assert.match(evidence, /Promise<boolean \| null>/);
  assert.match(evidence, /catch \{\s*return null;/);
  assert.match(adoption, /if \(evidence !== true\)/);
  assert.match(adoption, /if \(evidence === false\)/);
});
