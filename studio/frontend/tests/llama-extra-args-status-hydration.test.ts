// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Where a tab learns what the RUNNING server was invoked with.
//
// The switch path records it, so a tab that performed the load knows. A tab opened or
// refreshed while a model was already loaded does not, and its baseline stays null.
// That baseline is what a failed switch resends: the failed target is left resident,
// so an omitted llama_extra_args cannot inherit across models (the route refuses to),
// and the previous model comes back without the arguments it was running.
//
// The fix is that /api/inference/status publishes requested_llama_extra_args and the
// applier seeds the baseline from it. Checked at the source, like the chat-template
// seed test next door: the applier is one large object literal with no seam to call.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const read = (relative: string) =>
  readFileSync(path.join(HERE, "..", relative), "utf8");

const APPLIER = read("src/features/chat/lib/apply-inference-status-to-store.ts");
const RUNTIME = read("src/features/chat/hooks/use-chat-model-runtime.ts");
const API_TYPES = read("src/features/chat/types/api.ts");

test("the status type carries the running arguments", () => {
  assert.match(API_TYPES, /requested_llama_extra_args\?: string\[\] \| null;/);
});

test("the applier seeds the loaded baseline from the status echo", () => {
  assert.match(
    APPLIER,
    /loadedLlamaExtraArgs: status\.requested_llama_extra_args \?\? null/,
  );
});

test("an older backend that omits the field changes nothing", () => {
  // undefined is "this server does not publish it", which must leave a baseline this
  // tab recorded first-hand alone rather than clearing it to null.
  assert.match(APPLIER, /status\.requested_llama_extra_args !== undefined/);
});

test("the baseline is only adopted when it is not already known", () => {
  // Otherwise an echo that lags a load in flight would overwrite what the switch
  // just recorded, and the next rollback would resend the previous model's list.
  assert.match(
    APPLIER,
    /prevState\.loadedLlamaExtraArgs === null \|\|\s*\n?\s*hydratingExistingModel \|\|\s*\n?\s*slotsModelChanged/,
  );
});

test("the rollback still resends that baseline explicitly", () => {
  assert.match(
    RUNTIME,
    /stateBeforeUnload\.loadedLlamaExtraArgs != null\s*\n?\s*\? \{ llama_extra_args: stateBeforeUnload\.loadedLlamaExtraArgs \}/,
  );
});
