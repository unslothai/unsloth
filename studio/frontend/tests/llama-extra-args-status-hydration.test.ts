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

test("the CLI adoption path hydrates settings while it owns the load lease", () => {
  assert.match(
    APPLIER,
    /const seedLoadParams = options\.seedLoadParams \?\? !prevState\.modelLoading/,
  );
  assert.match(
    APPLIER,
    /seedLoadParams: options\?\.allowWhileModelLoading/,
  );
  assert.match(
    APPLIER,
    /!status\.active_model \|\|\s*\(status\.loading\?\.length \?\? 0\) > 0 \|\|\s*isSpeechOnlyStatus\(status\)/,
  );
});

test("an older backend that omits the field changes nothing", () => {
  // undefined is "this server does not publish it", which must leave a baseline this
  // tab recorded first-hand alone rather than clearing it to null.
  assert.match(APPLIER, /status\.requested_llama_extra_args !== undefined/);
});

test("the baseline follows a same-model reload from elsewhere", () => {
  // Another tab or an API client can reload the same model and variant with other
  // arguments, or with none: a baseline pinned at the first read would resend the
  // old list from the rollback path and resurrect arguments that are not running.
  // The in-flight guard stays, since performLoad owns these values mid-switch.
  assert.match(
    APPLIER,
    /requested_llama_extra_args !== undefined &&\s*\n\s*\(status\.is_gguf \?\? true\) &&\s*\n\s*seedLoadParams/,
  );
});

test("the rollback still resends that baseline explicitly", () => {
  assert.match(
    RUNTIME,
    /stateBeforeUnload\.loadedLlamaExtraArgs != null\s*\n?\s*\? \{ llama_extra_args: stateBeforeUnload\.loadedLlamaExtraArgs \}/,
  );
});

test("an explicit empty list is kept apart from an unknown one", () => {
  // The rollback sends this field only when it has one, and omitting it is what
  // makes /load inherit: a model launched with no extras would otherwise come back
  // carrying the arguments of the load that just failed. null stays for "never told".
  assert.match(RUNTIME, /loadLlamaExtraArgs !== undefined\s*\n?\s*\? \(loadLlamaExtraArgs \?\? \[\]\)/);
  // The status echo goes in as it arrives, so a server running none reads as [].
  assert.match(
    APPLIER,
    /loadedLlamaExtraArgs: status\.requested_llama_extra_args \?\? null/,
  );
});
