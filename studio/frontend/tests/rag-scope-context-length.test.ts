// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { ragScopeContextLength } from "../src/features/chat/api/rag-context-length.ts";

const UNGUARDED_WINDOW = /context_length:\s*\n?\s*runtime\./;
const GUARDED_WINDOW = /context_length: ragScopeContextLength\(\{/g;

const read = (path: string) =>
  readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");

test("a resident GGUF's window is not reported on a hosted turn", () => {
  assert.equal(
    ragScopeContextLength({
      isExternalRequest: true,
      loadedContextLength: 4096,
      maxSeqLength: 4096,
    }),
    undefined,
  );
});

test("a local turn reports the window the model serves", () => {
  assert.equal(
    ragScopeContextLength({
      isExternalRequest: false,
      loadedContextLength: 4096,
      maxSeqLength: 8192,
    }),
    4096,
  );
});

test("a local load with no GGUF window falls back to maxSeqLength", () => {
  assert.equal(
    ragScopeContextLength({
      isExternalRequest: false,
      loadedContextLength: null,
      maxSeqLength: 32768,
    }),
    32768,
  );
  assert.equal(
    ragScopeContextLength({
      isExternalRequest: false,
      loadedContextLength: null,
      maxSeqLength: null,
    }),
    undefined,
  );
});

test("every rag_scope window goes through the guard", () => {
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  assert.doesNotMatch(adapter, UNGUARDED_WINDOW);
  assert.equal((adapter.match(GUARDED_WINDOW) ?? []).length, 2);
});

// llama-server reduces the pinned n_ctx for a memory fit or a --parallel slot split, and
// the reduced window is what `loadedContextLength` carries. A budget sized off the pin
// would inject a document the served window cannot hold.
test("the served window is budgeted, never the n_ctx the load asked for", () => {
  const helper = read("../src/features/chat/api/rag-context-length.ts");
  assert.doesNotMatch(
    helper.slice(helper.indexOf("export function")),
    /loadedCustomContextLength/,
  );
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  for (const call of adapter.match(/ragScopeContextLength\(\{[^}]*\}\)/g) ?? []) {
    assert.doesNotMatch(call, /loadedCustomContextLength/);
  }
});

// A turn sent while a model is still loading runs later against the runtime the queue
// snapshotted. A field this budget reads but `QUEUED_SETTING_KEYS` does not carry would
// be read off whichever model happens to be visible when the turn finally executes.
test("every runtime field the budget reads survives a queued run", () => {
  const queuedKeys = new Set(
    (
      read("../src/features/chat/utils/queued-chat-run-settings.ts")
        .match(/const QUEUED_SETTING_KEYS = \[([\s\S]*?)\] as const;/)?.[1] ?? ""
    ).match(/"([^"]+)"/g)?.map((quoted) => quoted.slice(1, -1)) ?? [],
  );
  assert.ok(queuedKeys.size > 0);
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  const calls = adapter.match(/ragScopeContextLength\(\{[^}]*\}\)/g) ?? [];
  assert.equal(calls.length, 2);
  for (const call of calls) {
    for (const [, field] of call.matchAll(/runtime\.(\w+)/g)) {
      assert.ok(queuedKeys.has(field), `runtime.${field} is not snapshotted`);
    }
  }
});

const VALUES = [null, undefined, 0, 1, 4096, 8192, 200_000] as const;

test("an external request NEVER reports a window, whatever is resident", () => {
  for (const a of VALUES) for (const b of VALUES) {
    assert.equal(
      ragScopeContextLength({ isExternalRequest: true,
        loadedContextLength: a, maxSeqLength: b }),
      undefined,
      `leaked with ${a}/${b}`);
  }
});

test("a local request reports the first field that is set, in order", () => {
  for (const a of VALUES) for (const b of VALUES) {
    const got = ragScopeContextLength({ isExternalRequest: false,
      loadedContextLength: a, maxSeqLength: b });
    assert.equal(got, a ?? b ?? undefined, `${a}/${b}`);
  }
});

test("the result is only ever a number or undefined, never null", () => {
  for (const a of VALUES) {
    for (const ext of [true, false]) {
      const got = ragScopeContextLength({ isExternalRequest: ext,
        loadedContextLength: a, maxSeqLength: null });
      assert.ok(got === undefined || typeof got === "number", `got ${got}`);
      assert.notEqual(got, null);
    }
  }
});

test("a zero window is passed through, not treated as absent", () => {
  // `0` is falsy but `??` keeps it: a model serving no window must not silently fall
  // through to a stale larger one.
  assert.equal(ragScopeContextLength({ isExternalRequest: false,
    loadedContextLength: 0, maxSeqLength: 8192 }), 0);
});
