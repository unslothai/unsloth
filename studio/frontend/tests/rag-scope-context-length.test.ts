// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { ragScopeContextLength } from "../src/features/chat/api/rag-context-length.ts";

const UNGUARDED_WINDOW = /context_length:\s*\n?\s*runtime\./;
const GUARDED_WINDOW = /context_length: ragScopeContextLength\(\{/g;

test("a resident GGUF's window is not reported on a hosted turn", () => {
  assert.equal(
    ragScopeContextLength({
      isExternalRequest: true,
      loadedCustomContextLength: null,
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
      loadedCustomContextLength: null,
      loadedContextLength: 4096,
      maxSeqLength: 8192,
    }),
    4096,
  );
});

test("an applied Context Length wins over the reported window", () => {
  assert.equal(
    ragScopeContextLength({
      isExternalRequest: false,
      loadedCustomContextLength: 16384,
      loadedContextLength: 4096,
      maxSeqLength: 4096,
    }),
    16384,
  );
});

test("a local load with no GGUF window falls back to maxSeqLength", () => {
  assert.equal(
    ragScopeContextLength({
      isExternalRequest: false,
      loadedCustomContextLength: null,
      loadedContextLength: null,
      maxSeqLength: 32768,
    }),
    32768,
  );
  assert.equal(
    ragScopeContextLength({
      isExternalRequest: false,
      loadedCustomContextLength: null,
      loadedContextLength: null,
      maxSeqLength: null,
    }),
    undefined,
  );
});

test("every rag_scope window goes through the guard", () => {
  const adapter = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    ),
    "utf8",
  );
  assert.doesNotMatch(adapter, UNGUARDED_WINDOW);
  assert.equal((adapter.match(GUARDED_WINDOW) ?? []).length, 2);
});

const VALUES = [null, undefined, 0, 1, 4096, 8192, 200_000] as const;

test("an external request NEVER reports a window, whatever is resident", () => {
  for (const a of VALUES) for (const b of VALUES) for (const c of VALUES) {
    assert.equal(
      ragScopeContextLength({ isExternalRequest: true,
        loadedCustomContextLength: a, loadedContextLength: b, maxSeqLength: c }),
      undefined,
      `leaked with ${a}/${b}/${c}`);
  }
});

test("a local request reports the first field that is set, in order", () => {
  for (const a of VALUES) for (const b of VALUES) for (const c of VALUES) {
    const got = ragScopeContextLength({ isExternalRequest: false,
      loadedCustomContextLength: a, loadedContextLength: b, maxSeqLength: c });
    const want = a ?? b ?? c ?? undefined;
    assert.equal(got, want, `${a}/${b}/${c}`);
  }
});

test("the result is only ever a number or undefined, never null", () => {
  for (const a of VALUES) for (const b of VALUES) {
    for (const ext of [true, false]) {
      const got = ragScopeContextLength({ isExternalRequest: ext,
        loadedCustomContextLength: a, loadedContextLength: b, maxSeqLength: null });
      assert.ok(got === undefined || typeof got === "number", `got ${got}`);
      assert.notEqual(got, null);
    }
  }
});

test("a zero window is passed through, not treated as absent", () => {
  // `0` is falsy but `??` keeps it: a model serving no window must not silently fall
  // through to a stale larger one.
  assert.equal(ragScopeContextLength({ isExternalRequest: false,
    loadedCustomContextLength: 0, loadedContextLength: 4096, maxSeqLength: 8192 }), 0);
});
