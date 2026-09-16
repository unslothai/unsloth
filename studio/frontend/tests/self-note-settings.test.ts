// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_SELF_NOTE_ENABLED,
  DEFAULT_SELF_NOTE_RESERVE_TOKENS,
  ggufCompactionRequestFields,
  sanitizeSelfNoteEnabled,
  sanitizeSelfNoteReserveTokens,
} from "../src/features/chat/utils/auto-compaction.ts";

import { readSrc } from "./helpers/kit.ts";

test("the self-note feature defaults off", () => {
  assert.equal(DEFAULT_SELF_NOTE_ENABLED, false);
});

test("the enabled flag takes a boolean and nothing else", () => {
  assert.equal(sanitizeSelfNoteEnabled(true), true);
  assert.equal(sanitizeSelfNoteEnabled(false), false);
  assert.equal(sanitizeSelfNoteEnabled("true"), undefined);
  assert.equal(sanitizeSelfNoteEnabled(undefined), undefined);
});

test("the reserve clamps into the range the backend accepts", () => {
  // The backend is ge=64 le=4096 and 400s the whole save on one bad field, so
  // the client must never send an out-of-range value.
  assert.equal(sanitizeSelfNoteReserveTokens(8), 64);
  assert.equal(sanitizeSelfNoteReserveTokens(100000), 4096);
  assert.equal(sanitizeSelfNoteReserveTokens(512), 512);
});

test("a fractional reserve rounds to an integer", () => {
  assert.equal(sanitizeSelfNoteReserveTokens(512.7), 513);
});

test("a reserve that is not a finite number is rejected", () => {
  assert.equal(sanitizeSelfNoteReserveTokens("512"), undefined);
  assert.equal(sanitizeSelfNoteReserveTokens(Number.NaN), undefined);
  assert.equal(sanitizeSelfNoteReserveTokens(Number.POSITIVE_INFINITY), undefined);
});

test("the default reserve is inside its own range", () => {
  assert.equal(
    sanitizeSelfNoteReserveTokens(DEFAULT_SELF_NOTE_RESERVE_TOKENS),
    DEFAULT_SELF_NOTE_RESERVE_TOKENS,
  );
});

// --------------------------------------------------------------------------
// Wiring: selfNoteEnabled/selfNoteReserveTokens reaching the outbound request
// body through ggufCompactionRequestFields, the same builder the chat adapter
// calls. These exercise the request-field shape a caller (the adapter) sees,
// not selfNoteRequestFields directly -- it is not exported, by design.
// --------------------------------------------------------------------------

test("toggle on with a reserve set sends self_note_enabled true and the sanitized reserve", () => {
  const fields = ggufCompactionRequestFields({
    isGguf: true,
    autoCompactEnabled: true,
    contextPolicy: "inherit",
    compactionHeadroomRatio: 0.25,
    selfNoteEnabled: true,
    selfNoteReserveTokens: 512,
  });
  assert.equal(fields.self_note_enabled, true);
  assert.equal(fields.self_note_reserve_tokens, 512);
});

test("toggle off sends self_note_enabled false and no reserve", () => {
  const fields = ggufCompactionRequestFields({
    isGguf: true,
    autoCompactEnabled: true,
    contextPolicy: "inherit",
    compactionHeadroomRatio: 0.25,
    selfNoteEnabled: false,
    selfNoteReserveTokens: 512,
  });
  assert.equal(fields.self_note_enabled, false);
  assert.equal("self_note_reserve_tokens" in fields, false);
});

test("an out-of-range reserve is clamped before it reaches the request body", () => {
  // The backend 400s the whole save on a reserve outside ge=64/le=4096, so the
  // client must never forward a raw out-of-range value.
  const tooLow = ggufCompactionRequestFields({
    isGguf: true,
    autoCompactEnabled: true,
    contextPolicy: "inherit",
    compactionHeadroomRatio: 0.25,
    selfNoteEnabled: true,
    selfNoteReserveTokens: 1,
  });
  assert.equal(tooLow.self_note_enabled, true);
  assert.equal(tooLow.self_note_reserve_tokens, 64);

  const tooHigh = ggufCompactionRequestFields({
    isGguf: true,
    autoCompactEnabled: true,
    contextPolicy: "inherit",
    compactionHeadroomRatio: 0.25,
    selfNoteEnabled: true,
    selfNoteReserveTokens: 999999,
  });
  assert.equal(tooHigh.self_note_enabled, true);
  assert.equal(tooHigh.self_note_reserve_tokens, 4096);
});

test("selfNoteEnabled undefined omits both fields, leaving the server env default in force", () => {
  const fields = ggufCompactionRequestFields({
    isGguf: true,
    autoCompactEnabled: true,
    contextPolicy: "inherit",
    compactionHeadroomRatio: 0.25,
  });
  assert.equal("self_note_enabled" in fields, false);
  assert.equal("self_note_reserve_tokens" in fields, false);
});

// --------------------------------------------------------------------------
// Store + adapter wiring: selfNoteRequestFields is only useful once its two
// inputs are threaded from the runtime store through the adapter's request
// body, mirroring compactionHeadroomRatio's journey. Source-level checks
// here pin that wiring so it cannot silently regress back to "never called".
// --------------------------------------------------------------------------

test("the chat runtime store holds selfNoteEnabled/selfNoteReserveTokens with the documented defaults", () => {
  const store = readSrc("features/chat/stores/chat-runtime-store.ts");
  assert.match(store, /selfNoteEnabled: DEFAULT_SELF_NOTE_ENABLED/);
  assert.match(
    store,
    /selfNoteReserveTokens: DEFAULT_SELF_NOTE_RESERVE_TOKENS/,
  );
  assert.match(store, /setSelfNoteEnabled:/);
  assert.match(store, /setSelfNoteReserveTokens:/);
  // Persisted like every other scalar setting, or a reload silently reverts it.
  assert.match(store, /"selfNoteEnabled"/);
  assert.match(store, /"selfNoteReserveTokens"/);
});

test("the chat adapter passes the store's self-note fields into the shared request builder", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /ggufCompactionRequestFields\(/);
  assert.match(adapter, /selfNoteEnabled: runtime\.selfNoteEnabled/);
  assert.match(
    adapter,
    /selfNoteReserveTokens: runtime\.selfNoteReserveTokens/,
  );
});

test("the persisted chat settings type and sanitizer carry the self-note fields, like compactionHeadroomRatio", () => {
  const api = readSrc("features/chat/api/chat-settings-api.ts");
  assert.match(api, /selfNoteEnabled\?: boolean/);
  assert.match(api, /selfNoteReserveTokens\?: number/);

  const storage = readSrc("features/chat/utils/chat-settings-storage.ts");
  assert.match(storage, /sanitizeSelfNoteEnabled\(value\.selfNoteEnabled\)/);
  assert.match(
    storage,
    /sanitizeSelfNoteReserveTokens\(\s*value\.selfNoteReserveTokens,?\s*\)/,
  );
});

// Store instantiation (hydration, cross-tab mirroring, etc.) needs the
// registerStoreStubResolver + installLocalStorageFake harness that
// min-p-settings-store.test.ts uses; that harness is real, reusable test
// infra (not one-off mocking), so the defaults are also verified live below.
test("a freshly constructed store carries selfNoteEnabled=false and selfNoteReserveTokens=256", async () => {
  const { installLocalStorageFake, registerStoreStubResolver } = await import(
    "./helpers/kit.ts"
  );
  installLocalStorageFake();
  registerStoreStubResolver();
  const { useChatRuntimeStore } = await import(
    "../src/features/chat/stores/chat-runtime-store.ts"
  );
  const state = useChatRuntimeStore.getState();
  assert.equal(state.selfNoteEnabled, false);
  assert.equal(state.selfNoteReserveTokens, 256);
});
