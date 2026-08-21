// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { LegacyStoreGate } from "../src/features/chat/utils/legacy-store-gate.ts";

test("a store that answers is passed straight through", async () => {
  const gate = new LegacyStoreGate(50);
  assert.deepEqual(await gate.read(async () => ["thread"], []), ["thread"]);
  assert.equal(gate.responds, true);
});

test("a store that never answers yields the fallback instead of hanging", async () => {
  const gate = new LegacyStoreGate(20);
  const started = Date.now();
  const threads = await gate.read(() => new Promise<string[]>(() => {}), []);
  assert.deepEqual(threads, []);
  assert.ok(Date.now() - started < 1_000, "the read must not wait indefinitely");
  assert.equal(gate.responds, false);
});

test("a store that rejects yields the fallback", async () => {
  const gate = new LegacyStoreGate(50);
  const threads = await gate.read(async () => {
    throw new Error("VersionError");
  }, []);
  assert.deepEqual(threads, []);
  assert.equal(gate.responds, false);
});

test("a refusal latches, so later reads cost nothing", async () => {
  const gate = new LegacyStoreGate(20);
  await gate.read(() => new Promise<string[]>(() => {}), []);
  let called = false;
  const started = Date.now();
  const threads = await gate.read(async () => {
    called = true;
    return ["thread"];
  }, []);
  assert.equal(called, false, "the store must not be consulted again");
  assert.deepEqual(threads, []);
  assert.ok(Date.now() - started < 20, "a latched gate must return immediately");
});

test("a slow but healthy store keeps its answer", async () => {
  const gate = new LegacyStoreGate(200);
  const threads = await gate.read(
    () => new Promise<string[]>((resolve) => setTimeout(() => resolve(["late"]), 20)),
    [],
  );
  assert.deepEqual(threads, ["late"]);
  assert.equal(gate.responds, true);
});
