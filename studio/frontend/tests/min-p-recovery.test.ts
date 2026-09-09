// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";
registerBundlerResolver();
const {
  createMinPRecoveryGuard,
  invalidateMinPRecoveries,
  shouldOfferMinPRecovery,
} = await import("../src/features/chat/lib/min-p-recovery.ts");
const message =
  "The min_p and logit_bias sampling parameters are not yet supported with speculative decoding";

test("recovery distinguishes omitted retained zero from explicit zero", () => {
  assert.equal(
    shouldOfferMinPRecovery(message, "vllm", {
      minP: 0,
      minPMode: "server-default",
    }),
    true,
  );
  assert.equal(
    shouldOfferMinPRecovery(message, "vllm", { minP: 0, minPMode: "custom" }),
    false,
  );
  assert.equal(
    shouldOfferMinPRecovery(message, "vllm", {
      minP: 0.01,
      minPMode: "custom",
    }),
    true,
  );
  assert.equal(
    shouldOfferMinPRecovery(message, "llama_cpp", { minP: 0.01 }),
    false,
  );
  assert.equal(
    shouldOfferMinPRecovery("min_p value is invalid", "vllm", { minP: 0.01 }),
    false,
  );
});

function fixture() {
  let context: unknown[] = [
    "thread-a",
    "model-a",
    0.01,
    "server-default",
    "connection-a",
  ];
  const listeners = new Set<() => void>();
  const guard = createMinPRecoveryGuard(
    () => context,
    (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  );
  return {
    guard,
    listeners,
    change(next: unknown[]) {
      context = next;
      for (const listener of listeners) listener();
    },
  };
}

test("a switch away and back cannot revive a stale recovery", () => {
  const f = fixture();
  assert.equal(f.guard.isValid(), true);
  f.change(["thread-b", "model-a", 0.01, "server-default", "connection-a"]);
  f.change(["thread-a", "model-a", 0.01, "server-default", "connection-a"]);
  assert.equal(f.guard.isValid(), false);
  f.guard.dispose();
  assert.equal(f.listeners.size, 0);
});

test("a new send invalidates identical-settings error actions", () => {
  const f = fixture();
  invalidateMinPRecoveries();
  assert.equal(f.guard.isValid(), false);
  f.guard.dispose();
});

test("changed settings or connections invalidate recovery and disposal is idempotent", () => {
  for (const next of [
    ["thread-a", "model-a", 0.2, "custom", "connection-a"],
    ["thread-a", "model-a", 0.01, "server-default", "connection-b"],
  ]) {
    const f = fixture();
    f.change(next);
    assert.equal(f.guard.isValid(), false);
    f.guard.dispose();
    f.guard.dispose();
    assert.equal(f.listeners.size, 0);
  }
});
