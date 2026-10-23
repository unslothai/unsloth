// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * adopt_load_intent_if_matched also checks server-wide memory state and VRAM fraction,
 * which a load intent cannot see; without these the setting never reaches the child.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { serverWideReloadRequired } = await import(
  "../src/features/chat/lib/server-wide-reload.ts"
);

const NO = { reloadRequired: false } as const;
const YES = { reloadRequired: true } as const;

test("either signal alone declines the shortcut", () => {
  assert.equal(
    serverWideReloadRequired({ modelMemory: YES, vramBudget: NO }),
    true,
  );
  assert.equal(
    serverWideReloadRequired({ modelMemory: NO, vramBudget: YES }),
    true,
  );
  assert.equal(
    serverWideReloadRequired({ modelMemory: YES, vramBudget: YES }),
    true,
  );
});

test("a settled server adopts", () => {
  assert.equal(
    serverWideReloadRequired({ modelMemory: NO, vramBudget: NO }),
    false,
  );
});

test("a route this backend does not serve is not a reload", () => {
  assert.equal(
    serverWideReloadRequired({
      modelMemory: "unsupported",
      vramBudget: "unsupported",
    }),
    false,
  );
  assert.equal(
    serverWideReloadRequired({ modelMemory: "unsupported", vramBudget: NO }),
    false,
  );
});

test("an answer that could not be had declines the shortcut", () => {
  // Asymmetric on purpose: an extra reload is cheaper than silently keeping the old policy.
  assert.equal(
    serverWideReloadRequired({ modelMemory: "unknown", vramBudget: NO }),
    true,
  );
  assert.equal(
    serverWideReloadRequired({ modelMemory: NO, vramBudget: "unknown" }),
    true,
  );
  assert.equal(
    serverWideReloadRequired({
      modelMemory: "unknown",
      vramBudget: "unsupported",
    }),
    true,
  );
});
