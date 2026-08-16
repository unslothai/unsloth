// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The two reload conditions the resident shortcut cannot see in a load intent.
 *
 * `adopt_load_intent_if_matched` (llama_cpp.py) returns False when
 * `memory_state_satisfies_settings` fails or `_vram_fraction_launched` differs from the
 * active fraction, both of them server-wide. A Model Memory or VRAM budget save between two
 * picks of one model leaves the pick, its config and the status identical, so without these
 * the setting would be advertised as applying on the next load and then never reach the
 * child.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { serverWideReloadRequired } = await import(
  "../src/features/chat/lib/server-wide-reload.ts"
);

const NO = { reloadRequired: false };
const YES = { reloadRequired: true };

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

test("an answer that could not be had is not a reload", () => {
  // The endpoints ship with the backend that carries the predicates, so a null here is a
  // failed read or an older install. Declining on it would turn one flaky GET into the
  // stop-chats prompt this PR removes.
  assert.equal(
    serverWideReloadRequired({ modelMemory: null, vramBudget: null }),
    false,
  );
  assert.equal(
    serverWideReloadRequired({ modelMemory: null, vramBudget: YES }),
    true,
  );
});
