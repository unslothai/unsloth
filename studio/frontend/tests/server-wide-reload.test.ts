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
  // An older install has no such state to disagree about, so the shortcut keeps working.
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
  // Not symmetric with the case above. One extra reload costs the prompt this PR
  // removes; adopting on a read that failed right after a policy save leaves the child
  // on the old policy with nothing on screen to say so.
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
