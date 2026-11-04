// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const overrides = await import("../src/features/model-picker/api/model-overrides.ts");

test("every saved-settings write tells its listeners once it settles", async () => {
  let heard = 0;
  const off = overrides.onModelOverrideWrite(() => heard++);
  // The test stub's authFetch rejects: a failed write still re-reads, it may have landed.
  await overrides.putModelOverride("org/model", null, null).catch(() => {});
  assert.equal(heard, 1);
  off();
  await overrides.putModelOverride("org/model", null, null).catch(() => {});
  assert.equal(heard, 1);
});

test("Flows drops its cached load overrides on any write, not only its own", () => {
  const src = readSrc("features/flows/editor/llm-settings-panel.tsx");
  assert.match(src, /^onModelOverrideWrite\(overridesChanged\);$/m);
  assert.doesNotMatch(src, /setTimeout\(overridesChanged/);
});
