// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

register("./bundler-resolver.mjs", import.meta.url);
const { connectionPickerValue, isOpenCodeProviderType, OPENCODE_PLANS } = await import(
  "../src/features/chat/external-providers.ts"
);

const dialog = readFileSync(
  new URL("../src/features/chat/chat-providers-dialog.tsx", import.meta.url),
  "utf8",
);

test("both OpenCode plans share one picker entry", () => {
  assert.deepEqual(
    OPENCODE_PLANS.map((plan: { providerType: string }) => plan.providerType),
    ["opencode", "opencode_go"],
  );
  assert.deepEqual(
    OPENCODE_PLANS.map((plan: { label: string }) => plan.label),
    ["Console", "Go"],
  );
  assert.equal(connectionPickerValue("opencode"), "opencode");
  assert.equal(connectionPickerValue("opencode_go"), "opencode");
  assert.match(dialog, /HIDDEN_PROVIDER_TYPES = new Set\(\[[^\]]*"opencode_go"/);
});

test("other connections keep their own picker entry and show no plan", () => {
  for (const type of ["openai", "minimax", "custom"]) {
    assert.equal(connectionPickerValue(type), type);
    assert.equal(isOpenCodeProviderType(type), false);
  }
});

test("a saved connection cannot change plan", () => {
  assert.match(dialog, /id="provider-opencode-plan"[\s\S]{0,160}disabled=\{editingProviderId != null\}/);
  assert.match(dialog, /chooseProviderType = \(value: string\) => \{\s*if \(editingProviderId\) return;/);
});
