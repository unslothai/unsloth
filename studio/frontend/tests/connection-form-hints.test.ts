// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

test("the local tools note no longer names Docs", () => {
  const dialog = read("features/chat/chat-providers-dialog.tsx");
  assert.doesNotMatch(dialog, /MCP and Docs run on this machine/);
  assert.match(dialog, /Search, Tools and MCP run on this machine/);
});

test("a new vLLM or SGLang connection points at Unsloth's own engine", () => {
  const dialog = read("features/chat/chat-providers-dialog.tsx");
  assert.match(
    dialog,
    /editingProviderId == null && isManagedEngineType\(providerType\) \? \(\s+<ManagedEngineHint engine=\{providerType\} \/>/,
  );
  const hint = read("features/chat/managed-engine-hint.tsx");
  assert.match(hint, /status\.unsupported_reason/);
  assert.match(hint, /scrollTarget: "resources-engines"/);
  assert.match(hint, /\{isOwner \? \(\s+<>\s+\{" "\}\s+<button/);
  const resources = read("features/settings/tabs/resources-tab.tsx");
  assert.match(resources, /<InferenceEnginesSection ref=\{enginesRef\} \/>/);
});
