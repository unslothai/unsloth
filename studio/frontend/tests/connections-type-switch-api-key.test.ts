// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
const chooseProviderType =
  dialog.match(/const chooseProviderType = \(value: string\) => \{[\s\S]*?\n {2}\};/)?.[0] ?? "";

test("changing the connection type drops the typed API key", () => {
  assert.notEqual(chooseProviderType, "");
  assert.match(chooseProviderType, /setApiKey\(""\)/);
});

test("switching between the two OpenCode plans keeps the API key", () => {
  assert.match(
    chooseProviderType,
    /if \(!\(isOpenCodeProviderType\(providerType\) && isOpenCodeProviderType\(value\)\)\) \{\s*setApiKey\(""\);\s*\}/,
  );
});
