// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);
const MISSING_DISCLAIMER_DEFAULT_PATTERN =
  /showModelDisclaimer: saved\?\.showModelDisclaimer \?\? false/;

test("the model disclaimer is hidden by default", () => {
  const fresh = useChatPreferencesStore.getInitialState();
  assert.equal(fresh.showModelDisclaimer, false);
});

test("a saved payload without the model disclaimer key defaults to hidden", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/stores/chat-preferences-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, MISSING_DISCLAIMER_DEFAULT_PATTERN);
});
