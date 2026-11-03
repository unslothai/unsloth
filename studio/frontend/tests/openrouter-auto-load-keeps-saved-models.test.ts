// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer, type ViteDevServer } from "vite";

let vite: ViteDevServer;
let withSavedModelIds: (catalog: string[], saved: string[]) => string[];

before(async () => {
  vite = await createServer({ appType: "custom", server: { middlewareMode: true } });
  const loaded = await vite.ssrLoadModule(
    "/src/features/chat/chat-providers-dialog.tsx",
  );
  withSavedModelIds = loaded.withSavedModelIds;
});

after(async () => {
  await vite.close();
});

test("saved ids missing from the live catalog stay pickable", () => {
  assert.deepEqual(
    withSavedModelIds(
      ["openai/gpt-5.5", "deepseek/deepseek-v4-pro"],
      ["deepseek/deepseek-v4-pro:nitro", "openai/gpt-5.5"],
    ),
    ["openai/gpt-5.5", "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro:nitro"],
  );
});
