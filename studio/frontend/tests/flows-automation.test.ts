// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { pendingItems, webhookUrl } = await import(
  "../src/features/flows/capability-items.ts"
);

const none = {
  python: [],
  hosts: [],
  mcp: false,
  providers: [],
  triggers: false,
};

test("nothing pending shows no approval banner", () => {
  assert.deepEqual(pendingItems(none), []);
  assert.deepEqual(pendingItems(null), []);
});

test("each pending capability is one readable line", () => {
  const items = pendingItems({
    python: ["sha256:a", "sha256:b"],
    hosts: ["api.example.com"],
    mcp: true,
    providers: ["openai"],
    triggers: true,
  });
  assert.deepEqual(
    items.map((i) => i.label),
    [
      "Runs Python code (2 blocks)",
      "Calls api.example.com",
      "Uses your MCP servers' tools",
      "Uses your openai connection",
      "Starts on its own (schedule or webhook) while Studio is open",
    ],
  );
  assert.equal(new Set(items.map((i) => i.key)).size, items.length);
});

test("the webhook URL joins the page origin and the route path", () => {
  assert.equal(
    webhookUrl("http://127.0.0.1:8888/", "/api/flows/hooks/abc"),
    "http://127.0.0.1:8888/api/flows/hooks/abc",
  );
});

test("the webhook URL uses the backend origin and never shows the last block's secret", async () => {
  const { readFileSync } = await import("node:fs");
  const source = readFileSync(
    new URL("../src/features/flows/editor/trigger-panel.tsx", import.meta.url),
    "utf8",
  );
  // Desktop serves the page from the webview, not the HTTP server the hook lives on.
  assert.match(source, /webhookUrl\(getApiBase\(\) \|\| window\.location\.origin/);
  assert.match(source, /setTrigger\(null\);\s*\}, \[flowId, nodeId\]\)/);
});

test("the approval banner never shows or approves another flow's report", async () => {
  const { readFileSync } = await import("node:fs");
  const source = readFileSync(
    new URL("../src/features/flows/editor/capability-banner.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /current\.current === flowId && setReport\(found\)/);
  assert.match(source, /setReport\(null\);\s*\}, \[flowId\]\)/);
  // An approval that returns after the user opened another flow must not land on that flow's banner.
  assert.match(source, /if \(current\.current === flowId\) setReport\(approved\)/);
});
