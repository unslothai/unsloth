// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Each tool_output chunk replaces toolLiveOutput: a group subscribed to the map re-renders on every chunk.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const GROUP = readSrc("components/assistant-ui/tool-group.tsx");

test("tool groups select a live-output flag, not the toolLiveOutput map", () => {
  assert.doesNotMatch(
    GROUP,
    /useChatRuntimeStore\(\s*\(\w+\)\s*=>\s*\w+\.toolLiveOutput\s*,?\s*\)/,
  );
  const selector = GROUP.match(
    /const hasLiveOutput = useChatRuntimeStore\(\(s\) =>([\s\S]*?)\);\n/,
  );
  assert.ok(selector, "hasLiveOutput must come from its own store selector");
  assert.match(selector[1], /\.some\(/);
  assert.match(selector[1], /s\.toolLiveOutput/);
});

test("both pane scopes still count as live output", () => {
  assert.match(GROUP, /toolOutputKey\(paneScope, /);
  assert.match(GROUP, /toolOutputKey\(unresolvedScope, /);
});
