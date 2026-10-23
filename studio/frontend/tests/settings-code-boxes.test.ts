// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const COMMAND_BLOCK = readSrc(
  "features/settings/components/agent-command-block.tsx",
);
const USAGE = readSrc("features/settings/components/usage-examples.tsx");
const AUDIO = readSrc("features/settings/components/audio-api-examples.tsx");

const commandBlock = COMMAND_BLOCK.slice(
  COMMAND_BLOCK.indexOf("export function CommandBlock"),
);
const highlightedCode = USAGE.slice(
  USAGE.indexOf("export function HighlightedCode"),
  USAGE.indexOf("export function UsageExamples"),
);

test("command boxes wrap long commands instead of scrolling under the copy button", () => {
  const pre = commandBlock.match(/<pre className="([^"]+)"/)?.[1] ?? "";
  const code = commandBlock.match(/<code className="([^"]+)"/)?.[1] ?? "";
  assert.doesNotMatch(pre, /overflow-x-auto/);
  assert.match(pre, /\bpr-11\b/);
  assert.match(code, /\bwhitespace-pre-wrap\b/);
  assert.match(code, /\bbreak-all\b/);
});

test("long API examples are capped and scroll inside a keyboard-reachable region", () => {
  assert.match(highlightedCode, /<section\b/);
  assert.match(highlightedCode, /tabIndex=\{0\}/);
  assert.match(highlightedCode, /aria-label=\{label\}/);
  assert.match(
    highlightedCode,
    /className="max-h-96 [^"]*\boverflow-auto\b[^"]*\bscroll-rounded rounded-lg\b[^"]*\bfocus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset\b/,
  );
  // redaction must stay on the scroller that holds the snippet text
  assert.match(
    highlightedCode,
    /aria-label=\{label\}[\s\S]*data-reload-snapshot-sensitive=\{redactFromReload \? "" : undefined\}/,
  );
  for (const caller of [USAGE, AUDIO]) {
    assert.match(
      caller,
      /<HighlightedCode[\s\S]*?label=\{t\("settings\.apiKeys\./,
    );
  }
});
