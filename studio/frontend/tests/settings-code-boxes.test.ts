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
const MEDIA = readSrc("features/settings/components/media-api-examples.tsx");
const PARTS = readSrc("features/settings/components/api-example-parts.tsx");

const commandBlock = COMMAND_BLOCK.slice(
  COMMAND_BLOCK.indexOf("export function CommandBlock"),
);
const highlightedCode = PARTS.slice(
  PARTS.indexOf("export function HighlightedCode"),
  PARTS.indexOf("export function ExampleCodeBox"),
);

test("command boxes wrap long commands instead of scrolling under the copy button", () => {
  const pre = commandBlock.match(/<pre\s[^>]*?className="([^"]+)"/)?.[1] ?? "";
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
    /className="max-h-96 [^"]*\bhover-scrollbar overflow-auto\b[^"]*\bfocus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset\b/,
  );
  // square under the header: a rounded scroller cut its scrollbar's top off; the box clips the ends
  assert.doesNotMatch(highlightedCode, /scroll-rounded|rounded-lg/);
  assert.match(PARTS, /"relative min-w-0 overflow-hidden rounded-xl border/);
  // redaction must stay on the scroller that holds the snippet text
  assert.match(
    highlightedCode,
    /aria-label=\{label\}[\s\S]*data-reload-snapshot-sensitive=\{redactFromReload \? "" : undefined\}/,
  );
  // wrapped lines broke mid-token and ran back to the line-number gutter
  const cls =
    highlightedCode.match(/<section[\s\S]*?className="([^"]+)"/)?.[1] ?? "";
  assert.match(cls, /\[&_pre\]:!whitespace-pre(?!-)/);
  assert.doesNotMatch(cls, /break-words|break-all|whitespace-pre-wrap/);
  assert.match(cls, /\[&_\[data-streamdown=code-block\]\]:!min-w-max\b/);
  // all three example sections share one code box
  for (const caller of [USAGE, AUDIO, MEDIA]) {
    assert.match(
      caller,
      /<ExampleCodeBox[\s\S]*?label=\{t\("settings\.apiKeys\./,
    );
    assert.doesNotMatch(caller, /<HighlightedCode\b/);
  }
});
