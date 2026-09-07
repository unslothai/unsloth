// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const componentSources = [
  "../src/components/assistant-ui/thread.tsx",
  "../src/components/assistant-ui/chat-dictation-bar.tsx",
  "../src/features/chat/shared-composer.tsx",
].map((path) => readFileSync(new URL(path, import.meta.url), "utf8"));

const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");

function tags(source: string, component: string): string[] {
  return source.match(new RegExp(`<${component}\\b[^>]*\\/>`, "g")) ?? [];
}

const COMPOSER_GLYPH_RULES = [
  /& svg\.unsloth-send-icon \{([\s\S]*?)\n\t\}/,
  /& svg\.aui-composer-cancel-icon \{([\s\S]*?)\n\t\}/,
];
const TRANSLATE = /transform:\s*translate/;
const RETIRED_CANCEL_CLASS = /aui-composer-cancel-icon/;

test("every composer send arrow uses the shared sizing class", () => {
  const arrows = componentSources.flatMap((source) => tags(source, "ArrowUpIcon"));

  assert.equal(arrows.length, 5);
  for (const arrow of arrows) assert.match(arrow, /unsloth-send-icon/);
});

test("every isolated composer stop square keeps its size-3 glyph", () => {
  const stops = componentSources
    .flatMap((source) => tags(source, "SquareIcon"))
    .filter((tag) => /\bsize-3\b/.test(tag));

  assert.equal(stops.length, 6);
});

// A sub-pixel translate is calibrated for one DPR and skews the glyph on the
// others, so centering has to come from the geometry.
test("neither composer action glyph carries a device-pixel nudge", () => {
  for (const rule of COMPOSER_GLYPH_RULES) {
    const body = css.match(rule)?.[1];
    if (body !== undefined) assert.doesNotMatch(body, TRANSLATE);
  }
  // The stop square is styled by size-3 alone now, so the class it used to
  // carry the nudge on must be gone from the components too.
  for (const source of componentSources) {
    assert.doesNotMatch(source, RETIRED_CANCEL_CLASS);
  }
});
