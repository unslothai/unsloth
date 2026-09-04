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

test("every composer send arrow uses the optical centering class", () => {
  const arrows = componentSources.flatMap((source) => tags(source, "ArrowUpIcon"));

  assert.equal(arrows.length, 5);
  for (const arrow of arrows) assert.match(arrow, /unsloth-send-icon/);
  assert.match(
    css,
    /svg\.unsloth-send-icon[\s\S]*?transform: translateX\(-0\.25px\)/,
  );
});

test("every isolated composer stop square uses the optical centering class", () => {
  const stops = componentSources
    .flatMap((source) => tags(source, "SquareIcon"))
    .filter((tag) => /\bsize-3\b/.test(tag));

  assert.equal(stops.length, 6);
  for (const stop of stops) assert.match(stop, /aui-composer-cancel-icon/);
  assert.match(
    css,
    /svg\.aui-composer-cancel-icon[\s\S]*?transform: translateX\(-0\.5px\)/,
  );
});
