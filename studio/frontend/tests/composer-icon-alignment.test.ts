// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const componentSources = [
  "../src/components/assistant-ui/thread.tsx",
  "../src/components/assistant-ui/chat-dictation-bar.tsx",
  "../src/features/chat/shared-composer.tsx",
].map((path) => readFileSync(new URL(path, import.meta.url), "utf8"));

const css = readSrc("index.css");

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

test("every isolated composer stop square draws at size-3.5, about a third of its circle", () => {
  const stops = componentSources
    .flatMap((source) => tags(source, "HugeiconsIcon"))
    .filter((tag) => /icon=\{StopIcon\}/.test(tag) && /\bsize-3\.5(?![\w.-])/.test(tag));

  assert.equal(stops.length, 6);
});

test("neither composer action glyph carries a device-pixel nudge", () => {
  for (const rule of COMPOSER_GLYPH_RULES) {
    const body = css.match(rule)?.[1];
    if (body !== undefined) assert.doesNotMatch(body, TRANSLATE);
  }
  for (const source of componentSources) {
    assert.doesNotMatch(source, RETIRED_CANCEL_CLASS);
  }
});

test("the dictate glyph keeps a 1px optical nudge, not the 2px that read as low", () => {
  const body = css.match(/& svg\.unsloth-dictate-icon \{([\s\S]*?)\n\t\}/)?.[1];
  assert.ok(body, "dictate icon rule");
  assert.match(body, /transform:\s*translateY\(1px\);/);
});
