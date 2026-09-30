// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const DIALOG = readSrc("features/chat/components/chat-skills-dialog.tsx");

const SKILL_ROW = /\nfunction SkillRow\([\s\S]*?\n\}\n/;
const ROW_BUTTON =
  /<button\b[^>]*?onClick=\{onOpen\}[^>]*?className="absolute inset-0\b[^>]*?\/>/;
const CLASSED_TAG = /<(\w+)\b[^>]*?className="([^"]*)"/g;
const PAINTS_ABOVE =
  /(?:^|\s)(?:-?translate-|relative|absolute|fixed|sticky|z-)/;
const CLICK_THROUGH = /\bpointer-events-none\b/;
const INTERACTIVE = new Set(["Switch", "button", "a", "input"]);

test("nothing painted over a skill row swallows the click that opens it", () => {
  const row = SKILL_ROW.exec(DIALOG)?.[0];
  assert.ok(row, "SkillRow not found");
  const cover = ROW_BUTTON.exec(row);
  assert.ok(cover, "the full-row details button not found");

  const after = row.slice(cover.index + cover[0].length);
  for (const [, tag, classes] of after.matchAll(CLASSED_TAG)) {
    if (INTERACTIVE.has(tag) || !PAINTS_ABOVE.test(classes)) {
      continue;
    }
    assert.match(
      classes,
      CLICK_THROUGH,
      `<${tag} className="${classes}"> blocks the row`,
    );
  }
});
