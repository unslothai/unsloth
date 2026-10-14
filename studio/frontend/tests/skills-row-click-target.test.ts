// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const DIALOG = readSrc("features/chat/components/chat-skills-dialog.tsx");

const SKILL_ROW = /\nfunction SkillRow\([\s\S]*?\n\}\n/;
const ROW_ROOT = /return \(\s*<div\s+className=\{cn\(\s*"([^"]*)"/;
const TAKES_POINTER = /<(\w+)\b([^<]*?)\bpointer-events-auto\b/g;
const CLICK_THROUGH = /(?:^|\s)pointer-events-none(?:\s|$)/;

test("only the details button and the switch take clicks in a skill row", () => {
  const row = SKILL_ROW.exec(DIALOG)?.[0];
  assert.ok(row, "SkillRow not found");
  assert.match(ROW_ROOT.exec(row)?.[1] ?? "", CLICK_THROUGH);

  const targets = [...row.matchAll(TAKES_POINTER)].map(([, tag, attrs]) =>
    tag === "button" && attrs.includes("onClick={onOpen}") ? "details" : tag,
  );
  assert.deepEqual(targets, ["details", "Switch"]);
});
