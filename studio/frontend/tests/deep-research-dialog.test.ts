// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path: string) => readFile(new URL(path, import.meta.url), "utf8");
const RESEARCH = await read("../src/features/chat/components/deep-research-composer-button.tsx");

test("a Deep research domain list is one field, not a pill inside a pill", () => {
  const list = RESEARCH.slice(RESEARCH.indexOf("function DomainList("), RESEARCH.indexOf("export function DeepResearchComposerButton("));
  assert.doesNotMatch(list, /<Input\b/);
  assert.match(list, /<input\b/);
});

test("Deep research's time limit is a switch, and its header carries the tool's icon", () => {
  assert.match(RESEARCH, /<Switch\s+id="research-no-time-limit"\s+checked=\{unlimited\}/);
  assert.match(RESEARCH, /\{unlimited \? null : \(/);
  assert.doesNotMatch(RESEARCH, /"Use a limit"/);
  assert.match(RESEARCH, /icon=\{Telescope02Icon\} strokeWidth=\{1\.75\} className="size-5 text-primary"/);
});
