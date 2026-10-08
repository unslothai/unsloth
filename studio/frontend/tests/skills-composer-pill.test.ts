// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path: string) => readFile(new URL(path, import.meta.url), "utf8");
const PILL = await read("../src/features/chat/skills-composer-button.tsx");
const THREAD = await read("../src/components/assistant-ui/thread.tsx");
const SHARED = await read("../src/features/chat/shared-composer.tsx");

test("the Skills pill shows once a usable skill is on, in both composers", () => {
  assert.match(PILL, /skills\.filter\(\(skill\) => skill\.valid && !skill\.shadowed\)/);
  assert.match(PILL, /\{enabledCount > 0 \|\| menuOpen \? \(/);
  assert.match(THREAD, /<SkillsComposerButton side=\{effectiveMenuSide\} \/>/);
  assert.match(SHARED, /<SkillsComposerButton side="top" \/>/);
});

test("one list of every runnable skill under the Skills title, then Manage skills", () => {
  assert.match(PILL, /<DropdownMenuLabel>\{t\("skills\.title"\)\}<\/DropdownMenuLabel>/);
  assert.doesNotMatch(PILL, /source !== "bundled"|Settings02Icon/);
  assert.match(PILL, /<DropdownMenuSeparator \/>\s*<DropdownMenuItem/);
});

test("the Skills pill toggles in place and links to the full dialog", () => {
  assert.match(PILL, /event\.preventDefault\(\);\s*void toggle\(skill\.name, !skill\.enabled\);/);
  assert.match(PILL, /await setSkillEnabled\(name, enabled\);/);
  assert.match(PILL, /t\("skills\.manage"\)/);
  assert.match(PILL, /<ChatSkillsDialog open=\{dialogOpen\} onOpenChange=\{setDialogOpen\} \/>/);
});
