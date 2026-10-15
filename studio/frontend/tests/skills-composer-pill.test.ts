// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path: string) => readFile(new URL(path, import.meta.url), "utf8");
const PILL = await read("../src/features/chat/skills-composer-button.tsx");
const THREAD = await read("../src/components/assistant-ui/thread.tsx");
const SHARED = await read("../src/features/chat/shared-composer.tsx");

test("the Skills pill follows the Skills switch, in both composers", () => {
  assert.match(PILL, /skills\.filter\(\(skill\) => skill\.valid && !skill\.shadowed\)/);
  assert.match(PILL, /\{skillsOn \|\| menuOpen \? \(/);
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

test("unticking every skill keeps the pill; its X and the + menu switch Skills off", async () => {
  const { skillsSwitchedOn } = await import("../src/features/chat/stores/skills-switch-store.ts");
  assert.equal(skillsSwitchedOn(null, 0), false);
  assert.equal(skillsSwitchedOn(null, 2), true);
  assert.equal(skillsSwitchedOn(true, 0), true);
  assert.equal(skillsSwitchedOn(false, 2), false);
  assert.match(PILL, /role="button"[\s\S]*?setSkillsOn\(false\);[\s\S]*?<XIcon className="composer-pill-x" \/>/);
  for (const source of [THREAD, SHARED]) {
    assert.match(source, /onSelect=\{\(\) => setSkillsOn\(!skillsOn\)\}/);
  }
});

test("Skills switched off sends no skill tools, and ticking a skill switches it back on", async () => {
  const ADAPTER = await read("../src/features/chat/api/chat-adapter.ts");
  const API = await read("../src/features/chat/api/skills-api.ts");
  assert.match(ADAPTER, /useSkillsSwitchStore\.getState\(\)\.on === false \? \[\] : getSkillsSnapshot\(\)\.skills/);
  assert.match(API, /if \(updated\.enabled\) \{\s*useSkillsSwitchStore\.getState\(\)\.setOn\(true\);/);
});

test("an unset switch is pinned before a change can untick, delete or bulk-disable the last skill", async () => {
  const API = await read("../src/features/chat/api/skills-api.ts");
  for (const fn of ["setSkillEnabled", "setAllSkillsEnabled", "deleteSkill"]) {
    assert.match(API, new RegExp(`export async function ${fn}\\([\\s\\S]*?\\{\\s*pinSkillsSwitch\\(\\);`));
  }
  assert.match(API, /if \(on !== null\) \{\s*return;\s*\}\s*setOn\(\s*snapshot\.skills\.some\(/);
});

test("skill @mentions are offered only while Skills is on, in both composers", () => {
  assert.match(THREAD, /<SkillMentionPopover[\s\S]*?enabled=\{supportsTools && skillsSwitchOn\}/);
  assert.match(SHARED, /useTextareaSkillMentions\(\{[\s\S]*?enabled: supportsTools && skillsSwitchOn,/);
});
