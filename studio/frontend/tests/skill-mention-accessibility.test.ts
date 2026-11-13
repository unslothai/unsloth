// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const mentionsSource = readFileSync(
  new URL("../src/components/assistant-ui/skill-mentions.tsx", import.meta.url),
  "utf8",
);
const composerSource = readFileSync(
  new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
  "utf8",
);

test("the compare composer exposes its skill suggestions as a linked combobox", () => {
  assert.ok(composerSource.includes("{...skillMentions.inputProps}"));
  for (const semantic of [
    'role: "combobox"',
    '"aria-controls": open ? listboxId : undefined',
    '"aria-activedescendant": activeOptionId',
    'role="listbox"',
    'role="option"',
    "aria-selected={index === highlighted}",
  ]) {
    assert.ok(mentionsSource.includes(semantic), semantic);
  }
});

test("the popover caps the adapter search, so navigation never leaves the rendered rows", () => {
  assert.ok(
    mentionsSource.includes("rankMentionSkills(available, query, MAX_MENTION_RESULTS)"),
  );
  assert.ok(!mentionsSource.includes("results.slice(0, MAX_MENTION_RESULTS)"));
});

test("Tab accepts the highlighted row in the main composer, as Enter does", () => {
  assert.match(
    mentionsSource,
    /if \(event\.key !== "Tab" \|\| event\.shiftKey \|\| event\.isComposing\) return;/,
  );
  // reuse the library's Enter path so the token replacer still owns insertion.
  assert.match(mentionsSource, /key: "Enter",\s*shiftKey: false,/);
  assert.match(mentionsSource, /const active = open && items\.length > 0;/);
  // focus may leave while the popover is open; Tab elsewhere must stay with that field.
  assert.match(mentionsSource, /!scopeRef\.current\?\.contains\(event\.target\)/);
  assert.match(mentionsSource, /<MentionTabAccept scopeRef=\{composerRef\} \/>/);
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  assert.match(thread, /<SkillMentionPopover[\s\S]*?composerRef=\{editorRef\}/);
});
