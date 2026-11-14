// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// /skills: the command match, and that the composer opens the Skills dialog instead of sending it.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";
import { isSkillsCommand } from "../src/features/chat/utils/skills-command.ts";

test("only the bare /skills command matches", () => {
  for (const text of ["/skills", "  /skills\n", "/SKILLS", "/Skills "]) {
    assert.equal(isSkillsCommand(text), true, JSON.stringify(text));
  }
  for (const text of [
    "",
    "/skills list",
    "skills",
    "/skill",
    "use /skills",
    "/ skills",
  ]) {
    assert.equal(isSkillsCommand(text), false, JSON.stringify(text));
  }
});

test("the composer opens the Skills dialog instead of sending /skills", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const submit = thread.slice(
    thread.indexOf("const handleSubmit = useCallback("),
  );
  // An attachment or a staged audio clip means a message to send, as for /compact.
  const intercept = submit.search(
    /!hasAttachments &&\s*!hasPendingAudio &&\s*isSkillsCommand\(composerText\)/,
  );
  assert.ok(intercept > 0);
  assert.ok(intercept < submit.indexOf("if (disabled || shouldBlockSend())"));
  assert.ok(intercept < submit.indexOf("interceptSend(event)"));
  assert.match(
    submit.slice(intercept),
    /isSkillsCommand\(composerText\)\s*\) \{\s*event\.preventDefault\(\);\s*clearStoredDraft\(\);\s*aui\.composer\(\)\.setText\(""\);\s*setSkillsDialogOpen\(true\);\s*return;/,
  );
  assert.match(
    thread,
    /<ChatSkillsDialog\s+open=\{skillsDialogOpen\}\s+onOpenChange=\{setSkillsDialogOpen\}\s+\/>/,
  );
});
