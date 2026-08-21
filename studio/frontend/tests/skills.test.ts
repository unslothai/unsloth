// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  STUDIO_SKILL_TOOL_NAMES,
  isSuccessfulCreateSkillResult,
  readSkillToolDisplay,
} from "../src/features/chat/skill-tools.ts";

const adapterSource = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  ),
  "utf8",
);

test("read_skill calls explain progressive skill loading", () => {
  assert.deepEqual(readSkillToolDisplay({ name: "pr-9355-smoke" }), {
    actionLabel: "Read skill instructions",
    toolName: "pr-9355-smoke",
  });
  assert.deepEqual(
    readSkillToolDisplay({
      name: "pr-9355-smoke",
      resource: "references/phrase.txt",
    }),
    {
      actionLabel: "Read skill resource",
      toolName: "pr-9355-smoke · references/phrase.txt",
    },
  );
});

test("Studio skill tools stay paired", () => {
  assert.deepEqual(
    [...STUDIO_SKILL_TOOL_NAMES],
    ["create_skill", "read_skill"],
  );
});

test("only successful create_skill results announce catalog changes", () => {
  const cases = [
    [
      "create_skill",
      "Installed skill 'example'. It will be available on the next turn.",
      true,
    ],
    ["create_skill", "Installed skill 'example'. It remains disabled.", true],
    ["create_skill", "Error: invalid SKILL.md", false],
    ["read_skill", "Installed skill 'example'.", false],
    ["create_skill", undefined, false],
  ] as const;
  for (const [toolName, result, expected] of cases) {
    assert.equal(isSuccessfulCreateSkillResult(toolName, result), expected);
  }
});

test("chat requests and tool results keep skill wiring", () => {
  assert.match(
    adapterSource,
    /const skillToolsAvailableForThisTurn = Boolean\(\s*supportsStudioToolsForThisTurn &&\s*\(\s*isExternalRequest\s*\|\|\s*!imageBase64\s*\|\|\s*selectedModelSummary\?\.isGguf === true\s*\),?\s*\);/,
  );
  assert.equal(
    adapterSource.match(/\.\.\.STUDIO_SKILL_TOOL_NAMES/g)?.length,
    1,
  );
  assert.equal(
    adapterSource.match(
      /skillToolsAvailableForThisTurn\s*\?\s*STUDIO_SKILL_TOOL_NAMES/g,
    )?.length,
    2,
  );
  assert.match(
    adapterSource,
    /isSuccessfulCreateSkillResult\(completedToolName, rawEvent\)[\s\S]{0,80}announceSkillCatalogChanged\(\)/,
  );
});
