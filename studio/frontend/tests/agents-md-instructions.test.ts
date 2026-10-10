// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  EMPTY_AGENTS_MD,
  composeChatInstructions,
} from "../src/features/chat/utils/agents-md.ts";
import { readSrc } from "./helpers/kit.ts";

const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");
const DIALOG = readSrc(
  "features/chat/prompt-storage/prompt-storage-dialog.tsx",
);

// The composition before AGENTS.md support, kept verbatim as the oracle for "no file, no change".
function legacyCompose(
  projectInstructions: string,
  systemPrompt: string,
): string {
  return [
    projectInstructions
      ? `<project_instructions>\n${projectInstructions}\n</project_instructions>`
      : "",
    systemPrompt.trim(),
  ]
    .filter(Boolean)
    .join("\n\n");
}

test("without AGENTS.md the system prompt is byte-identical to before", () => {
  for (const [instructions, prompt] of [
    ["", ""],
    ["Use pnpm.", ""],
    ["", "  You are helpful.  "],
    ["Use pnpm.", "You are helpful."],
  ]) {
    const expected = legacyCompose(instructions, prompt);
    for (const agentsMd of [undefined, EMPTY_AGENTS_MD, { text: " \n\t" }]) {
      assert.equal(
        composeChatInstructions({
          agentsMd,
          projectInstructions: instructions,
          systemPrompt: prompt,
        }),
        expected,
      );
    }
  }
});

test("one AGENTS.md block comes first, the typed prompt last", () => {
  const agentsMd = {
    text: "# Source: ~/.agents/AGENTS.md\n\nBe brief.\n\n# Source: ~/proj/AGENTS.md\n\nUse pnpm.\n",
  };
  const text = composeChatInstructions({
    agentsMd,
    projectInstructions: "Cite sources.",
    systemPrompt: "You are helpful.",
  });
  assert.equal(
    text,
    [
      `<agents_md>\n${agentsMd.text.trim()}\n</agents_md>`,
      "<project_instructions>\nCite sources.\n</project_instructions>",
      "You are helpful.",
    ].join("\n\n"),
  );
  // Idempotent: the same inputs always give the same single block.
  assert.equal(
    composeChatInstructions({
      agentsMd,
      projectInstructions: "Cite sources.",
      systemPrompt: "You are helpful.",
    }),
    text,
  );
  assert.equal(text.split("<agents_md>").length - 1, 1);
});

test("only requests sent to the model ask for AGENTS.md; copies and exports do not", () => {
  const calls =
    ADAPTER.match(/await resolveChatInstructions\([\s\S]*?\);/g) ?? [];
  assert.equal(calls.length, 2, "chat send and deep research");
  for (const call of calls) {
    assert.match(call, /\{ agentsMd: true \}/);
  }
  assert.doesNotMatch(DIALOG, /agentsMd/);
  // The background token recount must measure the prompt the send will use.
  const recount = ADAPTER.slice(
    ADAPTER.indexOf("export async function buildLocalTokenCountHistory"),
    ADAPTER.indexOf("export function buildLocalTokenCountReasoning"),
  );
  assert.match(recount, /resolveAgentsMd\(projectId\)/);
  assert.match(recount, /composeChatInstructions\(/);
});
