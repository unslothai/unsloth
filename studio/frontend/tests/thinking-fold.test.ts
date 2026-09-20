// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Folding a turn's tool calls under the thinking block they came from. The parts are grouped by
// type, so the tool calls render as siblings of that block rather than inside it: what these
// assertions protect is the pairing between the two, and the calls that must never be folded away.

import assert from "node:assert/strict";
import test from "node:test";
import {
  countRoundToolParts,
  foldedToolSummary,
  governingReasoningEnd,
  reasoningRoundKey,
} from "../src/components/assistant-ui/thinking-fold.ts";
import { en } from "../src/i18n/locales/en.ts";
import { SETTINGS_SEARCH_INDEX } from "../src/features/settings/settings-search.ts";
import { readSrc, readSrcAsync } from "./helpers/kit.ts";

const parts = (types: string[]) => types.map((type) => ({ type }));

test("a run of tool calls belongs to the thinking block before it", () => {
  const message = parts([
    "reasoning",
    "tool-call",
    "tool-call",
    "reasoning",
    "tool-call",
  ]);
  assert.equal(governingReasoningEnd(message, 1), 0);
  // The second call of the same run answers the same round as the first.
  assert.equal(governingReasoningEnd(message, 2), 0);
  assert.equal(governingReasoningEnd(message, 4), 3);
});

test("tool calls with no thinking before them are never folded", () => {
  // Nothing to fold them under.
  assert.equal(governingReasoningEnd(parts(["tool-call"]), 0), null);
  // An answer ends the round: calls after it belong to the answer, not to the earlier thinking.
  assert.equal(
    governingReasoningEnd(parts(["reasoning", "text", "tool-call"]), 2),
    null,
  );
});

test("a round counts only the calls that follow it", () => {
  const message = parts([
    "reasoning",
    "tool-call",
    "tool-call",
    "reasoning",
    "tool-call",
  ]);
  assert.equal(countRoundToolParts(message, 0), 2);
  assert.equal(countRoundToolParts(message, 3), 1);
  // A block whose round holds nothing says nothing.
  assert.equal(countRoundToolParts(parts(["reasoning", "text"]), 0), 0);
});

test("the header says how many calls it is holding", () => {
  assert.equal(foldedToolSummary(0), null);
  assert.equal(foldedToolSummary(1), "1 tool call");
  assert.equal(foldedToolSummary(12), "12 tool calls");
  // The same wording the tool group trigger uses, so the count reads the same once unfolded.
  const toolGroup = readSrc("components/assistant-ui/tool-group.tsx");
  assert.match(
    toolGroup,
    /const label = `\$\{count\} tool \$\{count === 1 \? "call" : "calls"\}`;/,
  );
});

test("two blocks in one message stay apart", () => {
  assert.notEqual(reasoningRoundKey("m1", 0), reasoningRoundKey("m1", 3));
  assert.notEqual(reasoningRoundKey("m1", 0), reasoningRoundKey("m2", 0));
});

// A round publishes its open state for the calls that follow it. In a layout effect: the calls
// render after the block in the same commit, so a plain effect would show them for one frame.
test("the thinking block publishes its open state before the frame", async () => {
  const reasoning = await readSrcAsync("components/assistant-ui/reasoning.tsx");
  assert.match(
    reasoning,
    /useLayoutEffect\(\(\) => \{\n\s*if \(!foldToolActivity\) return;\n\s*setReasoningRoundOpen\(roundKey, isOpen\);\n\s*\}, \[foldToolActivity, isOpen, roundKey\]\);/,
  );
  assert.match(
    reasoning,
    /useLayoutEffect\(\(\) => \(\) => clearReasoningRound\(roundKey\), \[roundKey\]\);/,
  );
  // A round nobody has published counts as closed, so nothing flashes before its block is read.
  const store = readSrc("features/chat/stores/reasoning-round-store.ts");
  assert.match(store, /open: Record<string, boolean>;/);
  const toolGroup = readSrc("components/assistant-ui/tool-group.tsx");
  assert.match(toolGroup, /state\.open\[roundKey\] \?\? false/);
});

test("a folded run is hidden, not unmounted", () => {
  const toolGroup = readSrc("components/assistant-ui/tool-group.tsx");
  // One wrapper either way: the cards keep their state, their scroll and any live output.
  assert.match(
    toolGroup,
    /className=\{cn\(\n\s*roundOpen\n\s*\? "mt-1 border-muted-foreground\/25 border-l pl-3"\n\s*: "hidden",\n\s*\)\}/,
  );
  assert.match(toolGroup, /data-slot="tool-run-under-thinking"/);
});

test("output and approvals are never folded away", () => {
  const toolGroup = readSrc("components/assistant-ui/tool-group.tsx");
  assert.match(
    toolGroup,
    /const foldable =\n\s*foldToolActivity &&\n\s*roundKey !== null &&\n\s*!containsUngroupedTool &&\n\s*!hasPendingConfirmation;/,
  );
  // With the preference off the tree is exactly what it was, wrapper included.
  assert.match(toolGroup, /if \(!foldable\) \{\n\s*return group;\n\s*\}/);
});

test("the preference ships off and survives a reload", () => {
  const prefs = readSrc("features/chat/stores/chat-preferences-store.ts");
  assert.match(prefs, /foldToolActivityIntoThinking: false,/);
  assert.match(
    prefs,
    /foldToolActivityIntoThinking:\n\s*saved\?\.foldToolActivityIntoThinking \?\? false,/,
  );
});

test("the setting is in the Chat tab and findable", () => {
  const chatTab = readSrc("features/settings/tabs/chat-tab.tsx");
  assert.match(chatTab, /t\("settings\.chat\.tools\.foldIntoThinking"\)/);
  assert.match(
    chatTab,
    /checked=\{foldToolActivityIntoThinking\}\n\s*onCheckedChange=\{setFoldToolActivityIntoThinking\}/,
  );
  assert.ok(
    SETTINGS_SEARCH_INDEX.chat.includes(
      "settings.chat.tools.foldIntoThinking",
    ),
    "the setting cannot be found by search",
  );
  assert.equal(
    en.settings.chat.tools.foldIntoThinking,
    "Fold tool calls into Thinking",
  );
  assert.ok(en.settings.chat.tools.foldIntoThinkingDescription.length > 0);
});
