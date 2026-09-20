// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Folding a turn's run-up under its first Thinking block. The parts are grouped by type, so the
// tool calls and later thinking render as siblings of that block rather than inside it: what
// these assertions protect is which siblings the block holds, and the calls it must never hide.

import assert from "node:assert/strict";
import test from "node:test";
import {
  countFoldedToolParts,
  foldEnd,
  foldedToolSummary,
  foldedTurnDuration,
  governingReasoningEnd,
  isFoldedReasoningGroup,
  leadReasoningEnd,
  reasoningRoundKey,
} from "../src/components/assistant-ui/thinking-fold.ts";
import { en } from "../src/i18n/locales/en.ts";
import { SETTINGS_SEARCH_INDEX } from "../src/features/settings/settings-search.ts";
import { readSrc, readSrcAsync } from "./helpers/kit.ts";

const parts = (types: string[]) => types.map((type) => ({ type }));

// reasoning, tools, reasoning, tools, answer, one more call after the answer.
const twoRounds = parts([
  "reasoning",
  "tool-call",
  "tool-call",
  "reasoning",
  "reasoning",
  "tool-call",
  "text",
  "tool-call",
]);

test("the first thinking block heads the turn", () => {
  assert.equal(leadReasoningEnd(twoRounds), 0);
  // Two reasoning parts in a row are one block.
  assert.equal(leadReasoningEnd(parts(["reasoning", "reasoning", "text"])), 1);
  // A call before any thinking does not stop the first block from leading.
  assert.equal(leadReasoningEnd(parts(["tool-call", "reasoning", "text"])), 1);
  // An answer before any thinking, or no thinking at all, and nothing leads.
  assert.equal(leadReasoningEnd(parts(["text", "reasoning"])), null);
  assert.equal(leadReasoningEnd(parts(["tool-call", "text"])), null);
});

test("the fold runs from the lead to the answer", () => {
  assert.equal(foldEnd(twoRounds, 0), 6);
  assert.equal(foldEnd(parts(["reasoning", "tool-call"]), 0), 2);
});

test("every group before the answer folds under the lead", () => {
  for (const start of [1, 2, 3, 5]) {
    assert.equal(governingReasoningEnd(twoRounds, start), 0);
  }
  // The lead itself heads the fold rather than joining it.
  assert.equal(governingReasoningEnd(twoRounds, 0), null);
  assert.equal(isFoldedReasoningGroup(twoRounds, 3), true);
  assert.equal(isFoldedReasoningGroup(twoRounds, 0), false);
});

test("calls with no thinking before them, or after the answer, stay put", () => {
  assert.equal(governingReasoningEnd(parts(["tool-call"]), 0), null);
  assert.equal(
    governingReasoningEnd(parts(["tool-call", "reasoning"]), 0),
    null,
  );
  assert.equal(governingReasoningEnd(twoRounds, 7), null);
  // Thinking after the answer keeps its own block.
  assert.equal(
    isFoldedReasoningGroup(parts(["reasoning", "text", "reasoning"]), 2),
    false,
  );
});

test("the lead counts every call it holds and none after the answer", () => {
  assert.equal(countFoldedToolParts(twoRounds), 3);
  assert.equal(countFoldedToolParts(parts(["reasoning", "text"])), 0);
  assert.equal(countFoldedToolParts(parts(["text", "tool-call"])), 0);
});

test("the lead reports the turn's thinking time added up", () => {
  const byStart: Record<number, number | undefined> = { 0: 4, 3: 6 };
  const resolve = (_parts: readonly { type: string }[], start: number) =>
    byStart[start];
  assert.equal(foldedTurnDuration(twoRounds, resolve), 10);
  // One round still unmeasured and the caller keeps its own clock.
  assert.equal(
    foldedTurnDuration(twoRounds, (_p, start) => (start === 0 ? 4 : undefined)),
    undefined,
  );
  assert.equal(foldedTurnDuration(parts(["text"]), resolve), undefined);
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

// The lead publishes its open state for the parts folded under it. In a layout effect: they
// render after the block in the same commit, so a plain effect would show them for one frame.
test("the lead publishes its open state before the frame", async () => {
  const reasoning = await readSrcAsync("components/assistant-ui/reasoning.tsx");
  assert.match(
    reasoning,
    /useLayoutEffect\(\(\) => \{\n\s*if \(!foldLead\) return;\n\s*setReasoningRoundOpen\(roundKey, isOpen\);\n\s*\}, \[foldLead, isOpen, roundKey\]\);/,
  );
  assert.match(
    reasoning,
    /useLayoutEffect\(\(\) => \(\) => clearReasoningRound\(roundKey\), \[roundKey\]\);/,
  );
  // A block nobody has published counts as closed, so nothing flashes before it is read.
  const store = readSrc("features/chat/stores/reasoning-round-store.ts");
  assert.match(store, /open: Record<string, boolean>;/);
  const toolGroup = readSrc("components/assistant-ui/tool-group.tsx");
  assert.match(toolGroup, /state\.open\[roundKey\] \?\? false/);
  assert.match(reasoning, /state\.open\[roundKey\] \?\? false/);
});

test("later thinking rounds render inside the lead, with no header of their own", async () => {
  const reasoning = await readSrcAsync("components/assistant-ui/reasoning.tsx");
  assert.match(
    reasoning,
    /if \(folded\) \{\n\s*return <FoldedReasoningRound \{\.\.\.props\} \/>;\n\s*\}/,
  );
  assert.match(reasoning, /data-slot="reasoning-folded-round"/);
  // The lead keeps working through those rounds, so its clock covers the whole turn.
  assert.match(
    reasoning,
    /if \(type !== "tool-call" && !\(foldLead && type === "reasoning"\)\)/,
  );
});

test("a folded run is hidden, not unmounted", () => {
  const toolGroup = readSrc("components/assistant-ui/tool-group.tsx");
  // One wrapper either way: the cards keep their state, their scroll and any live output.
  assert.match(toolGroup, /className=\{cn\(!roundOpen && "hidden"\)\}/);
  assert.match(toolGroup, /data-slot="tool-run-under-thinking"/);
  const reasoning = readSrc("components/assistant-ui/reasoning.tsx");
  assert.match(reasoning, /!open && "hidden"/);
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
    SETTINGS_SEARCH_INDEX.chat.includes("settings.chat.tools.foldIntoThinking"),
    "the setting cannot be found by search",
  );
  assert.equal(
    en.settings.chat.tools.foldIntoThinking,
    "Fold tool calls into Thinking",
  );
  assert.ok(en.settings.chat.tools.foldIntoThinkingDescription.length > 0);
});
