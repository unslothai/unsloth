// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Folding a turn's run-up under its first Thinking block. The parts are grouped by type, so the
// tool calls and later thinking render as siblings of that block rather than inside it: what
// these assertions protect is which siblings the block holds, and the calls it must never hide.

import assert from "node:assert/strict";
import test from "node:test";
import {
  countFoldedToolParts,
  endsFoldedSpan,
  foldEnd,
  foldedToolSummary,
  foldedTurnDuration,
  foldRun,
  governingReasoningEnd,
  isBlankTextPart,
  isFoldedReasoningGroup,
  leadReasoningEnd,
  reasoningRoundKey,
} from "../src/components/assistant-ui/thinking-fold.ts";
import {
  awaitsConfirmation,
  holdsOwnOutput,
  toolRunIsExempt,
} from "../src/components/assistant-ui/tool-fold-exemptions.ts";
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

test("the first thinking block of a span heads it", () => {
  assert.equal(leadReasoningEnd(twoRounds, 0), 0);
  assert.equal(leadReasoningEnd(twoRounds, 5), 0);
  // Two reasoning parts in a row are one block.
  assert.equal(
    leadReasoningEnd(parts(["reasoning", "reasoning", "text"]), 1),
    1,
  );
  // A call before any thinking does not stop the first block from leading.
  assert.equal(
    leadReasoningEnd(parts(["tool-call", "reasoning", "text"]), 0),
    1,
  );
  // Answer text is not in any span, and a run with no thinking has no lead.
  assert.equal(leadReasoningEnd(parts(["text", "reasoning"]), 0), null);
  assert.equal(leadReasoningEnd(parts(["tool-call", "text"]), 0), null);
  // The call after the answer sits in a span of its own with nothing to lead it.
  assert.equal(leadReasoningEnd(twoRounds, 7), null);
});

test("a span runs from its lead to the answer", () => {
  assert.deepEqual(foldRun(twoRounds, 0), { start: 0, end: 6 });
  assert.deepEqual(foldRun(twoRounds, 7), { start: 7, end: 8 });
  assert.equal(foldRun(twoRounds, 6), null);
  assert.equal(foldEnd(twoRounds, 0), 6);
  assert.equal(foldEnd(parts(["reasoning", "tool-call"]), 0), 2);
});

test("blank text between thinking and a call does not end the span", () => {
  // A provider can leave a newline between a closed think block and its native tool event.
  const blank = [
    { type: "reasoning" },
    { type: "text", text: "\n" },
    { type: "tool-call" },
    { type: "reasoning" },
    { type: "text", text: "The answer." },
  ];
  assert.equal(isBlankTextPart(blank[1]), true);
  assert.equal(isBlankTextPart(blank[4]), false);
  assert.equal(isBlankTextPart({ type: "reasoning" }), false);
  assert.deepEqual(foldRun(blank, 0), { start: 0, end: 4 });
  assert.equal(governingReasoningEnd(blank, 2), 0);
  assert.equal(isFoldedReasoningGroup(blank, 3), true);
  assert.equal(countFoldedToolParts(blank, 0), 1);
  // The rule closes the last thing shown, not a trailing blank.
  const trailing = [
    { type: "reasoning" },
    { type: "tool-call" },
    { type: "text", text: " " },
    { type: "text", text: "The answer." },
  ];
  assert.equal(endsFoldedSpan(trailing, 1), true);
  assert.equal(endsFoldedSpan(trailing, 2), false);
});

test("a continuation's seeded reply sits before the new round, not in it", () => {
  // Continue seeds the new message with the previous reply as its first text part.
  const continued = [
    { type: "text", text: "What I had so far." },
    { type: "reasoning" },
    { type: "tool-call" },
    { type: "tool-call" },
    { type: "reasoning" },
    { type: "text", text: "And the rest." },
  ];
  assert.equal(leadReasoningEnd(continued, 1), 1);
  assert.equal(governingReasoningEnd(continued, 2), 1);
  assert.equal(isFoldedReasoningGroup(continued, 4), true);
  assert.equal(countFoldedToolParts(continued, 1), 2);
  assert.equal(endsFoldedSpan(continued, 4), true);
  // Two answers, two spans, each with its own lead.
  const twoSpans = [
    { type: "reasoning" },
    { type: "tool-call" },
    { type: "text", text: "First." },
    { type: "reasoning" },
    { type: "tool-call" },
    { type: "text", text: "Second." },
  ];
  assert.equal(leadReasoningEnd(twoSpans, 1), 0);
  assert.equal(leadReasoningEnd(twoSpans, 4), 3);
  assert.equal(governingReasoningEnd(twoSpans, 4), 3);
  assert.equal(countFoldedToolParts(twoSpans, 3), 1);
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
  assert.equal(countFoldedToolParts(twoRounds, 0), 3);
  assert.equal(countFoldedToolParts(parts(["reasoning", "text"]), 0), 0);
});

test("a run the thread keeps visible is not counted as held", () => {
  // The first run (parts 1..2) is exempt, say a call awaiting approval; the second is not.
  const exempt = (start: number) => start === 1;
  assert.equal(countFoldedToolParts(twoRounds, 0, exempt), 1);
  assert.equal(
    countFoldedToolParts(twoRounds, 0, () => true),
    0,
  );
  // Runs are passed whole, as the tool group sees them.
  const seen: [number, number][] = [];
  countFoldedToolParts(twoRounds, 0, (start, end) => {
    seen.push([start, end]);
    return false;
  });
  assert.deepEqual(seen, [
    [1, 2],
    [5, 5],
  ]);
});

test("calls whose output lives only in their card stay visible", () => {
  const call = (toolName: string, result?: unknown) => ({
    type: "tool-call",
    toolName,
    toolCallId: `c-${toolName}`,
    result,
  });
  // A generated image has no message part of its own: hide the card and the image is gone.
  assert.equal(holdsOwnOutput(call("image_generation")), true);
  assert.equal(holdsOwnOutput(call("render_html")), true);
  assert.equal(holdsOwnOutput(call("python")), true);
  assert.equal(
    holdsOwnOutput(
      call("terminal", {
        text: "",
        images: [],
        sessionId: "s",
        files: [{ name: "a.txt", size: 3 }],
      }),
    ),
    true,
  );
  // An MCP tool that returned images shows them only in its card.
  const mcpImages = {
    text: "two charts",
    images: [{ data: "iVBORw0KGgo=", mimeType: "image/png" }],
  };
  assert.equal(holdsOwnOutput(call("mcp__charts__plot", mcpImages)), true);
  assert.equal(
    holdsOwnOutput(call("mcp__charts__plot", { text: "none", images: [] })),
    false,
  );
  assert.equal(
    holdsOwnOutput(call("mcp__charts__plot", { text: "bad", images: ["x"] })),
    false,
  );
  assert.equal(holdsOwnOutput(call("web_search")), false);
  assert.equal(holdsOwnOutput(call("terminal", "plain output")), false);
  assert.equal(holdsOwnOutput({ type: "reasoning" }), false);
  // An approval prompt keeps its run visible; a resolved one does not.
  assert.equal(
    awaitsConfirmation(call("terminal"), { "c-terminal": {} }),
    true,
  );
  assert.equal(awaitsConfirmation(call("terminal"), {}), false);
  const run = [
    call("web_search"),
    call("image_generation"),
    call("web_search"),
  ];
  assert.equal(toolRunIsExempt(run, 0, 2, {}), true);
  assert.equal(toolRunIsExempt(run, 0, 0, {}), false);
  assert.equal(toolRunIsExempt(run, 0, 0, { "c-web_search": {} }), true);
});

test("the tool group and the header share one exemption rule", () => {
  const rules = readSrc("components/assistant-ui/tool-fold-exemptions.ts");
  assert.match(rules, /export function holdsOwnOutput/);
  assert.match(rules, /export function awaitsConfirmation/);
  assert.match(rules, /export function toolRunIsExempt/);
  const toolGroup = readSrc("components/assistant-ui/tool-group.tsx");
  assert.match(toolGroup, /\.some\(holdsOwnOutput\)/);
  assert.match(toolGroup, /awaitsConfirmation\(part, toolConfirmations\)/);
  const reasoning = readSrc("components/assistant-ui/reasoning.tsx");
  assert.match(
    reasoning,
    /countFoldedToolParts\(message\.parts, endIndex, \(start, end\) =>\n\s*toolRunIsExempt\(message\.parts, start, end, toolConfirmations\),/,
  );
});

test("the rule that closes the trace goes on the last thing before the answer", () => {
  // Part 5 is the last call before the answer at 6.
  assert.equal(endsFoldedSpan(twoRounds, 5), true);
  assert.equal(endsFoldedSpan(twoRounds, 3), false);
  assert.equal(endsFoldedSpan(twoRounds, 4), false);
  assert.equal(endsFoldedSpan(twoRounds, 7), false);
  // A lone block with the answer right after it closes itself.
  assert.equal(endsFoldedSpan(parts(["reasoning", "text"]), 0), true);
  // Nothing follows, nothing to separate from.
  assert.equal(endsFoldedSpan(parts(["reasoning", "tool-call"]), 1), false);
  const reasoning = readSrc("components/assistant-ui/reasoning.tsx");
  const toolGroup = readSrc("components/assistant-ui/tool-group.tsx");
  // Every candidate for last place renders the rule: the lead, a folded round, a tool run.
  assert.equal(
    reasoning.match(/closesTrace && <ReasoningEndRule \/>/g)?.length,
    2,
  );
  assert.match(
    toolGroup,
    /closesTrace && \(\n\s*<div\n\s*data-slot="reasoning-end-rule"/,
  );
  // Without the fold, the lead asks whether the answer is next.
  assert.match(
    reasoning,
    /: message\.parts\[endIndex \+ 1\]\?\.type === "text",/,
  );
});

test("opening by hand grows the block downward instead of pinning the bottom", () => {
  // The viewport follows content growth while attached to the bottom. A click that opens a
  // block, a tool group or a tool card first detaches, so the header stays put. A stream still
  // follows: its auto-open is not a click.
  const reasoning = readSrc("components/assistant-ui/reasoning.tsx");
  assert.match(
    reasoning,
    /if \(open && !isReasoningStreaming\) \{\n\s*detachFromBottom\(\);\n\s*\}/,
  );
  for (const file of ["tool-group.tsx", "tool-fallback.tsx"]) {
    const source = readSrc(`components/assistant-ui/${file}`);
    assert.match(
      source,
      /if \(!open\) \{\n\s*lockScroll\(\);\n\s*\} else if \(!messageRunning\) \{\n\s*detachFromBottom\(\);\n\s*\}/,
      file,
    );
  }
  const hook = readSrc(
    "components/assistant-ui/use-intent-aware-autoscroll.tsx",
  );
  assert.match(
    hook,
    /export function useDetachThreadFromBottom\(\): \(\) => void/,
  );
});

test("the header row keeps its height when Copy appears", () => {
  const reasoning = readSrc("components/assistant-ui/reasoning.tsx");
  // A bare span would take the paragraph line-height and push the label down when opened.
  assert.match(
    reasoning,
    /<span className="ml-auto flex items-center leading-none">\n\s*<ReasoningCopyButton/,
  );
});

test("the lead reports the turn's thinking time added up", () => {
  const byStart: Record<number, number | undefined> = { 0: 4, 3: 6 };
  const resolve = (_parts: readonly { type: string }[], start: number) =>
    byStart[start];
  assert.equal(foldedTurnDuration(twoRounds, 0, resolve), 10);
  // A reply saved with only the legacy last-round duration still reports it.
  assert.equal(
    foldedTurnDuration(twoRounds, 0, (_p, start) =>
      start === 3 ? 6 : undefined,
    ),
    6,
  );
  // Nothing known and the caller keeps its own clock.
  assert.equal(
    foldedTurnDuration(twoRounds, 0, () => undefined),
    undefined,
  );
  assert.equal(foldedTurnDuration(parts(["text"]), 0, resolve), undefined);
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
  // A folded round pages a long trace the same way the lead does.
  const foldedRound = reasoning.slice(
    reasoning.indexOf("const FoldedReasoningRound"),
    reasoning.indexOf("const ReasoningGroupBlock"),
  );
  assert.match(foldedRound, /const pages = useReasoningPages\(/);
  assert.match(foldedRound, /<ReasoningBody\n/);
  // Copy on the lead reaches the folded rounds, which have no Copy of their own.
  assert.match(
    reasoning,
    /foldLead \? foldEnd\(message\.parts, endIndex\) - 1 : endIndex/,
  );
  assert.match(reasoning, /endIndex=\{copyEndIndex\}/);
  // The count sits outside the label, so it shows while the block is still working.
  const trigger = reasoning.slice(
    reasoning.indexOf("function ReasoningTrigger"),
    reasoning.indexOf("function ReasoningContent"),
  );
  assert.ok(
    trigger.indexOf("{foldedSummary ? (") >
      trigger.indexOf("</span>\n      {/*"),
    "the count is inside the label and lost while active",
  );
  // The lead keeps working through those rounds, so its clock covers the whole turn.
  assert.match(
    reasoning,
    /if \(foldLead && \(part\?\.type === "reasoning" \|\| isBlankTextPart\(part\)\)\)/,
  );
});

test("a folded run is hidden, not unmounted", () => {
  const toolGroup = readSrc("components/assistant-ui/tool-group.tsx");
  // One wrapper whether the run is exempt or not: an approval arriving or clearing changes
  // visibility only, so the cards keep their state, their scroll and any live output.
  assert.match(
    toolGroup,
    /className=\{cn\(!\(roundOpen \|\| exempt\) && "hidden"\)\}/,
  );
  assert.match(toolGroup, /data-slot="tool-run-under-thinking"/);
  const reasoning = readSrc("components/assistant-ui/reasoning.tsx");
  assert.match(reasoning, /!open && "hidden"/);
});

test("output and approvals are never folded away", () => {
  const toolGroup = readSrc("components/assistant-ui/tool-group.tsx");
  assert.match(
    toolGroup,
    /const underThinking = foldToolActivity && roundKey !== null;\n\s*const exempt = containsUngroupedTool \|\| hasPendingConfirmation;/,
  );
  // With the preference off the tree is exactly what it was, wrapper included.
  assert.match(toolGroup, /if \(!underThinking\) \{\n\s*return group;\n\s*\}/);
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
  // The switch reads as off, and refuses input, exactly when the fold cannot apply.
  assert.match(
    chatTab,
    /checked=\{foldToolActivityIntoThinking && !foldBlockedByAlwaysExpanded\}\n\s*disabled=\{foldBlockedByAlwaysExpanded\}\n\s*onCheckedChange=\{setFoldToolActivityIntoThinking\}/,
  );
  assert.match(
    chatTab,
    /const foldBlockedByAlwaysExpanded = toolVisibility === "expanded";/,
    "the row no longer works out when the fold is unavailable",
  );
  assert.ok(
    SETTINGS_SEARCH_INDEX.chat.includes("settings.chat.tools.foldIntoThinking"),
    "the setting cannot be found by search",
  );
  assert.equal(
    en.settings.chat.tools.foldIntoThinking,
    "Group tool calls under Thinking",
  );
  assert.ok(en.settings.chat.tools.foldIntoThinkingDescription.length > 0);
  assert.ok(en.settings.chat.tools.foldIntoThinkingBlocked.length > 0);
});

test("the two visibility rows replaced the pair of collapse switches", () => {
  const chatTab = readSrc("features/settings/tabs/chat-tab.tsx");
  for (const key of [
    "settings.chat.thinking.visibility",
    "settings.chat.tools.visibility",
  ]) {
    assert.ok(
      SETTINGS_SEARCH_INDEX.chat.includes(key as never),
      `${key} cannot be found by search`,
    );
    assert.ok(
      chatTab.includes(`t("${key}")`),
      `${key} is indexed but not rendered, so search scrolls to nothing`,
    );
  }
  // All three states are offered, and no old switch is left claiming the same job.
  for (const value of ["collapsed", "auto", "expanded"]) {
    assert.ok(
      chatTab.includes(`<SelectItem value="${value}">`),
      `the ${value} option is missing from Display`,
    );
  }
  assert.equal(
    /collapseThinkingByDefault|collapseToolActivityByDefault/.test(chatTab),
    false,
    "an old collapse switch is still wired up alongside its replacement",
  );
});
