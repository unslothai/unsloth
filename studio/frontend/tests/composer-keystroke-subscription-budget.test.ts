// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Typing one character in the composer costs work proportional to the whole thread, none of it
// rendering.
//
// assistant-ui has ONE notification manager. `useAuiState` subscribes to it through
// `useSyncExternalStore` with the selector AS getSnapshot, so a single store write runs every
// selector in the tree, and writing a character calls `composer.setText`. Measured on a synthetic
// heavy thread with the manager instrumented to count live subscriptions and selector runs:
//
//     20 messages   955 subscriptions   1,020 selector runs per keystroke    1.8ms
//     80 messages  3,726 subscriptions  3,791 selector runs per keystroke    7.2ms
//    220 messages 10,193 subscriptions 10,258 selector runs per keystroke   19.6ms
//
// Layout and style are flat across the same range, so the fan-out, not the DOM, is what grows.
//
// The two seams below stop a subscription being minted per markdown BLOCK and per non-newest
// message. Neither shows in the output -- undo either and the thread renders identically and
// every other test passes -- so, like chat-autoscroll-frame-budget.test.ts and
// drag-costs-no-render.test.ts, the wiring is pinned at the source. The counts above are the
// motivation, not an assertion: a regression in them alone would not fail these source-shape
// checks.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function source(path: string): string {
  return readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
}

const markdown = source("components/assistant-ui/markdown-text.tsx");
const thread = source("components/assistant-ui/thread.tsx");

/** The body of the named function or component declaration, up to its closing brace. */
function body(text: string, start: string, terminator = "\n}"): string {
  const index = text.indexOf(start);
  assert.notEqual(index, -1, `${start} is gone; this test needs rewriting`);
  const rest = text.slice(index + start.length);
  const end = rest.indexOf(terminator);
  assert.notEqual(end, -1, `${start} has no closing brace`);
  return rest.slice(0, end);
}

test("a markdown block reads the render_html presence from context, not the store", () => {
  const block = body(
    markdown,
    "function StreamdownBlockContent(props: BlockProps) {",
  );
  assert.match(block, /useContext\(\s*RenderHtmlToolPresenceContext,?\s*\)/);
  // Mounted once per markdown block: 800 of the 10,193 subscriptions at 300K characters, each
  // re-scanning message.parts per keystroke.
  assert.doesNotMatch(
    block,
    /useAuiState\(/,
    "a markdown block subscribes to the assistant store, so every block pays per keystroke",
  );
});

test("the render_html scan happens once per message part, above the blocks", () => {
  const impl = body(markdown, "const MarkdownTextImpl = () => {", "\n};");
  assert.match(
    impl,
    /useAuiState\(\(\{ message \}\) =>\s*message\.parts\.some\(isRenderableRenderHtmlToolPart\),?\s*\)/,
  );
  // The value has to reach the blocks, or the context read above answers with its default.
  assert.match(
    impl,
    /<RenderHtmlToolPresenceContext\.Provider\s+value=\{messageHasRenderableRenderHtmlTool\}/,
  );
});

test("the continue bar subscribes once on a message that is not the newest", () => {
  const gate = body(thread, "const ContinueMessageBar: FC = () => {", "\n};");
  // Exactly one subscription before the gate, and it is the gate's own condition.
  const subscriptions = gate.match(/useAuiState\(/g) ?? [];
  assert.equal(
    subscriptions.length,
    1,
    "the continue bar subscribes more than once before it knows the message is the newest",
  );
  assert.match(gate, /useAuiState\(\(\{ message \}\) => message\.isLast\)/);
  assert.match(gate, /if \(!isLast\) \{\s*return null;\s*\}/);
  assert.match(gate, /<ContinueMessageBarForLastMessage \/>/);
});

test("the composer asks the thread-wide research question through the cache", () => {
  // The composer's own subscription walked every message per keystroke. The answer's semantics
  // live in thread-research-presence.test.ts; this pins the composer as the caller, since an
  // orphaned helper would leave the scan where it was.
  assert.match(
    thread,
    /state\.latestRunByThreadId\[researchThreadId\]/,
    "the live run must override stale assistant-message status after retry or stop",
  );
  assert.match(
    thread,
    /useAuiState\(\(\{ thread \}\) =>\s*threadHasResearchMessage\(thread\.messages, liveResearchRunId\),?\s*\)/,
  );
  assert.doesNotMatch(
    thread,
    /useAuiState\(\(\{ thread \}\) =>\s*thread\.messages\.some\(/,
    "the composer scans every message inside a selector again",
  );
});

test("the newest message still gets the whole continue bar", () => {
  // The gate must delegate rather than have deleted the work: the bar Max Tokens, Stop and a
  // dropped stream rely on is the one below it. The shared availability conditions now live in
  // useContinuationAvailability, reused by both this bar and the "Save & Continue" edit action,
  // instead of being duplicated here.
  const full = body(
    thread,
    "const ContinueMessageBarForLastMessage: FC = () => {",
    "\n};",
  );
  for (const marker of [
    /useAuiState\(\(\{ message \}\) => message\.status\)/,
    /useAuiState\(\(\{ message \}\) => message\.metadata\)/,
    /assistantMessageText\(message\.content\)/,
    /useContinuationAvailability\(messageContent\)/,
  ]) {
    assert.match(full, marker);
  }

  const availability = body(
    thread,
    "function useContinuationAvailability(",
    "\n}",
  );
  for (const marker of [
    /isContinuableContent\(content\)/,
    /findLatestUserAudioBase64\(thread\.messages, false\)/,
    /modeAllowsContinuation\(\{/,
  ]) {
    assert.match(availability, marker);
  }
});

test("Save & Continue only subscribes while the edit textarea is open", () => {
  // useContinuationAvailability's useAuiState subscriptions are the expensive part; this pins
  // them to a component mounted only for the message being edited, not to AssistantMessage,
  // which renders once per message in the thread.
  const editButton = body(
    thread,
    "const SaveAndContinueButton: FC<{",
    "\n};",
  );
  assert.match(editButton, /useContinuationAvailability\(messageContent\)/);

  const assistantMessage = body(thread, "const AssistantMessage: FC = () => {", "\n};");
  assert.doesNotMatch(
    assistantMessage,
    /useContinuationAvailability\(/,
    "AssistantMessage calls useContinuationAvailability directly, so it runs on every assistant message again",
  );
});
