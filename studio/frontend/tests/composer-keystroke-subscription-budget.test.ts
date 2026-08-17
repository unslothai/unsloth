// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Typing one character in the composer costs work proportional to the whole thread, and none of
// it is rendering.
//
// assistant-ui gives the client ONE notification manager. `useAuiState` subscribes every caller
// to it through `useSyncExternalStore`, and the selector IS the getSnapshot, so a single store
// write runs every selector in the tree. Writing one character calls `composer.setText`, so a
// keystroke pays one pass over every subscription the thread holds. Measured on
// tests/studio/playwright_heavy_thread.py's fixture, instrumented at the notification manager:
//
//     20 messages   955 subscriptions   1,020 selector runs per keystroke    1.8ms
//     80 messages  3,726 subscriptions  3,791 selector runs per keystroke    7.2ms
//    220 messages 10,193 subscriptions 10,258 selector runs per keystroke   19.6ms
//
// Layout and style recalculation are flat across the same range (1.8ms -> 2.1ms layout, 1.7ms ->
// 1.6ms style), so this fan-out, not the DOM, is what grows.
//
// The two seams below are what stops a subscription being minted per markdown BLOCK and per
// non-newest message. Neither is visible in the rendered output: undo either and the thread
// still renders identically, every other test here still passes, and the keystroke goes back to
// carrying the extra subscriptions. So, as thread-delete-render-budget.test.ts and
// chat-autoscroll-frame-budget.test.ts do, the wiring is pinned at the source.

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
  const block = body(markdown, "function StreamdownBlockContent(props: BlockProps) {");
  assert.match(block, /useContext\(\s*RenderHtmlToolPresenceContext,?\s*\)/);
  // The block component is mounted once per markdown block: 800 of the 10,193 subscriptions at
  // 300K characters came from this one call, and each re-scanned message.parts per keystroke.
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
  // The one subscription the composer itself holds walked every message on every keystroke.
  // thread-research-presence.test.ts covers what the answer is; this covers that the composer is
  // the caller, since an orphaned helper would leave the scan exactly where it was.
  assert.match(
    thread,
    /useAuiState\(\(\{ thread \}\) =>\s*threadHasResearchMessage\(thread\.messages\),?\s*\)/,
  );
  assert.doesNotMatch(
    thread,
    /useAuiState\(\(\{ thread \}\) =>\s*thread\.messages\.some\(/,
    "the composer scans every message inside a selector again",
  );
});

test("the newest message still gets the whole continue bar", () => {
  // The gate must delegate to the full component rather than having deleted its work: the bar
  // that Max Tokens, Stop and a dropped stream rely on is the one below the gate.
  const full = body(
    thread,
    "const ContinueMessageBarForLastMessage: FC = () => {",
    "\n};",
  );
  for (const marker of [
    /useAuiState\(\(\{ message \}\) => message\.status\)/,
    /useAuiState\(\(\{ message \}\) => message\.metadata\)/,
    /assistantMessageText\(message\.content\)/,
    /isContinuableContent\(message\.content\)/,
    /findLatestUserAudioBase64\(thread\.messages, false\)/,
    /modeAllowsContinuation\(\{/,
  ]) {
    assert.match(full, marker);
  }
});
