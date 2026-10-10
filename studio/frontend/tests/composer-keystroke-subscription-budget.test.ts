// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every useAuiState selector runs per keystroke, so per-block subscriptions scale with the
// thread. These source-shape checks pin the two seams.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const markdown = readSrc("components/assistant-ui/markdown-text.tsx");
const thread = readSrc("components/assistant-ui/thread.tsx");

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
  assert.doesNotMatch(
    block,
    /useAuiState\(/,
    "a markdown block subscribes to the assistant store, so every block pays per keystroke",
  );
});

test("the render_html scan happens once per message part, above the blocks", () => {
  const impl = body(markdown, "const MarkdownTextImpl = () => {", "\n};");
  const renderer = body(
    markdown,
    "function MarkdownTextRenderer({",
    "\nconst MarkdownTextImpl",
  );
  assert.match(
    impl,
    /useAuiState\(\(\{ message \}\) =>\s*partsHaveRenderableRenderHtmlTool\(message\.parts\),?\s*\)/,
  );
  assert.match(
    renderer,
    /<RenderHtmlToolPresenceContext\.Provider\s+value=\{messageHasRenderableRenderHtmlTool\}/,
  );
});

test("the continue bar subscribes once on a message that is not the newest", () => {
  const gate = body(thread, "const ContinueMessageBar: FC = () => {", "\n};");
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
  const full = body(
    thread,
    "const ContinueMessageBarForLastMessage: FC = () => {",
    "\n};",
  );
  assert.match(full, /useContinuation\(\);/);
  const shared = body(thread, "function useContinuation() {", "\n}\n");
  for (const marker of [
    /useAuiState\(\(\{ message \}\) => message\.status\)/,
    /useAuiState\(\(\{ message \}\) => message\.metadata\)/,
    /readContinuationSource\(message\.content\)/,
    /isContinuableContent\(messageContent, \{[\s\S]{0,160}thought: thoughtResumable,[\s\S]{0,160}replay: geminiReplayTurns\.length > 0/,
    /findLatestUserAudioBase64\(thread\.messages, false\)/,
    /modeAllowsContinuation\(\{/,
  ]) {
    assert.match(shared, marker);
  }
});
