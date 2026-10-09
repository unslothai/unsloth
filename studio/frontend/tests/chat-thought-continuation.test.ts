// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** GGUF continuation seeds preserve reasoning and join it with new streamed deltas. */

import assert from "node:assert/strict";
import test from "node:test";

import {
  continuationSeed,
  isContinuableContent,
  readContinuationRequest,
  readContinuationSource,
} from "../src/features/chat/utils/continuation.ts";
import { parseAssistantContent } from "../src/features/chat/utils/parse-assistant-content.ts";
import { createReasoningDurationTracker } from "../src/features/chat/utils/reasoning-duration.ts";

import { readSrc } from "./helpers/kit.ts";

const THOUGHT = "The user wants three primes. Small ones are";
const COLLECT_GEMINI_THOUGHTS =
  /collectGeminiThoughtReplayParts\(messageContent\)/;
const CARRY_GEMINI_THOUGHTS =
  /thoughtParts\.length > 0 \? \{ thoughtParts \} : \{\}/;
const REPLAY_GEMINI_THOUGHTS =
  /continuation\.thoughtParts && continuation\.thoughtParts\.length > 0/;
const SEED_GEMINI_THOUGHTS =
  /continuation\?\.thoughtParts\?\.map\(\(part\) => \(\{[\s\S]*afterToolCalls: 0/;

test("the source splits a reply into the answer and the thought before it", () => {
  assert.deepEqual(
    readContinuationSource([
      { type: "reasoning", text: THOUGHT },
      { type: "text", text: "2, 3" },
      { type: "text", text: " and" },
    ]),
    { partial: "2, 3 and", reasoning: THOUGHT },
  );
  assert.deepEqual(readContinuationSource([{ type: "reasoning", text: THOUGHT }]), {
    partial: "",
    reasoning: THOUGHT,
  });
  // A thought after the answer cannot be replayed before it, so it is not carried.
  assert.deepEqual(
    readContinuationSource([
      { type: "text", text: "2, 3" },
      { type: "reasoning", text: "Maybe 5 too." },
      { type: "text", text: " and 5" },
    ]),
    { partial: "2, 3 and 5", reasoning: "" },
  );
  assert.deepEqual(readContinuationSource(undefined), { partial: "", reasoning: "" });
});

test("a thought-only turn is continuable only where a thought resumes", () => {
  const cut = [{ type: "reasoning", text: THOUGHT }];
  assert.equal(isContinuableContent(cut), false);
  assert.equal(isContinuableContent(cut, { thought: true }), true);
  assert.equal(
    isContinuableContent([{ type: "reasoning", text: "  " }], { thought: true }),
    false,
  );
  // A tool call still blocks it: the sibling would replay the call without its result.
  assert.equal(
    isContinuableContent([...cut, { type: "tool-call", toolName: "web_search" }], {
      thought: true,
    }),
    false,
  );
});

test("a thought-only request is read, and its duration only travels with a thought", () => {
  assert.deepEqual(
    readContinuationRequest({
      custom: {
        unslothContinuation: {
          partial: "",
          reasoning: THOUGHT,
          reasoningDuration: 7,
          thoughtParts: [
            { text: "first thought", thoughtSignature: "SIG-THOUGHT-1" },
            { text: "second thought", thoughtSignature: "SIG-THOUGHT-2" },
          ],
        },
      },
    }),
    {
      partial: "",
      reasoning: THOUGHT,
      reasoningDuration: 7,
      thoughtParts: [
        { text: "first thought", thoughtSignature: "SIG-THOUGHT-1" },
        { text: "second thought", thoughtSignature: "SIG-THOUGHT-2" },
      ],
    },
  );
  assert.deepEqual(
    readContinuationRequest({
      custom: { unslothContinuation: { partial: "2, 3", reasoningDuration: 7 } },
    }),
    { partial: "2, 3" },
  );
  assert.equal(
    readContinuationRequest({
      custom: { unslothContinuation: { partial: "", reasoning: " \n" } },
    }),
    null,
  );
});

test("Gemini signed thought parts travel through a continuation sibling", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(thread, COLLECT_GEMINI_THOUGHTS);
  assert.match(thread, CARRY_GEMINI_THOUGHTS);
  assert.match(adapter, REPLAY_GEMINI_THOUGHTS);
  assert.match(adapter, SEED_GEMINI_THOUGHTS);
});

/** What the adapter's stream loop appends, per delta, to a seeded buffer. */
function stream(
  seed: string,
  openAtStart: boolean,
  deltas: Array<{ reasoning?: string; content?: string }>,
): string {
  let text = seed;
  let open = openAtStart;
  for (const { reasoning, content } of deltas) {
    if (reasoning) {
      text += open ? reasoning : `<think>${reasoning}`;
      open = true;
    }
    if (content) {
      if (open) {
        text += "</think>";
        open = false;
      }
      text += content;
    }
  }
  return open ? `${text}</think>` : text;
}

test("a resumed thought and its answer render as one thought and one answer", () => {
  const seed = continuationSeed("", THOUGHT);
  assert.equal(seed, `<think>${THOUGHT}`);
  // Exactly what llama-server streamed for this request on b11160 (Qwen3.5-4B).
  const final = stream(seed, true, [
    { reasoning: " 2, 3, 5." },
    { content: "2, 3 and 5." },
  ]);
  assert.deepEqual(parseAssistantContent(final), [
    { type: "reasoning", text: `${THOUGHT} 2, 3, 5.` },
    { type: "text", text: "2, 3 and 5." },
  ]);
});

test("a continued answer keeps the thought it followed", () => {
  const seed = continuationSeed("2, 3", "Easy.");
  assert.deepEqual(parseAssistantContent(seed), [
    { type: "reasoning", text: "Easy." },
    { type: "text", text: "2, 3" },
  ]);
  assert.deepEqual(parseAssistantContent(stream(seed, false, [{ content: " and 5." }])), [
    { type: "reasoning", text: "Easy." },
    { type: "text", text: "2, 3 and 5." },
  ]);
  // Without a thought the seed is the partial, byte for byte, as before.
  assert.equal(continuationSeed("2, 3 ", ""), "2, 3 ");
});

function clock(start = 1_000_000) {
  let now = start;
  return {
    now: () => now,
    advance: (seconds: number) => {
      now += seconds * 1000;
    },
  };
}

test("a resumed thought keeps counting from the time it already took", () => {
  const time = clock();
  const tracker = createReasoningDurationTracker(time.now);
  tracker.seedThought({ duration: 12, open: true, textLength: THOUGHT.length });
  time.advance(3);
  tracker.finishGroup();
  assert.deepEqual(tracker.metadata(), { reasoningDuration: 15, reasoningDurations: [15] });
});

test("a server summary of the resumed tail does not replace the whole thought's time", () => {
  const time = clock();
  const tracker = createReasoningDurationTracker(time.now);
  tracker.seedThought({ duration: 12, open: true, textLength: THOUGHT.length });
  time.advance(3);
  tracker.recordServerDuration(3000);
  tracker.finishGroup();
  assert.equal(tracker.metadata().reasoningDuration, 15);
});

test("a closed thought keeps its duration while the answer grows", () => {
  const time = clock();
  const tracker = createReasoningDurationTracker(time.now);
  tracker.seedThought({ duration: 12, open: false, textLength: 5 });
  time.advance(4);
  // The adapter re-offers the group on every chunk; unchanged text must not reopen it.
  tracker.resumeGroup(0, 5);
  tracker.finishGroup();
  assert.deepEqual(tracker.metadata(), { reasoningDuration: 12, reasoningDurations: [12] });
  // A new thought after it is a group of its own.
  tracker.startGroup();
  time.advance(2);
  tracker.finishGroup();
  assert.deepEqual(tracker.metadata().reasoningDurations, [12, 2]);
});

test("a closed thought with no recorded time stays untimed", () => {
  const time = clock();
  const tracker = createReasoningDurationTracker(time.now);
  tracker.seedThought({ duration: undefined, open: false, textLength: 5 });
  time.advance(4);
  tracker.resumeGroup(0, 5);
  tracker.finishGroup();
  assert.deepEqual(tracker.metadata(), {});
});

test("Continue response leads the More menu and yields to the Resume bar", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  // Continue response is the More menu's first item, before Edit response.
  assert.match(
    thread,
    /<ContinueResponseMenuItem \/>\n\s*\{!inlineEdit && <EditAssistantMessageMenuItem \/>\}/,
  );
  // Bar order: Edit, Fork, Read aloud (or Stop reading), Refresh.
  assert.match(
    thread,
    /\{inlineEdit && <EditAssistantMessageButton \/>\}\n\s*<ForkCountBadge \/>\n\s*<ForkMessageButton \/>\n\s*\{ttsEnabled && \(\n\s*<MessagePrimitive\.If speaking=\{false\}>\n\s*<ActionBarPrimitive\.Speak[\s\S]*?<\/ActionBarPrimitive\.StopSpeaking>\n\s*<\/MessagePrimitive\.If>\n\s*\{!researchRunId && !researchActive && \(\n\s*<ActionBarPrimitive\.Reload/,
  );
  // Delete is the More menu's last item, not a bar button.
  assert.match(thread, /<DeleteMessageMenuItem \/>\n\s*<\/div>\n\s*<\/ActionBarMorePrimitive\.Content>/);
  // A stopped reply already offers Resume; a reply being edited continues from the saved text;
  // a cited reply would lose its sources in the sibling.
  assert.match(
    thread,
    /if \(!completed \|\| reason \|\| !canResume \|\| editing \|\| cited\) \{\n\s*return null;/,
  );
  // Both controls start the same run.
  assert.equal(thread.match(/\[CONTINUATION_RUN_CONFIG_KEY\]/g)?.length, 1);
});
