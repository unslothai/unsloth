// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Thinking box (#11703): bounded under `auto`, lifted by its Expand toggle, never a scroller (#11704).

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { DISPLAY_VISIBILITIES } = await import(
  "../src/features/chat/utils/display-visibility.ts"
);
const { reasoningTailCapped, reasoningTailFlexBox, resolveReasoningOpen } = await import(
  "../src/features/chat/utils/reasoning-visibility.ts"
);

const reasoning = readSrc("components/assistant-ui/reasoning.tsx");
const transcript = readSrc("components/assistant-ui/reasoning-transcript.tsx");

test("only the auto setting bounds a block, whether the stream or a hand opened it", () => {
  for (const visibility of DISPLAY_VISIBILITIES) {
    assert.equal(
      reasoningTailCapped({ visibility }),
      visibility === "auto",
      visibility,
    );
  }
});

// Mirrors useTailFlexBox: the first state is `capped`, then each render feeds the last result back.
function flexBoxTrace(steps: readonly { capped: boolean; streaming: boolean }[]): boolean[] {
  let previous = steps[0]?.capped ?? false;
  return steps.map((step) => {
    previous = reasoningTailFlexBox({ ...step, previous });
    return previous;
  });
}

test("the flex box survives Expand/Collapse once finished and never flips when a stream ends", () => {
  const capped = { capped: true, streaming: false };
  const expanded = { capped: false, streaming: false };
  const live = { capped: true, streaming: true };
  const liveExpanded = { capped: false, streaming: true };

  // Hand-opened finished trace: flex the whole way, so neither toggle restyles the trace.
  assert.deepEqual(flexBoxTrace([capped, expanded, capped, expanded]), [true, true, true, true]);
  // Expanded while streaming drops it (block, as before), the stream ending keeps block, and only
  // the next Collapse restores it; after that the toggles are free again.
  assert.deepEqual(
    flexBoxTrace([live, liveExpanded, expanded, capped, expanded]),
    [true, false, false, true, true],
  );
  // A live tail that is never expanded stays flex through the end of the stream.
  assert.deepEqual(flexBoxTrace([live, capped, expanded]), [true, true, true]);
  // A block that first renders already expanded and finished starts without the flex box.
  assert.deepEqual(flexBoxTrace([expanded, expanded]), [false, false]);
  // A new round re-caps, which restores it even after a live Expand.
  assert.deepEqual(flexBoxTrace([live, liveExpanded, live]), [true, false, true]);
});

test("the box clips at the top instead of scrolling, and Expand toggles without a permanent open", () => {
  const tail = reasoning.slice(
    reasoning.indexOf("function ReasoningTail("),
    reasoning.indexOf("// With the fold preference on"),
  );
  assert.match(tail, /capped && "max-h-\[[^\]]+\] overflow-hidden"/);
  // Expand/Collapse on a finished box must not toggle the flex box or the child rule: either
  // restyles every row of the trace.
  assert.match(tail, /cappable && "\[&>\*\]:shrink-0"/);
  assert.match(
    reasoning,
    /const next = reasoningTailFlexBox\(\{ capped, streaming, previous: flexBox \}\);/,
  );
  assert.match(tail, /cappable && flexBox && "flex flex-col"/);
  assert.doesNotMatch(tail, /overflow-(y-)?(auto|scroll)|scrollTop|scrollTo\(/);
  assert.match(
    reasoning,
    /const cappedNow = !foldLead && reasoningTailCapped\(\{ visibility \}\);/,
  );
  assert.match(tail, /\{cappable && overflowing && \(/);
  assert.match(tail, /onClick=\{\(\) => onExpandedChange\(!expanded\)\}/);
  assert.match(tail, /aria-expanded=\{expanded\}/);
  assert.match(tail, /\{expanded \? "Collapse thinking" : "Expand thinking"\}/);
  assert.match(reasoning, /bounded=\{capped && !tailExpanded\}/);
  assert.match(reasoning, /setOverride\(null\);\s*setTailExpanded\(false\);/);
});

test("changing the setting drops a hand Expand along with the hand open", () => {
  assert.match(
    reasoning,
    /if \(lastVisibility !== visibility\) \{\s*setLastVisibility\(visibility\);\s*setOverride\(null\);\s*setTailExpanded\(false\);\s*\}/,
  );
});

test("focus landing in the clipped lines lifts the cap instead of staying invisible", () => {
  const tail = reasoning.slice(
    reasoning.indexOf("function ReasoningTail("),
    reasoning.indexOf("// With the fold preference on"),
  );
  assert.match(
    tail,
    /onFocus=\{\(event\) => \{[\s\S]*?clipped &&\s*\(streaming \? target\.top < box\.top : target\.bottom > box\.bottom\)[\s\S]*?onExpandedChange\(true\);/,
  );
});

test("a bounded transcript corrects nothing through the thread and mounts the tail", () => {
  assert.match(
    transcript,
    /const adjustAbove = useCallback\(\s*\(delta: number\) => !boundedRef\.current && adjustThread\(delta\),/,
    "a bounded transcript still writes thread corrections",
  );
  assert.equal(
    (transcript.match(/adjustThread\(/g) ?? []).length,
    1,
    "every thread correction goes through the bounded guard",
  );
  assert.match(transcript, /boundedRef\.current = bounded;/);
  assert.match(
    transcript,
    /if \(!bounded \|\| !streaming \|\| covered\) return;[\s\S]*?BOUNDED_TAIL_PX[\s\S]*?end: fragments\.length[\s\S]*?setIslands\(next\);/,
    "the newest rows are not mounted while the prefix catches up",
  );
  assert.match(transcript, /recapture\.current\(true\);\s*\}, \[bounded\]\);/);
});

test("a closing block keeps the cap it had while open", () => {
  // Hand-closing a full trace re-capped it mid-collapse and skipped the transcript's corrections.
  assert.match(
    reasoning,
    /if \(isOpen && closingCapped !== cappedNow\) \{\s*setClosingCapped\(cappedNow\);\s*\}\s*const capped = isOpen \? cappedNow : closingCapped;/,
  );
  const open = reasoning.indexOf("const isOpen = resolveReasoningOpen(");
  assert.ok(open !== -1 && open < reasoning.indexOf("const cappedNow ="));
});

test("Expand thinking ends in the header's chevron: right while bounded, down once expanded", () => {
  const tail = reasoning.slice(
    reasoning.indexOf("function ReasoningTail("),
    reasoning.indexOf("// With the fold preference on"),
  );
  assert.match(
    tail,
    /\{expanded \? "Collapse thinking" : "Expand thinking"\}\s*<ChevronDownIcon[\s\S]*?!expanded && "-rotate-90"/,
  );
});

test("a streaming box shows the newest lines, a finished one shows the start", () => {
  const tail = reasoning.slice(
    reasoning.indexOf("function ReasoningTail("),
    reasoning.indexOf("// With the fold preference on"),
  );
  assert.match(tail, /capped && \(streaming \? "justify-end" : "justify-start"\)/);
  assert.match(tail, /to_bottom,transparent[\s\S]*?to_top,transparent/);
  assert.match(reasoning, /streaming=\{isReasoningStreaming\}\s*onExpandedChange=/);
});
