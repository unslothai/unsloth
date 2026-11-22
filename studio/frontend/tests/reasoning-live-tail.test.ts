// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Thinking box (#11703): bounded under `auto`, scrolling on its own with a bottom fade, no Expand toggle.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { DISPLAY_VISIBILITIES } = await import(
  "../src/features/chat/utils/display-visibility.ts"
);
const { reasoningTailCapped, resolveReasoningOpen } = await import(
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

const tail = reasoning.slice(
  reasoning.indexOf("function ReasoningTail("),
  reasoning.indexOf("// With the fold preference on"),
);

test("the box scrolls on its own with no border, its scrollbar always showing, close to the text", () => {
  assert.match(
    tail,
    /\(capped \|\| expanded\) &&\s*"-ml-\[[^\]]+\] -mr-\[[^\]]+\] mt-2 min-w-0 overflow-x-hidden overflow-y-auto pl-\[[^\]]+\] pr-\[[^\]]+\] pb-3 \[scrollbar-gutter:stable\] \[&>\[data-slot=reasoning-text\]\]:pt-1"/,
  );
  // Only the cap drops on expand: the box keeps its gutter, so the width holds.
  assert.match(tail, /capped && "max-h-\[min\(22rem,48vh\)\]"/);
  assert.doesNotMatch(tail, /hover-scrollbar/);
  assert.doesNotMatch(tail, /\bborder\b|Expand thinking|overflow-y-clip/);
  assert.doesNotMatch(reasoning, /reasoning-show-full/);
  assert.match(reasoning, /const cappedNow = !foldLead && reasoningTailCapped\(\{ visibility \}\);/);
});

test("the header's expand button lifts the cap for the round and appears only when the box overflows", () => {
  assert.match(reasoning, /const showExpand = isOpen && capped && \(tailOverflowing \|\| tailExpanded\);/);
  assert.match(reasoning, /label=\{tailExpanded \? "Collapse thinking" : "Expand thinking"\}/);
  assert.match(reasoning, /icon=\{tailExpanded \? ArrowShrink01Icon : ArrowExpand01Icon\}/);
  assert.match(reasoning, /capped=\{capped && !tailExpanded\}/);
  assert.match(reasoning, /expanded=\{capped && tailExpanded\}/);
  assert.match(reasoning, /bounded=\{capped && !tailExpanded\}/);
  assert.match(reasoning, /setOverride\(null\);\s*setTailExpanded\(false\);/);
  assert.match(tail, /if \(now !== overflowing\.current\) \{\s*overflowing\.current = now;\s*onOverflowRef\.current\(now\);/);
});

test("a streaming box follows its newest line until the reader scrolls up", () => {
  assert.match(tail, /if \(streaming\) following\.current = true;/);
  assert.match(tail, /if \(streaming && following\.current\) box\.scrollTop = box\.scrollHeight;/);
  assert.match(tail, /if \(atBottom \|\| pressedHere \|\| performance\.now\(\) - readerAt < 250\) following\.current = atBottom;/);
  assert.match(reasoning, /const READER_SCROLL_EVENTS = \["wheel", "touchmove", "keydown"\] as const;/);
});

test("a scrollbar drag stops the follow for as long as the press that started in the box is held", () => {
  assert.match(tail, /box\.addEventListener\("pointerdown", onPress, \{ passive: true \}\);/);
  assert.match(tail, /pressedHere &&= inputPointerHeld\(\);/);
});

test("scrolling inside the capped box mounts the passages it reaches", () => {
  assert.match(transcript, /const tail = element\.closest<HTMLElement>\('\[data-slot="reasoning-tail"\]'\);/);
  assert.match(transcript, /tail\?\.addEventListener\("scroll", onTailScroll, \{ passive: true \}\);/);
  assert.match(transcript, /tailFrame = 0;\s*demandNow\.current\(\);/);
});

test("each edge fades only while more is past it, set without a re-render", () => {
  assert.match(tail, /box\.toggleAttribute\("data-more-below", below\(\) > 1\)/);
  assert.match(tail, /box\.toggleAttribute\("data-more-above", box\.scrollTop > 0\)/);
  const css = readSrc("index.css");
  assert.match(css, /\[data-slot="reasoning-tail"\]\[data-capped\]\[data-more-above\] \{\s*--trace-fade-top: 1\.25rem;/);
  assert.match(css, /\[data-slot="reasoning-tail"\]\[data-capped\]\[data-more-below\] \{\s*--trace-fade-bottom: 1\.5rem;/);
  assert.match(css, /linear-gradient\(#000, #000\);\s*mask-size: 100% 100%, calc\(1rem \* var\(--ui-space-scale, 1\)\) 100%;\s*mask-position: 0 0, right;/);
  assert.match(reasoning, /capped \? "mt-2" : "mt-6", "mb-2 border-border border-t"/);
  assert.doesNotMatch(tail, /setState|useState/);
});
