// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Thinking box (#11703): bounded under the `auto` setting however the block was opened, lifted
// only by its own Expand toggle, and never an inner scroller (#11704 review).

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

test("the box clips at the top instead of scrolling, and Expand toggles without a permanent open", () => {
  const tail = reasoning.slice(
    reasoning.indexOf("function ReasoningTail("),
    reasoning.indexOf("// With the fold preference on"),
  );
  assert.match(
    tail,
    /flex max-h-\[[^\]]+\] flex-col justify-end overflow-hidden/,
  );
  assert.doesNotMatch(tail, /overflow-(y-)?(auto|scroll)|scrollTop|scrollTo\(/);
  assert.match(
    reasoning,
    /const cappedNow = !foldLead && reasoningTailCapped\(\{ visibility \}\);/,
  );
  // The toggle stays mounted in both states, so Collapse is always reachable after Expand.
  assert.match(tail, /\{cappable && overflowing && \(/);
  assert.match(tail, /onClick=\{\(\) => onExpandedChange\(!expanded\)\}/);
  assert.match(tail, /aria-expanded=\{expanded\}/);
  assert.match(tail, /\{expanded \? "Collapse thinking" : "Expand thinking"\}/);
  assert.match(reasoning, /bounded=\{capped && !tailExpanded\}/);
  // A new round starts bounded again.
  assert.match(reasoning, /setOverride\(null\);\s*setTailExpanded\(false\);/);
});

test("focus landing in the clipped lines lifts the cap instead of staying invisible", () => {
  const tail = reasoning.slice(
    reasoning.indexOf("function ReasoningTail("),
    reasoning.indexOf("// With the fold preference on"),
  );
  assert.match(
    tail,
    /onFocus=\{\(event\) => \{[\s\S]*?clipped &&[\s\S]*?event\.target\.getBoundingClientRect\(\)\.top <\s*event\.currentTarget\.getBoundingClientRect\(\)\.top[\s\S]*?onExpandedChange\(true\);/,
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
    /if \(!bounded \|\| covered\) return;[\s\S]*?BOUNDED_TAIL_PX[\s\S]*?end: fragments\.length[\s\S]*?setIslands\(next\);/,
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
