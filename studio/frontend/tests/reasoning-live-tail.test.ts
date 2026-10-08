// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The live Thinking box (#11703): bounded only while the `auto` setting holds a block open for
// its stream, lifted by a hand open, and never an inner scroller (#11704 review).

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

test("only a block the auto setting opened for its stream is bounded", () => {
  for (const visibility of DISPLAY_VISIBILITIES) {
    for (const override of [null, false, true]) {
      for (const isStreaming of [false, true]) {
        const open = resolveReasoningOpen({
          isStreaming,
          visibility,
          override,
        });
        const capped = reasoningTailCapped({ visibility, override });
        const shown = open && capped;
        assert.equal(
          shown,
          visibility === "auto" && override === null && isStreaming,
          `${visibility} override=${override} streaming=${isStreaming}`,
        );
      }
    }
  }
});

test("the box clips at the top instead of scrolling, and Show full is a hand open", () => {
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
    /const capped = !foldLead && reasoningTailCapped\(\{ visibility, override \}\);/,
  );
  assert.match(
    reasoning,
    /<ReasoningTail\s+capped=\{capped\}\s+onShowFull=\{\(\) => handleOpenChange\(true\)\}/,
  );
  assert.match(reasoning, /bounded=\{capped\}/);
});

test("a bounded transcript corrects nothing through the thread and mounts the tail", () => {
  assert.match(transcript, /!bounded &&\s*virtualizer\.scrollElement !== null/);
  assert.match(transcript, /if \(passage && !bounded\)\s*adjustAbove/);
  assert.match(transcript, /if \(readingAnchor && !bounded\)/);
  assert.match(
    transcript,
    /if \(bounded\) \{[\s\S]*?fragments\.length - 1[\s\S]*?BOUNDED_TAIL_PX[\s\S]*?mounted\.add\(i\)/,
  );
  assert.match(transcript, /\}, \[viewport, virtualizer, bounded\]\);/);
});
