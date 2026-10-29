// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const TRANSCRIPT = readSrc("components/assistant-ui/reasoning-transcript.tsx");

test("thinking code is coloured by the worker: once settled near the screen, and while growing only on screen", () => {
  assert.match(TRANSCRIPT, /\{tokens === null\s*\? codeRowText\(fragment, leading\)\s*: colouredRow\(fragment, leading, tokens\)\}/);
  assert.match(TRANSCRIPT, /if \(column === 0 && line > 0 && !\(leading && at === 0\)\) nodes\.push\("\\n"\);\s*if \(text !== ""\) nodes\.push\(text\);/, "one text node per line and per line break: a streamed append rewrites only the growing line, and a reading anchor never spans a row's line break");
  assert.match(TRANSCRIPT, /useThinkingCodeColour\(\s*surface,\s*code\.source,\s*code\.language,[\s\S]{0,120}!code\.incomplete \|\| !streaming,\s*\)/);
  const hook = readSrc("components/assistant-ui/use-thinking-code-colour.ts");
  assert.match(hook, /export const LIVE_COLOUR_MS = 250;/, "a growing fence is re-coloured a few times a second at most");
  assert.match(hook, /new IntersectionObserver\([\s\S]*rootMargin: "100% 0px"/, "a settled block is asked for once it nears the viewport");
  assert.match(hook, /if \(settled \|\| !visible \|\| caughtUp \|\| state\.busy \|\| state\.off\) return;/, "a growing block off screen, or with a request in flight, asks for nothing");
  assert.match(hook, /if \(source\.length > MAX_HIGHLIGHT_CHARS\) return;/, "a growing block past the highlight cap waits for its close");
  assert.equal((hook.match(/requestFullHighlight\([\s\S]*?,\s*true,\s*\)/g) ?? []).length, 2, "thinking colour is speculative, behind the answer's own fences");
  assert.equal(/highlightFenceSource|code\.highlight|useReasoningHighlight/.test(TRANSCRIPT), false, "no main-thread tokenizer for thinking code");
  assert.match(TRANSCRIPT, /<CodeHighlightModeContext\.Provider value="settled">/, "a fence inside a prose row of the trace follows the same rule");
  assert.match(
    readSrc("components/assistant-ui/reasoning.tsx"),
    /<CodeHighlightModeContext\.Provider value="settled">\s*<MarkdownText \/>/,
    "and so does a fence in a trace too short for the transcript",
  );
});

test("a code group the fence continues past owns the line break after it, in a block that is not a <pre>", () => {
  assert.match(TRANSCRIPT, /const breaksAfter =\s*!last\.last &&\s*fragments\[indices\[indices\.length - 1\] \+ 1\]\?\.code\?\.lines\[0\]\?\.column === 0;/);
  assert.match(TRANSCRIPT, /\{breaksAfter && "\\n"\}\s*<\/code>\s*<\/div>/, "a blank last line needs a break after it to take up its row, and a copy across the seam keeps it");
  assert.equal(TRANSCRIPT.includes('<pre className="!m-0 min-h-[1lh]'), false, "Firefox copies a blank line between two <pre> blocks");
});

test("a code group's block still reads as a passage and takes the chosen code size, as its <pre> did", () => {
  assert.match(
    readSrc("components/assistant-ui/reasoning-reading-anchor.ts"),
    /const passages =\s*"p, pre, \.aui-reasoning-code-lines, li, /,
    "with only thinking code on screen, a width change moved the reader by the whole height the trace above it gained",
  );
  assert.match(
    readSrc("index.css"),
    /html\[data-code-font-size\] :is\(pre, code, kbd, samp\),\nhtml\[data-code-font-size\] \.aui-reasoning-code-lines \{\n\tfont-size: var\(--custom-code-font-size\) !important;/,
    "a smaller code font kept the block's default line spacing",
  );
});
