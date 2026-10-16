// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const TRANSCRIPT = readSrc("components/assistant-ui/reasoning-transcript.tsx");

test("thinking code stays plain while streaming and after completion", () => {
  assert.match(TRANSCRIPT, /\{codeRowText\(fragment, leading\)\}/);
  assert.match(TRANSCRIPT, /if \(column === 0 && line > 0 && !\(leading && at === 0\)\) nodes\.push\("\\n"\);\s*if \(text !== ""\) nodes\.push\(text\);/, "one text node per line and per line break: a streamed append rewrites only the growing line, and a reading anchor never spans a row's line break");
  assert.equal(/useThinkingCodeColour|requestFullHighlight|FenceLine|colouredRow|code\.highlight/.test(TRANSCRIPT), false, "transcript code creates no highlight work or token spans");
  assert.match(TRANSCRIPT, /<CodeHighlightModeContext\.Provider value="plain">/, "a fence inside a prose row of the trace is plain too");
  assert.match(
    readSrc("components/assistant-ui/reasoning.tsx"),
    /<CodeHighlightModeContext\.Provider value="plain">\s*<MarkdownText \/>/,
    "and so is a fence in a trace too short for the transcript",
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
