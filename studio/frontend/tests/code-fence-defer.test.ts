// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

/**
 * The one property that makes this change different from the viewport gate that came before it,
 * pinned as source facts.
 *
 * The earlier attempt gated a fence on viewport entry AND on viewport exit. Because the gate ran
 * both ways, collapsing a reasoning pane pushed fences out of view and generated re-highlight
 * work instead of saving it: predicted -55% on `reasoning_toggle.close_ms`, measured +12.7%
 * slower, and closed on that number.
 *
 * Everything below exists so that reintroducing a downgrade edge fails a test rather than a
 * benchmark two days later. These are deliberately source-level assertions: the module is a React
 * hook over IntersectionObserver, so a behavioural test would need a DOM, and the invariant worth
 * protecting is structural anyway -- "no code path sets this back to false".
 */

const SOURCE = readFileSync(
  new URL("../src/components/assistant-ui/code-fence-defer.tsx", import.meta.url),
  "utf8",
);

const MARKDOWN_TEXT = readFileSync(
  new URL("../src/components/assistant-ui/markdown-text.tsx", import.meta.url),
  "utf8",
);

test("the latch is only ever set to true", () => {
  const writes = SOURCE.match(/setLatched\([^)]*\)/g) ?? [];
  assert.ok(writes.length > 0, "expected at least one write to the latch");
  for (const write of writes) {
    assert.equal(
      write,
      "setLatched(true)",
      `the latch must never be cleared; found ${write}. A downgrade edge is what made the ` +
        "previous viewport gate measure slower than doing nothing.",
    );
  }
});

test("nothing clears the print latch, and nothing listens for afterprint", () => {
  assert.ok(SOURCE.includes("printed = true"), "the print latch must be settable");
  assert.ok(
    !/printed\s*=\s*false/.test(SOURCE.replace("let printed = false;", "")),
    "the print latch must never be cleared after it is set",
  );
  // The word appears in the comment that explains why there is no listener, so the assertion is
  // on the registration rather than on the string.
  assert.ok(
    !/addEventListener\(\s*["']afterprint/.test(SOURCE),
    "reverting on afterprint would reintroduce the bidirectional edge",
  );
});

test("the observer disconnects itself on the upgrade", () => {
  const callback = SOURCE.slice(
    SOURCE.indexOf("new IntersectionObserver"),
    SOURCE.indexOf("observer.observe(node)"),
  );
  assert.ok(
    callback.indexOf("observer.disconnect()") < callback.indexOf("setLatched(true)"),
    "the observer must disconnect before the state write, so an upgraded fence carries no " +
      "residual per-scroll cost",
  );
});

test("the flag defaults off, and off means today's behaviour", () => {
  assert.ok(
    SOURCE.includes('return raw === "1" || raw === "defer"'),
    "the mode is decided from the flag value",
  );
  assert.ok(
    /:\s*"off";/.test(SOURCE),
    "any value that is not an understood mode must fall through to off",
  );
  assert.ok(
    MARKDOWN_TEXT.includes('const immediate = mode === "off" || Boolean(isIncomplete)'),
    "with the flag off every fence must render immediately, exactly as it does today",
  );
});

test("a streaming fence never defers", () => {
  assert.ok(
    MARKDOWN_TEXT.includes("Boolean(isIncomplete)"),
    "an incomplete (streaming) fence must be immediate: deferring it would change what " +
      "streaming renders rather than what a settled thread costs",
  );
});

test("the shell carries the same streamdown hooks the real block does", () => {
  for (const attribute of [
    'data-streamdown="code-block"',
    'data-streamdown="code-block-header"',
    'data-streamdown="code-block-body"',
  ]) {
    assert.ok(
      SOURCE.includes(attribute),
      `the shell must carry ${attribute} or the stylesheet rules that size a code block do ` +
        "not apply to it and the two arms lay out differently",
    );
  }
});

test("the shell trims trailing newlines the way streamdown does", () => {
  const trim = (text: string): string => {
    let end = text.length;
    while (end > 0 && text[end - 1] === "\n") end -= 1;
    return text.slice(0, end);
  };
  assert.equal(trim("a\nb\n\n\n"), "a\nb");
  assert.equal(trim("a\nb"), "a\nb");
  assert.equal(trim("\n\n"), "");
  assert.ok(
    SOURCE.includes("trimTrailingNewlines"),
    "an untrimmed shell is one blank line taller than the block it stands in for",
  );
});

test("the gate does not mount a wrapper element of its own", () => {
  assert.ok(
    !SOURCE.includes("<div ref={host}>"),
    "an extra div between a list item and its code block breaks the direct-child selector in " +
      "index.css and pushes the block a level deeper than the :last-child margin chain walks",
  );
  assert.ok(
    MARKDOWN_TEXT.includes('<div className="relative isolate" ref={host}>'),
    "the intersection target must be the wrapper markdown-text already rendered",
  );
});

test("the tokenize arm is measurement only and is not reachable from a boolean flag", () => {
  assert.ok(
    SOURCE.includes('raw === "tokenize"'),
    "the tokenize arm is selected by an explicit string, never by a truthy flag",
  );
  assert.ok(
    MARKDOWN_TEXT.includes('const pretokenize = mode === "tokenize" && !reached'),
    "pretokenizing must be confined to the tokenize arm",
  );
});
