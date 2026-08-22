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





test("a completing stream cannot downgrade a fence that was highlighted while it streamed", () => {
  // `streaming` goes true -> FALSE at the closing delimiter. Deriving `reached` from it alone
  // hands a finished fence back the plain shell, which is the reverse edge in miniature.
  assert.ok(
    /if\s*\(!enabled\s*\|\|\s*latched\s*\|\|\s*!streaming\)\s*return;/.test(SOURCE),
    "a streaming fence must LATCH, not merely read as reached while the flag is live",
  );
  const derived = SOURCE.match(/const reached = [^;]+;/)?.[0] ?? "";
  assert.ok(
    derived.includes("latched"),
    `the derived value must include the latch; found ${derived}`,
  );
});

test("the observer is rooted at the nearest SCROLLING ancestor, found not named", () => {
  // Two failures this pins, and they are different from each other.
  //
  // `root: null` is the document viewport, so `rootMargin` expands a rectangle that is not the
  // one clipping and the lookahead is worth nothing. That was the review item.
  //
  // Matching two known selectors walks past the reasoning pane, which while streaming is an
  // `overflow-y-auto` `max-h-64` window holding an arbitrarily long trace. Intersection was still
  // correct there, because intermediate scrollers clip, but the one-viewport lookahead was not:
  // measured 3 of 10 fences intersecting with and without the margin when rooted at the thread
  // viewport, against 5 of 10 rooted at the 256px pane.
  assert.ok(
    /root:\s*scrollerOf\(node\)/.test(SOURCE),
    "the observer root must be the fence's own scrolling ancestor",
  );
  assert.ok(
    !/closest<HTMLElement>\("\[data-slot='thread-viewport'\]"\)/.test(SOURCE),
    "a named-selector lookup walks past the reasoning pane's scroller, which matches neither name",
  );
  const fn = SOURCE.slice(SOURCE.indexOf("const scrollerOf"), SOURCE.indexOf("const scrollerOf") + 320);
  assert.ok(
    fn.includes("el.parentElement") && fn.includes("isScrollable(el)"),
    "it must WALK to the nearest scrollable ancestor rather than matching known names",
  );
  const pred = SOURCE.slice(SOURCE.indexOf("const isScrollable"), SOURCE.indexOf("const scrollerOf"));
  for (const token of ['"auto"', '"scroll"', '"overlay"', "scrollHeight > el.clientHeight"]) {
    assert.ok(pred.includes(token), `the scrollable test must consider ${token}`);
  }
});

test("with the flag off the hook writes no state, builds no observer and reads no layout", () => {
  const hook = SOURCE.slice(SOURCE.indexOf("export function useFenceReached"));
  for (const guard of ["if (!enabled || latched || !streaming) return;", "if (reached) return;"]) {
    assert.ok(hook.includes(guard), `expected the early return ${guard}`);
  }
  assert.ok(
    /const reached = !enabled \|\|/.test(hook),
    "the disabled path must short-circuit to reached, so every effect below takes its early return",
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
    MARKDOWN_TEXT.includes('useFenceReached(host, mode !== "off", Boolean(isIncomplete))'),
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

test("there is no print machinery left to leak", () => {
  // A `beforeprint` path that latched every mounted fence, warmed grammars at idle and flushed
  // twice was removed after it was measured: synchronous dispatch and read in one task, 53 of 56
  // blocks still printed on streamdown's raw fallback at every delay out to twenty seconds. The
  // shipped build is fully highlighted after about three seconds. Neither a warm grammar nor a
  // second flushSync makes an effect-driven, lazily imported highlighter produce tokens inside
  // the synchronous print task.
  //
  // Three review rounds went on leaks inside that machinery. This test exists so it does not come
  // back without a measurement showing it works.
  for (const gone of ["flushSync", "printListeners", "claimChunkWarm", "beforeprint"]) {
    assert.ok(
      !new RegExp(`(addEventListener|const|let)[^\\n]*${gone}`).test(SOURCE),
      `${gone} was removed because it did not work; re-adding it needs a measurement first`,
    );
  }
  assert.ok(
    !/warmedGrammars|pendingGrammars/.test(MARKDOWN_TEXT),
    "the grammar warm-up was removed with the rest of the print path",
  );
});

test("the print limitation is written down where it is decided", () => {
  assert.ok(
    SOURCE.includes("PRINT: NOT HANDLED") && SOURCE.includes("53 of 56"),
    "the measured limitation belongs next to the code that causes it, with its number",
  );
});
