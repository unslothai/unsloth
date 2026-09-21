// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isBlankLine,
  plainLineText,
} from "../src/components/assistant-ui/code-fence-window.ts";

import { readSrc } from "./helpers/kit.ts";

/**
 * The fence body that replaced streamdown's `Block`.
 *
 * What is RUN here is the part that decides what characters a line shows, because that is the one
 * thing the window may never change. The React side is a `.tsx` and the runner cannot load JSX, so
 * it is pinned by regexes over source in the style `code-fence-defer.test.ts` already uses -- not
 * because that is good evidence, but because the alternative is no evidence at all.
 */

const DEFER = readSrc("components/assistant-ui/code-fence-defer.tsx");
const MARKDOWN_TEXT = readSrc("components/assistant-ui/markdown-text.tsx");

test("a line shows the same characters inside the window and outside it", () => {
  // The invariant the whole mechanism rests on. Read back out of the tokens, so it holds by
  // construction rather than by two code paths agreeing.
  const line = [
    { content: "const " },
    { content: "x" },
    { content: " = " },
    { content: "1;" },
  ];
  assert.equal(plainLineText(line), "const x = 1;");
});

test("the plain form is the tokens, never a slice of the source", () => {
  // Shiki drops the CR of a CRLF pair from token content. A source slice would put it back, so a
  // line leaving the window would gain a character the highlighted form never showed, and a fence
  // would change its text as the reader scrolled past it.
  assert.equal(plainLineText([{ content: "x = 1" }]), "x = 1");
  assert.ok(
    !/source\.slice|split\(["'`]\\n["'`]\)/.test(
      DEFER.slice(DEFER.indexOf("const FenceLine = memo")),
    ),
    "the line renderer must not reach for the source",
  );
});

test("a blank line is one line tall, not nothing", () => {
  // A <span> holding an empty text node has no line box at all, so a blank line in the middle of a
  // fence would close up and everything below it would move by one line.
  assert.equal(isBlankLine([]), true);
  assert.equal(isBlankLine([{ content: "" }]), true);
  assert.equal(isBlankLine([{ content: " " }]), false);
  assert.equal(isBlankLine([{ content: "" }, { content: "" }]), false);
  assert.ok(
    /if \(isBlankLine\(line\)\) \{\s*return <span className=\{LINE_CLASS\}>\{"\\n"\}<\/span>;/
      .test(DEFER),
    "a blank line must render the newline streamdown renders for it",
  );
});

test("the line component is memoized, which is the whole point of it", () => {
  // `code-plugin.ts` commits a line once and never rebuilds it, so the array identity is already
  // stable. Without `memo` on top of that, a fence growing by one character still reconciles every
  // line it has, which is the cost this change exists to remove.
  assert.ok(
    /const FenceLine = memo\(function FenceLine\(/.test(DEFER),
    "FenceLine must be memoized",
  );
  assert.ok(
    !/memo\(\s*function FenceLine[\s\S]*?\},\s*\(/.test(DEFER),
    "and on the default shallow comparison: a custom comparator here would have to know that a " +
      "committed line is immutable, which is a second place for that to stop being true",
  );
});

test("the fence branch no longer renders streamdown's Block", () => {
  // `Block` maps the whole token array on every render and memoizes on `prev.result ===
  // next.result`, and the plugin returns a fresh object every frame, so the memo never hit.
  const from = MARKDOWN_TEXT.indexOf("function FenceBlock(");
  const to = MARKDOWN_TEXT.indexOf("const StreamdownBlock = memo(", from);
  assert.ok(from > 0 && to > from, "FenceBlock still bounds a branch of its own");
  const body = MARKDOWN_TEXT.slice(from, to);
  // Comment lines stripped: this branch still EXPLAINS `<Block>` at length, and should.
  const code = body.replace(/^\s*(?:\/\/|\/?\*).*$/gm, "");
  assert.ok(
    !/<Block\b/.test(code),
    "the reached fence must render FenceBody, not Block",
  );
  assert.ok(
    /<FenceBody[\s\S]{0,80}language=\{languageToken\}/.test(code),
    "and it must be handed the parsed language token, not the raw info string",
  );
});

test("a streaming open fence is highlighted rather than shown plain", () => {
  // This is the route #10769 measured and #10779 fixed by giving up the colours. `getCodeFence`
  // needs the closing delimiter, so an open fence had no `codeFence` and fell to the bare `Block`.
  assert.ok(
    /if \(props\.isIncomplete\) \{\s*const openFence = markdownBlockFallback\(props\.content\);\s*if \(openFence\.fenced\) \{\s*return \(\s*<StreamingFenceBlock/m
      .test(MARKDOWN_TEXT),
    "an open fence must leave the bare Block route",
  );
  const streaming = MARKDOWN_TEXT.slice(
    MARKDOWN_TEXT.indexOf("function StreamingFenceBlock("),
  );
  assert.ok(
    /<FenceBody[\s\S]{0,200}result=\{tokens\}/.test(streaming),
    "and it must render the tokens, not a plain shell: keeping the highlighting while it streams " +
      "is the difference between this and the change that was rejected",
  );
});

test("a completed fence on a non-``` form keeps the bounded renderer", () => {
  // The other half of the same route. `getCodeFence` accepts exactly three unindented backticks, so
  // a tilde / four-backtick / indented fence that streamed through `StreamingFenceBlock` had no
  // `codeFence` once it completed and fell to the branch below -- streamdown's whole-token `Block`,
  // which remounts every span at the final frame. A stopped reply does the same and never settles.
  assert.ok(
    /const settledFence = props\.isIncomplete \? null : markdownBlockFallback\(props\.content\);\s*if \(settledFence\?\.fenced\)/.test(
      MARKDOWN_TEXT,
    ),
    "a completed fence must be recognised by the CommonMark-complete scanner",
  );
  const settled = MARKDOWN_TEXT.slice(MARKDOWN_TEXT.indexOf("const settledFence ="));
  const branch = settled.slice(0, settled.indexOf("if (props.isIncomplete)"));
  assert.ok(
    /<StreamingFenceBlock language=\{settledFence\.language\} source=\{settledFence\.text\} \/>/.test(
      branch,
    ),
    "and it must render the per-line body rather than Block, or the spans remount",
  );
  // Deliberately NOT `FenceBlock`: that branch owns the reach latch, the action bar and the
  // fence-mode switch, which are wired to `getCodeFence`'s narrower form on purpose.
  assert.ok(
    !/<FenceBlock[\s\S]{0,200}settledFence/.test(MARKDOWN_TEXT),
    "widening FenceBlock would add controls and artifact paths to blocks that do not have them",
  );
});

test("markdownBlockFallback is what recognises the open fence, not getCodeFence", () => {
  // `CODE_FENCE_RE` behind `getCodeFence` accepts exactly three unindented backticks. CommonMark
  // also allows tildes, four or more backticks, and up to three spaces of indent, and a fence this
  // route fails to recognise goes back to the renderer that cannot afford it.
  assert.ok(
    MARKDOWN_TEXT.includes(
      'import { markdownBlockFallback } from "./markdown-block-fallback";',
    ),
    "the streaming route must use the CommonMark-complete scanner",
  );
});

test("the window decision is not reimplemented in the component", () => {
  // Same rule `code-fence-defer.test.ts` holds for the mode table: one place decides, and it is
  // the one a test can execute.
  assert.ok(
    /import \{[\s\S]*?selectLineWindow,[\s\S]*?\} from "\.\/code-fence-window";/.test(DEFER),
    "the component must consume the decision",
  );
  assert.ok(
    !/WINDOW_CAP_LINES\s*=|OVERSCAN_VIEWPORTS\s*=|HYSTERESIS_VIEWPORTS\s*=/.test(DEFER),
    "no copy of the window constants may live here as well",
  );
});

test("the text is never removed from the document, only its colour", () => {
  // The difference between this and virtualization, and the reason find-in-page, select-all, copy
  // and print are untouched. `progressive-mount-controller.ts` rejects unmounting on exactly these
  // grounds, and `index.css` forces `content-visibility: visible` back on for code blocks because
  // WebKit before Safari 26 cannot find-in-page skipped content.
  const body = DEFER.slice(DEFER.indexOf("export const FenceBody = memo("));
  assert.ok(
    /tokens\.map\(\(line, index\) => \(/.test(body),
    "every line must still be rendered; only what is INSIDE a line changes",
  );
  assert.ok(
    !/contentVisibility:\s*"hidden"|display:\s*"none"/.test(body),
    "no line may be hidden from the engine",
  );
});

test("one scroll listener serves every windowed fence on the page", () => {
  // A listener per fence forces layout per fence per scroll, which is how the measurement costs
  // more than the rendering it saves. Same shape `watchScrolling` uses for the reach latch.
  assert.ok(
    /const windowedFences = new Set<\(\) => void>\(\);/.test(DEFER),
    "the registry must be shared",
  );
  assert.ok(
    /document\.addEventListener\("scroll", scheduleRemeasure, \{\s*capture: true,\s*passive: true,\s*\}\);/
      .test(DEFER),
    "capturing, so the nested reasoning pane's scroller is seen too, and passive",
  );
  assert.ok(
    /windowFrame = requestAnimationFrame\(remeasureWindows\);/.test(DEFER),
    "and coalesced into a frame: scroll fires faster than the screen updates",
  );
});

test("the scrolling ancestor is resolved from outside the code block", () => {
  // A code block carries `overflow-x: auto`, which makes `overflow-y` compute to `auto` as well,
  // so walking up from the <code> is one scrollHeight away from rooting the whole calculation
  // inside the fence's own horizontal scroller.
  assert.ok(
    /const scroller = scrollerOf\(outer\);/.test(DEFER),
    "resolve from the fence's outermost element, not from the code element",
  );
});

test("the body reproduces streamdown's language class and incomplete flag", () => {
  // Both were MEASURED missing against the merge base, on the same scene, and neither has a
  // reader in the tree today. That is what makes them worth pinning rather than shrugging at: a
  // published rendering contract with no current consumer is exactly the kind of thing that goes
  // quietly and is found by a user stylesheet months later.
  const body = DEFER.slice(DEFER.indexOf("export const FenceBody = memo("));
  assert.ok(
    /const languageClass = language === null \? null : `language-\$\{language\}`;/.test(DEFER),
    "the first word of the info string, which is what remark-rehype emits",
  );
  assert.ok(
    /className=\{joinClasses\(\s*languageClass,/.test(body),
    "the body div carries it, as streamdown's does",
  );
  assert.ok(
    /<pre className=\{joinClasses\(languageClass, PRE_CLASS\)\}/.test(body),
    "and so does the pre",
  );
  assert.ok(
    /data-incomplete=\{isIncomplete \|\| undefined\}/.test(body),
    "an open fence carries data-incomplete, and a settled one carries no attribute at all",
  );
});

test("a print colours the whole fence, and the window comes back afterwards", () => {
  // MEASURED: a 3,000 line fence printed with 342 spans against the 23,139 the merge base
  // printed. `upgradeEverythingForPrint` makes this argument for a deferred fence already; the
  // line window reintroduced the same defect one level down.
  assert.ok(
    /let printing = false;/.test(DEFER),
    "the print state has to be module-global: a print is a document-wide event",
  );
  assert.ok(
    /window\.addEventListener\("afterprint", \(\) => setPrinting\(false\)\);/.test(DEFER),
    "and it must revert, or one Ctrl+P un-windows every huge fence for the life of the tab. " +
      "Unlike the reach latch this costs nothing to undo, because the tokens are already cached",
  );
  assert.ok(
    /if \(event\.matches\) \{\s*upgradeEverythingForPrint\(\);\s*setPrinting\(true\);\s*\} else \{\s*setPrinting\(false\);\s*\}/
      .test(DEFER),
    "both doors: beforeprint for Ctrl+P, the media query for page.pdf() and devtools emulation",
  );
  assert.ok(
    /flushSync\(remeasureWindows\);/.test(DEFER),
    "synchronously, because there is no next paint before the print snapshot",
  );
  assert.ok(
    /if \(printing\) \{\s*if \(current\.current === null\) return;/.test(DEFER),
    "and the measurement has to honour it",
  );
});
