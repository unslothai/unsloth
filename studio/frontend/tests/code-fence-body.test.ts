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
 * The fence body that replaced streamdown's `Block`. What is RUN is the part deciding what
 * characters a line shows; the `.tsx` side is pinned by regex over source, since the runner cannot
 * load JSX.
 */

const DEFER = readSrc("components/assistant-ui/code-fence-defer.tsx");
const MARKDOWN_TEXT = readSrc("components/assistant-ui/markdown-text.tsx");

test("a line shows the same characters inside the window and outside it", () => {
  const line = [
    { content: "const " },
    { content: "x" },
    { content: " = " },
    { content: "1;" },
  ];
  assert.equal(plainLineText(line), "const x = 1;");
});

test("the plain form is the tokens, never a slice of the source", () => {
  // Shiki drops the CR of a CRLF pair, so a source slice would make a line gain a character on
  // leaving the window.
  assert.equal(plainLineText([{ content: "x = 1" }]), "x = 1");
  assert.ok(
    !/source\.slice|split\(["'`]\\n["'`]\)/.test(
      DEFER.slice(DEFER.indexOf("const FenceLine = memo")),
    ),
    "the line renderer must not reach for the source",
  );
});

test("a blank line is one line tall, not nothing", () => {
  // An empty text node has no line box, so the fence would close up and everything below it moves.
  assert.equal(isBlankLine([]), true);
  assert.equal(isBlankLine([{ content: "" }]), true);
  assert.equal(isBlankLine([{ content: " " }]), false);
  assert.equal(isBlankLine([{ content: "" }, { content: "" }]), false);
  assert.ok(
    /if \(!inline && isBlankLine\(line\)\) \{\s*return <span className=\{LINE_CLASS\}>\{"\\n"\}<\/span>;/
      .test(DEFER),
    "a blank line must render the newline streamdown renders for it",
  );
});

test("the line component is memoized, which is the whole point of it", () => {
  // A committed line's identity is already stable, so without `memo` a one-character growth still
  // reconciles every line: the cost this change removes.
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
  // A completed tilde / four-backtick / indented fence used to fall to `Block` and remount every
  // span; a stopped reply takes the same route.
  assert.ok(
    /const settledFence = props\.isIncomplete \? null : markdownBlockFallback\(props\.content\);\s*if \(settledFence\?\.fenced/.test(
      MARKDOWN_TEXT,
    ),
    "a completed fence must be recognised by the CommonMark-complete scanner",
  );
  const settled = MARKDOWN_TEXT.slice(MARKDOWN_TEXT.indexOf("const settledFence ="));
  const branch = settled.slice(0, settled.indexOf("if (props.isIncomplete)"));
  assert.ok(
    /<StreamingFenceBlock[\s\S]{0,120}isIncomplete=\{false\}[\s\S]{0,120}settledFence\.language/.test(
      branch,
    ),
    "and it must render the per-line body rather than Block, or the spans remount",
  );
  // Not `FenceBlock`: that branch also owns the reach latch, the action bar and the mode switch.
  assert.ok(
    !/<FenceBlock[\s\S]{0,200}settledFence/.test(MARKDOWN_TEXT),
    "widening FenceBlock would add controls and artifact paths to blocks that do not have them",
  );
});

test("markdownBlockFallback is what recognises the open fence, not getCodeFence", () => {
  // `getCodeFence` claims only unindented triple backticks; a fence this route misses goes back to
  // the renderer that cannot afford it.
  assert.ok(
    MARKDOWN_TEXT.includes(
      'import { markdownBlockFallback } from "./markdown-block-fallback";',
    ),
    "the streaming route must use the CommonMark-complete scanner",
  );
});

test("the window decision is not reimplemented in the component", () => {
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
  // Why this is not virtualization: the text never leaves the document, so find-in-page, copy and
  // print are untouched (`progressive-mount-controller.ts`, and the `content-visibility` ban).
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
  // A listener per fence reads layout per fence per scroll, costing more than it saves.
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

test("a block is scanned for a mermaid fence once per render, not twice", () => {
  // `findMermaidFence` splits the block and regex-scans every line. Asking "is it open" and "what
  // is the source" as two calls walked it twice on every render of every block, on the hot path
  // this change exists to shorten. Measured on a 3,000 line fence: 0.118 ms per walk, so the
  // redundant one cost 0.092 ms per render, about 5.5 ms per second at 60 fps.
  const body = MARKDOWN_TEXT.slice(MARKDOWN_TEXT.indexOf("function StreamdownBlockContent("));
  const walks = body.match(/findMermaidFence\(props\.content\)/g) ?? [];
  assert.equal(walks.length, 1, "the walk must happen once and both answers come off it");
  assert.ok(
    !/isMermaidFenceOpener\(/.test(MARKDOWN_TEXT),
    "the second walk's wrapper must be gone, not merely unused",
  );
  assert.ok(
    /const mermaidSource = mermaidSourceOf\(props\.content, mermaidFence\);/.test(body),
    "the source must be derived from the walk that already ran",
  );
});

test("a fence source is highlighted once per revision, not twice", () => {
  // This was a passive effect and a layout effect, same inputs and same deps. `code.highlight`
  // caches, but the plugin's throttled `approximateResult` hands back a FRESH object each call, so
  // both setters were observable and every source change scheduled a second render of the body.
  const hook = MARKDOWN_TEXT.slice(
    MARKDOWN_TEXT.indexOf("function useFenceTokens("),
    MARKDOWN_TEXT.indexOf("function StreamingFenceBlock("),
  );
  assert.ok(hook.length > 0, "useFenceTokens still bounds a hook of its own");
  assert.equal(
    (hook.match(/code\.highlight\(/g) ?? []).length,
    1,
    "one highlight call per revision",
  );
  assert.equal(
    (hook.match(/useLayoutEffect\(|useEffect\(/g) ?? []).length,
    1,
    "and one effect, which must be the LAYOUT one so an already-cached fence is coloured before " +
      "its first paint",
  );
  assert.ok(
    /useLayoutEffect\(\(\) => \{/.test(hook),
    "the surviving effect runs before paint",
  );
});
