// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

import {
  markdownBlockFallback,
} from "../src/components/assistant-ui/markdown-block-fallback.ts";

/**
 * Streamdown loads the syntax highlighted code body and the Mermaid renderer
 * through `React.lazy`, and it fetches them the first time a reply contains
 * that construct. A rejected import rethrows during render. Before the boundary
 * these tests protect, the nearest catcher was TanStack Router's, so ONE chunk
 * that would not load replaced all of Unsloth with "Something went wrong!",
 * unmounted the assistant-ui runtime with it, and left the reply's stream with
 * nothing consuming it.
 *
 * Measured on an unmodified tree by aborting exactly that one request: the
 * document went from 122 elements to 21, the pane's readable text went from
 * 11,968 characters to 0, and the stream stopped at 720 of 12,000 characters
 * and never resumed.
 *
 * So what these tests hold is not "the app does not crash". It is that the
 * READER STILL HAS THE CONTENT. A degraded fence has to be the same characters
 * in the same order, because that is the answer they asked for, and an error
 * card or an empty box in its place is a worse outcome than missing colour.
 */

const FENCE_BODY = [
  "def score(rows, cap):",
  "    total = 0.0",
  "",
  "    for row in rows:",
  "        total += min(cap, row.weight)",
  "    return total",
].join("\n");

test("a fenced block degrades to the code itself, without the fence scaffolding", () => {
  const fallback = markdownBlockFallback("```python\n" + FENCE_BODY + "\n```");

  assert.equal(
    fallback.text,
    FENCE_BODY,
    "the degraded fence is not the code it was carrying, so the reader lost the answer",
  );
  assert.equal(fallback.language, "python");
  assert.equal(
    fallback.fenced,
    true,
    "a fence that is not reported as fenced renders as prose, so its indentation and line breaks collapse",
  );
});

test("the degraded fence keeps every character and every blank line", () => {
  const fallback = markdownBlockFallback("```python\n" + FENCE_BODY + "\n```");

  assert.equal(
    fallback.text.split("\n").length,
    FENCE_BODY.split("\n").length,
    "the degraded fence lost a line, and a blank line inside a function is not decoration",
  );
  assert.ok(
    fallback.text.includes("    total = 0.0"),
    "the degraded fence lost its leading whitespace, which in python is the program",
  );
});

test("a fence that is still arriving degrades too", () => {
  // The failure happens MID-fence: the chunk is fetched when the fence first
  // renders, which is long before it closes. A fallback that only handled
  // closed fences would show backticks at exactly the moment it is needed.
  const fallback = markdownBlockFallback("```python\n" + FENCE_BODY);

  assert.equal(
    fallback.text,
    FENCE_BODY,
    "an unclosed fence was not recognised, so a reader sees the opening backticks and the language tag as text",
  );
  assert.equal(fallback.fenced, true);
});

test("a fence with no language tag still degrades to its body", () => {
  const fallback = markdownBlockFallback("```\n" + FENCE_BODY + "\n```");

  assert.equal(fallback.text, FENCE_BODY);
  assert.equal(
    fallback.language,
    null,
    "an absent language tag has to be absent, not the empty string, or the header renders blank",
  );
});

test("a tilde fence degrades the same way", () => {
  const fallback = markdownBlockFallback("~~~ts\nconst a = 1;\n~~~");

  assert.equal(fallback.text, "const a = 1;");
  assert.equal(fallback.language, "ts");
});

test("prose is handed back unchanged and is not treated as a fence", () => {
  const prose = "The scorer skips empty rows, then applies the cap.";
  const fallback = markdownBlockFallback(prose);

  assert.equal(
    fallback.text,
    prose,
    "a paragraph was rewritten on its way to the fallback",
  );
  assert.equal(
    fallback.fenced,
    false,
    "prose rendered as a code block is a visible change to a reply that had nothing wrong with it",
  );
});

test("a closing fence longer than the opening one still closes the block", () => {
  // CommonMark 0.31.2 requires the close to carry AT LEAST as many characters
  // as the open, so a four-backtick close is legitimate and is how a model
  // closes a fence whose body contains a three-backtick one. Demanding the same
  // run left the close on screen as though it were the last line of the code.
  const fallback = markdownBlockFallback("```python\n" + FENCE_BODY + "\n````");

  assert.equal(
    fallback.text,
    FENCE_BODY,
    "a longer closing fence was not recognised, so the reader sees stray backticks below their code",
  );
  assert.equal(fallback.language, "python");
  assert.equal(fallback.fenced, true);
});

test("a fence closed by a longer run keeps an inner fence in the body", () => {
  // The reason the rule exists: the four-backtick fence is carrying a
  // three-backtick one, which has to survive as content.
  const fallback = markdownBlockFallback("````md\n```py\nx = 1\n```\n````");

  assert.equal(fallback.text, "```py\nx = 1\n```");
  assert.equal(fallback.language, "md");
});

test("an empty fence degrades to nothing, not to its own closing backticks", () => {
  const fallback = markdownBlockFallback("```\n```");

  assert.equal(
    fallback.text,
    "",
    "the closing fence was returned as the body, so an empty code block renders ``` as if the model had written it",
  );
  assert.equal(fallback.fenced, true);
});

test("an opening fence the reply ends on is still a fence", () => {
  // The repository's own parser (`mdast-util-from-markdown`, via Streamdown's
  // `parseMarkdownIntoBlocks`) reads "```python" with no line break as
  // code(lang="python", value=""), so the highlighter chunk IS requested for it
  // and this fallback is reachable. CommonMark 0.31.2 closes an unclosed block
  // at the end of the document. Read as prose, the reader gets the delimiter and
  // the language tag as literal text where their answer should be.
  for (const [content, language] of [
    ["```python", "python"],
    ["```", null],
    ["~~~py", "py"],
    ["   ```py", "py"],
  ] as const) {
    const fallback = markdownBlockFallback(content);
    assert.equal(
      fallback.fenced,
      true,
      `an EOF terminated opening fence rendered as prose: ${JSON.stringify(content)}`,
    );
    assert.equal(fallback.text, "");
    assert.equal(fallback.language, language);
  }
});

test("a backtick fence whose info string carries a backtick is prose", () => {
  // "If the info string comes after a backtick fence, it may not contain any
  // backtick characters" (CommonMark 0.31.2), so this line does not open a
  // fence and the parser keeps it as a paragraph. Accepting it as an opener
  // discarded the opening line AND the closing one, which is the model's answer
  // silently going missing in the one view of it that still exists.
  const content = "```py`bad\nabc\n```";
  const fallback = markdownBlockFallback(content);

  assert.equal(
    fallback.text,
    content,
    "an invalid backtick fence opener was accepted, so the block degraded to part of itself",
  );
  assert.equal(fallback.fenced, false);
  assert.equal(fallback.language, null);
});

test("a backtick anywhere in the info string disqualifies the opener", () => {
  // Not just the first word: the restriction is on the whole info string.
  const content = "```py meta`x\nabc\n```";
  assert.equal(markdownBlockFallback(content).text, content);
  assert.equal(markdownBlockFallback("```py`bad").text, "```py`bad");
});

test("a tilde fence may carry backticks in its info string", () => {
  // The restriction is backtick fences only, and the parser agrees:
  // "~~~py`ok\nabc\n~~~" is code(lang="py`ok", value="abc").
  const fallback = markdownBlockFallback("~~~py`ok\nabc\n~~~");

  assert.equal(fallback.text, "abc");
  assert.equal(fallback.fenced, true);
  assert.equal(fallback.language, "py`ok");
});

test("an indented fence loses the opener's indentation, as the renderer does", () => {
  // "If the leading code fence is indented N spaces, then up to N spaces of
  // indentation are removed from each line of the content" (CommonMark 0.31.2),
  // and the parser agrees: "   ```python\n   x = 1\n   ```" is code(value="x = 1").
  // Leaving it on gives the reader text their model did not write, and pasting
  // it into a file is an IndentationError.
  const fallback = markdownBlockFallback("   ```python\n   x = 1\n   ```");

  assert.equal(fallback.text, "x = 1");
  assert.equal(fallback.language, "python");
  assert.equal(fallback.fenced, true);
});

test("only the opener's own indentation comes off, never the code's", () => {
  // UP TO N. A line indented deeper than the opener keeps the remainder, which
  // in Python is the program, and a line indented less loses only what it has.
  assert.equal(
    markdownBlockFallback("  ```py\n  def f():\n      return 1\n  ```").text,
    "def f():\n    return 1",
    "an indented fence lost the body's own indentation, so the code changed meaning",
  );
  assert.equal(
    markdownBlockFallback("   ```py\nx = 1\n   ```").text,
    "x = 1",
    "a content line shallower than the opener was over-stripped",
  );
  assert.equal(
    markdownBlockFallback("```py\n    x = 1\n```").text,
    "    x = 1",
    "an unindented fence must not strip anything at all",
  );
});

test("a block that continues past its fence is not read as one fence", () => {
  // Streamdown 2.5's parseMarkdownIntoBlocks returns the WHOLE reply as a single
  // block once it contains a footnote. Measured on the installed version: the
  // same reply is 5 blocks without the footnote and 1 block with it. So a fence
  // followed by prose really does arrive here as one string, and reading it all
  // as the fence's body showed the closing delimiter, the prose and the footnote
  // as though the model had written them as Python.
  const content = "```python\nx=1\n```\n\nAfter code.[^1]\n\n[^1]: note";
  const fallback = markdownBlockFallback(content);

  assert.equal(
    fallback.text,
    content,
    "a multi-construct block was degraded as a single fence, so prose rendered as code",
  );
  assert.equal(fallback.fenced, false);
  assert.equal(fallback.language, null);
});

test("the close still has to be the last line, not merely present", () => {
  // The three positions the close can take, which are the whole rule.
  assert.equal(markdownBlockFallback("```py\nx\n```").text, "x");
  assert.equal(markdownBlockFallback("```py\nx\n```\n").text, "x");
  assert.equal(markdownBlockFallback("```py\nx").text, "x");
  assert.equal(
    markdownBlockFallback("```py\nx\n```\ntail").fenced,
    false,
    "content after the closing fence was swallowed into the code body",
  );
});

test("an inner fence shorter than the opener does not end the block early", () => {
  // The scan runs from the top now, so the first line that could be mistaken for
  // a close is a four-backtick fence's three-backtick content.
  const fallback = markdownBlockFallback("````md\n```py\nx = 1\n```\n````");

  assert.equal(fallback.text, "```py\nx = 1\n```");
  assert.equal(fallback.language, "md");
  assert.equal(fallback.fenced, true);
});

test("a block with content never degrades to nothing", () => {
  for (const content of [
    "```python\nx = 1\n```",
    "plain",
    "| a | b |\n|---|---|\n| 1 | 2 |",
  ]) {
    const fallback = markdownBlockFallback(content);
    assert.ok(
      fallback.text.length > 0,
      `a non-empty block degraded to an empty string: ${JSON.stringify(content)}`,
    );
  }
});

const MARKDOWN_TEXT_PATH = new URL(
  "../src/components/assistant-ui/markdown-text.tsx",
  import.meta.url,
);
const source = ts.createSourceFile(
  MARKDOWN_TEXT_PATH.pathname,
  readFileSync(MARKDOWN_TEXT_PATH, "utf8"),
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);

/** The JSX element names that wrap `StreamdownBlockContent` where it is rendered. */
function wrappersAroundBlockContent(): string[] {
  const wrappers: string[] = [];
  const visit = (node: ts.Node, open: string[]): void => {
    if (
      ts.isJsxSelfClosingElement(node) &&
      node.tagName.getText(source) === "StreamdownBlockContent"
    ) {
      wrappers.push(...open);
    }
    const next =
      ts.isJsxElement(node)
        ? [...open, node.openingElement.tagName.getText(source)]
        : open;
    node.forEachChild((child) => visit(child, next));
  };
  source.forEachChild((node) => visit(node, []));
  return wrappers;
}

test("every markdown block is rendered inside the boundary", () => {
  // A source check because no output test can tell them apart: an unwrapped tree
  // renders identically until a lazy chunk fails, and then it takes the whole
  // application with it. This is the only thing standing between that and a
  // quiet revert.
  assert.ok(
    wrappersAroundBlockContent().includes("MarkdownBlockBoundary"),
    "the block component is rendered outside MarkdownBlockBoundary, so a fence whose highlighter fails to load unmounts all of Unsloth through the router's error boundary again",
  );
});

/** The JSX element names enclosing every occurrence of `tag`, innermost last. */
function wrappersAround(tag: string): string[][] {
  const found: string[][] = [];
  const visit = (node: ts.Node, open: string[]): void => {
    const name =
      ts.isJsxSelfClosingElement(node) || ts.isJsxOpeningElement(node)
        ? node.tagName.getText(source)
        : null;
    if (name === tag) found.push(open);
    const next = ts.isJsxElement(node)
      ? [...open, node.openingElement.tagName.getText(source)]
      : open;
    node.forEachChild((child) => visit(child, next));
  };
  source.forEachChild((node) => visit(node, []));
  return found;
}

const INNER_BOUNDARY = "MarkdownRendererBoundary";

test("no Block renders outside the renderer boundary", () => {
  /*
   * The test that was missing, and its absence shipped a boundary that never
   * fired on the path it was written for.
   *
   * `Block` is the only thing that loads a chunk at render time, so EVERY place
   * it is rendered has to be inside the narrower boundary. Checking only the
   * places that have controls beside them is not enough: `getCodeFence` needs
   * the CLOSING fence, so a fence that is still streaming falls past the fence
   * branch to the bare `Block` at the end of `StreamdownBlockContent`, and that
   * is precisely when the highlighter is first requested and fails. Leaving that
   * one unguarded let the whole-block boundary catch and LATCH, so the block
   * never re-entered `FenceBlock` when its closing fence arrived and the copy
   * and download bar never mounted.
   *
   * Measured before this assertion existed: a streamed abort produced a document
   * identical to the commit before the inner boundary was added, 1350 elements
   * and 0 copy buttons on both.
   */
  const sites = wrappersAround("Block");
  assert.ok(sites.length > 0, "no <Block> is rendered at all");
  const unguarded = sites.filter((open) => !open.includes(INNER_BOUNDARY));
  assert.deepEqual(
    unguarded,
    [],
    `a <Block> renders outside ${INNER_BOUNDARY}, so a rejected chunk there escapes to the whole-block boundary and latches it for the rest of the stream`,
  );
});

test("a failed renderer does not take the block's controls with it", () => {
  /*
   * The whole point, asserted as a RELATIONSHIP so it cannot pass vacuously: a
   * tree with no inner boundary at all would satisfy "the control is not inside
   * one", which is exactly the tree this test exists to reject.
   *
   * `Block` is the only thing in a block that loads a chunk at render time, so
   * it is what the rejected `React.lazy` throws from. `CodeBlockActions` (copy
   * and download) and `MermaidCopyButton` are independent of it, and they are
   * what a reader reaches for when a block did not render. So each control must
   * be a SIBLING of the boundary that catches the renderer, not a descendant:
   * the control's ancestors must be exactly the Block's ancestors with the
   * boundary taken out.
   */
  for (const control of ["CodeBlockActions", "MermaidCopyButton"]) {
    const sites = wrappersAround(control);
    assert.ok(sites.length > 0, `${control} is not rendered at all`);
    for (const open of sites) {
      assert.ok(
        !open.includes(INNER_BOUNDARY),
        `${control} is rendered inside ${INNER_BOUNDARY}, so it is unmounted along with the renderer that failed and the reader loses it exactly when they need it`,
      );
      assert.ok(
        wrappersAround("Block").some(
          (blockOpen) =>
            blockOpen.includes(INNER_BOUNDARY) &&
            blockOpen.filter((w) => w !== INNER_BOUNDARY).join(">") ===
              open.join(">"),
        ),
        `no boundaried <Block> sits beside ${control}, so a rejected lazy chunk still escapes to the whole-block boundary and takes ${control} down with it`,
      );
    }
  }
});

test("the whole-block boundary is still the catch-all above them", () => {
  // The inner boundary narrows what is REPLACED, never what is CAUGHT. Anything
  // that throws outside `Block` -- a hook, a sanitizer, an artifact card -- must
  // still reach the outer boundary rather than the router.
  assert.ok(
    wrappersAroundBlockContent().includes("MarkdownBlockBoundary"),
    "the outer boundary no longer wraps the block, so narrowing the inner one narrowed the coverage too",
  );
});

test("the boundary does not retry the import it caught", () => {
  const boundary = readFileSync(
    new URL(
      "../src/components/assistant-ui/markdown-block-boundary.tsx",
      import.meta.url,
    ),
    "utf8",
  );

  // React and the browser's module map both cache a rejected dynamic import
  // (whatwg/html#6768), so a boundary that resets on new props rethrows on every
  // frame of a streaming reply and issues no new request for its trouble.
  assert.ok(
    !boundary.includes("getDerivedStateFromProps"),
    "the boundary resets itself from props, which on a streaming reply means throwing and catching on every chunk for an import that can never succeed again",
  );
});

test("a carriage return only closes a fence as the closing line's last character", () => {
  // A rewrite of the closing-fence scan once accepted a CR ANYWHERE in the tail,
  // silently closing fences on 393,620 inputs of a differential corpus. Nothing
  // failed, because nothing asserted the rule.
  const trailing = markdownBlockFallback("```py\nx\n```\r ");
  assert.equal(trailing.text, "x\n```\r ", "a CR before other tail text does not close the fence");
  assert.equal(trailing.fenced, true);

  const closing = markdownBlockFallback("```py\nx\n```\r");
  assert.equal(closing.text, "x", "a CR as the closing line's last character does close it");
});
