// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

import {
  markdownBlockFallback,
} from "../src/components/assistant-ui/markdown-block-fallback.ts";

import { readSrc } from "./helpers/kit.ts";

/** A failed lazy Streamdown chunk must degrade to the same characters, not crash the app. */

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
  // The chunk is fetched while the fence is still open.
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
  // CommonMark: a close may be longer than the open.
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
  // The parser reads an unterminated opener as an empty code block, so this is reachable.
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
  // CommonMark: a backtick fence info string may not contain backticks.
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
  const content = "```py meta`x\nabc\n```";
  assert.equal(markdownBlockFallback(content).text, content);
  assert.equal(markdownBlockFallback("```py`bad").text, "```py`bad");
});

test("a tilde fence may carry backticks in its info string", () => {
  // The restriction applies only to backtick fences.
  const fallback = markdownBlockFallback("~~~py`ok\nabc\n~~~");

  assert.equal(fallback.text, "abc");
  assert.equal(fallback.fenced, true);
  assert.equal(fallback.language, "py`ok");
});

test("an indented fence loses the opener's indentation, as the renderer does", () => {
  // CommonMark: up to N spaces of opener indentation are stripped from content.
  const fallback = markdownBlockFallback("   ```python\n   x = 1\n   ```");

  assert.equal(fallback.text, "x = 1");
  assert.equal(fallback.language, "python");
  assert.equal(fallback.fenced, true);
});

test("only the opener's own indentation comes off, never the code's", () => {
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
  // Streamdown returns the whole reply as one block once it has a footnote.
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
  // Source check: no output test can tell a missing boundary until a chunk fails.
  assert.ok(
    wrappersAroundBlockContent().includes("MarkdownBlockBoundary"),
    "the block component is rendered outside MarkdownBlockBoundary, so a fence whose highlighter fails to load unmounts all of Unsloth through the router's error boundary again",
  );
});

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
  /* Every Block site, including the unclosed streaming fence, must sit inside the inner boundary. */
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
  /* Controls must be siblings of the boundary that catches Block, not descendants. */
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
  // The inner boundary narrows what is replaced, never what is caught.
  assert.ok(
    wrappersAroundBlockContent().includes("MarkdownBlockBoundary"),
    "the outer boundary no longer wraps the block, so narrowing the inner one narrowed the coverage too",
  );
});

test("the boundary does not retry the import it caught", () => {
  const boundary = readSrc("components/assistant-ui/markdown-block-boundary.tsx");

  // Rejected dynamic imports are cached (whatwg/html#6768), so resetting on props just rethrows.
  assert.ok(
    !boundary.includes("getDerivedStateFromProps"),
    "the boundary resets itself from props, which on a streaming reply means throwing and catching on every chunk for an import that can never succeed again",
  );
});

test("a carriage return only closes a fence as the closing line's last character", () => {
  const trailing = markdownBlockFallback("```py\nx\n```\r ");
  assert.equal(trailing.text, "x\n```\r ", "a CR before other tail text does not close the fence");
  assert.equal(trailing.fenced, true);

  const closing = markdownBlockFallback("```py\nx\n```\r");
  assert.equal(closing.text, "x", "a CR as the closing line's last character does close it");
});
