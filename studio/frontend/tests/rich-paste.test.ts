// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  htmlNodeToMarkdown,
  type PasteNode,
  richPasteMarkdown,
} from "../src/features/chat/utils/rich-paste.ts";

// No DOM in this project: build the trees the converter reads by hand.
type Child = PasteNode | string;
function el(tag: string, attrs: Record<string, string> | Child = {}, ...kids: Child[]): PasteNode {
  const attributes = typeof attrs === "object" && !("nodeType" in attrs) ? attrs : {};
  const all = typeof attrs === "object" && !("nodeType" in attrs) ? kids : [attrs as Child, ...kids];
  const childNodes = all.map((kid) =>
    typeof kid === "string"
      ? { nodeType: 3, nodeName: "#text", childNodes: [], textContent: kid }
      : kid,
  );
  const text = (node: PasteNode): string =>
    node.nodeType === 3 ? (node.textContent ?? "") : Array.from(node.childNodes).map(text).join("");
  const node: PasteNode = {
    nodeType: 1,
    nodeName: tag.toUpperCase(),
    childNodes,
    textContent: "",
    getAttribute: (name) => attributes[name] ?? null,
  };
  node.textContent = text(node);
  return node;
}

function clipboard(html: string, plain: string, types: string[] = ["text/html", "text/plain"]) {
  return {
    types,
    getData: (type: string) => (type === "text/html" ? html : type === "text/plain" ? plain : ""),
  };
}

test("links keep their target, and bare URLs stay bare", () => {
  const body = el("body", el("p", "See ", el("a", { href: "https://unsloth.ai/docs" }, "the docs"), " or ", el("a", { href: "https://x.com/a b" }, "https://x.com/a b"), "."));
  assert.equal(htmlNodeToMarkdown(body), "See [the docs](https://unsloth.ai/docs) or https://x.com/a%20b.");
});

test("unsafe or relative links paste as their text only", () => {
  const body = el("body", el("a", { href: "javascript:alert(1)" }, "click"), " ", el("a", { href: "/relative" }, "here"));
  assert.equal(htmlNodeToMarkdown(body), "click here");
});

test("tables become GFM tables with escaped pipes and ragged rows padded", () => {
  const body = el(
    "body",
    el(
      "table",
      el("thead", el("tr", el("th", "Model"), el("th", "VRAM"))),
      el("tbody", el("tr", el("td", "Qwen | 7B"), el("td", el("b", "8 GB"))), el("tr", el("td", "Llama"))),
    ),
  );
  assert.equal(
    htmlNodeToMarkdown(body),
    ["| Model | VRAM |", "| --- | --- |", "| Qwen \\| 7B | **8 GB** |", "| Llama |  |"].join("\n"),
  );
});

test("merged cells keep the other values under their own headers", () => {
  const body = el(
    "body",
    el(
      "table",
      el("tr", el("th", "Group"), el("th", "Value"), el("th", "Note")),
      el("tr", el("td", { rowspan: "2" }, "A"), el("td", "10"), el("td", "x")),
      el("tr", el("td", { colspan: "2" }, "20")),
    ),
  );
  assert.equal(
    htmlNodeToMarkdown(body),
    ["| Group | Value | Note |", "| --- | --- | --- |", "| A | 10 | x |", "|  | 20 |  |"].join("\n"),
  );
});

test("lists, nesting, headings, quotes and emphasis", () => {
  const body = el(
    "body",
    el("h2", "Setup"),
    el("ol", el("li", "Install ", el("code", "uv")), el("li", "Run", el("ul", el("li", el("em", "fast")), el("li", "safe")))),
    el("blockquote", el("p", "Quoted ", el("strong", "text"))),
  );
  assert.equal(
    htmlNodeToMarkdown(body),
    ["## Setup", "", "1. Install `uv`", "2. Run", "   - *fast*", "   - safe", "", "> Quoted **text**"].join("\n"),
  );
});

test("code blocks keep their indentation and language", () => {
  const body = el("body", el("pre", el("code", { class: "language-python" }, "def f():\n    return 1\n")));
  assert.equal(htmlNodeToMarkdown(body), "```python\ndef f():\n    return 1\n```");
});

test("code inside a list item keeps its blank lines", () => {
  const body = el("body", el("ul", el("li", "Example", el("pre", el("code", 'message = """first\n\nthird"""')))));
  assert.equal(htmlNodeToMarkdown(body), '- Example\n  ```\n  message = """first\n\n  third"""\n  ```');
});

test("Google Docs: the normal-weight wrapper adds nothing, styled spans become emphasis", () => {
  const body = el(
    "body",
    el("b", { style: "font-weight:normal;" }, el("span", { style: "font-weight:700" }, "Bold"), " and ", el("span", { style: "font-style:italic" }, "italic")),
  );
  assert.equal(htmlNodeToMarkdown(body), "**Bold** and *italic*");
});

test("rich paste falls back to the plain text when it would add nothing", () => {
  const parse = () => el("body", el("b", { style: "font-weight:normal" }, "just words"));
  assert.equal(richPasteMarkdown(clipboard('<b style="font-weight:normal">just words</b>', "just words"), parse), null);
  // No structural tag at all: spans and divs from a styled copy paste as before.
  assert.equal(richPasteMarkdown(clipboard("<div><span>x</span></div>", "x"), () => el("body", "x")), null);
  assert.equal(richPasteMarkdown(clipboard("", "plain only")), null);
});

test("code copied from VS Code is never converted", () => {
  const data = clipboard("<div><a href='https://a.b'>x</a></div>", "x", ["text/plain", "text/html", "vscode-editor-data"]);
  assert.equal(richPasteMarkdown(data, () => el("body", el("a", { href: "https://a.b" }, "x"))), null);
});

test("the copied text's edge spaces survive conversion", () => {
  const parse = () => el("body", el("strong", "Hello "));
  assert.equal(richPasteMarkdown(clipboard("<strong>Hello </strong>", "Hello "), parse), "**Hello** ");
});

test("a structural paste returns the Markdown", () => {
  const html = '<p>Read <a href="https://unsloth.ai">Unsloth</a></p>';
  const parse = () => el("body", el("p", "Read ", el("a", { href: "https://unsloth.ai" }, "Unsloth")));
  assert.equal(richPasteMarkdown(clipboard(html, "Read Unsloth"), parse), "Read [Unsloth](https://unsloth.ai)");
});
