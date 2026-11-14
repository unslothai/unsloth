// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  htmlNodeToMarkdown,
  type PasteNode,
  richPasteMarkdown,
} from "../src/features/chat/utils/rich-paste.ts";

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

test("spaces inside a link's tag still separate it from its neighbours", () => {
  const body = el("body", el("p", "Read", el("a", { href: "https://unsloth.ai/docs" }, " the docs "), "now"));
  assert.equal(htmlNodeToMarkdown(body), "Read [the docs](https://unsloth.ai/docs) now");
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

test("a copy holding only a code block or styled spans still converts", () => {
  const code = () => el("body", el("pre", el("code", { class: "language-python" }, "def f():\n    return 1\n")));
  const fenced = richPasteMarkdown(
    clipboard('<pre><code class="language-python">def f():\n    return 1\n</code></pre>', "def f():\n    return 1\n"),
    code,
  );
  assert.match(fenced ?? "", /^```python\ndef f\(\):\n {4}return 1\n```/);
  const styled = () => el("body", el("span", { style: "font-weight:700" }, "Bold"), " text");
  assert.equal(
    richPasteMarkdown(clipboard('<span style="font-weight:700">Bold</span> text', "Bold text"), styled),
    "**Bold** text",
  );
});

test("a table caption and line breaks inside code survive", () => {
  assert.equal(
    htmlNodeToMarkdown(el("table", el("caption", "Use these limits"), el("tr", el("th", "A")), el("tr", el("td", "1")))),
    "Use these limits\n\n| A |\n| --- |\n| 1 |",
  );
  assert.equal(
    htmlNodeToMarkdown(el("pre", "def f():", el("br"), "    return 1")),
    "```\ndef f():\n    return 1\n```",
  );
});

test("text styled to keep its whitespace keeps its line breaks and indentation", () => {
  const body = el(
    "body",
    el("p", el("b", "x"), " question"),
    el("div", { style: "white-space: pre-wrap" }, "Call ", el("code", "f"), ":\ndef f():\n    return 1\n\n\ndef g():\n\treturn 2"),
  );
  assert.equal(htmlNodeToMarkdown(body), "**x** question\n\nCall `f`:\ndef f():\n    return 1\n\n\ndef g():\n\treturn 2");
});

test("Google Docs code lines keep their indentation", () => {
  const line = (text: string) => el("p", el("span", { style: "font-family:'Courier New';white-space:pre;white-space:pre-wrap;" }, text));
  const body = el("body", el("b", { style: "font-weight:normal;" }, line("def f():"), line("    return 1")));
  assert.equal(htmlNodeToMarkdown(body), "def f():\n\n    return 1");
});

test("non-breaking space indents survive, and blank non-breaking paragraphs do not", () => {
  const body = el("body", el("p", el("b", "Retry")), el("p", " "), el("p", "def f():"), el("p", "    return 1"));
  assert.equal(htmlNodeToMarkdown(body), "**Retry**\n\ndef f():\n\n    return 1");
});

test("Google Docs bold, italic and links keep valid Markdown around their spaces", () => {
  const docs = "white-space:pre;white-space:pre-wrap;";
  const span = (text: string, style = "") => el("span", { style: style + docs }, text);
  const body = el(
    "body",
    el(
      "b",
      { style: "font-weight:normal;" },
      el("p", span("This is "), span("bold ", "font-weight:700;"), span("and"), span(" italic", "font-style:italic;")),
      el("p", el("a", { href: "https://x.com" }, span("https://x.com")), span(" and "), el("a", { href: "https://y.com" }, span("site "))),
    ),
  );
  assert.equal(htmlNodeToMarkdown(body), "This is **bold** and *italic*\n\nhttps://x.com and [site](https://y.com)");
});

test("preformatted edge whitespace is restored once, and bold code keeps its indent", () => {
  const docs = "white-space:pre;white-space:pre-wrap;";
  const body = el(
    "body",
    el("b", { style: "font-weight:normal;" }, el("p", el("span", { style: docs }, "    if x:")), el("p", el("span", { style: `${docs}font-weight:700` }, "        return 1"))),
  );
  const plain = "    if x:\n        return 1";
  const data = { types: ["text/html"], getData: (type: string) => (type === "text/html" ? "<b>x</b>" : plain) };
  assert.equal(richPasteMarkdown(data, () => body), "    if x:\n\n        **return 1**");
});

test("a non-breaking space inside code is kept as data", () => {
  const body = el("body", el("pre", el("code", "s = 'a\u00a0b'")));
  assert.equal(htmlNodeToMarkdown(body), "```\ns = 'a\u00a0b'\n```");
});

test("a descendant that resets white-space collapses its own text again", () => {
  const body = el("body", el("div", { style: "white-space:pre-wrap" }, el("b", "Q"), ":\n", el("span", { style: "white-space: normal" }, "hello\n    world")));
  assert.equal(htmlNodeToMarkdown(body), "**Q**:\nhello world");
});

test("blank lines in a list item survive adjacent preformatted spans", () => {
  const pre = { style: "white-space:pre-wrap" };
  const body = el("body", el("ul", el("li", el("span", pre, "a\nb"), el("span", pre, "c\n\nd"))));
  assert.equal(htmlNodeToMarkdown(body), "- a\n  bc\n\n  d");
});

test("pre-line keeps line breaks but collapses spaces", () => {
  const body = el("body", el("div", { style: "white-space:pre-line" }, el("b", "Note"), el("br"), el("br"), "    run  fast\n  next"));
  assert.equal(htmlNodeToMarkdown(body), "**Note**\n\nrun fast\nnext");
});
