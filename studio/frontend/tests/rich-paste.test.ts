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

test("pre-line keeps runs of blank lines", () => {
  const body = el("body", el("div", { style: "white-space:pre-line" }, el("b", "Code"), '\nmessage = """first\n\n\nfourth"""'));
  assert.equal(htmlNodeToMarkdown(body), '**Code**\nmessage = """first\n\n\nfourth"""');
});

test("long whitespace runs convert in linear time", () => {
  const run = (ch: string) => `a${ch.repeat(60_000)}b`;
  const bodies = [
    el("body", el("p", el("b", "x"), run(" "))),
    el("body", el("div", { style: "white-space:pre-wrap" }, el("b", run(" ")))),
    el("body", el("table", el("tr", el("td", { style: "white-space:pre-wrap" }, run(" ")), el("td", run("\u00a0"))))),
    el("body", el("div", { style: "white-space:pre-wrap" }, el("b", "a\n\n".repeat(20_000)))),
  ];
  const start = performance.now();
  for (const body of bodies) htmlNodeToMarkdown(body);
  const plain = run(" ");
  richPasteMarkdown({ types: ["text/html"], getData: (type: string) => (type === "text/html" ? "<b>x</b>" : plain) }, () => el("body", el("b", "q")));
  assert.ok(performance.now() - start < 1000);
});

test("layout non-breaking spaces in preformatted spans become spaces", () => {
  const body = el("body", el("div", { style: "white-space:pre-wrap" }, el("b", "Run"), ":\nx\u00a0=\u00a01"));
  assert.equal(htmlNodeToMarkdown(body), "**Run**:\nx = 1");
});

test("a non-breaking space inside inline code is kept as data", () => {
  const body = el("body", el("p", el("b", "Check:"), " ", el("code", "ord('\u00a0') == 160")));
  assert.equal(htmlNodeToMarkdown(body), "**Check:** `ord('\u00a0') == 160`");
});

test("emphasis closes and reopens around preserved paragraph breaks", () => {
  const body = el("body", el("div", { style: "white-space:pre-wrap" }, el("b", "first\n\nsecond"), " tail"));
  assert.equal(htmlNodeToMarkdown(body), "**first**\n\n**second** tail");
});

test("splitting emphasis across a preserved blank-line run leaks no marker", () => {
  const body = el("body", el("b", { style: "white-space:pre-line" }, "first\n\n\nsecond"));
  assert.equal(htmlNodeToMarkdown(body), "**first**\n\n\n**second**");
});

test("a non-breaking space in inline code inside a table cell is kept", () => {
  const body = el("body", el("table", el("tr", el("th", "k")), el("tr", el("td", el("code", "ord(' ')")))));
  assert.equal(htmlNodeToMarkdown(body), "| k |\n| --- |\n| `ord(' ')` |");
});

test("random nested markup never leaks internal markers", () => {
  let seed = 7;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  const pick = <T>(items: T[]) => items[Math.floor(rnd() * items.length)];
  const atoms = ["a", " ", "  ", "\n", "\n\n", "\n\n\n", "\t", " ", "    x", "`", "|", "*"];
  const styles = ["", "white-space:pre-wrap", "white-space:pre-line", "white-space:normal", "font-weight:700", "font-style:italic"];
  const tags = ["b", "i", "s", "span", "p", "div", "code", "a", "br", "ul", "li", "table", "tr", "td", "caption", "pre", "blockquote", "h2"];
  const gen = (depth: number): Child =>
    depth <= 0 || rnd() < 0.3
      ? Array.from({ length: 1 + Math.floor(rnd() * 3) }, () => pick(atoms)).join("")
      : el(pick(tags), { style: pick(styles), href: "https://x.com" }, ...Array.from({ length: Math.floor(rnd() * 4) }, () => gen(depth - 1)));
  for (let i = 0; i < 3000; i++) {
    const out = htmlNodeToMarkdown(el("body", gen(5)));
    assert.doesNotMatch(out, /[\u0001-\u0003]/);
  }
});

test("emphasis around a code block never enters the code", () => {
  const body = el("body", el("del", el("p", "before"), el("pre", el("code", "x = 1\n\ny = 2")), el("p", "after")));
  assert.equal(htmlNodeToMarkdown(body), "~~before~~\n\n```\nx = 1\n\ny = 2\n```\n\n~~after~~");
});

test("preserved text starting with a backtick still splits emphasis at paragraphs", () => {
  const body = el("body", el("b", { style: "white-space:pre-wrap" }, "`x`\n\nsecond"));
  assert.equal(htmlNodeToMarkdown(body), "**`x`**\n\n**second**");
});

test("internal marker characters in copied text are dropped", () => {
  const body = el("body", el("p", el("b", "a\u0002b"), el("code", "c\u0003d")), el("pre", "e\u0004f\u0005"));
  assert.equal(htmlNodeToMarkdown(body), "**ab**`cd`\n\n```\nef\n```");
});

test("list items and table cells that reset white-space collapse their text", () => {
  const body = el(
    "body",
    el("ul", { style: "white-space:pre-wrap" }, el("li", { style: "white-space:normal" }, "first\n    second")),
    el("table", { style: "white-space:pre-wrap" }, el("tr", el("th", "k")), el("tr", el("td", { style: "white-space:normal" }, el("b", "a\n    b")))),
  );
  assert.equal(htmlNodeToMarkdown(body), "- first second\n\n| k |\n| --- |\n| **a b** |");
});

test("a preserved link label is split into one link per paragraph", () => {
  const body = el("body", el("a", { href: "https://x.com", style: "white-space:pre-wrap" }, "first\n\nsecond"));
  assert.equal(htmlNodeToMarkdown(body), "[first](https://x.com)\n\n[second](https://x.com)");
});
