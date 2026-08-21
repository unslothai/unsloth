// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  SEARCH_IMAGES_MARKER,
  answerTextFromParts,
  collectSearchImages,
  extractListSubjects,
  extractSearchImages,
  holdBackPartialSearchImageToken,
  isSearchImagesToolResult,
  missingListSubjects,
  parseSearchImagesSignature,
  placeSubjectImages,
  rewriteSearchImageTokens,
  searchImagePath,
  searchImagesSignature,
  searchResultText,
  stripSearchImageTokens,
} from "../src/features/chat/search-images/search-images.ts";
import { safeMarkdownUrl } from "../src/lib/safe-markdown-url.ts";

const ENTRY = {
  id: "0123456789ab",
  title: "Golden Retriever",
  domain: "akc.org",
  source: "https://www.akc.org/golden",
};
const OTHER = { ...ENTRY, id: "abcdef012345", title: "Labrador" };
const KNOWN = new Set([ENTRY.id, OTHER.id]);

test("extractSearchImages splits a valid envelope off the result text", () => {
  const raw = `Title: A\nURL: https://a.test\nSnippet: s${SEARCH_IMAGES_MARKER}${JSON.stringify([ENTRY])}`;
  const { text, images } = extractSearchImages(raw);
  assert.equal(text, "Title: A\nURL: https://a.test\nSnippet: s");
  assert.deepEqual(images, [ENTRY]);
});

test("extractSearchImages leaves malformed or foreign envelopes alone", () => {
  for (const raw of [
    "plain text",
    `x${SEARCH_IMAGES_MARKER}not json`,
    `x${SEARCH_IMAGES_MARKER}[]`,
    `x${SEARCH_IMAGES_MARKER}[{"id":"zz"}]`,
    `x${SEARCH_IMAGES_MARKER}[{"id":"0123456789ab","title":"t","domain":"d","source":"https://d.test/p","thumbnail":"https://leak"}]`.replace(
      ',"thumbnail":"https://leak"',
      "",
    ) + "\n__OTHER__:tail",
  ]) {
    const { text, images } = extractSearchImages(raw);
    if (raw.endsWith("tail")) {
      // A later sentinel bounds the payload and survives the strip.
      assert.deepEqual(
        images.map((e) => e.id),
        ["0123456789ab"],
      );
      assert.equal(text, "x\n__OTHER__:tail");
    } else {
      assert.equal(text, raw);
      assert.deepEqual(images, []);
    }
  }
});

test("isSearchImagesToolResult accepts only the wrapper shape", () => {
  assert.ok(isSearchImagesToolResult({ text: "t", webImages: [ENTRY] }));
  assert.ok(!isSearchImagesToolResult({ text: "t", webImages: [] }));
  assert.ok(
    !isSearchImagesToolResult({
      text: "t",
      images: [{ data: "", mimeType: "" }],
    }),
  );
  assert.ok(!isSearchImagesToolResult("string"));
  assert.ok(
    !isSearchImagesToolResult({
      text: "t",
      webImages: [{ ...ENTRY, id: "../x" }],
    }),
  );
  assert.ok(
    !isSearchImagesToolResult({
      text: "t",
      webImages: [{ ...ENTRY, source: "javascript:alert(1)" }],
    }),
  );
});

test("rewriteSearchImageTokens swaps known tokens for elements and drops unknown ones", () => {
  const out = rewriteSearchImageTokens(
    `Lab:\n[[img:${OTHER.id}]]\nand [[img:ffffffffffff]] gone`,
    KNOWN,
  );
  assert.equal(
    out,
    `Lab:\n<search-image token="${OTHER.id}"></search-image>\nand  gone`,
  );
});

test("rewriteSearchImageTokens never touches code", () => {
  const fenced = "```\n[[img:0123456789ab]]\n```\n`[[img:0123456789ab]]` text";
  assert.equal(rewriteSearchImageTokens(fenced, KNOWN), fenced);
});

test("rewriteSearchImageTokens is a no-op without tokens", () => {
  const text = "plain [[not an image]] text";
  assert.equal(rewriteSearchImageTokens(text, KNOWN), text);
});

test("holdBackPartialSearchImageToken trims a token still arriving", () => {
  for (const partial of [
    "[",
    "[[",
    "[[i",
    "[[img",
    "[[img:",
    "[[img:0123",
    "[[img:0123456789ab",
    "[[img:0123456789ab]",
  ]) {
    assert.equal(
      holdBackPartialSearchImageToken(`Golden ${partial}`, true),
      "Golden ",
    );
  }
  assert.equal(
    holdBackPartialSearchImageToken("done [[img:0123456789ab]]", true),
    "done [[img:0123456789ab]]",
  );
  assert.equal(
    holdBackPartialSearchImageToken("array[0][[1]", true),
    "array[0][[1]",
  );
  assert.equal(holdBackPartialSearchImageToken("see [link", true), "see [link");
  assert.equal(
    holdBackPartialSearchImageToken("Golden [[img:", false),
    "Golden [[img:",
  );
  // Inside closed code nothing is held back; an open fence is still streaming, so trimming is harmless.
  assert.equal(
    holdBackPartialSearchImageToken("```\n[[img:\n```", true),
    "```\n[[img:\n```",
  );
});

test("collectSearchImages reads only web_search wrappers and the signature is stable", () => {
  const parts = [
    { type: "text", text: "hi" },
    {
      type: "tool-call",
      toolName: "python",
      result: { text: "x", webImages: [OTHER] },
    },
    { type: "tool-call", toolName: "web_search", result: "plain" },
    {
      type: "tool-call",
      toolName: "web_search",
      result: { text: "t", webImages: [ENTRY] },
    },
    {
      type: "tool-call",
      toolName: "web_search",
      result: { text: "t", webImages: [ENTRY, OTHER] },
    },
  ];
  const images = collectSearchImages(parts);
  assert.deepEqual([...images.keys()], [ENTRY.id, OTHER.id]);
  const signature = searchImagesSignature(parts);
  assert.equal(signature, searchImagesSignature([...parts]));
  assert.deepEqual(
    [...parseSearchImagesSignature(signature).values()],
    [ENTRY, OTHER],
  );
  assert.equal(searchImagesSignature([{ type: "text" }]), "");
  assert.equal(parseSearchImagesSignature("").size, 0);
});

test("placeSubjectImages puts a subject's image under the paragraph naming it", () => {
  const shepherd = { ...ENTRY, id: "aaaaaaaaaaaa", subject: "German Shepherd" };
  const lab = { ...ENTRY, id: "bbbbbbbbbbbb", subject: "Labrador" };
  const pug = { ...ENTRY, id: "cccccccccccc", subject: "Pug" };
  const images = new Map([
    [shepherd.id, shepherd],
    [lab.id, lab],
    [pug.id, pug],
  ]);
  const text =
    "Top breeds:\n\n1. **German Shepherd** - loyal.\n2. **Labrador** - friendly.\n\nEnjoy!";
  const out = placeSubjectImages(text, images, false);
  // Pug is never named in the answer, so its image is left out rather than piled
  // at the end; the tool card still shows it.
  assert.equal(
    out,
    "Top breeds:\n\n1. **German Shepherd** - loyal.\n\n   [[img:aaaaaaaaaaaa]]\n2. **Labrador** - friendly.\n\n   [[img:bbbbbbbbbbbb]]\n\nEnjoy!",
  );
  assert.equal(
    placeSubjectImages(
      "## German Shepherd\nLoyal.\n\nNext",
      new Map([[shepherd.id, shepherd]]),
      false,
    ),
    "## German Shepherd\n\n[[img:aaaaaaaaaaaa]]\nLoyal.\n\nNext",
  );
  assert.equal(
    placeSubjectImages(
      "The German Shepherd is loyal.\nVery.\n\nNext",
      new Map([[shepherd.id, shepherd]]),
      false,
    ),
    "The German Shepherd is loyal.\nVery.\n\n[[img:aaaaaaaaaaaa]]\n\nNext",
  );
});

test("placeSubjectImages never splits a list item a model wrapped over lines", () => {
  const golden = { ...ENTRY, id: "aaaaaaaaaaaa", subject: "Golden Retriever" };
  const images = new Map([[golden.id, golden]]);
  // The screenshot bug: the card landed after "and", cutting the sentence in half.
  const wrapped =
    "4. **Golden Retriever:** Known for being gentle, patient, and\ndevoted, making them excellent family companions.\n5. **Poodle:** Clever.";
  const out = placeSubjectImages(wrapped, images, false);
  assert.ok(
    out.includes("patient, and\ndevoted, making them excellent family companions."),
    "the sentence must stay intact",
  );
  // Its own block, indented to the item's content column so the list keeps numbering.
  assert.ok(out.includes("family companions.\n\n   [[img:aaaaaaaaaaaa]]\n5. "));
});

test("placeSubjectImages respects tokens the model placed, streaming, and code", () => {
  const shepherd = { ...ENTRY, id: "aaaaaaaaaaaa", subject: "German Shepherd" };
  const images = new Map([[shepherd.id, shepherd]]);
  const placed = "German Shepherd\n\n[[img:aaaaaaaaaaaa]]\n\ndone";
  assert.equal(placeSubjectImages(placed, images, false), placed);
  assert.equal(
    placeSubjectImages("German Shepherd is", images, true),
    "German Shepherd is",
  );
  // Named only inside code, or not named at all: nothing is inserted anywhere.
  const code = "```\nGerman Shepherd\n```";
  assert.equal(placeSubjectImages(code, images, false), code);
  assert.equal(placeSubjectImages("Nothing here", images, false), "Nothing here");
  // Entries without a subject (web_search images) are never auto-placed.
  assert.equal(
    placeSubjectImages("Labrador", new Map([[ENTRY.id, ENTRY]]), false),
    "Labrador",
  );
  // Word boundaries: "Pug" must not match inside "Pugilist", so nothing is placed.
  const pug = { ...ENTRY, id: "cccccccccccc", subject: "Pug" };
  assert.equal(
    placeSubjectImages("Pugilist", new Map([[pug.id, pug]]), false),
    "Pugilist",
  );
  assert.equal(
    placeSubjectImages("A Pug naps.", new Map([[pug.id, pug]]), false),
    "A Pug naps.\n\n[[img:cccccccccccc]]",
  );
});

test("answerTextFromParts keeps the answer and drops reasoning", () => {
  const parts = [
    { type: "reasoning", text: "1. **Analyze the Request:** plan\n2. **Pick a tool:** web" },
    { type: "text", text: "1. **Labrador Retriever:** friendly." },
    { type: "tool-call", toolName: "web_search" },
    { type: "text", text: "2. **Poodle:** clever." },
  ];
  const answer = answerTextFromParts(parts);
  assert.equal(answer, "1. **Labrador Retriever:** friendly.\n\n2. **Poodle:** clever.");
  // The reasoning list must never become a subject to illustrate.
  assert.deepEqual(extractListSubjects(answer), ["Labrador Retriever", "Poodle"]);
  assert.equal(answerTextFromParts([{ type: "reasoning", text: "x" }]), "");
});

test("extractListSubjects reads the lead of each listed item", () => {
  const answer = [
    "Here are five popular breeds:",
    "",
    "1. **Labrador Retriever:** Friendly and outgoing.",
    "2. **French Bulldog** - Compact companion.",
    "3. German Shepherd: Loyal.",
    "- **Poodle** (Standard, Miniature, Toy)",
    "### Golden Retriever",
    "4. **labrador retriever:** duplicate",
    "5. **See https://akc.org for the full 2026 list:** not a subject",
    "6. **This is a long descriptive sentence about a dog that goes on:** no",
  ].join("\n");
  assert.deepEqual(extractListSubjects(answer), [
    "Labrador Retriever",
    "French Bulldog",
    "German Shepherd",
    "Poodle",
    "Golden Retriever",
  ]);
  // One item is not a list; prose is not a list.
  assert.deepEqual(extractListSubjects("1. **Only one:** thing"), []);
  assert.deepEqual(extractListSubjects("Dogs are great. Cats too."), []);
  // Section labels are skipped; procedures and code answers are never illustrated.
  assert.deepEqual(
    extractListSubjects("- **Pros:** fast\n- **Cons:** pricey\n- **Honda Civic:** a\n- **Mazda 3:** b"),
    ["Honda Civic", "Mazda 3"],
  );
  // A mostly-step list is a procedure and is never illustrated.
  assert.deepEqual(
    extractListSubjects(
      "1. **Install Python:** from python.org.\n2. **Open the terminal:** and run it.",
    ),
    [],
  );
  assert.deepEqual(
    extractListSubjects(
      "1. **Download the installer:** a\n2. **Run it:** b\n3. **Homebrew:** c\n4. **Verify the install:** d",
    ),
    [],
  );
  // A comparison that merely ends in a "Choose X if" line keeps its subjects.
  assert.deepEqual(
    extractListSubjects(
      "## Honda Civic\nSporty.\n\n## Toyota Corolla\nCalm.\n\n## Choose the Civic if\nyou want fun.",
    ),
    ["Honda Civic", "Toyota Corolla"],
  );
  assert.deepEqual(
    extractListSubjects(
      "1. **Labrador Retriever:** a\n2. **Poodle:** b\n\n```py\nprint(1)\n```",
    ),
    [],
  );
});

test("the inline card is block-level, so a list item cannot flow text around it", () => {
  // A list item styles its paragraphs `[&>p]:inline`. An inline card lands in the
  // middle of the sentence and the text wraps around it, which is what shipped once.
  const source = readFileSync(
    new URL("../src/components/assistant-ui/search-image.tsx", import.meta.url),
    "utf8",
  );
  const wrapper = /data-search-image=\{entry\.id\}/.test(source)
    ? source.slice(source.indexOf("if (!entry) return null;"))
    : "";
  assert.match(wrapper, /className="[^"]*\bflex\b/, "wrapper must not be inline");
  assert.match(wrapper, /empty:hidden/, "an unloaded card must not leave a gap");
});

test("searchResultText reaches the citations inside an image-bearing result", () => {
  const blocks = "Title: A\nURL: https://a.test\nSnippet: s";
  assert.equal(searchResultText(blocks), blocks);
  // The shape that used to make the whole Sources row vanish.
  assert.equal(searchResultText({ text: blocks, webImages: [ENTRY] }), blocks);
  assert.equal(searchResultText({ text: "t", images: [{ data: "", mimeType: "" }] }), "");
  assert.equal(searchResultText(undefined), "");
});

test("placeSubjectImages never splices a token into a code fence", () => {
  const pug = { ...ENTRY, id: "cccccccccccc", subject: "Pug" };
  const images = new Map([[pug.id, pug]]);
  const text =
    "The Pug is small. Here is code:\n```python\nprint(1)\n\nprint(2)\n```\nDone.";
  const out = placeSubjectImages(text, images, false);
  // The card goes under the paragraph that names the Pug, and the fence is
  // reached neither as an insertion point nor as somewhere to splice into.
  assert.ok(!/```python\nprint\(1\)\n\n\[\[img:/.test(out));
  assert.equal(
    out,
    "The Pug is small. Here is code:\n\n[[img:cccccccccccc]]\n```python\nprint(1)\n\nprint(2)\n```\nDone.",
  );
});

test("placeSubjectImages illustrates a subject once per message", () => {
  const pug = { ...ENTRY, id: "cccccccccccc", subject: "Pug" };
  const images = new Map([[pug.id, pug]]);
  const first = "A Pug is small.";
  assert.equal(
    placeSubjectImages(first, images, false, ""),
    "A Pug is small.\n\n[[img:cccccccccccc]]",
  );
  // The later text part sees what the earlier one already named, so it adds nothing.
  assert.equal(placeSubjectImages("The Pug again.", images, false, first), "The Pug again.");
});

test("missingListSubjects matches coverage on word boundaries", () => {
  const answer = "1. **Caterpillar:** a\n2. **Catalina Island:** b\n3. **Pug:** c";
  const covered = (subject: string) => [
    {
      type: "tool-call",
      toolName: "web_search",
      result: { text: "t", webImages: [{ ...ENTRY, subject }] },
    },
  ];
  // "cat" must not swallow Caterpillar or Catalina Island.
  assert.deepEqual(missingListSubjects(answer, covered("cat")), [
    "Caterpillar",
    "Catalina Island",
    "Pug",
  ]);
  // A longer subject that contains the item as a whole word still covers it.
  assert.deepEqual(missingListSubjects(answer, covered("a Pug dog")), [
    "Caterpillar",
    "Catalina Island",
  ]);
});

test("missingListSubjects leaves out what the model already fetched", () => {
  const answer =
    "1. **Pug:** small.\n2. **Golden Retriever:** kind.\n3. **Beagle:** loud.";
  assert.deepEqual(missingListSubjects(answer, [{ type: "text" }]), [
    "Pug",
    "Golden Retriever",
    "Beagle",
  ]);
  const partial = [
    {
      type: "tool-call",
      toolName: "web_search",
      result: {
        text: "t",
        webImages: [{ ...ENTRY, subject: "golden retriever dog" }],
      },
    },
  ];
  assert.deepEqual(missingListSubjects(answer, partial), ["Pug", "Beagle"]);
  const generic = [
    {
      type: "tool-call",
      toolName: "web_search",
      result: { text: "t", webImages: [ENTRY] },
    },
  ];
  assert.deepEqual(missingListSubjects(answer, generic), [
    "Pug",
    "Golden Retriever",
    "Beagle",
  ]);
  assert.deepEqual(missingListSubjects("Just prose.", [{ type: "text" }]), []);
});

test("thumbnails load from Studio's own endpoint, which the img policy allows", () => {
  const path = searchImagePath(ENTRY.id);
  assert.equal(path, "/api/inference/search-images/0123456789ab");
  const imgNode = { tagName: "img" } as Parameters<typeof safeMarkdownUrl>[2];
  assert.equal(safeMarkdownUrl(path, "src", imgNode), path);
  assert.equal(
    safeMarkdownUrl("blob:https://studio/abc", "src", imgNode),
    "blob:https://studio/abc",
  );
  // The deny rule this feature routes around stays in force for remote images.
  assert.equal(
    safeMarkdownUrl("https://img.example.com/x.png", "src", imgNode),
    null,
  );
  assert.equal(
    safeMarkdownUrl("//img.example.com/x.png", "src", imgNode),
    null,
  );
});

test("extractListSubjects stays linear on a bullet padded with whitespace", () => {
  // The single-regex form backtracked at ~O(n^3.5): 250 leading spaces in one
  // bullet blocked the render thread for 13 s, and this runs on every finished
  // answer while the setting is on.
  const padded = `- ${" ".repeat(5000)}${"x".repeat(80)}y\n- Beagle: small`;
  const started = Date.now();
  assert.deepEqual(extractListSubjects(padded), []);
  assert.ok(Date.now() - started < 1000, "must not backtrack");
});

test("extractListSubjects reads the same leads after the marker split", () => {
  const text = [
    "1. **German Shepherd:** loyal",
    "  10. Golden Retriever: great family dog",
    "3) Poodle (standard)",
    "* Labrador Retriever - friendly",
  ].join("\n");
  assert.deepEqual(extractListSubjects(text), [
    "German Shepherd",
    "Golden Retriever",
    "Poodle",
    "Labrador Retriever",
  ]);
});

test("placeSubjectImages never splices a token into display math", () => {
  const pug = { ...ENTRY, id: "cccccccccccc", subject: "Pug" };
  const images = new Map([[pug.id, pug]]);
  // A blank line inside `$$ ... $$` used to read as the end of the block, so the
  // card landed mid-equation and KaTeX was handed markup instead of LaTeX.
  const text = "The Pug weighs:\n$$\na = 1\n\nb = 2\n$$\nDone.";
  const out = placeSubjectImages(text, images, false);
  assert.ok(!/a = 1\n\n\s*\[\[img:/.test(out), "the equation must stay intact");
  assert.ok(out.includes("$$\na = 1\n\nb = 2\n$$"));
  assert.ok(out.includes("[[img:cccccccccccc]]"));
});

test("stripSearchImageTokens takes the tokens and their blank line", () => {
  assert.equal(
    stripSearchImageTokens("Golden Retriever\n\n[[img:0123456789ab]]\n\nDone."),
    "Golden Retriever\n\nDone.",
  );
  // Untouched without a token, and an unknown-length id is not a token.
  assert.equal(stripSearchImageTokens("plain answer"), "plain answer");
  assert.equal(stripSearchImageTokens("[[img:nothex]]"), "[[img:nothex]]");
  // Two cards in a row collapse to one gap, not three blank lines.
  assert.equal(
    stripSearchImageTokens("A\n\n[[img:0123456789ab]]\n\n[[img:abcdef012345]]\n\nB"),
    "A\n\nB",
  );
  // Inline, and inside code — the same rule rewriteSearchImageTokens follows.
  assert.equal(stripSearchImageTokens("see [[img:0123456789ab]] here"), "see  here");
  assert.equal(
    stripSearchImageTokens("```\n[[img:0123456789ab]]\n```"),
    "```\n[[img:0123456789ab]]\n```",
  );
});

test("placeSubjectImages steps past a subject that code mentions first", () => {
  const go = { ...ENTRY, id: "cccccccccccc", subject: "Go" };
  const images = new Map([[go.id, go]]);
  // The first occurrence is inside code, which shows no card. Abandoning the subject
  // there dropped the picture for the prose item that names it further down.
  const text = "Run `Go` first.\n\n1. **Go:** compiled and fast\n2. **Rust:** strict";
  const out = placeSubjectImages(text, images, false);
  assert.ok(out.includes("Run `Go` first."), "the snippet must stay as written");
  assert.ok(out.includes("compiled and fast\n\n   [[img:cccccccccccc]]"));
});

test("placeSubjectImages ignores an earlier part that only named a subject in code", () => {
  const go = { ...ENTRY, id: "cccccccccccc", subject: "Go" };
  const images = new Map([[go.id, go]]);
  // A code-only mention earlier carries no card, so it must not suppress this one.
  const out = placeSubjectImages("Go is compiled.", images, false, "Type `Go` to start.");
  assert.equal(out, "Go is compiled.\n\n[[img:cccccccccccc]]");
  // A real earlier mention still wins: one card per message.
  assert.equal(
    placeSubjectImages("Go is compiled.", images, false, "Go is a language."),
    "Go is compiled.",
  );
});
