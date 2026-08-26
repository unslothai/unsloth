// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

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
  precedingTextForMessagePart,
  rewriteSearchImageTokens,
  searchImagePath,
  searchImagesSignature,
  searchResultText,
  stripSearchImageTokens,
} from "../src/features/chat/search-images/search-images.ts";
import { toolArgText } from "../src/components/assistant-ui/tool-arg-text.ts";
import { safeMarkdownUrl } from "../src/lib/safe-markdown-url.ts";

const ENTRY = {
  id: "0123456789ab",
  title: "Golden Retriever",
  domain: "akc.org",
  source: "https://www.akc.org/golden",
};
const OTHER = { ...ENTRY, id: "abcdef012345", title: "Labrador" };
const KNOWN = new Set([ENTRY.id, OTHER.id]);

const adapterSource = readFileSync(
  fileURLToPath(new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url)),
  "utf8",
);

function liftAdapterFunction(opener: string): string {
  const start = adapterSource.indexOf(opener);
  assert.ok(start >= 0, `${opener} is no longer defined in chat-adapter.ts`);
  const end = adapterSource.indexOf("\n}", start);
  assert.ok(end > start, `could not find the end of ${opener}`);
  return adapterSource.slice(start, end + 2).replace("export function", "function");
}

const privateContentPredicateJs = ts.transpileModule(
  [
    liftAdapterFunction("export function messagesContainImage("),
    liftAdapterFunction("function isPrivateMediaPart("),
    liftAdapterFunction("export function messagesUsePrivateContent("),
    "return messagesUsePrivateContent;",
  ].join("\n\n"),
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
).outputText;

const messagesUsePrivateContent = new Function(privateContentPredicateJs)() as (
  messages: unknown[],
) => boolean;

test("private message history blocks automatic image lookup", () => {
  assert.equal(
    messagesUsePrivateContent([
      {
        role: "user",
        content: [{ type: "text", text: "read this" }],
        attachments: [
          {
            type: "document",
            content: [{ type: "text", text: "private product name" }],
          },
        ],
      },
      { role: "user", content: [{ type: "text", text: "list those names" }] },
    ]),
    true,
  );
  assert.equal(
    messagesUsePrivateContent([
      {
        role: "user",
        content: [{ type: "image", image: "data:image/png;base64,cHJpdmF0ZQ==" }],
      },
    ]),
    true,
  );
  assert.equal(
    messagesUsePrivateContent([
      {
        role: "user",
        content: [],
        attachments: [
          {
            type: "image",
            content: [
              { type: "image", image: "data:image/png;base64,cHJpdmF0ZQ==" },
            ],
          },
        ],
      },
    ]),
    true,
  );
  assert.equal(
    messagesUsePrivateContent([
      {
        role: "assistant",
        content: [{ type: "tool-call", toolName: "search_knowledge_base" }],
      },
    ]),
    true,
  );
});

test("ordinary text and web search history allow automatic image lookup", () => {
  assert.equal(
    messagesUsePrivateContent([
      { role: "user", content: [{ type: "text", text: "list dog breeds" }] },
      {
        role: "assistant",
        content: [{ type: "tool-call", toolName: "web_search" }],
      },
    ]),
    false,
  );
});

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

test("equal text parts derive preceding text from the current part position", () => {
  const text = "A Pug is small.";
  const parts = [
    { type: "text", text },
    { type: "tool-call", toolName: "web_search" },
    { type: "text", text },
  ];
  const precedingText = precedingTextForMessagePart(parts, 2);
  assert.equal(precedingText, text);

  const pug = { ...ENTRY, id: "cccccccccccc", subject: "Pug" };
  assert.equal(
    placeSubjectImages(text, new Map([[pug.id, pug]]), false, precedingText),
    text,
  );
});

test("a token in a later text part prevents an earlier duplicate card", () => {
  const pug = { ...ENTRY, id: "cccccccccccc", subject: "Pug" };
  const images = new Map([[pug.id, pug]]);
  const first = "A Pug is small.";
  const later = "Here it is.\n\n[[img:cccccccccccc]]";
  assert.equal(placeSubjectImages(first, images, false, "", [first, later]), first);
  const codeOnly = "```\n[[img:cccccccccccc]]\n```";
  assert.equal(
    placeSubjectImages(first, images, false, "", [first, codeOnly]),
    `${first}\n\n[[img:cccccccccccc]]`,
  );
  const unclosedFence = `${first}\n\n\`\`\``;
  assert.equal(
    placeSubjectImages(unclosedFence, images, false, "", [unclosedFence, later]),
    unclosedFence,
  );
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

test("placeSubjectImages still places a subject whose token only sits in code", () => {
  const pug = { ...ENTRY, id: "cccccccccccc", subject: "Pug" };
  const images = new Map([[pug.id, pug]]);
  // rewriteSearchImageTokens leaves a token inside a fence alone, so it renders no
  // picture; counting it as "already placed" meant the answer showed nothing at all.
  const text = "Here is the token:\n\n```\n[[img:cccccccccccc]]\n```\n\nThe Pug is small.";
  const out = placeSubjectImages(text, images, false);
  assert.ok(out.includes("```\n[[img:cccccccccccc]]\n```"), "the fence stays as written");
  assert.ok(
    out.endsWith("The Pug is small.\n\n[[img:cccccccccccc]]"),
    `a real card must still be placed, got ${JSON.stringify(out)}`,
  );
  // A token the model placed in prose still wins, so nothing is placed twice.
  const placed = "The Pug is small.\n\n[[img:cccccccccccc]]";
  assert.equal(placeSubjectImages(placed, images, false), placed);
});

// A replayed turn keeps its tokens in the model's context, but they resolve
// against the message whose search produced them: repeating one in a later
// answer renders nothing at all. So history is replayed without them, and the
// model has to search again to get a token this message can resolve.
const sanitizeAssistantReplayJs = ts.transpileModule(
  [
    liftAdapterFunction("function sanitizeAssistantReplayText("),
    "return sanitizeAssistantReplayText;",
  ].join("\n\n"),
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
).outputText;

const sanitizeAssistantReplayText = new Function(
  "stripSearchImageTokens",
  sanitizeAssistantReplayJs,
)(stripSearchImageTokens) as (text: string) => string;

test("a token from an earlier turn is unresolvable in the message that repeats it", () => {
  const earlier = [
    {
      type: "tool-call",
      toolName: "web_search",
      result: { text: `- [[img:${ENTRY.id}]] ${ENTRY.title}`, webImages: [ENTRY] },
    },
  ];
  const repeatText = `Here it is again:\n\n[[img:${ENTRY.id}]]`;
  const later = [{ type: "text", text: repeatText }];

  assert.equal(collectSearchImages(earlier).size, 1);
  assert.equal(collectSearchImages(later).size, 0);
  // Silently nothing, which is why the token must not survive into replay.
  assert.equal(
    rewriteSearchImageTokens(repeatText, collectSearchImages(later)),
    "Here it is again:\n\n",
  );
});

test("replayed assistant text carries no image tokens", () => {
  assert.equal(
    sanitizeAssistantReplayText(`The retriever:\n\n[[img:${ENTRY.id}]]\n\nand more.`),
    "The retriever:\n\nand more.",
  );
  // The audio placeholder this shares the chokepoint with is untouched.
  assert.equal(
    sanitizeAssistantReplayText("clip: data:audio/mp3;base64,QUJD"),
    "clip: [audio]",
  );
  assert.equal(sanitizeAssistantReplayText("plain answer"), "plain answer");
});

test("a replayed web_search result carries no image tokens either", () => {
  const branch = adapterSource.slice(
    adapterSource.indexOf("function serializeToolResultPart("),
    adapterSource.indexOf("function sanitizeAssistantReplayText("),
  );
  assert.ok(branch.length > 0, "serializeToolResultPart moved");
  assert.match(
    branch,
    /isSearchImagesToolResult\(result\)\s*\?\s*stripSearchImageTokens\(result\.text\)/,
    "the replayed web_search result must be stripped of its tokens",
  );
});

// Search images is read by the backend out of SQLite when the tool schema is
// picked, not carried in the request, and the mirror to /api/chat/settings is a
// 400 ms trailing-edge debounce -- so a message sent right after the toggle used
// to run on the previous value. The store and the adapter cannot be imported in
// a bare node test (a .tsx barrel sits in both graphs), so these pin the source
// the way the sibling store tests do.
const storeSource = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/stores/chat-runtime-store.ts", import.meta.url),
  ),
  "utf8",
);

test("a queued settings patch is sent before a run reads it", () => {
  const flush = storeSource.slice(
    storeSource.indexOf("export async function flushPendingChatSettings("),
  );
  assert.ok(flush.length > 0, "flushPendingChatSettings is gone");
  const body = flush.slice(0, flush.indexOf("\n}\n") + 2);
  // Nothing queued and nothing in flight is the common case and must not cost a
  // send anything.
  assert.match(body, /if \(!queued && unsettledFlushes === 0\) return;/);
  // The debounce is cut short rather than waited out.
  assert.match(body, /clearTimeout\(pendingTimer\)/);
  // Bounded: a wedged PATCH must not hold the composer open.
  assert.match(body, /Promise\.race\(/);
  assert.match(storeSource, /const SETTINGS_FLUSH_TIMEOUT_MS = \d+;/);
});

test("a patch already handed to the server is waited for too", () => {
  // Between the debounce firing and the response, pendingPatch and pendingTimer
  // are both empty while the value the backend reads is still the old one, so
  // the fast path above cannot be decided from those two alone.
  const enqueue = storeSource.slice(
    storeSource.indexOf("function enqueueSettingsFlush("),
  );
  const body = enqueue.slice(0, enqueue.indexOf("\n}\n") + 2);
  assert.match(body, /unsettledFlushes \+= 1;/);
  assert.match(body, /\.finally\(\(\) => \{\s*unsettledFlushes -= 1;/);
  // Chained onto the same queue, so two flushes cannot overlap.
  assert.match(body, /inflightFlush = inflightFlush/);
  // The debounce goes through it rather than posting its own chain.
  const schedule = storeSource.slice(
    storeSource.indexOf("function scheduleSettingsFlush("),
  );
  assert.match(
    schedule.slice(0, schedule.indexOf("\n}\n")),
    /void enqueueSettingsFlush\(\);/,
  );
});

test("the run flushes those settings after it hydrates and before it sends", () => {
  const start = adapterSource.indexOf("await flushPendingChatSettings();");
  assert.ok(start > 0, "the run no longer flushes pending chat settings");
  const hydrate = adapterSource.indexOf(
    "await useChatRuntimeStore.getState().hydratePersistedSettings();",
  );
  // After, not before: a setting changed while the initial GET was still out is
  // held back and only reaches the debounce when hydration replays it.
  assert.ok(hydrate > 0 && hydrate < start, "the flush must follow the hydrate");
  // Awaited, not fired and forgotten, or the run races the patch it just sent.
  assert.match(
    adapterSource.slice(start - 6, start + 34),
    /await flushPendingChatSettings\(\);/,
  );
});

test("the automatic lookup reads this run's approval level, not the open chat's", () => {
  const gate = adapterSource.slice(
    adapterSource.indexOf("const toolCallsNeedApproval ="),
    adapterSource.indexOf("const subjects = missingListSubjects("),
  );
  assert.ok(gate.length > 0, "the automatic-lookup gate moved");
  // confirmToolCalls and permissionMode are per-chat, and a background run can
  // finish after the user has opened a chat on "auto": reading the store here
  // would look images up for an answer whose own chat asked to be consulted.
  assert.match(
    gate,
    /const toolCallsNeedApproval = confirmToolCalls && permissionMode === "ask";/,
  );
  assert.doesNotMatch(gate, /getState\(\)\.confirmToolCalls/);
  assert.doesNotMatch(gate, /getState\(\)\.permissionMode/);
  // Both come out of the runtime the run captured before it started.
  const captured = adapterSource.slice(
    adapterSource.indexOf("let runtime = useChatRuntimeStore.getState();"),
    adapterSource.indexOf("const toolCallsNeedApproval ="),
  );
  assert.match(captured, /\n\s*confirmToolCalls,\n/);
  assert.match(captured, /\n\s*permissionMode,\n/);
});

test("every export path strips the tokens, not just the clipboard", () => {
  // The tokens are renderer markup. A per-message export, a reply saved as a
  // project source and a whole chat saved as one all reach disk (or back into
  // model context) by a different route than the copy button.
  const threadSource = readFileSync(
    fileURLToPath(
      new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    ),
    "utf8",
  );
  const exporter = threadSource.slice(
    threadSource.indexOf("async function exportMessageMarkdown("),
    threadSource.indexOf("const AssistantActionBar"),
  );
  assert.ok(exporter.length > 0, "exportMessageMarkdown moved");
  assert.match(
    exporter,
    /downloadFile\(\s*(\/\/[^\n]*\n\s*)*stripSearchImageTokens\(content\)/,
    "the per-message markdown export must strip the tokens",
  );
  assert.match(
    threadSource,
    /stripSearchImageTokens\(\s*replySourceMarkdown\(/,
    "a reply saved as a project source must strip the tokens",
  );
  const dialogSource = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  const saveSource = dialogSource.slice(
    dialogSource.indexOf("async function saveConversationAsProjectSource("),
    dialogSource.indexOf("export async function saveChatItemAsProjectSource("),
  );
  assert.ok(saveSource.length > 0, "saveConversationAsProjectSource moved");
  assert.match(
    saveSource,
    /stripSearchImageTokens\(messageToMarkdown\(msg\)\)/,
    "a chat saved as a project source must strip the tokens",
  );
});

test("the web search card survives a query that is not a string", () => {
  // Local models emit `"query": 42` and `"query": {}` routinely, and .trim() on
  // one threw straight through the renderer.
  const cardSource = readFileSync(
    fileURLToPath(
      new URL(
        "../src/components/assistant-ui/tool-ui-web-search.tsx",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  const args = cardSource.slice(
    cardSource.indexOf("const query ="),
    cardSource.indexOf("const isUrlFetch ="),
  );
  assert.ok(args.length > 0, "the args derivation moved");
  // Passed in because the derivation now calls it: one coercion shared by every
  // card (see tool-card-arg-coercion.test.ts).
  const derived = new Function(
    "args",
    "toolArgText",
    ts.transpileModule(`${args}\nreturn [query.trim(), url];`, {
      compilerOptions: { target: ts.ScriptTarget.ES2022 },
    }).outputText,
  ) as (args: unknown, coerce: typeof toolArgText) => [string, string];
  const derive = (args: unknown): [string, string] => derived(args, toolArgText);
  assert.deepEqual(derive({ query: 42 }), ["42", ""]);
  assert.deepEqual(derive({ query: null, url: 7 }), ["", "7"]);
  assert.deepEqual(derive({}), ["", ""]);
  assert.deepEqual(derive({ query: " dogs " }), ["dogs", ""]);
});

test("a thumbnail response that lands after the id changed is ignored", () => {
  // Render falls through to idle for a state written under the previous id, and
  // the effect has no reason to run again: a skeleton that never resolves.
  const source = readFileSync(
    fileURLToPath(
      new URL("../src/components/assistant-ui/search-image.tsx", import.meta.url),
    ),
    "utf8",
  );
  const effect = source.slice(
    source.indexOf("authFetch(searchImagePath(id)"),
    source.indexOf("function useNearViewport"),
  );
  assert.ok(effect.length > 0, "the thumbnail effect moved");
  const notOk = effect.slice(effect.indexOf("if (!response.ok)"));
  assert.match(
    notOk.slice(0, notOk.indexOf("return;") + 7),
    /controller\.signal\.aborted/,
    "the not-ok branch must check the abort like the success branch does",
  );
});


// A voice note or a clip is the user's own content in the sense the text-attachment rule
// already covers, and the answer's subjects go to external image engines. Both arrive
// with no text part beside them: AudioAttachmentAdapter.send emits exactly
// [{ type: "audio", ... }], so nothing else in the predicate could have caught them.
test("an audio or video input blocks automatic image lookup", () => {
  assert.equal(
    messagesUsePrivateContent([
      {
        role: "user",
        content: [],
        attachments: [
          {
            type: "file",
            content: [
              { type: "audio", audio: { data: "cHJpdmF0ZQ==", format: "wav" } },
            ],
          },
        ],
      },
      { role: "user", content: [{ type: "text", text: "list what it mentions" }] },
    ]),
    true,
    "a recording the user attached must not have its subjects searched for",
  );
  assert.equal(
    messagesUsePrivateContent([
      {
        role: "user",
        content: [{ type: "audio", audio: "cHJpdmF0ZQ==" }],
      },
    ]),
    true,
    "the compare view puts audio on the message itself, not in an attachment",
  );
  assert.equal(
    messagesUsePrivateContent([
      {
        role: "user",
        content: [],
        attachments: [
          {
            type: "file",
            content: [{ type: "file", data: "cHJpdmF0ZQ==", mimeType: "video/mp4" }],
          },
        ],
      },
    ]),
    true,
    "video reaches the model the same way and leaks the same way",
  );
  // Matched on the type, not on a payload that parses: a clip whose base64 is malformed
  // is still a clip, and failing open on it is the wrong way round.
  assert.equal(
    messagesUsePrivateContent([
      { role: "user", content: [{ type: "audio", audio: { data: "", format: "wav" } }] },
    ]),
    true,
    "an audio part with an unusable payload is still private",
  );
  // The complement: a plain typed question is what the lookup exists for.
  assert.equal(
    messagesUsePrivateContent([
      { role: "user", content: [{ type: "text", text: "name three dog breeds" }] },
    ]),
    false,
    "an ordinary text turn must still be eligible",
  );
  // A non-video file part is not media this rule speaks for; the attachment rule below
  // it already decides those on their text.
  assert.equal(
    messagesUsePrivateContent([
      { role: "user", content: [{ type: "file", data: "eA==", mimeType: "application/pdf" }] },
    ]),
    false,
    "the video test is on the mime type, not on being a file part",
  );
});
