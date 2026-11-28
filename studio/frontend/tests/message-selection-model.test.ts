// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Native selection snapshots use the existing HTML-to-Markdown converter, so
// the fidelity cases are pinned here without a DOM dependency.

import assert from "node:assert/strict";
import test from "node:test";

import {
  QUOTE_PRUNE_SELECTOR,
  formatQuoteMarkdown,
  keyboardSelectionTarget,
  keyboardTargetBlocksDocumentSelection,
  placeToolbar,
  sameSelectionPosition,
  selectionChangeAllowed,
  selectionEndpointsInside,
  shouldTrackKeyboardSelection,
  shortMessageId,
  unionSelectionRects,
} from "../src/components/assistant-ui/message-selection-model.ts";
import { htmlNodeToMarkdown } from "../src/features/chat/utils/rich-paste.ts";

type FakeNode = {
  nodeType: number;
  nodeName: string;
  childNodes: FakeNode[];
  textContent: string;
  getAttribute?: (name: string) => string | null;
};

const text = (value: string): FakeNode => ({
  nodeType: 3,
  nodeName: "#text",
  childNodes: [],
  textContent: value,
});

const element = (
  nodeName: string,
  childNodes: FakeNode[],
  attributes: Record<string, string> = {},
): FakeNode => ({
  nodeType: 1,
  nodeName,
  childNodes,
  textContent: childNodes.map((child) => child.textContent).join(""),
  getAttribute: (name) => attributes[name] ?? null,
});

const paragraph = (...children: FakeNode[]): FakeNode =>
  element("P", children);

test("a paragraph becomes plain Markdown", () => {
  assert.equal(
    htmlNodeToMarkdown(paragraph(text("Hello world"))),
    "Hello world",
  );
});

test("bold text and links keep their Markdown", () => {
  const node = paragraph(
    text("Use "),
    element("STRONG", [text("bold")]),
    text(" and "),
    element("A", [text("Unsloth")], { href: "https://unsloth.ai" }),
  );

  assert.equal(
    htmlNodeToMarkdown(node),
    "Use **bold** and [Unsloth](https://unsloth.ai)",
  );
});

test("lists become Markdown bullets", () => {
  const node = element("UL", [
    element("LI", [text("one")]),
    element("LI", [text("two")]),
  ]);

  assert.equal(htmlNodeToMarkdown(node), "- one\n- two");
});

test("tables become Markdown tables", () => {
  const node = element("TABLE", [
    element("THEAD", [
      element("TR", [
        element("TH", [text("A")]),
        element("TH", [text("B")]),
      ]),
    ]),
    element("TBODY", [
      element("TR", [
        element("TD", [text("1")]),
        element("TD", [text("2")]),
      ]),
    ]),
  ]);

  assert.equal(
    htmlNodeToMarkdown(node),
    "| A | B |\n| --- | --- |\n| 1 | 2 |",
  );
});

test("code blocks become fenced Markdown", () => {
  const node = element("PRE", [
    element("CODE", [text("const x = 1;")], { class: "language-ts" }),
  ]);

  assert.equal(htmlNodeToMarkdown(node), "```ts\nconst x = 1;\n```");
});

test("quotes prefix every line, including blank lines and fences", () => {
  const quote = formatQuoteMarkdown(
    "Intro\n\n```ts\nconst x = 1;\n```",
    "Quoted from assistant response 2 (12345678):",
  );

  assert.equal(
    quote,
    [
      "Quoted from assistant response 2 (12345678):",
      "> Intro",
      ">",
      "> ```ts",
      "> const x = 1;",
      "> ```",
    ].join("\n"),
  );
});

test("an empty quote still produces a valid blockquote", () => {
  assert.equal(formatQuoteMarkdown("   ", "Source:"), "Source:\n>");
});

test("message ids use a short, readable prefix", () => {
  assert.equal(shortMessageId("12345678-90ab-cdef"), "12345678");
});

test("both selection endpoints must belong to the thread", () => {
  const inside = {} as Node;
  const outside = {} as Node;
  const contains = (node: Node) => node === inside;

  assert.equal(selectionEndpointsInside(inside, inside, contains), true);
  assert.equal(selectionEndpointsInside(inside, outside, contains), false);
  assert.equal(selectionEndpointsInside(outside, inside, contains), false);
  assert.equal(selectionEndpointsInside(null, inside, contains), false);
  assert.equal(selectionEndpointsInside(inside, null, contains), false);
  assert.equal(selectionEndpointsInside(null, null, contains), false);

  // A collapsed caret has both endpoints at the same contained position.
  assert.equal(selectionEndpointsInside(inside, inside, contains), true);
});

test("null and non-element keyboard targets are normalised safely", () => {
  assert.equal(keyboardSelectionTarget(null), null);
  assert.equal(keyboardSelectionTarget({}), null);
  assert.equal(keyboardSelectionTarget("not-an-element"), null);

  const target = { closest: () => null };
  assert.equal(keyboardSelectionTarget(target), target);

  assert.equal(
    keyboardTargetBlocksDocumentSelection(null, "textarea"),
    false,
  );
  assert.equal(
    keyboardTargetBlocksDocumentSelection(
      { closest: () => null } as unknown as Element,
      "textarea",
    ),
    false,
  );
  assert.equal(
    keyboardTargetBlocksDocumentSelection(
      { closest: () => ({}) } as unknown as Element,
      "textarea",
    ),
    true,
  );
});

test("keyboard selection only follows real selection-extending actions", () => {
  const base = {
    key: "ArrowLeft",
    shiftKey: true,
    ctrlOrMetaKey: false,
    targetBlocksDocumentSelection: false,
    targetInThread: true,
    targetIsDocumentBody: false,
    selectionInThread: true,
  };

  assert.equal(shouldTrackKeyboardSelection(base), true);
  for (const key of ["Home", "End", "PageUp", "PageDown"]) {
    assert.equal(shouldTrackKeyboardSelection({ ...base, key }), true);
  }

  for (const key of ["Tab", "Enter", "A", "Delete", " "]) {
    assert.equal(shouldTrackKeyboardSelection({ ...base, key }), false);
  }

  assert.equal(
    shouldTrackKeyboardSelection({
      ...base,
      targetBlocksDocumentSelection: true,
    }),
    false,
  );
  assert.equal(
    shouldTrackKeyboardSelection({ ...base, selectionInThread: false }),
    false,
  );
  assert.equal(
    shouldTrackKeyboardSelection({ ...base, targetInThread: false }),
    false,
  );
  assert.equal(
    shouldTrackKeyboardSelection({
      ...base,
      targetInThread: false,
      targetIsDocumentBody: true,
    }),
    true,
  );
});

test("Ctrl/Cmd+A only tracks a thread-rooted current selection", () => {
  const base = {
    key: "a",
    shiftKey: false,
    ctrlOrMetaKey: true,
    targetBlocksDocumentSelection: false,
    targetInThread: true,
    targetIsDocumentBody: false,
    selectionInThread: true,
  };

  assert.equal(shouldTrackKeyboardSelection(base), true);
  assert.equal(
    shouldTrackKeyboardSelection({ ...base, key: "A" }),
    true,
  );

  assert.equal(
    shouldTrackKeyboardSelection({
      ...base,
      targetBlocksDocumentSelection: true,
    }),
    false,
  );
  assert.equal(
    shouldTrackKeyboardSelection({ ...base, targetInThread: false }),
    false,
  );
  assert.equal(
    shouldTrackKeyboardSelection({
      ...base,
      targetInThread: false,
      targetIsDocumentBody: true,
    }),
    false,
  );
  assert.equal(
    shouldTrackKeyboardSelection({ ...base, selectionInThread: false }),
    false,
  );
});

test("selection changes are gated by the pointer origin", () => {
  assert.equal(selectionChangeAllowed(null, false), true);
  assert.equal(selectionChangeAllowed(true, false), true);
  assert.equal(selectionChangeAllowed(true, true), true);
  assert.equal(selectionChangeAllowed(false, false), false);
  assert.equal(selectionChangeAllowed(false, true), false);
  assert.equal(selectionChangeAllowed(null, true), false);
});

test("selection identity compares nodes and offsets, not text", () => {
  const first = {} as Node;
  const second = {} as Node;
  const a = {
    startContainer: first,
    startOffset: 2,
    endContainer: first,
    endOffset: 8,
  };

  assert.equal(sameSelectionPosition(a, { ...a }), true);
  assert.equal(sameSelectionPosition(a, { ...a, startOffset: 3 }), false);
  assert.equal(sameSelectionPosition(a, { ...a, endOffset: 9 }), false);
  assert.equal(
    sameSelectionPosition(a, { ...a, startContainer: second }),
    false,
  );
  assert.equal(
    sameSelectionPosition(a, { ...a, endContainer: second }),
    false,
  );
});

test("selection bounds union every selected line", () => {
  assert.deepEqual(
    unionSelectionRects([
      { top: 100, left: 100, right: 700, bottom: 120 },
      { top: 124, left: 100, right: 400, bottom: 144 },
    ]),
    {
      top: 100,
      left: 100,
      right: 700,
      bottom: 144,
      width: 600,
      height: 44,
    },
  );
  assert.equal(unionSelectionRects([]), null);
  assert.equal(
    unionSelectionRects([
      { top: 10, left: 10, right: 10, bottom: 10 },
    ]),
    null,
  );
});

test("toolbar placement centers on the selection and stays inside the viewport", () => {
  const anchor = {
    top: 100,
    left: 100,
    right: 200,
    bottom: 120,
    width: 100,
    height: 20,
  };
  const size = { width: 100, height: 32 };
  const viewport = { left: 0, top: 0, width: 800, height: 600 };

  assert.deepEqual(placeToolbar(anchor, size, viewport), {
    left: 100,
    top: 60,
  });

  assert.equal(
    placeToolbar({ ...anchor, left: 0, width: 20 }, size, viewport)?.left,
    8,
  );
  assert.equal(
    placeToolbar({ ...anchor, left: 780, width: 20 }, size, viewport)?.left,
    692,
  );
});

test("toolbar placement moves below when there is no room above", () => {
  const anchor = {
    top: 20,
    left: 100,
    right: 200,
    bottom: 40,
    width: 100,
    height: 20,
  };
  const size = { width: 100, height: 32 };

  assert.deepEqual(
    placeToolbar(anchor, size, { left: 0, top: 0, width: 800, height: 600 }),
    {
      left: 100,
      top: 48,
    },
  );
});

test("toolbar placement uses the chat viewport origin", () => {
  const viewport = { left: 300, top: 50, width: 600, height: 500 };
  const size = { width: 100, height: 32 };

  assert.deepEqual(
    placeToolbar(
      {
        top: 200,
        left: 400,
        right: 500,
        bottom: 220,
        width: 100,
        height: 20,
      },
      size,
      viewport,
    ),
    { left: 400, top: 160 },
  );

  assert.deepEqual(
    placeToolbar(
      {
        top: 80,
        left: 400,
        right: 500,
        bottom: 100,
        width: 100,
        height: 20,
      },
      size,
      viewport,
    ),
    { left: 400, top: 108 },
  );

  const anchor = {
    top: 200,
    left: 300,
    right: 320,
    bottom: 220,
    width: 20,
    height: 20,
  };
  assert.equal(placeToolbar(anchor, size, viewport)?.left, 308);
  assert.equal(
    placeToolbar(
      { ...anchor, left: 880, right: 900 },
      size,
      viewport,
    )?.left,
    792,
  );
});

test("toolbar placement never overlaps the selection when neither side fits", () => {
  const size = { width: 40, height: 32 };
  const shortViewport = { left: 0, top: 0, width: 100, height: 60 };

  assert.equal(
    placeToolbar(
      {
        top: 30,
        left: 40,
        right: 60,
        bottom: 50,
        width: 20,
        height: 20,
      },
      size,
      shortViewport,
    ),
    null,
  );

  assert.equal(
    placeToolbar(
      {
        top: 40,
        left: 100,
        right: 300,
        bottom: 560,
        width: 200,
        height: 520,
      },
      { width: 100, height: 32 },
      { left: 0, top: 0, width: 800, height: 600 },
    ),
    null,
  );
});

test("toolbar placement returns null when it cannot fit horizontally", () => {
  assert.equal(
    placeToolbar(
      {
        top: 30,
        left: 40,
        right: 60,
        bottom: 50,
        width: 20,
        height: 20,
      },
      { width: 120, height: 20 },
      { left: 0, top: 0, width: 100, height: 60 },
    ),
    null,
  );
});

test("rendered Markdown chrome is pruned from quotes", () => {
  for (const selector of [
    "button",
    "[data-streamdown='code-block-header']",
    "[data-streamdown='code-block-actions']",
    "[data-streamdown='mermaid-block-actions']",
    "[data-streamdown='table-fullscreen']",
    ".aui-md-table-actions",
  ]) {
    assert.ok(
      QUOTE_PRUNE_SELECTOR.includes(selector),
      `${selector} should be pruned`,
    );
  }
});
