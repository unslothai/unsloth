// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A web_search call carrying a `url` fetches that exact page, so the backend gates it even in
// "Approve for me" (see _web_search_fetches_url in the backend's tools.py). The card is pinned
// open while it waits, but the expanded content of a *running* call said only
// `Reading example.com…`: the path, query and fragment the decision turns on are invisible, and an
// innocuous and a sensitive url on the same host read identically. The finished card already shows
// the url -- that half landed on main with #5787 -- so this test owns the parked case only.
//
// The card is .tsx and this runner cannot execute JSX, so its claims go through the TypeScript AST
// rather than a source substring: a substring passes on broken code and fails on a reformat, which
// is backwards. Rendered behaviour lives in tests/studio/playwright_tool_activity.py.

import assert from "node:assert/strict";
import test from "node:test";

import ts from "typescript";

import { readSrc } from "./helpers/kit.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

const CARD_PATH = "components/assistant-ui/tool-ui-web-search.tsx";
const URL_SLOT = "tool-web-fetch-url";

const source = ts.createSourceFile(
  CARD_PATH,
  readSrc(CARD_PATH),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

function walk(node: ts.Node, visit: (node: ts.Node) => void): void {
  visit(node);
  node.forEachChild((child) => walk(child, visit));
}

function find<T extends ts.Node>(
  root: ts.Node,
  match: (node: ts.Node) => node is T,
): T[] {
  const hits: T[] = [];
  walk(root, (node) => {
    if (match(node)) hits.push(node);
  });
  return hits;
}

/** The opening tag of every element named `name`, paired or self-closing. */
function tagsNamed(name: string): ts.JsxOpeningLikeElement[] {
  const hits: ts.JsxOpeningLikeElement[] = [];
  walk(source, (node) => {
    const opening = openingTag(node);
    if (
      opening &&
      ts.isIdentifier(opening.tagName) &&
      opening.tagName.text === name
    ) {
      hits.push(opening);
    }
  });
  return hits;
}

/** The attribute named `name`, whichever of its two initializer forms it uses. */
function attributeOf(
  opening: ts.JsxOpeningLikeElement,
  name: string,
): ts.JsxAttribute | undefined {
  return opening.attributes.properties.find(
    (property): property is ts.JsxAttribute =>
      ts.isJsxAttribute(property) && property.name.getText(source) === name,
  );
}

/** The literal `data-slot` value of an opening tag, if it has one. */
function dataSlotOf(opening: ts.JsxOpeningLikeElement): string | undefined {
  const initializer = attributeOf(opening, "data-slot")?.initializer;
  return initializer && ts.isStringLiteral(initializer)
    ? initializer.text
    : undefined;
}

function classNameOf(opening: ts.JsxOpeningLikeElement): string {
  return opening.attributes.properties
    .filter(
      (property): property is ts.JsxAttribute =>
        ts.isJsxAttribute(property) &&
        property.name.getText(source) === "className",
    )
    .map((attribute) => attribute.initializer?.getText(source) ?? "")
    .join(" ");
}

/**
 * The condition a node renders under. Walked from the node up rather than searched for: the row
 * has to sit inside a condition naming the running state, and a sibling's condition would
 * satisfy a source-wide scan.
 */
function guardCondition(node: ts.Node): ts.Expression | undefined {
  for (let parent = node.parent; parent; parent = parent.parent) {
    if (ts.isConditionalExpression(parent)) {
      return parent.condition;
    }
  }
  return undefined;
}

/** The element whose opening tag carries this data-slot, so its full extent is comparable. */
function slotElement(slot: string): ts.JsxElement | undefined {
  return find(
    source,
    (node): node is ts.JsxElement =>
      ts.isJsxElement(node) && dataSlotOf(node.openingElement) === slot,
  )[0];
}

function requireRow(): ts.JsxElement {
  const row = slotElement(URL_SLOT);
  assert.ok(
    row,
    `the expanded card has no ${URL_SLOT} row, so a url fetch parked on Allow/Deny still shows only the hostname it is reading`,
  );
  return row;
}

const [contentTag] = tagsNamed("ToolFallbackContent");

test("the parked web-fetch card shows the complete raw url", () => {
  const row = requireRow();

  assert.ok(
    contentTag && row.getStart(source) > contentTag.getEnd(),
    "the url must render inside the expanded content; the collapsed trigger stays hostname-only",
  );

  // Rendered while the call runs: that is when it is parked on the decision.
  const guard = guardCondition(row.openingElement);
  assert.ok(
    guard,
    "the url row renders unconditionally, so the completed card would show it twice",
  );
  const condition = guard.getText(source);
  assert.match(
    condition,
    /isRunning/,
    "the row must cover the running (parked) arm, which is where the decision is made",
  );
  assert.match(
    condition,
    /url/,
    "the row must be gated on a url being present, or a plain search gains an empty URL line",
  );

  // The raw argument, not the parsed host the trigger names.
  const rendered = find(row, ts.isJsxExpression).map(
    (expression) => expression.expression?.getText(source) ?? "",
  );
  assert.ok(
    rendered.some(
      (expression) =>
        expression.includes("url") && !expression.includes("safeUrl"),
    ),
    `the row must render the raw argument; it renders ${JSON.stringify(rendered)}`,
  );
});

test("the raw url is inert text that wraps inside the card", () => {
  const row = requireRow();
  const code = find(
    row,
    (node): node is ts.JsxElement =>
      ts.isJsxElement(node) &&
      ts.isIdentifier(node.openingElement.tagName) &&
      node.openingElement.tagName.text === "code",
  );
  assert.equal(
    code.length,
    1,
    "the raw url must be rendered as one <code> element",
  );

  const dir = attributeOf(code[0].openingElement, "dir");
  assert.equal(
    dir?.initializer?.getText(source),
    '"ltr"',
    "a url must keep a stable left-to-right reading direction",
  );

  assert.match(
    classNameOf(code[0].openingElement),
    /break-all/,
    "a long url must wrap instead of clipping",
  );
  // A flex child defaults to min-width: auto and would overflow instead of shrinking.
  const rowClasses = [
    classNameOf(row.openingElement),
    ...find(row, ts.isJsxElement)
      .filter(
        (element) =>
          ts.isIdentifier(element.openingElement.tagName) &&
          element.openingElement.tagName.text === "ScrollPane",
      )
      .map((element) => classNameOf(element.openingElement)),
  ].join(" ");
  assert.match(
    rowClasses,
    /min-w-0/,
    "a flex item without min-w-0 refuses to shrink below its content, which clips the row",
  );

  // An untrusted tool argument must never become a clickable link.
  assert.deepEqual(
    find(
      row,
      (node): node is ts.JsxElement =>
        ts.isJsxElement(node) &&
        ts.isIdentifier(node.openingElement.tagName) &&
        node.openingElement.tagName.text === "a",
    ),
    [],
    "the url row must not render an anchor: the argument is provider-controlled, and the " +
      "completed card already links the url",
  );
  assert.equal(
    attributeOf(row.openingElement, "href"),
    undefined,
    "the url row must not carry an href",
  );
});

test("a bidi control in the url is escaped before it reaches the row", () => {
  // U+202E obeys in `unicode-bidi: plaintext`, so the row could read a different path
  // than the one being approved.
  const row = requireRow();
  const rendered = find(row, ts.isJsxExpression).map(
    (expression) => expression.expression?.getText(source) ?? "",
  );
  assert.ok(
    rendered.some((expression) => expression.includes("escapeBidiControls(")),
    `the row must escape bidi controls; it renders ${JSON.stringify(rendered)}`,
  );
  const imported = find(source, (node): node is ts.ImportDeclaration =>
    ts.isImportDeclaration(node),
  ).some((declaration) =>
    declaration.getText(source).includes("escape-bidi-controls"),
  );
  assert.ok(imported, "the row must import its escape from the shared helper");
});

test("the url row is height-capped so the approval controls stay reachable", () => {
  // An unbounded url would push Allow/Deny below the viewport while the decision is
  // being made.
  const row = requireRow();
  const pane = find(
    row,
    (node): node is ts.JsxElement =>
      ts.isJsxElement(node) &&
      ts.isIdentifier(node.openingElement.tagName) &&
      node.openingElement.tagName.text === "ScrollPane",
  );
  assert.equal(
    pane.length,
    1,
    "the url must sit in exactly one scroller rather than setting the card's height",
  );
  // `className` is the padded wrapper and `scrollerClassName` the inner <pre> that owns the
  // overflow, so the cap has to sit on the scroller or the text is clipped instead of scrollable.
  const scroller =
    attributeOf(
      pane[0].openingElement,
      "scrollerClassName",
    )?.initializer?.getText(source) ?? "";
  assert.match(
    scroller,
    /max-h-/,
    "the scroller needs an explicit max height, or it still grows with the url",
  );
  assert.match(
    scroller,
    /overflow-auto/,
    "the capped pane must scroll, so the complete value stays reachable",
  );
  assert.doesNotMatch(
    classNameOf(pane[0].openingElement),
    /max-h-/,
    "a cap on the wrapper leaves the scroller at full height, so its overflow never scrolls",
  );
  assert.ok(
    pane[0].getStart(source) >= row.getStart(source) &&
      pane[0].getEnd() <= row.getEnd(),
    "the scroller must wrap the url row's content",
  );
});

test("the completed card keeps the url link that landed with #5787", () => {
  // The finished card's link is main's own landed behaviour and stays out of scope here.
  const anchors = tagsNamed("a");
  assert.equal(
    anchors.length,
    1,
    "the card has exactly one anchor: the completed card's url link",
  );
  assert.equal(
    attributeOf(anchors[0], "href")?.initializer?.getText(source),
    "{safeUrl}",
    "the completed card's link must keep reading the scheme-checked url",
  );
  assert.ok(
    requireRow().getEnd() < anchors[0].getStart(source),
    "the parked row renders above the completed-card link, so it is not that element",
  );
});

test("the collapsed trigger does not leak the raw url", () => {
  const [trigger] = tagsNamed("ToolFallbackTrigger");
  assert.ok(trigger, "the card no longer has a trigger");
  // The label comes from webSearchToolName, which web-search-action-variants.test.ts pins. What
  // matters here is that this change did not put the raw argument in the collapsed row too.
  const identifiers = new Set<string>();
  walk(trigger, (node) => {
    if (ts.isIdentifier(node)) identifiers.add(node.text);
  });
  assert.ok(
    !identifiers.has("url"),
    `the trigger must stay hostname-only; it reads ${JSON.stringify([...identifiers])}`,
  );
});
