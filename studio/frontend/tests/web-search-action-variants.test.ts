// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

import {
  type WebSearchToolNameState,
  webSearchToolName,
} from "../src/components/assistant-ui/tool-arg-text.ts";

const CARD = readSrc("components/assistant-ui/tool-ui-web-search.tsx");

test("the card tells the three action types apart", () => {
  // open_page and find_in_page carry no `action.query`.
  assert.match(
    CARD,
    /actionType === "find_in_page" \|\| \(!!url && !!pattern\.trim\(\)\)/,
  );
  assert.match(CARD, /const isUrlFetch = !!url && !isFindInPage;/);
  assert.match(CARD, /!isUrlFetch && !isFindInPage && !query\.trim\(\)/);
});

test("a finished find_in_page does not claim it found the pattern", () => {
  // The action names the pattern, not whether it matched.
  const finished = webSearchToolName({
    isRunning: false,
    isFindInPage: true,
    isUrlFetch: false,
    isImageOnly: false,
    foundImages: false,
    displayDomain: "example.com",
    pattern: "needle",
    query: "",
    imageLabel: "",
  });
  assert.equal(finished, 'Searched for "needle" in example.com');
  assert.doesNotMatch(finished, /Found/);
});

test("the url variants show the page they read, whatever else the card has", () => {
  // The link must survive the citation backfill replacing this card's result with sources.
  const body = CARD.slice(
    CARD.indexOf("      <ToolFallbackContent>"),
    CARD.indexOf("</ToolFallbackContent>"),
  );
  assert.notEqual(body.length, 0, "the card body moved");
  const link = body.indexOf("href={safeUrl}");
  assert.ok(link !== -1, "the link is gone");
  for (const branch of [
    "{sources.length === 0 && images.length > 0 ? (",
    ") : sources.length > 0 ? (",
    ") : resultText ? (",
  ]) {
    const at = body.indexOf(branch);
    assert.notEqual(at, -1, `branch moved: ${branch}`);
    assert.ok(link < at, `the link must render above ${branch.trim()}`);
  }
  // Only http(s) reaches an href: the url is provider-controlled.
  assert.match(
    CARD,
    /const safeUrl = isSafeHttpUrl\(candidateUrl\) \? candidateUrl : "";/,
  );
});

test("the read-page link opens in Desktop, not just the browser", () => {
  // A bare target="_blank" does nothing in the Tauri webview; use openLink.
  assert.match(CARD, /import \{ openLink \} from "@\/lib\/open-link";/);
  const anchor = CARD.slice(
    CARD.indexOf("href={safeUrl}"),
    CARD.indexOf("</a>", CARD.indexOf("href={safeUrl}")),
  );
  assert.notEqual(anchor.length, 0, "the link moved");
  assert.match(anchor, /onClick=\{\(e\) => \{/);
  assert.match(anchor, /openLink\(safeUrl\)/);
  assert.match(anchor, /e\.preventDefault\(\)/);
});

const BASE: WebSearchToolNameState = {
  isRunning: false,
  isFindInPage: false,
  isUrlFetch: false,
  isImageOnly: false,
  foundImages: false,
  displayDomain: "",
  pattern: "",
  query: "",
  imageLabel: "",
};

const name = (state: Partial<WebSearchToolNameState>) =>
  webSearchToolName({ ...BASE, ...state });

test("a running call is never named in the past tense", () => {
  const running = [
    name({
      isRunning: true,
      isFindInPage: true,
      pattern: "todo",
      displayDomain: "example.com",
    }),
    name({ isRunning: true, isFindInPage: true, displayDomain: "example.com" }),
    name({ isRunning: true, isFindInPage: true }),
    name({ isRunning: true, isUrlFetch: true, displayDomain: "example.com" }),
    name({ isRunning: true, isUrlFetch: true }),
    name({ isRunning: true, isImageOnly: true, imageLabel: "otters" }),
    name({ isRunning: true, query: "unsloth" }),
    name({ isRunning: true }),
  ];
  for (const label of running) {
    assert.doesNotMatch(
      label,
      /^(Searched|Read|Found|No images)\b/,
      `"${label}" reads as a finished call`,
    );
  }
  assert.deepEqual(running, [
    'Finding "todo" in example.com…',
    "Searching example.com…",
    "Searching page…",
    "Reading example.com…",
    "Reading page…",
    "Finding images for “otters”",
    'Searching for "unsloth"…',
    "Searching…",
  ]);
});

test("a settled call keeps the wording it shipped with", () => {
  assert.equal(
    name({ isFindInPage: true, pattern: "todo", displayDomain: "example.com" }),
    'Searched for "todo" in example.com',
  );
  assert.equal(
    name({ isFindInPage: true, displayDomain: "example.com" }),
    "Searched example.com",
  );
  assert.equal(name({ isFindInPage: true }), "Searched page");
  assert.equal(
    name({ isUrlFetch: true, displayDomain: "example.com" }),
    "Read example.com",
  );
  assert.equal(name({ isUrlFetch: true }), "Read page");
  assert.equal(name({ query: "unsloth" }), 'Searched "unsloth"');
  assert.equal(name({}), "Web Search");
});

test("the image header only claims images that came back", () => {
  assert.equal(
    name({ isImageOnly: true, imageLabel: "otters", foundImages: true }),
    "Found images for “otters”",
  );
  assert.equal(
    name({ isImageOnly: true, imageLabel: "otters" }),
    "No images for “otters”",
  );
  assert.equal(
    name({ query: "otters", imageLabel: "otters", foundImages: true }),
    'Searched "otters" · images for otters',
  );
  assert.equal(
    name({ query: "otters", imageLabel: "otters" }),
    'Searched "otters"',
  );
  assert.equal(
    name({ query: "otters", foundImages: true }),
    'Searched "otters"',
  );
});

test("a url fetch names the page, not the query the call may carry", () => {
  assert.equal(
    name({ isUrlFetch: true, displayDomain: "example.com", query: "unsloth" }),
    "Read example.com",
  );
});
