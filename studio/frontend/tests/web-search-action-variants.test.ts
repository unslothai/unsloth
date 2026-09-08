// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const CARD = readFileSync(
  fileURLToPath(
    new URL(
      "../src/components/assistant-ui/tool-ui-web-search.tsx",
      import.meta.url,
    ),
  ),
  "utf8",
);

test("the card tells the three action types apart", () => {
  // Reading only `action.query` rendered open_page and find_in_page as an empty
  // `Searching ""`, which is the bug this branch exists for.
  assert.match(
    CARD,
    /actionType === "find_in_page" \|\| \(!!url && !!pattern\.trim\(\)\)/,
  );
  assert.match(CARD, /const isUrlFetch = !!url && !isFindInPage;/);
  // and an image-only call is still none of the above
  assert.match(CARD, /!isUrlFetch && !isFindInPage && !query\.trim\(\)/);
});

test("a finished find_in_page does not claim it found the pattern", () => {
  // The action names the pattern the model looked for and nothing about whether
  // it matched, so a completed call is not evidence that it did.
  assert.doesNotMatch(CARD, /`Found "\$\{pattern\}"/);
  assert.match(
    CARD,
    /`Searched for "\$\{pattern\}" in \$\{displayDomain \|\| "page"\}`/,
  );
});

test("the url variants show the page they read, whatever else the card has", () => {
  // open_page and find_in_page report no text of their own. The link also has to
  // survive the terminal citation backfill replacing this card's result with the
  // run's sources, which would otherwise take the sources branch and drop it.
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
  // A bare target="_blank" does nothing in the Tauri webview. Every external
  // link in the app pairs it with an opener; the Source pills in this same card
  // use openLink, so this one does too or Desktop users get a dead link.
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
