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

test("the url variants show the page they read", () => {
  // open_page and find_in_page report no text of their own, so without the link
  // the card body is empty.
  const start = CARD.indexOf("        ) : safeUrl || resultText ? (");
  assert.notEqual(start, -1, "the url/result branch changed shape");
  const branch = CARD.slice(
    start,
    CARD.indexOf("</ToolFallbackContent>", start),
  );
  assert.ok(branch.indexOf("href={safeUrl}") !== -1, "the link is gone");
  assert.ok(branch.indexOf("{resultText}") !== -1, "the result body is gone");
  // Only http(s) reaches an href: the url is provider-controlled.
  assert.match(
    CARD,
    /const safeUrl = isSafeHttpUrl\(candidateUrl\) \? candidateUrl : "";/,
  );
});
