// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const list = readFileSync(
  new URL("../src/features/translate/history-list.tsx", import.meta.url),
  "utf8",
);
const page = readFileSync(
  new URL("../src/features/translate/translate-page.tsx", import.meta.url),
  "utf8",
);

function listBody(): string {
  const start = list.indexOf("export const TranslateHistoryList = memo(");
  assert.notEqual(start, -1, "TranslateHistoryList must be wrapped in memo");
  return list.slice(start);
}

test("the history list is memoized, so streamed frames and keystrokes skip its rows", () => {
  listBody();
  assert.doesNotMatch(list, /export function TranslateHistoryList\b/);
  assert.match(list, /import \{[^}]*\bmemo\b[^}]*\} from "react";/);
});

test("the page renders the list without props, so memo always bails out on page renders", () => {
  const uses = page.match(/<TranslateHistoryList\b[^>]*>/g) ?? [];
  assert.deepEqual(uses, ["<TranslateHistoryList />"]);
});

test("the memoized list subscribes to the 12h/24h setting it formats with", () => {
  const body = listBody();
  const hook = body.indexOf("useTimeFormat();");
  assert.notEqual(hook, -1, "the list must call useTimeFormat()");
  assert.ok(
    hook < body.indexOf("withTimeFormat("),
    "subscribe before formatting",
  );
  assert.match(
    list,
    /import \{[^}]*\buseTimeFormat\b[^}]*\} from "@\/lib\/time-format";/,
  );
});
