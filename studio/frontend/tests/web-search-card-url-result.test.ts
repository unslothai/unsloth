// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const CARD = fileURLToPath(
  new URL(
    "../src/components/assistant-ui/tool-ui-web-search.tsx",
    import.meta.url,
  ),
);

/** The trailing branch of the card body, after the sources and images cases. */
function urlBranch(): string {
  const source = readFileSync(CARD, "utf8");
  const start = source.indexOf("        ) : safeUrl || resultText ? (");
  assert.notEqual(start, -1, "the url/result branch changed shape");
  return source.slice(start, source.indexOf("</ToolFallbackContent>", start));
}

test("a url does not hide the tool result", () => {
  // The local web_search url mode returns the fetched page text -- or why the
  // fetch failed -- as the result. A branch that renders the link INSTEAD of it
  // drops both without a trace.
  const branch = urlBranch();
  const link = branch.indexOf("href={safeUrl}");
  const body = branch.indexOf("{resultText}");
  assert.ok(link !== -1, "the link is gone");
  assert.ok(body !== -1, "the result body is gone");
  assert.ok(
    link < body,
    "the result must render below the link, not instead of it",
  );
});

test("only OpenAI's synthesized card label is suppressed", () => {
  assert.match(urlBranch(), /resultText && !resultIsCardLabel/);
  // Scoped to safeUrl so a query-only OpenAI search still shows its result and
  // the branch can never render an empty box.
  assert.match(
    readFileSync(CARD, "utf8"),
    /const resultIsCardLabel = !!safeUrl && !!actionType;/,
  );
});
