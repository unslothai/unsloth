// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import { isSafeNavigableSourceUrl } from "../src/features/chat/utils/document-citation-source.ts";

// open-link.ts imports `@/lib/api-base`; load it as the web build (isTauri = false) with window.open recorded.
const source = readFileSync(new URL("../src/lib/open-link.ts", import.meta.url), "utf8").replace(
  /^import \{ isTauri \} from "@\/lib\/api-base";$/m,
  "const isTauri = false;",
);
const js = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const opened: string[] = [];
(globalThis as Record<string, unknown>).window = {
  open: (url: string) => opened.push(url),
  location: { hash: "" },
};
const { openLink } = await import(`data:text/javascript,${encodeURIComponent(js)}`);

test("https: without slashes opens as a web link instead of navigating Studio's page", () => {
  for (const [href, expected] of [
    ["https:evil.example/x", "https://evil.example/x"],
    ["https:\t//evil.example/x", "https://evil.example/x"],
    ["HTTP:evil.example", "http://evil.example/"],
  ]) {
    opened.length = 0;
    assert.equal(openLink(href), true, href);
    assert.deepEqual(opened, [expected], href);
  }
});

test("relative, anchor and ordinary web links are unchanged", () => {
  opened.length = 0;
  assert.equal(openLink("/chat?thread=1"), false);
  assert.equal(openLink("docs/page"), false);
  assert.equal(openLink("https://unsloth.ai/docs"), true);
  assert.deepEqual(opened, ["https://unsloth.ai/docs"]);
});

test("citation sources come back in their parsed form", () => {
  assert.equal(isSafeNavigableSourceUrl("https:\t//a.example/x"), "https://a.example/x");
  assert.equal(isSafeNavigableSourceUrl("https:a.example"), "https://a.example/");
  assert.equal(isSafeNavigableSourceUrl("https://a.example/x?y=1"), "https://a.example/x?y=1");
  assert.equal(isSafeNavigableSourceUrl("javascript:alert(1)"), "");
});
