// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Studio drives Shiki itself rather than letting `@streamdown/code` do it, because the package
 * memoises every tokenisation forever and a streamed fence hands it one prefix per refresh window.
 * Owning the call means owning the LANGUAGE RESOLUTION the package used to do, and that path has
 * one way to fail silently: `createHighlighter` REJECTS on a grammar it does not have, and the
 * rejection lands in a `.catch`, so the fence is left unstyled forever with nothing but a console
 * line. A model writes ```mycustomdsl often enough for that to matter.
 *
 * So: an unknown fence tag must still produce tokens (falling back to plaintext, which is what
 * the package did), and a tag Shiki publishes as an alias must resolve to the same grammar as its
 * canonical id rather than being treated as unknown.
 */

import assert from "node:assert/strict";
import test from "node:test";
import { createCodePlugin } from "../src/components/assistant-ui/code-plugin.ts";

const THEMES: [string, string] = ["github-light", "github-dark"];

const plugin = createCodePlugin();

/** The plugin answers a cold ask with null and a callback, so wait for the callback. */
function highlight(code: string, language: string): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`no tokens for language "${language}"`)),
      60_000,
    );
    const immediate = plugin.highlight(
      // The plugin's own types narrow `language` to Shiki's union, and an unknown tag is the case
      // under test, so it arrives the way a fence really delivers it: as a string.
      { code, language, themes: THEMES } as never,
      (result) => {
        clearTimeout(timer);
        resolve(result);
      },
    );
    if (immediate) {
      clearTimeout(timer);
      resolve(immediate);
    }
  });
}

const tokenText = (result: unknown): string =>
  (result as { tokens: { content: string }[][] }).tokens
    .map((line) => line.map((token) => token.content).join(""))
    .join("\n");

test("an unknown fence tag still produces tokens instead of an unstyled block", async () => {
  const source = "cfg unit {\n  retries = 3\n}\n";
  const result = await highlight(source, "definitelynotarealgrammar");
  assert.equal(tokenText(result), source);
});

test("a Shiki alias resolves to the same grammar as its canonical id", async () => {
  // `dockerfile` is an alias Shiki publishes for `docker` and is NOT in Studio's own override
  // table, so it can only resolve through the alias map read out of `bundledLanguagesInfo`.
  const source = "FROM python:3.12\nRUN pip install unsloth\n";
  const viaAlias = await highlight(source, "dockerfile");
  const viaCanonical = await highlight(source, "docker");
  assert.deepEqual(viaAlias, viaCanonical);
  // A plaintext fallback would produce one token per line with no colour, so assert the grammar
  // really ran rather than only that the two agree.
  const lines = (viaAlias as { tokens: unknown[][] }).tokens;
  assert.ok(
    lines[0].length > 1,
    `expected the docker grammar to split the first line, got ${lines[0].length} token(s)`,
  );
});
