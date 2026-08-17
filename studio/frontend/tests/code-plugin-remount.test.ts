// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

import type { CodePluginOptions } from "@streamdown/code";

import { createCodePlugin } from "../src/components/assistant-ui/code-plugin.ts";

/**
 * The chat Markdown renderer remounts <Streamdown> whenever the incremental
 * cache has to move its render identity, which throws away the whole block
 * subtree. That is only affordable because none of the highlighting state
 * lives in the component tree: `@streamdown/code` keeps its highlighters and
 * its tokenized results in module-scope Maps, and the wrapper below it is
 * built once per module rather than once per render. So a remount re-asks for
 * tokens it already has and gets them back in the same tick, with no unstyled
 * frame in between.
 *
 * These two tests pin that property. Move the highlight cache into component
 * state, or build the plugin inside the component, and a remount becomes a
 * visible flash back to unhighlighted code on every fence in the reply.
 */

// Enough lines to clear the wrapper's MIN_INCREMENTAL_CHARS, so the fence goes
// through the per-fence slot path a streaming reply uses rather than the small
// fence shortcut straight to the underlying plugin.
const LINES = 90;
const MIN_INCREMENTAL_CHARS = 2000;

/** Unique per run, so the module-scope cache starts cold for this test. */
const freshCode = (tag: string): string =>
  Array.from(
    { length: LINES },
    (_, index) => `export const ${tag}_${index} = ${index};`,
  ).join("\n");

async function waitForHighlight(
  ask: () => unknown,
  timeoutMs = 30_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (ask()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error("the highlighter never produced tokens");
}

test("a remount gets already highlighted code back in the same tick", async () => {
  const themes: CodePluginOptions["themes"] = ["github-light", "github-dark"];
  const mounted = createCodePlugin({ themes });
  const code = freshCode(`remount${process.pid}`);
  assert.ok(
    code.length > MIN_INCREMENTAL_CHARS,
    "the fixture fence dropped below the incremental threshold, so this test no longer covers the slot path",
  );
  const options = {
    code,
    language: "ts",
    themes: mounted.getThemes(),
  } as Parameters<typeof mounted.highlight>[0];

  // Cold: the grammar has to load, so the first ask cannot answer inline. The
  // callback is what the renderer would repaint from; this test only needs the
  // tokens to reach the cache, so it drops them.
  const discard = (): void => undefined;
  assert.equal(
    mounted.highlight(options, discard),
    null,
    "a cold highlight is expected to answer through its callback",
  );
  await waitForHighlight(() => mounted.highlight(options));

  // A remount builds a new component tree, and with it a new plugin wrapper.
  // The tokens have to survive that, because they do not live in either.
  const remounted = createCodePlugin({ themes });
  const afterRemount = remounted.highlight(options);

  assert.ok(
    afterRemount,
    "a remount had to wait for the highlighter again, so every fence in the reply would flash back to unhighlighted code",
  );
  assert.equal(
    afterRemount.tokens.length,
    LINES,
    "the remount got a different tokenization than the mount it replaced",
  );
});

const MARKDOWN_TEXT_PATH = new URL(
  "../src/components/assistant-ui/markdown-text.tsx",
  import.meta.url,
);
const markdownText = ts.createSourceFile(
  MARKDOWN_TEXT_PATH.pathname,
  readFileSync(MARKDOWN_TEXT_PATH, "utf8"),
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);

/** Every `createCodePlugin(...)` call in the file, with the scope it sits in. */
function codePluginCalls(): { atModuleScope: boolean }[] {
  const calls: { atModuleScope: boolean }[] = [];
  const visit = (node: ts.Node, insideFunction: boolean): void => {
    if (
      ts.isCallExpression(node) &&
      node.expression.getText(markdownText) === "createCodePlugin"
    ) {
      calls.push({ atModuleScope: !insideFunction });
    }
    const entersFunction =
      insideFunction ||
      ts.isFunctionDeclaration(node) ||
      ts.isFunctionExpression(node) ||
      ts.isArrowFunction(node) ||
      ts.isMethodDeclaration(node);
    node.forEachChild((child) => visit(child, entersFunction));
  };
  markdownText.forEachChild((node) => visit(node, false));
  return calls;
}

test("the chat renderer builds its code plugin once, outside the component", () => {
  const calls = codePluginCalls();
  assert.equal(
    calls.length,
    1,
    "the chat renderer should build exactly one code plugin",
  );
  assert.equal(
    calls[0].atModuleScope,
    true,
    "the code plugin is built inside a component, so its incremental fence slots are discarded on every remount",
  );
});
