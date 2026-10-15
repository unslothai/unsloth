// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

import type { CodePluginOptions } from "@streamdown/code";

import {
  createCodePlugin,
  MIN_INCREMENTAL_CHARS,
} from "../src/components/assistant-ui/code-plugin.ts";

/** Remounts must reuse module-scope highlight state, or every fence flashes unhighlighted. */

// Past MIN_INCREMENTAL_CHARS so the fence takes the per-fence slot path.
const LINES = 90;

const freshCode = (tag: string): string =>
  Array.from(
    { length: LINES },
    (_, index) => `export const ${tag}_${index} = ${index};`,
  ).join("\n");

async function withTimeout(
  arrived: Promise<void>,
  timeoutMs = 30_000,
): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      arrived,
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error("the highlighter never produced tokens")),
          timeoutMs,
        );
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
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

  let arrived!: () => void;
  const tokensArrived = new Promise<void>((resolve) => {
    arrived = resolve;
  });
  assert.equal(
    mounted.highlight(options, () => arrived()),
    null,
    "a cold highlight is expected to answer through its callback",
  );
  await withTimeout(tokensArrived);

  const afterRemount = mounted.highlight(options);

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
