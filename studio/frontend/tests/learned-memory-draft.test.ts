// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";
import ts from "@typescript/typescript6";

test("memory editor initializes on a plain HTTP origin without randomUUID", () => {
  const source = readFileSync(
    new URL(
      "../src/features/settings/components/learned-memory-section.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const tree = ts.createSourceFile(
    "memory.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const draft = tree.statements.find(
    (node) =>
      ts.isFunctionDeclaration(node) && node.name?.text === "newMemoryDraft",
  );
  assert.ok(draft);
  const nonce = readFileSync(
    new URL("../src/lib/navigation-nonce.ts", import.meta.url),
    "utf8",
  ).replace("export function", "function");
  const code = ts.transpileModule(nonce + "\n" + draft.getText(tree), {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.None,
    },
  }).outputText;
  const value = runInNewContext(code + "; newMemoryDraft()", { crypto: {} });
  assert.equal(value.kind, "new");
  assert.equal(value.value, "");
  assert.match(value.key, /^[a-z][a-z0-9_]{0,63}$/);
});
