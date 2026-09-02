// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The compat endpoint now stamps Ollama rows "ollama" (#9986 bug 1), so every
// consumer of LocalModelInfo.source must name that source rather than fall to
// its generic default. The recipe selector's sourceLabel switch lives inside a
// component, so its ollama case is pinned via the AST, and its label text is
// held in parity with the hub inventory's label for the same source.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { localSourceLabel } = await import(
  "../src/features/hub/inventory/view-models.ts"
);

const SELECTOR = new URL(
  "../src/features/recipe-studio/dialogs/models/local-recipe-model-selector.tsx",
  import.meta.url,
);
const source = ts.createSourceFile(
  "local-recipe-model-selector.tsx",
  readFileSync(SELECTOR, "utf8"),
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);

/** The string literal returned by sourceLabel's `case "ollama"` clause, if any. */
function ollamaCaseLabel(): string | null {
  let label: string | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isFunctionDeclaration(node) &&
      node.name?.text === "sourceLabel" &&
      node.body
    ) {
      const visitCase = (inner: ts.Node): void => {
        if (
          ts.isCaseClause(inner) &&
          ts.isStringLiteral(inner.expression) &&
          inner.expression.text === "ollama"
        ) {
          for (const statement of inner.statements) {
            if (
              ts.isReturnStatement(statement) &&
              statement.expression &&
              ts.isStringLiteral(statement.expression)
            ) {
              label = statement.expression.text;
            }
          }
        }
        ts.forEachChild(inner, visitCase);
      };
      ts.forEachChild(node.body, visitCase);
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(source, visit);
  return label;
}

test("the recipe selector names the ollama source instead of falling to Local", () => {
  assert.equal(ollamaCaseLabel(), "Ollama");
});

test("the recipe selector and hub inventory agree on the ollama label", () => {
  assert.equal(ollamaCaseLabel(), localSourceLabel("ollama"));
});
