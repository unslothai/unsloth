// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Ollama rows are stamped "ollama"; the label must match the hub inventory's label.

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
