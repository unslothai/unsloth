// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const DATASET_SECTION_PATH = fileURLToPath(
  new URL(
    "../src/features/studio/sections/dataset-section.tsx",
    import.meta.url,
  ),
);
const sourceText = readFileSync(DATASET_SECTION_PATH, "utf8");
const sourceFile = ts.createSourceFile(
  DATASET_SECTION_PATH,
  sourceText,
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);

function findOpenDataRecipesButton(): ts.JsxElement | null {
  let result: ts.JsxElement | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isJsxElement(node) &&
      node.openingElement.tagName.getText(sourceFile) === "Button" &&
      node.getText(sourceFile).includes('t("studio.dataset.openDataRecipes")')
    ) {
      result = node;
      return;
    }
    node.forEachChild(visit);
  };
  sourceFile.forEachChild(visit);
  return result;
}

function attribute(
  opening: ts.JsxOpeningElement,
  name: string,
): ts.JsxAttribute | null {
  for (const property of opening.attributes.properties) {
    if (
      ts.isJsxAttribute(property) &&
      property.name.getText(sourceFile) === name
    ) {
      return property;
    }
  }
  return null;
}

function findVariableDeclaration(name: string): ts.VariableDeclaration | null {
  let result: ts.VariableDeclaration | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === name
    ) {
      result = node;
      return;
    }
    node.forEachChild(visit);
  };
  sourceFile.forEachChild(visit);
  return result;
}

test("the empty local-dataset action uses side-effect-free SPA navigation", () => {
  const button = findOpenDataRecipesButton();
  assert.ok(button, "Open Data Recipes must be rendered by a Button");

  const onClick = attribute(button.openingElement, "onClick");
  assert.equal(
    onClick?.initializer?.getText(sourceFile),
    "{handleOpenDataRecipes}",
  );
  assert.equal(
    attribute(button.openingElement, "type")?.initializer?.getText(sourceFile),
    '"button"',
  );
  assert.equal(attribute(button.openingElement, "asChild"), null);
  assert.doesNotMatch(
    button.getText(sourceFile),
    /<a\b|href\s*=/,
    "the action must not fall back to a full-document anchor",
  );

  const handler = findVariableDeclaration("handleOpenDataRecipes");
  assert.ok(handler, "Open Data Recipes must have a dedicated SPA handler");
  const handlerText = handler.getText(sourceFile);
  assert.match(
    handlerText,
    /navigate\s*\(\s*\{\s*to:\s*["']\/data-recipes["']/,
  );
  assert.doesNotMatch(
    handlerText,
    /sessionStorage|OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY|setDocumentRedirectOpen/,
    "opening Data Recipes must not force the Learning Recipes dialog",
  );
});

test("the dataset section has no raw Data Recipes document navigation", () => {
  assert.doesNotMatch(sourceText, /href\s*=\s*["']\/data-recipes["']/);
});
