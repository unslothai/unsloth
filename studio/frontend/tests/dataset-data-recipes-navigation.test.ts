// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const DATASET_LISTS_PATH = fileURLToPath(
  new URL(
    "../src/features/dataset-picker/components/dataset-selector-lists.tsx",
    import.meta.url,
  ),
);
const DATASET_SELECTOR_PATH = fileURLToPath(
  new URL(
    "../src/features/dataset-picker/components/dataset-selector.tsx",
    import.meta.url,
  ),
);
const DATASET_SECTION_PATH = fileURLToPath(
  new URL(
    "../src/features/studio/sections/dataset-section.tsx",
    import.meta.url,
  ),
);
const RECIPE_EXECUTION_TRACKER_PATH = fileURLToPath(
  new URL(
    "../src/features/recipe-studio/executions/tracker.ts",
    import.meta.url,
  ),
);
const sourceText = readFileSync(DATASET_LISTS_PATH, "utf8");
const sourceFile = ts.createSourceFile(
  DATASET_LISTS_PATH,
  sourceText,
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);
const selectorText = readFileSync(DATASET_SELECTOR_PATH, "utf8");
const selectorFile = ts.createSourceFile(
  DATASET_SELECTOR_PATH,
  selectorText,
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);
const sectionText = readFileSync(DATASET_SECTION_PATH, "utf8");
const trackerText = readFileSync(RECIPE_EXECUTION_TRACKER_PATH, "utf8");
const trackerFile = ts.createSourceFile(
  RECIPE_EXECUTION_TRACKER_PATH,
  trackerText,
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TS,
);

function findOpenDataRecipesButton(): ts.JsxElement | null {
  let result: ts.JsxElement | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isJsxElement(node) &&
      node.openingElement.tagName.getText(sourceFile) === "Button" &&
      node
        .getText(sourceFile)
        .includes('t("studio.datasetPicker.openDataRecipes")')
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

function findSelectorVariableDeclaration(
  name: string,
): ts.VariableDeclaration | null {
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
  selectorFile.forEachChild(visit);
  return result;
}

test("the empty local-dataset action uses side-effect-free SPA navigation", () => {
  const button = findOpenDataRecipesButton();
  assert.ok(button, "Open Data Recipes must be rendered by a Button");

  const onClick = attribute(button.openingElement, "onClick");
  assert.equal(
    onClick?.initializer?.getText(sourceFile),
    "{onOpenDataRecipes}",
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

  const handler = findSelectorVariableDeclaration("openDataRecipes");
  assert.ok(handler, "Open Data Recipes must have a dedicated SPA handler");
  const handlerText = handler.getText(selectorFile);
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
  assert.doesNotMatch(sectionText, /href\s*=\s*["']\/data-recipes["']/);
  assert.doesNotMatch(sourceText, /href\s*=\s*["']\/data-recipes["']/);
  assert.doesNotMatch(selectorText, /href\s*=\s*["']\/data-recipes["']/);
});

test("successful full recipe runs invalidate the shared dataset inventory", () => {
  const inventoryImport = trackerFile.statements.find(
    (statement): statement is ts.ImportDeclaration =>
      ts.isImportDeclaration(statement) &&
      statement.moduleSpecifier.getText(trackerFile) === '"@/features/hub"',
  );
  assert.ok(
    inventoryImport,
    "the tracker must import the inventory invalidator",
  );
  assert.match(
    inventoryImport.getText(trackerFile),
    /\bbumpInventoryVersion\b/,
  );

  const fullRunInvalidations: ts.IfStatement[] = [];
  const visit = (node: ts.Node): void => {
    if (
      ts.isIfStatement(node) &&
      node.expression.getText(trackerFile) === 'kind === "full"' &&
      /\bbumpInventoryVersion\s*\(\s*\)/.test(
        node.thenStatement.getText(trackerFile),
      )
    ) {
      fullRunInvalidations.push(node);
    }
    node.forEachChild(visit);
  };
  trackerFile.forEachChild(visit);

  assert.equal(
    fullRunInvalidations.length,
    1,
    "only completed full runs should invalidate the selectable dataset inventory",
  );
});
