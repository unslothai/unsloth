// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const WHITESPACE = /\s+/g;

function parse(relativePath: string, kind: ts.ScriptKind): ts.SourceFile {
  const path = fileURLToPath(new URL(relativePath, import.meta.url));
  return ts.createSourceFile(
    path,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    kind,
  );
}

function findDescendant<T extends ts.Node>(
  root: ts.Node,
  predicate: (node: ts.Node) => node is T,
  message: string,
): T {
  let match: T | undefined;
  const visit = (node: ts.Node) => {
    if (!match && predicate(node)) {
      match = node;
    }
    if (!match) {
      ts.forEachChild(node, visit);
    }
  };
  visit(root);
  if (!match) {
    throw new Error(message);
  }
  return match;
}

function findFunction(
  source: ts.SourceFile,
  name: string,
): ts.FunctionLikeDeclaration {
  const declaration = findDescendant(
    source,
    (node): node is ts.FunctionDeclaration | ts.VariableDeclaration =>
      (ts.isFunctionDeclaration(node) && node.name?.text === name) ||
      (ts.isVariableDeclaration(node) &&
        ts.isIdentifier(node.name) &&
        node.name.text === name &&
        node.initializer !== undefined &&
        (ts.isArrowFunction(node.initializer) ||
          ts.isFunctionExpression(node.initializer))),
    `function ${name} not found`,
  );
  return ts.isVariableDeclaration(declaration)
    ? (declaration.initializer as ts.FunctionLikeDeclaration)
    : declaration;
}

function findVariable(
  source: ts.SourceFile,
  name: string,
): ts.VariableDeclaration {
  return findDescendant(
    source,
    (node): node is ts.VariableDeclaration =>
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === name,
    `variable ${name} not found`,
  );
}

function compact(node: ts.Node, source: ts.SourceFile): string {
  return node.getText(source).replace(WHITESPACE, "");
}

function bodyOf(declaration: ts.FunctionLikeDeclaration): ts.ConciseBody {
  if (!declaration.body) {
    throw new Error("function body not found");
  }
  return declaration.body;
}

function initializerOf(declaration: ts.VariableDeclaration): ts.Expression {
  if (!declaration.initializer) {
    throw new Error("variable initializer not found");
  }
  return declaration.initializer;
}

function calls(root: ts.Node, source: ts.SourceFile, name: string): boolean {
  let found = false;
  const visit = (node: ts.Node) => {
    if (ts.isCallExpression(node) && node.expression.getText(source) === name) {
      found = true;
      return;
    }
    ts.forEachChild(node, visit);
  };
  visit(root);
  return found;
}

const API_SOURCE = parse(
  "../src/features/chat/api/skills-api.ts",
  ts.ScriptKind.TS,
);
const ADAPTER_SOURCE = parse(
  "../src/features/chat/api/chat-adapter.ts",
  ts.ScriptKind.TS,
);
const DIALOG_SOURCE = parse(
  "../src/features/chat/chat-skills-dialog.tsx",
  ts.ScriptKind.TSX,
);

test("skill mutations and cross-tab events invalidate cached availability", () => {
  const reset = findFunction(API_SOURCE, "resetSkillCatalogCache");
  const resetText = compact(bodyOf(reset), API_SOURCE);
  assert.equal(resetText.includes("catalogRevision+=1"), true);
  assert.equal(resetText.includes("enabledSkillsCache=null"), true);
  assert.equal(resetText.includes("enabledSkillsRequest=null"), true);

  for (const name of ["importSkillBundle", "setSkillEnabled", "deleteSkill"]) {
    assert.equal(
      calls(
        findFunction(API_SOURCE, name),
        API_SOURCE,
        "clearSkillCatalogCache",
      ),
      true,
      `${name} leaves the enabled-skill cache stale`,
    );
  }

  const listText = compact(findFunction(API_SOURCE, "listSkills"), API_SOURCE);
  assert.equal(listText.includes("while(true)"), true);
  assert.equal(
    listText.includes("if(revision!==catalogRevision){continue;}"),
    true,
  );
  assert.equal(
    calls(
      findFunction(API_SOURCE, "notifySkillCatalogChanged"),
      API_SOURCE,
      "resetSkillCatalogCache",
    ),
    true,
  );
  assert.equal(
    compact(findFunction(API_SOURCE, "getCatalogChannel"), API_SOURCE).includes(
      "catalogChannel.onmessage=notifySkillCatalogChanged",
    ),
    true,
  );
  assert.equal(
    calls(
      findFunction(DIALOG_SOURCE, "ChatSkillsDialog"),
      DIALOG_SOURCE,
      "subscribeSkillCatalogChanges",
    ),
    true,
  );
});

test("the import input passes its selected ZIP to the API path", () => {
  const input = findDescendant(
    DIALOG_SOURCE,
    (node): node is ts.JsxSelfClosingElement =>
      ts.isJsxSelfClosingElement(node) &&
      node.tagName.getText(DIALOG_SOURCE) === "input",
    "skill file input not found",
  );
  const onChange = input.attributes.properties.find(
    (attribute) =>
      ts.isJsxAttribute(attribute) &&
      attribute.name.getText(DIALOG_SOURCE) === "onChange",
  );
  assert.ok(onChange && ts.isJsxAttribute(onChange));
  assert.equal(onChange.initializer?.getText(DIALOG_SOURCE), "{onImportFile}");

  const handler = compact(
    findFunction(DIALOG_SOURCE, "onImportFile"),
    DIALOG_SOURCE,
  );
  assert.equal(handler.includes("event.target.files?.[0]"), true);
  assert.equal(handler.includes("if(file){importBundle(file,false);}"), true);
});

test("vision support gates the loader and every request list uses that decision", () => {
  const availability = findVariable(
    ADAPTER_SOURCE,
    "skillLoaderAvailableForThisTurn",
  );
  assert.equal(
    compact(initializerOf(availability), ADAPTER_SOURCE),
    "Boolean(supportsStudioToolsForThisTurn&&(isExternalRequest||!imageBase64||selectedModelSummary?.isGguf===true),)",
  );
  const enabled = findVariable(ADAPTER_SOURCE, "skillsEnabledForThisTurn");
  assert.equal(
    compact(initializerOf(enabled), ADAPTER_SOURCE),
    "skillLoaderAvailableForThisTurn&&(awaithasEnabledSkills())",
  );

  const loaderConditions: string[] = [];
  const visit = (node: ts.Node) => {
    if (
      ts.isConditionalExpression(node) &&
      ts.isArrayLiteralExpression(node.whenTrue) &&
      node.whenTrue.elements.some(
        (element) =>
          ts.isStringLiteral(element) && element.text === "read_skill",
      )
    ) {
      loaderConditions.push(node.condition.getText(ADAPTER_SOURCE));
    }
    ts.forEachChild(node, visit);
  };
  visit(ADAPTER_SOURCE);
  assert.deepEqual(loaderConditions.sort(), [
    "skillsEnabledForThisTurn",
    "skillsEnabledForThisTurn",
    "skillsOn",
  ]);

  assert.equal(
    compact(ADAPTER_SOURCE, ADAPTER_SOURCE).includes(
      "!anyWebEnabledForThisTurn&&!codeExecEnabledForThisTurn&&!imageGenerationEnabledForThisTurn&&!skillsEnabledForThisTurn",
    ),
    true,
  );
});

test("token counts query skill availability and include its schema", () => {
  const tokenCount = findFunction(ADAPTER_SOURCE, "buildLocalTokenCountExtras");
  const skillsOn = findDescendant(
    tokenCount,
    (node): node is ts.VariableDeclaration =>
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === "skillsOn",
    "token-count skill availability not found",
  );
  assert.equal(
    compact(initializerOf(skillsOn), ADAPTER_SOURCE),
    "awaithasEnabledSkills()",
  );
  assert.equal(
    compact(tokenCount, ADAPTER_SOURCE).includes(
      '...(skillsOn?["read_skill"]:[])',
    ),
    true,
  );
});
