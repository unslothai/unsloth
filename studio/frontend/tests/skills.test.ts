// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { readSkillToolDisplay } from "../src/components/assistant-ui/read-skill-tool-display.ts";

const WHITESPACE = /\s+/g;

test("read_skill calls explain progressive skill loading", () => {
  assert.deepEqual(readSkillToolDisplay({ name: "pr-9355-smoke" }), {
    actionLabel: "Read skill instructions",
    toolName: "pr-9355-smoke",
  });
  assert.deepEqual(
    readSkillToolDisplay({
      name: "pr-9355-smoke",
      resource: "references/phrase.txt",
    }),
    {
      actionLabel: "Read skill resource",
      toolName: "pr-9355-smoke · references/phrase.txt",
    },
  );
});

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
const THREAD_SOURCE = parse(
  "../src/components/assistant-ui/thread.tsx",
  ts.ScriptKind.TSX,
);
const SHARED_COMPOSER_SOURCE = parse(
  "../src/features/chat/shared-composer.tsx",
  ts.ScriptKind.TSX,
);

function agentSkillsLivesInMore(
  source: ts.SourceFile,
  functionName: string,
): boolean {
  const declaration = findFunction(source, functionName);
  const label = findDescendant(
    declaration,
    (node): node is ts.JsxText =>
      ts.isJsxText(node) && node.text.trim() === "Agent Skills",
    `Agent Skills menu item not found in ${functionName}`,
  );
  let ancestor: ts.Node | undefined = label.parent;
  while (ancestor && ancestor !== declaration) {
    if (
      ts.isJsxElement(ancestor) &&
      ancestor.openingElement.tagName.getText(source) === "DropdownMenuSub"
    ) {
      return compact(ancestor, source).includes(
        "More</DropdownMenuSubTrigger>",
      );
    }
    ancestor = ancestor.parent;
  }
  return false;
}

test("Agent Skills is available under More in both chat composers", () => {
  assert.equal(
    agentSkillsLivesInMore(THREAD_SOURCE, "ComposerToolsMenu"),
    true,
  );
  assert.equal(
    agentSkillsLivesInMore(SHARED_COMPOSER_SOURCE, "SharedComposer"),
    true,
  );
});

test("skill mutations broadcast once and keep same-tab updates local", () => {
  for (const name of ["importSkillBundle", "setSkillEnabled", "deleteSkill"]) {
    assert.equal(
      calls(
        findFunction(API_SOURCE, name),
        API_SOURCE,
        "broadcastSkillCatalogChanged",
      ),
      true,
      `${name} leaves other tabs stale`,
    );
  }
  assert.equal(API_SOURCE.text.includes("clearSkillCatalogCache"), false);

  assert.equal(
    compact(
      bodyOf(findFunction(API_SOURCE, "notifySkillCatalogChanged")),
      API_SOURCE,
    ).includes(
      "catalogRevision+=1;for(constlistenerofcatalogListeners){listener();}",
    ),
    true,
  );
  assert.equal(
    compact(
      bodyOf(findFunction(API_SOURCE, "broadcastSkillCatalogChanged")),
      API_SOURCE,
    ).includes(
      'catalogRevision+=1;getCatalogChannel()?.postMessage("changed");',
    ),
    true,
  );
  assert.equal(
    calls(
      findFunction(API_SOURCE, "broadcastSkillCatalogChanged"),
      API_SOURCE,
      "notifySkillCatalogChanged",
    ),
    false,
  );
  assert.equal(
    compact(findFunction(API_SOURCE, "listSkills"), API_SOURCE).includes(
      "if(revision===catalogRevision){returnbody.skills;}",
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
  assert.equal(
    calls(findFunction(DIALOG_SOURCE, "importBundle"), DIALOG_SOURCE, "refresh"),
    true,
  );
  for (const name of ["toggleSkill", "removeSkill"]) {
    assert.equal(
      calls(findFunction(DIALOG_SOURCE, name), DIALOG_SOURCE, "setSkills"),
      true,
    );
  }
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

test("every installed skill has a delete control", () => {
  const dialog = compact(
    findFunction(DIALOG_SOURCE, "ChatSkillsDialog"),
    DIALOG_SOURCE,
  );
  assert.equal(dialog.includes("setConfirmingDelete(skill)"), true);
  assert.equal(dialog.includes("skill.bundled"), false);
  assert.equal(compact(API_SOURCE, API_SOURCE).includes("bundled:"), false);
});

test("vision support gates skill tools and request lists keep create and read together", () => {
  const availability = findVariable(
    ADAPTER_SOURCE,
    "skillToolsAvailableForThisTurn",
  );
  assert.equal(
    compact(initializerOf(availability), ADAPTER_SOURCE),
    "Boolean(supportsStudioToolsForThisTurn&&(isExternalRequest||!imageBase64||selectedModelSummary?.isGguf===true),)",
  );
  const createConditions: string[] = [];
  let pairedToolLists = 0;
  const visit = (node: ts.Node) => {
    if (ts.isArrayLiteralExpression(node)) {
      const toolNames = node.elements
        .filter(ts.isStringLiteral)
        .map((element) => element.text);
      if (toolNames.includes("create_skill")) {
        assert.equal(toolNames.includes("read_skill"), true);
        pairedToolLists += 1;
      }
    }
    if (
      ts.isConditionalExpression(node) &&
      ts.isArrayLiteralExpression(node.whenTrue) &&
      node.whenTrue.elements.some(
        (element) =>
          ts.isStringLiteral(element) && element.text === "create_skill",
      )
    ) {
      createConditions.push(node.condition.getText(ADAPTER_SOURCE));
    }
    ts.forEachChild(node, visit);
  };
  visit(ADAPTER_SOURCE);
  assert.deepEqual(createConditions.sort(), [
    "skillToolsAvailableForThisTurn",
    "skillToolsAvailableForThisTurn",
  ]);
  assert.equal(pairedToolLists, 3);

  assert.equal(
    compact(ADAPTER_SOURCE, ADAPTER_SOURCE).includes(
      "!anyWebEnabledForThisTurn&&!codeExecEnabledForThisTurn&&!imageGenerationEnabledForThisTurn&&!skillToolsAvailableForThisTurn",
    ),
    true,
  );
});

test("token counts include create and server-filtered read schemas", () => {
  const tokenCount = findFunction(ADAPTER_SOURCE, "buildLocalTokenCountExtras");
  const text = compact(tokenCount, ADAPTER_SOURCE);
  assert.equal(text.includes('"create_skill","read_skill"'), true);
  assert.equal(text.includes("hasEnabledSkills"), false);
});
