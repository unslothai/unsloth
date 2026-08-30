// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Asserted at the source level: a future edit calling the tombstone helpers directly would
// reintroduce the leak on whichever delete path a runtime test did not cover.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const WRAPPERS = new Set(["forgetChatThread", "forgetChatThreads"]);
const TOMBSTONES = new Set(["markChatThreadDeleted", "markChatThreadsDeleted"]);

const MODULE_PATH = fileURLToPath(
  new URL("../src/features/chat/utils/chat-history-storage.ts", import.meta.url),
);

function parseModule(): ts.SourceFile {
  return ts.createSourceFile(
    MODULE_PATH,
    readFileSync(MODULE_PATH, "utf8"),
    ts.ScriptTarget.ES2022,
    true,
  );
}

/** Names of functions calling `markChatThread(s)Deleted`, by enclosing declaration. */
function tombstoneCallers(sf: ts.SourceFile): string[] {
  const callers: string[] = [];
  const visit = (node: ts.Node, enclosing: string | null): void => {
    let scope = enclosing;
    if (
      (ts.isFunctionDeclaration(node) || ts.isMethodDeclaration(node)) &&
      node.name &&
      ts.isIdentifier(node.name)
    ) {
      scope = node.name.text;
    }
    if (
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      TOMBSTONES.has(node.expression.text)
    ) {
      const { line } = sf.getLineAndCharacterOfPosition(node.getStart(sf));
      callers.push(`${node.expression.text} at line ${line + 1} in ${scope ?? "<top level>"}`);
    }
    ts.forEachChild(node, (child) => visit(child, scope));
  };
  visit(sf, null);
  return callers;
}

test("the tombstone helpers are only called from the wrappers that clear the map", () => {
  const sf = parseModule();
  const offenders = tombstoneCallers(sf).filter(
    (entry) => ![...WRAPPERS].some((wrapper) => entry.endsWith(`in ${wrapper}`)),
  );
  assert.deepEqual(
    offenders,
    [],
    "call forgetChatThread/forgetChatThreads instead, so server-owned markers are cleared too",
  );
});

test("both wrappers exist and each clears the map", () => {
  const text = readFileSync(MODULE_PATH, "utf8");
  for (const wrapper of WRAPPERS) {
    assert.match(
      text,
      new RegExp(`function ${wrapper}\\b`),
      `${wrapper} must exist for the delete paths to use`,
    );
  }
  const sf = parseModule();
  const cleared: string[] = [];
  const visit = (node: ts.Node, enclosing: string | null): void => {
    let scope = enclosing;
    if (ts.isFunctionDeclaration(node) && node.name) scope = node.name.text;
    if (
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "clearServerOwnedChatMessages" &&
      scope !== null
    ) {
      cleared.push(scope);
    }
    ts.forEachChild(node, (child) => visit(child, scope));
  };
  visit(sf, null);
  for (const wrapper of WRAPPERS) {
    assert.ok(
      cleared.includes(wrapper),
      `${wrapper} must call clearServerOwnedChatMessages`,
    );
  }
});

test("at least one real delete path is wired to a wrapper", () => {
  const sf = parseModule();
  const used: string[] = [];
  const visit = (node: ts.Node): void => {
    if (
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      WRAPPERS.has(node.expression.text)
    ) {
      used.push(node.expression.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  assert.ok(used.length >= 2, `expected the delete paths to use the wrappers, saw ${used}`);
});
