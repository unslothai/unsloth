// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// generation-recovery-monotonic.test.ts holds the rule. This holds the wiring,
// because the rule is enforced at exactly one place: the object a recovery
// publish hands to `view.thread().import`. Deleting the call there restores the
// rewind while leaving every behavioural test green, so the call site is pinned
// rather than trusted.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const SOURCE = fileURLToPath(
  new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
);

const parsed = ts.createSourceFile(
  "runtime-provider.tsx",
  readFileSync(SOURCE, "utf8"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

const findRecoveryFunction = (): ts.FunctionDeclaration | null => {
  let found: ts.FunctionDeclaration | null = null;
  const walk = (node: ts.Node): void => {
    if (
      ts.isFunctionDeclaration(node) &&
      node.name?.text === "scheduleGenerationRecovery"
    ) {
      found = node;
      return;
    }
    ts.forEachChild(node, walk);
  };
  walk(parsed);
  return found;
};

const propertyName = (property: ts.ObjectLiteralElementLike): string => {
  const name = property.name;
  return name !== undefined && ts.isIdentifier(name) ? name.text : "";
};

/**
 * The object literals the publish builds for the thread import: the ones that
 * set both a `content` and a `status`, which is the message body swap.
 */
const publishedMessageObjects = (
  root: ts.Node,
): ts.ObjectLiteralExpression[] => {
  const objects: ts.ObjectLiteralExpression[] = [];
  const walk = (node: ts.Node): void => {
    if (ts.isObjectLiteralExpression(node)) {
      const names = node.properties.map(propertyName);
      if (names.includes("content") && names.includes("status")) {
        objects.push(node);
      }
    }
    ts.forEachChild(node, walk);
  };
  walk(root);
  return objects;
};

test("the recovery publish swaps the body through the monotonicity guard", () => {
  const recovery = findRecoveryFunction();
  assert.ok(recovery, "scheduleGenerationRecovery was not found");
  const objects = publishedMessageObjects(recovery);
  assert.equal(
    objects.length,
    1,
    "expected exactly one published message body in the recovery",
  );

  const content = objects[0].properties.find(
    (property) =>
      ts.isPropertyAssignment(property) &&
      ts.isIdentifier(property.name) &&
      property.name.text === "content",
  );
  assert.ok(
    content && ts.isPropertyAssignment(content),
    "the published body must assign `content` rather than pass it through",
  );

  const initializer = content.initializer;
  assert.ok(
    ts.isCallExpression(initializer) &&
      ts.isIdentifier(initializer.expression) &&
      initializer.expression.text === "recoveredContentToImport",
    "`content` must come from recoveredContentToImport, or the recovery can rewind the reply again",
  );
  assert.equal(
    initializer.arguments.length,
    2,
    "recoveredContentToImport takes the view's body first, then the recovered one",
  );
  assert.equal(
    initializer.arguments[0].getText(parsed),
    "item.message.content",
    "the view's own body must be the first argument",
  );
});
