// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

// No DOM renderer here and the frame pulls in React plus the runtime store, so
// assert the wiring in the source the way artifact-source-key.test.ts does.
const sourceFile = (relative: string): ts.SourceFile => {
  const path = fileURLToPath(new URL(relative, import.meta.url));
  return ts.createSourceFile(
    path,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );
};

const SURFACE = "../src/features/chat/artifacts/artifact-surface.tsx";
const FRAME = "../src/features/chat/artifacts/html-frame.tsx";

/** The opening tag of `node`, for both `<x>` and `<x />`. */
const openingTag = (node: ts.Node): ts.JsxOpeningLikeElement | null => {
  if (ts.isJsxSelfClosingElement(node)) return node;
  if (ts.isJsxElement(node)) return node.openingElement;
  return null;
};

/** The `allowNetworkAccess` expression on the surface's frame, or null. */
function readAllowNetworkAccessProp(): string | null {
  const source = sourceFile(SURFACE);
  let value: string | null = null;
  const visit = (node: ts.Node): void => {
    const opening = openingTag(node);
    if (opening?.tagName.getText() === "ArtifactHtmlFrame") {
      for (const attribute of opening.attributes.properties) {
        if (
          ts.isJsxAttribute(attribute) &&
          attribute.name.getText() === "allowNetworkAccess"
        ) {
          value = attribute.initializer?.getText() ?? "";
        }
      }
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return value;
}

// The bug: a ```html fence is source "fence", so gating on "tool" left every
// fenced canvas on the strict CSP and a CDN import (three.js) silently died.
test("the preview frame is offered network access regardless of canvas source", () => {
  const value = readAllowNetworkAccessProp();
  assert.ok(value, "<ArtifactHtmlFrame> has no allowNetworkAccess prop");
  assert.doesNotMatch(
    value,
    /\bsource\b/,
    "allowNetworkAccess must not discriminate on artifact.source",
  );
  assert.equal(value, "{true}");
});

/** The `if` condition guarding the `allow_network` query flag, or null. */
function readAllowNetworkGuard(): string | null {
  const source = sourceFile(FRAME);
  let condition: string | null = null;
  const visit = (node: ts.Node): void => {
    if (ts.isIfStatement(node) && node.thenStatement.getText().includes("allow_network")) {
      condition = node.expression.getText();
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return condition;
}

// Without this the suite passes with the setting ignored, which would hand every
// canvas the permissive CSP whether or not the user opted in.
test("the permissive CSP still requires the user's network setting", () => {
  const condition = readAllowNetworkGuard();
  assert.ok(condition, "no guard found around the allow_network flag");
  assert.match(condition, /\bnetworkAccessEnabled\b/);
  assert.match(condition, /\ballowNetworkAccess\b/);
});

/** The default for the frame's `allowNetworkAccess` parameter, or null. */
function readAllowNetworkAccessDefault(): string | null {
  const source = sourceFile(FRAME);
  let initializer: string | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isFunctionDeclaration(node) &&
      node.name?.getText() === "ArtifactHtmlFrame"
    ) {
      const binding = node.parameters[0]?.name;
      if (binding && ts.isObjectBindingPattern(binding)) {
        for (const element of binding.elements) {
          if (element.name.getText() === "allowNetworkAccess") {
            initializer = element.initializer?.getText() ?? "";
          }
        }
      }
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return initializer;
}

// A future caller that forgets the prop must land on the strict CSP, not inherit
// the permissive one.
test("the frame defaults to no network access", () => {
  assert.equal(readAllowNetworkAccessDefault(), "false");
});
