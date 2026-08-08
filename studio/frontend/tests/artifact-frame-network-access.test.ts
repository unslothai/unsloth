// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { openingTag } from "./helpers/tsx-ast.ts";

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

/** Every `<ArtifactHtmlFrame>` opening tag in the artifact surface. */
function readFrameOpeningTags(): string[] {
  const source = sourceFile(SURFACE);
  const tags: string[] = [];
  const visit = (node: ts.Node): void => {
    const opening = openingTag(node);
    if (opening?.tagName.getText() === "ArtifactHtmlFrame") {
      tags.push(opening.getText());
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return tags;
}

// The bug: a fenced html block is source "fence", so gating on "tool" left every
// fenced canvas on the strict CSP and a CDN import (three.js) silently died.
// Every call site is checked, so a source-gated one cannot hide behind an
// ungated one that happens to be visited later.
test("no canvas preview is gated on the artifact source", () => {
  const tags = readFrameOpeningTags();
  assert.ok(
    tags.length > 0,
    "<ArtifactHtmlFrame> not found in the artifact surface",
  );
  for (const tag of tags) {
    assert.doesNotMatch(
      tag,
      /\bsource\b/,
      "the preview frame must not discriminate on artifact.source",
    );
  }
});

/** Every condition guarding an `allow_network` query flag in the frame. */
function readAllowNetworkGuards(): string[] {
  const source = sourceFile(FRAME);
  const conditions: string[] = [];
  const visit = (node: ts.Node): void => {
    if (
      ts.isIfStatement(node) &&
      node.thenStatement.getText().includes("allow_network")
    ) {
      conditions.push(node.expression.getText());
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return conditions;
}

// The permissive CSP is opt-in, and that setting is the whole gate. Asserting the
// condition exactly is the point: a second operand is how the gate gets defeated
// (an || that short-circuits past it, or a source check smuggled back in), and
// there is no legitimate reason for one here.
test("the permissive CSP is gated on the user's network setting alone", () => {
  const conditions = readAllowNetworkGuards();
  assert.equal(conditions.length, 1, "expected exactly one allow_network guard");
  assert.equal(conditions[0], "networkAccessEnabled");
});
