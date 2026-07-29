// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import {
  buildArtifactSourceKey,
  createArtifactId,
  createChatArtifact,
  hashArtifactCode,
} from "../src/features/chat/artifacts/types.ts";

// The shipped helper the component keys on, not a copy of it.
const sourceKey = buildArtifactSourceKey;

const toolInput = (code: string) => ({
  code,
  source: "tool" as const,
  threadId: "thread-1",
  sourceMessageId: "msg-1",
  sourceToolCallId: "call_0",
});

const fenceInput = (code: string) => ({
  code,
  source: "fence" as const,
  threadId: "thread-1",
  sourceMessageId: "msg-1",
});

test("tool artifact IDs are stable across code changes, so the ID alone is not enough", () => {
  const first = createArtifactId(toolInput("<p>first</p>"));
  const second = createArtifactId(toolInput("<p>second</p>"));
  assert.equal(first, second);
});

test("the source key changes when a tool artifact's code changes", () => {
  const first = createChatArtifact(toolInput("<p>first</p>"));
  const second = createChatArtifact(toolInput("<p>second</p>"));
  assert.notEqual(sourceKey(first), sourceKey(second));
});

test("the source key changes when switching between fence artifacts", () => {
  const first = createChatArtifact(fenceInput("<p>alpha</p>"));
  const second = createChatArtifact(fenceInput("<p>bravo</p>"));
  assert.notEqual(sourceKey(first), sourceKey(second));
});

test("the source key is stable for an unchanged artifact, so no needless remount", () => {
  const code = "<p>same</p>";
  assert.equal(
    sourceKey(createChatArtifact(toolInput(code))),
    sourceKey(createChatArtifact(toolInput(code))),
  );
});

// Equal line count, the shape where Streamdown's comparator sees no change.
test("the source key changes for two canvases with the same shape", () => {
  const first = createChatArtifact(
    toolInput("<html>\n<body>\n<h1>Alpha</h1>\n</body>\n</html>"),
  );
  const second = createChatArtifact(
    toolInput("<html>\n<body>\n<h1>Bravo</h1>\n</body>\n</html>"),
  );
  assert.equal(first.code.length, second.code.length);
  assert.equal(first.code.split("\n").length, second.code.split("\n").length);
  assert.notEqual(sourceKey(first), sourceKey(second));
});

test("hashArtifactCode separates same-length codes and empty from whitespace", () => {
  assert.notEqual(hashArtifactCode("<p>ab</p>"), hashArtifactCode("<p>ba</p>"));
  assert.notEqual(hashArtifactCode(""), hashArtifactCode(" "));
});

const KEYED_BY_HELPER = /^\{buildArtifactSourceKey\(\s*artifact\s*\)\}$/;

const SURFACE_PATH = fileURLToPath(
  new URL(
    "../src/features/chat/artifacts/artifact-surface.tsx",
    import.meta.url,
  ),
);

/** The opening tag of `node`, for both `<x>` and `<x />`. */
const openingTag = (node: ts.Node): ts.JsxOpeningLikeElement | null => {
  if (ts.isJsxSelfClosingElement(node)) return node;
  if (ts.isJsxElement(node)) return node.openingElement;
  return null;
};

/** The `key` expression on the source view's Streamdown, or null if unkeyed. */
function readStreamdownKey(): string | null {
  const source = ts.createSourceFile(
    SURFACE_PATH,
    readFileSync(SURFACE_PATH, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );
  let key: string | null = null;
  const visit = (node: ts.Node): void => {
    const opening = openingTag(node);
    if (opening?.tagName.getText() === "Streamdown") {
      for (const attribute of opening.attributes.properties) {
        if (
          ts.isJsxAttribute(attribute) &&
          attribute.name.getText() === "key"
        ) {
          key = attribute.initializer?.getText() ?? "";
        }
      }
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return key;
}

// Without this the suite passes with the key deleted, which is the regression.
// No DOM renderer is available here, so assert the wiring in the source.
test("the source view's Streamdown is keyed by the shipped helper", () => {
  const key = readStreamdownKey();
  assert.ok(key, "source view <Streamdown> has no key prop");
  assert.match(key, KEYED_BY_HELPER);
});
