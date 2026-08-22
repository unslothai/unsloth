// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

// No DOM renderer here and the frame pulls in React plus the runtime store, so
// assert the wiring in the source, the way artifact-frame-network-access.test.ts does.
const FRAME = "../src/features/chat/mcp-apps/mcp-app-frame.tsx";

const path = fileURLToPath(new URL(FRAME, import.meta.url));
const text = readFileSync(path, "utf8");
const source = ts.createSourceFile(
  path,
  text,
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);

/** The body of the `const <name> = ...` initializer, whatever it is wrapped in. */
function declarationText(name: string): string {
  let found: string | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === name &&
      node.initializer
    ) {
      found = node.initializer.getText();
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(found, `${name} is no longer declared in mcp-app-frame.tsx`);
  return found as unknown as string;
}

test("a load the host did not seed revokes the bridge", () => {
  // A sandboxed frame keeps one contentWindow and reports origin "null" whatever
  // it navigates to, so the source/origin pair cannot tell the widget from the
  // page an ordinary in-widget link moved it to. Verified in Chromium: a document
  // fetched from another site passes both checks. Without a third gate that page
  // inherits the server's app-visible tools through tools/call.
  const onLoad = declarationText("onLoad");
  assert.ok(
    /bridgeLiveRef\.current\s*=\s*false/.test(onLoad),
    "onLoad must revoke the bridge on a load it did not initiate",
  );
  assert.ok(
    /bridgeLiveRef\.current\s*=\s*true/.test(onLoad),
    "onLoad must arm the bridge for the document it seeds",
  );
  // Revocation before the early return, arming only after it.
  assert.ok(
    onLoad.indexOf("bridgeLiveRef.current = false") <
      onLoad.indexOf("bridgeLiveRef.current = true"),
    "the revoking branch must be the early return, not the seeded path",
  );
});

test("the message handler refuses a frame whose bridge is not armed", () => {
  assert.ok(
    /if\s*\(!bridgeLiveRef\.current\)\s*return;/.test(text),
    "the postMessage handler must drop messages once the bridge is revoked",
  );
  // Re-seeding a frame must start from revoked, or a self-navigated document
  // stays trusted across a template refetch.
  assert.ok(
    /pendingPostRef\.current\s*=\s*true;\s*\n\s*bridgeLiveRef\.current\s*=\s*false;/.test(
      text,
    ),
    "arming pendingPost must also reset the bridge to revoked",
  );
});

test("the view is seeded from the tool's own text, never the flattened body", () => {
  // _flatten_result builds the model-facing transcript: an image-only result reads
  // "[1 image attached; displayed to the user]" and a structuredContent-only one is
  // a Python repr of the payload. Neither is in the server's CallToolResult, so a
  // fallback to it hands the view host prose as the tool's answer.
  const seedView = declarationText("seedView");
  assert.ok(
    /ui\.text\s*\?\s*\[\{\s*type:\s*"text",\s*text:\s*ui\.text\s*\}\]/.test(seedView),
    "the text content block must come from ui.text alone",
  );
  assert.ok(
    !/resultText/.test(seedView),
    "seedView must not fall back to the flattened result text",
  );
  assert.ok(
    !/resultText/.test(text),
    "the resultText prop is the flattened body; it has no seeding role left",
  );
});
