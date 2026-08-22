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

test("only the document the host seeded can reach the bridge", () => {
  // A sandboxed frame keeps one contentWindow and reports origin "null" whatever
  // it navigates to, and the replacement document's inline scripts run BEFORE the
  // iframe's load event -- both verified in Chromium by
  // tests/studio/playwright_mcp_app_bridge_smoke.py, which also shows a load-flag
  // gate accepting the navigated document's tools/call. So the token the shim
  // stamps in is what names the sender.
  assert.ok(
    /envelope\.__unslothMcpApp !== bridgeToken/.test(text),
    "the handler must require the seeded document's token",
  );
  assert.ok(
    /const data = envelope\.message/.test(text),
    "the handled payload must be the envelope's message, not the raw event data",
  );
  // bridgeShim is a function declaration, not a const, so slice it out directly.
  const shimStart = text.indexOf("export function bridgeShim");
  assert.ok(shimStart >= 0, "bridgeShim is no longer declared in mcp-app-frame.tsx");
  const shim = text.slice(shimStart, text.indexOf("\n}\n", shimStart));
  assert.ok(
    /Object\.defineProperty\(window, name, \{ value: proxy, configurable: true \}\)/.test(
      shim,
    ),
    "the shim must shadow the frame's handle on the host so every message is stamped",
  );
  assert.ok(
    /for \(const name of \["parent", "top"\]\)/.test(shim),
    "shadowing parent alone leaves top as an unstamped handle on the host",
  );
});

test("the token is minted per fetched template", () => {
  // A token reused across re-seeds would let a document that captured one earlier
  // keep talking after the frame moved on.
  const token = declarationText("bridgeToken");
  assert.ok(
    /crypto\.randomUUID\(\)/.test(token),
    "the bridge token must be unguessable",
  );
  assert.ok(
    /\[resource\]/.test(token),
    "the bridge token must be re-minted whenever the template is refetched",
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
