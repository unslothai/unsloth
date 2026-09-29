// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

// No DOM renderer: assert the wiring in the source.
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

function liftFunction<T>(signature: string): T {
  const start = text.indexOf(signature);
  assert.ok(start >= 0, `${signature} is no longer in mcp-app-frame.tsx`);
  const end = text.indexOf("\n}\n", start);
  assert.ok(end > start, `${signature} has no top-level closing brace`);
  const declaration = text.slice(start, end + 3).replace(/^export /, "");
  const name = /function (\w+)/.exec(declaration)?.[1];
  return new Function(
    `${
      ts.transpileModule(declaration, {
        compilerOptions: { target: ts.ScriptTarget.ES2020 },
      }).outputText
    }; return ${name};`,
  )() as T;
}

test("the view's handle on the host is its own port", () => {
  const shimStart = text.indexOf("export function bridgeShim");
  assert.ok(shimStart >= 0, "bridgeShim is no longer declared in mcp-app-frame.tsx");
  const shim = text.slice(shimStart, text.indexOf("\n}\n", shimStart));
  assert.ok(
    /Object\.defineProperty\(window, name, \{ value: port, configurable: true \}\)/.test(
      shim,
    ),
    "window.parent must BE the port, so event.source === window.parent still holds",
  );
  assert.ok(
    /for \(const name of \["parent", "top"\]\)/.test(shim),
    "shadowing parent alone leaves top as a second, unbound handle on the host",
  );
  assert.ok(
    /source: port,/.test(shim),
    "a re-dispatched reply must name the port as its source, or the identity check fails",
  );
  assert.ok(
    /Array\.isArray\(a\) \? a : Array\.isArray\(b\) \? b : \[\]/.test(shim),
    "the port must tolerate postMessage(message, targetOrigin) as a Window would",
  );
  assert.ok(
    /envelope\.__unslothMcpApp !== bridgeToken/.test(text),
    "the handshake must still require the seeded document's token",
  );
  assert.ok(
    /port\.onmessage = handler/.test(text),
    "protocol traffic must be read off the port, not the window",
  );
});

// withBridgeShim is asserted in a real browser: tests/studio/playwright_mcp_app_bridge_smoke.py.

test("the token is minted per fetched template", () => {
  const token = declarationText("bridgeToken");
  assert.ok(
    /newBridgeToken\(\)/.test(token),
    "the bridge token must come from the minter, which handles a non-secure origin",
  );
  assert.ok(
    /\[resource\]/.test(token),
    "the bridge token must be re-minted whenever the template is refetched",
  );
});

test("the view is seeded from the server's own blocks", () => {
  const seedView = declarationText("seedView");
  assert.ok(
    /for \(const block of ui\.content \?\? \[\]\)/.test(seedView),
    "the seed must walk the server's blocks in order",
  );
  assert.ok(
    /block\.data === undefined/.test(seedView) && /images\.shift\(\)/.test(seedView),
    "an image block arrives without data and must be refilled from the image sentinel",
  );
  assert.ok(
    !/resultText/.test(text),
    "the flattened body has no seeding role left",
  );
});

test("the frame is armed and listening inside the commit, not after it", () => {
  assert.ok(
    /useLayoutEffect\(\(\) => \{\s*pendingPostRef\.current = true;/.test(text),
    "arming pendingPostRef must happen in the commit, not in a passive effect",
  );
  assert.ok(
    /useLayoutEffect\(\(\) => \{\s*const respond =/.test(text),
    "the message listener must be attached in the commit, before the view can post",
  );
  assert.equal(
    (text.match(/initializedRef\.current = false/g) ?? []).length,
    1,
    "initializedRef must be reset once, alongside the other per-load state",
  );
});

test("the bridge token survives a non-secure Studio origin", () => {
  const mint = liftFunction<() => string | null>("export function newBridgeToken(");
  const realCrypto = globalThis.crypto;
  const withCrypto = (value: unknown, run: () => void): void => {
    Object.defineProperty(globalThis, "crypto", {
      value,
      configurable: true,
      writable: true,
    });
    try {
      run();
    } finally {
      Object.defineProperty(globalThis, "crypto", {
        value: realCrypto,
        configurable: true,
        writable: true,
      });
    }
  };

  withCrypto({ randomUUID: () => "from-random-uuid" }, () => {
    assert.equal(mint(), "from-random-uuid");
  });

  withCrypto({ getRandomValues: realCrypto.getRandomValues.bind(realCrypto) }, () => {
    const first = mint();
    assert.match(String(first), /^[0-9a-f]{32}$/, "128 bits of hex from getRandomValues");
    assert.notEqual(first, mint(), "a fresh token every call");
  });

  withCrypto(undefined, () => assert.equal(mint(), null));
  assert.ok(
    /resource && !bridgeToken/.test(text),
    "a frame with no token must report the failure rather than render",
  );
});
