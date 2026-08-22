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

/** A standalone `export function` from the frame, evaluated on its own. */
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
    /newBridgeToken\(\)/.test(token),
    "the bridge token must come from the minter, which handles a non-secure origin",
  );
  assert.ok(
    /\[resource\]/.test(token),
    "the bridge token must be re-minted whenever the template is refetched",
  );
});

test("the view is seeded from the server's own blocks", () => {
  // _flatten_result builds the model-facing transcript: an image-only result reads
  // "[1 image attached; displayed to the user]" and a structuredContent-only one is
  // a Python repr of the payload. Neither is in the server's CallToolResult, so the
  // seed comes from the blocks the envelope carries, image bytes put back from the
  // image sentinel rather than duplicated on the seed line.
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
  // The iframe starts fetching the shell the moment it is committed, and onLoad
  // reads pendingPostRef. A passive effect is queued during that same commit and
  // normally wins -- 10/10 against a real local server here -- but they are
  // different task sources with nothing ordering them, and losing once leaves the
  // widget on the empty shell permanently, with no second load to retry it.
  assert.ok(
    /useLayoutEffect\(\(\) => \{\s*pendingPostRef\.current = true;/.test(text),
    "arming pendingPostRef must happen in the commit, not in a passive effect",
  );
  assert.ok(
    /useLayoutEffect\(\(\) => \{\s*const respond =/.test(text),
    "the message listener must be attached in the commit, before the view can post",
  );
  // initializedRef used to be reset in its own passive effect, which could land
  // after the view had already said `initialized` and silently stop theme updates.
  assert.equal(
    (text.match(/initializedRef\.current = false/g) ?? []).length,
    1,
    "initializedRef must be reset once, alongside the other per-load state",
  );
});

test("the bridge token survives a non-secure Studio origin", () => {
  // Studio is reachable over plain HTTP on a LAN address, where crypto.randomUUID
  // is simply undefined; calling it unconditionally threw on render and took every
  // widget with it. getRandomValues is not secure-context gated, so it is the
  // fallback -- not the Date.now()+Math.random() the attachment adapters use, which
  // is fine for an id and not for the value that names the seeded document.
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

  // No Web Crypto at all: a guessable token would be worse than none, and the
  // component renders the failure instead.
  withCrypto(undefined, () => assert.equal(mint(), null));
  assert.ok(
    /resource && !bridgeToken/.test(text),
    "a frame with no token must report the failure rather than render",
  );
});
