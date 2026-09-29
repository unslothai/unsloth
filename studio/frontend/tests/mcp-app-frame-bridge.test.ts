// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  RESIZE_FALLBACK,
  bridgeShim,
  newBridgeToken,
} from "../src/features/chat/mcp-apps/mcp-ui.ts";

// No DOM renderer here; the shim and the resize fallback run in a real browser in
// tests/studio/playwright_mcp_app_bridge_smoke.py.
const frame = readFileSync(
  new URL("../src/features/chat/mcp-apps/mcp-app-frame.tsx", import.meta.url),
  "utf8",
);

test("the frame is sandboxed without same-origin and talks only through the port", () => {
  assert.match(frame, /sandbox="allow-scripts"\n/);
  assert.doesNotMatch(frame, /allow-same-origin"/);
  assert.match(frame, /data\?\.__unslothMcpApp !== frame\.token/);
  assert.match(frame, /port\.onmessage = handler;/);
  // Exactly one window listener: the handshake. Protocol traffic is never read off the window.
  assert.equal(frame.match(/addEventListener\("message"/g)?.length, 1);
});

test("a widget's tools/call is sent as approved only when the user said so", () => {
  const sends = [...frame.matchAll(/\bsend\(([^)]*)\)/g)].map((m) => m[1]);
  assert.deepEqual(sends.sort(), ["alwaysAllowed", "true"]);
  assert.match(
    frame,
    /if \(!allow\) \{\s*return \{[^\n]*DECLINED[^\n]*\n\s*\}\s*return send\(true\);/,
  );
  assert.match(
    frame,
    /const \{ serverId, threadId, sessionId \} = latest\.current\.props;/,
  );
});

test("the shim binds window.parent and window.top to its own port", () => {
  const shim = bridgeShim("tok-1", "https://studio.test");
  assert.match(shim, /for \(const name of \["parent", "top"\]\)/);
  assert.match(shim, /source: port, origin: "https:\/\/studio\.test"/);
  assert.match(shim, /__unslothMcpApp: "tok-1"/);
});

test("the size fallback measures content, so a widget can shrink", () => {
  assert.doesNotMatch(RESIZE_FALLBACK, /scrollHeight/);
  assert.match(RESIZE_FALLBACK, /html\.style\.height="max-content"/);
});

test("the bridge token survives a non-secure Studio origin", () => {
  const real = globalThis.crypto;
  const withCrypto = (value: unknown, run: () => void) => {
    Object.defineProperty(globalThis, "crypto", {
      value,
      configurable: true,
      writable: true,
    });
    try {
      run();
    } finally {
      Object.defineProperty(globalThis, "crypto", {
        value: real,
        configurable: true,
        writable: true,
      });
    }
  };
  withCrypto({ randomUUID: () => "uuid" }, () =>
    assert.equal(newBridgeToken(), null),
  );
  withCrypto({ getRandomValues: real.getRandomValues.bind(real) }, () => {
    const first = newBridgeToken();
    assert.match(String(first), /^[0-9a-f]{32}$/);
    assert.notEqual(first, newBridgeToken());
  });
  withCrypto(undefined, () => assert.equal(newBridgeToken(), null));
});
