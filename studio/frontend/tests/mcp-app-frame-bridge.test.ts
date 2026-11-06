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

// Shim and resize fallback run in tests/studio/playwright_mcp_app_bridge_smoke.py.
const frame = readFileSync(
  new URL("../src/features/chat/mcp-apps/mcp-app-frame.tsx", import.meta.url),
  "utf8",
);

test("the frame is sandboxed without same-origin and talks only through the port", () => {
  assert.match(frame, /sandbox="allow-scripts"\n/);
  assert.doesNotMatch(frame, /allow-same-origin"/);
  assert.match(frame, /data\?\.__unslothMcpApp !== frame\.token/);
  assert.match(frame, /port\.onmessage = handler;/);
  assert.equal(frame.match(/addEventListener\("message"/g)?.length, 1);
});

test("a widget's tools/call is sent as approved only when the user said so", () => {
  const sends = [...frame.matchAll(/\bsend\(([^)]*)\)/g)].map((m) => m[1]);
  assert.deepEqual(sends.sort(), ["alwaysAllowed", "true"]);
  assert.match(
    frame,
    /if \(!allow\) \{\s*return \{[^\n]*DECLINED[^\n]*\n\s*\}\s*const result = await send\(true\);/,
  );
  assert.match(
    frame,
    /const \{ serverId, threadId, sessionId \} = frame\.scope;/,
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

test("the host advertises each server method the bridge proxies", () => {
  for (const [method, capability] of [
    ["tools/call", "serverTools"],
    ["resources/read", "serverResources"],
  ]) {
    assert.match(frame, new RegExp(`case "${method}"`));
    assert.match(
      frame,
      new RegExp(`${capability}: \\{ listChanged: false \\}`),
    );
  }
});

test("Always allow is granted only after the approved call went through", () => {
  assert.match(
    frame,
    /const result = await send\(true\);[\s\S]{0,300}if \(always\) \{\n\s*useChatRuntimeStore\.getState\(\)\.allowToolAlways\(scope, toolKey\);/,
  );
  assert.equal(frame.match(/allowToolAlways\(/g)?.length, 1);
});

test("server-bound bridge requests are capped per frame", () => {
  assert.match(
    frame,
    /const SERVER_METHODS = new Set\(\["tools\/call", "resources\/read"\]\);/,
  );
  assert.match(
    frame,
    /if \(counted && inFlight >= MAX_IN_FLIGHT_SERVER_CALLS\) \{\n\s*run = Promise\.reject\(new RpcError\("Too many requests in flight"\)\);/,
  );
  assert.match(frame, /\.finally\(\(\) => \{\n\s*inFlight -= 1;/);
});

test("a widget's session pairs its thread with the provider's project, as the adapter does", () => {
  const card = readFileSync(
    new URL(
      "../src/components/assistant-ui/tool-fallback.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const widget = card.slice(card.indexOf("function ToolFallbackMcpApp"));
  assert.match(widget, /const projectId = useChatProjectScope\(\);/);
  assert.doesNotMatch(widget, /activeProjectId/);
});

test("a widget's link opens only from the user's Open click, never from the request", () => {
  const openLinkCase = frame.slice(
    frame.indexOf('case "ui/open-link"'),
    frame.indexOf('case "ui/request-display-mode"'),
  );
  assert.doesNotMatch(openLinkCase, /openLink\(/);
  assert.match(openLinkCase, /link: url,/);
  assert.match(openLinkCase, /return opened \? \{\} : \{ isError: true \};/);
  assert.match(frame, /if \(allow && asking\.link\) openLink\(asking\.link\);/);
});

test("a widget's server-bound requests use the scope its template was fetched for", () => {
  assert.match(frame, /scope: \{ serverId, threadId, sessionId \}/);
  assert.equal(
    frame.match(/const \{ serverId, threadId, sessionId \} = frame\.scope;/g)
      ?.length,
    2,
  );
  assert.doesNotMatch(frame, /latest\.current\.props;\n\s*const scope/);
  assert.doesNotMatch(frame, /readMcpUiResource\(now\.serverId/);
});
