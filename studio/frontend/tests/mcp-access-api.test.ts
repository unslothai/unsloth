// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type * as McpAccessApi from "../src/features/settings/api/mcp-access.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const API = readSrc("features/settings/api/mcp-access.ts");

function client(response: () => Response) {
  const requests: { path: string; init?: RequestInit }[] = [];
  const api = loadWithStubs<typeof McpAccessApi>(
    new URL("../src/features/settings/api/mcp-access.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (path: string, init?: RequestInit) => {
          requests.push({ path, init });
          return response();
        },
      },
      "@/lib/format-fastapi-error": formatFastApiError,
    },
  );
  return { api, requests };
}

const BODY = {
  enabled: true,
  forced_by_env: false,
  url: "http://127.0.0.1:8888/mcp/",
};

test("loading reads the owner settings route and maps it to camelCase", async () => {
  for (const forced of [false, true]) {
    const c = client(() => Response.json({ ...BODY, forced_by_env: forced }));
    assert.deepEqual(await c.api.loadMcpAccess(), {
      enabled: true,
      forcedByEnv: forced,
      url: "http://127.0.0.1:8888/mcp/",
      tools: null,
    });
    assert.deepEqual(c.requests, [
      { path: "/api/settings/mcp-access", init: undefined },
    ]);
  }
});

test("saving sends only the switch value", async () => {
  const c = client(() => Response.json({ ...BODY, enabled: false }));
  const saved = await c.api.updateMcpAccess(false);
  assert.equal(saved.enabled, false);
  assert.deepEqual(c.requests, [
    {
      path: "/api/settings/mcp-access",
      init: {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: false }),
      },
    },
  ]);
});

test("the tool list is kept in server order, and an older backend without one maps to null", async () => {
  const tools = [
    { name: "studio_status", area: "status", enabled: true },
    { name: "cancel", area: "cancel", enabled: false },
  ];
  const c = client(() => Response.json({ ...BODY, tools }));
  assert.deepEqual((await c.api.loadMcpAccess()).tools, tools);
  assert.equal(c.api.mcpAccessFromApi(BODY).tools, null);
});

test("saving tools sends only the full turned-off list", async () => {
  const c = client(() =>
    Response.json({
      ...BODY,
      tools: [{ name: "cancel", area: "cancel", enabled: false }],
    }),
  );
  const saved = await c.api.updateMcpTools({ cancel: false });
  assert.deepEqual(saved.tools, [
    { name: "cancel", area: "cancel", enabled: false },
  ]);
  assert.deepEqual(c.requests, [
    {
      path: "/api/settings/mcp-access",
      init: {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tool_states: { cancel: false } }),
      },
    },
  ]);
});

test("a forced setting and an API-key caller surface the server's reason", async () => {
  for (const [status, detail] of [
    [409, "Set by UNSLOTH_STUDIO_ENABLE_MCP."],
    [403, "Agent access (MCP) can only be changed from the Unsloth UI."],
  ] as const) {
    const c = client(() => Response.json({ detail }, { status }));
    await assert.rejects(c.api.updateMcpAccess(true), (error: Error) => {
      assert.equal(error.message, detail);
      return true;
    });
  }
});

test("a body without a detail falls back to a status line", async () => {
  const c = client(() => new Response("bad gateway", { status: 502 }));
  await assert.rejects(
    c.api.loadMcpAccess(),
    /Failed to load agent access settings \(502\)/,
  );
});

test("requests run one at a time, in call order, and a failure does not stall the next", async () => {
  // A read sent during a write (Settings > API mounting while Agents saves) must see the write.
  const started: string[] = [];
  const pending: (() => void)[] = [];
  const api = loadWithStubs<typeof McpAccessApi>(
    new URL("../src/features/settings/api/mcp-access.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: (_path: string, init?: RequestInit) => {
          started.push(init?.method ?? "GET");
          return new Promise<Response>((resolve) =>
            pending.push(() =>
              resolve(
                started.length === 1
                  ? new Response("down", { status: 502 })
                  : Response.json(BODY),
              ),
            ),
          );
        },
      },
      "@/lib/format-fastapi-error": formatFastApiError,
    },
  );
  const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
  const write = api.updateMcpTools({ cancel: false });
  const read = api.loadMcpAccess();
  await tick();
  assert.deepEqual(started, ["PUT"]);
  pending.shift()?.();
  await assert.rejects(write, /Failed to update agent access tools \(502\)/);
  await tick();
  assert.deepEqual(started, ["PUT", "GET"]);
  pending.shift()?.();
  assert.equal((await read).enabled, true);
});

test("the client is built on authFetch and readFastApiError", () => {
  assert.match(API, /import \{ authFetch \} from "@\/features\/auth";/);
  assert.match(API, /readFastApiError\(res, /);
  assert.match(API, /forcedByEnv: settings\.forced_by_env/);
});
