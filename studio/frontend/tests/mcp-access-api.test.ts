// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The agent access (MCP) switch reads and writes one owner setting. Pinned here: the
// snake_case schema maps to camelCase, the PUT sends only `enabled`, and the 409 (set by
// the environment) and 403 (changed from an API key) details reach the caller verbatim.

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

test("the schema maps from snake_case to camelCase", () => {
  const { api } = client(() => Response.json(BODY));
  assert.deepEqual(api.mcpAccessFromApi({ ...BODY, forced_by_env: true }), {
    enabled: true,
    forcedByEnv: true,
    url: "http://127.0.0.1:8888/mcp/",
  });
});

test("loading reads the owner settings route", async () => {
  const c = client(() => Response.json(BODY));
  assert.deepEqual(await c.api.loadMcpAccess(), {
    enabled: true,
    forcedByEnv: false,
    url: "http://127.0.0.1:8888/mcp/",
  });
  assert.deepEqual(c.requests, [
    { path: "/api/settings/mcp-access", init: undefined },
  ]);
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

test("the client is built on authFetch and readFastApiError", () => {
  assert.match(API, /import \{ authFetch \} from "@\/features\/auth";/);
  assert.match(API, /readFastApiError\(res, /);
  assert.match(API, /forcedByEnv: settings\.forced_by_env/);
});
