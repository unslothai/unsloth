// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type OfflineDocsApi = {
  loadOfflineDocsSettings: () => Promise<Record<string, unknown>>;
  updateOfflineDocsSettings: (update: {
    path?: string | null;
    autoUpdate?: boolean;
  }) => Promise<Record<string, unknown>>;
  refreshOfflineDocs: () => Promise<Record<string, unknown>>;
};

const API_STATUS = {
  path: "/home/user/Documents/Unsloth",
  auto_update: true,
  file: "/home/user/Documents/Unsloth/Unsloth Docs LLMs Full Text [Documentation] (2026-10-26).md",
  updated_at: "2026-10-26T01:00:00+00:00",
  release: "2026.10.1",
  current_release: "2026.10.1",
  error: null,
  refreshing: false,
};

function loadApi(
  respond: (url: string, init?: RequestInit) => Response,
): OfflineDocsApi {
  return loadWithStubs<OfflineDocsApi>(
    new URL("../src/features/settings/api/offline-docs.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (url: string, init?: RequestInit) => respond(url, init),
      },
      "@/lib/format-fastapi-error": {
        readFastApiError: async (_response: Response, fallback: string) =>
          fallback,
      },
    },
  );
}

test("the status is mapped to camelCase", async () => {
  const api = loadApi(() => Response.json(API_STATUS));

  assert.deepEqual(await api.loadOfflineDocsSettings(), {
    path: API_STATUS.path,
    autoUpdate: true,
    file: API_STATUS.file,
    updatedAt: API_STATUS.updated_at,
    release: "2026.10.1",
    currentRelease: "2026.10.1",
    error: null,
    refreshing: false,
  });
});

test("an update sends only the fields it changes", async () => {
  const bodies: unknown[] = [];
  const api = loadApi((_url, init) => {
    bodies.push(JSON.parse(String(init?.body)));
    return Response.json(API_STATUS);
  });

  await api.updateOfflineDocsSettings({ autoUpdate: false });
  await api.updateOfflineDocsSettings({ path: null });
  await api.updateOfflineDocsSettings({ path: "/data/docs", autoUpdate: true });

  assert.deepEqual(bodies, [
    { auto_update: false },
    { path: null },
    { path: "/data/docs", auto_update: true },
  ]);
});

test("a refresh posts to the refresh route and surfaces failures", async () => {
  const calls: string[] = [];
  const api = loadApi((url, init) => {
    calls.push(`${init?.method ?? "GET"} ${url}`);
    return new Response(null, { status: 502 });
  });

  await assert.rejects(api.refreshOfflineDocs(), /Failed to download the Unsloth docs/);
  assert.deepEqual(calls, ["POST /api/settings/offline-docs/refresh"]);
});
