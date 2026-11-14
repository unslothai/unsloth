// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import * as routeAbsent from "../src/features/settings/api/settings-route-absent.ts";
import type * as TrustedFoldersApi from "../src/features/settings/api/trusted-tool-folders.ts";

function client(response: () => Response) {
  const requests: { path: string; init?: RequestInit }[] = [];
  const api = loadWithStubs<typeof TrustedFoldersApi>(
    new URL("../src/features/settings/api/trusted-tool-folders.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (path: string, init?: RequestInit) => {
          requests.push({ path, init });
          return response();
        },
      },
      "@/lib/format-fastapi-error": formatFastApiError,
      "./settings-route-absent": routeAbsent,
    },
  );
  return { api, requests };
}

test("one folder per line, blank lines and padding dropped", () => {
  const { api } = client(() => new Response("{}"));
  assert.deepEqual(api.parseTrustedFolders("  /opt/tools \r\n\n C:\\Tools\\bin\n"), [
    "/opt/tools",
    "C:\\Tools\\bin",
  ]);
  assert.deepEqual(api.parseTrustedFolders("\n \n"), []);
});

test("saves the list and returns what the backend stored", async () => {
  const { api, requests } = client(
    () => new Response(JSON.stringify({ folders: ["/real/tools"] }), { status: 200 }),
  );
  assert.deepEqual(await api.updateTrustedToolFolders(["/opt/tools"], "fallback"), ["/real/tools"]);
  assert.equal(requests[0].path, "/api/settings/trusted-tool-folders");
  assert.equal(requests[0].init?.method, "PUT");
  assert.deepEqual(JSON.parse(String(requests[0].init?.body)), { folders: ["/opt/tools"] });
});

test("a refusal surfaces the backend's reason", async () => {
  const { api } = client(
    () =>
      new Response(JSON.stringify({ detail: "Folder 1 is not an absolute path." }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }),
  );
  await assert.rejects(api.updateTrustedToolFolders(["tools"], "fallback"), /absolute path/);
});

test("an older backend without the route reads as absent", async () => {
  const { api } = client(() => new Response("", { status: 404 }));
  await assert.rejects(api.loadTrustedToolFolders("fallback"), {
    name: "SettingsRouteAbsentError",
  });
});

test("Save stays disabled until the stored list has loaded", () => {
  const tab = readFileSync(
    new URL("../src/features/settings/tabs/sandbox-tab.tsx", import.meta.url),
    "utf8",
  );
  const section = tab.slice(tab.indexOf("function TrustedFoldersSection()"));
  // A failed load leaves `saved` null; an enabled Save would send [] over the stored folders.
  assert.match(section, /const unsaveable =\s*saved === null \|\|/);
  assert.match(section, /disabled=\{saving \|\| unsaveable\}/);
});
