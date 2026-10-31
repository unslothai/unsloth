// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Settings = {
  maxChars: number;
  defaultChars: number;
  minChars: number;
  maxAllowedChars: number;
  lockedByEnvironment: boolean;
};

type Api = {
  loadToolResultLimit: (fallback: string) => Promise<Settings>;
  updateToolResultLimit: (maxChars: number, fallback: string) => Promise<Settings>;
  toolResultLimitChoices: (settings: Settings) => number[];
};

type Call = { url: string; init?: RequestInit };

function loadApi(respond: (call: Call) => Response): { api: Api; calls: Call[] } {
  const calls: Call[] = [];
  const api = loadWithStubs<Api>(
    new URL("../src/features/settings/api/tool-result-limit.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (url: string, init?: RequestInit) => {
          const call = { url, init };
          calls.push(call);
          return respond(call);
        },
      },
      "@/lib/format-fastapi-error": {
        readFastApiError: async (response: Response, fallback: string) => {
          const body = (await response.json().catch(() => null)) as {
            detail?: string;
          } | null;
          return body?.detail ?? fallback;
        },
      },
    },
    { relativePassthrough: true },
  );
  return { api, calls };
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const API_BODY = {
  max_chars: 16000,
  default_chars: 16000,
  min_chars: 2000,
  max_allowed_chars: 200000,
  locked_by_environment: false,
};

test("loads the limit from the settings route", async () => {
  const { api, calls } = loadApi(() => json(API_BODY));
  const settings = await api.loadToolResultLimit("fallback");
  assert.equal(calls[0].url, "/api/settings/tool-result-limit");
  assert.deepEqual(settings, {
    maxChars: 16000,
    defaultChars: 16000,
    minChars: 2000,
    maxAllowedChars: 200000,
    lockedByEnvironment: false,
  });
});

test("saves with a PUT of max_chars", async () => {
  const { api, calls } = loadApi(() => json({ ...API_BODY, max_chars: 64000 }));
  const settings = await api.updateToolResultLimit(64000, "fallback");
  assert.equal(calls[0].init?.method, "PUT");
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), { max_chars: 64000 });
  assert.equal(settings.maxChars, 64000);
});

test("a refused save surfaces the server's reason", async () => {
  const { api } = loadApi(() =>
    json({ detail: "UNSLOTH_TOOL_RESULT_MAX_CHARS is set" }, 409),
  );
  await assert.rejects(
    api.updateToolResultLimit(64000, "fallback"),
    /UNSLOTH_TOOL_RESULT_MAX_CHARS is set/,
  );
});

test("choices keep the presets in range and add a value that is not one", () => {
  const { api } = loadApi(() => json(API_BODY));
  const base = {
    maxChars: 16000,
    defaultChars: 16000,
    minChars: 2000,
    maxAllowedChars: 200000,
    lockedByEnvironment: false,
  };
  assert.deepEqual(api.toolResultLimitChoices(base), [
    4000, 8000, 16000, 32000, 64000, 128000, 200000,
  ]);
  assert.deepEqual(
    api.toolResultLimitChoices({ ...base, maxChars: 50000, lockedByEnvironment: true }),
    [4000, 8000, 16000, 32000, 50000, 64000, 128000, 200000],
  );
});
