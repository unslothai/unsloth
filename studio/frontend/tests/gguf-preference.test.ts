// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as PreferenceApi from "../src/features/model-picker/api/gguf-preference.ts";
import { readFastApiError } from "../src/lib/format-fastapi-error.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

/** Load the shipped API module with only its authenticated HTTP boundary replaced. */
function apiWithFetch(
  fetch: (url: string, init?: RequestInit) => Promise<Response>,
) {
  return loadWithStubs<typeof PreferenceApi>(
    new URL(
      "../src/features/model-picker/api/gguf-preference.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": { authFetch: fetch },
      "@/lib/format-fastapi-error": { readFastApiError },
    },
  );
}

test("reads a saved quant using an encoded model identity and forwards cancellation", async () => {
  const controller = new AbortController();
  const model = "C:/Models/Model & Quant";
  const api = apiWithFetch(async (url, init) => {
    assert.equal(
      new URL(url, "http://localhost").searchParams.get("model_id"),
      model,
    );
    assert.equal(init?.signal, controller.signal);
    return Response.json({ gguf_variant: "Q8_0" });
  });
  assert.equal(await api.readGgufPreference(model, controller.signal), "Q8_0");
});

for (const variant of ["Q8_0", null]) {
  test(`persists ${variant ?? "a cleared default"} and returns the server-confirmed value`, async () => {
    const api = apiWithFetch(async (url, init) => {
      assert.equal(url, "/api/settings/model-gguf-preference");
      assert.equal(init?.method, "PUT");
      assert.equal(
        (init?.headers as Record<string, string>)["Content-Type"],
        "application/json",
      );
      assert.deepEqual(JSON.parse(String(init?.body)), {
        model_id: "org/Model",
        gguf_variant: variant,
      });
      return Response.json({ gguf_variant: variant });
    });
    assert.equal(await api.writeGgufPreference("org/Model", variant), variant);
  });
}

for (const operation of ["read", "write"] as const) {
  test(`${operation} surfaces the backend failure rather than reporting a saved default`, async () => {
    const api = apiWithFetch(async () =>
      Response.json({ detail: "Settings unavailable" }, { status: 503 }),
    );
    await assert.rejects(
      operation === "read"
        ? api.readGgufPreference("org/Model")
        : api.writeGgufPreference("org/Model", "Q8_0"),
      /Settings unavailable/,
    );
  });
}

test("an absent saved preference remains null", async () => {
  const api = apiWithFetch(async () => Response.json({ gguf_variant: null }));
  assert.equal(await api.readGgufPreference("org/Model"), null);
});
