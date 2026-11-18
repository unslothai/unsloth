// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Feature = "hardware" | "text" | "datasets" | "media" | "quantization";
type Settings = {
  features: Feature[];
  effectiveFeatures: Feature[];
  disabledByEnv: boolean;
};

function settingsApi(fetch: (url: string, init?: RequestInit) => Promise<Response>) {
  return loadWithStubs<{
    loadMlWarmupSettings(): Promise<Settings>;
    updateMlWarmupSettings(features: readonly Feature[]): Promise<Settings>;
    updateWarmupSelection(selected: readonly Feature[], feature: Feature, enabled: boolean): Feature[];
    hardwareWarmupRequired(features: readonly Feature[]): boolean;
  }>(
    new URL("../src/features/settings/api/ml-warmup.ts", import.meta.url),
    {
      "@/features/auth": { authFetch: fetch },
      "@/lib/format-fastapi-error": {
        readFastApiError: async (response: Response) => (await response.json()).detail,
      },
    },
  );
}

test("reads fresh requested and effective preferences independently", async () => {
  let features: Feature[] = [];
  const api = settingsApi(async (url) => {
    assert.equal(url, "/api/settings/ml-warmup");
    return Response.json({
      features,
      effective_features: features.length ? ["hardware", "text"] : [],
      disabled_by_env: true,
    });
  });
  assert.deepEqual(await api.loadMlWarmupSettings(), {
    features: [],
    effectiveFeatures: [],
    disabledByEnv: true,
  });
  features = ["text"];
  assert.deepEqual(await api.loadMlWarmupSettings(), {
    features: ["text"],
    effectiveFeatures: ["hardware", "text"],
    disabledByEnv: true,
  });
});

test("saves only the user's selections and returns expanded server features", async () => {
  const api = settingsApi(async (url, init) => {
    assert.equal(url, "/api/settings/ml-warmup");
    assert.equal(init?.method, "PUT");
    assert.deepEqual(JSON.parse(init?.body as string), { features: ["text"] });
    return Response.json({
      features: ["text"],
      effective_features: ["hardware", "text"],
      disabled_by_env: false,
    });
  });
  assert.deepEqual(await api.updateMlWarmupSettings(["text"]), {
    features: ["text"],
    effectiveFeatures: ["hardware", "text"],
    disabledByEnv: false,
  });
});

test("removing text restores hardware's explicit saved preference", () => {
  const api = settingsApi(async () => Response.json({}));
  assert.deepEqual(api.updateWarmupSelection(["text"], "text", false), []);
  assert.deepEqual(api.updateWarmupSelection(["hardware", "text"], "text", false), ["hardware"]);
  assert.deepEqual(api.updateWarmupSelection(["media"], "text", true), ["text", "media"]);
  assert.deepEqual(api.updateWarmupSelection(["text"], "text", true), ["text"]);
});

test("only text, media and quantization lock the hardware switch", () => {
  const api = settingsApi(async () => Response.json({}));
  for (const feature of ["text", "media", "quantization"] as const) {
    assert.equal(api.hardwareWarmupRequired([feature]), true);
  }
  assert.equal(api.hardwareWarmupRequired([]), false);
  assert.equal(api.hardwareWarmupRequired(["datasets"]), false);
  assert.equal(api.hardwareWarmupRequired(["hardware", "datasets"]), false);
});

test("failed reads and writes report the server error", async () => {
  const api = settingsApi(async () =>
    Response.json({ detail: "Only the owner can change this setting." }, { status: 403 }),
  );
  await assert.rejects(api.loadMlWarmupSettings(), /Only the owner/);
  await assert.rejects(api.updateMlWarmupSettings(["text"]), /Only the owner/);
});
