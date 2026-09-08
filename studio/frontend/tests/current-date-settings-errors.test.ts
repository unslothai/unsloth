// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type CurrentDateApi = {
  loadCurrentDatePrompt: (fallbackMessage: string) => Promise<unknown>;
  updateCurrentDatePrompt: (
    enabled: boolean,
    fallbackMessage: string,
  ) => Promise<unknown>;
};

function loadApi(): CurrentDateApi {
  return loadWithStubs<CurrentDateApi>(
    new URL(
      "../src/features/settings/api/current-date-prompt.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": {
        authFetch: async () => new Response(null, { status: 500 }),
      },
      "@/lib/format-fastapi-error": {
        readFastApiError: async (_response: Response, fallback: string) =>
          fallback,
      },
    },
  );
}

test("the load helper uses its translated fallback", async () => {
  const api = loadApi();

  await assert.rejects(
    api.loadCurrentDatePrompt("Échec du chargement"),
    /Échec du chargement/,
  );
});

test("the save helper uses its translated fallback", async () => {
  const api = loadApi();

  await assert.rejects(
    api.updateCurrentDatePrompt(true, "Échec de l’enregistrement"),
    /Échec de l’enregistrement/,
  );
});
