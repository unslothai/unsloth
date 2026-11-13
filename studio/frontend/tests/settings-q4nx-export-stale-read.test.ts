// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as zustand from "zustand";

import { loadWithStubs } from "./helpers/module-stubs.ts";

const pending: ((r: Response) => void)[] = [];
const api = loadWithStubs<
  typeof import("../src/features/settings/api/q4nx-export.ts")
>(new URL("../src/features/settings/api/q4nx-export.ts", import.meta.url), {
  "@/features/auth": {
    authFetch: () => new Promise<Response>((resolve) => pending.push(resolve)),
  },
  "@/lib/format-fastapi-error": { readFastApiError: async () => "x" },
  zustand,
});

const reply = (enabled: boolean) =>
  new Response(JSON.stringify({ enabled, npu_supported: false }));

test("a read that started before a save does not overwrite the saved setting", async () => {
  const read = api.loadQ4nxExportSettings();
  const save = api.updateQ4nxExportSettings(true);
  const [getResolve, putResolve] = pending.splice(0);
  putResolve(reply(true));
  await save;
  getResolve(reply(false));
  await read;
  assert.equal(api.useQ4nxExportStore.getState().settings?.enabled, true);
});
