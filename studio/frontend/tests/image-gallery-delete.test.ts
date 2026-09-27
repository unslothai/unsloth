// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";

test("gallery delete accepts an externally removed image but still reports real failures", async () => {
  const source = readSrc("features/images/api.ts");
  const start = source.indexOf("export async function deleteGalleryImage(");
  const end = source.indexOf("export async function clearGallery(", start);
  assert.ok(start >= 0 && end > start);
  const code = ts.transpileModule(source.slice(start, end), {
    compilerOptions: { module: ts.ModuleKind.CommonJS },
  }).outputText;

  for (const status of [200, 404, 401, 403, 500]) {
    const context = {
      exports: {} as { deleteGalleryImage: (id: string) => Promise<void> },
      authFetch: async (url: string, init: RequestInit) => {
        assert.equal(url, "/api/inference/images/gallery/missing-png");
        assert.equal(init.method, "DELETE");
        return new Response(null, { status });
      },
      readFastApiError: async () => `HTTP ${status}`,
    };
    runInNewContext(code, context);
    const deletion = context.exports.deleteGalleryImage("missing-png");
    if (status === 200 || status === 404) {
      await assert.doesNotReject(deletion);
    } else {
      await assert.rejects(deletion, new RegExp(`HTTP ${status}`));
    }
  }
});
