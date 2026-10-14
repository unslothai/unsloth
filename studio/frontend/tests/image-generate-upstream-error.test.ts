// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test, { afterEach } from "node:test";

register("./helpers/settings-api-resolver.mjs", import.meta.url);

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

const { generateDiffusionImage, GenerateResponseLostError } = await import(
  "../src/features/images/api.ts"
);

function respond(status: number, body: string, contentType: string) {
  globalThis.fetch = (async () =>
    new Response(body, { status, headers: { "content-type": contentType } })) as typeof fetch;
}

test("a JSON 502 from the app shows the upstream reason instead of waiting for a lost run", async () => {
  respond(
    502,
    JSON.stringify({ detail: "Image endpoint returned HTTP 401: invalid api key" }),
    "application/json",
  );
  await assert.rejects(generateDiffusionImage({ prompt: "a sloth" }), (err: unknown) => {
    assert.ok(!(err instanceof GenerateResponseLostError));
    assert.match((err as Error).message, /invalid api key/);
    return true;
  });
});

test("a proxy's HTML 502 is still a lost response", async () => {
  respond(502, "<html>Bad gateway</html>", "text/html");
  await assert.rejects(
    generateDiffusionImage({ prompt: "a sloth" }),
    (err: unknown) => err instanceof GenerateResponseLostError,
  );
});
