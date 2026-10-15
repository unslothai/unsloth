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

const { cancelDiffusionGeneration, generateDiffusionImage, GenerateResponseLostError } = await import(
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

test("Stop on a connection run targets the connection, a local Stop does not", async () => {
  const urls: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    urls.push(String(input));
    return new Response(JSON.stringify({ cancelled: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;
  await cancelDiffusionGeneration(undefined, "connection");
  await cancelDiffusionGeneration();
  assert.deepEqual(urls, [
    "/api/inference/images/generate/cancel?target=connection",
    "/api/inference/images/generate/cancel",
  ]);
});

test("a connected pick retires a staged local pick and skips the hidden seed", async () => {
  const { readSrc } = await import("./helpers/kit.ts");
  const page = readSrc("features/images/images-page.tsx");
  const branch = page.slice(page.indexOf('if (meta.source === "external") {'));
  const body = branch.slice(0, branch.indexOf("setExternalModelId(id);"));
  assert.match(body, /pickGuard\.cancel\(\);/);
  assert.match(body, /beginPick\(\);/);
  assert.match(page, /if \(seed\.trim\(\) && !external\) \{/);
});
