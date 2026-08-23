// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

register("./helpers/settings-api-resolver.mjs", import.meta.url);

type RecordedCall = {
  input: string;
  init?: RequestInit;
};

const calls: RecordedCall[] = [];
let respond = (): Response =>
  new Response(JSON.stringify({}), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

globalThis.fetch = ((
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> => {
  calls.push({ input: String(input), init });
  return Promise.resolve(respond());
}) as typeof fetch;

const {
  browseFolders,
  getCachedModelPath,
  listRecommendedFolders,
  revealCachedModel,
} = await import("../src/lib/model-filesystem-api.ts");

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("cached-model path and reveal requests preserve their wire contract", async () => {
  calls.length = 0;
  respond = () =>
    jsonResponse({
      path: "/cache/model.gguf",
      // biome-ignore lint/style/useNamingConvention: API response schema
      is_dir: false,
    });

  assert.deepEqual(
    await getCachedModelPath(
      "org/model",
      "Q4 K_M",
      "/previous/models--org--model",
    ),
    {
      path: "/cache/model.gguf",
      // biome-ignore lint/style/useNamingConvention: API response schema
      is_dir: false,
    },
  );
  assert.equal(
    calls[0]?.input,
    "/api/models/cached-model-path?repo_id=org%2Fmodel&variant=Q4+K_M&cache_path=%2Fprevious%2Fmodels--org--model",
  );
  assert.equal(calls[0]?.init, undefined);

  respond = () => jsonResponse({ ok: true });
  await revealCachedModel(
    "org/model",
    "Q4_K_M",
    "/previous/models--org--model",
  );
  assert.equal(calls[1]?.input, "/api/models/reveal-cached-model");
  assert.equal(calls[1]?.init?.method, "POST");
  assert.deepEqual(calls[1]?.init?.headers, {
    "Content-Type": "application/json",
  });
  assert.deepEqual(JSON.parse(String(calls[1]?.init?.body)), {
    // biome-ignore lint/style/useNamingConvention: API request schema
    repo_id: "org/model",
    variant: "Q4_K_M",
    // biome-ignore lint/style/useNamingConvention: API request schema
    cache_path: "/previous/models--org--model",
  });
});

test("folder requests preserve query, cancellation, and response shapes", async () => {
  calls.length = 0;
  const controller = new AbortController();
  respond = () =>
    jsonResponse({
      current: "/models",
      parent: "/",
      entries: [],
      suggestions: ["/models"],
    });

  const result = await browseFolders("/models/a b", true, controller.signal);
  assert.equal(result.current, "/models");
  assert.equal(
    calls[0]?.input,
    "/api/models/browse-folders?path=%2Fmodels%2Fa+b&show_hidden=true",
  );
  assert.equal(calls[0]?.init?.signal, controller.signal);

  respond = () => jsonResponse({ folders: ["/models", "/weights"] });
  assert.deepEqual(await listRecommendedFolders(), ["/models", "/weights"]);
  assert.equal(calls[1]?.input, "/api/models/recommended-folders");
});

test("filesystem requests retain Chat API error formatting", async () => {
  respond = () =>
    jsonResponse(
      {
        detail: [
          {
            loc: ["body", "repo_id"],
            msg: "Invalid repository",
            type: "value_error",
          },
        ],
      },
      422,
    );
  await assert.rejects(getCachedModelPath("bad"), {
    message: "repo_id: Invalid repository",
  });

  respond = () => new Response("<html>bad gateway</html>", { status: 503 });
  await assert.rejects(listRecommendedFolders(), {
    message: "Request failed (503)",
  });
});

test("Chat and Hub no longer duplicate filesystem API implementations", () => {
  const chatApi = readFileSync(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    "utf8",
  );
  const hubInventoryApi = readFileSync(
    new URL("../src/features/hub/inventory/api.ts", import.meta.url),
    "utf8",
  );
  const chatIndex = readFileSync(
    new URL("../src/features/chat/index.ts", import.meta.url),
    "utf8",
  );

  for (const source of [chatApi, hubInventoryApi]) {
    assert.ok(!source.includes("interface CachedModelPath"));
    assert.ok(!source.includes("interface BrowseFoldersResponse"));
    assert.ok(!source.includes("function getCachedModelPath"));
    assert.ok(!source.includes("function revealCachedModel"));
    assert.ok(!source.includes("function browseFolders"));
    assert.ok(!source.includes("function listRecommendedFolders"));
  }
  // Nor does the Chat barrel re-export them: every consumer imports the owning module
  // directly, so @/features/chat stops looking like the home of a filesystem API.
  assert.ok(!chatIndex.includes("@/lib/model-filesystem-api"));
});
