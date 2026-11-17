// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
const { ModelAliasConflictError, loadModelAliases, saveModelAliases } =
  await import("../src/features/settings/api/model-aliases.ts");

test("a renamed alias keeps every field it was read with, and the save names the etag", async () => {
  // Settings this page cannot show, including ones from a newer release, travel back untouched.
  const config = {
    max_seq_length: 98304,
    kv_cache_dtype: "q5_1",
    speculative_type: "ngram",
    rope_scale: 2,
  };
  const stored = [
    { name: "long", model: "org/model", load_config: config, note: "kept" },
  ];
  let written: unknown;
  globalThis.fetch = (async (_url, init) => {
    if (init?.method === "PUT") {
      written = JSON.parse(String(init.body));
      return Response.json({
        aliases: (written as { aliases: unknown }).aliases,
        etag: "v2",
      });
    }
    return Response.json({ aliases: stored, etag: "v1" });
  }) as typeof fetch;
  const loaded = await loadModelAliases();
  assert.equal(loaded.etag, "v1");
  loaded.aliases[0].name = "research";
  const saved = await saveModelAliases(loaded.aliases, loaded.etag);
  const expected = [{ ...stored[0], name: "research" }];
  assert.deepEqual(written, { aliases: expected, etag: "v1" });
  assert.deepEqual(saved, { aliases: expected, etag: "v2" });
});

test("ordinary aliases keep their original wire representation", async () => {
  globalThis.fetch = (async () =>
    Response.json({
      aliases: [{ name: "plain", model: "org/model" }],
      etag: "v1",
    })) as typeof fetch;
  assert.deepEqual(await loadModelAliases(), {
    aliases: [{ name: "plain", model: "org/model" }],
    etag: "v1",
  });
});

test("a refused stale save surfaces as a conflict with the server's message", async () => {
  globalThis.fetch = (async () =>
    Response.json(
      {
        detail:
          "The model aliases changed since this page loaded. Reload the page and try again.",
      },
      { status: 409 },
    )) as typeof fetch;
  await assert.rejects(
    saveModelAliases([{ name: "x", model: "org/model" }], "v0"),
    (error: unknown) =>
      error instanceof ModelAliasConflictError &&
      error.message.includes("Reload the page"),
  );
});
