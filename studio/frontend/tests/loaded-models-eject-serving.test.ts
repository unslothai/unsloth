// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/loaded-models-eject-resolver.mjs", import.meta.url);

const stub = await import("./helpers/loaded-models-eject-stub.ts");
const auth = await import("./helpers/store-stubs/auth.ts");
const { ejectLoadedModel } = await import(
  "../src/features/loaded-models/loaded-models-api.ts"
);

const ACTIVE = "unsloth/Qwen3-4B-GGUF";
const KEPT = "/models/kept-b.gguf";

function servingRow() {
  return {
    id: `chat:${KEPT}`,
    kind: "text" as const,
    source: "chat" as const,
    name: "kept-b",
    detail: "Also serving",
    servingCheckpoint: KEPT,
  };
}

test("a kept model unloads by its checkpoint and leaves the active one", async () => {
  stub.resetStub({
    active: ACTIVE,
    loaded: [ACTIVE, "kept-b"],
    servingCheckpoints: [ACTIVE, KEPT],
  });
  assert.deepEqual(await ejectLoadedModel(servingRow()), { status: "ejected" });
  assert.deepEqual(
    stub.state.unloads.map((u) => u.model_path),
    [KEPT],
  );
  assert.equal(stub.state.active, ACTIVE);
  // The running-chat check and the unload must address the same slot.
  assert.deepEqual(
    stub.state.confirmCalls.map((c) => c.model),
    [KEPT],
  );
});

test("a kept model already gone unloads nothing", async () => {
  stub.resetStub({
    active: ACTIVE,
    loaded: [ACTIVE],
    servingCheckpoints: [ACTIVE],
  });
  assert.deepEqual(await ejectLoadedModel(servingRow()), {
    status: "alreadyFree",
  });
  assert.deepEqual(stub.state.unloads, []);
});

function embeddingRow() {
  return {
    id: "embedding:rag",
    kind: "embedding" as const,
    source: "embedding" as const,
    name: "org/E",
    detail: "Documents",
  };
}

function answerEmbeddingStatus() {
  auth.setAuthFetchHandler((url) => {
    assert.equal(url, "/api/settings/embedding-model");
    return Response.json({
      embedding_model: "org/E",
      loaded: stub.state.embeddingLoaded,
      backend_loaded: stub.state.embeddingLoaded,
    });
  });
}

test("the embedder ejects through its own unload and is confirmed gone", async (t) => {
  t.after(() => auth.setAuthFetchHandler(null));
  stub.resetStub({ embeddingLoaded: true });
  answerEmbeddingStatus();
  assert.deepEqual(await ejectLoadedModel(embeddingRow()), {
    status: "ejected",
  });
  assert.equal(stub.state.embeddingLoaded, false);
});

test("an embedder already released is reported as free", async (t) => {
  t.after(() => auth.setAuthFetchHandler(null));
  stub.resetStub({ embeddingLoaded: false });
  answerEmbeddingStatus();
  assert.deepEqual(await ejectLoadedModel(embeddingRow()), {
    status: "alreadyFree",
  });
});

test("an embedder swapped in since the poll is reported, not unloaded", async (t) => {
  t.after(() => auth.setAuthFetchHandler(null));
  stub.resetStub({ embeddingLoaded: true });
  answerEmbeddingStatus();
  const outcome = await ejectLoadedModel({ ...embeddingRow(), name: "org/Old" });
  assert.deepEqual(outcome, { status: "replaced", resident: "org/E" });
  assert.equal(stub.state.embeddingLoaded, true);
});
