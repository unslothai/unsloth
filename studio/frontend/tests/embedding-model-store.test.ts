// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

// The api module reaches authFetch through the auth barrel, which re-exports
// login-page.tsx, and the hub barrel, which reads import.meta.env.
register("./helpers/vite-env-loader.mjs", import.meta.url);
register("./helpers/settings-api-resolver.mjs", import.meta.url);
register("./helpers/hub-stub-resolver.mjs", import.meta.url);

const { useEmbeddingModelStore } = await import(
  "../src/features/settings/stores/embedding-model-store.ts"
);

type Settings = {
  embeddingModel: string;
  embeddingGgufRepo: string;
  defaultEmbeddingModel: string;
  defaultEmbeddingGgufRepo: string;
  isCustom: boolean;
};

function settings(model: string): Settings {
  return {
    embeddingModel: model,
    embeddingGgufRepo: "",
    defaultEmbeddingModel: "unsloth/bge-small-en-v1.5",
    defaultEmbeddingGgufRepo: "",
    isCustom: model !== "unsloth/bge-small-en-v1.5",
  };
}

/** The GET, answering with `model` after `release` resolves. */
function respondWith(model: string, release?: Promise<void>): void {
  globalThis.fetch = (async () => {
    if (release) await release;
    return {
      ok: true,
      status: 200,
      json: async () => ({
        // biome-ignore lint/style/useNamingConvention: API schema
        embedding_model: model,
        // biome-ignore lint/style/useNamingConvention: API schema
        embedding_gguf_repo: "",
        // biome-ignore lint/style/useNamingConvention: API schema
        default_embedding_model: "unsloth/bge-small-en-v1.5",
        // biome-ignore lint/style/useNamingConvention: API schema
        default_embedding_gguf_repo: "",
        // biome-ignore lint/style/useNamingConvention: API schema
        is_custom: model !== "unsloth/bge-small-en-v1.5",
      }),
    } as unknown as Response;
  }) as typeof fetch;
}

function reset(): void {
  useEmbeddingModelStore.setState({
    settings: null,
    loadError: null,
    revision: 0,
  });
}

test("a mount reads the setting", async () => {
  reset();
  respondWith("unsloth/bge-m3");
  await useEmbeddingModelStore.getState().load();
  assert.equal(
    useEmbeddingModelStore.getState().settings?.embeddingModel,
    "unsloth/bge-m3",
  );
});

test("a save that lands mid-read is not undone by it", async () => {
  reset();
  // The other tab's read is in flight, and answers with the OLD model.
  let release = (): void => undefined;
  const gate = new Promise<void>((resolve) => {
    release = () => resolve();
  });
  respondWith("unsloth/bge-small-en-v1.5", gate);
  const reading = useEmbeddingModelStore.getState().load();

  // The save from the tab the user just left commits first.
  useEmbeddingModelStore.getState().applySettings(settings("unsloth/bge-m3"));
  release();
  await reading;

  assert.equal(
    useEmbeddingModelStore.getState().settings?.embeddingModel,
    "unsloth/bge-m3",
    "the saved model stands, not the value the read started before it",
  );
});

test("a read that finishes first is still replaced by the save", async () => {
  reset();
  respondWith("unsloth/bge-small-en-v1.5");
  await useEmbeddingModelStore.getState().load();
  useEmbeddingModelStore.getState().applySettings(settings("unsloth/bge-m3"));
  assert.equal(
    useEmbeddingModelStore.getState().settings?.embeddingModel,
    "unsloth/bge-m3",
  );
});

test("a slow read cannot report over the one that overtook it", async () => {
  reset();
  // General mounts and its read hangs; Data mounts behind it and answers.
  let release = (): void => undefined;
  const gate = new Promise<void>((resolve) => {
    release = () => resolve();
  });
  globalThis.fetch = (async () => {
    await gate;
    throw new Error("network unreachable");
  }) as typeof fetch;
  const first = useEmbeddingModelStore.getState().load();

  respondWith("unsloth/bge-m3");
  await useEmbeddingModelStore.getState().load();
  release();
  await first;

  assert.equal(
    useEmbeddingModelStore.getState().loadError,
    null,
    "the newer read succeeded, so no error is raised over it",
  );
  assert.equal(
    useEmbeddingModelStore.getState().settings?.embeddingModel,
    "unsloth/bge-m3",
  );
});

test("a failed read reports the backend's reason", async () => {
  reset();
  globalThis.fetch = (async () =>
    ({
      ok: false,
      status: 500,
      json: async () => ({ detail: "storage is offline" }),
    }) as unknown as Response) as typeof fetch;
  await useEmbeddingModelStore.getState().load();
  assert.equal(
    useEmbeddingModelStore.getState().loadError,
    "storage is offline",
  );
});

test("an older save cannot land on top of a newer one", async () => {
  reset();
  const store = useEmbeddingModelStore.getState();
  // General submits, the user switches to Data, and Data submits its own: the
  // second mount carries its own pending flag, so nothing stopped it.
  const first = store.beginSave();
  const second = store.beginSave();

  assert.ok(store.applySettings(settings("unsloth/bge-m3"), second));
  assert.equal(
    store.applySettings(settings("unsloth/bge-small-en-v1.5"), first),
    false,
    "the superseded save reports that it did not stand",
  );
  assert.equal(
    useEmbeddingModelStore.getState().settings?.embeddingModel,
    "unsloth/bge-m3",
  );
});

test("a save still bumps the revision the reads check", async () => {
  reset();
  const store = useEmbeddingModelStore.getState();
  store.applySettings(settings("unsloth/bge-m3"), store.beginSave());
  assert.equal(useEmbeddingModelStore.getState().revision, 1);
  // A superseded save is not a mutation, so it must not move the revision.
  const stale = store.beginSave();
  store.beginSave();
  store.applySettings(settings("unsloth/bge-small-en-v1.5"), stale);
  assert.equal(useEmbeddingModelStore.getState().revision, 1);
});
