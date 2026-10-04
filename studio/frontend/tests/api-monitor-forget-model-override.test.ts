// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { store } = installLocalStorageFake();

const {
  FORGET_MODEL_OVERRIDE_FAILED,
  FORGET_MODEL_OVERRIDE_LOCAL_FAILED,
  forgetModelOverride,
} = await import("../src/features/api-monitor/forget-model-override.ts");
const { putModelOverride, syncModelOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const {
  DEFAULT_PER_MODEL_CONFIG,
  PER_MODEL_CONFIG_STORAGE_KEY,
  deletePerModelConfigsForOverrideKeys,
  listPerModelConfigs,
  resolveInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { modelStorageKey } = await import(
  "../src/features/model-picker/model-config/model-identity.ts"
);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

type Trace = {
  deps: Parameters<typeof forgetModelOverride>[1];
  removedRemote: [string, string | null][];
  removedLocal: (readonly string[])[];
  errors: string[];
  reloads: number;
  sequence: string[];
};

type Answer = {
  overrides: Readonly<Record<string, unknown>>;
  removedKeys: readonly string[];
};

function answer(removedKeys: readonly string[] = []): Answer {
  return { overrides: {}, removedKeys };
}

function trace(
  remote: () => Promise<Answer> = () => Promise.resolve(answer()),
  local = true,
  listedKeys: readonly string[] = [],
): Trace {
  const state: Trace = {
    removedRemote: [],
    removedLocal: [],
    errors: [],
    reloads: 0,
    sequence: [],
    deps: {
      listedKeys,
      removeRemote: (modelId, ggufVariant) => {
        state.sequence.push("remote");
        state.removedRemote.push([modelId, ggufVariant]);
        return remote();
      },
      removeLocal: (overrideKeys) => {
        state.sequence.push("local");
        state.removedLocal.push(overrideKeys);
        return local;
      },
      reload: () => {
        state.sequence.push("reload");
        state.reloads += 1;
        return Promise.resolve();
      },
      onError: (message) => {
        state.errors.push(message);
      },
    },
  };
  return state;
}

const PATH_ID = "/home/santiago/Temp-GGUF/qwen38/UD-IQ3_XXS";
const PATH_KEY = `${PATH_ID}:UD-IQ3_XXS`;

test("a forget clears the server entry, then every record it reports, then refetches", async () => {
  // A listed key that left the map for another reason is not read as forgotten once it reports.
  const state = trace(
    () => Promise.resolve(answer([PATH_KEY, PATH_ID])),
    true,
    [PATH_KEY, "unsloth/Other-GGUF:Q8_0"],
  );

  await forgetModelOverride(PATH_KEY, state.deps);

  assert.deepEqual(state.removedRemote, [[PATH_ID, "UD-IQ3_XXS"]]);
  assert.deepEqual(state.removedLocal, [[PATH_KEY, PATH_ID]]);
  assert.deepEqual(state.sequence, ["remote", "local", "reload"]);
  assert.deepEqual(state.errors, []);
});

test("a server that reports nothing still forgets the clicked key here", async () => {
  const state = trace();

  await forgetModelOverride(PATH_KEY, state.deps);

  assert.deepEqual(state.removedLocal, [[PATH_KEY]]);
});

// Without this an old backend leaves the snapshot record, and the next load applies it.
test("a server that reports nothing is read off the map it returns", async () => {
  const repoKey = "unsloth/Repo-GGUF:Q4_K_M";
  const snapshotKey =
    "/home/u/.cache/hub/models--unsloth--Repo-GGUF/snapshots/2f1c9ab:Q4_K_M";
  const state = trace(
    () =>
      Promise.resolve({
        overrides: { "unsloth/Other-GGUF:Q8_0": {} },
        removedKeys: [],
      }),
    true,
    [repoKey, snapshotKey, "unsloth/Other-GGUF:Q8_0"],
  );

  await forgetModelOverride(repoKey, state.deps);

  assert.deepEqual(state.removedLocal, [[repoKey, snapshotKey]]);
});

// Another client re-saved the row under the server's casing mid-forget; it is still held.
test("a listed key the map still holds under another spelling is not read as cleared", async () => {
  const state = trace(
    () =>
      Promise.resolve({
        overrides: { "unsloth/b-gguf:q8_0": {} },
        removedKeys: [],
      }),
    true,
    [PATH_KEY, "Unsloth/B-GGUF:Q8_0"],
  );

  await forgetModelOverride(PATH_KEY, state.deps);

  assert.deepEqual(state.removedLocal, [[PATH_KEY]]);
});

test("a key with no quant suffix forgets the whole id", async () => {
  const state = trace();

  await forgetModelOverride("unsloth/Qwen3-4B-GGUF", state.deps);

  assert.deepEqual(state.removedRemote, [["unsloth/Qwen3-4B-GGUF", null]]);
});

test("a refused remove leaves this browser's copy and the list alone", async () => {
  const state = trace(() =>
    Promise.reject(new Error("Settings are read-only")),
  );

  await assert.doesNotReject(() => forgetModelOverride(PATH_KEY, state.deps));

  assert.deepEqual(state.errors, ["Settings are read-only"]);
  assert.deepEqual(state.removedLocal, []);
  assert.equal(state.reloads, 0);
});

test("a rejection that is not an Error still reports", async () => {
  const state = trace(() => Promise.reject("offline"));

  await forgetModelOverride(PATH_KEY, state.deps);

  assert.deepEqual(state.errors, [FORGET_MODEL_OVERRIDE_FAILED]);
  assert.deepEqual(state.removedLocal, []);
});

test("a browser copy that could not be deleted is reported, not swallowed", async () => {
  const state = trace(() => Promise.resolve(answer([PATH_KEY])), false);

  await forgetModelOverride(PATH_KEY, state.deps);

  assert.deepEqual(state.errors, [FORGET_MODEL_OVERRIDE_LOCAL_FAILED]);
  // The server entry is gone whatever the browser did, so the list still refetches.
  assert.equal(state.reloads, 1);
});

// A quant can name a directory or a whole filename stem (is_qualified_gguf_variant_key in
// hub/utils/gguf.py); the key still joins on one colon.
test("a path-qualified variant splits off the repo it belongs to", async () => {
  const state = trace();

  await forgetModelOverride(
    "unsloth/Repo-GGUF:distilled/model-Q6_K",
    state.deps,
  );

  assert.deepEqual(state.removedRemote, [
    ["unsloth/Repo-GGUF", "distilled/model-Q6_K"],
  ]);
});

test("a filename-stem variant splits off the repo too", async () => {
  const state = trace();

  await forgetModelOverride(
    "unsloth/H3-GGUF:minimax_h3_ref2va_pruned-Q6_K",
    state.deps,
  );

  assert.deepEqual(state.removedRemote, [
    ["unsloth/H3-GGUF", "minimax_h3_ref2va_pruned-Q6_K"],
  ]);
});

test("a colon inside a local path is part of the name, not a separator", async () => {
  const state = trace();

  await forgetModelOverride("/home/u/models/foo:bar/baz.gguf", state.deps);

  assert.deepEqual(state.removedRemote, [
    ["/home/u/models/foo:bar/baz.gguf", null],
  ]);
});

const REPO_ID = "unsloth/Repo-GGUF";
const SNAPSHOT_PATH =
  "/home/u/.cache/huggingface/hub/models--unsloth--Repo-GGUF/snapshots/2f1c9ab";
const OTHER_PATH = "/home/u/models/Repo-Q4_K_M.gguf";
const QUANT = "Q4_K_M";
const SAVED = config(32768);

function config(maxSeqLength: number) {
  return { ...DEFAULT_PER_MODEL_CONFIG, maxSeqLength };
}

function respond(status: number, body: unknown): void {
  setAuthFetchHandler(
    () =>
      new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json" },
      }),
  );
}

// The local half of a forget: the records this browser drops for the keys a server reports.

test("every reported key's record goes, whichever spelling it uses", () => {
  store.clear();
  savePerModelConfig(REPO_ID, QUANT, config(32768));
  savePerModelConfig(SNAPSHOT_PATH, QUANT, config(4096));
  savePerModelConfig(REPO_ID, null, config(2048));

  assert.equal(
    deletePerModelConfigsForOverrideKeys([
      `${REPO_ID}:${QUANT}`,
      REPO_ID,
      `${SNAPSHOT_PATH}:${QUANT}`,
    ]),
    true,
  );

  assert.equal(listPerModelConfigs().length, 0);
});

test("records the server did not report stay", () => {
  store.clear();
  savePerModelConfig(REPO_ID, QUANT, config(32768));
  savePerModelConfig(REPO_ID, "Q8_0", config(16384));
  savePerModelConfig(OTHER_PATH, QUANT, config(4096));

  assert.equal(
    deletePerModelConfigsForOverrideKeys([`${REPO_ID}:${QUANT}`]),
    true,
  );

  assert.equal(resolveInitialConfig(REPO_ID, QUANT).remembered, false);
  assert.equal(resolveInitialConfig(REPO_ID, "Q8_0").remembered, true);
  assert.equal(resolveInitialConfig(OTHER_PATH, QUANT).remembered, true);
});

test("the server row's casing still finds the lowercased record", () => {
  store.clear();
  savePerModelConfig(REPO_ID, QUANT, config(32768));

  assert.equal(
    deletePerModelConfigsForOverrideKeys([`${REPO_ID}:q4_k_m`]),
    true,
  );

  assert.equal(resolveInitialConfig(REPO_ID, QUANT).remembered, false);
});

// A colon is legal in a POSIX filename, so the key cannot be parsed apart.
test("a directory-qualified variant under a path resolves to its stored record", () => {
  store.clear();
  savePerModelConfig(
    "/home/u/models/repo",
    "distilled/model-Q6_K",
    config(32768),
  );

  assert.equal(
    deletePerModelConfigsForOverrideKeys([
      "/home/u/models/repo:distilled/model-Q6_K",
    ]),
    true,
  );

  assert.equal(
    resolveInitialConfig("/home/u/models/repo", "distilled/model-Q6_K")
      .remembered,
    false,
  );
});

test("two records that spell the same key both go", () => {
  store.clear();
  savePerModelConfig(
    "/home/u/models/repo",
    "distilled/model-Q6_K",
    config(32768),
  );
  savePerModelConfig(
    "/home/u/models/repo:distilled/model-Q6_K",
    null,
    config(4096),
  );

  assert.equal(
    deletePerModelConfigsForOverrideKeys([
      "/home/u/models/repo:distilled/model-Q6_K",
    ]),
    true,
  );

  assert.equal(listPerModelConfigs().length, 0);
});

test("a bare key deletes the record with no variant and not the quants", () => {
  store.clear();
  savePerModelConfig(REPO_ID, null, config(2048));
  savePerModelConfig(REPO_ID, QUANT, config(32768));

  assert.equal(deletePerModelConfigsForOverrideKeys([REPO_ID]), true);

  assert.equal(resolveInitialConfig(REPO_ID, null).remembered, false);
  assert.equal(resolveInitialConfig(REPO_ID, QUANT).remembered, true);
});

test("a key this browser has no record for is nothing to delete", () => {
  store.clear();
  savePerModelConfig(REPO_ID, QUANT, config(32768));

  assert.equal(
    deletePerModelConfigsForOverrideKeys(["unsloth/Other-GGUF:Q8_0"]),
    true,
  );

  assert.equal(resolveInitialConfig(REPO_ID, QUANT).remembered, true);
});

test("a record a newer build wrote is left behind and reported", () => {
  store.clear();
  const key = modelStorageKey(REPO_ID, QUANT);
  store.set(
    PER_MODEL_CONFIG_STORAGE_KEY,
    JSON.stringify({ [key]: { version: 99, maxSeqLength: 32768 } }),
  );

  assert.equal(
    deletePerModelConfigsForOverrideKeys([`${REPO_ID}:${QUANT}`]),
    false,
  );

  assert.ok(
    JSON.parse(store.get(PER_MODEL_CONFIG_STORAGE_KEY) ?? "{}")[key],
    "the newer record must survive an older build's forget",
  );
});

// The settings page's forget, mirrored through syncModelOverride.

/** syncModelOverride is fire-and-forget; its cleanup lands after the response is read. */
function settled(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function quietly(run: () => Promise<void>): Promise<void> {
  const warn = console.warn;
  console.warn = () => undefined;
  try {
    await run();
  } finally {
    console.warn = warn;
  }
}

test("a forget resolves to the keys the server cleared", async () => {
  respond(200, {
    overrides: {},
    // biome-ignore lint/style/useNamingConvention: API schema
    removed_keys: [`${REPO_ID}:${QUANT}`, REPO_ID],
  });

  assert.deepEqual(await putModelOverride(REPO_ID, QUANT, null), {
    overrides: {},
    removedKeys: [`${REPO_ID}:${QUANT}`, REPO_ID],
  });
});

test("a server that predates the field resolves to none, with the map it returned", async () => {
  respond(200, { overrides: { "unsloth/Other-GGUF:Q8_0": {} } });

  assert.deepEqual(await putModelOverride(REPO_ID, QUANT, null), {
    overrides: { "unsloth/Other-GGUF:Q8_0": {} },
    removedKeys: [],
  });
});

test("a mirrored forget drops the records the server reports", async () => {
  store.clear();
  savePerModelConfig(REPO_ID, QUANT, SAVED);
  savePerModelConfig(SNAPSHOT_PATH, QUANT, SAVED);
  savePerModelConfig(REPO_ID, null, SAVED);
  respond(200, {
    overrides: {},
    // biome-ignore lint/style/useNamingConvention: API schema
    removed_keys: [`${REPO_ID}:${QUANT}`, REPO_ID, `${SNAPSHOT_PATH}:${QUANT}`],
  });

  syncModelOverride(REPO_ID, QUANT, null);
  await settled();

  assert.equal(listPerModelConfigs().length, 0);
});

// The settings page deletes its own record before the mirror answers; cleanup carries on past it.
test("a mirrored forget from the settings page still drops the other spellings", async () => {
  store.clear();
  savePerModelConfig(SNAPSHOT_PATH, QUANT, SAVED);
  savePerModelConfig(REPO_ID, null, SAVED);
  respond(200, {
    overrides: {},
    // biome-ignore lint/style/useNamingConvention: API schema
    removed_keys: [`${REPO_ID}:${QUANT}`, REPO_ID, `${SNAPSHOT_PATH}:${QUANT}`],
  });

  syncModelOverride(REPO_ID, QUANT, null);
  await settled();

  assert.equal(listPerModelConfigs().length, 0);
});

test("a mirrored save leaves this browser's records alone", async () => {
  store.clear();
  savePerModelConfig(REPO_ID, QUANT, SAVED);
  respond(200, { overrides: {} });

  syncModelOverride(REPO_ID, QUANT, SAVED);
  await settled();

  assert.equal(resolveInitialConfig(REPO_ID, QUANT).remembered, true);
});

test("a forget the server refused keeps this browser's records", async () => {
  store.clear();
  savePerModelConfig(REPO_ID, QUANT, SAVED);
  respond(403, { detail: "Settings are read-only" });

  await quietly(async () => {
    syncModelOverride(REPO_ID, QUANT, null);
    await settled();
  });

  assert.equal(resolveInitialConfig(REPO_ID, QUANT).remembered, true);
});
