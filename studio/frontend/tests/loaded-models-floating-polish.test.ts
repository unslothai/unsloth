// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import test from "node:test";

import {
  type LoadedModelEntry,
  describeDiffusionStatus,
  describeEmbeddingStatus,
  describeInferenceStatus,
  describeVideoStatus,
  formatResidentMib,
  heldByAnotherAccount,
  loadedModelLabel,
  loadedModelTarget,
  loadedModelsCardState,
  sameLoadedModels,
} from "../src/features/loaded-models/loaded-models-sources.ts";
import { sentinelAfterSettle } from "../src/features/loaded-models/use-drag-position.ts";

type Inference = Parameters<typeof describeInferenceStatus>[0];

function inference(overrides: Record<string, unknown>): Inference {
  return {
    active_model: null,
    is_vision: false,
    loading: [],
    loaded: [],
    ...overrides,
  } as Inference;
}

const SNAPSHOT =
  "/home/u/.cache/huggingface/hub/models--org--Flux/snapshots/0a1b2c3d4e5f";

test("an image row shows the Hub id but ejects by the snapshot it loaded", () => {
  const [row] = describeDiffusionStatus({
    loaded: true,
    repo_id: SNAPSHOT,
    display_repo_id: "org/Flux",
  } as never);
  assert.equal(row.name, SNAPSHOT);
  assert.equal(loadedModelLabel(row), "org/Flux");
});

test("a video row does the same, and falls back to the repo id without one", () => {
  const [labelled] = describeVideoStatus({
    loaded: true,
    repo_id: SNAPSHOT,
    display_repo_id: "org/Wan",
  } as never);
  assert.equal(loadedModelLabel(labelled), "org/Wan");
  const [plain] = describeVideoStatus({
    loaded: true,
    repo_id: "org/Wan",
  } as never);
  assert.equal(plain.label, undefined);
  assert.equal(loadedModelLabel(plain), "org/Wan");
});

test("models kept alongside the active one are serving, not idle", () => {
  const rows = describeInferenceStatus(
    inference({
      active_model: "org/A-GGUF",
      model_identifier: "org/A-GGUF",
      is_gguf: true,
      loaded: ["org/A-GGUF", "kept-b", "org/Cached"],
      serving: ["org/A-GGUF", "kept-b"],
      serving_checkpoints: ["org/A-GGUF", "/models/kept-b.gguf"],
    }),
  );
  assert.deepEqual(
    rows.map((row) => [row.id, Boolean(row.inactive), row.servingCheckpoint]),
    [
      ["chat:org/A-GGUF", false, undefined],
      ["chat:/models/kept-b.gguf", false, "/models/kept-b.gguf"],
      ["chat:org/Cached", true, undefined],
    ],
  );
  assert.equal(rows[1].name, "kept-b");
});

test("two kept models sharing a label stay two rows", () => {
  const rows = describeInferenceStatus(
    inference({
      active_model: "org/A",
      loaded: ["org/A", "model"],
      serving: ["org/A", "model", "model"],
      serving_checkpoints: ["org/A", "/x/model.gguf", "/y/model.gguf"],
    }),
  );
  assert.equal(rows.filter((row) => row.servingCheckpoint).length, 2);
});

test("an older backend without serving keeps the cached rows as before", () => {
  const rows = describeInferenceStatus(
    inference({ active_model: "org/A", loaded: ["org/A", "org/B"] }),
  );
  assert.equal(rows[1].inactive, true);
});

test("resident sizes ride along and format in binary GiB", () => {
  const [chat] = describeInferenceStatus(
    inference({ active_model: "org/A", resident_mib: 5325 }),
  );
  assert.equal(chat.residentMib, 5325);
  assert.equal(formatResidentMib(5325), "5.2 GiB");
  assert.equal(formatResidentMib(20 * 1024), "20 GiB");
  for (const unknown of [null, undefined, 0, -1, Number.NaN]) {
    assert.equal(formatResidentMib(unknown), null);
  }
  const [image] = describeDiffusionStatus({
    loaded: true,
    repo_id: "org/Flux",
    resident_mib: null,
  } as never);
  assert.equal(image.residentMib, undefined);
});

test("the embedder is listed only while one is resident", () => {
  assert.deepEqual(describeEmbeddingStatus(null), []);
  assert.deepEqual(
    describeEmbeddingStatus({
      embedding_model: "org/E",
      loaded: false,
      backend_loaded: false,
    }),
    [],
  );
  const [row] = describeEmbeddingStatus({
    embedding_model: "org/E",
    loaded: true,
    backend_loaded: true,
  });
  assert.equal(row.name, "org/E");
  assert.equal(row.source, "embedding");
  assert.deepEqual(loadedModelTarget(row.source), {
    open: "settings",
    tab: "data",
    label: "Data settings",
  });
  const [stale] = describeEmbeddingStatus({
    embedding_model: "org/New",
    loaded: false,
    backend_loaded: true,
  });
  assert.equal(stale.name, "Embedding model");
});

test("another account's resident shows only when nothing of this account's is loaded", () => {
  assert.equal(heldByAnotherAccount([{ yours: false }, null, {}]), true);
  assert.equal(heldByAnotherAccount([{ yours: true }, null, {}]), false);
  assert.equal(loadedModelsCardState(0, true, false), "otherAccount");
  assert.equal(loadedModelsCardState(2, true, false), "rows");
  assert.equal(loadedModelsCardState(0, false, true), "empty");
  assert.equal(loadedModelsCardState(0, false, false), null);
});

test("a poll that changed nothing is the same rows", () => {
  const row: LoadedModelEntry = {
    id: "chat:a",
    kind: "text",
    source: "chat",
    name: "a",
    detail: "GGUF",
    residentMib: 100,
  };
  assert.equal(sameLoadedModels([row], [{ ...row }]), true);
  assert.equal(sameLoadedModels([row], [{ ...row, residentMib: 200 }]), false);
  assert.equal(sameLoadedModels([row], [{ ...row, detail: "MLX" }]), false);
  assert.equal(sameLoadedModels([row], []), false);
  assert.equal(
    sameLoadedModels(
      [{ ...row, workflows: ["speak"] }],
      [{ ...row, workflows: ["music"] }],
    ),
    false,
  );
});

test("a drag from a handle no click follows clears the sentinel", () => {
  assert.equal(sentinelAfterSettle(true, true), true);
  assert.equal(sentinelAfterSettle(true, false), false);
  assert.equal(sentinelAfterSettle(false, true), false);
});

test("a kept model sharing the active model's label still gets its own row", () => {
  const rows = describeInferenceStatus({
    active_model: "model",
    model_identifier: "/x/model.gguf",
    is_gguf: true,
    loaded: ["model"],
    serving: ["model", "model"],
    serving_checkpoints: ["/x/model.gguf", "/y/model.gguf"],
  } as never);
  assert.deepEqual(
    rows.filter((r) => r.servingCheckpoint).map((r) => r.servingCheckpoint),
    ["/y/model.gguf"],
  );
});
