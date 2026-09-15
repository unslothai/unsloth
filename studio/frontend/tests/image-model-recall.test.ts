// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  matchesRememberedModel,
  readImageModel,
  rememberImageModel,
} from "../src/features/images/image-model-recall.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";

const { storage } = installLocalStorageFake();

test("recall keeps the exact GGUF artifact", () => {
  const model = {
    repoId: "unsloth/model-GGUF",
    kind: "gguf" as const,
    filename: "model-Q4.gguf",
  };
  rememberImageModel(model);
  assert.deepEqual(readImageModel(), model);
  assert.equal(
    matchesRememberedModel(model, {
      loaded: true,
      repo_id: model.repoId,
      model_kind: "gguf",
      gguf_filename: "model-Q8.gguf",
    }),
    false,
  );
  assert.equal(
    matchesRememberedModel(model, {
      loaded: true,
      repo_id: model.repoId,
      model_kind: "gguf",
      gguf_filename: model.filename,
    }),
    true,
  );
});

test("invalid or incomplete stored targets cannot trigger automatic loading", () => {
  for (const value of [
    "broken",
    "null",
    "{}",
    '{"repoId":"org/model","kind":"gguf"}',
    '{"repoId":"org/model","kind":"unknown"}',
  ]) {
    storage.setItem("unsloth:images:last-model", value);
    assert.equal(readImageModel(), null);
  }
});
