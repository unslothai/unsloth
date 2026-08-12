// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  looksLikeLocalPath,
  routableToMediaPage,
} from "../src/features/hub/lib/local-path.ts";

test("recognizes local paths across supported operating systems", () => {
  for (const path of [
    "/datasets/train",
    "./datasets/train",
    "../datasets/train",
    "~/datasets/train",
    "~user/datasets/train",
    "C:/datasets/train",
    String.raw`C:\datasets\train`,
    String.raw`C:datasets\train`,
    String.raw`\datasets\train`,
    String.raw`\\server\datasets\train`,
  ]) {
    assert.equal(looksLikeLocalPath(path), true, path);
  }
});

test("preserves repository identifiers that contain no local path syntax", () => {
  assert.equal(looksLikeLocalPath("model:variant"), false);
});

test("does not classify Hugging Face repository identifiers as paths", () => {
  assert.equal(looksLikeLocalPath("HuggingFaceH4/ultrachat_200k"), false);
  assert.equal(looksLikeLocalPath("unsloth/Qwen3-4B"), false);
});

test("an HF-cache row routes to the media pages, a filesystem row does not", () => {
  // Inventory dedup can leave a complete hf_cache LOCAL row as the only row for a repo
  // (use-models-selection drops the partial cached row beside it), and that row carries the
  // Hub id. Excluding it by kind sent a diffusion GGUF to the chat loader, where the backend
  // refuses it, instead of to Images or Video.
  assert.equal(routableToMediaPage("local", "hf_cache"), true);
  assert.equal(routableToMediaPage("local", "models_dir"), false);
  assert.equal(routableToMediaPage("local", "lmstudio"), false);
  assert.equal(routableToMediaPage("local", "ollama"), false);
  assert.equal(routableToMediaPage("local", null), false);
  assert.equal(routableToMediaPage("cache", null), true);
  assert.equal(routableToMediaPage("discover", null), true);
});
