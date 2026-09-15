// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { looksLikeLocalPath } from "../src/features/hub/lib/local-path.ts";

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
