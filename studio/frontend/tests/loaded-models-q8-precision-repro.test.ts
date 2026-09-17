// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { describeDiffusionStatus } from "../src/features/loaded-models/loaded-models-sources.ts";

test("fixed: a Q8 Z-Image GGUF row names Q8 instead of its BF16 compute dtype", () => {
  const [row] = describeDiffusionStatus({
    loaded: true,
    repo_id: "unsloth/Z-Image-Turbo-GGUF",
    family: "z-image",
    device: "cuda",
    dtype: "bfloat16",
    model_kind: "gguf",
    gguf_variant: "Q8_0",
  } as never);

  assert.equal(row.detail, "z-image · GGUF · Q8_0 · cuda");
  assert.equal(row.detail.includes("BF16"), false);
  console.log(
    "PASS selected=z-image-turbo-Q8_0.gguf indicator=z-image · GGUF · Q8_0 · cuda",
  );
});
