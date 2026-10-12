// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  componentFileFields,
  splitComponentFileList,
} from "../src/features/images/component-files.ts";

test("splitComponentFileList accepts newlines and commas, trims, drops blanks", () => {
  assert.deepEqual(splitComponentFileList(""), []);
  assert.deepEqual(splitComponentFileList(undefined), []);
  assert.deepEqual(splitComponentFileList("  \n , \n"), []);
  assert.deepEqual(
    splitComponentFileList(" ../text_encoders/clip_l.safetensors \n\n/abs/t5xxl.safetensors, owner/repo/te.safetensors ,"),
    ["../text_encoders/clip_l.safetensors", "/abs/t5xxl.safetensors", "owner/repo/te.safetensors"],
  );
});

test("componentFileFields sends only for single-file / GGUF and only non-empty values", () => {
  assert.deepEqual(componentFileFields("pipeline", ["a.safetensors"], "vae.safetensors"), {});
  assert.deepEqual(componentFileFields(undefined, ["a.safetensors"], "vae.safetensors"), {});
  assert.deepEqual(componentFileFields("gguf", undefined, undefined), {});
  assert.deepEqual(componentFileFields("gguf", [" ", ""], "  "), {});
  assert.deepEqual(componentFileFields("single_file", ["a.safetensors"], undefined), {
    text_encoder_file: ["a.safetensors"],
  });
  assert.deepEqual(componentFileFields("gguf", "a.safetensors\nb.safetensors", " vae.safetensors "), {
    text_encoder_file: ["a.safetensors", "b.safetensors"],
    vae_file: "vae.safetensors",
  });
});
