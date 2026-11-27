// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

const PICKER =
  "../src/components/assistant-ui/composer-model-picker/composer-model-picker.tsx";

test("Image/video gen lists only image and video rows, and only under its own filter", async () => {
  const { matchesFormat } = await import(
    "../src/components/assistant-ui/composer-model-picker/picker-model.ts"
  );
  const image = { kind: "gguf", format: "gguf", media: "images" } as const;
  const video = { kind: "cached", format: "checkpoint", media: "video" } as const;
  const audio = { kind: "gguf", format: "gguf", media: "audio" } as const;
  const chat = { kind: "gguf", format: "gguf" } as const;
  for (const row of [image, video]) {
    assert.equal(matchesFormat(row, "media"), true);
    for (const other of ["all", "gguf", "safetensors", "mlx"] as const) {
      assert.equal(matchesFormat(row, other), false, `${row.media} under ${other}`);
    }
  }
  assert.equal(matchesFormat(audio, "media"), false);
  assert.equal(matchesFormat(chat, "media"), false);
  assert.equal(matchesFormat(audio, "gguf"), true, "audio rows keep their format");
  assert.equal(matchesFormat(chat, "all"), true);
});

test("the composer offers Image/video gen on On Device only, and only with such models", async () => {
  const src = await readText(PICKER);
  assert.match(src, /formatGroup\(deviceFormat, hasGenerationModels\)/);
  assert.match(src, /formatGroup\(formatFilter, false\)/);
  assert.match(src, /chosenFormat === "media" \? "all" : chosenFormat/, "Discover reads it as All");
});
