// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getEstimatedSize } from "../src/features/export/constants.ts";
import { formatBytes } from "../src/features/hub/lib/format.ts";
import { formatSize as formatLibrarySize } from "../src/features/library/format.ts";
import { sttModelSize } from "../src/features/settings/stores/stt-model-catalog.ts";
import { readSrc } from "./helpers/kit.ts";

const GGUF_BYTES = 4_900_000_000;

test("one byte count reads the same on the Export page as in the picker and Hub", () => {
  assert.equal(getEstimatedSize("merged", [], GGUF_BYTES), "~4.9 GB");
  assert.equal(formatBytes(GGUF_BYTES), "4.9 GB");
  assert.equal(formatBytes(GGUF_BYTES, ""), "4.9GB");
});

test("Settings, chat and update screens size files with the shared formatter", () => {
  for (const path of [
    "features/settings/tabs/agents-tab.tsx",
    "features/chat/components/project-files-panel.tsx",
    "features/settings/components/uploaded-files-dialog.tsx",
    "features/rag/components/project-source-dropzone.tsx",
    "components/assistant-ui/sandbox-files-view.tsx",
    "components/llama-update-banner.tsx",
    "features/model-picker/components/model-config-page.tsx",
    "features/model-picker/components/model-selector/pickers.tsx",
  ]) {
    const source = readSrc(path);
    assert.ok(/\bformatBytes\(/.test(source), path);
    assert.ok(
      !/function (formatBytes|formatSize|formatFileSize|formatTableSize)\b/.test(source),
      path,
    );
    assert.ok(!/\/ \(?1024\b(?! \*\* 3)/.test(source), path);
  }
});

test("a Library file reads the same size as in chat and Settings", () => {
  assert.equal(formatBytes(1_500_000), "1.5 MB");
  const t = (key: string, values?: { value?: unknown }) =>
    key === "library.size.megabytes" ? `${String(values?.value)} MB` : key;
  assert.equal(formatLibrarySize(1_500_000, "en", t as never), "1.5 MB");
});

test("an audio.cpp dictation model reads the same size as the Hub file", () => {
  // qwen3-asr-1.7b-q8_0.gguf is 2,473,017,600 bytes on Hugging Face.
  assert.equal(sttModelSize("audiocpp-qwen3-asr-1.7b"), "2.5 GB");
  assert.equal(sttModelSize("qwen3-asr-1.7b"), "2.5 GB");
});
