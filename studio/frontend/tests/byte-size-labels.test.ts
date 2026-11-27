// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { AUDIO_CPP_DICTATION_MODELS } from "../src/features/audio/audio-cpp-catalog.ts";
import { getEstimatedSize } from "../src/features/export/constants.ts";
import { formatBytes } from "../src/features/hub/lib/format.ts";
import { formatSize as formatLibrarySize } from "../src/features/library/format.ts";
import {
  sttModelBytes,
  sttModelSize,
} from "../src/features/settings/stores/stt-model-catalog.ts";
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
    "components/assistant-ui/tool-confirmation-controls.tsx",
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

test("Library and Hub pick the same unit and round the same way", () => {
  const t = (key: string, values?: { value?: unknown }) =>
    `${String(values?.value)} ${key.split(".").pop()}`;
  for (const [bytes, hub, library] of [
    [999_999, "1.0 MB", "1.0 megabytes"],
    [9_960, "10 KB", "10 kilobytes"],
    [999, "999 B", "999 bytes"],
    [2_550_000, "2.5 MB", "2.5 megabytes"],
  ] as const) {
    assert.equal(formatBytes(bytes), hub);
    assert.equal(formatLibrarySize(bytes, "en", t as never), library);
  }
});

test("an audio.cpp dictation model reads the same size as the Hub file", () => {
  // qwen3-asr-1.7b-q8_0.gguf is 2,473,017,600 bytes on Hugging Face.
  assert.equal(sttModelSize("audiocpp-qwen3-asr-1.7b"), "2.5 GB");
  assert.equal(sttModelSize("qwen3-asr-1.7b"), "2.5 GB");
});

test("the Audio picker sorts dictation models by their byte size, not their label", () => {
  const model = AUDIO_CPP_DICTATION_MODELS.find((m) => m.key === "audiocpp-qwen3-asr-1.7b");
  assert.ok(model);
  assert.equal(sttModelBytes(model.key), model.sizeBytes);
  assert.equal(sttModelBytes("qwen3-asr-1.7b"), 2.5e9);
  const page = readSrc("features/audio/audio-page.tsx");
  assert.match(page, /deviceSizeBytes: sttModelBytes\(/);
  assert.ok(!/deviceSizeBytes\(size\)/.test(page));
});
