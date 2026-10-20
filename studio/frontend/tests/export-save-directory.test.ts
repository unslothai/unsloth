// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { saveDirectoryField } from "../src/features/export/lib/save-directory.ts";

const DEFAULT = "qwen3-4b-GGUF";

test("the save folder input shows the default until the user edits it", () => {
  assert.deepEqual(saveDirectoryField(null, DEFAULT), {
    inputValue: DEFAULT,
    saveDirectory: DEFAULT,
  });
});

test("a cleared save folder input stays empty and exports to the default", () => {
  assert.deepEqual(saveDirectoryField("", DEFAULT), {
    inputValue: "",
    saveDirectory: DEFAULT,
  });
});

test("a trailing space survives typing and is trimmed from the export path", () => {
  assert.deepEqual(saveDirectoryField("D:\\My ", DEFAULT), {
    inputValue: "D:\\My ",
    saveDirectory: "D:\\My",
  });
});

test("a pasted absolute path is used as typed", () => {
  assert.deepEqual(saveDirectoryField("/Volumes/ext/exports", DEFAULT), {
    inputValue: "/Volumes/ext/exports",
    saveDirectory: "/Volumes/ext/exports",
  });
});
