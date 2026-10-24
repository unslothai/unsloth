// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Custom-folder models used to omit Pin from the row menu, so only Hub downloads could
// surface in the Pinned section. The picker now pins those local rows by id and moves
// them out of Custom folders, the same way fine-tunes leave their section.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PICKERS = readSrc(
  "features/model-picker/components/model-selector/pickers.tsx",
);

test("custom-folder rows expose Pin and leave the Custom folders list when pinned", () => {
  assert.match(
    PICKERS,
    /const renderLocalRowMenu = \(m: LocalModelInfo, allowPin = false\)/,
  );
  assert.match(PICKERS, /\{renderLocalRowMenu\(m, true\)\}/);
  assert.match(
    PICKERS,
    /sortedCustomFolderModels\.filter\(\(m\) => pinnedSet\.has\(pinKey\(m\.id\)\)\)/,
  );
  assert.match(PICKERS, /unpinnedCustomFolderModels\.map\(\(m\) =>/);
  assert.match(PICKERS, /row\.local \? \(\s*renderCustomFolderModelRow\(row\.local\)/);
  assert.match(
    PICKERS,
    /makeModelOptionKey\("custom-folder", row\.local\.id\)/,
  );
});

test("custom-folder pins use the whole-model pinKey(m.id), not a GGUF quant suffix", () => {
  assert.match(PICKERS, /onToggle: \(\) => togglePinned\(m\.id\)/);
  assert.match(
    PICKERS,
    /sortedCustomFolderModels\.filter\(\(m\) => pinnedSet\.has\(pinKey\(m\.id\)\)\)/,
  );
  assert.doesNotMatch(
    PICKERS,
    /renderLocalRowMenu\(m, true\)[\s\S]{0,200}togglePinned\(m\.id,/,
  );
});
