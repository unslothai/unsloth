// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Custom-folder models used to omit Pin from the row menu, so only Hub downloads could
// surface in the Pinned section. The picker now pins those local rows by id and moves
// them out of Custom folders, the same way fine-tunes leave their section.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  readSrc,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const { pinKey, pinnedQuantEntries, usePinnedModelsStore } = await import(
  "../src/features/model-picker/components/model-selector/pinned-models.ts"
);

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

test("LM Studio and ./models rows still omit Pin", () => {
  assert.equal(PICKERS.split("{renderLocalRowMenu(m)}").length - 1, 2);
  assert.equal(PICKERS.split("{renderLocalRowMenu(m, true)}").length - 1, 1);
});

test("deleting a pinned custom-folder row drops its pin", () => {
  assert.match(
    PICKERS,
    /await deleteLocalModel\(m\.path, m\.id\);\s*if \(isPinned\) unpinRepo\(m\.id\);/,
  );
});

test("Custom folders stays empty only when nothing from that list is pinned either", () => {
  assert.match(
    PICKERS,
    /unpinnedCustomFolderModels\.length === 0 &&\s*pinnedCustomFolderRows\.length === 0/,
  );
});

test("a custom-folder load id pins as a whole model, not a phantom GGUF quant", () => {
  const id = "/home/user/models/Qwen3-4B-Instruct.gguf";
  usePinnedModelsStore.setState({ pinned: [] });
  usePinnedModelsStore.getState().togglePinned(id);
  const pinned = usePinnedModelsStore.getState().pinned;
  assert.deepEqual(pinned, [pinKey(id)]);
  assert.equal(pinKey(id), id);
  assert.deepEqual(pinnedQuantEntries(pinned), []);
  usePinnedModelsStore.getState().unpinRepo(id);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, []);
});
