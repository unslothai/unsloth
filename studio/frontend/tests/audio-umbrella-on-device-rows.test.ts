// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The backend lists each cached folder of the audio.cpp umbrella repo as its own row, next to
// the one umbrella row. Only the Audio pages that offer a folder may list it; there it reads as
// its folder name, sits with the Unsloth rows, links to its tree, and can be revealed or deleted.

import assert from "node:assert/strict";
import test from "node:test";

import type { CachedInventoryRow } from "../src/features/hub/inventory/types.ts";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { umbrellaFolderRowAllowed } =
  await import("../src/features/model-picker/components/model-selector/audio-picker-policy.ts");
const { isFeaturedOnDeviceRepoId, onDeviceRepoLabel } =
  await import("../src/features/model-picker/components/model-selector/row-meta.ts");
const { buildCachedInventoryRow, withUmbrellaFolderRows } =
  await import("../src/features/hub/inventory/view-models.ts");

const UMBRELLA = "audio-cpp/audio.cpp-gguf";
const CLONE_FOLDER = `${UMBRELLA}/Qwen3-TTS-12Hz-0.6B-Base-GGUF`;
const ASR_FOLDER = `${UMBRELLA}/Qwen3-ASR-0.6B-GGUF`;

const PICKERS = readSrc(
  "features/model-picker/components/model-selector/pickers.tsx",
);
const HUB_INVENTORY = readSrc("features/hub/inventory/use-hub-inventory.ts");
const PICKER_INVENTORY = readSrc(
  "features/model-picker/inventory/use-chat-picker-inventory.ts",
);

function folderRow(repoId: string): CachedInventoryRow {
  return buildCachedInventoryRow(
    {
      repo_id: repoId,
      load_id: repoId,
      size_bytes: 1024,
      cache_path: "/c/hub/models--audio-cpp--audio.cpp-gguf",
      task: "text-to-speech",
      audio_type: "audiocpp_tts",
      audio_workflows: ["clone"],
      umbrella_folder: true,
      model_format: "gguf",
      capabilities: { can_chat: false },
    },
    "gguf",
  );
}

test("a folder row keeps its folder flag and workflows through the view model", () => {
  const row = folderRow(CLONE_FOLDER);
  assert.equal(row.umbrellaFolder, true);
  assert.deepEqual(row.audioWorkflows, ["clone"]);
  assert.equal(row.capabilities.canChat, false);
  const plain = buildCachedInventoryRow(
    { repo_id: UMBRELLA, size_bytes: 1, model_format: "gguf" },
    "gguf",
  );
  assert.equal(plain.umbrellaFolder, false);
  assert.equal(plain.audioWorkflows, null);
});

test("hub inventory drops folder rows unless a caller opts in", () => {
  const umbrella = buildCachedInventoryRow(
    { repo_id: UMBRELLA, size_bytes: 1, model_format: "gguf" },
    "gguf",
  );
  const rows = [umbrella, folderRow(CLONE_FOLDER)];
  assert.deepEqual(
    withUmbrellaFolderRows(rows, false).map((row) => row.repoId),
    [UMBRELLA],
  );
  assert.equal(withUmbrellaFolderRows(rows, true).length, 2);
  assert.match(
    HUB_INVENTORY,
    /const includeUmbrellaFolders = options\.includeUmbrellaFolders \?\? false;/,
  );
  assert.match(
    HUB_INVENTORY,
    /withUmbrellaFolderRows\(\s*cachedGguf\.map\(\(row\) => buildCachedInventoryRow\(row, "gguf"\)\),\s*includeUmbrellaFolders,\s*\)/,
  );
});

test("the picker lists a folder row only for a page that passed its exact id", () => {
  const row = folderRow(CLONE_FOLDER);
  assert.equal(
    umbrellaFolderRowAllowed(row, undefined),
    false,
    "chat passes none",
  );
  assert.equal(umbrellaFolderRowAllowed(row, new Set()), false);
  assert.equal(
    umbrellaFolderRowAllowed(row, new Set([ASR_FOLDER.toLowerCase()])),
    false,
    "another page's folder",
  );
  assert.equal(
    umbrellaFolderRowAllowed(row, new Set([CLONE_FOLDER.toLowerCase()])),
    true,
  );
  assert.equal(
    umbrellaFolderRowAllowed({ repoId: UMBRELLA }, undefined),
    true,
    "any other row is left to the hidden-model gate",
  );
  assert.match(PICKER_INVENTORY, /includeUmbrellaFolders: true,/);
  assert.match(
    PICKER_INVENTORY,
    /row\.modelFormat === "gguf" &&\s*isListableCachedRow\(row\) &&\s*umbrellaFolderRowAllowed\(row, options\.allowedHiddenModelIds\) &&/,
  );
  assert.match(
    PICKER_INVENTORY,
    /audio_workflows: row\.audioWorkflows \?\? null,/,
  );
});

test("On Device rows name a folder by its folder and group it with Unsloth's", () => {
  assert.equal(
    onDeviceRepoLabel(CLONE_FOLDER),
    "Qwen3-TTS-12Hz-0.6B-Base-GGUF",
  );
  assert.equal(onDeviceRepoLabel(UMBRELLA), UMBRELLA);
  assert.equal(onDeviceRepoLabel("someone/Model-GGUF"), "someone/Model-GGUF");
  assert.equal(isFeaturedOnDeviceRepoId(CLONE_FOLDER), true);
  assert.equal(isFeaturedOnDeviceRepoId(ASR_FOLDER), true);
  assert.equal(isFeaturedOnDeviceRepoId("unsloth/Qwen3-8B-GGUF"), true);
  assert.equal(isFeaturedOnDeviceRepoId("unslothai/Foo"), true);
  assert.equal(isFeaturedOnDeviceRepoId(UMBRELLA), false);
  assert.equal(isFeaturedOnDeviceRepoId("someone/Model-GGUF"), false);
});

test("pickers render folder rows through the label, group, link and menu helpers", () => {
  assert.match(
    PICKERS,
    /function isUnslothPublisherRepoId\(repoId: string\): boolean \{\n\s*return isFeaturedOnDeviceRepoId\(repoId\);\n\}/,
  );
  assert.match(
    PICKERS,
    /huggingface\.co\/\$\{audioCppHubPath\(trimmed\) \?\? trimmed\}/,
  );
  for (const fn of [
    "const renderSoleQuantGgufRow",
    "const renderDownloadedGgufRow",
    "const renderAdditionalOnDeviceModelRow",
  ]) {
    const start = PICKERS.indexOf(fn);
    assert.ok(start > 0, fn);
    const body = PICKERS.slice(start, PICKERS.indexOf("\n  };\n", start));
    assert.match(body, /label=\{[^}]*onDeviceRepoLabel\(/, fn);
  }
  const start = PICKERS.indexOf("const renderAdditionalOnDeviceModelRow");
  const body = PICKERS.slice(start, PICKERS.indexOf("\n  };\n", start));
  assert.match(
    body,
    /isAudioCppFolderId\(model\.id\) \? \(\s*<span className=\{ROW_ACTIONS_CLASS\}>\s*<ModelRowMenu[\s\S]*cachePath=\{\{ repoId: model\.id \}\}[\s\S]*deleteCachedModel\(model\.id, undefined, hfToken \|\| undefined\)/,
  );
  assert.match(
    body,
    /<span aria-hidden="true" className=\{cn\(ROW_ACTIONS_CLASS, "h-6"\)\} \/>/,
  );
});
