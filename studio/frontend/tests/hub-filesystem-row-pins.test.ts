// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { inventoryPinId } from "../src/features/hub/catalog/inventory-pin.ts";
import { readSrc } from "./helpers/kit.ts";

const hubRows = readSrc("features/hub/catalog/models-catalog-rows.tsx");
const hubLists = readSrc("features/hub/catalog/models-catalog-lists.tsx");

type Row = Parameters<typeof inventoryPinId>[0];
const row = (fields: Record<string, unknown>) => fields as unknown as Row;

test("a cached repo pins by repo id, a filesystem row by its load id", () => {
  assert.equal(
    inventoryPinId(row({ kind: "cache", repoId: "unsloth/Qwen3-0.6B", loadId: "/c/snap" })),
    "unsloth/Qwen3-0.6B",
  );
  assert.equal(
    inventoryPinId(
      row({ kind: "local", source: "hf_cache", repoId: "org/m", loadId: "/hf/org/m" }),
    ),
    "org/m",
  );
  for (const source of ["custom", "lmstudio", "omlx", "models_dir"]) {
    assert.equal(
      inventoryPinId(row({ kind: "local", source, repoId: null, loadId: "/data/models/m" })),
      "/data/models/m",
    );
  }
});

test("a load id holding the repo::quant separator is not pinnable", () => {
  assert.equal(
    inventoryPinId(row({ kind: "local", source: "custom", repoId: null, loadId: "/m/a::b" })),
    null,
  );
});

test("filesystem rows get a Pin menu and sort with the pinned section", () => {
  assert.match(hubRows, /const pinId = isDataset \? null : inventoryPinId\(row\);/);
  assert.match(hubRows, /onToggle: \(\) => toggleRepoPinned\(pinId\),/);
  assert.match(hubRows, /deletableRepoId \|\| \(pinId && !localMenuRow\) \? \(/);
  assert.match(hubRows, /pin=\{pinMenu\}\s*onReveal=\{\(\) => revealLocalModel\(localActionPath, row\.loadId\)\}/);
  assert.match(hubRows, /if \(pinId\) usePinnedModelsStore\.getState\(\)\.unpinRepo\(pinId\);/);
  assert.match(hubLists, /return pinId \? rank\(pinId\) : Number\.MAX_SAFE_INTEGER;/);
  assert.match(hubLists, /return pinId != null && pinCoversId\(pinnedIds, pinId\);/);
});
