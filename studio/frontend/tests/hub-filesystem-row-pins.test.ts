// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const hubPin = readSrc("features/hub/catalog/inventory-pin.ts");
const hubRows = readSrc("features/hub/catalog/models-catalog-rows.tsx");
const hubLists = readSrc("features/hub/catalog/models-catalog-lists.tsx");

test("the Model Hub pins a filesystem row by its load id", () => {
  assert.match(
    hubPin,
    /if \(row\.kind === "cache" \|\| row\.source === "hf_cache"\) \{\s*return row\.repoId;\s*\}/,
  );
  // A "::" id would read back as a repo::quant pin.
  assert.match(hubPin, /return row\.loadId\.includes\("::"\) \? null : row\.loadId;/);
});

test("a filesystem row gets a row menu with Pin even when it cannot be deleted", () => {
  assert.match(hubRows, /const pinId = isDataset \? null : inventoryPinId\(row\);/);
  assert.match(hubRows, /const rowMenu =\s*deletableRepoId \|\| pinId \? \(/);
  assert.match(hubRows, /onToggle: \(\) => togglePinned\(pinId\),/);
});

test("pinned filesystem rows sort and count with the pinned section", () => {
  assert.match(
    hubLists,
    /const pinId = inventoryPinId\(item\.row\);\s*return pinId \? rank\(pinKey\(pinId\)\) : Number\.MAX_SAFE_INTEGER;/,
  );
  assert.match(
    hubLists,
    /const pinId = inventoryPinId\(item\.row\);\s*return pinId != null && pinnedSet\.has\(pinKey\(pinId\)\);/,
  );
  assert.match(hubLists, /const itemPinId = isDataset \? null : inventoryPinId\(item\.row\);/);
});
