// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { firstIndexByKey } from "../src/features/model-picker/components/model-selector/option-index.ts";

import { readSrc } from "./helpers/kit.ts";

const PICKERS = readSrc("features/model-picker/components/model-selector/pickers.tsx");

test("the index map answers exactly what indexOf and includes did, first match on duplicates", () => {
  const keys = ["hub::a", "connected::b", "hub::a", "local::c", ""];
  const index = firstIndexByKey(keys);
  for (const key of [...keys, "missing", "hub::"]) {
    assert.equal(index.get(key) ?? -1, keys.indexOf(key), key);
    assert.equal(index.has(key), keys.includes(key), key);
  }
  assert.equal(firstIndexByKey([]).size, 0);
});

test("the roving list no longer scans optionKeys per row", () => {
  const start = PICKERS.indexOf("function useRovingModelList(");
  const body = PICKERS.slice(start, PICKERS.indexOf("\n}\n", start));
  assert.doesNotMatch(body, /optionKeys\.(indexOf|includes)\(/);
  assert.match(body, /useMemo\(\(\) => firstIndexByKey\(optionKeys\), \[optionKeys\]\)/);
});
