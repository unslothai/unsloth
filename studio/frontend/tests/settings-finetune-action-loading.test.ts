// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../src/features/settings/tabs/data-tab.tsx", import.meta.url),
  "utf8",
);
const STATIC_FINE_TUNE_IMPORT = /finetune-recipe/;
const ACTION_FINE_TUNE_IMPORT =
  /await import\(\s*"\.\.\/components\/finetune-recipe"\s*\)/g;

test("fine-tuning workflow dependencies load only when their actions run", () => {
  // Bundle membership follows the static import graph, so this structural
  // assertion guards startup loading more directly than exercising the actions.
  const imports = source.slice(0, source.indexOf("export function DataTab"));
  assert.doesNotMatch(
    imports,
    STATIC_FINE_TUNE_IMPORT,
    "the Data tab must not pull fine-tuning workflows into Studio startup",
  );

  const actionImports = source.match(ACTION_FINE_TUNE_IMPORT);
  assert.equal(actionImports?.length, 2);
});
