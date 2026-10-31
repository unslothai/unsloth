// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import { CANVAS_DELETE_KEYS } from "../src/features/recipe-studio/constants.ts";
import { readSrc } from "./helpers/kit.ts";

test("recipe canvas deletes selected blocks on Delete as well as Backspace", () => {
  assert.deepEqual([...CANVAS_DELETE_KEYS].sort(), ["Backspace", "Delete"]);
  const src = readSrc("features/recipe-studio/recipe-studio-page.tsx");
  const canvas = src.slice(src.indexOf("<ReactFlow<"));
  assert.match(
    canvas.slice(0, canvas.indexOf("\n      >\n")),
    /\bdeleteKeyCode=\{CANVAS_DELETE_KEYS\}/,
  );
});
