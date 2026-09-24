// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { parentFolder } from "../src/features/library/paths.ts";

test("the picker starts in the folder above, drive and share roots included", () => {
  assert.equal(parentFolder("/Users/me/Pictures/Unsloth Images"), "/Users/me/Pictures");
  assert.equal(parentFolder("/Unsloth Images"), "/");
  assert.equal(parentFolder("/Users/me/Pictures/"), "/Users/me");
  // `D:` alone is drive D's current folder, not its root.
  assert.equal(parentFolder("D:\\Unsloth Images"), "D:\\");
  assert.equal(parentFolder("D:/Unsloth Images"), "D:/");
  assert.equal(parentFolder("C:\\Users\\me\\Unsloth"), "C:\\Users\\me");
  assert.equal(parentFolder("\\\\server\\share\\Unsloth Images"), "\\\\server\\share\\");
  assert.equal(parentFolder("\\\\server\\share\\a\\b"), "\\\\server\\share\\a");
});

test("a root has nothing above it to start from", () => {
  for (const root of ["/", "D:\\", "\\\\server\\share", "relative", ""]) {
    assert.equal(parentFolder(root), undefined, root);
  }
});
