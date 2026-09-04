// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { scanFolderStatusCopy } = await import(
  "../src/features/hub/lib/scan-folder-status.ts"
);

const MAC = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)";
const WINDOWS = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)";
const LINUX = "Mozilla/5.0 (X11; Linux x86_64)";

test("a healthy folder shows nothing", () => {
  assert.equal(scanFolderStatusCopy("ok", MAC), null);
});

test("a folder from an older backend shows nothing", () => {
  assert.equal(scanFolderStatusCopy(undefined, MAC), null);
});

test("a denied folder says so instead of looking empty", () => {
  const copy = scanFolderStatusCopy("permission_denied", MAC);
  assert.ok(copy);
  assert.match(copy.title, /not allowed to read/i);
});

test("the permission hint names the screen that fixes it", () => {
  assert.match(
    scanFolderStatusCopy("permission_denied", MAC)!.hint,
    /System Settings > Privacy & Security > Files and Folders/,
  );
  assert.match(
    scanFolderStatusCopy("permission_denied", WINDOWS)!.hint,
    /Controlled Folder Access/,
  );
  // No screen worth naming on Linux, so the hint stays generic.
  assert.match(
    scanFolderStatusCopy("permission_denied", LINUX)!.hint,
    /permissions/i,
  );
});

test("a missing folder is not reported as a permission problem", () => {
  const copy = scanFolderStatusCopy("missing", MAC);
  assert.ok(copy);
  assert.match(copy.title, /no longer there/i);
  assert.doesNotMatch(copy.hint, /permission/i);
});

test("a partial folder does not claim the whole folder is blocked", () => {
  const copy = scanFolderStatusCopy("partial", MAC);
  assert.ok(copy);
  assert.match(copy.title, /some models/i);
  assert.doesNotMatch(copy.title, /not allowed/i);
  // Same fix, so the same hint.
  assert.match(copy.hint, /Files and Folders/);
});

test("an unreadable folder points at the drive", () => {
  const copy = scanFolderStatusCopy("unreadable", MAC);
  assert.ok(copy);
  assert.match(copy.hint, /drive/i);
});

test("the model picker renders the status on its folder rows too", async () => {
  const source = await readFile(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, /scanFolderStatusCopy\(f\.status\)/);
  assert.match(source, /\{problem\.title\}/);
});

test("the folders dialog renders the status on the row", async () => {
  const source = await readFile(
    new URL(
      "../src/features/hub/catalog/on-device-folders-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, /scanFolderStatusCopy\(folder\.status\)/);
  assert.match(source, /\{problem\.title\}\. \{problem\.hint\}/);
});
