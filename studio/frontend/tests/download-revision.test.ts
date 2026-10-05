// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { sameDownloadRevision } from "../src/features/hub/download-manager/download-manager-types.ts";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");
}

const ADAPTER = read(
  "../src/features/hub/download-manager/download-api-adapter.ts",
);
const HYDRATION = read("../src/features/hub/download-manager/hydration.ts");
const CONFLICTS = read(
  "../src/features/hub/download-manager/transport-conflict.ts",
);
const FORWARDS_REVISION = /\{ revision: req\.revision \}/;
const HYDRATES_REVISION =
  /download\.revision\?\.trim\(\).*revision: download\.revision\.trim\(\)/s;
const COMPARES_REVISION =
  /sameDownloadRevision\(job\.revision, req\.revision\)/;

test("immutable scoped revisions compare by commit identity", () => {
  assert.equal(sameDownloadRevision("A".repeat(40), "a".repeat(40)), true);
  assert.equal(sameDownloadRevision("a".repeat(40), "b".repeat(40)), false);
  assert.equal(sameDownloadRevision(undefined, null), true);
  assert.equal(sameDownloadRevision(undefined, "a".repeat(40)), false);
});

test("the manager sends and rehydrates a scoped pinned revision", () => {
  assert.match(ADAPTER, FORWARDS_REVISION);
  assert.match(HYDRATION, HYDRATES_REVISION);
  assert.match(CONFLICTS, COMPARES_REVISION);
});
