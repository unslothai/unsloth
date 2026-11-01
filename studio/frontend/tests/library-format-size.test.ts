// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { formatSize } from "../src/features/library/format.ts";

const t = (key: string, values?: Record<string, unknown>) =>
  `${values?.value} ${key.split(".").pop()}`;
const size = (bytes: number) => formatSize(bytes, "en", t as never);

test("formatSize keeps a unit while its rounded figure is below 1024", () => {
  assert.equal(size(500), "500 bytes");
  assert.equal(size(1023 * 1024), "1,023 kilobytes");
  assert.equal(size(2 * 1024 ** 2), "2.0 megabytes");
});

test("formatSize moves to the next unit when rounding reaches 1024", () => {
  assert.equal(size(1023.6 * 1024), "1.0 megabytes");
  assert.equal(size(1024 ** 2 - 1), "1.0 megabytes");
  assert.equal(size(1024 ** 3 - 1), "1.0 gigabytes");
  assert.equal(size(1024 ** 4 - 1), "1.0 terabytes");
});
