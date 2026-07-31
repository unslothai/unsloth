// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { resolveResumeRemoteCodeCache } from "../src/features/training/lib/resume-remote-code-cache.ts";

test("resume consent targets the attested model snapshot", () => {
  const snapshot = "/cache/models--org--model/snapshots/original";

  assert.deepEqual(
    resolveResumeRemoteCodeCache({
      modelKnownCached: false,
      modelLocalPath: "/cache/models--org--model",
      modelSnapshotPath: snapshot,
    }),
    {
      preferLocalCache: true,
      modelLocalPath: snapshot,
    },
  );
});

test("legacy resume consent retains cached-model selection", () => {
  const cachePath = "/cache/models--org--model";

  assert.deepEqual(
    resolveResumeRemoteCodeCache({
      modelKnownCached: true,
      modelLocalPath: cachePath,
      modelSnapshotPath: null,
    }),
    {
      preferLocalCache: true,
      modelLocalPath: cachePath,
    },
  );
});

test("uncached resume consent scans the repository target", () => {
  assert.deepEqual(
    resolveResumeRemoteCodeCache({
      modelKnownCached: false,
      modelLocalPath: "/stale/cache/hint",
    }),
    {
      preferLocalCache: false,
      modelLocalPath: null,
    },
  );
});
