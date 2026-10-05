// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { resolveResumeRemoteCodeCache } from "../src/features/training/lib/resume-remote-code-cache.ts";

test("resume consent targets the attested model snapshot", () => {
  const snapshot = "/cache/models--org--actual-model/snapshots/original";

  assert.deepEqual(
    resolveResumeRemoteCodeCache({
      actualModelRepoId: "org/actual-model",
      modelKnownCached: false,
      modelLocalPath: "/cache/models--org--actual-model",
      modelSnapshotPath: snapshot,
    }),
    {
      preferLocalCache: true,
      modelLocalPath: snapshot,
      modelSnapshotPath: snapshot,
      modelSnapshotRepoId: "org/actual-model",
    },
  );
});

test("direct snapshot consent lets the backend validate against the selected repo", () => {
  const snapshot = "/cache/models--org--model/snapshots/original";

  assert.deepEqual(
    resolveResumeRemoteCodeCache({
      modelSnapshotPath: snapshot,
    }),
    {
      preferLocalCache: true,
      modelLocalPath: snapshot,
      modelSnapshotPath: snapshot,
      modelSnapshotRepoId: null,
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
      modelSnapshotPath: null,
      modelSnapshotRepoId: null,
    },
  );
});

test("resume consent honors a stored cache path without a legacy cache flag", () => {
  const cachePath = "/cache/models--org--model";

  assert.deepEqual(
    resolveResumeRemoteCodeCache({
      modelKnownCached: false,
      modelLocalPath: cachePath,
    }),
    {
      preferLocalCache: true,
      modelLocalPath: cachePath,
      modelSnapshotPath: null,
      modelSnapshotRepoId: null,
    },
  );
});

test("uncached resume consent scans the repository target", () => {
  assert.deepEqual(resolveResumeRemoteCodeCache({ modelKnownCached: false }), {
    preferLocalCache: false,
    modelLocalPath: null,
    modelSnapshotPath: null,
    modelSnapshotRepoId: null,
  });
});
