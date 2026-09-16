// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  pendingDrafterPresentation,
  presentationForExpectedBytesUpdate,
  presentationForJobStart,
  presentedProgress,
} from "../src/features/hub/download-manager/download-presentation.ts";
import { readSrc } from "./helpers/kit.ts";
import { carriesOverSeed } from "../src/features/hub/download-manager/adopt-rules.ts";

test("a pending MTP file becomes the download manager presentation", () => {
  assert.deepEqual(
    pendingDrafterPresentation({
      filename: "Qwen3.8-Flash-Next-Q4.gguf",
      quant: "Q4",
      size_bytes: 100,
      pending_drafter_filename: "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
      pending_drafter_size_bytes: 20,
    }),
    {
      label: "MTP companion",
      filename: "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
      expectedBytes: 20,
    },
  );
});

test("companion presentation subtracts the cached main model from progress", () => {
  const mainBytes = 112_238_658_784;
  const mtpBytes = 2_786_568_256;
  const mtpTransferred = 540_000_000;

  assert.deepEqual(
    presentedProgress({
      expectedBytes: mainBytes + mtpBytes,
      downloadedBytes: mainBytes + mtpTransferred,
      fraction: (mainBytes + mtpTransferred) / (mainBytes + mtpBytes),
      presentation: {
        label: "MTP companion",
        filename: "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
        expectedBytes: mtpBytes,
      },
    }),
    {
      expectedBytes: mtpBytes,
      downloadedBytes: mtpTransferred,
      fraction: mtpTransferred / mtpBytes,
    },
  );
});

test("companion presentation keeps backend baseline-adjusted progress", () => {
  const presentation = presentationForJobStart(
    {
      label: "MTP companion",
      filename: "mtp-shared-Q8_0.gguf",
      expectedBytes: 20,
    },
    undefined,
    120,
    false,
  );

  assert.deepEqual(
    presentedProgress({
      // snapshot_progress has already removed the cached 100-byte main model.
      expectedBytes: 20,
      downloadedBytes: 5,
      fraction: 0.25,
      presentation,
    }),
    { expectedBytes: 20, downloadedBytes: 5, fraction: 0.25 },
  );
});

test("ordinary downloads retain their plan-wide counters", () => {
  assert.deepEqual(
    presentedProgress({
      expectedBytes: 100,
      downloadedBytes: 25,
      fraction: 0.25,
    }),
    { expectedBytes: 100, downloadedBytes: 25, fraction: 0.25 },
  );
});

test("backend-active adoption keeps and stabilizes persisted presentation", () => {
  const existing = {
    label: "MTP companion",
    filename: "mtp-shared-Q8_0.gguf",
    expectedBytes: 20,
  };
  assert.deepEqual(presentationForJobStart(undefined, existing, 120, true), {
    ...existing,
    cachedPlanPrefixBytes: 100,
  });
  assert.equal(
    presentationForJobStart(undefined, existing, 120, false),
    undefined,
  );
});

test("adoption defers stabilization while the backend total is unknown", () => {
  const existing = {
    label: "MTP companion",
    filename: "mtp-shared-Q8_0.gguf",
    expectedBytes: 20,
  };
  assert.deepEqual(presentationForJobStart(undefined, existing, 0, true), existing);
});

test("a different backend generation cannot inherit stale presentation", () => {
  const existing = {
    label: "MTP companion",
    filename: "old-mtp.gguf",
    expectedBytes: 20,
  };
  const carryExisting = carriesOverSeed(true, 1, 2);
  assert.equal(carryExisting, false);
  assert.equal(
    presentationForJobStart(undefined, existing, 120, carryExisting),
    undefined,
  );
  assert.match(
    readSrc("features/hub/download-manager/poll-loop.ts"),
    /presentationForJobStart\([\s\S]*?carryOverSeed,\s*\)/,
  );
});

test("a later plan-total increase cannot move companion progress backwards", () => {
  const presentation = presentationForJobStart(
    {
      label: "MTP companion",
      filename: "mtp-shared-Q8_0.gguf",
      expectedBytes: 20,
    },
    undefined,
    120,
    false,
  );
  assert.deepEqual(
    presentedProgress({
      expectedBytes: 200,
      downloadedBytes: 105,
      fraction: 0.525,
      presentation,
    }),
    { expectedBytes: 20, downloadedBytes: 5, fraction: 0.25 },
  );
});

test("poll totals freeze the first authoritative prefix across later growth", () => {
  const initial = {
    label: "MTP companion",
    filename: "mtp-shared-Q8_0.gguf",
    expectedBytes: 20,
  };
  const first = presentationForExpectedBytesUpdate(initial, 0, 120);
  const grown = presentationForExpectedBytesUpdate(first, 120, 200);
  assert.deepEqual(first, { ...initial, cachedPlanPrefixBytes: 100 });
  assert.deepEqual(grown, first);
  assert.deepEqual(
    presentedProgress({
      expectedBytes: 200,
      downloadedBytes: 105,
      fraction: 0.525,
      presentation: grown,
    }),
    { expectedBytes: 20, downloadedBytes: 5, fraction: 0.25 },
  );
  assert.match(
    readSrc("features/hub/download-manager/poll-loop.ts"),
    /applyProgressUpdate[\s\S]*?presentationForExpectedBytesUpdate\(/,
  );
});

test("Models Hub forwards companion presentation on start and update", () => {
  const card = readSrc("features/hub/catalog/gguf-download-card.tsx");
  assert.match(card, /presentation:\s*selectedPresentation/);
  assert.match(card, /pendingDrafterPresentation\(updateTargetVariant\)/);
  assert.match(card, /presentation\s*\?\s*\{ presentation \}/);
});
