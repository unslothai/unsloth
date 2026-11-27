// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { exportRemoteCodeConsent } from "../src/features/export/lib/remote-code-consent.ts";

test("a checkpoint export scans the checkpoint folder and starts untrusted", () => {
  assert.deepEqual(
    exportRemoteCodeConsent({
      sourceMode: "checkpoint",
      // The model-mode toggle may still read "local" from an earlier visit.
      modelSource: "local",
      source: "checkpoint-60",
      checkpointPath: "/studio/outputs/k2-horizon/checkpoint-60",
    }),
    {
      scanTarget: "/studio/outputs/k2-horizon/checkpoint-60",
      requiresTrustRemoteCode: false,
      trustByDefault: false,
    },
  );
});

test("an HF model export scans the repo and needs approval to trust it", () => {
  assert.deepEqual(
    exportRemoteCodeConsent({
      sourceMode: "model",
      modelSource: "hf",
      source: "IFM/K2-Horizon-3.7B",
      checkpointPath: null,
    }),
    {
      scanTarget: "IFM/K2-Horizon-3.7B",
      requiresTrustRemoteCode: true,
      trustByDefault: false,
    },
  );
});

test("a local model export stays trusted by default", () => {
  assert.deepEqual(
    exportRemoteCodeConsent({
      sourceMode: "model",
      modelSource: "local",
      source: "/models/k2",
      checkpointPath: null,
    }),
    { scanTarget: "/models/k2", requiresTrustRemoteCode: false, trustByDefault: true },
  );
});
