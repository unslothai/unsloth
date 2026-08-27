// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  INITIAL_PREPARATION,
  backendIdle,
  downloadPercent,
  preparationStatus,
  restartPlan,
  stagingDecision,
  type StagedUpdateStatus,
  type UpdatePreparation,
} from "../src/lib/update-preparation.ts";

function prep(patch: Partial<UpdatePreparation>): UpdatePreparation {
  return { ...INITIAL_PREPARATION, ...patch };
}

function staged(patch: Partial<StagedUpdateStatus>): StagedUpdateStatus {
  return { state: "none", backendVersion: null, shellVersion: null, ...patch };
}

test("nothing is ready until the app bundle is downloaded", () => {
  assert.equal(preparationStatus(INITIAL_PREPARATION), "preparing");
  assert.equal(preparationStatus(prep({ backend: "ready" })), "preparing");
  assert.equal(preparationStatus(prep({ shell: "downloading", backend: "ready" })), "preparing");
});

test("a downloaded bundle is ready once the backend has settled either way", () => {
  assert.equal(preparationStatus(prep({ shell: "done", backend: "ready" })), "ready");
  assert.equal(preparationStatus(prep({ shell: "done", backend: "failed" })), "ready");
  assert.equal(preparationStatus(prep({ shell: "done", backend: "skipped" })), "ready");
  assert.equal(preparationStatus(prep({ shell: "done", backend: "staging" })), "preparing");
  assert.equal(preparationStatus(prep({ shell: "done", backend: "waiting" })), "preparing");
});

test("a failed bundle download falls back to the plain offer", () => {
  assert.equal(preparationStatus(prep({ shell: "failed", backend: "ready" })), "available");
});

test("only a staged backend with a downloaded bundle restarts on the fast path", () => {
  assert.equal(restartPlan(prep({ shell: "done", backend: "ready" })), "fast");
  assert.equal(restartPlan(prep({ shell: "done", backend: "failed" })), "classic");
  assert.equal(restartPlan(prep({ shell: "done", backend: "skipped" })), "classic");
  assert.equal(restartPlan(prep({ shell: "downloading", backend: "ready" })), "classic");
});

test("staging is skipped outside the in-app updater and for external servers", () => {
  const base = { inApp: true, isExternalServer: false, offeredVersion: "0.1.900-beta", staged: staged({}) };
  assert.equal(stagingDecision({ ...base, inApp: false }), "skip");
  assert.equal(stagingDecision({ ...base, isExternalServer: true }), "skip");
  assert.equal(stagingDecision(base), "stage");
});

test("a stage already prepared for the offered version is reused", () => {
  const ready = staged({ state: "ready", shellVersion: "0.1.900-beta" });
  const args = { inApp: true, isExternalServer: false, offeredVersion: "v0.1.900-beta", staged: ready };
  assert.equal(stagingDecision(args), "already-ready");
  assert.equal(stagingDecision({ ...args, offeredVersion: "0.1.901-beta" }), "stage");
});

test("a version that rolled back is never staged again", () => {
  const failed = staged({ state: "failed", shellVersion: "0.1.900-beta" });
  const args = { inApp: true, isExternalServer: false, offeredVersion: "0.1.900-beta", staged: failed };
  assert.equal(stagingDecision(args), "skip");
  assert.equal(stagingDecision({ ...args, offeredVersion: "0.1.901-beta" }), "stage");
});

test("the backend is idle only when reachable, not generating and not training", () => {
  assert.equal(backendIdle(null, false), false);
  assert.equal(backendIdle({}, false), true);
  assert.equal(backendIdle({ inference_active: true }, false), false);
  assert.equal(backendIdle({}, true), false);
});

test("download progress clamps to a whole percentage", () => {
  assert.equal(downloadPercent(0, null), 0);
  assert.equal(downloadPercent(50, 200), 25);
  assert.equal(downloadPercent(300, 200), 100);
});
