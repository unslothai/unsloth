// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  INITIAL_PREPARATION,
  backendIdle,
  desktopDownloadDecision,
  downloadPercent,
  preparationStatus,
  restartPlan,
  settleWithin,
  stagingDecision,
  waitForBackendIdle,
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

test("a reloaded renderer waits for the native shell download", () => {
  const status = {
    version: "0.1.901",
    downloaded: false,
    downloading: true,
  };
  assert.equal(desktopDownloadDecision(status, "0.1.901"), "wait");
  assert.equal(desktopDownloadDecision(status, "0.1.902"), "wait");
  assert.equal(
    desktopDownloadDecision({ ...status, downloading: false }, "0.1.901"),
    "download",
  );
  assert.equal(
    desktopDownloadDecision({ ...status, downloaded: true, downloading: false }, "v0.1.901"),
    "ready",
  );
});

function clock() {
  let now = 0;
  return {
    now: () => now,
    sleep: async (ms: number) => {
      now += ms;
    },
  };
}

test("the idle wait stops at the first idle probe", async () => {
  const c = clock();
  let probes = 0;
  const outcome = await waitForBackendIdle({
    cancelled: () => false,
    probe: async () => ++probes >= 3,
    sleep: c.sleep,
    now: c.now,
  });
  assert.equal(outcome, "idle");
  assert.equal(probes, 3);
});

test("an unreachable backend gives up instead of waiting forever", async () => {
  const c = clock();
  let probes = 0;
  const outcome = await waitForBackendIdle({
    cancelled: () => false,
    // What fetchHealth() returns when the backend is stopped or will not start.
    probe: async () => {
      probes += 1;
      return false;
    },
    sleep: c.sleep,
    now: c.now,
    pollMs: 20_000,
    timeoutMs: 60_000,
  });
  assert.equal(outcome, "timeout");
  assert.equal(probes, 4);
  // And the offer still settles, so the pill gets its Restart and the restart runs
  // the classic update rather than leaving the user with no way to update at all.
  assert.equal(preparationStatus(prep({ shell: "done", backend: "skipped" })), "ready");
  assert.equal(restartPlan(prep({ shell: "done", backend: "skipped" })), "classic");
});

test("a newer offer cancels the wait", async () => {
  const c = clock();
  let probes = 0;
  const outcome = await waitForBackendIdle({
    cancelled: () => probes >= 2,
    probe: async () => {
      probes += 1;
      return false;
    },
    sleep: c.sleep,
    now: c.now,
  });
  assert.equal(outcome, "cancelled");
});

test("a probe that never answers still settles, and aborts what it started", async () => {
  let aborted = false;
  // The wedged backend: the connection is accepted, the response never comes.
  const value = await settleWithin<string | null>(
    (signal) =>
      new Promise(() => {
        signal.addEventListener("abort", () => {
          aborted = true;
        });
      }),
    null,
    20,
  );
  assert.equal(value, null);
  assert.equal(aborted, true);
});

test("a probe that answers keeps its answer", async () => {
  const value = await settleWithin(async () => "ok", null, 1_000);
  assert.equal(value, "ok");
});

test("a probe that throws falls back", async () => {
  const value = await settleWithin(async () => {
    throw new Error("connection refused");
  }, null, 1_000);
  assert.equal(value, null);
});

test("an unanswerable backend still reaches the idle deadline", async () => {
  // The end-to-end shape of the P1: bounded probes let the wait's own deadline fire.
  let now = 0;
  let probes = 0;
  const outcome = await waitForBackendIdle({
    cancelled: () => false,
    probe: async () => {
      probes += 1;
      // Every probe burns its full timeout and answers "not idle".
      now += 10_000;
      return false;
    },
    sleep: async (ms: number) => {
      now += ms;
    },
    now: () => now,
    pollMs: 20_000,
    timeoutMs: 60_000,
  });
  assert.equal(outcome, "timeout");
  assert.ok(probes <= 4, `expected the deadline to stop the loop, ran ${probes} probes`);
});

test("only a running stage for the offered version is adopted", () => {
  // The webview reloaded while start_staged_update was running: the hook lost its
  // own record of it, and the native side rejects a second request.
  assert.equal(
    stagingDecision({
      inApp: true,
      isExternalServer: false,
      offeredVersion: "0.1.901",
      staged: staged({
        state: "partial",
        staging: true,
        stagingShellVersion: "0.1.901",
      }),
    }),
    "adopt",
  );
  assert.equal(
    stagingDecision({
      inApp: true,
      isExternalServer: false,
      offeredVersion: "0.1.902",
      staged: staged({
        state: "partial",
        staging: true,
        stagingShellVersion: "0.1.901",
      }),
    }),
    "wait",
  );
  // A partial stage with nothing running is leftover, and is restaged.
  assert.equal(
    stagingDecision({
      inApp: true,
      isExternalServer: false,
      offeredVersion: "0.1.901",
      staged: staged({ state: "partial" }),
    }),
    "stage",
  );
  // A finished stage for this offer is still reused rather than adopted.
  assert.equal(
    stagingDecision({
      inApp: true,
      isExternalServer: false,
      offeredVersion: "0.1.901",
      staged: staged({
        state: "ready",
        shellVersion: "0.1.901",
        staging: true,
        stagingShellVersion: "0.1.901",
      }),
    }),
    "already-ready",
  );
});
