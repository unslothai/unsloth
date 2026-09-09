// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type PrefetchState = "none" | "ready" | "noop" | "partial" | "stale";

interface PrefetchStatus {
  state: PrefetchState;
  backendVersion: string | null;
  shellVersion: string | null;
  cacheDir: string | null;
  createdAt: number | null;
  running: boolean;
  runningShellVersion: string | null;
}

interface UpdatePreparation {
  shell: "pending" | "downloading" | "done" | "failed";
  backend: "pending" | "prefetching" | "ready" | "failed" | "skipped";
  shellProgress: number;
}

const preparation = loadWithStubs<{
  INITIAL_PREPARATION: UpdatePreparation;
  preparationStatus: (p: UpdatePreparation) => string;
  prefetchDecision: (args: {
    inApp: boolean;
    isExternalServer: boolean;
    offeredVersion: string;
    prefetch: PrefetchStatus;
  }) => string;
  desktopDownloadDecision: (
    status: { version: string | null; downloaded: boolean; downloading: boolean },
    offeredVersion: string,
  ) => string;
  preparationShortLabel: (p: UpdatePreparation) => string;
}>(new URL("../src/lib/update-preparation.ts", import.meta.url), {
  "@/lib/tauri-updater": {
    // The shipped predicate, which tolerates a leading "v" on either side.
    sameUpdateVersion: (left: string | null | undefined, right: string) =>
      left ? left.replace(/^v/, "") === right.replace(/^v/, "") : false,
  },
});

function status(overrides: Partial<PrefetchStatus> = {}): PrefetchStatus {
  return {
    state: "none",
    backendVersion: null,
    shellVersion: null,
    cacheDir: null,
    createdAt: null,
    running: false,
    runningShellVersion: null,
    ...overrides,
  };
}

function decide(prefetch: PrefetchStatus, offeredVersion = "2.0.0") {
  return preparation.prefetchDecision({
    inApp: true,
    isExternalServer: false,
    offeredVersion,
    prefetch,
  });
}

test("nothing on disk means a prefetch has to be started", () => {
  assert.equal(decide(status()), "prefetch");
});

test("a finished prefetch for the offered version is adopted as it stands", () => {
  for (const state of ["ready", "noop", "partial"] as const) {
    assert.equal(
      decide(status({ state, shellVersion: "2.0.0" })),
      "already-ready",
      state,
    );
  }
  // The leading "v" is a spelling, not a different release.
  assert.equal(
    decide(status({ state: "ready", shellVersion: "v2.0.0" })),
    "already-ready",
  );
});

test("a stale marker is not treated as a warm cache", () => {
  assert.equal(decide(status({ state: "stale", shellVersion: "2.0.0" })), "prefetch");
});

test("a finished prefetch for another version is redone", () => {
  assert.equal(decide(status({ state: "ready", shellVersion: "1.9.0" })), "prefetch");
});

test("a running prefetch is joined when it is preparing this offer", () => {
  assert.equal(
    decide(status({ running: true, runningShellVersion: "2.0.0" })),
    "adopt",
  );
});

test("a running prefetch for an older offer is stopped and started again", () => {
  assert.equal(
    decide(status({ running: true, runningShellVersion: "1.9.0" })),
    "restart",
  );
  // A reload that lost the record leaves no version to compare, which is the
  // same situation: the run in flight is not known to be preparing this offer.
  assert.equal(decide(status({ running: true })), "restart");
});

test("a running prefetch outranks the marker its previous run left", () => {
  assert.equal(
    decide(
      status({
        state: "ready",
        shellVersion: "1.9.0",
        running: true,
        runningShellVersion: "2.0.0",
      }),
    ),
    "adopt",
  );
});

test("there is nothing to prepare off the in-app path or against another server", () => {
  assert.equal(
    preparation.prefetchDecision({
      inApp: false,
      isExternalServer: false,
      offeredVersion: "2.0.0",
      prefetch: status(),
    }),
    "skip",
  );
  assert.equal(
    preparation.prefetchDecision({
      inApp: true,
      isExternalServer: true,
      offeredVersion: "2.0.0",
      prefetch: status(),
    }),
    "skip",
  );
});

test("the offer is ready once the app bundle is down, whatever the backend did", () => {
  const shellDone = { ...preparation.INITIAL_PREPARATION, shell: "done" as const };

  assert.equal(preparation.preparationStatus(preparation.INITIAL_PREPARATION), "preparing");
  assert.equal(preparation.preparationStatus(shellDone), "preparing");
  for (const backend of ["ready", "failed", "skipped"] as const) {
    assert.equal(
      preparation.preparationStatus({ ...shellDone, backend }),
      "ready",
      backend,
    );
  }
  assert.equal(
    preparation.preparationStatus({ ...shellDone, backend: "prefetching" }),
    "preparing",
  );
});

test("a failed app download puts the offer back on the ordinary Update button", () => {
  assert.equal(
    preparation.preparationStatus({
      shell: "failed",
      backend: "ready",
      shellProgress: 0,
    }),
    "available",
  );
});

test("the bundle decision separates a finished download from one in flight", () => {
  assert.equal(
    preparation.desktopDownloadDecision(
      { version: "2.0.0", downloaded: true, downloading: false },
      "2.0.0",
    ),
    "ready",
  );
  // Downloaded, but not this version: it has to be fetched again.
  assert.equal(
    preparation.desktopDownloadDecision(
      { version: "1.9.0", downloaded: true, downloading: false },
      "2.0.0",
    ),
    "download",
  );
  assert.equal(
    preparation.desktopDownloadDecision(
      { version: null, downloaded: false, downloading: true },
      "2.0.0",
    ),
    "wait",
  );
});

test("the pill line names whichever half is still working", () => {
  assert.equal(
    preparation.preparationShortLabel({
      shell: "downloading",
      backend: "pending",
      shellProgress: 42,
    }),
    "downloading 42%",
  );
  assert.equal(
    preparation.preparationShortLabel({
      shell: "done",
      backend: "prefetching",
      shellProgress: 100,
    }),
    "preparing packages",
  );
  assert.equal(
    preparation.preparationShortLabel(preparation.INITIAL_PREPARATION),
    "starting",
  );
});
