// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The bar renders the stored etaSeconds rather than deriving one, so a total
// that grows underneath it leaves an ETA measured against the old, smaller one.
// GGUF metadata arriving after a job was adopted does exactly that. Hiding the
// ETA for one poll beats showing one for a size that is no longer the target.
//
// Also pins that the poll loop ignores a hidden tab's throttled callbacks: its
// gaps time the poller, not the transfer, and would read as the burst rate.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();
Object.assign(globalThis.window, { addEventListener: () => {} });

const { getState, putJob, jobKeyOf, setExpectedBytesForJob } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);

const KEY = jobKeyOf("model", "org/gguf-repo", null);

test("a total that grows drops the ETA measured against the old one", () => {
  putJob({
    key: KEY,
    kind: "model",
    repoId: "org/gguf-repo",
    variant: null,
    state: "running",
    downloadedBytes: 900_000_000,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 1_000_000_000,
    fraction: 0.9,
    bytesPerSec: 10_000_000,
    // 10s left, against a 1 GB total.
    etaSeconds: 10,
    error: null,
    startedAt: 1,
  });

  // GGUF metadata lands: the job is really 8 GB, not 1 GB.
  setExpectedBytesForJob("model", "org/gguf-repo", null, 8_000_000_000);

  const job = getState().jobs[KEY];
  assert.equal(job.expectedBytes, 8_000_000_000);
  assert.equal(
    job.etaSeconds,
    0,
    "the ETA for the old total must not survive the total changing",
  );
});

test("the poll loop drops a hidden tab's throttled samples", () => {
  // Voice used to run a second estimator of its own for a progress bar beside
  // the shared panel. Both are gone, so this is the only one left to guard.
  const voice = readFileSync(
    new URL("../src/features/settings/tabs/voice-tab.tsx", import.meta.url),
    "utf8",
  );
  assert.ok(
    !voice.includes("downloadSamplesRef"),
    "voice no longer estimates a rate of its own",
  );

  const source = readFileSync(
    new URL("../src/features/hub/download-manager/poll-loop.ts", import.meta.url),
    "utf8",
  );
  const guard = source.indexOf("document.hidden");
  assert.ok(guard > 0, "the poll loop should skip a hidden tab");
  const clear = source.indexOf("rt.speedSamples.length = 0", guard);
  assert.ok(clear > 0, "a hidden tab should clear the samples");
  assert.ok(
    clear - guard < 120,
    "the hidden-tab branch must clear rather than sample",
  );
});
