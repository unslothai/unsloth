// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The bar renders the stored etaSeconds rather than deriving one, so a total
// that grows underneath it leaves an ETA measured against the old, smaller one.
// GGUF metadata arriving after a job was adopted does exactly that. Hiding the
// ETA for one poll beats showing one for a size that is no longer the target.
//
// Also pins that the voice poller ignores a hidden tab's throttled callbacks,
// the way the hub poll loop already does.

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

test("the voice poller drops a hidden tab's throttled samples", () => {
  const source = readFileSync(
    new URL("../src/features/settings/tabs/voice-tab.tsx", import.meta.url),
    "utf8",
  );
  const guard = source.indexOf("document.hidden");
  assert.ok(guard > 0, "the voice poller should skip a hidden tab");
  const clear = source.indexOf("downloadSamplesRef.current.length = 0", guard);
  const sample = source.indexOf(
    "appendSample(downloadSamplesRef.current",
    guard,
  );
  assert.ok(clear > 0, "a hidden tab should clear the samples");
  assert.ok(
    sample < 0 || clear < sample,
    "the hidden-tab branch must clear rather than sample",
  );
});
