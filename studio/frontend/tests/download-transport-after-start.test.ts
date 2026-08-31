// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readFileSync } from "node:fs";

import {
  adoptedTransports,
  mismatchStartAction,
  probeDescribesCurrentRun,
  TRANSPORT,
  transportAfterStart,
} from "../src/features/hub/download-manager/constants.ts";

test("a start of its own keeps the transport it resolved", () => {
  assert.equal(transportAfterStart("http", null), "http");
  assert.equal(transportAfterStart("xet", undefined), "xet");
});

test("adopting another client's run takes that run's transport", () => {
  // Requested HTTP, attached to a Xet job: Pause would promise a resume the
  // Xet worker cannot give.
  assert.equal(transportAfterStart("http", "xet"), "xet");
  assert.equal(transportAfterStart("xet", "http"), "http");
});

test("an unresolved or unknown report is ignored, not trusted", () => {
  assert.equal(transportAfterStart("http", "auto"), "http");
  assert.equal(transportAfterStart("xet", "ftp"), "xet");
  assert.equal(transportAfterStart("xet", 3), "xet");
});

test("a probe for the run a job is on may patch it", () => {
  assert.equal(probeDescribesCurrentRun(7, 7), true);
});

test("a probe from before a cancel and restart may not", () => {
  // The old response would otherwise hand the restarted job the old
  // transport, and the fresh start's own reply cannot repair it.
  assert.equal(probeDescribesCurrentRun(8, 7), false);
});

test("a job with no generation yet takes what the probe reports", () => {
  assert.equal(probeDescribesCurrentRun(undefined, 7), true);
  assert.equal(probeDescribesCurrentRun(null, 0), true);
});

test("a probe with no generation proves nothing", () => {
  assert.equal(probeDescribesCurrentRun(undefined, undefined), false);
  assert.equal(probeDescribesCurrentRun(7, undefined), false);
  assert.equal(probeDescribesCurrentRun(7, 7.5), false);
});

test("adoption keeps a persisted marker when the probe carries none", () => {
  // `/download-status` has no cancel marker and can win the hydration race
  // against `/active-downloads`, which does.
  assert.deepEqual(
    adoptedTransports(
      { transport: "http" },
      { transport: "http", cancelTransport: "xet" },
    ),
    { transport: "http", cancelTransport: "xet" },
  );
});

test("a probe that reports both wins over what was stored", () => {
  assert.deepEqual(
    adoptedTransports(
      { transport: "xet", cancelTransport: "http" },
      { transport: "http", cancelTransport: "xet" },
    ),
    { transport: "xet", cancelTransport: "http" },
  );
});

test("an adoption with nothing stored and nothing reported holds neither", () => {
  assert.deepEqual(adoptedTransports({}, undefined), {
    transport: undefined,
    cancelTransport: undefined,
  });
});

test("the persisted transport survives a probe that omits it too", () => {
  assert.deepEqual(
    adoptedTransports({ cancelTransport: "xet" }, { transport: "http" }),
    { transport: "http", cancelTransport: "xet" },
  );
});

test("a reported null clears a marker left by an earlier run", () => {
  // /active-downloads always carries the field, so null is the backend saying
  // this run has no marker, not a source that cannot report one.
  assert.deepEqual(
    adoptedTransports(
      { transport: "http", cancelTransport: null },
      { transport: "xet", cancelTransport: "xet" },
    ),
    { transport: "http", cancelTransport: undefined },
  );
});

test("a reported null with nothing stored still holds no marker", () => {
  assert.deepEqual(
    adoptedTransports({ transport: "http", cancelTransport: null }, undefined),
    { transport: "http", cancelTransport: undefined },
  );
});

test("auto continues over HTTP after a Xet stall leftover", () => {
  // Health demotes Auto to HTTP while the .transport marker is still xet.
  assert.equal(
    mismatchStartAction(TRANSPORT.AUTO, TRANSPORT.HTTP, TRANSPORT.XET),
    TRANSPORT.HTTP,
  );
});

test("auto resumes an HTTP partial instead of restarting on Xet", () => {
  // HTTP fallback already wrote the marker; Auto still resolving to Xet must
  // not send the user to the Hub conflict banner.
  assert.equal(
    mismatchStartAction(TRANSPORT.AUTO, TRANSPORT.XET, TRANSPORT.HTTP),
    TRANSPORT.HTTP,
  );
});

test("an explicit HTTP preference discards a Xet partial without a dialog", () => {
  assert.equal(
    mismatchStartAction(TRANSPORT.HTTP, TRANSPORT.HTTP, TRANSPORT.XET),
    TRANSPORT.HTTP,
  );
});

test("an explicit Xet preference still asks before throwing away HTTP bytes", () => {
  assert.equal(
    mismatchStartAction(TRANSPORT.XET, TRANSPORT.XET, TRANSPORT.HTTP),
    "conflict",
  );
});

test("an explicit Xet preference restarts an unresumable HTTP partial", () => {
  assert.equal(
    mismatchStartAction(TRANSPORT.XET, TRANSPORT.XET, TRANSPORT.HTTP, false),
    TRANSPORT.XET,
  );
});

test("unavailable Xet with an HTTP partial continues over HTTP", () => {
  // Preference is Xet, but the machine already demoted the resolved transport.
  assert.equal(
    mismatchStartAction(TRANSPORT.XET, TRANSPORT.HTTP, TRANSPORT.HTTP),
    TRANSPORT.HTTP,
  );
});

test("a matching pair is left alone", () => {
  assert.equal(
    mismatchStartAction(TRANSPORT.AUTO, TRANSPORT.XET, TRANSPORT.XET),
    TRANSPORT.XET,
  );
});

test("a transport mismatch start uses the mismatch helper", () => {
  const source = readFileSync(
    new URL(
      "../src/features/hub/download-manager/transport-conflict.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    source,
    /mismatchStartAction\(\s*preferred,\s*resolved,\s*last,\s*status\.resumable,\s*\)/,
  );
  assert.match(
    source,
    /siblingTransport !== mode &&\s*preferred !== TRANSPORT\.AUTO/,
  );
  assert.match(source, /mode = action;[\s\S]*?siblingTransport !== mode/);
});

test("staging owns cleanup while Hub resolves exact or scoped conflicts", () => {
  const source = readFileSync(
    new URL(
      "../src/features/hub/download-manager/use-repo-download.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const exactIndex = source.indexOf(
    "const exact = state.conflicts[conflictKey]",
  );
  const scopedIndex = source.indexOf(
    "const scoped = Object.entries(state.conflicts)",
  );
  assert.ok(exactIndex >= 0 && scopedIndex > exactIndex);
  assert.match(source, /__staged_download_idle__/);
  assert.match(source, /__hub_autoload_idle__/);
  assert.match(source, /preservedConflictKeyRef\.current = conflictKey/);
  assert.match(source, /cancelConflict\(preservedConflictKeyRef\.current\)/);
  assert.match(source, /resumeConflict\(visibleConflictKey\)/);
  assert.match(source, /restartConflict\(visibleConflictKey\)/);
  assert.match(source, /cancelConflict\(visibleConflictKey\)/);
  assert.match(
    source,
    /downloadManager\.cancelConflict\(conflictKey\);\s*\},\s*\[conflictKey\]/,
  );
});
