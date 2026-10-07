// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  shouldRecheckSttReplacement,
  SttDownloadTrackers,
  sttReplacementAction,
} from "../src/features/settings/lib/stt-download-trackers.ts";

test("a second model's download does not stop the first", () => {
  const trackers = new SttDownloadTrackers();
  const stopped: string[] = [];

  trackers.start("qwen3-asr-0.6b", () => stopped.push("qwen3-asr-0.6b"));
  trackers.start("whisper-small", () => stopped.push("whisper-small"));

  // Each engine has its own download state, so both transfers are still live.
  assert.deepEqual(stopped, []);
  assert.equal(trackers.has("qwen3-asr-0.6b"), true);
  assert.equal(trackers.has("whisper-small"), true);
});

test("restarting the same model replaces its poller", () => {
  const trackers = new SttDownloadTrackers();
  const stopped: string[] = [];

  trackers.start("whisper-small", () => stopped.push("first"));
  trackers.start("whisper-small", () => stopped.push("second"));

  assert.deepEqual(stopped, ["first"], "the old interval would keep polling");
  assert.equal(trackers.has("whisper-small"), true);
});

test("stopping one model leaves the others tracked", () => {
  const trackers = new SttDownloadTrackers();
  const stopped: string[] = [];

  trackers.start("qwen3-asr-0.6b", () => stopped.push("qwen3-asr-0.6b"));
  trackers.start("whisper-small", () => stopped.push("whisper-small"));
  trackers.stop("whisper-small");

  assert.deepEqual(stopped, ["whisper-small"]);
  assert.equal(trackers.has("whisper-small"), false);
  assert.equal(trackers.has("qwen3-asr-0.6b"), true);
});

test("stopping an untracked model is a no-op", () => {
  const trackers = new SttDownloadTrackers();
  trackers.stop("whisper-small");
  assert.equal(trackers.has("whisper-small"), false);
});

test("an authoritative replacement survives its previous tracker settling", () => {
  assert.equal(
    sttReplacementAction(false, undefined, "attempt-x", "attempt-y"),
    "track",
  );
});

test("replacement confirmation cannot displace a newer local attempt", () => {
  assert.equal(
    sttReplacementAction(true, "attempt-z", "attempt-x", "attempt-y"),
    "retry",
  );
  assert.equal(
    sttReplacementAction(true, "attempt-y", "attempt-x", "attempt-y"),
    "ignore",
  );
});

test("a transient confirmation failure retries after the prior tracker settles", () => {
  assert.equal(shouldRecheckSttReplacement(undefined, "attempt-y"), true);
  assert.equal(
    shouldRecheckSttReplacement("attempt-x", "attempt-y"),
    true,
  );
  assert.equal(
    shouldRecheckSttReplacement("attempt-y", "attempt-y"),
    false,
  );
});
