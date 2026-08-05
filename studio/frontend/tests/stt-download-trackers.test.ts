// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { SttDownloadTrackers } from "../src/features/settings/lib/stt-download-trackers.ts";

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
