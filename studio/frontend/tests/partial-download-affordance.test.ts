// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The "Partial" badge used to say "Click to continue" while being a status dot with no handler,
// and the button beside it said "Redownload" for a Xet partial even though completed shards are
// kept. Both read as a 56 GB refetch nobody asked for (issue #8927).

import assert from "node:assert/strict";
import test from "node:test";

import {
  downloadActionLabel,
  partialDownloadHint,
  partialIsResumable,
  partialResumeLabel,
} from "../src/features/hub/catalog/use-download-card-state.ts";

const RESUMABLE = true;

test("only an HTTP partial on a hub that reopens it can be resumed", () => {
  assert.equal(partialIsResumable("http", RESUMABLE), true);
  assert.equal(partialIsResumable("http", false), false);
  assert.equal(partialIsResumable("xet", RESUMABLE), false);
  assert.equal(partialIsResumable(null, RESUMABLE), false);
});

test("a Xet partial is continued, never redownloaded", () => {
  // "Redownload" priced the whole repo. The transfer skips every completed file.
  assert.equal(partialResumeLabel("xet", RESUMABLE), "Continue");
  assert.equal(partialResumeLabel("xet", false), "Continue");
});

test("a resumable HTTP partial says so", () => {
  assert.equal(partialResumeLabel("http", RESUMABLE), "Resume");
});

test("an HTTP partial no writer can reopen does not promise a resume", () => {
  assert.equal(partialResumeLabel("http", false), "Continue");
});

test("an unknown transport continues rather than retrying from scratch", () => {
  assert.equal(partialResumeLabel(null, RESUMABLE), "Continue");
  assert.equal(partialResumeLabel(undefined, false), "Continue");
});

test("a repo with no partial is still a plain download", () => {
  assert.equal(downloadActionLabel(false, "http", RESUMABLE), "Download");
  assert.equal(downloadActionLabel(true, "http", RESUMABLE), "Resume");
  assert.equal(downloadActionLabel(true, "xet", RESUMABLE), "Continue");
});

test("the badge tooltip names the button, since the badge is not one", () => {
  for (const hint of [
    partialDownloadHint("xet", RESUMABLE),
    partialDownloadHint("http", RESUMABLE),
    partialDownloadHint(null, false),
  ]) {
    assert.match(hint, /^Partial download\. Click (Continue|Resume) /);
  }
});

test("a restart-only partial says what survives it", () => {
  const hint = partialDownloadHint("xet", RESUMABLE);
  assert.match(hint, /Click Continue/);
  assert.match(hint, /Files already downloaded are kept/);
});

test("a resumable partial promises the bytes on disk, and nothing more", () => {
  const hint = partialDownloadHint("http", RESUMABLE);
  assert.match(hint, /Click Resume/);
  assert.match(hint, /pick up where it stopped/);
});
