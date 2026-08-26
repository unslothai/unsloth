// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The "Partial" badge used to say "Click to continue" while being a status dot with no handler,
// and the button beside it said "Redownload" for a Xet partial even though completed shards are
// kept. Both read as a 56 GB refetch nobody asked for (issue #8927).
//
// The resume wording is driven by the backend's verdict on THIS partial, never by the installed
// huggingface_hub alone: a cache shared with a newer environment holds nonce-named partials that
// even a resuming writer will not reopen.

import assert from "node:assert/strict";
import test from "node:test";

import {
  downloadActionLabel,
  partialDownloadHint,
  partialResumeLabel,
} from "../src/features/hub/catalog/use-download-card-state.ts";

const RESUMABLE = true;

test("a partial the backend cannot resume is continued, never redownloaded", () => {
  // "Redownload" priced the whole repo. The transfer skips every completed file.
  assert.equal(partialResumeLabel(false), "Continue");
});

test("a resumable partial says so", () => {
  assert.equal(partialResumeLabel(RESUMABLE), "Resume");
});

test("an unstated verdict never promises a resume", () => {
  assert.equal(partialResumeLabel(), "Continue");
  assert.equal(partialResumeLabel(undefined), "Continue");
});

test("a repo with no partial is still a plain download", () => {
  assert.equal(downloadActionLabel(false, RESUMABLE), "Download");
  assert.equal(downloadActionLabel(true, RESUMABLE), "Resume");
  assert.equal(downloadActionLabel(true, false), "Continue");
});

test("the badge tooltip names the button, since the badge is not one", () => {
  for (const hint of [partialDownloadHint(RESUMABLE), partialDownloadHint(false)]) {
    assert.match(hint, /^Partial download\. Click (Continue|Resume) /);
  }
});

test("a restart-only partial leads with the file that starts over", () => {
  // A one-file quant has nothing to keep, so "files are kept" must not come first.
  const hint = partialDownloadHint(false);
  assert.match(hint, /Click Continue/);
  assert.match(
    hint,
    /The interrupted file starts over; other files already on disk are kept\.$/,
  );
});

test("a resumable partial promises the bytes on disk, and nothing more", () => {
  const hint = partialDownloadHint(RESUMABLE);
  assert.match(hint, /Click Resume/);
  assert.match(hint, /pick up where it stopped/);
});
