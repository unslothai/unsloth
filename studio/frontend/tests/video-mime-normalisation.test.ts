// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A file accepted by extension can arrive as "" or octet-stream, and only ^video/ parts are
// sent as video, so an un-normalised type silently drops the clip.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isVideoFile, videoMimeForFile } = await import("../src/lib/video-utils.ts");

const file = (name: string, type: string) =>
  ({ name, type }) as unknown as File;

test("a browser that names the container is believed", () => {
  assert.equal(videoMimeForFile(file("clip.mp4", "video/mp4")), "video/mp4");
  assert.equal(
    videoMimeForFile(file("clip.mkv", "video/x-matroska")),
    "video/x-matroska",
  );
  assert.equal(videoMimeForFile(file("clip.mkv", "video/mp2t")), "video/mp2t");
});

test("an octet-stream is replaced by the container the extension names", () => {
  // Chromium on Windows with no codec pack registered.
  assert.equal(
    videoMimeForFile(file("clip.mkv", "application/octet-stream")),
    "video/x-matroska",
  );
  assert.equal(
    videoMimeForFile(file("holiday.MOV", "application/octet-stream")),
    "video/quicktime",
  );
});

test("an empty type is replaced too, which is the case that already worked", () => {
  assert.equal(videoMimeForFile(file("clip.mkv", "")), "video/x-matroska");
  assert.equal(videoMimeForFile(file("clip.avi", "")), "video/x-msvideo");
  assert.equal(videoMimeForFile(file("clip.webm", "")), "video/webm");
});

test("every extension the picker accepts normalises to a video type", () => {
  for (const ext of [".mp4", ".mov", ".webm", ".mkv", ".avi"]) {
    const picked = file(`clip${ext}`, "application/octet-stream");
    assert.ok(isVideoFile(picked), `${ext} is offered by the picker`);
    assert.match(
      videoMimeForFile(picked),
      /^video\//,
      `${ext} would be dropped by the request builder`,
    );
  }
});

test("a name with no known extension still sends something a video route accepts", () => {
  assert.equal(videoMimeForFile(file("clip", "application/octet-stream")), "video/mp4");
});
