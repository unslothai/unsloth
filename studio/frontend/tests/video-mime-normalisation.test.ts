// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The accept list carries extensions as well as mime types because the browser's answer is
// unreliable for mkv and some mov files. A file taken on its extension can arrive as "" or as
// application/octet-stream, and the request builder only recognises a file part whose mimeType
// matches ^video/, so an un-normalised type costs the clip silently: it is attached, it is sent,
// and the model answers as though nothing were there.

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
  // An unexpected video/* subtype is still a video type, so it is not rewritten.
  assert.equal(videoMimeForFile(file("clip.mkv", "video/mp2t")), "video/mp2t");
});

test("an octet-stream is replaced by the container the extension names", () => {
  // Chromium on a Windows box with no codec pack registered.
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
