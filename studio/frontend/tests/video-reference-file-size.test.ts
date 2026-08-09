// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The reference picker used to read ANY file the MIME check let through straight to a base64 data
 * URL. The backend caps the base64 STRING (96 MiB for a reference video, 32 MiB for its
 * soundtrack), so the raw file limits are three quarters of those, and a data URL costs about
 * 2.33x the file in renderer memory before the 422 can arrive. H3 reference clips are 2 to 15
 * seconds by spec, so a 15 second 4K phone clip clears the cap routinely.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_REFERENCE_BYTES,
  readReferenceFile,
  referenceFileRejection,
} from "../src/features/video/reference-budget.ts";

/** A File stub: readReferenceFile only ever reads .type, .size and hands the object on. */
function fileStub(type: string, size: number): File {
  return { type, size, name: "clip.mp4" } as unknown as File;
}

function withFileReaderSpy<T>(run: (constructed: () => number) => T): T {
  let built = 0;
  const previous = (globalThis as { FileReader?: unknown }).FileReader;
  class SpyFileReader {
    result: string | null = null;
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    constructor() {
      built += 1;
    }
    readAsDataURL() {
      this.result = "data:video/mp4;base64,AAAA";
      this.onload?.();
    }
  }
  (globalThis as { FileReader?: unknown }).FileReader = SpyFileReader;
  try {
    return run(() => built);
  } finally {
    (globalThis as { FileReader?: unknown }).FileReader = previous;
  }
}

test("the raw file limits are three quarters of the backend's base64 caps", () => {
  assert.equal(MAX_REFERENCE_BYTES.video, 72 * 1024 * 1024);
  assert.equal(MAX_REFERENCE_BYTES.audio, 24 * 1024 * 1024);
});

test("an oversized reference is refused before a FileReader ever exists", () => {
  withFileReaderSpy((constructed) => {
    const loaded: (string | null)[] = [];
    const errors: string[] = [];

    readReferenceFile("video", fileStub("video/mp4", MAX_REFERENCE_BYTES.video + 1), {
      onLoaded: (dataUrl) => loaded.push(dataUrl),
      onError: (message) => errors.push(message),
    });

    assert.equal(constructed(), 0, "the file must not be read into memory at all");
    assert.deepEqual(loaded, []);
    assert.equal(errors.length, 1);
    assert.match(errors[0], /too large \(limit 72 MB\)/);

    // The soundtrack slot carries its own, smaller cap.
    errors.length = 0;
    readReferenceFile("audio", fileStub("audio/wav", MAX_REFERENCE_BYTES.audio + 1), {
      onLoaded: (dataUrl) => loaded.push(dataUrl),
      onError: (message) => errors.push(message),
    });
    assert.equal(constructed(), 0);
    assert.match(errors[0], /too large \(limit 24 MB\)/);
  });
});

test("a file inside the cap is still read normally", () => {
  withFileReaderSpy((constructed) => {
    const loaded: (string | null)[] = [];
    const errors: string[] = [];

    readReferenceFile("video", fileStub("video/mp4", MAX_REFERENCE_BYTES.video), {
      onLoaded: (dataUrl) => loaded.push(dataUrl),
      onError: (message) => errors.push(message),
    });

    assert.equal(constructed(), 1);
    assert.deepEqual(loaded, ["data:video/mp4;base64,AAAA"]);
    assert.deepEqual(errors, []);
  });
});

test("the wrong media kind is still refused first", () => {
  assert.equal(
    referenceFileRejection("video", { type: "image/png", size: 10 }),
    "Please choose a video file",
  );
  assert.equal(referenceFileRejection("audio", { type: "audio/wav", size: 10 }), null);
});
