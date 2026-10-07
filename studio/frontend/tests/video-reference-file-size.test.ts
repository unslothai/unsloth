// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The backend caps the base64 STRING (96 MiB video, 32 MiB soundtrack), so the raw file
 * limits are below three quarters of those. */

import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_REFERENCE_BYTES,
  readReferenceFile,
  referenceFileRejection,
} from "../src/features/video/reference-budget.ts";

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

test("a file at the cap still encodes to a data URL the backend accepts", () => {
  // A raw cap of exactly 3/4 encodes to exactly the limit, and the data-URL prefix tips it over.
  const caps = { video: 96 * 1024 * 1024, audio: 32 * 1024 * 1024 } as const;
  for (const kind of ["video", "audio"] as const) {
    const raw = MAX_REFERENCE_BYTES[kind];
    const prefix = `data:${kind}/x-matroska;base64,`.length;
    assert.ok(
      Math.ceil(raw / 3) * 4 + prefix <= caps[kind],
      `${kind}: ${Math.ceil(raw / 3) * 4 + prefix} exceeds ${caps[kind]}`,
    );
    assert.ok(Math.ceil((caps[kind] * 3) / 4 / 3) * 4 + prefix > caps[kind]);
  }
});

test("the headroom does not move the limit the user is shown", () => {
  assert.equal(Math.round(MAX_REFERENCE_BYTES.video / (1024 * 1024)), 72);
  assert.equal(Math.round(MAX_REFERENCE_BYTES.audio / (1024 * 1024)), 24);
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
    // deepEqual against [] narrows loaded to never[] and breaks the push below.
    assert.equal(loaded.length, 0);
    assert.equal(errors.length, 1);
    assert.match(errors[0], /too large \(limit 72 MB\)/);

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
