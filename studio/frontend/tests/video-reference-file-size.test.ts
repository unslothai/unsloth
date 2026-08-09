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

test("a file at the cap still encodes to a data URL the backend accepts", () => {
  // models/inference.py bounds the STRING, not the file: 96 MiB for the video field and 32 MiB
  // for the soundtrack. FileReader emits `data:<mime>;base64,` plus 4 characters per 3 bytes, so
  // a raw cap of exactly three quarters encodes to exactly the limit and the prefix puts it over.
  // The picker accepted such a file and request validation then rejected it.
  const caps = { video: 96 * 1024 * 1024, audio: 32 * 1024 * 1024 } as const;
  for (const kind of ["video", "audio"] as const) {
    const raw = MAX_REFERENCE_BYTES[kind];
    // One of the longer MIME strings the OS can report, not the friendly mp4 case.
    const prefix = `data:${kind}/x-matroska;base64,`.length;
    assert.ok(
      Math.ceil(raw / 3) * 4 + prefix <= caps[kind],
      `${kind}: ${Math.ceil(raw / 3) * 4 + prefix} exceeds ${caps[kind]}`,
    );
    // Three quarters exactly would have failed that, so this is the assertion that moved.
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
    // Length rather than deepEqual against []: node's deepEqual is an assertion
    // signature, so comparing to an empty literal narrows loaded to never[] and
    // the audio push below stops typechecking.
    assert.equal(loaded.length, 0);
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
