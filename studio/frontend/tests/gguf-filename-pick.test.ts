// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { pickGgufFilename } from "../src/lib/gguf-filename-pick.ts";

const Q4 = {
  filename: "z-image-turbo-Q4_K_S.gguf",
  quant: "Q4_K_S",
  downloaded: true,
};
const Q8 = {
  filename: "z-image-turbo-Q8_0.gguf",
  quant: "Q8_0",
  downloaded: false,
};

test("a repo holding one quant names it without a label", () => {
  // The On Device row with a lone quant chip: the click carries only the repo.
  assert.equal(pickGgufFilename([Q4]), Q4.filename);
});

test("one downloaded quant wins over undownloaded siblings", () => {
  assert.equal(pickGgufFilename([Q4, Q8]), Q4.filename);
});

test("a quant label resolves to that quant's real filename", () => {
  // Filenames do not follow the repo name, so a label is never a filename.
  assert.equal(pickGgufFilename([Q4, Q8], "Q8_0"), Q8.filename);
  assert.equal(pickGgufFilename([Q4, Q8], "q8_0"), Q8.filename);
});

test("a label prefers the copy on disk when both are listed", () => {
  const remoteDup = {
    ...Q4,
    filename: "mirror/z-image.gguf",
    downloaded: false,
  };
  assert.equal(pickGgufFilename([remoteDup, Q4], "Q4_K_S"), Q4.filename);
});

test("a label that matches nothing does not fall back to the sole file", () => {
  // A pin left from a deleted quant must prompt, not load another quant.
  assert.equal(pickGgufFilename([Q4], "Q2_K"), null);
});

test("an exact filename passes through, normalised to the listing", () => {
  assert.equal(pickGgufFilename([Q4], Q4.filename), Q4.filename);
  assert.equal(
    pickGgufFilename([Q4], "Z-IMAGE-TURBO-Q4_K_S.GGUF"),
    Q4.filename,
  );
  // Unlisted (offline or failed listing): the caller's own name still routes.
  assert.equal(
    pickGgufFilename([], "some-model-Q6_K.gguf"),
    "some-model-Q6_K.gguf",
  );
});

test("several downloaded quants stay ambiguous", () => {
  const both = [Q4, { ...Q8, downloaded: true }];
  assert.equal(pickGgufFilename(both), null);
});

test("an empty or malformed listing resolves to nothing", () => {
  assert.equal(pickGgufFilename([]), null);
  assert.equal(pickGgufFilename([{ filename: 42, quant: null }]), null);
  // A companion .safetensors row is not a GGUF checkpoint.
  assert.equal(
    pickGgufFilename([{ filename: "model.safetensors", quant: "BF16" }]),
    null,
  );
});
