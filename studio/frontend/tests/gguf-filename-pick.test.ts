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

// A repo's second build at one quant is advertised under a qualified key (`model-Q4_K_M-mtp`),
// and a saved bare label is the legacy spelling the backend's download and load paths still
// accept for a LONE such build. Refusing it here left "Pick a quantization" on a hint that
// resolves everywhere else.
const TAGGED = {
  filename: "model-Q4_K_M-mtp.gguf",
  quant: "model-Q4_K_M-mtp",
  downloaded: true,
};
const TAGGED_FP16 = {
  filename: "model-Q4_K_M-fp16.gguf",
  quant: "model-Q4_K_M-fp16",
  downloaded: true,
};
const PLAIN = {
  filename: "model-Q4_K_M.gguf",
  quant: "Q4_K_M",
  downloaded: false,
};

test("a bare label reaches the lone tagged build advertised under a qualified key", () => {
  assert.equal(pickGgufFilename([TAGGED, Q8], "Q4_K_M"), TAGGED.filename);
  assert.equal(pickGgufFilename([TAGGED, Q8], "q4_k_m"), TAGGED.filename);
});

test("a bare label that two tagged builds carry names neither", () => {
  assert.equal(pickGgufFilename([TAGGED, TAGGED_FP16], "Q4_K_M"), null);
});

test("a plain row that owns the label wins over its tagged sibling", () => {
  assert.equal(pickGgufFilename([TAGGED, PLAIN], "Q4_K_M"), PLAIN.filename);
});

test("a bit-width modifier is part of the token, not a tag past it", () => {
  const bpw = {
    filename: "model-IQ4_XS-3.53bpw.gguf",
    quant: "IQ4_XS-3.53bpw",
    downloaded: true,
  };
  assert.equal(pickGgufFilename([bpw], "IQ4_XS"), null);
});

test("a label that is only the start of the tagged build's token names nothing", () => {
  assert.equal(pickGgufFilename([TAGGED], "Q4_K"), null);
  const ud = {
    filename: "model-UD-Q4_K_XL-mtp.gguf",
    quant: "model-UD-Q4_K_XL-mtp",
    downloaded: true,
  };
  assert.equal(pickGgufFilename([ud], "Q4_K_XL"), null);
  assert.equal(pickGgufFilename([ud], "UD-Q4_K_XL"), ud.filename);
});

test("an H3 denoiser partition never answers to its bare quant", () => {
  const fl2va = {
    filename: "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
    quant: "minimax_h3_fl2va_pruned-UD-Q2_K_XL",
    downloaded: true,
  };
  assert.equal(pickGgufFilename([fl2va], "UD-Q2_K_XL"), null);
});

test("a build keyed by its parent directory answers to that directory's token", () => {
  const nested = {
    filename: "Q6_K/model-3.5bpw.gguf",
    quant: "Q6_K/model-3.5bpw",
    downloaded: true,
  };
  assert.equal(pickGgufFilename([nested], "Q6_K-3.5bpw"), nested.filename);
  assert.equal(pickGgufFilename([nested], "Q6_K"), null);
});

test("a tagged root build outranks a subordinate checkpoint sharing its quant", () => {
  const distilled = {
    filename: "distilled/model-Q4_K_M.gguf",
    quant: "distilled/model-Q4_K_M",
    downloaded: true,
  };
  assert.equal(
    pickGgufFilename([distilled, TAGGED], "Q4_K_M"),
    TAGGED.filename,
  );
  // A quant-named directory leaves a build at the root, so it ties with the tagged root.
  const quantDir = {
    filename: "Q4_K_M/model-a.gguf",
    quant: "Q4_K_M/model-a",
    downloaded: true,
  };
  assert.equal(pickGgufFilename([quantDir, TAGGED], "Q4_K_M"), null);
  assert.equal(
    pickGgufFilename([distilled, TAGGED, TAGGED_FP16], "Q4_K_M"),
    null,
  );
});
