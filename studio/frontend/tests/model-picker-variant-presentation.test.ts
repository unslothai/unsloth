// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ggufQuantChipLabel,
  ggufQuantDetailLabel,
  ggufVariantPickerLabel,
  groupGgufVariantsForPicker,
  h3PickerHasOnlyPrunedBuilds,
  preferredGgufVariantByGroup,
} from "../src/features/model-picker/components/model-selector/variant-presentation.ts";

const GGUF_SUFFIX = /\.gguf$/i;

const variant = (filename: string, size: number) => ({
  filename,
  quant: filename.replace(GGUF_SUFFIX, ""),
  display_label: filename.includes("ref2va")
    ? "References · Q4_K · Pruned"
    : "Text & frames · Q4_K · Pruned",
  size_bytes: size,
});

test("MiniMax H3 variants are grouped by generation workflow", () => {
  const textQ8 = variant("minimax_h3_fl2va_pruned-Q8_0.gguf", 20);
  const referenceQ8 = variant("minimax_h3_ref2va_pruned-Q8_0.gguf", 19);
  const textQ4 = variant("minimax_h3_fl2va_pruned-Q4_K.gguf", 11);
  const referenceQ4 = variant("minimax_h3_ref2va_pruned-Q4_K.gguf", 10);

  const groups = groupGgufVariantsForPicker([
    textQ8,
    referenceQ8,
    textQ4,
    referenceQ4,
  ]);

  assert.deepEqual(
    groups.map(({ key, title, description, variants }) => ({
      key,
      title,
      description,
      files: variants.map((entry) => entry.filename),
    })),
    [
      {
        key: "text-frames",
        title: "Text / first and last frames",
        description:
          "Generate from a prompt, optionally using first and last frame images.",
        files: [textQ8.filename, textQ4.filename],
      },
      {
        key: "reference-media",
        title: "Reference media",
        description: "Generate using reference images, video, or audio.",
        files: [referenceQ8.filename, referenceQ4.filename],
      },
    ],
  );
});

test("MiniMax H3 rows show only their quantization", () => {
  const pruned = variant("minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf", 9);
  const full = variant("minimax_h3_ref2va-Q4_K_M.gguf", 11);

  assert.equal(
    ggufVariantPickerLabel(pruned, {
      h3Grouped: true,
      hideH3PrunedBuild: true,
    }),
    "UD-Q3_K_XL",
  );
  assert.equal(
    ggufVariantPickerLabel(full, {
      h3Grouped: true,
      hideH3PrunedBuild: true,
    }),
    "Q4_K_M · Full",
  );
});

test("MiniMax H3 rows omit shard suffixes from quantization labels", () => {
  const sharded = variant(
    "minimax_h3_ref2va_pruned-Q4_K_M-00001-of-00002.gguf",
    11,
  );

  assert.equal(
    ggufVariantPickerLabel(sharded, {
      h3Grouped: true,
      hideH3PrunedBuild: true,
    }),
    "Q4_K_M",
  );
  assert.equal(
    groupGgufVariantsForPicker([sharded])[0]?.key,
    "reference-media",
  );
});

test("Pruned is hidden only when every visible H3 build is pruned", () => {
  const textPruned = variant("minimax_h3_fl2va_pruned-Q4_K_M.gguf", 11);
  const referencePruned = variant("minimax_h3_ref2va_pruned-Q4_K_M.gguf", 10);
  const full = variant("minimax_h3_ref2va-Q4_K_M.gguf", 18);

  assert.equal(
    h3PickerHasOnlyPrunedBuilds([textPruned, referencePruned]),
    true,
  );
  assert.equal(h3PickerHasOnlyPrunedBuilds([textPruned, full]), false);
  assert.equal(
    ggufVariantPickerLabel(textPruned, { h3Grouped: true }),
    "Q4_K_M · Pruned",
  );
});

test("generic and mixed repositories retain their existing presentation", () => {
  const generic = {
    filename: "model-Q4_K_M.gguf",
    quant: "Q4_K_M",
    display_label: "Q4_K_M · distilled",
    size_bytes: 4,
  };
  const h3 = variant("minimax_h3_fl2va_pruned-Q4_K.gguf", 11);

  assert.equal(ggufVariantPickerLabel(generic), "Q4_K_M · distilled");
  assert.deepEqual(groupGgufVariantsForPicker([h3, generic]), [
    {
      key: "quantizations",
      title: null,
      description: null,
      variants: [h3, generic],
    },
  ]);
});

test("each workflow prefers the same quantization and build as the default", () => {
  const textRecommended = variant("minimax_h3_fl2va-Q4_K_M.gguf", 20_000);
  const referenceMatching = variant("minimax_h3_ref2va-Q4_K_M.gguf", 30_000);
  const referenceCloser = variant("minimax_h3_ref2va_pruned-Q6_K.gguf", 20_001);
  const groups = groupGgufVariantsForPicker([
    referenceCloser,
    textRecommended,
    referenceMatching,
  ]);

  const preferred = preferredGgufVariantByGroup(groups, textRecommended.quant);

  assert.equal(preferred.get("text-frames"), textRecommended);
  assert.equal(preferred.get("reference-media"), referenceMatching);
});

test("each workflow uses the nearest-size fallback for different quant names", () => {
  const textRecommended = variant(
    "minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf",
    9_559,
  );
  const referenceQ3 = variant("minimax_h3_ref2va_pruned-Q3_K.gguf", 8_716);
  const referenceQ4 = variant("minimax_h3_ref2va_pruned-Q4_K.gguf", 11_381);
  const groups = groupGgufVariantsForPicker([
    referenceQ4,
    textRecommended,
    referenceQ3,
  ]);

  const preferred = preferredGgufVariantByGroup(groups, textRecommended.quant);

  assert.equal(preferred.get("text-frames"), textRecommended);
  assert.equal(preferred.get("reference-media"), referenceQ3);
});

test("an H3 quant chip is the quant alone, because the column is capped", () => {
  // The column fits UD-Q4_K_XL and no more.
  assert.equal(ggufQuantChipLabel("minimax_h3_fl2va_pruned-Q4_K_M"), "Q4_K_M");
  assert.equal(
    ggufQuantChipLabel("minimax_h3_ref2va-UD-Q3_K_XL"),
    "UD-Q3_K_XL",
  );
  // A suffix still on, and a shard counter, read alike.
  assert.equal(
    ggufQuantChipLabel("minimax_h3_ref2va_pruned-Q4_K_M.gguf"),
    "Q4_K_M",
  );
  assert.equal(
    ggufQuantChipLabel("minimax_h3_fl2va_pruned-Q4_K_M-00001-of-00002"),
    "Q4_K_M",
  );
});

test("the tooltip keeps the workflow the chip has no room for", () => {
  assert.equal(
    ggufQuantDetailLabel("minimax_h3_fl2va_pruned-Q4_K_M"),
    "Text & frames · Q4_K_M · Pruned",
  );
  assert.equal(
    ggufQuantDetailLabel("minimax_h3_ref2va-UD-Q3_K_XL"),
    "References · UD-Q3_K_XL · Full",
  );
});

test("an ordinary quant key is left exactly as it is", () => {
  for (const label of [ggufQuantChipLabel, ggufQuantDetailLabel]) {
    assert.equal(label("Q4_K_M"), "Q4_K_M");
    assert.equal(label("UD-Q2_K_XL"), "UD-Q2_K_XL");
    assert.equal(label("distilled-1.1/Q4_K_M"), "distilled-1.1/Q4_K_M");
    assert.equal(
      label("qwen3vl_32b_minimax_h3-Q4_K_M"),
      "qwen3vl_32b_minimax_h3-Q4_K_M",
    );
  }
});

test("the picker label is unchanged by the key-shaped parse", () => {
  // The optional suffix must not change what a ROW reads under its heading.
  const pruned = variant("minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf", 9);
  assert.equal(
    ggufVariantPickerLabel(pruned, {
      h3Grouped: true,
      hideH3PrunedBuild: true,
    }),
    "UD-Q3_K_XL",
  );
  assert.equal(groupGgufVariantsForPicker([pruned])[0]?.key, "text-frames");
});
