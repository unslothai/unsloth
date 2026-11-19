// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Images picker header names the precision the resident build runs at, the same value the Loaded build row and the
// loaded-models chip show: an INT8 load of the BF16 MOSS-Image row reads INT8, not BF16.

import assert from "node:assert/strict";
import test from "node:test";

import { describeDiffusionStatus } from "../src/features/loaded-models/loaded-models-sources.ts";
import { withLoadedPrecision } from "../src/features/model-picker/components/model-selector/loaded-precision-trigger.ts";
import {
  IMAGE_CATALOG,
  catalogToModelOptions,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import {
  denseTransformerBuildLabel,
  formatResolvedValue,
  loadedTransformerPrecisionLabel,
} from "../src/lib/resolved-precision.ts";
import { readSrc } from "./helpers/kit.ts";

const MOSS = "OpenMOSS-Team/MOSS-Image-SFT";

function status(extra: Record<string, unknown>) {
  return {
    loaded: true,
    repo_id: MOSS,
    family: "moss-image",
    base_repo: MOSS,
    device: "cuda",
    dtype: "bfloat16",
    model_kind: "pipeline",
    cpu_offload: false,
    transformer_quant: null,
    ...extra,
  } as never;
}

// The Loaded build row's Transformer value, as images-page LoadedBuildSummary computes it.
function settingsRow(s: { transformer_quant?: string | null; model_kind?: string | null; dtype?: string | null }) {
  return s.transformer_quant
    ? formatResolvedValue("transformer_quant", s.transformer_quant)
    : denseTransformerBuildLabel(s);
}

function headerFor(s: never, schemes: readonly string[] = ["int8"]) {
  const option = catalogToModelOptions(IMAGE_CATALOG, "dense-quant", schemes).find((o) => o.id === MOSS);
  assert.ok(option, "MOSS-Image row");
  return withLoadedPrecision(option, loadedTransformerPrecisionLabel(s));
}

test("an INT8 load of the BF16 MOSS-Image row reads INT8 in the header, settings and chip", () => {
  const s = status({ transformer_quant: "int8" });
  const header = headerFor(s);
  assert.equal(header.descriptionSuffix, "INT8");
  assert.doesNotMatch(`${header.name} ${header.descriptionSuffix}`, /BF16|FP8/);
  assert.equal(settingsRow(status({ transformer_quant: "int8" }) as never), "INT8");
  const chip = describeDiffusionStatus(s)[0]?.detail ?? "";
  assert.match(chip, /\bINT8\b/);
  assert.doesNotMatch(chip, /BF16/);
});

test("FP8 and BF16 loads name what runs", () => {
  assert.equal(headerFor(status({ transformer_quant: "fp8" })).descriptionSuffix, "FP8");
  const bf16 = headerFor(status({}));
  assert.equal(bf16.descriptionSuffix, "BF16");
  // The row's "(Fast FP8)" speed tier is not the loaded build.
  assert.doesNotMatch(bf16.name, /Fast/);
  assert.equal(settingsRow(status({}) as never), "BF16");
  assert.equal(headerFor(status({ dtype: "float16" })).descriptionSuffix, "FP16");
});

test("a GGUF keeps its quant, and a dense fast path on a GGUF pick reads the engaged quant", () => {
  assert.equal(
    loadedTransformerPrecisionLabel(status({ model_kind: "gguf", gguf_variant: "Q4_K_M" })),
    "GGUF · Q4_K_M",
  );
  assert.equal(
    loadedTransformerPrecisionLabel(status({ model_kind: "gguf", gguf_variant: "Q4_K_M", transformer_quant: "int8" })),
    "INT8",
  );
  assert.equal(loadedTransformerPrecisionLabel({ loaded: false, transformer_quant: "int8" }), null);
  assert.equal(loadedTransformerPrecisionLabel(status({ transformer_quant: "none" })), "BF16");
});

test("nothing known leaves the row label alone", () => {
  const row = { name: "MOSS-Image (Fast FP8)", descriptionSuffix: "BF16" };
  assert.deepEqual(withLoadedPrecision(row, null), row);
});

test("the Images page feeds the header from the backend status", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /loadedPrecisionOverride=\{[\s\S]{0,120}loadedTransformerPrecisionLabel\(status\)/);
  const selector = readSrc("features/model-picker/components/model-selector.tsx");
  // Ahead of the GGUF branch, so a dense build on a GGUF pick does not read as the GGUF.
  assert.ok(
    selector.indexOf("withLoadedPrecision(") < selector.indexOf("const desc = `GGUF · ${triggerGgufVariant}`"),
  );
});
