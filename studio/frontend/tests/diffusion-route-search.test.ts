// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type DiffusionRouteSearch,
  diffusionRouteSearch,
} from "../src/lib/diffusion-route-search.ts";

const REPO = "unsloth/Z-Image-Turbo-GGUF";

test("an expander pick routes its exact filename as the quant", () => {
  assert.deepEqual(
    diffusionRouteSearch(REPO, {
      ggufFilename: "z-image-turbo-Q4_K_S.gguf",
      ggufVariant: "Q4_K_S",
    }),
    { model: REPO, quant: "z-image-turbo-Q4_K_S.gguf" },
  );
});

test("a pinned pick routes its label, and never as the quant", () => {
  // The target reads `quant` verbatim as a filename, so a label there routes a file that does not exist.
  const search: DiffusionRouteSearch = diffusionRouteSearch(REPO, {
    ggufVariant: "Q4_K_S",
  });
  // Asserted before the shape below: deepEqual narrows `search` to the literal it matched.
  assert.equal(search.quant, undefined);
  assert.deepEqual(search, { model: REPO, ggufQuant: "Q4_K_S" });
});

test("a non-catalog repo keeps its label too", () => {
  // The gap this closes: the page cannot recognise a GGUF repo it has no catalog entry for, so the label is the only evidence.
  assert.deepEqual(
    diffusionRouteSearch("QuantStack/SomeDiffusion-GGUF", {
      ggufVariant: "Q6_K",
    }),
    { model: "QuantStack/SomeDiffusion-GGUF", ggufQuant: "Q6_K" },
  );
});

test("a curated non-GGUF pick routes the model alone", () => {
  assert.deepEqual(diffusionRouteSearch("unsloth/FLUX.1-schnell", {}), {
    model: "unsloth/FLUX.1-schnell",
  });
});

test("blank metadata is dropped rather than routed", () => {
  assert.deepEqual(
    diffusionRouteSearch(REPO, { ggufFilename: "  ", ggufVariant: "" }),
    { model: REPO },
  );
  // A filename wins over a label, and both are trimmed.
  assert.deepEqual(
    diffusionRouteSearch(REPO, {
      ggufFilename: " a-Q8_0.gguf ",
      ggufVariant: " Q8_0 ",
    }),
    { model: REPO, quant: "a-Q8_0.gguf" },
  );
  assert.deepEqual(
    diffusionRouteSearch(REPO, { ggufFilename: null, ggufVariant: " Q8_0 " }),
    { model: REPO, ggufQuant: "Q8_0" },
  );
});
