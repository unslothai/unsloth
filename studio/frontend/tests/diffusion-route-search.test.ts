// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type DiffusionRouteSearch,
  diffusionRouteSearch,
  routedGgufFilename,
  routedGgufLabel,
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
  // The gap this closes: with no catalog entry for the repo, the label is the page's only evidence that it is GGUF.
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

test("an arrival naming a real file loads it, label or no label", () => {
  const search = { model: "m", quant: "a-Q8_0.gguf", ggufQuant: "Q4_K_S" };
  assert.equal(routedGgufFilename(search), "a-Q8_0.gguf");
  assert.equal(
    routedGgufLabel(search),
    null,
    "the exact filename wins and needs no listing",
  );
});

test("a label left in the filename slot is resolved, not posted", () => {
  // A hand-built link, or a producer predating the split: posting "Q4_K_S" as a filename is a certain error.
  assert.equal(routedGgufLabel({ quant: "Q4_K_S" }), "Q4_K_S");
  assert.equal(routedGgufFilename({ quant: "Q4_K_S" }), null);
});

test("an arrival with only the label slot resolves it", () => {
  assert.equal(routedGgufLabel({ ggufQuant: "Q6_K" }), "Q6_K");
  assert.equal(
    routedGgufFilename({}),
    null,
    "and nothing lands in the filename slot",
  );
});

test("an arrival with neither leaves the old path alone", () => {
  assert.equal(routedGgufLabel({}), null);
  assert.equal(routedGgufFilename({}), null);
  assert.equal(routedGgufLabel({ quant: "  ", ggufQuant: "" }), null);
});

test("case and whitespace do not change which slot a value belongs to", () => {
  assert.equal(routedGgufFilename({ quant: " A-Q8_0.GGUF " }), "A-Q8_0.GGUF");
  assert.equal(routedGgufLabel({ ggufQuant: " Q4_K_S " }), "Q4_K_S");
});
