// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { importRecipePayload } =
  await import("../src/features/recipe-studio/utils/import/importer.ts");
const { buildRecipePayload } =
  await import("../src/features/recipe-studio/utils/payload/build-payload.ts");
const { applyRecipeConnection } =
  await import("../src/features/recipe-studio/utils/graph/recipe-graph-connection.ts");
const { applyRenameToConfig, applyRemovalToConfig } =
  await import("../src/features/recipe-studio/stores/helpers/reference-sync.ts");
const { makeImageConfig, makeModelConfig, makeSamplerConfig } =
  await import("../src/features/recipe-studio/utils/config-factories.ts");
const { HANDLE_IDS } =
  await import("../src/features/recipe-studio/utils/handles.ts");
const { resolveImagePreview } =
  await import("../src/features/recipe-studio/utils/image-preview.ts");

const PRODUCT_PHOTOS = readFileSync(
  new URL(
    "../src/features/data-recipes/learning-recipes/product-photos.json",
    import.meta.url,
  ),
  "utf8",
);

function rebuild(input: string) {
  const imported = importRecipePayload(input);
  assert.deepEqual(imported.errors, []);
  assert.ok(imported.snapshot);
  const { configs, nodes, edges, processors, layoutDirection } =
    imported.snapshot;
  return {
    configs,
    built: buildRecipePayload(
      configs,
      nodes,
      edges,
      processors,
      layoutDirection,
    ),
  };
}

test("the product photos template rebuilds into the same image column and preset", () => {
  const source = JSON.parse(PRODUCT_PHOTOS).recipe;
  const { built } = rebuild(PRODUCT_PHOTOS);

  // As saved: JSON drops the builders' undefined fields.
  const saved = JSON.parse(JSON.stringify(built.payload.recipe));
  assert.deepEqual(built.errors, []);
  assert.deepEqual(saved.columns, source.columns);
  assert.deepEqual(saved.model_configs, source.model_configs);
});

test("an image preset and Image block survive a second save and reopen", () => {
  const first = rebuild(PRODUCT_PHOTOS).built.payload;
  const second = rebuild(JSON.stringify(first)).built.payload;

  assert.deepEqual(second.recipe, first.recipe);
});

test("an AI text step on an image preset, or an Image block on a text preset, is refused", () => {
  const recipe = JSON.parse(PRODUCT_PHOTOS);
  recipe.recipe.model_configs.push({
    alias: "writer",
    model: "unsloth/Qwen3-0.6B",
    provider: "Local Model",
  });
  recipe.recipe.columns.push({
    column_type: "llm-text",
    name: "caption",
    model_alias: "image_model",
    prompt: "Describe {{ subject }}",
  });
  recipe.recipe.columns[2].model_alias = "writer";

  const { built } = rebuild(JSON.stringify(recipe));

  assert.ok(
    built.errors.includes(
      "LLM caption: image_model is an image model preset; pick a text one.",
    ),
  );
  assert.ok(
    built.errors.includes(
      "Image photo: writer is a text model preset; pick an image one.",
    ),
  );
});

test("an Image block needs a prompt and an image preset", () => {
  const image = makeImageConfig("i1", []);
  const { built } = rebuild(
    JSON.stringify({
      recipe: {
        columns: [
          {
            column_type: "image",
            name: image.name,
            model_alias: "",
            prompt: "",
          },
        ],
      },
    }),
  );

  assert.ok(built.errors.includes("image_1: Prompt is required."));
  assert.equal(
    built.errors.filter((error) => /prompt/i.test(error)).length,
    1,
    "an empty prompt is reported once",
  );
  assert.ok(
    built.errors.some((error) =>
      error.includes("Choose an image model preset"),
    ),
  );
});

test("connecting an image preset sets the block's model and a field adds its reference", () => {
  const preset = { ...makeModelConfig("m1", []), generation: "image" as const };
  const subject = makeSamplerConfig("s1", "category", []);
  const image = makeImageConfig("i1", []);
  const configs = { m1: preset, s1: subject, i1: image };

  const withModel = applyRecipeConnection(
    {
      source: "m1",
      target: "i1",
      sourceHandle: HANDLE_IDS.semanticOut,
      targetHandle: HANDLE_IDS.dataIn,
    },
    configs,
    [],
  );
  assert.equal(withModel.edges[0].type, "semantic");
  const modelled = withModel.configs?.i1;
  assert.ok(modelled?.kind === "image");
  assert.equal(modelled.model_alias, preset.name);

  const withField = applyRecipeConnection(
    {
      source: "s1",
      target: "i1",
      sourceHandle: HANDLE_IDS.dataOut,
      targetHandle: HANDLE_IDS.dataIn,
    },
    { ...configs, i1: modelled },
    withModel.edges,
  );
  const prompted = withField.configs?.i1;
  assert.ok(prompted?.kind === "image");
  assert.equal(prompted.prompt, `{{ ${subject.name} }}`);
});

test("renaming or removing a field or preset updates the Image block", () => {
  const image = {
    ...makeImageConfig("i1", []),
    prompt: "A {{ style }} photo",
    model_alias: "painter",
  };

  const renamed = applyRenameToConfig(
    applyRenameToConfig(image, "style", "look"),
    "painter",
    "artist",
  );
  assert.ok(renamed.kind === "image");
  assert.equal(renamed.prompt, "A {{ look }} photo");
  assert.equal(renamed.model_alias, "artist");

  const removed = applyRemovalToConfig(renamed, "artist");
  assert.ok(removed.kind === "image");
  assert.equal(removed.model_alias, "");
});

test("a cell holding several generated images previews the first with its count", () => {
  const data =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

  assert.deepEqual(resolveImagePreview({ type: "image", data, count: 3 }), {
    kind: "ready",
    src: `data:image/jpeg;base64,${data}`,
    count: 3,
  });
  assert.equal(
    resolveImagePreview({ type: "image", data, count: 1 })?.kind,
    "ready",
  );
  assert.equal(
    "count" in (resolveImagePreview({ type: "image", data }) ?? {}),
    false,
  );
});
