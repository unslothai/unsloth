// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { savedLabel } from "../src/features/export/constants.ts";
import {
  catalogFileName,
  matchCatalogId,
} from "../src/features/images/train/diffusion-train-deploy.ts";

import { readSrcAsync } from "./helpers/kit.ts";

test("a published adapter is matched by its stem, or its full name when the stem collides", () => {
  assert.equal(
    catalogFileName(
      "/home/u/.unsloth/studio/loras/diffusion/sksdog.safetensors",
    ),
    "sksdog.safetensors",
  );
  assert.equal(
    catalogFileName("C:\\Users\\u\\loras\\diffusion\\my.style.gguf"),
    "my.style.gguf",
  );
  const st = (id: string) => ({ id, format: "safetensors" });
  const gg = (id: string) => ({ id, format: "gguf" });
  assert.equal(
    matchCatalogId([st("sksdog"), st("other")], "sksdog.safetensors"),
    "sksdog",
  );
  assert.equal(
    matchCatalogId([gg("s.gguf"), st("s.safetensors")], "s.safetensors"),
    "s.safetensors",
  );
  // A stale run whose safetensors is gone never picks the same-stem GGUF.
  assert.equal(matchCatalogId([gg("style")], "style.safetensors"), undefined);
  assert.equal(
    matchCatalogId([st("dog.safetensors"), st("dog")], "dog.safetensors"),
    "dog",
  );
  assert.equal(matchCatalogId([st("other")], "sksdog.safetensors"), undefined);
});

test("the export result names the exports folder unless the path is absolute", () => {
  assert.equal(
    savedLabel("image-loras/sksdog.safetensors"),
    "exports/image-loras/sksdog.safetensors",
  );
  assert.equal(
    savedLabel("/data/loras/sksdog.safetensors"),
    "/data/loras/sksdog.safetensors",
  );
  assert.equal(
    savedLabel("D:\\loras\\sksdog.safetensors"),
    "D:\\loras\\sksdog.safetensors",
  );
});

test("the finished training run hands its LoRA to the Export page", async () => {
  const panel = await readSrcAsync(
    "features/images/train/diffusion-train-panel.tsx",
  );
  assert.match(
    panel,
    /to: "\/export",\s*search: \{ imageLora: catalogFileName\(catalogPath\)\s*\}/,
  );
  assert.equal(panel.match(/exportAdapter\(/g)?.length, 2);
  // A run stopped early with a saved adapter can be exported too, not only a completed one.
  assert.doesNotMatch(panel, /completed && status\?\.catalog_path && \(/);
  const route = await readSrcAsync("app/routes/export.tsx");
  assert.match(route, /imageLora:\s*typeof search\.imageLora === "string"/);
  const page = await readSrcAsync("features/export/export-page.tsx");
  assert.match(
    page,
    /<ImageLoraExportCard preselect=\{preselectImageLora\} \/>/,
  );
});
