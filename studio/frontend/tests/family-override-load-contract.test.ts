// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  diffusionPipelineStagingEntries,
  diffusionPipelineLoadTarget,
} from "../src/lib/diffusion-pipeline-load-target.ts";

function source(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");
}

for (const [name, page] of [
  ["images", "../src/features/images/images-page.tsx"],
  ["video", "../src/features/video/video-page.tsx"],
] as const) {
  test(`${name} pins family override through plan and load`, () => {
    const text = source(page);
    assert.ok(text.includes('"family_override"'));
    assert.ok(text.includes("family_override: advanced.family_override"));
    assert.ok(text.includes("opaqueKind={overrideArtifactKind}"));
    assert.ok(text.includes("resolvedFamilyOverrideSelection("));
    assert.ok(
      text.includes("familyOverrideOptions(status?.supported_families)"),
    );
  });
}

test("image defaults use explicit and resolved family keys for opaque paths", () => {
  const text = source("../src/features/images/images-page.tsx");
  assert.ok(
    text.includes("defaultsFor(defaultsKeyFor(repoId, effectiveFamilyOverride))"),
  );
  assert.ok(
    text.includes(
      'const nextFamilyOverride = familyOverrideRequired ? familyOverride : "auto"',
    ),
  );
  assert.ok(text.includes("applyImageModelDefaults(id, nextFamilyOverride)"));
  assert.ok(text.includes('applyImageModelDefaults(wanted, "auto")'));
  assert.ok(text.includes("const seedKey = `${repoId}\\0${residentDefaults}`"));
  assert.ok(text.includes("defaultsFor(residentDefaults)"));
});

test("video defaults use the explicit family for opaque paths", () => {
  const text = source("../src/features/video/video-page.tsx");
  assert.ok(
    text.includes("defaultsFor(defaultsKeyFor(repoId, effectiveFamilyOverride))"),
  );
  assert.ok(
    text.includes(
      'const nextFamilyOverride = familyOverrideRequired ? familyOverride : "auto"',
    ),
  );
  assert.ok(text.includes("applyVideoModelDefaults(id, nextFamilyOverride)"));
  assert.ok(text.includes('loadGgufRepoPick(wanted, routedLabel, "hub", null, "auto")'));
  assert.ok(
    text.includes("MODEL_DEFAULTS.some((entry) => id.includes(entry.match))"),
  );
});

test("pinned pipeline paths retain their Hub selector identity across remounts", () => {
  for (const file of [
    "../src/features/images/images-page.tsx",
    "../src/features/video/video-page.tsx",
  ]) {
    const text = source(file);
    assert.ok(text.includes("display_repo_id: opts.displayRepoId"), file);
    assert.ok(text.includes("displayRepoId: l.displayRepoId"), file);
    assert.ok(text.includes("status.display_repo_id ?? status.repo_id"), file);
    assert.ok(!text.includes("pinnedPipelineDisplayIds"), file);
    assert.ok(text.includes("loadedModelIdOverride={selectorModelId}"), file);
  }
  const picker = source(
    "../src/features/model-picker/components/model-selector/pickers.tsx",
  );
  const inventory = source(
    "../src/features/model-picker/inventory/use-chat-picker-inventory.ts",
  );
  const images = source("../src/features/images/images-page.tsx");
  assert.ok(
    images.includes("displayRepoId: status.display_repo_id ?? undefined"),
  );
  assert.ok(picker.includes("c.load_id.trim() !== c.repo_id.trim()"));
  assert.ok(inventory.includes("taskOpaqueArtifactSupportsFamilyOverride("));
  assert.ok(
    picker.includes("familyOverrideRequired: c.opaque === true"),
    "cached rows must say whether their explicit family admitted them",
  );
  assert.ok(
    picker.includes("m.opaque === true"),
    "filesystem rows must carry the same admission metadata",
  );
});

test("a pinned Hub pipeline keeps Hub planning separate from its physical load target", () => {
  assert.deepEqual(
    diffusionPipelineLoadTarget("MiniMaxAI/MiniMax-H3", {
      source: "hub",
      loadId: "/cache/snapshots/deadbeef",
    }),
    {
      repoId: "/cache/snapshots/deadbeef",
      displayRepoId: "MiniMaxAI/MiniMax-H3",
      source: "hub",
    },
  );
  assert.deepEqual(
    diffusionPipelineStagingEntries(
      "/cache/snapshots/deadbeef",
      "MiniMaxAI/MiniMax-H3",
      [
        { checkpoint: true, repoId: "MiniMaxAI/MiniMax-H3" },
        { checkpoint: false, repoId: "external/quant" },
      ],
    ),
    [{ checkpoint: false, repoId: "external/quant" }],
  );
  assert.deepEqual(
    diffusionPipelineStagingEntries(
      "MiniMaxAI/MiniMax-H3",
      "MiniMaxAI/MiniMax-H3",
      [{ checkpoint: true, repoId: "MiniMaxAI/MiniMax-H3" }],
    ),
    [{ checkpoint: true, repoId: "MiniMaxAI/MiniMax-H3" }],
  );

  for (const file of [
    "../src/features/images/images-page.tsx",
    "../src/features/video/video-page.tsx",
  ]) {
    const text = source(file);
    assert.ok(
      text.includes("const planRepoId = opts.displayRepoId ?? repoId"),
      file,
    );
    assert.ok(text.includes("familyOverrideRequired = false"), file);
    assert.ok(text.includes('setFamilyOverride("auto")'), file);
    assert.ok(text.includes("repoId,"), file);
    assert.ok(text.includes("diffusionPipelineStagingEntries("), file);
    assert.ok(text.includes("stage(entriesToStage)"), file);
  }
});
