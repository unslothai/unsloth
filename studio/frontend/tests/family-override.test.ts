// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  explicitFamily,
  familyOverrideArtifactKind,
  familyOverrideOptions,
  resolvedFamilyOverrideSelection,
  taskOpaqueArtifactSupportsFamilyOverride,
} from "../src/features/model-picker/components/model-selector/family-override.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

// diffusion-pipeline-load-target.ts imports through the `@/` alias.
registerBundlerResolver();
const { diffusionPipelineLoadTarget, diffusionStagingEntries } = await import(
  "../src/lib/diffusion-pipeline-load-target.ts"
);

test("an explicit family selects exactly one structurally loadable artifact kind", () => {
  for (const [family, modular, want] of [
    ["auto", undefined, undefined],
    [undefined, undefined, undefined],
    ["z-image", undefined, "diffusers_pipeline"],
    ["ltx-2", ["minimax-h3"], "diffusers_pipeline"],
    [" MINIMAX-H3 ", ["minimax-h3"], "diffusers_modular_pipeline"],
  ] as const) {
    assert.equal(familyOverrideArtifactKind(family, modular), want, String(family));
  }
  for (const [value, want] of [[" z-image ", "z-image"], ["AUTO", undefined], ["", undefined], [3, undefined]] as const) {
    assert.equal(explicitFamily(value), want);
  }
});

test("a family admits only a task-less row of a matching (or dual) manifest", () => {
  for (const [task, artifact, required, want] of [
    [null, "diffusers_pipeline", "diffusers_pipeline", true],
    [" ", "diffusers_dual_pipeline", "diffusers_pipeline", true],
    [null, "diffusers_dual_pipeline", "diffusers_modular_pipeline", true],
    [null, "diffusers_pipeline", "diffusers_modular_pipeline", false],
    [null, "diffusers_pipeline", undefined, false],
    ["text-to-image", "diffusers_pipeline", "diffusers_pipeline", false],
    ["text-to-video", "diffusers_dual_pipeline", "diffusers_modular_pipeline", false],
  ] as const) {
    assert.equal(taskOpaqueArtifactSupportsFamilyOverride(task, artifact, required), want, `${task} ${artifact} ${required}`);
  }
});

test("selector restoration prefers the canonical engaged family over an alias", () => {
  assert.equal(resolvedFamilyOverrideSelection({ source: "explicit", requested: "h3", value: "minimax-h3" }), "minimax-h3");
  assert.equal(resolvedFamilyOverrideSelection({ source: "explicit", requested: "h3", value: "" }), "h3");
  assert.equal(resolvedFamilyOverrideSelection({ source: "auto", requested: null, value: "minimax-h3" }), "auto");
  assert.equal(resolvedFamilyOverrideSelection(null), undefined);
});

test("family options follow the backend registry, deduplicated, and every backend family has a label", () => {
  assert.deepEqual(familyOverrideOptions(["z-image", "z-image", "flux.1"]), [
    ["auto", "Auto (detect)"],
    ["z-image", "Z-Image"],
    ["flux.1", "FLUX.1"],
  ]);
  const root = new URL("../../backend/core/inference/", import.meta.url);
  const names = ["diffusion_families.py", "video_families.py"].flatMap((f) =>
    [...readFileSync(new URL(f, root), "utf8").matchAll(/^\s+name = "([^"]+)"/gm)].map((m) => m[1]),
  );
  assert.ok(names.includes("flux.1") && names.includes("ltx-2"));
  const unlabeled = familyOverrideOptions(names).slice(1).filter(([value, label]) => value === label);
  assert.deepEqual(unlabeled, []);
});

test("a pinned cache snapshot loads by path, plans by its Hub id, and stages only companions", () => {
  const pinned = diffusionPipelineLoadTarget("MiniMaxAI/MiniMax-H3", { source: "hub", loadId: " /cache/snapshots/abc " });
  assert.deepEqual(pinned, {
    repoId: "/cache/snapshots/abc",
    displayRepoId: "MiniMaxAI/MiniMax-H3",
    source: "hub",
    onDevice: true,
  });
  for (const loadId of [undefined, "Org/Opaque", " "]) {
    assert.deepEqual(diffusionPipelineLoadTarget("Org/Opaque", { source: "hub", loadId }), {
      repoId: "Org/Opaque",
      source: "hub",
      onDevice: false,
    });
  }
  assert.equal(diffusionPipelineLoadTarget("/models/x", { source: "local" }).onDevice, true);

  const entry = (repo_id: string, checkpoint?: boolean) => ({ repo_id, files: ["a"], bytes: 1, gguf_filename: null, checkpoint });
  const plan = [entry("MiniMaxAI/MiniMax-H3"), entry("external/quant", false)];
  assert.deepEqual(
    diffusionStagingEntries(plan, pinned.repoId, { displayRepoId: pinned.displayRepoId }).map((e) => e.repoId),
    ["external/quant"],
  );
  assert.deepEqual(
    diffusionStagingEntries(plan, "MiniMaxAI/MiniMax-H3", {}).map((e) => [e.repoId, e.checkpoint]),
    [["MiniMaxAI/MiniMax-H3", true], ["external/quant", false]],
  );
});

for (const page of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
  test(`${page} pins the family and logical id through plan, load and selector`, () => {
    const text = readSrc(page);
    for (const needle of [
      "family_override: advanced.family_override",
      "display_repo_id: opts.displayRepoId",
      "displayRepoId: l.displayRepoId",
      "opaqueKind={opaqueKind}",
      "loadedModelIdOverride={selectorModelId}",
      "!pipelineTarget.onDevice",
    ]) {
      assert.ok(text.includes(needle), needle);
    }
    assert.match(text, /diffusionStagingEntries\(plan\.entries, repoId, \{\s*\.\.\.opts,/);
    if (page.includes("images")) assert.ok(text.includes("downloadOnly ? currentLoadAdvanced(repoId, false)"));
  });
}
