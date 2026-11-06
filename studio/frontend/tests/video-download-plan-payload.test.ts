// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** /video/download-plan refuses an unsupported precision, so the plan must carry it or GBs
 * get staged before the load refuses. */

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const source = readSrc("features/video/video-page.tsx");

test("the video download plan is asked with the selected precision", () => {
  const call = source.slice(
    source.indexOf("await getVideoDownloadPlan({"),
    source.indexOf("await getVideoDownloadPlan({") + 900,
  );
  assert.ok(call.length > 0, "the plan call must exist");
  assert.ok(
    call.includes("transformer_quant: advanced.transformer_quant"),
    "the plan must use the precision snapshot",
  );
  const snapshot = source.slice(
    source.indexOf("const currentLoadAdvanced = useCallback("),
    source.indexOf("const handleLoad = useCallback("),
  );
  assert.ok(snapshot.includes("loadControlsRef.current"));

  assert.ok(snapshot.includes('kind === "pipeline"'));
});

test("the staged plan pins its controls through the eventual load", () => {
  const flow = source.slice(
    source.indexOf("const loadOrStage = useCallback("),
    source.indexOf("// A GGUF pick can arrive"),
  );
  const advancedAt = flow.indexOf("const advanced = currentLoadAdvanced(opts.kind, familyOverrideRequired);");
  const planAt = flow.indexOf("await getVideoDownloadPlan({");
  assert.ok(advancedAt >= 0, "the staged flow must compute the advanced snapshot");
  assert.ok(planAt >= 0, "the staged flow must request the download plan");
  assert.ok(advancedAt < planAt, "the snapshot must precede the plan request");
  assert.match(flow, /pendingStagedLoad\.current = \{\s*repoId,\s*opts,\s*advanced,/);
  assert.ok(source.includes("pending.opts, pending.advanced"));

  assert.match(flow, /handleLoadRef\.current\(repoId, opts, advanced[,)]/);
});

test("the video picker resolves the full GGUF footprint", () => {
  assert.ok(source.includes("const resolveDownloadFootprint = useCallback("));
  assert.ok(source.includes("const requiredBytes = plan.required_bytes"));
  assert.ok(source.includes("resolveDownloadFootprint={resolveDownloadFootprint}"));
});
test("the staged plan carries the memory request too", () => {
  assert.equal(source.match(/memory_mode: advanced\.memory_mode/g)?.length, 3);
});

test("the selected H3 task reaches both the plan and the load", () => {
  // The STAGING plan, not the earlier footprint probe call for a named GGUF file.
  const flow = source.slice(
    source.indexOf("const loadOrStage = useCallback("),
    source.indexOf("// A GGUF pick can arrive"),
  );
  const planAt = flow.indexOf("await getVideoDownloadPlan({");
  assert.ok(planAt >= 0, "the staged flow must request the download plan");
  const planCall = flow.slice(planAt, planAt + 1500);
  const loadCall = source.slice(
    source.indexOf("const startRequest = loadVideoModel({"),
    source.indexOf("const startRequest = loadVideoModel({") + 1500,
  );
  assert.ok(planCall.includes("h3_task: opts.h3Task"));
  assert.ok(loadCall.includes("h3_task: opts.h3Task"));
  assert.ok(source.includes('chooseH3Task("fl2va")'));
  assert.ok(source.includes('chooseH3Task("ref2va")'));
});

test("a routed H3 pipeline pick asks for the task instead of loading a default", () => {
  // A ?model= pick from chat calls loadOrStage directly, so it needs the same H3 interception.
  const routeEffect = source.slice(
    source.indexOf("const pick = diffusionRoutePick("),
    source.indexOf("const chooseH3Task = useCallback"),
  );
  assert.ok(routeEffect.length > 0, "the routed pick branch must exist");
  assert.ok(
    routeEffect.includes("isH3PipelinePick(pick.repoId, pick.opts.kind)"),
    "the routed branch must intercept an H3 pipeline pick",
  );
  const intercept = routeEffect.indexOf("isH3PipelinePick(");
  const load = routeEffect.indexOf("void loadOrStage(pick.repoId");
  assert.ok(
    intercept >= 0 && load > intercept,
    "the interception must come before the unconditional load",
  );
  assert.ok(routeEffect.includes("setPendingH3Load({"));
  assert.ok(source.includes("function isH3PipelinePick("));
  assert.ok(source.includes("isH3PipelinePick(id, spec.kind, nextFamilyOverride)"));
});

test("reapply preserves the loaded H3 task", () => {
  const reapply = source.slice(
    source.indexOf("const handleReapply = useCallback"),
    source.indexOf("const handleReapply = useCallback") + 600,
  );
  assert.ok(reapply.includes("h3Task: l.h3Task"));
});
