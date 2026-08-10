// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The staged-plan request has to carry the same precision the load will send.
 *
 * /video/download-plan refuses a scheme this host cannot honour, so a plan asked without the
 * precision succeeded, staged tens of GB of pipeline weights, and left the refusal to the load
 * afterwards -- which is the whole failure the plan-time check was added to prevent.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

const source = readFileSync(
  fileURLToPath(new URL("../src/features/video/video-page.tsx", import.meta.url)),
  "utf8",
);

test("the video download plan is asked with the selected precision", () => {
  const call = source.slice(
    source.indexOf("await getVideoDownloadPlan({"),
    source.indexOf("await getVideoDownloadPlan({") + 900,
  );
  assert.ok(call.length > 0, "the plan call must exist");
  assert.ok(
    call.includes("transformer_quant:"),
    "the plan must be asked with the precision the load will use",
  );
  // Under the same pipeline-only rule the load applies: a GGUF / single-file DiT runs the
  // precision its checkpoint carries, and the stale control value must not reach either call.
  assert.ok(call.includes('opts.kind === "pipeline"'));
});

test("the staged plan reads the precision live, not the value it closed over", () => {
  // loadOrStage is memoized on [stage, pickGuard] so its consumers keep a stable identity. A
  // plain capture of transformerQuant therefore froze at the value selected when the callback
  // was built, and the ordinary auto -> FP8 change sent the plan no precision at all: the
  // pre-download refusal was skipped and tens of GB were staged before the load refused it.
  const call = source.slice(
    source.indexOf("await getVideoDownloadPlan({"),
    source.indexOf("await getVideoDownloadPlan({") + 900,
  );
  assert.ok(
    call.includes("transformerQuantRef.current"),
    "the plan must read the precision through the ref",
  );
  assert.ok(
    !/transformerQuant\s*[!=]==/.test(call),
    "a direct read of the memoized capture is the stale value",
  );
  assert.ok(
    source.includes("transformerQuantRef.current = transformerQuant"),
    "the ref must be kept current on every render",
  );
});

test("the staged plan carries the memory request too", () => {
  // The route refuses an explicit precision under balanced or low_vram only when it can see the
  // memory mode. Omitting it here meant the plan succeeded, tens of GB were staged, and the
  // identical pick was then rejected by /video/load -- the regression the plan gate exists for.
  const call = source.slice(
    source.indexOf("await getVideoDownloadPlan({"),
    source.indexOf("await getVideoDownloadPlan({") + 1200,
  );
  assert.ok(call.includes("memory_mode:"), "the plan must be asked with the memory request");
  assert.ok(
    call.includes("memoryModeRef.current"),
    "and read it live, like the precision",
  );
  assert.ok(source.includes("memoryModeRef.current = memoryMode"));
});

test("the selected H3 task reaches both the plan and the load", () => {
  const planCall = source.slice(
    source.indexOf("await getVideoDownloadPlan({"),
    source.indexOf("await getVideoDownloadPlan({") + 1500,
  );
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
  // The chat picker cannot load a diffusion model, so a pick there arrives on this page as
  // ?model=. That route calls loadOrStage directly: without the same interception the direct
  // pick makes, a cached MiniMax H3 silently staged the fl2va denoiser, tens of GB, and left no
  // way to ask for References.
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
  // One predicate, so the two entry points cannot drift apart again.
  assert.ok(source.includes("function isH3PipelinePick("));
  assert.ok(source.includes("isH3PipelinePick(id, spec.kind)"));
});

test("reapply preserves the loaded H3 task", () => {
  const reapply = source.slice(
    source.indexOf("const handleReapply = useCallback"),
    source.indexOf("const handleReapply = useCallback") + 600,
  );
  assert.ok(reapply.includes("h3Task: l.h3Task"));
});
