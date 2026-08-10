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
  const advancedAt = flow.indexOf("const advanced = currentLoadAdvanced(opts.kind);");
  const planAt = flow.indexOf("await getVideoDownloadPlan({");
  assert.ok(advancedAt >= 0, "the staged flow must compute the advanced snapshot");
  assert.ok(planAt >= 0, "the staged flow must request the download plan");
  assert.ok(advancedAt < planAt, "the snapshot must precede the plan request");
  assert.match(flow, /pendingStagedLoad\.current = \{\s*repoId,\s*opts,\s*advanced,/);
  assert.ok(source.includes("pending.opts, pending.advanced"));

  assert.ok(flow.includes("handleLoadRef.current(repoId, opts, advanced)"));
});

test("the video picker resolves the full GGUF footprint", () => {
  assert.ok(source.includes("const resolveDownloadFootprint = useCallback("));
  assert.ok(source.includes("const requiredBytes = plan.required_bytes"));
  assert.ok(source.includes("resolveDownloadFootprint={resolveDownloadFootprint}"));
});
test("the staged plan carries the memory request too", () => {
  assert.equal(source.match(/memory_mode: advanced\.memory_mode/g)?.length, 3);
});
