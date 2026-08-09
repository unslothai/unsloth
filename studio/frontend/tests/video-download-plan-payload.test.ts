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
