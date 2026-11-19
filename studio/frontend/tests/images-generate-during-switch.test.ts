// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Generate stays off while a model pick plans, downloads or loads: the resident model is on its way out and the form
// already carries the incoming model's steps and guidance. The backend refuses such a run too (409).

import assert from "node:assert/strict";
import test from "node:test";

import {
  SWITCHING_MODEL_NOTICE,
  generateGate,
  pickReleasedWhileNewest,
} from "../src/features/images/lib/generate-gate.ts";
import { createPickGuard } from "../src/lib/diffusion-gguf-pick.ts";
import { readSrc, readText } from "./helpers/kit.ts";

const idle = {
  busy: null,
  presetsHydrated: true,
  loaded: true,
  canRecallModel: true,
  external: false,
  switching: false,
  loraReload: null,
  queueWaiting: false,
  canQueue: true,
};

test("Generate is on when a model is resident and nothing is switching", () => {
  const gate = generateGate(idle);
  assert.equal(gate.disabled, false);
  assert.equal(gate.label, "Generate");
  assert.equal(gate.notice, null);
});

test("jobs waiting in the queue turn the button into Queue", () => {
  const gate = generateGate({ ...idle, queueWaiting: true });
  assert.equal(gate.action, "queue");
  assert.equal(gate.label, "Queue");
  assert.equal(generateGate({ ...idle, queueWaiting: true, canQueue: false }).action, "generate");
});

test("a pick that is planning or downloading turns Generate off and says why", () => {
  const gate = generateGate({ ...idle, switching: true });
  assert.equal(gate.disabled, true);
  assert.equal(gate.notice, SWITCHING_MODEL_NOTICE);
  // An external connection does not run on the local model.
  assert.equal(generateGate({ ...idle, switching: true, external: true }).disabled, false);
});

test("a load, unload or run in flight keeps it off", () => {
  for (const busy of ["loading", "unloading", "generating"]) {
    const gate = generateGate({ ...idle, busy });
    assert.equal(gate.disabled, true, busy);
    assert.equal(gate.notice, null, busy);
  }
  assert.equal(generateGate({ ...idle, loaded: false, canRecallModel: false }).disabled, true);
  assert.equal(generateGate({ ...idle, presetsHydrated: false }).disabled, true);
});

test("the Images page tracks every pick and gates the button on it", () => {
  const page = readSrc("features/images/images-page.tsx");
  const assignments = page.match(/quantRevert\.current = [^;\n]+;/g) ?? [];
  assert.ok(assignments.length > 10);
  // Each pick start and end updates the render state the button reads.
  const unpaired = page
    .split("\n")
    .map((line, i, lines) => [line, lines[i + 1] ?? ""])
    .filter(([line]) => /quantRevert\.current = [^;\n]+;/.test(line))
    .filter(([, next]) => !next.includes("setPickInFlight(quantRevert.current !== null)"));
  assert.deepEqual(unpaired, []);
  assert.match(page, /switching: pickInFlight,/);
  assert.match(page, /disabled=\{generateButton\.disabled\}/);
  // Compare, Vary and Queue reach handleGenerate directly.
  assert.match(page, /if \(!external && quantRevert\.current\) \{/);
});

test("the backend refuses an image generate while a load is in flight", () => {
  const routes = readText("../../backend/routes/inference.py");
  assert.match(routes, /if await asyncio\.to_thread\(_image_load_in_flight\):\s*raise HTTPException\(\s*status_code = 409/);
});

test("a pick whose page went hidden mid-plan is taken back, so Generate cannot stay held", () => {
  const guard = createPickGuard();
  const first = guard.claim();
  assert.equal(pickReleasedWhileNewest(guard, first), false);
  // The page switch: the plan that answers now stages and loads nothing.
  guard.release();
  assert.equal(pickReleasedWhileNewest(guard, first), true);
  // A newer pick or a cancel owns the settle instead.
  const second = guard.claim();
  assert.equal(pickReleasedWhileNewest(guard, first), false);
  guard.cancel();
  assert.equal(pickReleasedWhileNewest(guard, second), false);
  assert.equal(pickReleasedWhileNewest(guard, undefined), false);
  assert.equal(pickReleasedWhileNewest(guard, 0), false);

  // Both superseded returns of loadOrStage drop a released pick before answering "handled".
  const page = readSrc("features/images/images-page.tsx");
  const start = page.indexOf("const loadOrStage = useCallback(");
  const end = page.indexOf("const loadOrStageRef = useRef(loadOrStage);");
  const body = page.slice(start, end);
  assert.ok(start >= 0 && end > start);
  assert.match(body, /if \(!pickReleasedWhileNewest\(pickGuard, token\)\) return;/);
  assert.match(body, /if \(!owns\(\)\) \{\s*dropReleasedPick\(quantRevert\.current\);\s*return true;/);
  const superseded = body.match(/pickToast\.dismiss\(pickToastId\);\s*dropReleasedPick\(ownRevert\);\s*return true;/g) ?? [];
  assert.equal(superseded.length, 2);
  assert.equal((body.match(/pickToast\.dismiss\(pickToastId\);\s*return true;/g) ?? []).length, 0);
});
