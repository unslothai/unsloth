// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { restyleSetupView } from "../src/features/video/restyle-setup.ts";

import { readSrc } from "./helpers/kit.ts";

const fresh = {
  available: false,
  reason:
    "The Cosmos3-Nano-Sim-Transfer weights are not set up on this machine.",
  missing: ["weights", "engine"],
  can_setup: true,
  refusal: null,
  weights: "bf16" as const,
  setup: { active: false, phase: null },
};

test("a fresh install offers Set up Restyle with what it will do", () => {
  const view = restyleSetupView(fresh);
  assert.equal(view?.kind, "offer");
  assert.match(
    view.kind === "offer" ? view.message : "",
    /stable-diffusion\.cpp server/,
  );
  assert.match(view.kind === "offer" ? view.message : "", /about 35 GB/);
});

test("a 40 GB card is told about the 8-bit copy", () => {
  const view = restyleSetupView({ ...fresh, weights: "q8_0" });
  assert.match(view?.kind === "offer" ? view.message : "", /8-bit/);
});

test("a card that cannot run it gets the refusal and no button", () => {
  const view = restyleSetupView({
    ...fresh,
    can_setup: false,
    refusal:
      "Restyle needs a GPU with at least 31 GB of memory; this one has 22.0 GB.",
  });
  assert.deepEqual(view, {
    kind: "refused",
    message:
      "Restyle needs a GPU with at least 31 GB of memory; this one has 22.0 GB.",
  });
});

test("overridden weights that are missing are reported, not set up", () => {
  const view = restyleSetupView({
    ...fresh,
    missing: ["weights"],
    can_setup: false,
    reason: "Set X",
  });
  assert.deepEqual(view, { kind: "blocked", message: "Set X" });
});

test("a running setup shows its phase and byte progress", () => {
  const view = restyleSetupView({
    ...fresh,
    setup: {
      active: true,
      phase: "download",
      bytes_done: 8e9,
      bytes_total: 32e9,
    },
  });
  assert.equal(view?.kind, "running");
  if (view?.kind !== "running") return;
  assert.equal(view.percent, 25);
  assert.match(view.label, /Downloading .*8\.0 GB of 32\.0 GB/);
  const engine = restyleSetupView({
    ...fresh,
    setup: { active: true, phase: "engine" },
  });
  assert.equal(engine?.kind === "running" ? engine.percent : -1, null);
});

test("a failed setup keeps the offer and shows the error", () => {
  const view = restyleSetupView({
    ...fresh,
    setup: { active: false, phase: "error", error: "no disk" },
  });
  assert.equal(view?.kind === "offer" ? view.error : null, "no disk");
});

test("ready once nothing is missing", () => {
  assert.deepEqual(
    restyleSetupView({ available: true, reason: null, missing: [] }),
    { kind: "ready" },
  );
  assert.equal(restyleSetupView(null), null);
});

test("the Restyle tab wires the setup button to the setup route", () => {
  const panel = readSrc("features/video/restyle-panel.tsx");
  assert.match(panel, /startVideoRestyleSetup\(hfApiToken\(getHfToken\(\)\)\)/);
  assert.match(panel, /data-testid="restyle-setup-start"/);
  const api = readSrc("features/video/api.ts");
  assert.match(api, /\/api\/inference\/video\/restyle\/setup"/);
  assert.match(api, /\/api\/inference\/video\/restyle\/setup\/cancel"/);
});
