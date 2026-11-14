// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Diffusion forces tensorParallel false on every update, so an ungated row flips back.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_PAGE = readFileSync(
  path.join(HERE, "..", "src/features/model-picker/components/model-config-page.tsx"),
  "utf8",
);

function gateAbove(marker: string): string {
  const lines = CONFIG_PAGE.split("\n");
  const at = lines.findIndex((line) => line.includes(marker));
  assert.notEqual(at, -1, `missing marker: ${marker}`);
  for (let i = at; i >= 0; i--) {
    const line = lines[i].trim();
    if (line === "{!isDiffusion && (") return "!isDiffusion";
    if (line === ")}") return "closed";
  }
  return "none";
}

test("the reconciler still clears tensorParallel for a diffusion model", () => {
  assert.match(CONFIG_PAGE, /tensorParallel: false,/);
});

test("the Tensor Parallelism row is gated out for diffusion models", () => {
  assert.equal(gateAbove("checked={config.tensorParallel}"), "!isDiffusion");
});

test("the Vision row it sits beside stays gated too", () => {
  assert.equal(gateAbove("checked={!config.disableVision}"), "!isDiffusion");
});
