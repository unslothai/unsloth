// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Tensor Parallelism row is one of the llama-server controls a diffusion model
// cannot act on. withoutUnsupportedDiffusionSettings forces it back to false on every
// state update and the diffusion runner never reads it, so an ungated row is a switch
// that flips back under the pointer and would change nothing if it did not.

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

/** The nearest JSX conditional boundary above `marker`, walking upwards. */
function gateAbove(marker: string): string {
  const lines = CONFIG_PAGE.split("\n");
  const at = lines.findIndex((line) => line.includes(marker));
  assert.notEqual(at, -1, `missing marker: ${marker}`);
  for (let i = at; i >= 0; i--) {
    const line = lines[i].trim();
    if (line === "{!isDiffusion && (") return "!isDiffusion";
    // A `)}` first means the gate closed before the row, i.e. it is not inside one.
    if (line === ")}") return "closed";
  }
  return "none";
}

test("the reconciler still clears tensorParallel for a diffusion model", () => {
  // Without this the gate below would be guarding nothing, and the test would pass
  // while the row quietly became actionable again.
  assert.match(CONFIG_PAGE, /tensorParallel: false,/);
});

test("the Tensor Parallelism row is gated out for diffusion models", () => {
  assert.equal(gateAbove("checked={config.tensorParallel}"), "!isDiffusion");
});

test("the Vision row it sits beside stays gated too", () => {
  // Both rows are unsupported for the same reason; regating one and not the other is
  // the state this change exists to end.
  assert.equal(gateAbove("checked={!config.disableVision}"), "!isDiffusion");
});
