// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { normalizeReportedOffloadFitTiers } from "../src/lib/offload-fit-tiers.ts";

test("normalises /api/system.diffusers_offload_tiers and drops malformed tiers", () => {
  assert.deepEqual(normalizeReportedOffloadFitTiers(undefined), {});
  assert.deepEqual(normalizeReportedOffloadFitTiers([]), {});
  assert.deepEqual(normalizeReportedOffloadFitTiers({ X: "no" }), {});
  assert.deepEqual(
    normalizeReportedOffloadFitTiers({
      "MiniMaxAI/MiniMax-H3": [
        { gpu_gb: 11, system_ram_gb: 60, requires_quantised_streaming: true },
        { gpu_gb: 30, system_ram_gb: 60 },
        { gpu_gb: 24, system_ram_gb: 40, requires_quantised_streaming: false },
        { gpu_gb: "12", system_ram_gb: 60 },
        { gpu_gb: 0, system_ram_gb: 60 },
        { gpu_gb: Number.NaN, system_ram_gb: 60 },
        null,
      ],
    }),
    {
      "minimaxai/minimax-h3": [
        { gpuGb: 11, systemRamGb: 60, requiresQuantisedStreaming: true },
        // Missing flag reads as "requires streaming", the conservative side.
        { gpuGb: 30, systemRamGb: 60, requiresQuantisedStreaming: true },
        { gpuGb: 24, systemRamGb: 40, requiresQuantisedStreaming: false },
      ],
    },
  );
});
