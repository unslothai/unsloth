// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type CosmosGuardrailStatus,
  guardrailInstallOutcome,
  isGuardrailInstallRefusal,
  runGuardrailInstall,
} from "../src/features/video/cosmos-guardrail.ts";
import { readSrc } from "./helpers/kit.ts";

// The backend's two refusals, as video_cosmos3 words them.
const MISSING =
  "'nvidia/Cosmos3-Nano' needs NVIDIA's gated Cosmos guardrail. Install it with Install guardrail on the Video page " +
  "or run `uv pip install --python /home/me/.unsloth/studio/unsloth_studio/bin/python cosmos-guardrail`, then accept " +
  "its licence at https://huggingface.co/nvidia/Cosmos-1.0-Guardrail, add that account's Hugging Face token in " +
  "Settings, and load again. Or set UNSLOTH_COSMOS3_SAFETY_CHECKER=0 to run without it at your own responsibility.";
const NO_ACCESS =
  "'nvidia/Cosmos3-Nano' needs NVIDIA's gated Cosmos guardrail, and this Hugging Face token has no access: accept " +
  "its licence at https://huggingface.co/nvidia/Cosmos-1.0-Guardrail, add that account's Hugging Face token in " +
  "Settings, and load again. Or set UNSLOTH_COSMOS3_SAFETY_CHECKER=0 to run without it at your own responsibility.";

const status = (s: Partial<CosmosGuardrailStatus>): CosmosGuardrailStatus => ({
  state: "idle",
  installed: false,
  error: null,
  ...s,
});

test("only the missing-package refusal offers the install", () => {
  assert.equal(isGuardrailInstallRefusal(MISSING), true);
  assert.equal(isGuardrailInstallRefusal(NO_ACCESS), false);
  assert.equal(isGuardrailInstallRefusal("Failed to load model"), false);
  assert.equal(isGuardrailInstallRefusal(null), false);
});

test("an install is followed until it ends", async () => {
  const polls = [status({ state: "running" }), status({ state: "done", installed: true, added: ["cosmos-guardrail"] })];
  const waits: number[] = [];
  const result = await runGuardrailInstall(
    async () => status({ state: "running" }),
    async () => polls.shift()!,
    async (ms) => void waits.push(ms),
    { intervalMs: 5 },
  );
  assert.equal(result.ok, true);
  assert.match(result.message, /Added cosmos-guardrail\./);
  assert.match(result.message, /Cosmos-1\.0-Guardrail/);
  assert.deepEqual(waits, [5, 5]);
});

test("a failed install reports the backend's reason", async () => {
  const result = await runGuardrailInstall(
    async () => status({ state: "running" }),
    async () => status({ state: "failed", error: "transformers>=5.0.0 is required" }),
    async () => {},
  );
  assert.deepEqual(result, { ok: false, message: "transformers>=5.0.0 is required" });
  assert.deepEqual(guardrailInstallOutcome(status({ state: "idle" })), {
    finished: true,
    ok: false,
    message: "The cosmos-guardrail install did not start.",
  });
});

test("the video page wires the install action for the owner only", () => {
  const page = readSrc("features/video/video-page.tsx");
  assert.match(page, /isGuardrailInstallRefusal\(text\)/);
  assert.match(page, /action: isAccountOwner\(\)\s*\? \{ label: "Install guardrail"/);
  const api = readSrc("features/video/api.ts");
  assert.match(api, /"\/api\/inference\/video\/cosmos-guardrail\/install", \{ method: "POST" \}/);
});

test("a dropped status poll does not end a running install", async () => {
  const states: Array<CosmosGuardrailStatus | Error> = [
    new Error("network"),
    { state: "running", installed: false, error: null },
    { state: "done", installed: true, error: null, added: ["cosmos-guardrail"] },
  ];
  const result = await runGuardrailInstall(
    async () => ({ state: "running", installed: false, error: null }),
    async () => {
      const next = states.shift()!;
      if (next instanceof Error) throw next;
      return next;
    },
    async () => undefined,
  );
  assert.equal(result.ok, true);
});
